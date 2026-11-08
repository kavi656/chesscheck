"""The honest pipeline: log -> explore -> BH correction -> validate -> skeptic -> vault -> verdict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.hypothesis import Hypothesis, InvalidHypothesis, groups, validate
from src.ledger import Ledger, now
from src.tester import K_WINDOW, TestResult, cluster_test, far_baseline

Q = 0.05            # BH false discovery rate across all hypotheses in the ledger
ALPHA = 0.05        # validate / vault / concentration significance
BAND_MIN_TREATED = 500
TOP_ACTIVE_FRAC = 0.01
DATA = Path("data/processed/player_games.parquet")


class VaultAccessError(RuntimeError):
    pass


class Vault:
    """Holds the vault split. Each hypothesis id may read it exactly once; every access is logged."""

    def __init__(self, df: pd.DataFrame, log_path: str | Path | None = None):
        self._df = df
        self._read: set[str] = set()
        self.log_path = Path(log_path) if log_path else None
        self.accesses: list[dict] = []

    def read(self, hid: str) -> pd.DataFrame:
        rec = {"ts": now(), "id": hid, "granted": hid not in self._read}
        self.accesses.append(rec)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if hid in self._read:
            raise VaultAccessError(f"vault already read for {hid}; one shot only")
        self._read.add(hid)
        return self._df

    def __len__(self):
        return len(self._df)


def bh_threshold(pvals: list[float], q: float = Q) -> float:
    """Largest p_(k) with p_(k) <= k q / m; hypotheses with p <= threshold are discoveries."""
    p = np.sort(np.asarray(pvals, dtype=float))
    m = len(p)
    if m == 0:
        return -1.0
    ok = p <= q * np.arange(1, m + 1) / m
    return float(p[np.nonzero(ok)[0].max()]) if ok.any() else -1.0


def expected_sign(h: Hypothesis, explore_effect: float) -> int:
    return {"greater": 1, "less": -1}.get(h.direction, int(np.sign(explore_effect) or 1))


def directional_p(r: TestResult, sign: int, direction: str) -> float:
    """p-value in the pre-registered direction (for two-sided: two-sided p, sign checked separately)."""
    if r.p is None:
        return 1.0
    if direction == "two-sided":
        return r.p if np.sign(r.effect) == sign else 1.0
    return r.p


def add_adjusted_outcome(df: pd.DataFrame, y_col: str = "residual") -> pd.DataFrame:
    """y_adj = y - player's mean y over games > K positions away in the same split (see tester.py).
    Computed separately within each split, so no split's outcomes enter another split's tests."""
    group = df["player_code"].to_numpy().astype(np.int64) * 4 + df["split"].cat.codes.to_numpy()
    y = df[y_col].to_numpy(dtype=np.float64)
    df["y_adj"] = y - far_baseline(group, y, K_WINDOW)
    return df


def load_data(path: Path = DATA) -> pd.DataFrame:
    return prepare(pd.read_parquet(path))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop burn-in, sort by player/time, add integer codes and the adjusted outcome."""
    df = df[df["split"] != "burn_in"].reset_index(drop=True)
    df["player"] = df["player"].astype("category")
    df["split"] = df["split"].cat.remove_unused_categories()
    df["player_code"] = df["player"].cat.codes.astype(np.int32)
    df = df.sort_values(["player_code", "start", "game_idx"], kind="stable").reset_index(drop=True)
    df["game_code"] = df["game_idx"].astype(np.int64)
    if "hour_utc" in df.columns:
        df["local_hour"] = estimated_local_hour(df)
    return add_adjusted_outcome(df)


QUIET_HOURS = 6          # length of the assumed nightly sleep window
MIN_GAMES_FOR_CLOCK = 30  # players with fewer explore games get no local_hour (NaN)


def estimated_local_hour(df: pd.DataFrame) -> np.ndarray:
    """Player-relative clock. Lichess headers carry no timezone, so for each player we find the 6-hour
    UTC window in which they played the fewest EXPLORE games (their likely sleep), and map that window to
    01:00-06:59 'local'. Uses only game start times from the explore split - never outcomes and never
    validate/vault data - so it is a fixed per-player attribute learned before any testing."""
    code = df["player_code"].to_numpy()
    hour = df["hour_utc"].to_numpy().astype(np.int64)
    ex = (df["split"] == "explore").to_numpy()
    n_players = int(code.max()) + 1
    counts = np.bincount(code[ex] * 24 + hour[ex], minlength=n_players * 24).reshape(n_players, 24)
    window = sum(np.roll(counts, -k, axis=1) for k in range(QUIET_HOURS))  # window[:, s] = games in s..s+5
    quiet_start = window.argmin(axis=1)
    local = (hour - quiet_start[code] + 1) % 24
    known = counts.sum(axis=1)[code] >= MIN_GAMES_FOR_CLOCK
    return np.where(known, local, np.nan).astype(np.float32)


class Judge:
    def __init__(self, data: pd.DataFrame, ledger: Ledger, vault_log: str | Path | None = None,
                 mask_cache: dict | None = None, y_col: str = "y_adj"):
        ev = data[data["split"].isin(["explore", "validate"])].reset_index(drop=True)
        self.vault = Vault(data[data["split"] == "vault"].reset_index(drop=True), vault_log)
        self.ev = ev
        self.ledger = ledger
        self.y_col = y_col
        self.cache = mask_cache if mask_cache is not None else {}
        self.player = ev["player_code"].to_numpy()
        self.game = ev["game_code"].to_numpy()
        self.y = ev[y_col].to_numpy(dtype=np.float64)
        self.raw = ev["residual"].to_numpy(dtype=np.float64)
        self.is_explore = (ev["split"] == "explore").to_numpy()
        self.is_validate = ~self.is_explore
        mid = ev["start"].quantile(0.5)
        self.first_half = (ev["start"] < mid).to_numpy()
        counts = ev.groupby("player_code", observed=True).size()
        top = counts.sort_values(ascending=False).index[: max(1, int(len(counts) * TOP_ACTIVE_FRAC))]
        self.not_top = ~np.isin(self.player, top.to_numpy())
        self.band_masks = {b: (ev["rating_band"] == b).to_numpy() for b in ev["rating_band"].cat.categories}
        self.hyps: dict[str, Hypothesis] = {}

    # ---------------------------------------------------------------- intake
    def submit(self, raws: list[dict], source: str, batch: str | None = None) -> list[str]:
        """Log every proposal first (invalid and duplicates included), then validate."""
        ids = [self.ledger.propose(r, source, batch) for r in raws]
        seen = {h.spec_key() for h in self.hyps.values()}
        for hid, raw in zip(ids, raws):
            try:
                h = validate({**raw, "id": hid})
            except InvalidHypothesis as e:
                self.ledger.verdict(hid, "INVALID", "schema", str(e))
                continue
            if h.spec_key() in seen:
                self.ledger.verdict(hid, "DUPLICATE", "schema", f"same spec as an earlier hypothesis ({h.spec_key()})")
                continue
            seen.add(h.spec_key())
            self.hyps[hid] = h
        return ids

    # ---------------------------------------------------------------- testing
    def _masks(self, h: Hypothesis):
        key = ("ev", h.spec_key())
        if key not in self.cache:
            t, c = groups(self.ev, h)
            self.cache[key] = (np.packbits(t), np.packbits(c), len(t))
        t, c, n = self.cache[key]
        return np.unpackbits(t, count=n).astype(bool), np.unpackbits(c, count=n).astype(bool)

    def _test(self, t, c, extra, direction) -> TestResult:
        return cluster_test(self.y, t & extra, c & extra, self.player, self.game, direction, raw=self.raw)

    def explore(self, hid: str) -> None:
        h = self.hyps[hid]
        t, c = self._masks(h)
        r = self._test(t, c, self.is_explore, h.direction)
        self.ledger.record(hid, "explore", **r.to_dict())

    def _validate_and_skeptic(self, hid: str, st: dict) -> None:
        h = self.hyps[hid]
        t, c = self._masks(h)
        sign = expected_sign(h, st["explore"]["effect"])
        r = self._test(t, c, self.is_validate, h.direction)
        dp = directional_p(r, sign, h.direction)
        passed = (not r.underpowered) and dp < ALPHA and abs(r.effect) >= h.min_effect
        self.ledger.record(hid, "validate", **r.to_dict(), passed=passed, directional_p=dp)
        if not passed:
            return
        self.ledger.record(hid, "skeptic", **self.skeptic(h, t, c, sign))

    def skeptic(self, h: Hypothesis, t, c, sign: int) -> dict:
        halves = [self._test(t, c, self.first_half, h.direction), self._test(t, c, ~self.first_half, h.direction)]
        time_ok = all(r.effect is not None and np.sign(r.effect) == sign for r in halves)
        bands, band_ok = {}, True
        for b, m in self.band_masks.items():
            r = self._test(t, c, m, "two-sided")
            if r.effect is None:
                continue
            bands[b] = {"effect": round(r.effect, 5), "ci": [round(r.ci_low, 5), round(r.ci_high, 5)], "n_treated": r.n_treated}
            flipped = np.sign(r.effect) == -sign and (r.ci_high < 0 if sign > 0 else r.ci_low > 0)
            if r.n_treated >= BAND_MIN_TREATED and flipped:
                band_ok = False
        rc = self._test(t, c, self.not_top, "two-sided")
        conc_p = directional_p(rc, sign, "two-sided") / 2 if rc.p is not None else 1.0  # one-sided in expected direction
        conc_ok = rc.effect is not None and np.sign(rc.effect) == sign and conc_p < ALPHA
        leak_ok = not h.is_leaky()
        checks = {"time_stability": time_ok, "subgroup_stability": band_ok,
                  "player_concentration": conc_ok, "leakage": leak_ok}
        return {"passed": all(checks.values()), "checks": checks,
                "time_halves": [r.effect for r in halves], "bands": bands,
                "without_top_active": {"effect": rc.effect, "p_one_sided": conc_p},
                "leaky_features": sorted(f for f in h.features_used() if f == "termination")}

    def _vault(self, hid: str, st: dict) -> None:
        h = self.hyps[hid]
        vdf = self.vault.read(hid)
        key = ("vault", h.spec_key())
        if key not in self.cache:
            t, c = groups(vdf, h)
            self.cache[key] = (np.packbits(t), np.packbits(c), len(t))
        tp, cp, n = self.cache[key]
        t, c = np.unpackbits(tp, count=n).astype(bool), np.unpackbits(cp, count=n).astype(bool)
        r = cluster_test(vdf[self.y_col].to_numpy(dtype=np.float64), t, c, vdf["player_code"].to_numpy(),
                         vdf["game_code"].to_numpy(), h.direction, raw=vdf["residual"].to_numpy(dtype=np.float64))
        sign = expected_sign(h, st["explore"]["effect"])
        dp = directional_p(r, sign, h.direction)
        passed = (not r.underpowered) and dp < ALPHA and abs(r.effect) >= h.min_effect
        self.ledger.record(hid, "vault", **r.to_dict(), passed=passed, directional_p=dp)

    # ---------------------------------------------------------------- pipeline
    def run(self) -> dict[str, dict]:
        """Process everything pending. Safe to call after every new batch."""
        state = self.ledger.state()
        for hid in self.hyps:
            if "explore" not in state[hid]["stages"]:
                self.explore(hid)
        state = self.ledger.state()
        tested = {hid: state[hid]["stages"]["explore"] for hid in self.hyps
                  if not state[hid]["stages"]["explore"]["underpowered"]}
        thr = bh_threshold([r["p"] for r in tested.values()])
        for hid, ex in tested.items():
            h = self.hyps[hid]
            st = state[hid]["stages"]
            if ex["p"] <= thr and abs(ex["effect"]) >= h.min_effect and "validate" not in st:
                self._validate_and_skeptic(hid, st)
                st = self.ledger.state()[hid]["stages"]
                if st.get("skeptic", {}).get("passed") and "vault" not in st:
                    self._vault(hid, st)
        state = self.ledger.state()
        for hid in self.hyps:
            v, at, why = self.verdict(self.hyps[hid], state[hid]["stages"], thr, len(tested))
            if (state[hid]["verdict"], state[hid].get("killed_at")) != (v, at):
                self.ledger.verdict(hid, v, at, why)
        return self.ledger.state()

    @staticmethod
    def verdict(h: Hypothesis, st: dict, thr: float, m: int) -> tuple[str, str | None, str]:
        ex = st["explore"]
        if ex["underpowered"]:
            return "UNDERPOWERED", "explore", f"n_treated={ex['n_treated']}, players={ex['n_players']}"
        if ex["p"] >= ALPHA:
            return "REJECTED", "explore", f"explore p={ex['p']:.3g} >= {ALPHA}"
        if ex["p"] > thr:
            return "LUCKY", "correction", f"explore p={ex['p']:.3g} fails BH (q={Q}, m={m}, threshold={thr:.3g})"
        if abs(ex["effect"]) < h.min_effect:
            return "LUCKY", "min_effect", f"|effect|={abs(ex['effect']):.4f} < min_effect={h.min_effect}"
        va = st.get("validate")
        if va is None:
            return "PENDING", None, ""
        if va["underpowered"]:
            return "UNDERPOWERED", "validate", f"validate n_treated={va['n_treated']}"
        if not va["passed"]:
            return "LUCKY", "validate", f"validate effect={va['effect']:.4f}, p={va['directional_p']:.3g}"
        sk = st["skeptic"]
        if not sk["passed"]:
            failed = [k for k, ok in sk["checks"].items() if not ok]
            return "LUCKY", "skeptic", "failed: " + ", ".join(failed)
        vl = st.get("vault")
        if vl is None:
            return "PENDING", None, ""
        if vl["underpowered"]:
            return "UNDERPOWERED", "vault", f"vault n_treated={vl['n_treated']}"
        if not vl["passed"]:
            return "LUCKY", "vault", f"vault effect={vl['effect']:.4f}, p={vl['directional_p']:.3g}"
        return "REAL", None, f"vault effect={vl['effect']:.4f} [{vl['ci_low']:.4f}, {vl['ci_high']:.4f}], p={vl['directional_p']:.2g}"


def print_report(state: dict) -> None:
    for hid, s in state.items():
        ex = s["stages"].get("explore", {})
        eff = f"{ex['effect']:+.4f}" if ex.get("effect") is not None else "   n/a "
        p = f"{ex['p']:.2g}" if ex.get("p") is not None else "n/a"
        stmt = s["raw"].get("statement", "")[:70]
        print(f"{hid} {s['verdict']:<12} {str(s.get('killed_at') or ''):<11} eff={eff} p={p:<8} {stmt}")
        if s["verdict"] != "REJECTED":
            print(f"      {s.get('reason')}")


def _main() -> None:
    """Stage-3 acceptance: hand-written bank through the full honest pipeline."""
    import argparse

    from src.bank import HAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="results/ledger_hand.jsonl")
    a = ap.parse_args()
    for p in (Path(a.ledger), Path("results/vault_access_hand.jsonl")):
        p.unlink(missing_ok=True)  # fresh run each time (a new run id = a new ledger)
    led = Ledger(a.ledger, run_id="hand")
    j = Judge(load_data(), led, vault_log="results/vault_access_hand.jsonl")
    j.submit(HAND, source="hand")
    print_report(j.run())


if __name__ == "__main__":
    _main()
