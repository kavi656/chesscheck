"""Calibration harness: prove the false-discovery rate on null data and measure power.

Null datasets ("outcome shuffle"): within each player, the outcomes of their games (score, residual,
termination) are randomly re-ordered in time, and every outcome-derived sequence feature
(prev_result, prev_termination, streak) is RECOMPUTED from the shuffled order. Contexts (time, colour,
opponent, rating, session position) stay put. This keeps each player's skill and results distribution
but destroys any real link between context/history and outcome - while keeping every mechanical
artefact of building conditions out of a player's own results (the Miller-Sanjurjo bias). A null that
only permutes residuals with the labels held fixed would NOT contain that artefact and would pass a
biased test; we learned this the hard way (NOTES.md).

Planted runs: on a null dataset, subtract `effect` from residual for rows matching the planted
hypothesis's condition (all splits), and check whether the pipeline calls it REAL.

Usage: python -m src.calibrate [--seeds 50] [--plant-seeds 50]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.bank import HAND
from src.features import PREV_RESULTS
from src.hypothesis import all_mask, validate
from src.judge import ALPHA, Judge, add_adjusted_outcome, load_data
from src.ledger import Ledger
from src.tester import OUTCOME_FEATURES

OUT = Path("results/calibration.json")
LLM_BANK = Path("results/cache/llm_bank.json")
PLANT_INDEX = 0                      # plant the effect on HAND[0] (tilt) by default
EFFECTS = [0.005, 0.01, 0.02, 0.04]


OFFLINE_BANK = Path("results/cache/offline_bank.json")


def hypothesis_bank() -> list[dict]:
    """Hand-written + every cached generated hypothesis (LLM and offline). No API calls."""
    bank = [dict(h) for h in HAND]
    for f in (LLM_BANK, OFFLINE_BANK):
        if f.exists():
            bank += json.loads(f.read_text())
    return bank


def within_player_permutation(player: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Permute y within each player (any row order)."""
    order = np.argsort(player, kind="stable")
    perm = np.lexsort((rng.random(len(player)), player))
    out = np.empty_like(y)
    out[order] = y[perm]
    return out


class ShuffleNull:
    """Outcome-shuffle null generator. `df` must be sorted by (player_code, start) as load_data does."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        p = df["player_code"].to_numpy()
        assert np.all(p[1:] >= p[:-1]), "data must be sorted by player"
        self.player = p
        self.first = np.r_[True, p[1:] != p[:-1]]
        self.score = df["score"].to_numpy(dtype=np.float64)
        self.resid = df["residual"].to_numpy(dtype=np.float64)
        self.term_codes = df["termination"].cat.codes.to_numpy()
        self.term_cats = df["termination"].cat.categories
        self.pterm_cats = df["prev_termination"].cat.categories
        # map termination code -> prev_termination code
        self.term_to_pterm = np.array([self.pterm_cats.get_loc(c) for c in self.term_cats])
        self.orig_prev_result = df["prev_result"].cat.codes.to_numpy()
        self.orig_prev_term = df["prev_termination"].cat.codes.to_numpy()
        self.orig_streak = df["streak"].to_numpy().astype(np.int64)
        assert list(df["prev_result"].cat.categories) == PREV_RESULTS

    def sample(self, seed: int, plant: dict | None = None, effect: float = 0.0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = np.lexsort((rng.random(len(self.player)), self.player))  # df is player-sorted: within-player shuffle
        score, resid, term = self.score[idx], self.resid[idx], self.term_codes[idx]
        first = self.first
        # prev_result / prev_termination: from the previous (shuffled) game; first analysed game keeps
        # its real predecessor (from the burn-in day), which the shuffle doesn't touch.
        score_code = np.select([score == 1.0, score == 0.5], [0, 1], default=2)
        prev_result = np.where(first, self.orig_prev_result, np.r_[0, score_code[:-1]])
        prev_term = np.where(first, self.orig_prev_term, np.r_[0, self.term_to_pterm[term][:-1]])
        # streak: run length of identical decisive results ending at the previous game
        o = np.sign(score - 0.5).astype(np.int64)
        newrun = first | (o != np.r_[2, o[:-1]])
        run_start = np.nonzero(newrun)[0]
        run_id = np.cumsum(newrun) - 1
        cnt = np.arange(len(o)) - run_start[run_id] + 1
        carry_ok = first[run_start] & (np.sign(self.orig_streak[run_start]) == o[run_start])
        carry = np.where(carry_ok, np.abs(self.orig_streak[run_start]), 0)[run_id]
        after = (cnt + carry) * o
        streak = np.where(first, self.orig_streak, np.r_[0, after[:-1]])
        d = self.df.assign(
            score=score.astype(np.float32), residual=resid.astype(np.float32),
            termination=pd.Categorical.from_codes(term, self.term_cats),
            prev_result=pd.Categorical.from_codes(prev_result, PREV_RESULTS),
            prev_termination=pd.Categorical.from_codes(prev_term, self.pterm_cats),
            streak=streak.astype(np.int16))
        if plant is not None and effect:
            h = validate({**plant, "id": "plant"})
            m = all_mask(d, h.subset + h.condition)
            d["residual"] = (d["residual"].to_numpy(dtype=np.float64) - effect * m).astype(np.float32)
        return add_adjusted_outcome(d)


class Calibrator:
    def __init__(self, data: pd.DataFrame, bank: list[dict]):
        self.null = ShuffleNull(data)
        self.bank = bank
        self.shared_cache: dict = {}
        self.static_keys = set()
        for i, raw in enumerate(bank):
            try:
                h = validate({**raw, "id": f"C{i}"})
            except Exception:
                continue
            if not (h.features_used() & OUTCOME_FEATURES):
                self.static_keys.add(h.spec_key())

    def run_once(self, data: pd.DataFrame, tag: str) -> dict:
        cache = {k: v for k, v in self.shared_cache.items() if k[1] in self.static_keys}
        j = Judge(data, Ledger(None, tag), mask_cache=cache)
        j.submit(self.bank, source="calibration")
        st = j.run()
        self.shared_cache.update({k: v for k, v in cache.items() if k[1] in self.static_keys})
        rows = []
        for hid, s in st.items():
            ex = s["stages"].get("explore", {})
            rows.append({"id": hid, "verdict": s["verdict"], "killed_at": s.get("killed_at"),
                         "explore_p": ex.get("p"), "explore_effect": ex.get("effect"),
                         "naive_discovery": bool(ex.get("p") is not None and ex["p"] < ALPHA and not ex.get("underpowered"))})
        return {"tag": tag, "rows": rows}

    def null_runs(self, seeds: int) -> list[dict]:
        out = []
        for s in range(seeds):
            t = time.time()
            r = self.run_once(self.null.sample(s), tag=f"null-{s}")
            n_real = sum(x["verdict"] == "REAL" for x in r["rows"])
            n_naive = sum(x["naive_discovery"] for x in r["rows"])
            print(f"null seed {s:3d}: REAL={n_real} naive_discoveries={n_naive} ({time.time()-t:.1f}s)", flush=True)
            out.append(r)
        return out

    def planted_runs(self, seeds: int, effects=EFFECTS, plant_index: int = PLANT_INDEX) -> dict:
        global PLANT_INDEX
        PLANT_INDEX = plant_index
        out = {}
        for eff in effects:
            hits, others = [], []
            for s in range(seeds):
                d = self.null.sample(10_000 + s, plant=self.bank[PLANT_INDEX], effect=eff)
                r = self.run_once(d, tag=f"plant-{eff}-{s}")
                hits.append(r["rows"][PLANT_INDEX]["verdict"] == "REAL")
                others.append(sum(x["verdict"] == "REAL" for i, x in enumerate(r["rows"]) if i != PLANT_INDEX))
                if s % 10 == 0:
                    print(f"plant {eff}: seed {s} detected={hits[-1]} other REAL={others[-1]}", flush=True)
            out[str(eff)] = {"detected": int(sum(hits)), "seeds": seeds, "power": float(np.mean(hits)),
                             "other_real_per_run": float(np.mean(others))}
            print(f"effect {eff}: power {np.mean(hits):.2f}", flush=True)
        return out


def summarise(null: list[dict], planted: dict, bank: list[dict]) -> dict:
    tested = [x for r in null for x in r["rows"] if x["verdict"] not in ("INVALID", "DUPLICATE", "UNDERPOWERED")]
    n_real = sum(x["verdict"] == "REAL" for x in tested)
    any_real = np.mean([any(x["verdict"] == "REAL" for x in r["rows"]) for r in null])
    naive = [sum(x["naive_discovery"] for x in r["rows"]) for r in null]
    pvals = [x["explore_p"] for x in tested if x["explore_p"] is not None]
    stage_counts: dict[str, int] = {}
    for x in tested:
        k = x["verdict"] if x["verdict"] in ("REAL", "REJECTED") else f"LUCKY@{x['killed_at']}"
        stage_counts[k] = stage_counts.get(k, 0) + 1
    false_real_ids = sorted({x["id"] for x in tested if x["verdict"] == "REAL"})
    return {
        "null_type": "outcome shuffle within player, sequence features recomputed",
        "bank_size": len(bank), "null_seeds": len(null),
        "null_hypothesis_tests": len(tested),
        "false_real_rate_per_hypothesis": n_real / max(1, len(tested)),
        "false_real_count": n_real, "false_real_ids": false_real_ids,
        "fraction_runs_with_any_real": float(any_real),
        "naive_discoveries_per_run_mean": float(np.mean(naive)),
        "naive_discoveries_per_run": naive,
        "naive_false_rate_per_hypothesis": float(np.sum(naive) / max(1, len(tested))),
        "null_explore_p_hist": np.histogram(pvals, bins=20, range=(0, 1))[0].tolist(),
        "null_fate_counts": stage_counts,
        "planted_hypothesis": bank[PLANT_INDEX]["statement"],
        "power": planted,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--plant-seeds", type=int, default=50)
    ap.add_argument("--reuse-power", action="store_true",
                    help="keep the planted-signal results already in calibration.json (null runs only)")
    ap.add_argument("--power-only", type=int, metavar="PLANT_INDEX",
                    help="only run planted-signal runs on this bank index; writes results/power_<index>.json")
    a = ap.parse_args()
    data = load_data()
    if a.power_only is not None:
        bank = hypothesis_bank()
        cal = Calibrator(data, bank)
        t = time.time()
        pw = cal.planted_runs(a.plant_seeds, plant_index=a.power_only)
        h = validate({**bank[a.power_only], "id": "plant"})
        n_t = int(all_mask(cal.null.df[cal.null.df["split"] == "explore"], h.subset + h.condition).sum())
        out = {"planted_hypothesis": bank[a.power_only]["statement"], "plant_index": a.power_only,
               "explore_treated_rows": n_t, "power": pw, "runtime_s": round(time.time() - t, 1)}
        Path(f"results/power_{a.power_only}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return
    bank = hypothesis_bank()
    cal = Calibrator(data, bank)
    t = time.time()
    null = cal.null_runs(a.seeds)
    if a.reuse_power:
        prev = json.loads(OUT.read_text())
        planted = prev["power"]
        planted["_note"] = f"from earlier run with bank_size={prev['bank_size']}"
    else:
        planted = cal.planted_runs(a.plant_seeds)
    summary = summarise(null, planted, bank)
    summary["runtime_s"] = round(time.time() - t, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("naive_discoveries_per_run", "null_explore_p_hist")}, indent=2))
    ok = summary["false_real_rate_per_hypothesis"] <= 0.05
    print("ACCEPTANCE:", "PASS" if ok else "FAIL", f"(false REAL rate {summary['false_real_rate_per_hypothesis']:.4f} <= 0.05)")


if __name__ == "__main__":
    main()
