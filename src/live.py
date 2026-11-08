""""The vault is the future": commit surviving hypotheses, then test them on games played later.

    python -m src.live commit            # writes commitment.json, prints its SHA-256
    python -m src.live run --minutes 60  # fetch post-commit blitz games, update running estimates
    python -m src.live verify [HASH]     # recompute the hash; compare with commitment.sha256 / HASH

Data source (checked against the official Lichess OpenAPI spec, github.com/lichess-org/api):
  * with LICHESS_TOKEN: GET /api/games/user/{username}?since=..&perfType=blitz&rated=true (ndjson).
    Anonymous calls to this endpoint returned 404 when tested, so it is used only with a token.
  * without a token: GET /api/user/{username}/current-game (anonymous OK), polled round-robin over a
    fixed player pool chosen at commit time. Lower throughput; games played between polls can be missed.
  Rate limits (spec): one request at a time; on HTTP 429 wait a full minute.

Running estimates use an anytime-valid confidence sequence, so watching them live does not
inflate false positives (see NOTES.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features import clean_games, player_games
from src.hypothesis import groups, validate
from src.judge import bh_threshold
from src.tester import cluster_test, p_value

COMMIT = Path("commitment.json")
COMMIT_HASH = Path("commitment.sha256")
LIVE_DIR = Path("results/live")
GAMES_FILE = LIVE_DIR / "games.jsonl"
RESULTS_FILE = LIVE_DIR / "live_results.json"
API = "https://lichess.org"
UA = {"User-Agent": "honest-ai-researcher (hackathon research project)"}
POOL_PER_BAND = 60
MAX_PAUSE = 8.0
CANDIDATES_PER_BAND = 400   # most active October players per band, then filtered to recently online
RECENT_HOURS = 2
ALPHA = 0.05
HISTORY_LOOKBACK_H = 3
LIVE_TARGET_ROWS = 5000  # planned live sample; fixes the CS mixture parameter rho at commit time
FINAL_TEST_HOURS = 3.0  # default pre-registered time of the one-shot final test (after commit)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ HTTP (one request at a time)
class Lichess:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.s = requests.Session()
        self.s.headers.update(UA)
        tok = os.environ.get("LICHESS_TOKEN")
        self.authed = bool(tok)
        self.pause = self.min_pause = 1.0  # seconds between requests; adapts to 429s
        if tok:
            self.s.headers["Authorization"] = f"Bearer {tok}"

    def get(self, path, **kw):
        """One request at a time; on 429 wait a minute (per the API docs) and slow down."""
        while True:
            r = self.s.get(API + path, timeout=60, **kw)
            if r.status_code == 429:
                self.pause = min(self.pause * 2, MAX_PAUSE)
                print(f"429 rate-limited: sleeping 60s, pause now {self.pause:.1f}s", flush=True)
                time.sleep(60)
                continue
            self.pause = max(self.pause * 0.98, self.min_pause)
            time.sleep(self.pause)
            return r

    def post(self, path, data, **kw):
        while True:
            r = self.s.post(API + path, data=data, timeout=60, **kw)
            if r.status_code == 429:
                time.sleep(60)
                continue
            return r

    def existing_users(self, ids: list[str]) -> dict[str, int]:
        """{user id: seenAt ms} for open, non-flagged accounts."""
        out = {}
        for i in range(0, len(ids), 300):  # POST /api/users takes up to 300 ids
            r = self.post("/api/users", ",".join(ids[i:i + 300]), headers={"Content-Type": "text/plain"})
            r.raise_for_status()
            out.update({u["id"]: u.get("seenAt", 0) for u in r.json()
                        if not u.get("disabled") and not u.get("tosViolation")})
        return out

    def user_games(self, user: str, since_ms: int) -> list[dict]:
        r = self.get(f"/api/games/user/{user}", params={"since": since_ms, "perfType": "blitz", "rated": "true",
                                                        "moves": "false", "opening": "true", "finished": "true"},
                     headers={"Accept": "application/x-ndjson"}, stream=True)
        if r.status_code != 200:
            return []
        return [json.loads(l) for l in r.iter_lines() if l]

    def current_game(self, user: str) -> dict | None:
        r = self.get(f"/api/user/{user}/current-game", params={"moves": "false", "opening": "true"},
                     headers={"Accept": "application/json"})
        return r.json() if r.status_code == 200 else None


# ------------------------------------------------------------------ JSON game -> PGN-header row
TERM = {"mate": "Normal", "resign": "Normal", "stalemate": "Normal", "draw": "Normal", "outoftime": "Time forfeit",
        "timeout": "Abandoned", "noStart": "Abandoned", "aborted": None, "cheat": "Rules infraction",
        "variantEnd": "Normal", "insufficientMaterialClaim": "Insufficient material"}


def game_to_header(g: dict) -> dict | None:
    if not g.get("rated") or g.get("speed") != "blitz" or g.get("variant") != "standard":
        return None
    if g.get("status") in ("created", "started") or TERM.get(g.get("status"), "Normal") is None:
        return None
    w, b = g["players"]["white"], g["players"]["black"]
    if "user" not in w or "user" not in b or "rating" not in w or "rating" not in b:
        return None
    winner = g.get("winner")
    result = "1-0" if winner == "white" else "0-1" if winner == "black" else "1/2-1/2"
    t = datetime.fromtimestamp(g["createdAt"] / 1000, tz=timezone.utc)
    clock = g.get("clock") or {}
    return {"Event": "Rated Blitz tournament" if g.get("tournament") else "Rated Blitz game",
            "Site": f"https://lichess.org/{g['id']}", "White": w["user"]["name"], "Black": b["user"]["name"],
            "Result": result, "UTCDate": t.strftime("%Y.%m.%d"), "UTCTime": t.strftime("%H:%M:%S"),
            "WhiteElo": str(w["rating"]), "BlackElo": str(b["rating"]),
            "TimeControl": f"{clock.get('initial', 0)}+{clock.get('increment', 0)}",
            "ECO": (g.get("opening") or {}).get("eco", "?"), "Opening": (g.get("opening") or {}).get("name", ""),
            "Termination": TERM.get(g.get("status"), "Normal")}


# ------------------------------------------------------------------ anytime-valid confidence sequence
def confidence_sequence(s_centered: float, var_s: float, scale: float, rho: float, alpha: float = ALPHA):
    """Two-sided normal-mixture boundary (Robbins 1970; Howard et al. 2021, eq. 14):
        |S_t - E0 S_t| < sqrt((V_t + rho) * (log(1 + V_t/rho) + 2 log(1/alpha)))  for all t, w.p. >= 1-alpha
    applied to the centred within-player permutation sum (asymptotic CS, Waudby-Smith et al. 2021).
    Returns (lo, hi) on the effect scale."""
    radius = math.sqrt((var_s + rho) * (math.log(1 + var_s / rho) + 2 * math.log(1 / alpha)))
    eff = s_centered * scale
    return eff - radius * scale, eff + radius * scale


# ------------------------------------------------------------------ commands
def cmd_commit(args) -> None:
    from src.judge import load_data
    from src.ledger import Ledger

    led = Ledger(args.ledger)
    hyps = []
    for hid, s in led.state().items():
        st_ = s["stages"]
        if st_.get("validate", {}).get("passed") and st_.get("skeptic", {}).get("passed"):
            ex, va = st_["explore"], st_["validate"]
            spec = validate({**s["raw"], "id": hid})
            # rho = permutation variance expected at LIVE_TARGET_ROWS rows, scaled from validate. Fixed now,
            # before any live data, so the confidence sequence stays anytime-valid.
            rho = float(va["var_s"]) * LIVE_TARGET_ROWS / (va["n_treated"] + va["n_control"])
            hyps.append({"id": hid, "spec": spec.to_dict(), "predicted_direction": spec.direction, "cs_rho": rho,
                         "explore": {k: ex[k] for k in ("effect", "ci_low", "ci_high", "p", "n_treated")},
                         "validate": {k: va[k] for k in ("effect", "ci_low", "ci_high", "p", "n_treated")},
                         "historical_vault_verdict": s["verdict"]})
    if not hyps:
        sys.exit("No hypotheses passed validate + skeptic in this ledger; nothing to commit.")
    # Fixed player pool: the most active players of the most recent (vault) period, per rating band.
    df = load_data()
    vault = df[df["split"] == "vault"]
    counts = vault.groupby(["rating_band", "player"], observed=True).size().reset_index(name="n")
    cand = (counts.sort_values("n", ascending=False).groupby("rating_band", observed=True)
            .head(CANDIDATES_PER_BAND)).copy()
    li = Lichess()
    seen = li.existing_users([p.lower() for p in cand["player"].astype(str)])
    cand["seen_at"] = cand["player"].astype(str).str.lower().map(seen)
    recent = utcnow().timestamp() * 1000 - RECENT_HOURS * 3600 * 1000
    cand = cand[cand["seen_at"] >= recent].sort_values("seen_at", ascending=False)
    pool = (cand.groupby("rating_band", observed=True).head(POOL_PER_BAND)["player"].astype(str).tolist())
    # Player baseline for live y_adj: mean historical residual (October data, disjoint from live games).
    hist = df[df["player"].astype(str).isin(pool)].groupby(df["player"].astype(str), observed=True)["residual"].mean()
    baseline = {k.lower(): round(float(v), 6) for k, v in hist.items()}
    now = utcnow()
    final_at = (now + pd.Timedelta(hours=args.final_hours)).isoformat(timespec="seconds")
    commitment = {"committed_at": now.isoformat(timespec="seconds"), "ledger": args.ledger,
                  "final_test_at": final_at,
                  "final_test": "one-shot: games started in [committed_at, final_test_at); one-sided p in the "
                                "validated direction; Benjamini-Hochberg q=0.05 across all committed hypotheses; "
                                "pass = BH-significant and |effect| >= min_effect",
                  "source_ledger_sha256": sha256(Path(args.ledger)),
                  "test": "within-player permutation, player-adjusted mean residual difference; "
                          "pass = same direction, 95% anytime-valid CS excludes 0",
                  "hypotheses": hyps, "player_pool": pool, "player_baseline": baseline}
    COMMIT.write_text(json.dumps(commitment, indent=2, sort_keys=True))
    h = sha256(COMMIT)
    COMMIT_HASH.write_text(h + "\n")
    print(f"committed {len(hyps)} hypotheses, pool of {len(pool)} players at {commitment['committed_at']}")
    print(f"final one-shot test pre-registered at {final_at}")
    print(f"SHA-256: {h}")


def load_games() -> dict[str, dict]:
    if not GAMES_FILE.exists():
        return {}
    return {g["id"]: g for g in (json.loads(l) for l in GAMES_FILE.read_text().splitlines() if l)}


def save_new(games: dict, new: list[dict]) -> int:
    added = 0
    with GAMES_FILE.open("a") as f:
        for g in new:
            if g and g.get("id") and g["id"] not in games and g.get("status") not in ("created", "started"):
                games[g["id"]] = g
                f.write(json.dumps(g) + "\n")
                added += 1
    return added


def evaluate(commitment: dict, games: dict) -> dict:
    t0 = pd.Timestamp(commitment["committed_at"])
    heads = [h for g in games.values() if (h := game_to_header(g))]
    pool = {p.lower() for p in commitment["player_pool"]}
    out = {"updated_at": utcnow().isoformat(timespec="seconds"), "n_games": 0, "n_rows": 0, "results": []}
    if heads:
        pg = player_games(clean_games(pd.DataFrame(heads)))
        pg = pg[pg["player"].astype(str).str.lower().isin(pool) & (pg["start"] >= t0)].reset_index(drop=True)
        pg["player_code"] = pg["player"].astype(str).astype("category").cat.codes.astype(np.int32)
        base = commitment["player_baseline"]
        pg["y_adj"] = pg["residual"].astype(float) - pg["player"].astype(str).str.lower().map(base).astype(float)
        out["n_games"] = int(pg["game_idx"].nunique())
        out["n_rows"] = int(len(pg))
    for c in commitment["hypotheses"]:
        h = validate(c["spec"])
        row = {"id": c["id"], "statement": h.statement, "direction": h.direction,
               "predicted_effect": c["validate"]["effect"], "effect": None, "lo": None, "hi": None,
               "n_treated": 0, "n_control": 0, "status": "waiting for data"}
        if heads and len(pg):
            t, cmask = groups(pg, h)
            r = cluster_test(pg["y_adj"].to_numpy(dtype=np.float64), t, cmask, pg["player_code"].to_numpy(),
                             pg["game_idx"].to_numpy(), h.direction, raw=pg["residual"].to_numpy(dtype=np.float64))
            row.update(n_treated=r.n_treated, n_control=r.n_control)
            if r.var_s and r.n_treated >= 30:  # the CS is valid at any n; just avoid printing noise
                lo, hi = confidence_sequence(r.s_centered, r.var_s, r.scale, rho=c["cs_rho"])
                sign = 1 if c["validate"]["effect"] > 0 else -1
                status = ("confirmed" if (lo > 0 if sign > 0 else hi < 0) else
                          "contradicted" if (hi < 0 if sign > 0 else lo > 0) else "inconclusive")
                row.update(effect=r.effect, lo=lo, hi=hi, status=status)
        out["results"].append(row)
    if commitment.get("final_test_at") and heads:
        out["final_test"] = final_test(commitment, pg)
    return out


def final_test(commitment: dict, pg: pd.DataFrame) -> dict:
    """Pre-registered one-shot test. Only reported once final_test_at has passed, on games before it."""
    t_final = pd.Timestamp(commitment["final_test_at"])
    if pd.Timestamp(utcnow()) < t_final:
        return {"status": "pending", "at": commitment["final_test_at"]}
    d = pg[pg["start"] < t_final].reset_index(drop=True)
    rows = []
    for c in commitment["hypotheses"]:
        h = validate(c["spec"])
        sign = 1 if c["validate"]["effect"] > 0 else -1
        t, cmask = groups(d, h)
        r = cluster_test(d["y_adj"].to_numpy(dtype=np.float64), t, cmask, d["player_code"].to_numpy(),
                         d["game_idx"].to_numpy(), h.direction, raw=d["residual"].to_numpy(dtype=np.float64))
        p = p_value(r.z, "greater" if sign > 0 else "less") if r.z is not None else 1.0
        rows.append({"id": c["id"], "effect": r.effect, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": p,
                     "n_treated": r.n_treated, "min_effect": h.min_effect})
    thr = bh_threshold([x["p"] for x in rows])
    for x in rows:
        big = x["effect"] is not None and abs(x["effect"]) >= x["min_effect"]
        x["verdict"] = "CONFIRMED" if (x["p"] <= thr and big) else "NOT CONFIRMED"
    return {"status": "done", "at": commitment["final_test_at"], "n_rows": int(len(d)),
            "bh_threshold": thr, "results": rows}


def cmd_run(args) -> None:
    if not COMMIT.exists():
        sys.exit("run `python -m src.live commit` first")
    commitment = json.loads(COMMIT.read_text())
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    t_commit = pd.Timestamp(commitment["committed_at"])
    since_ms = int((t_commit - pd.Timedelta(hours=HISTORY_LOOKBACK_H)).timestamp() * 1000)
    li = Lichess()
    li.pause = li.min_pause = args.pause
    games = load_games()
    pool = commitment["player_pool"]
    deadline = time.time() + args.minutes * 60
    mode = "export (token)" if li.authed else "current-game polling (anonymous)"
    print(f"live run: {len(pool)} players, mode={mode}, until {args.minutes} min from now", flush=True)
    rnd = 0
    while True:
        rnd += 1
        added = 0
        for i, user in enumerate(pool):
            if li.authed:
                # incremental: only games newer than the latest one we already hold for this player
                mine = [g["createdAt"] for g in games.values()
                        if user.lower() in {(g["players"][c].get("user") or {}).get("id", "") for c in ("white", "black")}]
                added += save_new(games, li.user_games(user, max([since_ms] + [m + 1 for m in mine])))
            else:
                g = li.current_game(user)
                added += save_new(games, [g] if g else [])
            if time.time() > deadline:
                break
        res = evaluate(commitment, games)
        res["mode"] = mode
        RESULTS_FILE.write_text(json.dumps(res, indent=2, default=float))
        print(f"round {rnd}: +{added} games, post-commit player-rows={res['n_rows']}", flush=True)
        for r in res["results"]:
            eff = "n/a" if r["effect"] is None else f"{r['effect']:+.4f} CS[{r['lo']:+.4f}, {r['hi']:+.4f}]"
            print(f"   {r['id']} {r['status']:<16} {eff}  n_T={r['n_treated']}  {r['statement'][:60]}", flush=True)
        if time.time() > deadline or args.once:
            break


def cmd_verify(args) -> None:
    h = sha256(COMMIT)
    expected = args.hash or (COMMIT_HASH.read_text().strip() if COMMIT_HASH.exists() else None)
    print(f"SHA-256(commitment.json) = {h}")
    if expected is None:
        print("no published hash to compare against")
    elif h == expected:
        print("VERIFIED: matches the published hash; commitment unchanged.")
    else:
        print(f"MISMATCH: published {expected}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit")
    c.add_argument("--ledger", default="results/ledger_main.jsonl")
    c.add_argument("--final-hours", type=float, default=FINAL_TEST_HOURS,
                   help="hours after commit at which the one-shot final test is taken")
    r = sub.add_parser("run")
    r.add_argument("--minutes", type=float, default=60)
    r.add_argument("--pause", type=float, default=2.0, help="minimum seconds between requests (one at a time; grows after 429s)")
    r.add_argument("--once", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("hash", nargs="?")
    a = ap.parse_args()
    {"commit": cmd_commit, "run": cmd_run, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
