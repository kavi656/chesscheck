"""Naive mode (the strawman) and its fate under honest testing.

Naive = explore split only, no correction, no validation, no skeptic, no vault, and the
proposer may refine hypotheses after seeing explore p-values (up to 3 rounds). Anything
with p < 0.05 is a "discovery". Then every hypothesis the naive process tested is fed to
the honest pipeline (so the correction counts everything that was actually tried), and we
report what happened to each naive discovery.

Refiners:
  llm   - the same LLM proposer, shown explore p-values and asked to refine (spec behaviour)
  auto  - deterministic p-hacker: for each failed hypothesis, try rating-band subsets and
          nearby numeric thresholds, keep the best p. No API calls, so it can run on many
          null seeds.

Usage: python -m src.naive --data real --refiner llm
       python -m src.naive --data null --refiner auto --seeds 50
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.bank import HAND
from src.features import BAND_LABELS
from src.hypothesis import FEATURES, InvalidHypothesis, validate
from src.judge import ALPHA, Judge, load_data
from src.ledger import Ledger

ROUNDS = 3
OUT = Path("results")


def explore_test(judge: Judge, raw: dict, tag: str) -> dict | None:
    try:
        h = validate({**raw, "id": tag})
    except InvalidHypothesis:
        return None
    t, c = judge._masks(h)
    r = judge._test(t, c, judge.is_explore, h.direction)
    return {"raw": raw, "p": r.p, "effect": r.effect, "n_treated": r.n_treated, "underpowered": r.underpowered,
            "discovery": bool(r.p is not None and r.p < ALPHA and not r.underpowered)}


def auto_variants(raw: dict) -> list[dict]:
    """Forking paths: band subsets, shifted numeric thresholds, and direction flips."""
    out = []
    for b in BAND_LABELS:
        v = copy.deepcopy(raw)
        v["subset"] = [c for c in v.get("subset", []) if c["feature"] != "rating_band"] + \
                      [{"feature": "rating_band", "op": "==", "value": b}]
        v["statement"] = f"{raw['statement']} [only {b}]"
        out.append(v)
    for i, c in enumerate(raw["condition"]):
        if FEATURES[c["feature"]]["type"] == "numeric" and c["op"] in ("<", "<=", ">", ">="):
            for mult in (0.5, 2.0, 3.0):
                v = copy.deepcopy(raw)
                nv = c["value"] * mult
                v["condition"][i]["value"] = int(round(nv)) if float(c["value"]).is_integer() else nv
                v["statement"] = f"{raw['statement']} [threshold {c['value']} -> {v['condition'][i]['value']}]"
                out.append(v)
    if raw["direction"] != "two-sided":
        v = copy.deepcopy(raw)
        v["direction"] = "greater" if raw["direction"] == "less" else "less"
        v["statement"] = f"{raw['statement']} [direction flipped]"
        out.append(v)
    return out


def naive_run(judge: Judge, initial: list[dict], refiner: str, summary: dict | None = None,
              batch_prefix: str = "naive") -> dict:
    tested: list[dict] = []
    current = list(initial)
    for rnd in range(ROUNDS + 1):
        results = [r for i, raw in enumerate(current)
                   if (r := explore_test(judge, raw, f"N{rnd}-{i}")) is not None]
        for r in results:
            r["round"] = rnd
        tested += results
        failed = [r for r in results if not r["discovery"]]
        if rnd == ROUNDS or not failed:
            break
        if refiner == "auto":
            current = []
            for r in failed:
                trials = [x for v in auto_variants(r["raw"]) if (x := explore_test(judge, v, "tmp")) is not None]
                tested += [{**x, "round": rnd + 0.5, "variant_trial": True} for x in trials]
                best = min((x for x in trials if x["p"] is not None and not x["underpowered"]),
                           key=lambda x: x["p"], default=None)
                if best is not None:
                    current.append(best["raw"])
        else:
            from src.proposer import propose
            fb = "Explore-split results of your hypotheses (p-values are one-sided in your stated direction):\n" + \
                 "\n".join(f"- {r['raw']['statement']}: effect={r['effect'] if r['effect'] is None else round(r['effect'], 4)}, "
                           f"p={r['p'] if r['p'] is None else round(r['p'], 4)}, n_treated={r['n_treated']}"
                           for r in results) + \
                 f"\n\n{len(failed)} were not significant at p<0.05. Refine them (narrow the subset, change thresholds, " \
                 "flip the direction, or try related conditions) to find significant effects. Propose one refined " \
                 "hypothesis for each failure."
            current, _ = propose(summary, [r["raw"]["statement"] for r in tested], n=len(failed),
                                 batch=f"{batch_prefix}-refine{rnd + 1}", feedback=fb)
    discoveries = [r for r in tested if r["discovery"] and not r.get("variant_trial")]
    return {"tested": tested, "discoveries": discoveries}


def honest_fate(judge: Judge, naive: dict) -> list[dict]:
    """Feed everything the naive process tested into the honest pipeline; report fate of discoveries."""
    key = lambda raw: validate({**raw, "id": "k"}).spec_key()
    index, uniq = {}, []
    for r in naive["tested"]:  # each distinct spec enters the honest ledger once
        k = key(r["raw"])
        if k not in index:
            index[k] = len(uniq)
            uniq.append(r["raw"])
    ids = judge.submit(uniq, source="naive")
    st = judge.run()
    fate, seen = [], set()
    for d in naive["discoveries"]:
        k = key(d["raw"])
        if k in seen:
            continue
        seen.add(k)
        hid = ids[index[k]]
        s = st[hid]
        fate.append({"id": hid, "statement": d["raw"]["statement"], "naive_p": d["p"], "naive_effect": d["effect"],
                     "round": d["round"], "verdict": s["verdict"], "killed_at": s.get("killed_at"),
                     "reason": s.get("reason")})
    return fate


def initial_bank(refiner: str, summary: dict | None, batch: str) -> list[dict]:
    if refiner == "llm":
        from src.proposer import propose
        hyps, _ = propose(summary, [], n=20, batch=batch)
        return hyps
    out = [dict(h) for h in HAND]
    for f in ("results/cache/llm_bank.json", "results/cache/offline_bank.json"):
        if Path(f).exists():
            out += json.loads(Path(f).read_text())[:20]
    return out


def main() -> None:
    from src.calibrate import ShuffleNull
    from src.proposer import explore_summary, get_summary

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["real", "null"], default="real")
    ap.add_argument("--refiner", choices=["llm", "auto"], default="auto")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    df = load_data()
    runs, null = [], None
    for seed in range(a.seeds if a.data == "null" else 1):
        d = df
        if a.data == "null":
            null = null if seed else ShuffleNull(df)
            d = null.sample(20_000 + seed)  # same outcome-shuffle null as calibration (distinct seeds)
            summary = explore_summary(d[d["split"] == "explore"]) if a.refiner == "llm" else None
        else:
            summary = get_summary(df) if a.refiner == "llm" else None
        tag = f"naive-{a.data}-{a.refiner}-s{seed}"
        judge = Judge(d, Ledger(OUT / f"ledger_{tag}.jsonl" if seed == 0 else None, run_id=tag))
        if seed == 0:
            (OUT / f"ledger_{tag}.jsonl").unlink(missing_ok=True)
            judge.ledger = Ledger(OUT / f"ledger_{tag}.jsonl", run_id=tag)
        naive = naive_run(judge, initial_bank(a.refiner, summary, tag), a.refiner, summary, tag)
        fate = honest_fate(judge, naive)
        n_real = sum(f["verdict"] == "REAL" for f in fate)
        print(f"[{tag}] tested={len(naive['tested'])} naive discoveries={len(fate)} "
              f"-> honest REAL={n_real}", flush=True)
        runs.append({"seed": seed, "n_tested": len(naive["tested"]),
                     "n_discoveries": len(fate), "fate": fate})
    out = {"data": a.data, "refiner": a.refiner, "runs": runs,
           "mean_naive_discoveries": float(np.mean([r["n_discoveries"] for r in runs])),
           "mean_honest_real": float(np.mean([sum(f["verdict"] == "REAL" for f in r["fate"]) for r in runs]))}
    path = OUT / f"naive_{a.data}_{a.refiner}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))
    for f in runs[0]["fate"]:
        print(f"  {f['verdict']:<12} {str(f['killed_at'] or ''):<11} naive p={f['naive_p']:.3g}  {f['statement'][:80]}")


if __name__ == "__main__":
    main()
