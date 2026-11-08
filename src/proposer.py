"""LLM hypothesis proposer.

The LLM sees ONLY: the feature schema, aggregate summary stats from the EXPLORE split, and
the statements already in the ledger. It never sees validate or vault data.
Every batch's exact prompt and raw response are stored in results/proposer/.

Usage: python -m src.proposer --n 20      (proposes, then runs the honest pipeline)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.hypothesis import FEATURES

MODEL = "claude-sonnet-5-5"
OUT_DIR = Path("results/proposer")
SUMMARY_CACHE = Path("results/cache/explore_summary.json")
MAX_RETRIES = 3

CLAUSE = {
    "type": "object",
    "properties": {
        "feature": {"type": "string", "enum": sorted(FEATURES)},
        "op": {"type": "string", "enum": ["==", "!=", "<", "<=", ">", ">=", "between", "in", "not_in"]},
        "value": {"anyOf": [{"type": "string"}, {"type": "number"},
                            {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}}]},
    },
    "required": ["feature", "op", "value"],
    "additionalProperties": False,
}
HYP = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "subset": {"type": "array", "items": CLAUSE},
        "condition": {"type": "array", "items": CLAUSE},
        "control_condition": {"type": "array", "items": CLAUSE,
                              "description": "empty = compare against the rest of the subset"},
        "direction": {"type": "string", "enum": ["greater", "less", "two-sided"]},
        "min_effect": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["statement", "subset", "condition", "control_condition", "direction", "min_effect", "rationale"],
    "additionalProperties": False,
}
SCHEMA = {"type": "object", "properties": {"hypotheses": {"type": "array", "items": HYP}},
          "required": ["hypotheses"], "additionalProperties": False}

SYSTEM = """You are a quantitative researcher proposing testable hypotheses about Lichess rated blitz games.
The outcome is always `residual` = actual score (1/0.5/0) minus the Elo-expected score, per player per game.
A positive residual means the player did better than their rating predicts.

Each hypothesis defines a treated group (rows matching `subset` AND `condition`) and a control group
(the rest of the subset, or rows matching `control_condition` if given). The tested outcome is the residual
minus the player's own typical residual, so the test compares a player's games with their OTHER games:
conditions that are fixed for a player (e.g. their rating band alone) cannot be detected - use rating bands
in `subset`, and put things that vary game to game (previous result, streaks, time gaps, colour, time of
day, session position, opening, time control...) in `condition`.

Rules:
- Use only the features, operators and allowed values in the schema. Categorical features take `==`, `!=`,
  `in`, `not_in`; numeric features take `==`, `!=`, `<`, `<=`, `>`, `>=`, `between` ([lo, hi]).
- `streak` thresholds must be between -5 and 5.
- Fix `direction` in advance from your reasoning. Use `two-sided` only if you genuinely have no prior.
- `min_effect` is the smallest residual difference that would matter in practice (typically 0.005-0.02).
- Every feature must be known BEFORE the game starts.
- Do not repeat hypotheses already in the ledger (listed below). Aim for diversity across: tilt, streaks /
  hot hand, time-of-day, session fatigue, colour by rating band, opening family, time control, rest gaps.
- Treated groups need at least ~500 games from 50+ players; avoid ultra-narrow conditions."""


def explore_summary(ex: pd.DataFrame) -> dict:
    """Aggregate stats from the explore split ONLY."""
    out = {"split": "explore", "n_rows": int(len(ex)), "n_players": int(ex["player"].nunique()),
           "residual_mean": round(float(ex["residual"].mean()), 5),
           "residual_sd": round(float(ex["residual"].std()), 4), "marginal_residual_means": {}}
    for f, spec in FEATURES.items():
        if spec.get("leaky"):
            continue
        col = ex[f]
        if spec["type"] == "numeric":
            q = np.unique(np.nanquantile(col.astype(float), [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            if len(q) < 3:
                continue
            col = pd.cut(col.astype(float), bins=q, include_lowest=True, duplicates="drop").astype(str)
        g = ex.groupby(col.astype(str), observed=True)["residual"].agg(["mean", "size"])
        g = g[g["size"] >= 1000].sort_values("size", ascending=False).head(12)
        out["marginal_residual_means"][f] = {k: {"mean": round(float(r["mean"]), 5), "n": int(r["size"])}
                                             for k, r in g.iterrows()}
    return out


def schema_for_prompt() -> dict:
    return {f: {k: v for k, v in s.items() if k != "leaky"} for f, s in FEATURES.items() if not s.get("leaky")}


def build_prompt(summary: dict, existing: list[str], n: int, feedback: str | None = None) -> str:
    parts = [f"Propose exactly {n} new hypotheses.",
             "## Feature schema\n" + json.dumps(schema_for_prompt(), indent=1),
             "## Explore-split summary statistics (the only data you see)\n" + json.dumps(summary, indent=1),
             "## Hypotheses already in the ledger (do not duplicate)\n" + ("\n".join(f"- {s}" for s in existing) or "(none)")]
    if feedback:
        parts.append("## Feedback from previous round\n" + feedback)
    return "\n\n".join(parts)


def to_template(h: dict) -> dict:
    """Map the LLM's flat output onto the project hypothesis template."""
    comp = {"condition": h["control_condition"]} if h.get("control_condition") else "rest_of_subset"
    return {"statement": h["statement"], "subset": h.get("subset", []), "condition": h["condition"],
            "comparison": comp, "outcome": "residual", "direction": h["direction"],
            "min_effect": h["min_effect"], "rationale": h.get("rationale", "")}


def _client():
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    return anthropic.Anthropic()


def call_llm(prompt: str, client=None, system: str = SYSTEM) -> tuple[str, dict]:
    """Returns (raw text, metadata). Structured output guarantees schema-shaped JSON."""
    client = client or _client()
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
    text = next(b.text for b in resp.content if b.type == "text")
    meta = {"model": resp.model, "stop_reason": resp.stop_reason, "id": resp.id,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}}
    return text, meta


def propose(summary: dict, existing: list[str], n: int = 20, client=None, batch: str | None = None,
            feedback: str | None = None, system: str = SYSTEM) -> tuple[list[dict], str]:
    """Return (hypotheses in template form, batch id). Stores prompt + raw response for every attempt."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch = batch or time.strftime("B%Y%m%d-%H%M%S")
    prompt = build_prompt(summary, existing, n, feedback)
    attempts = []
    for attempt in range(1, MAX_RETRIES + 1):
        rec = {"batch": batch, "attempt": attempt, "model": MODEL, "system": system, "prompt": prompt}
        try:
            text, meta = call_llm(prompt, client, system)
            rec.update(raw_response=text, meta=meta)
            hyps = json.loads(text)["hypotheses"]
            if not isinstance(hyps, list) or not hyps:
                raise ValueError("empty hypothesis list")
            rec["ok"] = True
            attempts.append(rec)
            break
        except Exception as e:  # malformed output or API failure: log and retry
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            attempts.append(rec)
            hyps = None
    (OUT_DIR / f"{batch}.json").write_text(json.dumps(attempts, indent=2))
    if hyps is None:
        raise RuntimeError(f"proposer failed after {MAX_RETRIES} attempts; see {OUT_DIR / (batch + '.json')}")
    return [to_template(h) for h in hyps], batch


def get_summary(df: pd.DataFrame) -> dict:
    if SUMMARY_CACHE.exists():
        return json.loads(SUMMARY_CACHE.read_text())
    s = explore_summary(df[df["split"] == "explore"])
    SUMMARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CACHE.write_text(json.dumps(s, indent=1))
    return s


def main() -> None:
    from src.judge import Judge, load_data, print_report
    from src.ledger import Ledger
    from src.bank import HAND

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ledger", default="results/ledger_main.jsonl")
    ap.add_argument("--fresh", action="store_true", help="start a new ledger (deletes the old one)")
    ap.add_argument("--offline", action="store_true", help="template proposer instead of the LLM (no API key)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.fresh:
        Path(a.ledger).unlink(missing_ok=True)
        Path("results/vault_access_main.jsonl").unlink(missing_ok=True)
    df = load_data()
    summary = get_summary(df)
    led = Ledger(a.ledger, run_id="main")
    judge = Judge(df, led, vault_log="results/vault_access_main.jsonl")
    # Re-hydrate hypotheses already in this ledger so BH counts them and duplicates are caught.
    from src.hypothesis import InvalidHypothesis, validate
    for hid, s in led.state().items():
        try:
            judge.hyps[hid] = validate({**s["raw"], "id": hid}) if s["verdict"] not in ("INVALID", "DUPLICATE") else None
        except InvalidHypothesis:
            pass
    judge.hyps = {k: v for k, v in judge.hyps.items() if v is not None}
    if not led.entries:
        judge.submit(HAND, source="hand")
    existing = [s["raw"].get("statement", "") for s in led.state().values()]
    if a.offline:
        from src.offline_proposer import propose_offline, record_batch
        batch = time.strftime("O%Y%m%d-%H%M%S")
        hyps = propose_offline(summary, {h.spec_key() for h in judge.hyps.values()}, a.n, a.seed)
        record_batch(OUT_DIR / f"{batch}.json", summary, existing, hyps, a.seed)
    else:
        hyps, batch = propose(summary, existing, a.n)
    judge.submit(hyps, source="offline" if a.offline else "llm", batch=batch)
    # Cache LLM hypotheses for calibration (no API calls needed later).
    bank = Path("results/cache/offline_bank.json" if a.offline else "results/cache/llm_bank.json")
    prev = json.loads(bank.read_text()) if bank.exists() else []
    bank.write_text(json.dumps(prev + hyps, indent=1))
    print_report(judge.run())


if __name__ == "__main__":
    main()
