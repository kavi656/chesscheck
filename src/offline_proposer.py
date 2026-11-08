"""Offline hypothesis proposer: a stand-in for the LLM when no ANTHROPIC_API_KEY is available.

It sees exactly what the LLM would see - the feature schema, EXPLORE-only summary stats and the
ledger's existing hypotheses - and nothing else. It draws diverse hypotheses from templates covering
the same themes the LLM is asked for (tilt, streaks, time of day, session fatigue, colour by rating
band, opening family, time control, rest gaps). The direction is pre-registered from the sign of
the explore marginal mean for the condition's level (or two-sided when unknown), just as an LLM
reading the summary would. Everything downstream (ledger, BH, validate, skeptic, vault) is identical.
"""
from __future__ import annotations

import json
import random

from src.features import BAND_LABELS
from src.hypothesis import InvalidHypothesis, validate

THEMES = {
    "tilt": lambda r: [{"feature": "prev_result", "op": "==", "value": "loss"},
                       {"feature": "minutes_since_prev_game", "op": "<=", "value": r.choice([2, 3, 5, 10])}],
    "after_win_fast": lambda r: [{"feature": "prev_result", "op": "==", "value": "win"},
                                 {"feature": "minutes_since_prev_game", "op": "<=", "value": r.choice([2, 5])}],
    "after_draw": lambda r: [{"feature": "prev_result", "op": "==", "value": "draw"}],
    "win_streak": lambda r: [{"feature": "streak", "op": ">=", "value": r.choice([2, 3, 4, 5])}],
    "loss_streak": lambda r: [{"feature": "streak", "op": "<=", "value": -r.choice([2, 3, 4, 5])}],
    "session_fatigue": lambda r: [{"feature": "session_index", "op": ">=", "value": r.choice([5, 8, 15, 25])}],
    "daily_volume": lambda r: [{"feature": "games_played_today", "op": ">=", "value": r.choice([10, 20, 40])}],
    "rested": lambda r: [{"feature": "minutes_since_prev_game", "op": ">=", "value": r.choice([60, 240, 720])}],
    "time_of_day": lambda r: [{"feature": "hour_utc", "op": "between", "value": r.choice([[0, 5], [6, 11], [12, 17], [18, 23], [22, 23]])}],
    "colour": lambda r: [{"feature": "color", "op": "==", "value": r.choice(["white", "black"])}],
    "opening": lambda r: [{"feature": "eco_family", "op": "==", "value": r.choice(list("ABCDE"))}],
    "time_control": lambda r: [{"feature": "time_control", "op": "==", "value": r.choice(["180+0", "180+2", "300+0", "300+3"])}],
    "arena": lambda r: [{"feature": "event_type", "op": "==", "value": "tournament"}],
    "flagged_last": lambda r: [{"feature": "prev_termination", "op": "==", "value": "Time forfeit"}],
    "stronger_opp_band": lambda r: [{"feature": "elo_diff", "op": "<=", "value": -r.choice([50, 100, 150])}],
    "weekend": lambda r: [{"feature": "weekday", "op": "in", "value": ["Sat", "Sun"]}],
}


def _summary_sign(summary: dict, clause: dict) -> int:
    """+1/-1 if the explore marginal mean for this categorical level is above/below the overall mean."""
    m = summary.get("marginal_residual_means", {}).get(clause["feature"], {})
    v = clause["value"]
    if isinstance(v, str) and v in m:
        d = m[v]["mean"] - summary.get("residual_mean", 0.0)
        return 1 if d > 0 else -1
    return 0


def _statement(cond: list, subset: list, direction: str) -> str:
    def c(x):
        v = x["value"]
        return f"{x['feature']} {x['op']} {v if not isinstance(v, list) else '[' + ', '.join(map(str, v)) + ']'}"
    who = f"Among players in {subset[0]['value']}, " if subset else ""
    verb = {"greater": "outperform", "less": "underperform", "two-sided": "differ from"}[direction]
    return f"{who}games with {' and '.join(c(x) for x in cond)} {verb} the player's other games (rating-adjusted)"


def propose_offline(summary: dict, existing_keys: set[str], n: int = 20, seed: int = 0) -> list[dict]:
    r = random.Random(seed)
    out, keys, tries = [], set(existing_keys), 0
    themes = list(THEMES)
    while len(out) < n and tries < 2000:
        tries += 1
        theme = themes[(len(out) + tries) % len(themes)] if tries < len(themes) else r.choice(themes)
        cond = THEMES[theme](r)
        subset = [{"feature": "rating_band", "op": "==", "value": r.choice(BAND_LABELS)}] if r.random() < 0.35 else []
        s = _summary_sign(summary, cond[0]) if len(cond) == 1 else 0
        direction = {1: "greater", -1: "less"}.get(s, "two-sided")
        if theme == "tilt" or theme == "loss_streak":
            direction = "less"           # prior: tilt hurts
        elif theme in ("win_streak", "after_win_fast"):
            direction = "greater"        # prior: confidence / hot hand
        h = {"statement": _statement(cond, subset, direction), "subset": subset, "condition": cond,
             "comparison": "rest_of_subset", "outcome": "residual", "direction": direction,
             "min_effect": 0.005, "rationale": f"offline template proposer: theme '{theme}'"}
        try:
            k = validate({**h, "id": "tmp"}).spec_key()
        except InvalidHypothesis:
            continue
        if k in keys:
            continue
        keys.add(k)
        out.append(h)
    return out


def record_batch(path, summary, existing, hyps, seed) -> None:
    """Same transparency record the LLM proposer writes: inputs and raw output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"proposer": "offline-template", "seed": seed, "inputs": {
        "explore_summary": summary, "existing_statements": existing}, "raw_response": hyps, "ok": True}], indent=1))
