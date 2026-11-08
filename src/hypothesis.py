"""Hypothesis template: schema, validation, and evaluation of conditions to row masks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import BAND_LABELS, PREV_RESULTS, TERMINATIONS, WEEKDAYS

# Feature schema. `leaky=True` means the value is only known after the game starts.
FEATURES: dict[str, dict] = {
    "color": {"type": "categorical", "values": ["white", "black"], "desc": "player's colour"},
    "player_elo": {"type": "numeric", "range": [400, 3400], "desc": "player's rating before the game"},
    "opp_elo": {"type": "numeric", "range": [400, 3400], "desc": "opponent's rating"},
    "elo_diff": {"type": "numeric", "range": [-2000, 2000], "desc": "player_elo - opp_elo"},
    "rating_band": {"type": "categorical", "values": BAND_LABELS, "desc": "band of player_elo"},
    "opp_rating_band": {"type": "categorical", "values": BAND_LABELS, "desc": "band of opp_elo"},
    "prev_result": {"type": "categorical", "values": PREV_RESULTS, "desc": "player's previous game result ('none' = no earlier game in data)"},
    "prev_termination": {"type": "categorical", "values": TERMINATIONS, "desc": "how the player's previous game ended"},
    "minutes_since_prev_game": {"type": "numeric", "range": [0, 20000], "desc": "minutes between start of previous game and this one (missing if none)"},
    "streak": {"type": "numeric", "range": [-5, 5], "desc": "+k = k straight wins before this game, -k = k straight losses; draws reset to 0. Thresholds limited to |k|<=5 (the test's baseline excludes 6 games either side)"},
    "games_played_today": {"type": "numeric", "range": [0, 500], "desc": "games the player started earlier the same UTC day"},
    "session_index": {"type": "numeric", "range": [1, 500], "desc": "position of this game in the player's session (sessions split by >30 min gaps), 1 = first"},
    "hour_utc": {"type": "numeric", "range": [0, 23], "desc": "UTC hour the game started"},
    "local_hour": {"type": "numeric", "range": [0, 23], "desc": "ESTIMATED local hour for this player: their quietest 6-hour UTC window in the explore period is taken as 01:00-06:59 local. Missing for players with <30 explore games, so compare with an explicit comparison condition on local_hour (e.g. treated 0-4 vs comparison 10-21)"},
    "weekday": {"type": "categorical", "values": WEEKDAYS, "desc": "UTC weekday"},
    "start_minute_parity": {"type": "categorical", "values": ["odd", "even"], "desc": "whether the start minute is odd"},
    "time_control": {"type": "categorical", "values": ["180+0", "180+2", "300+0", "300+3", "240+0", "240+2", "300+2", "420+0"], "desc": "base seconds + increment"},
    "eco_family": {"type": "categorical", "values": ["A", "B", "C", "D", "E", "?"], "desc": "first letter of the ECO opening code"},
    "event_type": {"type": "categorical", "values": ["pool", "tournament"], "desc": "lobby/pool game or arena tournament game"},
    "termination": {"type": "categorical", "values": TERMINATIONS[:-1], "desc": "how THIS game ended", "leaky": True},
}
NUMERIC_OPS = {"==", "!=", "<", "<=", ">", ">=", "between"}
CAT_OPS = {"==", "!=", "in", "not_in"}
DIRECTIONS = {"greater", "less", "two-sided"}
OUTCOMES = {"residual"}


class InvalidHypothesis(ValueError):
    pass


@dataclass
class Hypothesis:
    id: str
    statement: str
    subset: list
    condition: list
    comparison: object  # "rest_of_subset" or {"condition": [...]}
    outcome: str
    direction: str
    min_effect: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def features_used(self) -> set[str]:
        clauses = self.subset + self.condition
        if isinstance(self.comparison, dict):
            clauses = clauses + self.comparison["condition"]
        return {c["feature"] for c in clauses}

    def is_leaky(self) -> bool:
        return any(FEATURES[f].get("leaky") for f in self.features_used())

    def spec_key(self) -> str:
        """Hash of the testable spec (ignores id/statement/rationale) for duplicate detection."""
        canon = lambda cl: sorted(json.dumps(c, sort_keys=True) for c in cl)
        comp = self.comparison if isinstance(self.comparison, str) else canon(self.comparison["condition"])
        key = json.dumps([canon(self.subset), canon(self.condition), comp, self.outcome, self.direction])
        return hashlib.sha1(key.encode()).hexdigest()[:12]


def _validate_clause(c: dict) -> dict:
    if not isinstance(c, dict) or set(c) - {"feature", "op", "value"} or not {"feature", "op", "value"} <= set(c):
        raise InvalidHypothesis(f"clause must have exactly feature/op/value: {c}")
    f, op, v = c["feature"], c["op"], c["value"]
    if f not in FEATURES:
        raise InvalidHypothesis(f"unknown feature '{f}'")
    spec = FEATURES[f]
    if spec["type"] == "numeric":
        if op not in NUMERIC_OPS:
            raise InvalidHypothesis(f"op '{op}' not allowed for numeric feature '{f}'")
        if op == "between":
            if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and v[0] <= v[1]):
                raise InvalidHypothesis(f"'between' needs [lo, hi] for '{f}'")
        elif not isinstance(v, (int, float)) or isinstance(v, bool):
            raise InvalidHypothesis(f"numeric value required for '{f}', got {v!r}")
        lo, hi = spec["range"]
        if any(not (lo <= x <= hi) for x in (v if isinstance(v, list) else [v])):
            raise InvalidHypothesis(f"value {v!r} outside allowed range {spec['range']} for '{f}'")
    else:
        if op not in CAT_OPS:
            raise InvalidHypothesis(f"op '{op}' not allowed for categorical feature '{f}'")
        vals = v if op in ("in", "not_in") else [v]
        if op in ("in", "not_in") and not isinstance(v, list):
            raise InvalidHypothesis(f"'{op}' needs a list for '{f}'")
        bad = [x for x in vals if x not in spec["values"]]
        if bad:
            raise InvalidHypothesis(f"values {bad} not allowed for '{f}' (allowed: {spec['values']})")
    return {"feature": f, "op": op, "value": v}


def validate(d: dict) -> Hypothesis:
    """Return a Hypothesis or raise InvalidHypothesis with a human-readable reason."""
    if not isinstance(d, dict):
        raise InvalidHypothesis("hypothesis must be a JSON object")
    for k in ("id", "statement", "condition", "direction", "min_effect"):
        if k not in d:
            raise InvalidHypothesis(f"missing field '{k}'")
    subset = [_validate_clause(c) for c in (d.get("subset") or [])]
    condition = [_validate_clause(c) for c in d["condition"]]
    if not condition:
        raise InvalidHypothesis("condition must not be empty")
    comp = d.get("comparison", "rest_of_subset")
    if isinstance(comp, dict):
        if "condition" not in comp or not comp["condition"]:
            raise InvalidHypothesis("explicit comparison needs a non-empty 'condition'")
        comp = {"condition": [_validate_clause(c) for c in comp["condition"]]}
    elif comp != "rest_of_subset":
        raise InvalidHypothesis(f"comparison must be 'rest_of_subset' or {{'condition': [...]}}, got {comp!r}")
    outcome = d.get("outcome", "residual")
    if outcome == "score":  # score with rating diff controlled == residual
        outcome = "residual"
    if outcome not in OUTCOMES:
        raise InvalidHypothesis(f"outcome must be 'residual' (got {outcome!r})")
    if d["direction"] not in DIRECTIONS:
        raise InvalidHypothesis(f"direction must be one of {sorted(DIRECTIONS)}")
    me = d["min_effect"]
    if not isinstance(me, (int, float)) or isinstance(me, bool) or not (0 < me < 0.5):
        raise InvalidHypothesis("min_effect must be a number in (0, 0.5)")
    return Hypothesis(id=str(d["id"]), statement=str(d["statement"]), subset=subset, condition=condition,
                      comparison=comp, outcome=outcome, direction=d["direction"], min_effect=float(me),
                      rationale=str(d.get("rationale", "")))


def clause_mask(df: pd.DataFrame, c: dict) -> np.ndarray:
    col, op, v = df[c["feature"]], c["op"], c["value"]
    if FEATURES[c["feature"]]["type"] == "numeric":
        x = col.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            m = {"==": lambda: x == v, "!=": lambda: x != v, "<": lambda: x < v, "<=": lambda: x <= v,
                 ">": lambda: x > v, ">=": lambda: x >= v,
                 "between": lambda: (x >= v[0]) & (x <= v[1])}[op]()
        return m & ~np.isnan(x)  # missing never satisfies a numeric clause
    x = col.astype(str).to_numpy() if not isinstance(col.dtype, pd.CategoricalDtype) else col
    if op == "==":
        return np.asarray(x == v)
    if op == "!=":
        return np.asarray(x != v)
    isin = np.asarray(pd.Series(x).isin(v))
    return isin if op == "in" else ~isin


def all_mask(df: pd.DataFrame, clauses: list) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for c in clauses:
        m &= clause_mask(df, c)
    return m


def groups(df: pd.DataFrame, h: Hypothesis) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (treated, control) masks over df rows."""
    sub = all_mask(df, h.subset)
    treated = sub & all_mask(df, h.condition)
    if h.comparison == "rest_of_subset":
        control = sub & ~treated
    else:
        control = sub & all_mask(df, h.comparison["condition"]) & ~treated
    return treated, control
