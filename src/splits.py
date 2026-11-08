"""Time-based explore / validate / vault splits.

Never random: games by the same player are correlated in time, so a random split
would leak a player's form across splits. The first UTC day is burn-in only (its games
feed sequence features of later games but are not analysed), because players' history
before the data window is unknown.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FRACTIONS = {"explore": 0.5, "validate": 0.3, "vault": 0.2}
BOUNDARY_FILE = Path("data/processed/split_boundaries.json")


def compute_boundaries(start: pd.Series) -> dict:
    """Cut points are row-count quantiles of game start time, floored to the hour."""
    first_day = start.min().floor("D")
    burn_end = first_day + pd.Timedelta(days=1)
    s = start[start >= burn_end]
    q1 = s.quantile(FRACTIONS["explore"]).floor("h")
    q2 = s.quantile(FRACTIONS["explore"] + FRACTIONS["validate"]).floor("h")
    return {"burn_in_start": str(first_day), "explore_start": str(burn_end),
            "validate_start": str(q1), "vault_start": str(q2), "data_end": str(start.max())}


def assign_splits(df: pd.DataFrame, boundaries: dict | None = None, save: bool = True) -> pd.DataFrame:
    b = boundaries or compute_boundaries(df["start"])
    ts = {k: pd.Timestamp(v) for k, v in b.items()}
    split = pd.Series("burn_in", index=df.index)
    split[df["start"] >= ts["explore_start"]] = "explore"
    split[df["start"] >= ts["validate_start"]] = "validate"
    split[df["start"] >= ts["vault_start"]] = "vault"
    df["split"] = pd.Categorical(split, categories=["burn_in", "explore", "validate", "vault"])
    if save:
        BOUNDARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        BOUNDARY_FILE.write_text(json.dumps(b, indent=2))
    return df
