"""Build the player-game table: one row per player per game, from that player's view.

All sequence features use only games that *started before* the current game, so no
feature leaks information from after the game starts (except `termination`, which is
deliberately kept and flagged leaky in the schema so the skeptic check can catch it).

Usage: python -m src.features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
SESSION_GAP_MIN = 30.0
N_CHUNKS = 4
BANDS = [(-np.inf, 1200, "<1200"), (1200, 1600, "1200-1600"), (1600, 2000, "1600-2000"),
         (2000, 2400, "2000-2400"), (2400, np.inf, "2400+")]
BAND_LABELS = [b[2] for b in BANDS]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PREV_RESULTS = ["win", "draw", "loss", "none"]
TERMINATIONS = ["Normal", "Time forfeit", "Abandoned", "Rules infraction", "Insufficient material", "none"]


def rating_band(elo: pd.Series) -> pd.Categorical:
    edges = [b[0] for b in BANDS] + [np.inf]
    return pd.cut(elo, bins=edges, labels=BAND_LABELS, right=False)


RAW_COLS = ["Event", "Site", "White", "Black", "Result", "UTCDate", "UTCTime", "WhiteElo", "BlackElo",
            "TimeControl", "ECO", "Termination"]
CAT_COLS = ["Event", "White", "Black", "TimeControl", "ECO", "Termination"]


def load_games(raw_dir: Path = RAW) -> pd.DataFrame:
    """Clean each part separately (bounded memory), then concatenate categoricals with unioned categories."""
    from pandas.api.types import union_categoricals
    parts = []
    for p in sorted(raw_dir.glob("blitz_*_part*.parquet")):
        g = clean_games(pd.read_parquet(p, columns=RAW_COLS))
        # dedupe happened inside clean_games; the dump is one sequential stream so parts can't overlap
        g = g.drop(columns=["Site", "UTCDate", "UTCTime", "game_id"])
        for c in CAT_COLS:
            g[c] = g[c].astype("category")
        parts.append(g)
    out = {}
    for c in parts[0].columns:
        out[c] = (union_categoricals([q[c] for q in parts]) if c in CAT_COLS
                  else np.concatenate([q[c].to_numpy() for q in parts]))
    g = pd.DataFrame(out)
    g["start"] = pd.to_datetime(g["start"], utc=True)
    return g


def clean_games(g: pd.DataFrame) -> pd.DataFrame:
    g = g[g["Result"].isin(["1-0", "0-1", "1/2-1/2"])].copy()
    for c in ("WhiteElo", "BlackElo"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["WhiteElo", "BlackElo"])
    g = g[(g["White"] != "") & (g["Black"] != "")]
    g["start"] = pd.to_datetime(g["UTCDate"] + " " + g["UTCTime"], format="%Y.%m.%d %H:%M:%S", utc=True)
    g["game_id"] = g["Site"].str.rsplit("/", n=1).str[-1]
    g["WhiteElo"] = g["WhiteElo"].astype(np.int16)
    g["BlackElo"] = g["BlackElo"].astype(np.int16)
    g = g.drop_duplicates("game_id")
    return g.reset_index(drop=True)


def player_rows(g: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (white view + black view), then sequence features per player."""
    white_score = g["Result"].map({"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}).to_numpy(dtype=np.float64)
    # Categoricals with shared categories keep memory ~4x lower than object strings at 20M+ rows.
    names = pd.unique(np.r_[np.asarray(g["White"], dtype=object), np.asarray(g["Black"], dtype=object)])
    cat = lambda col: pd.Categorical(g[col].to_numpy(), categories=names)
    common = {
        "game_idx": np.arange(len(g), dtype=np.int32), "start": g["start"].to_numpy(),
        "time_control": pd.Categorical(g["TimeControl"]), "eco": pd.Categorical(g["ECO"]),
        "termination": pd.Categorical(g["Termination"]),
        "event_type": pd.Categorical(np.where(g["Event"].astype(str).str.contains("tournament"), "tournament", "pool")),
    }
    w = pd.DataFrame({**common, "player": cat("White"), "opponent": cat("Black"),
                      "color": "white", "player_elo": g["WhiteElo"].to_numpy(),
                      "opp_elo": g["BlackElo"].to_numpy(), "score": white_score})
    b = pd.DataFrame({**common, "player": cat("Black"), "opponent": cat("White"),
                      "color": "black", "player_elo": g["BlackElo"].to_numpy(),
                      "opp_elo": g["WhiteElo"].to_numpy(), "score": 1.0 - white_score})
    df = pd.concat([w, b], ignore_index=True)
    df["color"] = df["color"].astype("category")
    df["elo_diff"] = df["player_elo"] - df["opp_elo"]
    df["expected_score"] = 1.0 / (1.0 + 10.0 ** (-df["elo_diff"] / 400.0))
    df["residual"] = df["score"] - df["expected_score"]
    return df


def player_games(g: pd.DataFrame) -> pd.DataFrame:
    return add_sequence_features(player_rows(g))


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "start", "game_idx"], kind="mergesort").reset_index(drop=True)
    same = df["player"].eq(df["player"].shift(1)).to_numpy()

    prev_score = df["score"].shift(1).where(same)
    df["prev_result"] = np.select([prev_score == 1.0, prev_score == 0.5, prev_score == 0.0],
                                  ["win", "draw", "loss"], default="none")
    df["prev_termination"] = df["termination"].astype(str).shift(1).where(same).fillna("none")
    mins = (df["start"] - df["start"].shift(1)).dt.total_seconds() / 60.0
    df["minutes_since_prev_game"] = mins.where(same)  # start-to-start; NaN if no prior game

    # Signed streak of identical decisive results ending at the previous game (draw/none -> 0).
    outcome = np.sign(df["score"].to_numpy() - 0.5)  # +1 win, -1 loss, 0 draw
    prev_outcome = np.r_[np.nan, outcome[:-1]]
    run_id = np.cumsum((~same) | (outcome != prev_outcome))
    streak_after = (pd.Series(run_id).groupby(run_id).cumcount().to_numpy() + 1) * outcome
    st = pd.Series(streak_after).shift(1)
    df["streak"] = st.where(same, 0.0).fillna(0.0).astype(int)

    day = df["start"].dt.floor("D")
    df["games_played_today"] = df.groupby(["player", day]).cumcount()  # earlier games same UTC day
    new_session = (~same) | (mins.fillna(np.inf) > SESSION_GAP_MIN).to_numpy()
    session_id = np.cumsum(new_session)
    df["session_index"] = df.groupby(session_id).cumcount() + 1  # 1 = first game of session

    df["hour_utc"] = df["start"].dt.hour
    df["weekday"] = pd.Categorical(df["start"].dt.day_of_week.map(dict(enumerate(WEEKDAYS))), categories=WEEKDAYS)
    df["start_minute_parity"] = np.where(df["start"].dt.minute % 2 == 1, "odd", "even")
    df["eco_family"] = df["eco"].str[:1].where(df["eco"].str[:1].isin(list("ABCDE")), "?")
    df["rating_band"] = rating_band(df["player_elo"])
    df["opp_rating_band"] = rating_band(df["opp_elo"])
    return df


FIXED_CATS = {"prev_result": PREV_RESULTS, "start_minute_parity": ["odd", "even"],
              "eco_family": list("ABCDE") + ["?"], "color": ["white", "black"]}


def compact(df: pd.DataFrame, term_cats: list[str] | None = None) -> pd.DataFrame:
    """Fixed categories everywhere so independently built chunks concatenate as categoricals."""
    for c, cats in FIXED_CATS.items():
        df[c] = pd.Categorical(df[c].astype(str), categories=cats)
    if term_cats is not None:
        df["prev_termination"] = pd.Categorical(df["prev_termination"].astype(str), categories=term_cats)
    for c in ("player", "opponent", "time_control", "eco", "termination", "event_type", "prev_termination"):
        if not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")
    for c in ("player_elo", "opp_elo", "elo_diff"):
        df[c] = df[c].astype("int16")
    for c in ("score", "expected_score", "residual", "minutes_since_prev_game"):
        df[c] = df[c].astype("float32")
    for c in ("streak", "games_played_today", "session_index", "hour_utc"):
        df[c] = df[c].astype("int16")
    return df


BOT_CACHE = OUT / "bot_accounts.json"
BOT_CHECK_TOP = 3000
MAX_GAMES_PER_DAY = 200


def bot_accounts(games: pd.DataFrame) -> set[str]:
    """BOT-titled accounts among the most active players (bots play nonstop and don't tilt).

    Headers were ingested without the title tag, so titles come from POST /api/users (<=300 ids per
    request, one request at a time per the Lichess API rules). Cached to disk."""
    import json
    import time

    import requests
    if BOT_CACHE.exists():
        return set(json.loads(BOT_CACHE.read_text())["bots"])
    counts = pd.concat([games["White"].astype(str), games["Black"].astype(str)]).value_counts()
    top = counts.index[:BOT_CHECK_TOP].tolist()
    bots = set()
    for i in range(0, len(top), 300):
        r = requests.post("https://lichess.org/api/users", data=",".join(top[i:i + 300]), timeout=60,
                          headers={"Content-Type": "text/plain", "User-Agent": "honest-ai-researcher (hackathon)"})
        if r.status_code == 429:
            time.sleep(60)
            r = requests.post("https://lichess.org/api/users", data=",".join(top[i:i + 300]), timeout=60,
                              headers={"Content-Type": "text/plain"})
        r.raise_for_status()
        bots |= {u["username"] for u in r.json() if u.get("title") == "BOT"}
    OUT.mkdir(parents=True, exist_ok=True)
    BOT_CACHE.write_text(json.dumps({"checked_top": BOT_CHECK_TOP, "bots": sorted(bots)}, indent=1))
    return bots


def main() -> None:
    from src.splits import assign_splits

    games = load_games()
    print(f"games after cleaning: {len(games):,}")
    bots = bot_accounts(games)
    day = games["start"].dt.floor("D")
    per_day = pd.concat([pd.DataFrame({"p": games["White"].astype(str), "d": day}),
                         pd.DataFrame({"p": games["Black"].astype(str), "d": day})]).value_counts()
    heavy = set(per_day[per_day > MAX_GAMES_PER_DAY].index.get_level_values("p"))
    print(f"accounts with >{MAX_GAMES_PER_DAY} games in a UTC day (likely unflagged bots): {len(heavy - bots)}")
    bots = bots | heavy
    is_bot = games["White"].astype(str).isin(bots) | games["Black"].astype(str).isin(bots)
    games = games[~is_bot.to_numpy()].reset_index(drop=True)
    for c in ("White", "Black"):
        games[c] = games[c].cat.remove_unused_categories()
    print(f"removed {int(is_bot.sum()):,} games involving {len(bots)} BOT accounts -> {len(games):,} games")
    rows = player_rows(games)
    del games
    term_cats = sorted(set(rows["termination"].cat.categories) | {"none"})
    chunk = rows["player"].cat.codes.to_numpy() % N_CHUNKS
    parts = []
    for k in range(N_CHUNKS):  # sequence features are per player, so player chunks are independent
        parts.append(compact(add_sequence_features(rows[chunk == k]), term_cats))
        print(f"  chunk {k + 1}/{N_CHUNKS}: {len(parts[-1]):,} rows", flush=True)
    del rows
    pg = pd.concat(parts, ignore_index=True)
    del parts
    pg = assign_splits(pg)
    OUT.mkdir(parents=True, exist_ok=True)
    pg.to_parquet(OUT / "player_games.parquet", index=False)
    print(f"player-game rows: {len(pg):,}")


if __name__ == "__main__":
    main()
