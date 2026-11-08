"""Stream a Lichess monthly dump (.pgn.zst) and keep rated-blitz game headers.

Headers only: we never parse moves. Header blocks are pulled out with a byte regex,
which is ~10x faster than a chess parser. Output: parquet part files in data/raw/.

Usage: python -m src.ingest --month 2026-10 --max-day 8 [--max-games N]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
import zstandard

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
KEEP = ["Event", "Site", "White", "Black", "Result", "UTCDate", "UTCTime", "WhiteElo",
        "BlackElo", "TimeControl", "ECO", "Opening", "Termination"]
HEADER_BLOCK = re.compile(rb'(?:^\[[A-Za-z]+ "[^"\n]*"\]\n)+', re.M)
TAG = re.compile(rb'\[([A-Za-z]+) "([^"\n]*)"\]')
OUT = Path("data/raw")


def parse_blocks(buf: bytes) -> list[dict]:
    rows = []
    for m in HEADER_BLOCK.finditer(buf):
        block = m.group(0)
        if not block.startswith(b'[Event "Rated Blitz'):
            continue
        tags = dict(TAG.findall(block))
        rows.append({k: tags.get(k.encode(), b"").decode("utf-8", "replace") for k in KEEP})
    return rows


def ingest(month: str, max_day: int, max_games: int | None, chunk: int = 1 << 24) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob(f"blitz_{month}_part*.parquet"):
        f.unlink()
    r = requests.get(URL.format(month=month), stream=True, timeout=60,
                     headers={"User-Agent": "honest-ai-researcher-hackathon"})
    r.raise_for_status()
    reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(r.raw)
    tail, rows, part, total, t0 = b"", [], 0, 0, time.time()
    stop_date = f"{month.replace('-', '.')}.{max_day + 1:02d}"
    done = False
    while not done:
        data = reader.read(chunk)
        if not data:
            buf, tail = tail, b""
            done = True
        else:
            buf = tail + data
            cut = buf.rfind(b'\n\n[Event "')
            if cut < 0:
                tail = buf
                continue
            buf, tail = buf[: cut + 2], buf[cut + 2:]
        new = parse_blocks(buf)
        if new and new[-1]["UTCDate"] >= stop_date:
            new = [g for g in new if g["UTCDate"] < stop_date]
            done = True
        rows.extend(new)
        if max_games is not None and total + len(rows) >= max_games:
            done = True
        if len(rows) >= 1_000_000 or done:
            df = pd.DataFrame(rows)
            if max_games is not None:
                df = df.iloc[: max(0, max_games - total)]
            df.to_parquet(OUT / f"blitz_{month}_part{part:03d}.parquet", index=False)
            total += len(df); part += 1; rows = []
            last = df["UTCDate"].iloc[-1] if len(df) else "?"
            print(f"[{time.time()-t0:7.0f}s] part {part} total blitz={total:,} last date={last}", flush=True)
            if max_games is not None and total >= max_games:
                done = True
    r.close()
    print(f"done: {total:,} rated blitz games in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-10")
    ap.add_argument("--max-day", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=None)
    a = ap.parse_args()
    ingest(a.month, a.max_day, a.max_games)
