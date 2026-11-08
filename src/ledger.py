"""Append-only JSONL ledger. Every hypothesis is logged *before* it is tested.

Lines are never rewritten; the current state of a hypothesis is the fold of all its lines.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Ledger:
    def __init__(self, path: str | Path | None = None, run_id: str = "run"):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.entries: list[dict] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.entries = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def _append(self, rec: dict) -> dict:
        rec = {"ts": now(), "run": self.run_id, **rec}
        with self._lock:
            self.entries.append(rec)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def next_id(self) -> str:
        n = sum(1 for e in self.entries if e["type"] == "proposed")
        return f"H{n + 1:04d}"

    def propose(self, raw: dict, source: str, batch: str | None = None) -> str:
        """Log a proposal before anything else happens. Assigns the ledger id."""
        hid = self.next_id()
        self._append({"type": "proposed", "id": hid, "source": source, "batch": batch, "raw": raw})
        return hid

    def record(self, hid: str, stage: str, **payload) -> None:
        self._append({"type": "stage", "id": hid, "stage": stage, **payload})

    def verdict(self, hid: str, verdict: str, killed_at: str | None, reason: str | None) -> None:
        self._append({"type": "verdict", "id": hid, "verdict": verdict, "killed_at": killed_at, "reason": reason})

    def state(self) -> dict[str, dict]:
        """Fold lines into {id: {"raw":..., "stages": {stage: payload}, "verdict":...}} in proposal order."""
        out: dict[str, dict] = {}
        for e in self.entries:
            if e["type"] == "proposed":
                out[e["id"]] = {"id": e["id"], "proposed_at": e["ts"], "source": e["source"], "batch": e.get("batch"),
                                "raw": e["raw"], "stages": {}, "verdict": None}
            elif e["type"] == "stage":
                out[e["id"]]["stages"][e["stage"]] = {k: v for k, v in e.items() if k not in ("type", "id", "stage", "run")}
            elif e["type"] == "verdict":
                out[e["id"]].update(verdict=e["verdict"], killed_at=e["killed_at"], reason=e["reason"])
        return out
