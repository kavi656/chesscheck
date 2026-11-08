"""Ask-a-question: plain English -> template (LLM) -> full honest pipeline, logged in the main ledger."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.proposer import SYSTEM, call_llm, get_summary, schema_for_prompt, to_template

LEDGER = "results/ledger_main.jsonl"
DEFAULT_MIN_EFFECT = 0.005  # same bar as the hand-written bank (~3.5 Elo)
SIZE_RE = re.compile(r"\d+(\.\d+)?\s*(elo|rating points?|score points?|points?|%|percent)", re.I)


def translate(question: str, client=None) -> dict:
    prompt = ("Convert this ONE plain-English hypothesis from a user into exactly one hypothesis in the schema. "
              "Keep the user's meaning; choose the direction they imply; do not make it more likely to be significant. "
              f"Use min_effect = {DEFAULT_MIN_EFFECT} (the project standard) unless the user states a size.\n\n"
              f"User hypothesis: {question}\n\n## Feature schema\n{json.dumps(schema_for_prompt(), indent=1)}")
    text, _ = call_llm(prompt, client)
    h = json.loads(text)["hypotheses"][0]
    t = to_template(h)
    t["rationale"] = f"asked by a user: {question}"
    if not SIZE_RE.search(question):
        # The size bar is a policy choice, not something the LLM should vary between near-identical questions:
        # without it, "within 5 min" got 0.005 and "within 4 min" got 0.01, so the same effect passed and failed.
        t["min_effect"] = DEFAULT_MIN_EFFECT
    return t


SANDBOX_LEDGER = Path("results/sandbox/ledger_ask.jsonl")


def run_question(spec: dict, sandbox: bool = False) -> dict:
    """Run one plain-English question through the full pipeline. sandbox=True works on a copy of the main
    ledger (created on first use), so demo questions still count toward the correction but cannot change
    the main ledger or the committed live set."""
    import shutil

    from src.hypothesis import InvalidHypothesis, validate
    from src.judge import Judge, load_data
    from src.ledger import Ledger

    path, vlog = Path(LEDGER), "results/vault_access_main.jsonl"
    if sandbox:
        SANDBOX_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        if not SANDBOX_LEDGER.exists():
            shutil.copy(LEDGER, SANDBOX_LEDGER)
        path, vlog = SANDBOX_LEDGER, str(SANDBOX_LEDGER.parent / "vault_access_ask.jsonl")
    led = Ledger(path, run_id="sandbox" if sandbox else "main")
    judge = Judge(load_data(), led, vault_log=vlog)
    for hid, s in led.state().items():  # existing hypotheses count toward the correction
        if s["verdict"] not in ("INVALID", "DUPLICATE"):
            try:
                judge.hyps[hid] = validate({**s["raw"], "id": hid})
            except InvalidHypothesis:
                pass
    (hid,) = judge.submit([spec], source="ask")
    return judge.run()[hid]
