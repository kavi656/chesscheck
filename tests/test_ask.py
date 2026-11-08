import json

import src.ask as ask

H = {"statement": "Losing then re-queueing within 4 minutes hurts", "subset": [],
     "condition": [{"feature": "prev_result", "op": "==", "value": "loss"},
                   {"feature": "minutes_since_prev_game", "op": "<=", "value": 4}],
     "control_condition": [], "direction": "less", "min_effect": 0.01, "rationale": "tilt"}


def fake(monkeypatch):
    monkeypatch.setattr(ask, "call_llm", lambda prompt, client=None: (json.dumps({"hypotheses": [H]}), None))


def test_default_min_effect_when_user_gives_no_size(monkeypatch):
    fake(monkeypatch)
    assert ask.translate("Players who lost within 4 minutes do worse")["min_effect"] == ask.DEFAULT_MIN_EFFECT


def test_user_stated_size_is_kept(monkeypatch):
    fake(monkeypatch)
    assert ask.translate("Players who lost within 4 minutes do worse by 7 Elo")["min_effect"] == 0.01
