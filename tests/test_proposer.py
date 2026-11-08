import json
from types import SimpleNamespace

import src.proposer as P
from src.hypothesis import validate
from tests.test_judge import synthetic_player_games

GOOD = {"hypotheses": [{"statement": "tilt", "subset": [],
                        "condition": [{"feature": "prev_result", "op": "==", "value": "loss"}],
                        "control_condition": [], "direction": "less", "min_effect": 0.01, "rationale": "r"},
                       {"statement": "3+0 vs 5+0", "subset": [],
                        "condition": [{"feature": "time_control", "op": "==", "value": "180+0"}],
                        "control_condition": [{"feature": "time_control", "op": "==", "value": "300+0"}],
                        "direction": "less", "min_effect": 0.01, "rationale": "r"}]}


class FakeClient:
    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        t = self.texts.pop(0)
        return SimpleNamespace(stop_reason="end_turn", model=P.MODEL, id="msg_x",
                               content=[SimpleNamespace(type="text", text=t)],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def test_propose_retries_and_maps_template(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUT_DIR", tmp_path)
    client = FakeClient(["not json", json.dumps(GOOD)])
    hyps, batch = P.propose({"n_rows": 1}, ["old one"], n=2, client=client, batch="T1")
    assert len(hyps) == 2
    assert hyps[1]["comparison"] == {"condition": GOOD["hypotheses"][1]["control_condition"]}
    for i, h in enumerate(hyps):
        validate({**h, "id": f"X{i}"})
    log = json.loads((tmp_path / "T1.json").read_text())
    assert [a["ok"] for a in log] == [False, True]
    assert "old one" in log[1]["prompt"] and log[1]["raw_response"] == json.dumps(GOOD)


def test_summary_uses_explore_only():
    df = synthetic_player_games(n_players=100, n_games=8000)
    s = P.explore_summary(df[df.split == "explore"])
    assert s["n_rows"] == int((df.split == "explore").sum())
    assert "termination" not in s["marginal_residual_means"]  # leaky feature hidden


def test_offline_proposer_valid_diverse_and_deduped():
    from src.offline_proposer import propose_offline
    s = {"residual_mean": 0.0, "marginal_residual_means": {"color": {"white": {"mean": 0.02, "n": 10}}}}
    hyps = propose_offline(s, set(), n=20, seed=1)
    assert len(hyps) == 20
    keys = {validate({**h, "id": f"X{i}"}).spec_key() for i, h in enumerate(hyps)}
    assert len(keys) == 20
    assert len({h["rationale"] for h in hyps}) >= 10          # many themes
    again = propose_offline(s, keys, n=5, seed=1)
    assert not ({validate({**h, "id": "y"}).spec_key() for h in again} & keys)


def test_ask_translate_maps_question_to_valid_template():
    from src.ask import translate
    client = FakeClient([json.dumps({"hypotheses": GOOD["hypotheses"][:1]})])
    spec = translate("Do players tilt after a loss?", client=client)
    assert "Do players tilt after a loss?" in client.prompts[0]
    assert spec["rationale"].startswith("asked by a user")
    validate({**spec, "id": "Q1"})
