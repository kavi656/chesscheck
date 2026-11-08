import numpy as np
import pandas as pd
import pytest

from src.bank import HAND
from src.features import clean_games, player_games
from src.judge import Judge, Vault, VaultAccessError, bh_threshold, prepare
from src.ledger import Ledger
from src.splits import assign_splits


def synthetic_player_games(n_players=300, n_games=60000, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"p{i}" for i in range(n_players)]
    elo = rng.normal(1600, 300, n_players).round()
    t0 = pd.Timestamp("2026-10-01", tz="UTC")
    starts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 8 * 86400, n_games)), unit="s")
    w = rng.integers(0, n_players, n_games)
    b = (w + rng.integers(1, n_players, n_games)) % n_players
    pexp = 1 / (1 + 10 ** (-(elo[w] - elo[b]) / 400))
    u = rng.random(n_games)
    res = np.where(u < pexp * 0.95, "1-0", np.where(u < pexp * 0.95 + 0.05, "1/2-1/2", "0-1"))
    raw = pd.DataFrame({"Event": "Rated Blitz game", "Site": [f"https://lichess.org/g{i}" for i in range(n_games)],
                        "White": np.array(names)[w], "Black": np.array(names)[b], "Result": res,
                        "UTCDate": starts.strftime("%Y.%m.%d"), "UTCTime": starts.strftime("%H:%M:%S"),
                        "WhiteElo": elo[w].astype(int).astype(str), "BlackElo": elo[b].astype(int).astype(str),
                        "TimeControl": rng.choice(["180+0", "300+0", "180+2"], n_games), "ECO": "B20",
                        "Opening": "x", "Termination": rng.choice(["Normal", "Time forfeit"], n_games)})
    from src.features import compact
    pg = player_games(clean_games(raw))
    pg = compact(pg, sorted(set(pg["termination"].astype(str)) | {"none"}))
    return prepare(assign_splits(pg, save=False))


@pytest.fixture(scope="module")
def data():
    return synthetic_player_games()


def test_bh_threshold():
    assert bh_threshold([0.001, 0.04, 0.5]) == 0.001  # 0.04 > 2*0.05/3
    assert bh_threshold([0.001, 0.03, 0.5]) == 0.03
    assert bh_threshold([0.2, 0.5]) == -1.0


def test_vault_one_shot():
    v = Vault(pd.DataFrame({"a": [1]}))
    v.read("H0001")
    with pytest.raises(VaultAccessError):
        v.read("H0001")
    assert [a["granted"] for a in v.accesses] == [True, False]


def test_pipeline_finds_planted_effect_and_kills_leak(data, tmp_path):
    d = data.copy()
    tilt = (d.prev_result == "loss").to_numpy()
    d["residual"] = d["residual"] - 0.06 * tilt  # strong planted tilt in all splits
    from src.judge import add_adjusted_outcome
    d = add_adjusted_outcome(d)
    led = Ledger(tmp_path / "ledger.jsonl")
    j = Judge(d, led, vault_log=tmp_path / "vault.jsonl")
    raws = [
        {"statement": "tilt", "condition": [{"feature": "prev_result", "op": "==", "value": "loss"}],
         "direction": "less", "min_effect": 0.01},
        {"statement": "leaky", "condition": [{"feature": "termination", "op": "==", "value": "Time forfeit"},
                                             {"feature": "prev_result", "op": "==", "value": "loss"}],
         "direction": "less", "min_effect": 0.01},
        {"statement": "bad", "condition": [{"feature": "nope", "op": "==", "value": 1}],
         "direction": "less", "min_effect": 0.01},
        {"statement": "tilt again", "condition": [{"feature": "prev_result", "op": "==", "value": "loss"}],
         "direction": "less", "min_effect": 0.01},
    ] + HAND[2:3]
    j.submit(raws, source="test")
    st = j.run()
    assert st["H0001"]["verdict"] == "REAL", st["H0001"]
    assert st["H0002"]["verdict"] == "LUCKY" and st["H0002"]["killed_at"] == "skeptic"
    assert "leakage" in st["H0002"]["reason"]
    assert st["H0003"]["verdict"] == "INVALID"
    assert st["H0004"]["verdict"] == "DUPLICATE"
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) > 10 and '"type": "proposed"' in lines[0]
    # Running again must not touch the vault a second time.
    j.run()
    assert sum(a["id"] == "H0001" for a in j.vault.accesses) == 1


def test_proposals_logged_before_tests(data, tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    j = Judge(data, led)
    j.submit(HAND[:3], source="hand")
    j.run()
    types = [e["type"] for e in led.entries]
    first_stage = types.index("stage")
    assert types[:first_stage] == ["proposed"] * 3
