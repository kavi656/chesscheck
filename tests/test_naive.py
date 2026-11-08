from src.bank import HAND
from src.calibrate import ShuffleNull
from src.judge import Judge
from src.ledger import Ledger
from src.naive import honest_fate, naive_run
from tests.test_judge import synthetic_player_games


def test_auto_naive_finds_noise_honest_kills_it():
    d = ShuffleNull(synthetic_player_games(n_players=300, n_games=60000, seed=2)).sample(0)
    j = Judge(d, Ledger(None, "t"))
    naive = naive_run(j, [dict(h) for h in HAND], "auto")
    fate = honest_fate(j, naive)
    assert len(fate) >= 1                       # p-hacking manufactures "discoveries" from noise
    assert sum(f["verdict"] == "REAL" for f in fate) == 0
    assert all(f["verdict"] not in ("DUPLICATE", "INVALID") for f in fate)
