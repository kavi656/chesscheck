import numpy as np

from src.bank import HAND
from src.calibrate import Calibrator, summarise, within_player_permutation
from tests.test_judge import synthetic_player_games


def test_permutation_stays_within_player():
    rng = np.random.default_rng(0)
    player = np.array([0, 0, 0, 1, 1, 2])
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 99.0])
    yn = within_player_permutation(player, y, rng)
    for p in (0, 1, 2):
        assert sorted(yn[player == p]) == sorted(y[player == p])


def test_calibrator_null_and_planted():
    data = synthetic_player_games(n_players=300, n_games=60000, seed=1)
    cal = Calibrator(data, HAND)
    null = cal.null_runs(5)
    planted = cal.planted_runs(3, effects=[0.08])
    s = summarise(null, planted, HAND)
    assert s["false_real_rate_per_hypothesis"] <= 0.05
    assert planted["0.08"]["power"] >= 2 / 3
