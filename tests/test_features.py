import numpy as np
import pandas as pd

from src.features import clean_games, player_games


def _game(i, white, black, result, t, welo=1500, belo=1500, term="Normal"):
    d, tm = t.split(" ")
    return {"Event": "Rated Blitz game", "Site": f"https://lichess.org/g{i}", "White": white,
            "Black": black, "Result": result, "UTCDate": d, "UTCTime": tm, "WhiteElo": str(welo),
            "BlackElo": str(belo), "TimeControl": "180+0", "ECO": "C20", "Opening": "x",
            "Termination": term}


def _table():
    raw = pd.DataFrame([
        _game(1, "A", "X", "1-0", "2026.10.01 10:00:00"),          # A win
        _game(2, "Y", "A", "0-1", "2026.10.01 10:04:00"),          # A win (black)
        _game(3, "A", "Z", "1/2-1/2", "2026.10.01 10:10:00"),      # A draw
        _game(4, "A", "X", "0-1", "2026.10.01 10:15:00", term="Time forfeit"),  # A loss
        _game(5, "A", "Y", "0-1", "2026.10.01 11:30:00"),          # A loss, new session (>30m gap)
        _game(6, "A", "Z", "1-0", "2026.10.02 00:01:00", welo=1900, belo=1500),  # next day
    ])
    pg = player_games(clean_games(raw))
    return pg[pg.player == "A"].reset_index(drop=True)


def test_prev_result_and_gap():
    a = _table()
    assert list(a.prev_result) == ["none", "win", "win", "draw", "loss", "loss"]
    assert np.isnan(a.minutes_since_prev_game[0])
    assert list(a.minutes_since_prev_game[1:5]) == [4.0, 6.0, 5.0, 75.0]
    assert list(a.prev_termination) == ["none", "Normal", "Normal", "Normal", "Time forfeit", "Normal"]


def test_streak_uses_only_previous_games():
    a = _table()
    assert list(a.streak) == [0, 1, 2, 0, -1, -2]


def test_sessions_and_daily_counts():
    a = _table()
    assert list(a.session_index) == [1, 2, 3, 4, 1, 1]
    assert list(a.games_played_today) == [0, 1, 2, 3, 4, 0]


def test_scores_and_residuals():
    a = _table()
    assert list(a.color) == ["white", "black", "white", "white", "white", "white"]
    assert list(a.score) == [1.0, 1.0, 0.5, 0.0, 0.0, 1.0]
    assert np.allclose(a.expected_score[:5], 0.5)
    assert np.isclose(a.expected_score[5], 1 / (1 + 10 ** (-400 / 400)))
    assert np.allclose(a.residual, a.score - a.expected_score)
    assert a.rating_band[5] == "1600-2000"
    assert a.weekday[0] == "Thu"  # 2026-10-01 is a Thursday


def test_residuals_sum_to_zero_per_game():
    raw = pd.DataFrame([_game(1, "A", "B", "1-0", "2026.10.01 10:00:00", welo=1700, belo=1450)])
    pg = player_games(clean_games(raw))
    assert np.isclose(pg.residual.sum(), 0.0)
