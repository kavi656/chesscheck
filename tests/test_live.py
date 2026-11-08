import numpy as np

from src.live import confidence_sequence, game_to_header


def test_confidence_sequence_time_uniform_coverage():
    # Peek after every batch; the CS must contain the true mean (0) at ALL peeks in >= ~95% of runs.
    rng = np.random.default_rng(0)
    fails = 0
    for _ in range(400):
        s, v, ok = 0.0, 0.0, True
        for _ in range(200):
            x = rng.normal(0, 1, 20)
            s += x.sum(); v += 20.0
            lo, hi = confidence_sequence(s, v, 1.0, rho=500.0)
            if not lo <= 0 <= hi:
                ok = False
                break
        fails += not ok
    assert fails / 400 <= 0.06


def test_naive_peeking_would_fail():
    # Sanity: a fixed-n 95% interval checked 200 times is violated far more than 5% of the time.
    rng = np.random.default_rng(1)
    fails = 0
    for _ in range(400):
        s = np.cumsum(rng.normal(0, 1, (200, 20)).sum(1)); v = 20.0 * np.arange(1, 201)
        fails += np.any(np.abs(s) > 1.96 * np.sqrt(v))
    assert fails / 400 > 0.2


def test_game_to_header():
    g = {"id": "abc", "rated": True, "variant": "standard", "speed": "blitz", "createdAt": 1794054681864,
         "status": "outoftime", "winner": "black", "clock": {"initial": 180, "increment": 2},
         "players": {"white": {"user": {"name": "A"}, "rating": 1500}, "black": {"user": {"name": "B"}, "rating": 1600}},
         "opening": {"eco": "C20", "name": "x"}}
    h = game_to_header(g)
    assert h["Result"] == "0-1" and h["TimeControl"] == "180+2" and h["Termination"] == "Time forfeit"
    assert game_to_header({**g, "status": "started"}) is None
