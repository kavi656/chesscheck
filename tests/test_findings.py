import numpy as np

from src.findings import cluster, similarity


def test_mirror_nested_and_unrelated():
    rng = np.random.default_rng(0)
    n = 10_000
    white = rng.random(n) < 0.5
    quick_loss = rng.random(n) < 0.08
    quick_loss_2min = quick_loss & (rng.random(n) < 0.5)   # nested variant of quick_loss
    quick_win = ~quick_loss & (rng.random(n) < 0.08)       # disjoint, opposite sign: NOT a mirror
    masks = {"white": (white, ~white), "black": (~white, white),
             "tilt5": (quick_loss, ~quick_loss), "tilt2": (quick_loss_2min, ~quick_loss_2min),
             "win5": (quick_win, ~quick_win)}
    signs = {"white": 1, "black": -1, "tilt5": -1, "tilt2": -1, "win5": 1}
    groups, _ = cluster(sorted(masks), masks, signs)
    assert sorted(map(sorted, groups)) == [["black", "white"], ["tilt2", "tilt5"], ["win5"]]
    # a small group overlaps White ~50% by chance: chance-adjusted similarity ~ 0
    assert abs(similarity(*masks["tilt5"], 1, *masks["white"], 1)) < 0.1
