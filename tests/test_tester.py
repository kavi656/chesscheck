import numpy as np
from scipy import stats

from src.tester import cluster_test, far_baseline


def _sequences(n_players=4000, games=30, seed=0, het_sd=0.03):
    """i.i.d. games per player with a persistent per-player offset; NO effect of history."""
    rng = np.random.default_rng(seed)
    player = np.repeat(np.arange(n_players), games)
    a = rng.normal(0, het_sd, n_players)[player]
    p_win = np.clip(0.48 + a, 0.05, 0.95)
    u = rng.random(len(player))
    score = np.where(u < p_win, 1.0, np.where(u < p_win + 0.04, 0.5, 0.0))
    y = score - 0.5
    first = np.r_[True, player[1:] != player[:-1]]
    prev = np.where(first, np.nan, np.r_[np.nan, score[:-1]])
    game = np.arange(len(player))  # each row its own game (no opponent rows here)
    return player, y, prev == 0.0, game


def test_far_baseline_matches_bruteforce():
    rng = np.random.default_rng(0)
    g = np.repeat([0, 1, 2], [20, 5, 14])
    y = rng.normal(size=len(g))
    b = far_baseline(g, y, k=3)
    for i in range(len(g)):
        idx = np.nonzero(g == g[i])[0]
        pos = i - idx[0]
        far = [j for j in idx if abs((j - idx[0]) - pos) > 3]
        expect = np.mean(y[far]) if far else np.nan
        assert (np.isnan(expect) and np.isnan(b[i])) or np.isclose(b[i], expect)


def test_plain_player_mean_is_biased_far_baseline_is_not():
    """The Miller-Sanjurjo / Nickell trap: conditioning on the player's own previous result."""
    player, y, after_loss, game = _sequences()
    plain = y - np.bincount(player, y)[player] / np.bincount(player)[player]   # ordinary fixed effect
    adj = y - far_baseline(player, y, k=6)
    r_plain = cluster_test(plain, after_loss, ~after_loss, player, game, "two-sided")
    r_adj = cluster_test(adj, after_loss, ~after_loss, player, game, "two-sided")
    assert r_plain.effect > 0.02 and r_plain.p < 1e-10        # spurious "after a loss they do better"
    assert abs(r_adj.effect) < 3 * r_adj.se                     # unbiased
    assert r_adj.p > 0.001


def test_null_p_values_roughly_uniform():
    ps = []
    for s in range(300):
        player, y, after_loss, game = _sequences(n_players=300, games=30, seed=s)
        adj = y - far_baseline(player, y, k=6)
        ps.append(cluster_test(adj, after_loss, ~after_loss, player, game, "two-sided").p)
    ps = np.array(ps)
    assert stats.kstest(ps, "uniform").pvalue > 0.01
    assert 0.02 < (ps < 0.05).mean() < 0.09


def test_detects_planted_effect():
    player, y, after_loss, game = _sequences(seed=3)
    y2 = y - 0.04 * after_loss
    adj = y2 - far_baseline(player, y2, k=6)
    r = cluster_test(adj, after_loss, ~after_loss, player, game, "less")
    assert r.p < 1e-4 and r.ci_low < -0.04 < r.ci_high


def test_two_way_clustering_handles_paired_game_rows():
    """Every game gives one treated (white) and one control (black) row with y_b = -y_w.
    One-way player clustering would understate the SE by ~sqrt(2); two-way must get it right."""
    zs = []
    for s in range(300):
        rng = np.random.default_rng(s)
        n_games, n_players = 3000, 300
        w, b = rng.integers(0, n_players, n_games), rng.integers(0, n_players, n_games)
        yw = rng.choice([0.5, -0.5], n_games)
        player = np.r_[w, b]
        y = np.r_[yw, -yw]
        game = np.r_[np.arange(n_games), np.arange(n_games)]
        white = np.r_[np.ones(n_games, bool), np.zeros(n_games, bool)]
        zs.append(cluster_test(y, white, ~white, player, game, "two-sided").z)
    assert 0.8 < np.var(zs) < 1.25


def test_underpowered_flag():
    player, y, after_loss, game = _sequences(n_players=20, games=10)
    r = cluster_test(y, after_loss, ~after_loss, player, game, "less")
    assert r.underpowered
