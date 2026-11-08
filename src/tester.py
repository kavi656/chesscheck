"""Run one hypothesis on one split.

Outcome: `y_adj` = residual minus the player's own baseline, where the baseline is the player's mean
residual over games at least K positions away in the same split (`far_baseline`).
  * Subtracting a player baseline removes between-player confounding (overrated players lose more
    AND keep underperforming).
  * Excluding the K games either side removes the mechanical bias that a plain within-player
    (fixed-effects) comparison has when the condition is built from the player's own recent results
    (Miller & Sanjurjo 2018 "hot hand" bias / Nickell 1981 dynamic-panel bias). With an ordinary
    player mean, "after a loss" rows look ~+0.05 better on pure noise; see tests/test_tester.py.

Test: difference in mean y_adj (treated - control), standard error two-way cluster-robust by player
and by game (Cameron, Gelbach & Miller 2011): rows of the same player are correlated, and the two rows
of one game have residuals that are exact negatives of each other.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

MIN_TREATED = 500
MIN_TREATED_PLAYERS = 50
K_WINDOW = 6          # games each side excluded from the player baseline
OUTCOME_FEATURES = {"prev_result", "streak", "prev_termination", "termination"}  # derived from outcomes


@dataclass
class TestResult:
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p: float | None = None
    z: float | None = None
    se: float | None = None
    raw_diff: float | None = None
    n_treated: int = 0
    n_control: int = 0
    n_players: int = 0          # distinct players in treated group
    underpowered: bool = False
    # information-scale quantities for the live confidence sequence: S = effect * V, V = 1/se^2
    s_centered: float | None = None
    var_s: float | None = None
    scale: float | None = None

    def to_dict(self) -> dict:
        return {k: (float(f"{v:.8g}") if isinstance(v, float) else v) for k, v in asdict(self).items()}


def p_value(z: float, direction: str) -> float:
    if direction == "greater":
        return float(stats.norm.sf(z))
    if direction == "less":
        return float(stats.norm.cdf(z))
    return float(2 * stats.norm.sf(abs(z)))


def far_baseline(group: np.ndarray, y: np.ndarray, k: int = K_WINDOW) -> np.ndarray:
    """For rows sorted by time within contiguous groups, the mean of y over the group's rows more
    than k positions away. NaN when no such row exists."""
    n = len(y)
    y = np.asarray(y, dtype=np.float64)
    start = np.r_[True, group[1:] != group[:-1]]
    gid = np.cumsum(start) - 1
    gstart = np.nonzero(start)[0]
    gsize = np.diff(np.r_[gstart, n])
    s0, sz = gstart[gid], gsize[gid]
    pos = np.arange(n) - s0
    cs = np.r_[0.0, np.cumsum(y)]
    lo = s0 + np.maximum(pos - k, 0)
    hi = s0 + np.minimum(pos + k, sz - 1)
    win_sum = cs[hi + 1] - cs[lo]
    rest_n = sz - (hi - lo + 1)
    tot = cs[s0 + sz] - cs[s0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rest_n > 0, (tot - win_sum) / rest_n, np.nan)


def _cluster_var(codes: np.ndarray, u: np.ndarray) -> float:
    s = np.bincount(codes, weights=u)
    return float((s * s).sum())


def cluster_test(y: np.ndarray, treated: np.ndarray, control: np.ndarray, player: np.ndarray,
                 game: np.ndarray, direction: str, raw: np.ndarray | None = None) -> TestResult:
    """Difference in means of y between treated and control with a two-way (player, game)
    cluster-robust SE. Rows with NaN y are dropped from both groups."""
    ok = ~np.isnan(y)
    t, c = treated & ok, control & ok
    nT, nC = int(t.sum()), int(c.sum())
    res = TestResult(n_treated=nT, n_control=nC)
    if nT < 2 or nC < 2:
        res.underpowered = True
        return res
    yt, yc = y[t], y[c]
    mt, mc = yt.mean(), yc.mean()
    D = float(mt - mc)
    if raw is not None:
        res.raw_diff = float(raw[t].mean() - raw[c].mean())
    pop = t | c
    u = np.zeros(len(y))
    u[t] = (yt - mt) / nT
    u[c] = -(yc - mc) / nC
    up = u[pop]
    v_player = _cluster_var(player[pop], up)
    v_game = _cluster_var(game[pop], up)
    v_row = float((up * up).sum())
    var = max(v_player + v_game - v_row, v_player, v_game)  # CGM two-way can dip negative; floor it
    res.n_players = int(len(np.unique(player[t])))
    res.underpowered = nT < MIN_TREATED or res.n_players < MIN_TREATED_PLAYERS or var <= 0
    if var <= 0:
        return res
    se = float(np.sqrt(var))
    res.effect, res.se = D, se
    res.z = D / se
    res.p = p_value(res.z, direction)
    res.ci_low, res.ci_high = D - 1.96 * se, D + 1.96 * se
    res.var_s, res.s_centered, res.scale = 1.0 / var, D / var, var
    return res


def run(df, h, y_col: str = "y_adj") -> TestResult:
    from src.hypothesis import groups
    treated, control = groups(df, h)
    return cluster_test(df[y_col].to_numpy(dtype=np.float64), treated, control, df["player_code"].to_numpy(),
                        df["game_code"].to_numpy(), h.direction, raw=df["residual"].to_numpy(dtype=np.float64))


def _main() -> None:
    """Stage-2 acceptance: three hand-written hypotheses on explore + null p-value histogram."""
    from src.bank import HAND
    from src.calibrate import ShuffleNull
    from src.hypothesis import validate
    from src.judge import load_data

    df = load_data()
    ex_mask = (df["split"] == "explore").to_numpy()
    ex = df[ex_mask].reset_index(drop=True)
    print(f"explore rows: {len(ex):,}  (rows with a player baseline: {int(ex['y_adj'].notna().sum()):,})")
    for i, raw in enumerate(HAND[:3]):
        h = validate({**raw, "id": f"S{i+1}"})
        r = run(ex, h)
        print(f"\n{h.statement}\n  effect={r.effect:+.4f}  95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  p={r.p:.3g}"
              f"  raw_diff={r.raw_diff:+.4f}  n_T={r.n_treated:,}  n_C={r.n_control:,}  players={r.n_players:,}")
    hyps = [validate({**raw, "id": f"B{i}"}) for i, raw in enumerate(HAND)]
    null = ShuffleNull(df)
    ps = []
    for s in range(10):
        e = null.sample(s)[ex_mask].reset_index(drop=True)
        for h in hyps:
            r = run(e, h)
            if r.p is not None and not r.underpowered:
                ps.append(r.p)
    hist = np.histogram(ps, bins=10, range=(0, 1))[0]
    print(f"\nnull p-values (outcomes shuffled in time within player, sequence features recomputed;"
          f" {len(ps)} tests = 10 shuffles x {len(hyps)} hypotheses), deciles:")
    for i, c in enumerate(hist):
        print(f"  {i/10:.1f}-{(i+1)/10:.1f} {'#' * int(60 * c / max(hist))} {c}")
    print(f"  fraction p<0.05: {np.mean(np.array(ps) < 0.05):.3f}   KS vs uniform p={stats.kstest(ps, 'uniform').pvalue:.3f}")


if __name__ == "__main__":
    _main()
