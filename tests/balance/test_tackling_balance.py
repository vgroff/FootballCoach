"""Balance test: tackling win rate. User's explicit target: a player with
0.8 tackling should win a challenge against a player with 0.6 dribbling
more than 70% but less than 90% of the time, at rng_reduction=0.3.
"""
from __future__ import annotations

import random

from footballcoach.engine.tackling import TacklingParams, attempt_tackle

N_TRIALS = 5000
RNG_REDUCTION = 0.3


def test_tackling_08_vs_dribbling_06_win_rate(balance_recorder):
    params = TacklingParams.from_config()
    wins = sum(
        1 for seed in range(N_TRIALS)
        if attempt_tackle(0.8, 0.6, RNG_REDUCTION, random.Random(seed), params).tackler_won
    )
    win_rate_pct = round(100 * wins / N_TRIALS, 2)
    stats = {"n_trials": N_TRIALS, "wins": wins, "win_rate_pct": win_rate_pct}
    balance_recorder.report("tackling_0.8_vs_dribbling_0.6", stats)
    assert 70.0 < win_rate_pct < 90.0


def test_goalkeeper_tackling_boost(balance_recorder):
    """Goalkeeper tackle boost (+100%) should win challenges far more often
    than an outfield player with the same attribute."""
    params = TacklingParams.from_config()
    n = 2000
    gk_wins = sum(
        1 for seed in range(n)
        if attempt_tackle(0.5, 0.8, RNG_REDUCTION, random.Random(seed), params, is_goalkeeper_tackle=True).tackler_won
    )
    outfield_wins = sum(
        1 for seed in range(n)
        if attempt_tackle(0.5, 0.8, RNG_REDUCTION, random.Random(seed), params, is_goalkeeper_tackle=False).tackler_won
    )
    gk_rate = round(100 * gk_wins / n, 2)
    outfield_rate = round(100 * outfield_wins / n, 2)
    stats = {"n": n, "gk_win_rate_pct": gk_rate, "outfield_win_rate_pct": outfield_rate}
    balance_recorder.report("goalkeeper_tackle_boost", stats)
    # GK (2.0 boost) should win majority even against a stronger dribbler (0.8 vs 0.5).
    assert gk_rate > 60.0
    # Outfield (1.2 boost, matched against 0.8 dribbler) should rarely win.
    assert outfield_rate < 40.0


def test_tackling_win_rate_table_across_attribute_pairs(balance_recorder):
    """Not a hard-target test - reports a table of win rates across a grid
    of tackling/dribbling values so the balance can be visually inspected."""
    params = TacklingParams.from_config()
    table = {}
    for tackling in (0.2, 0.4, 0.5, 0.6, 0.8, 0.95):
        for dribbling in (0.2, 0.4, 0.5, 0.6, 0.8, 0.95):
            wins = sum(
                1 for seed in range(1000)
                if attempt_tackle(tackling, dribbling, RNG_REDUCTION, random.Random(seed), params).tackler_won
            )
            table[f"tackling={tackling}_vs_dribbling={dribbling}"] = round(100 * wins / 1000, 1)
    balance_recorder.report("tackling_win_rate_grid_pct", table)
    # Sanity: equal-attribute matchups should favour the tackler due to the boost.
    assert table["tackling=0.5_vs_dribbling=0.5"] > 50.0
