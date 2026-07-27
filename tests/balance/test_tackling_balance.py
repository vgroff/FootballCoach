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
        attempt_tackle(0.8, 0.6, RNG_REDUCTION, random.Random(seed), params)
        for seed in range(N_TRIALS)
    )
    win_rate_pct = round(100 * wins / N_TRIALS, 2)
    stats = {"n_trials": N_TRIALS, "wins": wins, "win_rate_pct": win_rate_pct}
    balance_recorder.report("tackling_0.8_vs_dribbling_0.6", stats)
    assert 70.0 < win_rate_pct < 90.0


def test_tackling_win_rate_table_across_attribute_pairs(balance_recorder):
    """Not a hard-target test - reports a table of win rates across a grid
    of tackling/dribbling values so the balance can be visually inspected."""
    params = TacklingParams.from_config()
    table = {}
    for tackling in (0.2, 0.4, 0.5, 0.6, 0.8, 0.95):
        for dribbling in (0.2, 0.4, 0.5, 0.6, 0.8, 0.95):
            wins = sum(
                attempt_tackle(tackling, dribbling, RNG_REDUCTION, random.Random(seed), params)
                for seed in range(1000)
            )
            table[f"tackling={tackling}_vs_dribbling={dribbling}"] = round(100 * wins / 1000, 1)
    balance_recorder.report("tackling_win_rate_grid_pct", table)
    # Sanity: equal-attribute matchups should favour the tackler due to the boost.
    assert table["tackling=0.5_vs_dribbling=0.5"] > 50.0
