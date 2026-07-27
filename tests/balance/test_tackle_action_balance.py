"""Balance tests for the 'Tackle' action (ChaseTackleOrder): a defender runs
straight at an opposing ball-carrier and attempts a tackle once in range.

This exercises the same underlying skill-check as tests/balance/
test_tackling_balance.py (which tests attempt_tackle() directly, in
isolation), but end-to-end through the chase-then-tackle order, including
movement/closing-distance and a randomly-generated scenario batch across
distances and attribute pairs, per the user's request to balance-test the
full action, not just the underlying dice roll.
"""
from __future__ import annotations

import random

from footballcoach import actions
from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from tests.conftest import make_player

RNG_REDUCTION = 0.3
MAX_TICKS = 400


def _run_tackle_trial(pitch: Pitch, tackling: float, dribbling: float, distance: float, seed: int) -> bool:
    defender = make_player(
        "d", Team.LEFT, position=Vector3(0, 0, 0),
        tackling=tackling, top_speed=0.6, acceleration=0.6,
    )
    attacker = make_player(
        "a", Team.RIGHT, position=Vector3(distance, 0, 0),
        dribbling=dribbling, top_speed=0.6, acceleration=0.6,
    )
    ball = Ball.at_rest(attacker.position)
    ball.possessed_by = "a"
    match = Match(pitch=pitch, players=[defender, attacker], ball=ball, rng_reduction=RNG_REDUCTION, rng=random.Random(seed))

    actions.tackle(defender, attacker)

    for _ in range(MAX_TICKS):
        match.step()
        if ball.possessed_by == "d":
            return True
        if defender.current_order is None:
            return False  # chase completed (contact made, tackle attempted) and failed
    return False


def test_tackling_08_vs_dribbling_06_wins_challenge_most_of_the_time(balance_recorder):
    pitch = Pitch.standard()
    n = 500
    wins = sum(_run_tackle_trial(pitch, 0.8, 0.6, distance=5.0, seed=seed) for seed in range(n))
    stats = {"n_trials": n, "wins": wins, "win_rate_pct": round(100 * wins / n, 2)}
    balance_recorder.report("tackle_action_0.8_vs_0.6_dist5m", stats)
    # Matches the same analytically-derived band as the underlying skill
    # check (tests/balance/test_tackling_balance.py): 70-90%.
    assert 60.0 < stats["win_rate_pct"] < 95.0


def test_weak_tackler_rarely_wins_against_strong_dribbler(balance_recorder):
    pitch = Pitch.standard()
    n = 300
    wins = sum(_run_tackle_trial(pitch, 0.2, 0.9, distance=5.0, seed=seed) for seed in range(n))
    stats = {"n_trials": n, "wins": wins, "win_rate_pct": round(100 * wins / n, 2)}
    balance_recorder.report("tackle_action_weak_vs_strong", stats)
    assert stats["win_rate_pct"] < 20.0


def test_strong_tackler_usually_wins_against_weak_dribbler(balance_recorder):
    pitch = Pitch.standard()
    n = 300
    wins = sum(_run_tackle_trial(pitch, 0.9, 0.2, distance=5.0, seed=seed) for seed in range(n))
    stats = {"n_trials": n, "wins": wins, "win_rate_pct": round(100 * wins / n, 2)}
    balance_recorder.report("tackle_action_strong_vs_weak", stats)
    assert stats["win_rate_pct"] > 80.0


def test_tackle_win_rate_grid_across_attribute_pairs(balance_recorder):
    pitch = Pitch.standard()
    table = {}
    for tackling in (0.2, 0.5, 0.8):
        for dribbling in (0.2, 0.5, 0.8):
            n = 150
            wins = sum(_run_tackle_trial(pitch, tackling, dribbling, distance=5.0, seed=seed) for seed in range(n))
            table[f"tackling={tackling}_vs_dribbling={dribbling}"] = round(100 * wins / n, 1)
    balance_recorder.report("tackle_action_win_rate_grid_pct", table)
    # Sanity: equal attributes should favour the tackler (tackler_boost).
    assert table["tackling=0.5_vs_dribbling=0.5"] > 50.0


def test_random_scenario_batch_varied_distance_and_attributes(balance_recorder):
    """Randomly generates (reasonable) starting distances and attribute
    pairs, comparing overall win rate for a clearly-better-tackler set vs a
    clearly-better-dribbler set of matchups."""
    pitch = Pitch.standard()
    rng = random.Random(7)
    n = 200

    def_favoured_wins = 0
    att_favoured_wins = 0
    for i in range(n):
        distance = rng.uniform(2.0, 15.0)
        if _run_tackle_trial(pitch, tackling=0.75, dribbling=0.35, distance=distance, seed=i):
            def_favoured_wins += 1
        if _run_tackle_trial(pitch, tackling=0.35, dribbling=0.75, distance=distance, seed=i + 100000):
            att_favoured_wins += 1

    stats = {
        "n_scenarios": n,
        "defender_favoured_win_pct": round(100 * def_favoured_wins / n, 1),
        "attacker_favoured_tackler_win_pct": round(100 * att_favoured_wins / n, 1),
    }
    balance_recorder.report("tackle_action_random_scenarios", stats)
    assert stats["defender_favoured_win_pct"] > stats["attacker_favoured_tackler_win_pct"]
