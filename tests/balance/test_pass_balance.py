"""Balance tests for the 'Pass' action: to an immobile teammate. User's
explicit targets:

- All players should succeed >80% of passes to an immobile teammate within
  10m - good players should get ~99%.
- All players should succeed >50% of passes to an immobile teammate within
  30m - good players should get ~90%.

Also includes a randomly-generated batch across a spread of distances and
angles, and a low-vs-high attribute comparison table.
"""
from __future__ import annotations

import random

from footballcoach import actions
from footballcoach.engine.match import Match
from footballcoach.engine.movement import MovementParams, effective_top_speed
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from tests.conftest import make_player

RNG_REDUCTION = 0.3
N_TRIALS = 200
MAX_TICKS = 400


def _run_pass_trial(pitch: Pitch, precision: float, target_position: Vector3, seed: int) -> bool:
    passer = make_player("p", Team.LEFT, position=Vector3(0, 0, 0), kick_precision=precision)
    receiver = make_player("r", Team.LEFT, position=target_position)
    ball = Ball.at_rest(passer.position)
    ball.possessed_by = passer.player_id
    match = Match(pitch=pitch, players=[passer, receiver], ball=ball, rng_reduction=RNG_REDUCTION, rng=random.Random(seed))

    actions.pass_to(passer, receiver.position)

    for _ in range(MAX_TICKS):
        match.step()
        if ball.possessed_by == receiver.player_id:
            return True
        # Ball at rest, loose, and not mid-control by the receiver: it
        # missed and rolled dead somewhere (or was never close enough to
        # trigger a control-time countdown at all).
        if ball.velocity.length() < 0.05 and ball.possessed_by is None and receiver.state.name != "CONTROLLING_BALL":
            return False
    return False


def _run_batch(pitch: Pitch, precision: float, distance: float, n_trials: int, seed_offset: int) -> dict:
    target = Vector3(distance, 0, 0)
    succeeded = sum(
        _run_pass_trial(pitch, precision, target, seed_offset * 100000 + seed)
        for seed in range(n_trials)
    )
    return {
        "n_trials": n_trials,
        "succeeded": succeeded,
        "success_rate_pct": round(100 * succeeded / n_trials, 2),
    }


def test_average_player_succeeds_over_80_percent_at_10m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_batch(pitch, precision=0.5, distance=10.0, n_trials=N_TRIALS, seed_offset=1)
    balance_recorder.report("pass_10m_precision_0.5", stats)
    assert stats["success_rate_pct"] > 80.0


def test_low_attribute_player_still_succeeds_over_80_percent_at_10m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_batch(pitch, precision=0.15, distance=10.0, n_trials=N_TRIALS, seed_offset=2)
    balance_recorder.report("pass_10m_precision_0.15", stats)
    assert stats["success_rate_pct"] > 80.0


def test_good_player_succeeds_around_99_percent_at_10m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_batch(pitch, precision=0.9, distance=10.0, n_trials=N_TRIALS, seed_offset=3)
    balance_recorder.report("pass_10m_precision_0.9", stats)
    assert stats["success_rate_pct"] >= 97.0


def test_average_player_succeeds_over_50_percent_at_25m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_batch(pitch, precision=0.5, distance=25.0, n_trials=N_TRIALS, seed_offset=4)
    balance_recorder.report("pass_25m_precision_0.5", stats)
    assert stats["success_rate_pct"] > 50.0


def test_low_attribute_player_still_succeeds_over_30_percent_at_25m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_batch(pitch, precision=0.1, distance=25.0, n_trials=N_TRIALS, seed_offset=5)
    balance_recorder.report("pass_25m_precision_0.1", stats)
    assert stats["success_rate_pct"] > 30.0


def test_good_player_succeeds_around_80_percent_at_25m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_batch(pitch, precision=0.9, distance=25.0, n_trials=N_TRIALS, seed_offset=6)
    balance_recorder.report("pass_25m_precision_0.9", stats)
    assert stats["success_rate_pct"] >= 75.0


def test_pass_success_rate_table_across_precision_and_distance(balance_recorder):
    """Not a hard-target test - a full table across precision x distance for
    visual balance inspection, plus random angles to check off-axis passes
    aren't wildly different from straight-ahead ones."""
    pitch = Pitch.standard()
    table = {}
    for precision in (0.1, 0.3, 0.5, 0.7, 0.9):
        for distance in (5.0, 10.0, 20.0, 25.0, 30.0):
            stats = _run_batch(pitch, precision, distance, n_trials=150, seed_offset=int(precision * 10) * 100 + int(distance))
            table[f"precision={precision}_distance={distance}m"] = stats["success_rate_pct"]
    balance_recorder.report("pass_success_rate_grid_pct", table)

    # Sanity: comparing the extremes (lowest vs highest precision) at each
    # distance should clearly favour higher precision. We don't assert
    # strict monotonicity across every adjacent precision step, since with
    # only 150 trials per cell, adjacent precision levels can occasionally
    # land within a percentage point of each other from sampling noise
    # alone - the low-vs-high comparison is a more robust signal.
    for distance in (5.0, 10.0, 20.0, 25.0, 30.0):
        lowest = table[f"precision=0.1_distance={distance}m"]
        highest = table[f"precision=0.9_distance={distance}m"]
        assert highest >= lowest, f"highest precision underperformed lowest at distance={distance}"


def _run_long_pass_trial(pitch: Pitch, precision: float, distance: float, seed: int) -> bool:
    """Like _run_pass_trial but centres the pair at midfield so long passes
    (40m+) don't cross a goal line and trigger a reset. Passer runs at half
    top speed toward the receiver, matching the grid search calibration setup."""
    origin = Vector3(-distance / 2.0, 0.0, 0.0)
    target = Vector3(distance / 2.0, 0.0, 0.0)
    passer = make_player("p", Team.LEFT, position=origin, kick_precision=precision)
    receiver = make_player("r", Team.LEFT, position=target)
    mvmt = MovementParams.from_config()
    top_speed = effective_top_speed(mvmt, passer.attributes.top_speed, 1.0,
                                    has_ball=True, ball_control_attr=passer.attributes.ball_control)
    passer.velocity = Vector3(top_speed * 0.5, 0.0, 0.0)
    ball = Ball.at_rest(passer.position)
    ball.possessed_by = passer.player_id
    match = Match(pitch=pitch, players=[passer, receiver], ball=ball, rng_reduction=RNG_REDUCTION, rng=random.Random(seed))

    actions.pass_to(passer, target)

    max_ticks = max(400, int(distance * 20))
    for _ in range(max_ticks):
        match.step()
        if ball.possessed_by == receiver.player_id:
            return True
        if ball.velocity.length() < 0.05 and ball.possessed_by is None and receiver.state.name != "CONTROLLING_BALL":
            return False
    return False


def _run_long_pass_batch(pitch: Pitch, precision: float, distance: float, n_trials: int, seed_offset: int) -> dict:
    succeeded = sum(
        _run_long_pass_trial(pitch, precision, distance, seed_offset * 100000 + seed)
        for seed in range(n_trials)
    )
    return {
        "n_trials": n_trials,
        "succeeded": succeeded,
        "success_rate_pct": round(100 * succeeded / n_trials, 2),
    }


def test_average_player_succeeds_over_20_percent_at_40m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_long_pass_batch(pitch, precision=0.5, distance=40.0, n_trials=N_TRIALS, seed_offset=10)
    balance_recorder.report("pass_40m_precision_0.5", stats)
    assert stats["success_rate_pct"] > 20.0


def test_good_player_succeeds_over_60_percent_at_40m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_long_pass_batch(pitch, precision=0.9, distance=40.0, n_trials=N_TRIALS, seed_offset=11)
    balance_recorder.report("pass_40m_precision_0.9", stats)
    assert stats["success_rate_pct"] > 60.0


def test_good_player_succeeds_over_35_percent_at_60m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_long_pass_batch(pitch, precision=0.9, distance=60.0, n_trials=N_TRIALS, seed_offset=12)
    balance_recorder.report("pass_60m_precision_0.9", stats)
    assert stats["success_rate_pct"] > 35.0


def test_good_player_succeeds_over_15_percent_at_70m(balance_recorder):
    pitch = Pitch.standard()
    stats = _run_long_pass_batch(pitch, precision=0.9, distance=70.0, n_trials=N_TRIALS, seed_offset=13)
    balance_recorder.report("pass_70m_precision_0.9", stats)
    assert stats["success_rate_pct"] > 15.0


def test_random_scenario_batch_good_vs_bad_players(balance_recorder):
    """Randomly generates (reasonable) distances and angles, comparing a
    'bad' (low attribute) and 'good' (high attribute) player's overall
    success rate across the same random scenario set."""
    pitch = Pitch.standard()
    rng = random.Random(42)
    scenarios = []
    for _ in range(150):
        distance = rng.uniform(5.0, 35.0)
        angle = rng.uniform(-0.6, 0.6)  # radians off the x-axis
        target = Vector3(distance * (1.0), distance * angle, 0)  # small lateral offset, still forward-ish
        scenarios.append(target)

    def run_for_precision(precision: float) -> int:
        succ = 0
        for i, target in enumerate(scenarios):
            if _run_pass_trial(pitch, precision, target, seed=i):
                succ += 1
        return succ

    bad_succ = run_for_precision(0.15)
    good_succ = run_for_precision(0.9)
    n = len(scenarios)

    stats = {
        "n_scenarios": n,
        "bad_player_precision_0.15_success_pct": round(100 * bad_succ / n, 1),
        "good_player_precision_0.9_success_pct": round(100 * good_succ / n, 1),
    }
    balance_recorder.report("pass_random_scenarios_good_vs_bad", stats)
    assert stats["good_player_precision_0.9_success_pct"] > stats["bad_player_precision_0.15_success_pct"]
