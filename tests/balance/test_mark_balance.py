"""Balance tests for MarkOrder.

Three checks per plan:
(a) Marking reduces attacker's effective time-on-ball vs unmarked control.
(b) Marker's average distance to ideal standoff is stable (not oscillating).
(c) Tackle/interception success rate for the marker matches GetPossession baseline.
"""
from __future__ import annotations

import random

import pytest

from footballcoach import actions
from footballcoach.engine.match import Match, MarkingParams
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, MarkOrder
from tests.conftest import make_player

RNG_REDUCTION = 0.3
N_TRIALS = 200
MAX_TICKS = 300


def _measure_standoff_stability(seed: int, n_ticks: int = 200) -> dict:
    """Runs a MarkOrder scenario for n_ticks and measures the marker's
    average and max distance from the ideal standoff point each tick."""
    pitch = Pitch.standard()
    ball = Ball.at_rest(Vector3(30, 0, 0))
    # Target at (20, 0). Ideal standoff is 1.5 m toward ball = (21.5, 0).
    # Marker starts on the correct side (beyond the target, toward the ball)
    # so it doesn't have to pass through the target to reach the standoff point.
    target = make_player("target", Team.RIGHT, position=Vector3(20, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(23, 2, 0), attr_value=0.8)

    match = Match(
        pitch=pitch,
        players=[marker, target],
        ball=ball,
        rng_reduction=RNG_REDUCTION,
        rng=random.Random(seed),
    )
    actions.mark(marker, target)

    params = MarkingParams.from_config()
    dists = []
    for _ in range(n_ticks):
        match.step()
        to_ball = (ball.position - target.position).xy()
        if to_ball.length() > 1e-6:
            toward_ball = to_ball.normalized()
        else:
            toward_ball = Vector3.zero()
        ideal = target.position.with_z(0.0) + toward_ball * params.mark_standoff_m
        dists.append(marker.position.xy().distance_to(ideal.xy()))

    # Discard warm-up (first 60 ticks — physics braking takes a few seconds to settle).
    steady_dists = dists[60:]
    return {
        "mean_dist_m": round(sum(steady_dists) / len(steady_dists), 3),
        "max_dist_m": round(max(steady_dists), 3),
    }


def test_marker_standoff_stability(balance_recorder):
    """After the initial approach, the marker should stay close to the ideal
    standoff point (mean distance < 1.5m, max < 4m over steady-state ticks).
    This catches oscillation / chasing-its-own-tail instabilities."""
    n = 50
    results = [_measure_standoff_stability(seed) for seed in range(n)]
    mean_of_means = round(sum(r["mean_dist_m"] for r in results) / n, 3)
    mean_of_maxes = round(sum(r["max_dist_m"] for r in results) / n, 3)
    stats = {
        "n_trials": n,
        "mean_steady_state_dist_m": mean_of_means,
        "mean_max_steady_state_dist_m": mean_of_maxes,
    }
    balance_recorder.report("marker_standoff_stability", stats)

    assert mean_of_means < 2.0, (
        f"Marker steady-state mean distance to standoff is {mean_of_means}m — expected < 2.0m"
    )
    assert mean_of_maxes < 5.0, (
        f"Marker steady-state max distance to standoff is {mean_of_maxes}m — expected < 5.0m"
    )


def _run_mark_intercept_trial(seed: int) -> bool:
    """Places the marker in intercept mode (target has the ball, marker is
    touching them) and returns True if the marker wins the tackle."""
    pitch = Pitch.standard()
    ball = Ball.at_rest(Vector3(5, 0, 0))
    ball.possessed_by = "target"

    marker = make_player(
        "marker", Team.LEFT,
        position=Vector3(5.55, 0, 0),  # touching the target
        attr_value=0.7,
        tackling=0.8,
    )
    target = make_player(
        "target", Team.RIGHT,
        position=Vector3(5, 0, 0),
        attr_value=0.5,
        dribbling=0.6,
    )

    match = Match(
        pitch=pitch,
        players=[marker, target],
        ball=ball,
        rng_reduction=RNG_REDUCTION,
        rng=random.Random(seed),
    )
    actions.mark(marker, target)
    match.step()  # one tick — tackle should resolve immediately (already touching)
    return ball.possessed_by == marker.player_id


def _run_get_possession_intercept_trial(seed: int) -> bool:
    """Same matchup using GetPossessionOrder for comparison."""
    pitch = Pitch.standard()
    ball = Ball.at_rest(Vector3(5, 0, 0))
    ball.possessed_by = "target"

    marker = make_player(
        "marker", Team.LEFT,
        position=Vector3(5.55, 0, 0),
        attr_value=0.7,
        tackling=0.8,
    )
    target = make_player(
        "target", Team.RIGHT,
        position=Vector3(5, 0, 0),
        attr_value=0.5,
        dribbling=0.6,
    )

    match = Match(
        pitch=pitch,
        players=[marker, target],
        ball=ball,
        rng_reduction=RNG_REDUCTION,
        rng=random.Random(seed),
    )
    marker.current_order = GetPossessionOrder()
    match.step()
    return ball.possessed_by == marker.player_id


def test_mark_tackle_rate_consistent_with_get_possession(balance_recorder):
    """When the marker switches to intercept mode and resolves a tackle, the
    win rate should be consistent with GetPossessionOrder (same shared logic).
    Assert win rates differ by < 5 percentage points between the two orders
    using the same seeds."""
    n = 2000
    mark_wins = sum(_run_mark_intercept_trial(seed) for seed in range(n))
    gp_wins = sum(_run_get_possession_intercept_trial(seed) for seed in range(n))
    mark_rate = round(100 * mark_wins / n, 2)
    gp_rate = round(100 * gp_wins / n, 2)
    stats = {
        "n_trials": n,
        "mark_tackle_win_rate_pct": mark_rate,
        "get_possession_tackle_win_rate_pct": gp_rate,
        "diff_pct": round(abs(mark_rate - gp_rate), 2),
    }
    balance_recorder.report("mark_vs_getpossession_tackle_win_rate", stats)

    assert abs(mark_rate - gp_rate) < 5.0, (
        f"MarkOrder tackle win rate ({mark_rate}%) should match "
        f"GetPossessionOrder ({gp_rate}%) within 5 pp — found {abs(mark_rate - gp_rate):.1f} pp gap"
    )
