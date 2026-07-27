"""Balance tests for the 'Shoot' action: aimed at dead centre of the
opponent's goal, no goalkeeper. User's explicit target: all players should
score >50% of goals within the box on "shoot". Scenarios are randomly
generated (but constrained to reasonable in-box positions/angles) so the
target is validated across a spread of situations, not just one spot.
"""
from __future__ import annotations

import random

from footballcoach import actions
from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from tests.conftest import make_player

RNG_REDUCTION = 0.3
N_TRIALS = 500


def _random_in_box_position(pitch: Pitch, rng: random.Random) -> tuple[float, float]:
    """A random position inside the box, at least 3m from the goal line (a
    shot taken standing on the line isn't a meaningful scenario) and at
    least 1m from the touchline edges of the box."""
    x = pitch.half_length - rng.uniform(3.0, pitch.box_length_m - 0.5)
    half_box_w = pitch.box_width_m / 2.0
    y = rng.uniform(-half_box_w + 1.0, half_box_w - 1.0)
    return x, y


def _run_shoot_trial(pitch, precision: float, power: float, x: float, y: float, seed: int) -> bool:
    from footballcoach.mathutils import Vector3

    kicker = make_player(
        "k", Team.LEFT, position=Vector3(x, y, 0),
        kick_precision=precision, kick_power=power,
    )
    ball = Ball.at_rest(kicker.position)
    ball.possessed_by = "k"
    match = Match(pitch=pitch, players=[kicker], ball=ball, rng_reduction=RNG_REDUCTION, rng=random.Random(seed))

    actions.shoot(kicker, pitch)

    for _ in range(150):
        match.step()
        if match.scoreboard.left_goals > 0:
            return True
        if match.ball.position.x > pitch.half_length + 2.0:
            return False  # sailed past the goal line wide/high
        if match.ball.velocity.length() < 0.1 and match.ball.position.x < pitch.half_length - 1.0:
            return False  # stopped short
    return False


def _run_random_shoot_batch(precision: float, power: float, n_trials: int, seed_offset: int) -> dict:
    pitch = Pitch.standard()
    rng = random.Random(seed_offset)
    scored = 0
    for i in range(n_trials):
        x, y = _random_in_box_position(pitch, rng)
        if _run_shoot_trial(pitch, precision, power, x, y, seed_offset * 100000 + i):
            scored += 1
    return {
        "n_trials": n_trials,
        "scored": scored,
        "score_rate_pct": round(100 * scored / n_trials, 2),
    }


def test_low_attribute_player_scores_over_50_percent_in_box(balance_recorder):
    stats = _run_random_shoot_batch(precision=0.2, power=0.3, n_trials=N_TRIALS, seed_offset=1)
    balance_recorder.report("shoot_low_attrs_random_in_box", stats)
    assert stats["score_rate_pct"] > 50.0


def test_mid_attribute_player_scores_over_50_percent_in_box(balance_recorder):
    stats = _run_random_shoot_batch(precision=0.5, power=0.5, n_trials=N_TRIALS, seed_offset=2)
    balance_recorder.report("shoot_mid_attrs_random_in_box", stats)
    assert stats["score_rate_pct"] > 50.0


def test_high_attribute_player_scores_over_50_percent_in_box(balance_recorder):
    stats = _run_random_shoot_batch(precision=0.85, power=0.85, n_trials=N_TRIALS, seed_offset=3)
    balance_recorder.report("shoot_high_attrs_random_in_box", stats)
    assert stats["score_rate_pct"] > 50.0


def test_shoot_scoring_rate_table_across_attribute_and_position(balance_recorder):
    """Not a hard-target test - reports a table across precision levels and
    a few illustrative fixed positions (centre, near-post edge, far edge of
    box) for visual balance inspection."""
    pitch = Pitch.standard()
    half_box_w = pitch.box_width_m / 2.0
    positions = {
        "penalty_spot": (pitch.penalty_spot(left=False).x, 0.0),
        "edge_of_box_centre": (pitch.half_length - pitch.box_length_m + 1.0, 0.0),
        "edge_of_box_wide_angle": (pitch.half_length - pitch.box_length_m + 1.0, half_box_w - 2.0),
        "six_yard_box": (pitch.half_length - pitch.six_yard_length_m + 1.0, 3.0),
    }
    table = {}
    for precision in (0.2, 0.5, 0.8):
        for label, (x, y) in positions.items():
            scored = sum(
                _run_shoot_trial(pitch, precision, precision, x, y, seed)
                for seed in range(200)
            )
            table[f"precision={precision}_{label}"] = round(100 * scored / 200, 1)
    balance_recorder.report("shoot_scoring_rate_grid_pct", table)
    # Sanity: higher precision should never score less often than lower
    # precision from the same spot.
    for label in positions:
        assert table[f"precision=0.8_{label}"] >= table[f"precision=0.2_{label}"]
