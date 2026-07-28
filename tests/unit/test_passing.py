from __future__ import annotations

import random

from footballcoach.engine.kicking import KickingParams, PassingParams, kick_sigma_rad, pass_ball, pass_speed_mps
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3


def test_kick_sigma_never_zero_at_max_precision():
    """Kicks are never perfectly accurate even at precision=1.0."""
    kp = KickingParams.from_config()
    assert kick_sigma_rad(kp, 1.0, effective_power=0.0, rng_reduction=0.0) > 0.0


def test_low_power_pass_less_sigma_than_high_power_shot():
    """With the unified model, lower effective power → lower sigma."""
    kp = KickingParams.from_config()
    sigma_pass = kick_sigma_rad(kp, 0.5, effective_power=0.2, rng_reduction=0.0)
    sigma_shot = kick_sigma_rad(kp, 0.5, effective_power=1.0, rng_reduction=0.0)
    assert sigma_pass < sigma_shot


def test_higher_precision_always_lower_sigma():
    kp = KickingParams.from_config()
    for power in (0.0, 0.5, 1.0):
        low_prec = kick_sigma_rad(kp, 0.2, power, rng_reduction=0.3)
        high_prec = kick_sigma_rad(kp, 0.9, power, rng_reduction=0.3)
        assert high_prec < low_prec


def test_pass_speed_increases_with_distance():
    params = PassingParams.from_config()
    short = pass_speed_mps(params, 5.0, 9.81, 0.06)
    long = pass_speed_mps(params, 30.0, 9.81, 0.06)
    assert long > short


def test_pass_speed_clamped_to_bounds():
    params = PassingParams.from_config()
    very_short = pass_speed_mps(params, 0.01, 9.81, 0.06)
    very_long = pass_speed_mps(params, 200.0, 9.81, 0.06)
    assert very_short >= params.min_speed_mps
    assert very_long <= params.max_speed_mps


def test_pass_ball_releases_possession_and_moves_toward_target():
    ball = Ball.at_rest(Vector3(0, 0, 0.11))
    ball.possessed_by = "p1"
    rng = random.Random(1)
    pass_ball(ball, Vector3(0, 0, 0), Vector3(10, 0, 0), kick_precision=0.7, rng_reduction=1.0, rng=rng)
    assert ball.possessed_by is None
    assert ball.velocity.x > 0
    assert abs(ball.velocity.y) < 1e-6  # straight pass, rng_reduction=1.0 -> zero error


def test_pass_ball_deterministic_with_rng_reduction_one():
    params = PassingParams.from_config()
    ball1 = Ball.at_rest(Vector3(0, 0, 0.11))
    ball1.possessed_by = "p1"
    ball2 = Ball.at_rest(Vector3(0, 0, 0.11))
    ball2.possessed_by = "p1"

    pass_ball(ball1, Vector3(0, 0, 0), Vector3(10, 5, 0), 0.5, rng_reduction=1.0, rng=random.Random(1), params=params)
    pass_ball(ball2, Vector3(0, 0, 0), Vector3(10, 5, 0), 0.5, rng_reduction=1.0, rng=random.Random(999), params=params)

    assert ball1.velocity.x == ball2.velocity.x
    assert ball1.velocity.y == ball2.velocity.y


def test_pass_ball_manual_power_fraction_overrides_auto_pace():
    ball_auto = Ball.at_rest(Vector3(0, 0, 0.11))
    ball_auto.possessed_by = "p1"
    ball_manual = Ball.at_rest(Vector3(0, 0, 0.11))
    ball_manual.possessed_by = "p1"

    pass_ball(ball_auto, Vector3(0, 0, 0), Vector3(10, 0, 0), 0.5, rng_reduction=1.0, rng=random.Random(1))
    pass_ball(ball_manual, Vector3(0, 0, 0), Vector3(10, 0, 0), 0.5, rng_reduction=1.0, rng=random.Random(1), power_fraction=1.0)

    assert ball_manual.velocity.length() != ball_auto.velocity.length()
