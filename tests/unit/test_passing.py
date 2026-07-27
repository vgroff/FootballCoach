from __future__ import annotations

import random

from footballcoach.engine.kicking import PassingParams, pass_angle_error_sigma_rad, pass_ball, pass_speed_mps
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3


def test_pass_angle_error_never_zero_at_max_precision():
    params = PassingParams.from_config()
    assert pass_angle_error_sigma_rad(params, 1.0) > 0.0


def test_pass_angle_error_more_forgiving_than_shot_error():
    from footballcoach.engine.kicking import KickingParams, angle_error_sigma_rad

    pass_params = PassingParams.from_config()
    kick_params = KickingParams.from_config()
    for precision in (0.2, 0.5, 0.8):
        assert pass_angle_error_sigma_rad(pass_params, precision) < angle_error_sigma_rad(kick_params, precision)


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
