from __future__ import annotations

import random

from footballcoach.engine.kicking import (
    KickingParams,
    angle_error_sigma_rad,
    firsttime_difficulty_multiplier,
    kick_ball,
    max_kick_speed_mps,
    solve_launch_pitch_rad,
)
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3


def test_angle_error_never_zero_at_max_precision():
    params = KickingParams.from_config()
    assert angle_error_sigma_rad(params, 1.0) > 0.0


def test_angle_error_decreases_with_precision():
    params = KickingParams.from_config()
    assert angle_error_sigma_rad(params, 0.9) < angle_error_sigma_rad(params, 0.1)


def test_kick_power_scales_with_attribute():
    params = KickingParams.from_config()
    assert max_kick_speed_mps(params, 1.0) > max_kick_speed_mps(params, 0.0)


def test_firsttime_multiplier_increases_error_with_difficulty():
    params = KickingParams.from_config()
    base = firsttime_difficulty_multiplier(params, kick_precision=0.5, difficulty=0.0)
    harder = firsttime_difficulty_multiplier(params, kick_precision=0.5, difficulty=2.0)
    assert base == 1.0
    assert harder > base


def test_solve_launch_pitch_hits_higher_target_with_more_arc():
    flat_pitch = solve_launch_pitch_rad(10.0, 0.0, 20.0, 9.81)
    high_pitch = solve_launch_pitch_rad(10.0, 2.0, 20.0, 9.81)
    assert high_pitch > flat_pitch


def test_kick_ball_releases_possession_and_sets_velocity():
    ball = Ball.at_rest(Vector3(0, 0, 0.11))
    ball.possessed_by = "p1"
    rng = random.Random(1)
    kick_ball(
        ball,
        kicker_position=Vector3(0, 0, 0),
        aim_point=Vector3(10, 0, 2.0),
        power_fraction=0.5,
        kick_precision=0.8,
        kick_power_attr=0.8,
        spin=Vector3.zero(),
        rng_reduction=1.0,  # deterministic
        rng=rng,
    )
    assert ball.possessed_by is None
    assert ball.velocity.length() > 0
    assert ball.velocity.x > 0  # roughly forward
    assert ball.velocity.z > 0  # some lift, aiming at a raised aim_point


def test_kick_ball_deterministic_with_rng_reduction_one():
    params = KickingParams.from_config()
    ball1 = Ball.at_rest(Vector3(0, 0, 0.11))
    ball1.possessed_by = "p1"
    ball2 = Ball.at_rest(Vector3(0, 0, 0.11))
    ball2.possessed_by = "p1"

    kick_ball(
        ball1, Vector3(0, 0, 0), Vector3(10, 0, 0), 0.7, 0.6, 0.6, Vector3.zero(),
        rng_reduction=1.0, rng=random.Random(1), params=params,
    )
    kick_ball(
        ball2, Vector3(0, 0, 0), Vector3(10, 0, 0), 0.7, 0.6, 0.6, Vector3.zero(),
        rng_reduction=1.0, rng=random.Random(999), params=params,
    )
    # With rng_reduction=1.0, sigma=0, so different seeds should not matter.
    assert ball1.velocity.x == ball2.velocity.x
    assert ball1.velocity.y == ball2.velocity.y
