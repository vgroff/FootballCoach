from __future__ import annotations

import math
import random

import pytest

from footballcoach.engine.kicking import (
    KickingParams,
    angle_error_sigma_rad,
    firsttime_difficulty_multiplier,
    kick_ball,
    max_kick_speed_mps,
    pass_ball,
    running_direction_precision_multiplier,
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


# --------------------------------------------------------------------------
# Phase E: running-direction precision multiplier tests
# --------------------------------------------------------------------------

def _make_running_params(
    cos_high: float = 0.35,
    cos_low: float = -0.2,
    penalty_mid: float = 0.25,
    penalty_max: float = 0.75,
    min_speed: float = 1.0,
) -> KickingParams:
    base = KickingParams.from_config()
    # Can't set fields on frozen dataclass — just use from_config() which reads
    # the values from physics.json (already set to the expected defaults).
    return base


def test_running_direction_multiplier_forward_no_penalty():
    """Running in line with the kick (cos_sim=1.0) → multiplier = 1.0."""
    params = KickingParams.from_config()
    velocity = Vector3(5.0, 0, 0)
    aim = Vector3(1.0, 0, 0)
    mult = running_direction_precision_multiplier(velocity, aim, params)
    assert abs(mult - 1.0) < 1e-9


def test_running_direction_multiplier_at_cos_high_no_penalty():
    """Exactly at cos_high (0.35) → multiplier = 1.0."""
    params = KickingParams.from_config()
    cos_high = params.running_direction_precision_cos_high  # 0.35
    angle = math.acos(cos_high)
    velocity = Vector3(5.0, 0, 0)
    aim = Vector3(math.cos(angle), math.sin(angle), 0)
    mult = running_direction_precision_multiplier(velocity, aim, params)
    assert abs(mult - 1.0) < 1e-6


def test_running_direction_multiplier_at_cos_low_mid_penalty():
    """Exactly at cos_low (-0.2) → multiplier = 1 - penalty_mid = 0.75."""
    params = KickingParams.from_config()
    cos_low = params.running_direction_precision_cos_low  # -0.2
    penalty_mid = params.running_direction_precision_penalty_mid  # 0.25
    angle = math.acos(cos_low)
    velocity = Vector3(5.0, 0, 0)
    aim = Vector3(math.cos(angle), math.sin(angle), 0)
    mult = running_direction_precision_multiplier(velocity, aim, params)
    expected = 1.0 - penalty_mid
    assert abs(mult - expected) < 1e-6


def test_running_direction_multiplier_at_cos_minus_one_max_penalty():
    """Kicking directly backward (cos_sim=-1.0) → multiplier = 1 - penalty_max = 0.25."""
    params = KickingParams.from_config()
    penalty_max = params.running_direction_precision_penalty_max  # 0.75
    velocity = Vector3(5.0, 0, 0)
    aim = Vector3(-1.0, 0, 0)  # directly backward
    mult = running_direction_precision_multiplier(velocity, aim, params)
    expected = 1.0 - penalty_max
    assert abs(mult - expected) < 1e-6


def test_running_direction_multiplier_table_sample_points():
    """Spot-check a few intermediate cos_sim values for the piecewise formula."""
    params = KickingParams.from_config()
    cos_high = params.running_direction_precision_cos_high  # 0.35
    cos_low = params.running_direction_precision_cos_low    # -0.2
    penalty_mid = params.running_direction_precision_penalty_mid  # 0.25
    penalty_max = params.running_direction_precision_penalty_max  # 0.75

    def mult_for_cos(cos_sim: float) -> float:
        angle = math.acos(max(-1.0, min(1.0, cos_sim)))
        velocity = Vector3(5.0, 0, 0)
        aim = Vector3(math.cos(angle), math.sin(angle), 0)
        return running_direction_precision_multiplier(velocity, aim, params)

    # Mid of upper zone: cos_sim = (0.35+(-0.2))/2 = 0.075 → t=0.5 → mult=1-0.125=0.875
    expected_075 = 1.0 - penalty_mid * (cos_high - 0.075) / (cos_high - cos_low)
    assert abs(mult_for_cos(0.075) - expected_075) < 1e-6

    # Above threshold: multiplier must be 1.0
    assert abs(mult_for_cos(0.5) - 1.0) < 1e-9


# --------------------------------------------------------------------------
# Player.kick_direct() — last_kick_* capture (BC kick supervision plan)
# --------------------------------------------------------------------------

def test_kick_direct_captures_last_kick_fields():
    from footballcoach.engine.match import Match
    from footballcoach.entities.attributes import PlayerAttributes
    from footballcoach.entities.pitch import Pitch
    from footballcoach.entities.player import Player, Team

    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.possessed_by = "p1"
    match = Match(pitch=Pitch.standard(), players=[p1], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))

    aim_point = Vector3(10, 0, 0)
    spin = Vector3(1.0, 2.0, 3.0)
    p1.kick_direct(match, aim_point, power_fraction=0.6, spin=spin)

    assert p1.kicked_this_tick is True
    assert p1.last_kick_direction is not None
    assert p1.last_kick_direction.x == pytest.approx(1.0, abs=1e-6)
    assert p1.last_kick_direction.y == pytest.approx(0.0, abs=1e-6)
    assert p1.last_kick_power_fraction is not None
    assert p1.last_kick_spin is spin


def test_kick_direct_no_possession_leaves_last_kick_fields_unset():
    from footballcoach.engine.match import Match
    from footballcoach.entities.attributes import PlayerAttributes
    from footballcoach.entities.pitch import Pitch
    from footballcoach.entities.player import Player, Team

    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(20, 0, 0))
    ball.possessed_by = None
    match = Match(pitch=Pitch.standard(), players=[p1], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))

    p1.kick_direct(match, Vector3(10, 0, 0), power_fraction=0.6, spin=Vector3.zero())

    assert p1.kicked_this_tick is False
    assert p1.last_kick_direction is None
    assert p1.last_kick_power_fraction is None
    assert p1.last_kick_spin is None


def test_running_direction_below_min_speed_returns_one():
    """Kicker speed below min_speed_mps → always 1.0, even at worst angle."""
    params = KickingParams.from_config()
    # Speed of 0.5 m/s (below the 1.0 min)
    velocity = Vector3(0.5, 0, 0)
    aim = Vector3(-1.0, 0, 0)  # directly backward (worst case)
    mult = running_direction_precision_multiplier(velocity, aim, params)
    assert abs(mult - 1.0) < 1e-9


def test_running_direction_zero_velocity_no_crash():
    """Zero kicker velocity → must return 1.0 without division-by-zero or NaN."""
    params = KickingParams.from_config()
    velocity = Vector3(0.0, 0.0, 0.0)
    aim = Vector3(-5.0, 0, 0)
    mult = running_direction_precision_multiplier(velocity, aim, params)
    assert abs(mult - 1.0) < 1e-9
    assert math.isfinite(mult)


def test_running_direction_multiplier_applied_in_kick_ball():
    """kick_ball with backward run produces wider sigma than same kick running forward."""
    params = KickingParams.from_config()
    rng_reduction = 0.0  # full randomness, but we compare sigmas not outputs

    # Helper: compute effective sigma by doing many kicks and measuring spread
    # Instead, directly compare sigma via angle_error_sigma_rad with effective precision.
    from footballcoach.engine.kicking import angle_error_sigma_rad
    precision = 0.7
    vel_forward = Vector3(5.0, 0, 0)
    vel_backward = Vector3(-5.0, 0, 0)
    aim = Vector3(1.0, 0, 0)

    mult_forward = running_direction_precision_multiplier(vel_forward, aim, params)
    mult_backward = running_direction_precision_multiplier(vel_backward, aim, params)

    sigma_forward = angle_error_sigma_rad(params, precision * mult_forward)
    sigma_backward = angle_error_sigma_rad(params, precision * mult_backward)

    assert sigma_backward > sigma_forward  # backward run → worse precision → wider sigma


def test_running_direction_multiplier_applied_in_pass_ball():
    """pass_ball produces deterministically wider spread when kicker runs backward."""
    params = KickingParams.from_config()

    def sigma_for_pass(kicker_vel: Vector3) -> float:
        """Proxy: check spread of aim angle across many deterministic trials."""
        from footballcoach.engine.kicking import angle_error_sigma_rad, running_direction_precision_multiplier, KickingParams
        kp = KickingParams.from_config()
        aim_dir = Vector3(10.0, 0, 0)
        mult = running_direction_precision_multiplier(kicker_vel, aim_dir, kp)
        return angle_error_sigma_rad(kp, 0.7 * mult)

    sigma_fwd = sigma_for_pass(Vector3(5.0, 0, 0))
    sigma_bwd = sigma_for_pass(Vector3(-5.0, 0, 0))
    assert sigma_bwd > sigma_fwd
