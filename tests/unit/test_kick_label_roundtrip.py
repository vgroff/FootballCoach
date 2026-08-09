"""Verify that the BC label captured from a rules-AI kick round-trips through
the neural kick pathway (kick_with_direction) producing the same ball speed
and direction, with rng_reduction=1.0 (no noise).

Both rules-AI (kick_direct with compensate_for_run=True) and the neural path
(kick_with_direction, which also compensates) cancel run_mult, so ball speed
= max_kick * power_fraction regardless of player velocity.  The round-trip
must hold for stationary AND running players.
"""
from __future__ import annotations

import math
import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from tests.conftest import make_player


def _make_match_with_ball_carrier(
    position: Vector3,
    velocity: Vector3,
    heading_rad: float,
) -> tuple[Match, "Player"]:
    """Player at `position` moving at `velocity`, holding the ball; opponent far away."""
    player = make_player("trainee", Team.LEFT, position=position,
                         kick_precision=1.0, kick_power=0.8)
    opponent = make_player("opponent", Team.RIGHT, position=Vector3(50, 50, 0))
    ball = Ball.at_rest(position)
    match = Match(
        pitch=Pitch.standard(),
        players=[player, opponent],
        ball=ball,
        rng_reduction=1.0,   # no noise → deterministic
        rng=random.Random(0),
    )
    match._set_possession(player.player_id)
    player.velocity = velocity
    player.heading_rad = heading_rad
    return match, player


def _roundtrip_check(velocity: Vector3, heading_rad: float, power_fraction: float) -> None:
    """Assert kick_direct → label → kick_with_direction gives same ball speed and direction."""
    position = Vector3(0, 0, 0)
    aim_point = Vector3(20, 5, 0)

    # ── Step 1: rules-AI kick (compensate_for_run=True) ──
    match1, player1 = _make_match_with_ball_carrier(position, velocity, heading_rad)
    player1.kick_direct(match1, aim_point, power_fraction, Vector3.zero(), compensate_for_run=True)

    assert player1.kicked_this_tick
    label_direction = player1.last_kick_direction
    label_power     = player1.last_kick_power_fraction
    ball1_speed     = match1.ball.velocity.length()

    assert label_direction is not None
    assert label_power is not None
    assert ball1_speed > 0.0

    # ── Step 2: neural kick via kick_with_direction ──
    match2, player2 = _make_match_with_ball_carrier(position, velocity, heading_rad)
    player2.kick_with_direction(match2, label_direction, label_power, Vector3.zero())

    assert player2.kicked_this_tick
    ball2_speed = match2.ball.velocity.length()

    assert abs(ball1_speed - ball2_speed) < 1e-4, (
        f"velocity={velocity}  ball speeds differ: rules-AI={ball1_speed:.4f}  neural={ball2_speed:.4f}"
    )

    ball2_vel = match2.ball.velocity
    if ball2_vel.length() > 1e-6:
        ball2_dir = ball2_vel * (1.0 / ball2_vel.length())
        dot = (ball2_dir.x * label_direction.x
               + ball2_dir.y * label_direction.y
               + ball2_dir.z * label_direction.z)
        assert dot > 0.9999, (
            f"Direction mismatch: cosine similarity={dot:.6f} "
            f"velocity={velocity}  label={label_direction}  ball2_dir={ball2_dir}"
        )


def test_kick_direction_label_roundtrip_stationary():
    _roundtrip_check(velocity=Vector3.zero(), heading_rad=0.0, power_fraction=0.6)


def test_kick_direction_label_roundtrip_running_toward_kick():
    """Player sprinting in the same direction as the kick (run_mult > 1)."""
    _roundtrip_check(velocity=Vector3(6.0, 1.5, 0), heading_rad=math.atan2(1.5, 6.0), power_fraction=0.6)


def test_kick_direction_label_roundtrip_running_across():
    """Player running perpendicular to kick direction (run_mult ≈ 1)."""
    _roundtrip_check(velocity=Vector3(0, 5.0, 0), heading_rad=math.pi / 2, power_fraction=0.7)


def test_kick_direction_label_roundtrip_running_away():
    """Player running away from kick direction (run_mult < 1)."""
    _roundtrip_check(velocity=Vector3(-4.0, 0, 0), heading_rad=math.pi, power_fraction=0.8)


def test_kick_label_power_is_adjusted():
    """last_kick_power_fraction stores adjusted_power (what was passed to kick_ball),
    not the raw power_fraction arg — the network learns to output adjusted_power directly."""
    from footballcoach.engine.kicking import running_power_multiplier, compensate_power_for_run_mult, KickingParams
    from footballcoach.engine.movement import effective_top_speed

    position = Vector3(-10, 0, 0)
    aim_point = Vector3(30, 0, 0)   # straight ahead

    player = make_player("trainee", Team.LEFT, position=position,
                         kick_precision=1.0, kick_power=0.8)
    opponent = make_player("opponent", Team.RIGHT, position=Vector3(50, 0, 0))
    ball = Ball.at_rest(position)
    match = Match(
        pitch=Pitch.standard(),
        players=[player, opponent],
        ball=ball,
        rng_reduction=1.0,
        rng=random.Random(0),
    )
    match._set_possession(player.player_id)

    player.velocity = Vector3(5.0, 0, 0)
    player.heading_rad = 0.0

    power_fraction = 0.7
    player.kick_direct(match, aim_point, power_fraction, Vector3.zero(), compensate_for_run=True)

    top_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina,
        has_ball=True, ball_control_attr=player.attributes.ball_control,
    )
    run_mult = running_power_multiplier(
        match.kicking_params.running_power_coefficient,
        Vector3(5.0, 0, 0), aim_point - position, top_speed,
    )
    expected_adjusted = compensate_power_for_run_mult(power_fraction, run_mult)

    assert abs(player.last_kick_power_fraction - expected_adjusted) < 1e-6, (
        f"last_kick_power_fraction={player.last_kick_power_fraction} should equal "
        f"adjusted_power={expected_adjusted}, not raw power_fraction={power_fraction}"
    )
    assert abs(player.last_kick_power_fraction - power_fraction) > 1e-4, (
        "run_mult should differ from 1.0 for a running player — test setup issue"
    )
