from __future__ import annotations

import math

from footballcoach.engine.movement import (
    MovementParams,
    SpeedMode,
    ball_carry_speed_multiplier,
    drain_stamina,
    effective_acceleration,
    effective_top_speed,
    max_acceleration_mps2,
    max_turn_rate_rad_s,
    regen_stamina,
    stamina_multiplier,
    step_player_towards,
    top_speed_mps,
)
from footballcoach.mathutils import Vector3
from tests.conftest import make_player


def test_top_speed_range():
    params = MovementParams.from_config()
    assert math.isclose(top_speed_mps(params, 0.0), params.top_speed_base_mps)
    assert math.isclose(top_speed_mps(params, 1.0), params.top_speed_base_mps + params.top_speed_scale_mps)


def test_acceleration_range():
    params = MovementParams.from_config()
    assert math.isclose(max_acceleration_mps2(params, 0.0), params.accel_base_mps2)
    assert math.isclose(max_acceleration_mps2(params, 1.0), params.accel_base_mps2 + params.accel_scale_mps2)


def test_stamina_multiplier_bounds():
    params = MovementParams.from_config()
    assert stamina_multiplier(params, 1.0) == 1.0
    assert math.isclose(stamina_multiplier(params, 0.0), 1.0 - params.stamina_speed_penalty_max)


def test_ball_carry_speed_never_reaches_full():
    params = MovementParams.from_config()
    mult_at_max_control = ball_carry_speed_multiplier(params, 1.0)
    assert mult_at_max_control < 1.0
    mult_at_min_control = ball_carry_speed_multiplier(params, 0.0)
    assert mult_at_min_control < mult_at_max_control


def test_effective_top_speed_with_ball_is_slower():
    params = MovementParams.from_config()
    no_ball = effective_top_speed(params, 0.5, 1.0, has_ball=False)
    with_ball = effective_top_speed(params, 0.5, 1.0, has_ball=True, ball_control_attr=0.5)
    assert with_ball < no_ball


def test_turn_rate_decreases_with_speed():
    params = MovementParams.from_config()
    slow_turn = max_turn_rate_rad_s(params, 0.5, speed_mps=1.0, has_ball=False)
    fast_turn = max_turn_rate_rad_s(params, 0.5, speed_mps=8.0, has_ball=False)
    assert slow_turn > fast_turn


def test_turn_rate_with_ball_low_control_is_worse():
    params = MovementParams.from_config()
    no_ball = max_turn_rate_rad_s(params, 0.5, speed_mps=5.0, has_ball=False)
    with_ball_low_control = max_turn_rate_rad_s(params, 0.5, speed_mps=5.0, has_ball=True, ball_control_attr=0.0)
    with_ball_high_control = max_turn_rate_rad_s(params, 0.5, speed_mps=5.0, has_ball=True, ball_control_attr=1.0)
    assert with_ball_low_control < no_ball
    assert with_ball_low_control < with_ball_high_control
    assert math.isclose(with_ball_high_control, no_ball)


def test_stamina_drains_and_regenerates():
    params = MovementParams.from_config()
    drained = drain_stamina(params, 1.0, stamina_attr=0.5, effort=1.0, dt_s=10.0)
    assert drained < 1.0
    regened = regen_stamina(params, drained, stamina_attr=0.5, dt_s=10.0)
    assert regened > drained


def test_higher_stamina_attr_drains_slower():
    params = MovementParams.from_config()
    low_attr = drain_stamina(params, 1.0, stamina_attr=0.0, effort=1.0, dt_s=5.0)
    high_attr = drain_stamina(params, 1.0, stamina_attr=1.0, effort=1.0, dt_s=5.0)
    assert high_attr > low_attr  # drains less => higher remaining stamina


def test_step_player_towards_moves_in_target_direction():
    player = make_player(position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    for _ in range(120):
        step_player_towards(player, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=1 / 30)
    assert player.position.x > 5.0
    assert abs(player.position.y) < 1e-6


def test_step_player_towards_decelerates_on_zero_target():
    player = make_player(position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    player.velocity = Vector3(5.0, 0.0, 0.0)
    for _ in range(60):
        step_player_towards(player, Vector3.zero(), SpeedMode.STANDSTILL, dt_s=1 / 30)
    assert player.velocity.length() < 0.5


def test_standstill_decelerates_faster_than_jog():
    """STANDSTILL should stop a moving player faster than JOG (standstill_decel_multiplier > 1)."""
    params = MovementParams.from_config()

    def ticks_to_stop(mode: SpeedMode) -> int:
        player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
        player.heading_rad = 0.0
        player.velocity = Vector3(6.0, 0.0, 0.0)
        for i in range(300):
            step_player_towards(player, Vector3.zero(), mode, dt_s=1 / 30, params=params)
            if player.speed_mps < 0.05:
                return i
        return 300

    standstill_ticks = ticks_to_stop(SpeedMode.STANDSTILL)
    jog_ticks = ticks_to_stop(SpeedMode.JOG)
    assert standstill_ticks < jog_ticks, (
        f"STANDSTILL should stop faster than JOG: {standstill_ticks} vs {jog_ticks} ticks"
    )


def test_standstill_snap_clears_drift():
    """A player at near-zero speed in STANDSTILL mode should snap to exactly
    zero (physics-level snap), preventing infinite creep."""
    player = make_player(position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    player.velocity = Vector3(0.015, 0.0, 0.0)  # below _STOP_SNAP_THRESHOLD_MPS (0.02)
    step_player_towards(player, Vector3.zero(), SpeedMode.STANDSTILL, dt_s=1 / 30)
    assert player.speed_mps == 0.0, "velocity should be snapped to zero by physics-level guard"
