from __future__ import annotations

from footballcoach.engine.possession import ControlTimeParams, compute_difficulty, control_time_s, height_difficulty_factor


def test_height_factor_flat_below_knee():
    params = ControlTimeParams.from_config()
    assert height_difficulty_factor(params, 0.0) == params.height_factor_knee
    assert height_difficulty_factor(params, params.knee_height_m) == params.height_factor_knee


def test_height_factor_increases_monotonically():
    params = ControlTimeParams.from_config()
    heights = [0.0, 0.2, 0.49, 0.7, 0.95, 1.3, 1.8, 2.2]
    factors = [height_difficulty_factor(params, h) for h in heights]
    assert factors == sorted(factors)


def test_height_factor_capped_above_head():
    params = ControlTimeParams.from_config()
    very_high = height_difficulty_factor(params, 10.0)
    assert very_high <= params.height_factor_max


def test_control_time_increases_with_height():
    params = ControlTimeParams.from_config()
    t_ground = control_time_s(params, 0.1, 0.0, 0.0, ball_control_attr=0.5)
    t_head = control_time_s(params, 1.8, 0.0, 0.0, ball_control_attr=0.5)
    assert t_head > t_ground


def test_control_time_increases_with_relative_velocity():
    params = ControlTimeParams.from_config()
    t_slow = control_time_s(params, 0.3, relative_speed_mps=1.0, player_speed_mps=0.0, ball_control_attr=0.5)
    t_fast = control_time_s(params, 0.3, relative_speed_mps=10.0, player_speed_mps=0.0, ball_control_attr=0.5)
    assert t_fast > t_slow


def test_control_time_decreases_with_ball_control():
    params = ControlTimeParams.from_config()
    t_low_control = control_time_s(params, 1.0, 2.0, 1.0, ball_control_attr=0.1)
    t_high_control = control_time_s(params, 1.0, 2.0, 1.0, ball_control_attr=0.9)
    assert t_high_control < t_low_control


def test_goalkeeper_in_box_faster_than_outfield_for_same_ball():
    params = ControlTimeParams.from_config()
    t_outfield = control_time_s(params, 1.5, 5.0, 2.0, ball_control_attr=0.5, is_goalkeeper_in_box=False)
    t_gk = control_time_s(params, 1.5, 5.0, 2.0, ball_control_attr=0.5, is_goalkeeper_in_box=True)
    assert t_gk < t_outfield


def test_control_time_never_below_base_at_zero_difficulty():
    params = ControlTimeParams.from_config()
    t = control_time_s(params, 0.0, 0.0, 0.0, ball_control_attr=1.0)
    assert t >= params.t_base_s
