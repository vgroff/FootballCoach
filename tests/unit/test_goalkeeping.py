from __future__ import annotations

from footballcoach.engine.goalkeeping import (
    GoalkeepingParams,
    early_intercept_target,
    own_goal_x,
    predict_goal_line_crossing,
    save_target_position,
)
from footballcoach.engine.possession import ControlTimeParams, control_time_s
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3


def test_own_goal_x_left_and_right():
    pitch = Pitch.standard()
    assert own_goal_x(pitch, Team.LEFT) == -pitch.half_length
    assert own_goal_x(pitch, Team.RIGHT) == pitch.half_length


def test_predict_crossing_straight_shot_no_gravity_drift():
    # A shot moving purely in +x with no y/z velocity should cross at the
    # same y, and (due to gravity) at a lower z than launch height.
    crossing = predict_goal_line_crossing(
        ball_position=Vector3(0, 2, 1),
        ball_velocity=Vector3(20, 0, 0),
        target_x=10,
        gravity_mps2=9.81,
    )
    assert crossing is not None
    assert abs(crossing.y - 2) < 1e-9
    assert crossing.z < 1.0


def test_predict_crossing_none_when_moving_away():
    crossing = predict_goal_line_crossing(
        ball_position=Vector3(0, 0, 1),
        ball_velocity=Vector3(-5, 0, 0),
        target_x=10,
        gravity_mps2=9.81,
    )
    assert crossing is None


def test_predict_crossing_none_when_stationary():
    crossing = predict_goal_line_crossing(
        ball_position=Vector3(0, 0, 1),
        ball_velocity=Vector3(0, 0, 0),
        target_x=10,
        gravity_mps2=9.81,
    )
    assert crossing is None


def test_predict_crossing_diagonal_shot():
    crossing = predict_goal_line_crossing(
        ball_position=Vector3(0, 0, 0),
        ball_velocity=Vector3(10, 5, 0),
        target_x=10,
        gravity_mps2=9.81,
    )
    assert crossing is not None
    assert abs(crossing.y - 5) < 1e-6  # t=1s, y = 0 + 5*1


def test_save_target_clamped_to_goal_frame():
    pitch = Pitch.standard()
    params = GoalkeepingParams.from_config()
    # A shot aimed way outside the goal width should clamp to the post.
    wide_shot_velocity = Vector3(20, 50, 0)
    target = save_target_position(
        pitch, Team.LEFT, Vector3(0, 0, 1), wide_shot_velocity, 9.81, params
    )
    half_goal_w = pitch.goal_width_m / 2.0
    assert abs(target.y) <= half_goal_w + 1e-6


def test_save_target_defaults_to_goal_centre_when_no_shot_incoming():
    pitch = Pitch.standard()
    params = GoalkeepingParams.from_config()
    target = save_target_position(
        pitch, Team.LEFT, Vector3(-40, 3, 0), Vector3(0, 0, 0), 9.81, params
    )
    assert abs(target.y) < 1e-6
    assert target.x < 0  # somewhere near the LEFT team's own goal


def test_save_target_plane_in_front_of_true_goal_line():
    pitch = Pitch.standard()
    params = GoalkeepingParams.from_config()
    target = save_target_position(
        pitch, Team.LEFT, Vector3(-40, 0, 1), Vector3(20, 0, 0), 9.81, params
    )
    # Target plane should be strictly inside the pitch relative to the true
    # goal line (i.e. closer to the centre spot), per the "intercept before
    # crossing" design rationale.
    assert target.x > -pitch.half_length


# --------------------------------------------------------------------------
# Phase C: early_intercept_target tests
# --------------------------------------------------------------------------

def _make_gk_params(
    early_intercept_max_distance_m: float = 10.0,
    early_intercept_safety_margin: float = 0.85,
) -> GoalkeepingParams:
    base = GoalkeepingParams.from_config()
    # Reconstruct with custom values (frozen dataclass — must pass all fields).
    return GoalkeepingParams(
        goal_frame_margin_m=base.goal_frame_margin_m,
        default_position_fraction_of_half_length=base.default_position_fraction_of_half_length,
        early_intercept_max_distance_m=early_intercept_max_distance_m,
        early_intercept_safety_margin=early_intercept_safety_margin,
    )


def test_early_intercept_close_slow_shot_returns_point_closer_than_goal_line():
    """A slow close-range shot: GK should get an early intercept point that
    is strictly closer to the GK than the default goal-line target."""
    pitch = Pitch.standard()
    params = _make_gk_params()
    # GK near the left goal, ball just inside the box heading slowly toward goal
    gk_pos = Vector3(-pitch.half_length + 3.0, 0, 0)
    ball_pos = Vector3(-pitch.half_length + 8.0, 1.0, 0.5)  # ~5m in front of GK
    ball_vel = Vector3(-5.0, 0, 0)  # heading toward goal at moderate speed
    gk_top_speed = 7.0

    intercept = early_intercept_target(
        gk_position=gk_pos,
        gk_effective_top_speed_mps=gk_top_speed,
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        pitch=pitch,
        team=Team.LEFT,
        gravity_mps2=9.81,
        params=params,
    )
    goal_line_target = save_target_position(pitch, Team.LEFT, ball_pos, ball_vel, 9.81, params)

    assert intercept is not None, "Expected early intercept for a close slow shot"
    # Intercept should be closer to GK than the goal-line target
    dist_intercept = gk_pos.xy().distance_to(intercept.xy())
    dist_goal_line = gk_pos.xy().distance_to(goal_line_target.xy())
    assert dist_intercept < dist_goal_line


def test_early_intercept_ball_beyond_max_distance_returns_none():
    """Ball further than early_intercept_max_distance_m → must return None."""
    pitch = Pitch.standard()
    params = _make_gk_params(early_intercept_max_distance_m=10.0)
    gk_pos = Vector3(-pitch.half_length + 1.0, 0, 0)
    # Ball 15m away — well beyond the 10m gate
    ball_pos = Vector3(-pitch.half_length + 16.0, 0, 0.5)
    ball_vel = Vector3(-15.0, 0, 0)

    result = early_intercept_target(
        gk_position=gk_pos,
        gk_effective_top_speed_mps=7.0,
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        pitch=pitch,
        team=Team.LEFT,
        gravity_mps2=9.81,
        params=params,
    )
    assert result is None


def test_early_intercept_no_shot_incoming_returns_none():
    """Ball not moving toward the goal → no crossing → must return None."""
    pitch = Pitch.standard()
    params = _make_gk_params()
    gk_pos = Vector3(-pitch.half_length + 2.0, 0, 0)
    ball_pos = Vector3(-pitch.half_length + 5.0, 0, 0.5)
    ball_vel = Vector3(5.0, 0, 0)  # moving AWAY from goal (wrong direction for LEFT team)

    result = early_intercept_target(
        gk_position=gk_pos,
        gk_effective_top_speed_mps=7.0,
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        pitch=pitch,
        team=Team.LEFT,
        gravity_mps2=9.81,
        params=params,
    )
    assert result is None


def test_early_intercept_boundary_exactly_at_max_distance():
    """Ball exactly at max_distance_m: result should be None (boundary is exclusive on the outside)."""
    pitch = Pitch.standard()
    params = _make_gk_params(early_intercept_max_distance_m=10.0)
    gk_pos = Vector3(-pitch.half_length + 1.0, 0, 0)
    # Ball at exactly 10m from GK
    ball_pos = Vector3(-pitch.half_length + 11.0, 0, 0.5)
    ball_vel = Vector3(-10.0, 0, 0)

    result = early_intercept_target(
        gk_position=gk_pos,
        gk_effective_top_speed_mps=7.0,
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        pitch=pitch,
        team=Team.LEFT,
        gravity_mps2=9.81,
        params=params,
    )
    # Exactly at the boundary or beyond → None
    assert result is None


def test_early_intercept_fast_far_shot_falls_back_to_none():
    """A powerful shot from just inside the distance gate, but so fast the
    GK cannot intercept it earlier than at the goal line → None."""
    pitch = Pitch.standard()
    params = _make_gk_params(early_intercept_max_distance_m=10.0, early_intercept_safety_margin=0.85)
    gk_pos = Vector3(-pitch.half_length + 1.0, 0, 0)
    # Ball 9m from GK (just inside gate), travelling at 30 m/s (very fast)
    ball_pos = Vector3(-pitch.half_length + 10.0, 0, 1.0)
    ball_vel = Vector3(-30.0, 0, -2.0)

    result = early_intercept_target(
        gk_position=gk_pos,
        gk_effective_top_speed_mps=7.0,
        ball_position=ball_pos,
        ball_velocity=ball_vel,
        pitch=pitch,
        team=Team.LEFT,
        gravity_mps2=9.81,
        params=params,
    )
    # Ball arrives so quickly the GK can't intercept earlier → None
    assert result is None


# --------------------------------------------------------------------------
# Phase C: control_time jump height penalty tests
# --------------------------------------------------------------------------

def test_control_time_gk_jump_zone_strictly_harder_than_head_height():
    """GK control time above head height (1.8m) must be strictly greater than
    control time at head height (jump zone penalty activating)."""
    params = ControlTimeParams.from_config()
    kwargs = dict(relative_speed_mps=0.0, player_speed_mps=0.0, ball_control_attr=0.5, is_goalkeeper_in_box=True)
    t_at_head = control_time_s(params, ball_height_m=params.player_height_m, **kwargs)
    t_above_head = control_time_s(params, ball_height_m=params.player_height_m + 0.2, **kwargs)
    assert t_above_head > t_at_head


def test_control_time_outfield_jump_zone_strictly_harder():
    """Outfield control time above head height (1.8m) must be strictly greater
    than at head height, and steeper than GK at the same height."""
    params = ControlTimeParams.from_config()
    h_above = params.player_height_m + 0.1
    t_gk = control_time_s(params, ball_height_m=h_above, relative_speed_mps=0.0,
                           player_speed_mps=0.0, ball_control_attr=0.5, is_goalkeeper_in_box=True)
    t_outfield = control_time_s(params, ball_height_m=h_above, relative_speed_mps=0.0,
                                player_speed_mps=0.0, ball_control_attr=0.5, is_goalkeeper_in_box=False)
    t_at_head = control_time_s(params, ball_height_m=params.player_height_m, relative_speed_mps=0.0,
                               player_speed_mps=0.0, ball_control_attr=0.5, is_goalkeeper_in_box=False)
    # Jump zone is harder than at head height
    assert t_outfield > t_at_head
    # GK is easier than outfield at same height above 1.8m (GK jump advantage)
    assert t_gk < t_outfield


def test_control_time_outfield_below_head_height_unchanged():
    """Outfield control time at/below head height must be byte-for-byte
    identical to the pre-phase-C height_difficulty_factor output (pure regression)."""
    params = ControlTimeParams.from_config()
    for h in [0.0, 0.49, 0.95, 1.8]:
        t = control_time_s(params, ball_height_m=h, relative_speed_mps=0.0,
                           player_speed_mps=0.0, ball_control_attr=0.5, is_goalkeeper_in_box=False)
        # Must be strictly positive (no degenerate 0)
        assert t > 0.0
    # Below-head heights must be strictly less than above-head height
    t_below = control_time_s(params, ball_height_m=1.8, relative_speed_mps=0.0,
                             player_speed_mps=0.0, ball_control_attr=0.5, is_goalkeeper_in_box=False)
    t_above = control_time_s(params, ball_height_m=2.0, relative_speed_mps=0.0,
                             player_speed_mps=0.0, ball_control_attr=0.5, is_goalkeeper_in_box=False)
    assert t_above > t_below


def test_control_time_height_tables_monotone():
    """Both GK and outfield control-time curves are monotonically increasing
    with ball height, and the jump zone creates a steeper slope above 1.8m."""
    params = ControlTimeParams.from_config()
    gk_heights = [0.0, 0.49, 0.95, 1.8, 2.0, 2.2]
    outfield_heights = [0.0, 0.49, 0.95, 1.8, 1.9, 2.0]

    gk_times = [
        control_time_s(params, h, 0.0, 0.0, 0.5, is_goalkeeper_in_box=True)
        for h in gk_heights
    ]
    outfield_times = [
        control_time_s(params, h, 0.0, 0.0, 0.5, is_goalkeeper_in_box=False)
        for h in outfield_heights
    ]

    # Both must be strictly monotone increasing
    for i in range(1, len(gk_times)):
        assert gk_times[i] >= gk_times[i - 1], f"GK curve non-monotone at h={gk_heights[i]}"
    for i in range(1, len(outfield_times)):
        assert outfield_times[i] >= outfield_times[i - 1], f"Outfield curve non-monotone at h={outfield_heights[i]}"

    # GK time must be lower than outfield time at every shared height above 1.8m
    for h in [1.9, 2.0]:
        t_gk = control_time_s(params, h, 0.0, 0.0, 0.5, is_goalkeeper_in_box=True)
        t_of = control_time_s(params, h, 0.0, 0.0, 0.5, is_goalkeeper_in_box=False)
        assert t_gk < t_of, f"GK not easier than outfield at h={h}m: gk={t_gk:.4f} outfield={t_of:.4f}"
