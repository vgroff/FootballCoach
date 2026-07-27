from __future__ import annotations

from footballcoach.engine.goalkeeping import (
    GoalkeepingParams,
    own_goal_x,
    predict_goal_line_crossing,
    save_target_position,
)
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
