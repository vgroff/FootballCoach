"""Goalkeeper "Save" mechanics: predicting where an in-flight ball will
cross the keeper's own goal line and moving there. See engine/knowledge.md
for the full derivation, the tick-ordering subtlety this has to work around,
and how "good vs bad" keepers are differentiated (movement attributes get
them to the right spot in time; ball_control + the existing GK-in-box
control-time bonus determines whether they actually hold onto it once there).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from footballcoach.config import load_physics_config
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3


@dataclass(frozen=True)
class GoalkeepingParams:
    goal_frame_margin_m: float
    default_position_fraction_of_half_length: float

    @staticmethod
    def from_config() -> "GoalkeepingParams":
        d = load_physics_config()["goalkeeping"]
        return GoalkeepingParams(
            goal_frame_margin_m=d["goal_frame_margin_m"],
            default_position_fraction_of_half_length=d["default_position_fraction_of_half_length"],
        )


def own_goal_x(pitch: Pitch, team: Team) -> float:
    """The x-coordinate of the goal line `team` defends. By the same
    attacking-direction convention as engine/offside.py: Team.LEFT attacks
    towards +x and therefore defends the goal at -x; Team.RIGHT is the
    mirror image."""
    return -pitch.half_length if team == Team.LEFT else pitch.half_length


def predict_goal_line_crossing(
    ball_position: Vector3,
    ball_velocity: Vector3,
    target_x: float,
    gravity_mps2: float,
) -> Vector3 | None:
    """Predicts the (y, z) point at which the ball will cross the plane
    x = target_x, given its current position/velocity, under gravity alone
    (drag and Magnus are ignored for this prediction - the same
    simplification `kicking.solve_launch_pitch_rad` makes for aiming, i.e.
    the keeper "reads" the shot using straightforward physics judgement;
    heavily curved or backspun shots will be predicted slightly less
    accurately as a natural, tunable source of save difficulty).

    Returns None if the ball is not currently moving towards `target_x` (so
    there is no meaningful future crossing to react to) - e.g. it's
    stationary, or already moving away from that line.
    """
    vx = ball_velocity.x
    dx = target_x - ball_position.x

    # No meaningful x-motion, or moving away from target_x: no crossing.
    if abs(vx) < 1e-6 or dx * vx <= 0:
        return None

    t = dx / vx
    y = ball_position.y + ball_velocity.y * t
    z = ball_position.z + ball_velocity.z * t - 0.5 * gravity_mps2 * t * t
    return Vector3(target_x, y, max(0.0, z))


def save_target_position(
    pitch: Pitch,
    team: Team,
    ball_position: Vector3,
    ball_velocity: Vector3,
    gravity_mps2: float,
    params: GoalkeepingParams | None = None,
) -> Vector3:
    """Returns the world position a goalkeeper defending `team`'s goal
    should move towards this tick: the predicted crossing point of an
    incoming shot (if one is heading goalward), clamped to the goal frame
    plus a small margin, else a sensible default (goal centre, standing a
    short distance off the line rather than glued to it).

    The keeper's target plane is `goal_frame_margin_m` *in front of* the
    true goal line (not on it), so a keeper who arrives in time can
    intercept the ball before it actually crosses the line - if the target
    plane were the goal line itself, a fast shot can cross it in the same
    tick the keeper starts their pickup/control-time countdown (see
    engine/knowledge.md's tick-ordering note), turning every well-read save
    into a "too little, too late" concession.
    """
    params = params or GoalkeepingParams.from_config()
    goal_x = own_goal_x(pitch, team)
    sign = 1.0 if team == Team.LEFT else -1.0  # direction from goal line INTO the pitch
    target_plane_x = goal_x + sign * params.goal_frame_margin_m

    crossing = predict_goal_line_crossing(ball_position, ball_velocity, target_plane_x, gravity_mps2)

    half_goal_w = pitch.goal_width_m / 2.0
    if crossing is not None:
        clamped_y = max(-half_goal_w, min(half_goal_w, crossing.y))
        clamped_z = max(0.0, min(pitch.goal_height_m, crossing.z))
        return Vector3(target_plane_x, clamped_y, clamped_z)

    default_x = goal_x + sign * params.default_position_fraction_of_half_length * pitch.half_length
    return Vector3(default_x, 0.0, 0.0)
