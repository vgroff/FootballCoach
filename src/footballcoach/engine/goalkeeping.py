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
    early_intercept_max_distance_m: float  # max GK-to-ball distance for early intercept to activate
    early_intercept_safety_margin: float   # (legacy, unused in new logic) kept for config back-compat
    goal_line_save_weight: float           # multiplier on goal-line margin; >1 prefers goal-line saves

    @staticmethod
    def from_config() -> "GoalkeepingParams":
        d = load_physics_config()["goalkeeping"]
        return GoalkeepingParams(
            goal_frame_margin_m=d["goal_frame_margin_m"],
            default_position_fraction_of_half_length=d["default_position_fraction_of_half_length"],
            early_intercept_max_distance_m=d.get("early_intercept_max_distance_m", 10.0),
            early_intercept_safety_margin=d.get("early_intercept_safety_margin", 0.85),
            goal_line_save_weight=d.get("goal_line_save_weight", 1.0),
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


def early_intercept_target(
    gk_position: Vector3,
    gk_effective_top_speed_mps: float,
    ball_position: Vector3,
    ball_velocity: Vector3,
    pitch: Pitch,
    team: Team,
    gravity_mps2: float,
    params: GoalkeepingParams | None = None,
) -> Vector3 | None:
    """Chooses between an early intercept and a goal-line save by comparing
    the time margin the GK has for each option, picking whichever gives the
    larger safety buffer, weighted by ``goal_line_save_weight``.

    For each option the margin is defined as:
        margin = t_ball_arrives - t_gk_arrives

    A positive margin means the GK gets there before the ball.  Intercept
    is only considered if the ball is within ``early_intercept_max_distance_m``
    and both options are viable (positive margin). The goal-line margin is
    multiplied by ``goal_line_save_weight`` before comparison — at the default
    of 1.0 the choice is purely based on which margin is larger; values > 1.0
    bias toward goal-line saves, values < 1.0 bias toward early intercepts.

    Returns the chosen target Vector3, or None if no shot is incoming or
    neither option is viable (caller falls back to save_target_position's
    default positioning).
    """
    params = params or GoalkeepingParams.from_config()
    goal_x = own_goal_x(pitch, team)
    sign = 1.0 if team == Team.LEFT else -1.0
    target_plane_x = goal_x + sign * params.goal_frame_margin_m
    gk_speed = max(gk_effective_top_speed_mps, 0.1)
    half_goal_w = pitch.goal_width_m / 2.0

    # --- Goal-line save option ---
    crossing = predict_goal_line_crossing(ball_position, ball_velocity, target_plane_x, gravity_mps2)
    if crossing is None:
        return None  # ball not heading toward goal at all

    # Time for ball to reach goal-line plane.
    dx_ball_to_goal = abs(target_plane_x - ball_position.x)
    vx = abs(ball_velocity.x)
    t_ball_to_goal_line = dx_ball_to_goal / max(vx, 0.1)

    # Time for GK to reach goal-line crossing position.
    gk_to_goal_line = gk_position.xy().distance_to(crossing.xy())
    t_gk_to_goal_line = gk_to_goal_line / gk_speed

    goal_line_margin = t_ball_to_goal_line - t_gk_to_goal_line

    # --- Early intercept option ---
    # Only evaluate if ball is within the activation radius.
    gk_to_ball = ball_position.xy().distance_to(gk_position.xy())
    intercept_target: Vector3 | None = None
    intercept_margin = -math.inf

    if gk_to_ball <= params.early_intercept_max_distance_m:
        ball_speed_xy = ball_velocity.length_xy()
        # Estimate where the ball will be when the GK could reach it (one
        # linear step capped at 2 s, same approach as Match._intercept_target).
        t_estimate = min(gk_to_ball / gk_speed, 2.0)
        predicted_xy = ball_position + ball_velocity.xy() * t_estimate
        t_ball_at_intercept = (
            (predicted_xy.xy() - ball_position.xy()).length() / max(ball_speed_xy, 0.1)
            if ball_speed_xy > 0.1 else t_estimate
        )
        z_at_intercept = max(0.0,
            ball_position.z + ball_velocity.z * t_ball_at_intercept
            - 0.5 * gravity_mps2 * t_ball_at_intercept ** 2
        )
        candidate = predicted_xy.with_z(z_at_intercept)

        # Discard intercept candidates that are behind/on the goal line —
        # the ball would already have scored before the GK could get there.
        in_front_of_goal = (
            candidate.x > target_plane_x if team == Team.LEFT
            else candidate.x < target_plane_x
        )
        if in_front_of_goal:
            gk_to_candidate = gk_position.xy().distance_to(candidate.xy())
            t_gk_to_intercept = gk_to_candidate / gk_speed

            intercept_margin = t_ball_at_intercept - t_gk_to_intercept

            clamped_y = max(-half_goal_w, min(half_goal_w, candidate.y))
            clamped_z = max(0.0, min(pitch.goal_height_m, candidate.z))
            intercept_target = Vector3(candidate.x, clamped_y, clamped_z)

    # --- Pick the better option ---
    # Weight the goal-line margin so values > 1 bias toward staying on the line.
    weighted_goal_line_margin = goal_line_margin * params.goal_line_save_weight

    if intercept_margin > weighted_goal_line_margin and intercept_margin > 0.0:
        return intercept_target

    # Fall back to goal-line save if viable, else return None (caller uses
    # save_target_position's default positioning).
    if goal_line_margin > 0.0:
        clamped_y = max(-half_goal_w, min(half_goal_w, crossing.y))
        clamped_z = max(0.0, min(pitch.goal_height_m, crossing.z))
        return Vector3(target_plane_x, clamped_y, clamped_z)

    return None


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
