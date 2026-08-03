"""Ball physics: gravity, drag, Magnus effect, ground bounce, and rolling
friction. See engine/knowledge.md for the full derivation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3

# Pitch imported lazily inside resolve_goal_boundary to avoid a circular
# dependency at module load time (entities.pitch → config → nothing that
# imports ball_physics, but keeping it lazy is defensive).
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from footballcoach.entities.pitch import Pitch


@dataclass(frozen=True)
class BallPhysicsParams:
    gravity_mps2: float
    air_density_kgpm3: float
    ball_radius_m: float
    ball_mass_kg: float
    drag_coefficient: float
    magnus_coefficient: float
    bounce_restitution_vertical: float
    bounce_restitution_horizontal: float
    bounce_spin_retention: float
    rolling_friction_coefficient: float
    spin_decay_per_s: float
    block_restitution: float
    goal_net_restitution: float = 0.15
    just_bounced_display_duration_s: float = 0.3
    bounce_threshold_mps: float = 0.5  # min incoming vertical speed to count as a real bounce vs. rest/rolling

    @staticmethod
    def from_config() -> "BallPhysicsParams":
        cfg = load_physics_config()
        world = require_section(cfg, "world")
        ball = require_section(cfg, "ball")
        bp = require_section(cfg, "ball_physics")
        return BallPhysicsParams(
            gravity_mps2=world["gravity_mps2"],
            air_density_kgpm3=world["air_density_kgpm3"],
            ball_radius_m=ball["radius_m"],
            ball_mass_kg=ball["mass_kg"],
            drag_coefficient=bp["drag_coefficient"],
            magnus_coefficient=bp["magnus_coefficient"],
            bounce_restitution_vertical=bp["bounce_restitution_vertical"],
            bounce_restitution_horizontal=bp["bounce_restitution_horizontal"],
            bounce_spin_retention=bp["bounce_spin_retention"],
            rolling_friction_coefficient=bp["rolling_friction_coefficient"],
            spin_decay_per_s=bp["spin_decay_per_s"],
            block_restitution=bp["block_restitution"],
            goal_net_restitution=bp.get("goal_net_restitution", 0.15),
            just_bounced_display_duration_s=bp.get("just_bounced_display_duration_s", 0.3),
            bounce_threshold_mps=bp.get("bounce_threshold_mps", 0.5),
        )


def _cross_section_area(radius_m: float) -> float:
    return math.pi * radius_m * radius_m


def step_ball(ball: Ball, dt_s: float, params: BallPhysicsParams | None = None) -> None:
    """Advances the ball's position/velocity/spin by one physics tick.

    Only applies free-flight physics (gravity, drag, Magnus, ground
    bounce/roll). Does not handle possession "stuck to player" logic or
    kicks - those are handled by possession.py / kicking.py which set
    ball.velocity directly before/after calling this.
    """
    params = params or BallPhysicsParams.from_config()

    if ball.possessed_by is not None:
        return  # ball is glued to a player; possession.py handles that motion.

    area = _cross_section_area(params.ball_radius_m)

    # Gravity.
    gravity_force = Vector3(0.0, 0.0, -params.gravity_mps2 * params.ball_mass_kg)

    # Aerodynamic drag: F = -0.5 * rho * Cd * A * |v| * v
    speed = ball.velocity.length()
    if speed > 1e-9:
        drag_mag = 0.5 * params.air_density_kgpm3 * params.drag_coefficient * area * speed
        drag_force = ball.velocity.normalized() * (-drag_mag * speed)
    else:
        drag_force = Vector3.zero()

    # Magnus effect: F = rho * A * r * C_L * (omega x v)
    magnus_force = ball.spin.cross(ball.velocity) * (
        params.air_density_kgpm3 * area * params.ball_radius_m * params.magnus_coefficient
    )

    total_force = gravity_force + drag_force + magnus_force
    acceleration = total_force / params.ball_mass_kg

    new_velocity = ball.velocity + acceleration * dt_s
    new_position = ball.position + ball.velocity * dt_s + acceleration * (0.5 * dt_s * dt_s)

    # Spin decays exponentially over time (air resistance on rotation).
    spin_decay_factor = max(0.0, 1.0 - params.spin_decay_per_s * dt_s)
    new_spin = ball.spin * spin_decay_factor

    # Ground collision / bounce.
    #
    # IMPORTANT: only treat this as a genuine "bounce" (and apply
    # restitution, which scales horizontal velocity down too) if the
    # incoming vertical speed exceeds BOUNCE_THRESHOLD_MPS. Without this
    # threshold, a resting/rolling ball on the ground has gravity pull its
    # *next* position slightly below ball_radius_m every single tick, giving
    # new_velocity.z a small negative value purely from that one tick's
    # gravity integration - which used to be misread as a full bounce and
    # scaled horizontal velocity by bounce_restitution_horizontal (0.8)
    # EVERY tick (~30x/second), decaying speed almost instantly instead of
    # via the much gentler rolling_friction_coefficient below. This was a
    # real bug (see engine/knowledge.md's "ground contact" note) that made
    # slow, grounded balls (e.g. a gentle pass) stop dead within a few
    # ticks.
    real_bounce_this_tick = False
    if new_position.z <= params.ball_radius_m:
        new_position = new_position.with_z(params.ball_radius_m)
        if new_velocity.z < -params.bounce_threshold_mps:
            outgoing_vz = -new_velocity.z * params.bounce_restitution_vertical
            if outgoing_vz < params.bounce_threshold_mps:
                # The bounce would produce a smaller upward vz than the threshold.
                # Continuing to bounce would create a perpetual micro-bounce loop
                # (the ball never settles because restitution keeps it airborne by
                # a tiny amount each tick). Treat as grounded instead: zero vz and
                # let rolling friction take over.
                new_velocity = new_velocity.with_z(0.0)
            else:
                new_velocity = Vector3(
                    new_velocity.x * params.bounce_restitution_horizontal,
                    new_velocity.y * params.bounce_restitution_horizontal,
                    outgoing_vz,
                )
                new_spin = new_spin * params.bounce_spin_retention
                real_bounce_this_tick = True
        else:
            # Resting/rolling contact, not a real bounce: kill only the
            # (small, spurious) vertical velocity and leave horizontal
            # speed untouched here - rolling friction (below) is solely
            # responsible for decelerating a grounded ball.
            new_velocity = new_velocity.with_z(0.0)

        # Rolling friction always applies while in ground contact.
        horiz_speed = new_velocity.length_xy()
        if horiz_speed > 1e-9:
            friction_decel = params.rolling_friction_coefficient * params.gravity_mps2
            reduced_speed = max(0.0, horiz_speed - friction_decel * dt_s)
            scale = reduced_speed / horiz_speed
            new_velocity = Vector3(new_velocity.x * scale, new_velocity.y * scale, new_velocity.z)

    ball.position = new_position
    ball.velocity = new_velocity
    ball.spin = new_spin

    # Update the visual "just bounced" timer: set on a real bounce, decay each tick.
    if real_bounce_this_tick:
        ball.just_bounced_timer_s = params.just_bounced_display_duration_s
    elif ball.just_bounced_timer_s > 0.0:
        ball.just_bounced_timer_s = max(0.0, ball.just_bounced_timer_s - dt_s)

def _reflect_axis(ball: Ball, axis: str, boundary: float, net_e: float, moving_towards_boundary: bool) -> None:
    """Clamp `ball.position[axis]` to `boundary` and negate+damp the matching
    velocity component (scaling the other two components by `net_e` too),
    but only if the ball is actually moving towards the boundary.

    Shared by `resolve_goal_boundary`'s back-wall/side-post/crossbar checks,
    which otherwise repeat this "clamp position, negate velocity component,
    apply restitution" pattern three times.
    """
    if not moving_towards_boundary:
        return
    pos = {"x": ball.position.x, "y": ball.position.y, "z": ball.position.z}
    vel = {"x": ball.velocity.x, "y": ball.velocity.y, "z": ball.velocity.z}
    pos[axis] = boundary
    vel[axis] = -vel[axis] * net_e
    for other in ("x", "y", "z"):
        if other != axis:
            vel[other] *= net_e
    ball.position = Vector3(pos["x"], pos["y"], pos["z"])
    ball.velocity = Vector3(vel["x"], vel["y"], vel["z"])


def resolve_goal_boundary(ball: Ball, pitch: "Pitch", params: BallPhysicsParams) -> None:
    """Bounces the ball off the interior surfaces of whichever goal it has
    entered (back wall, side posts, crossbar).  Ground collisions inside the
    goal are already handled by step_ball's normal ground-contact logic.

    Call this after step_ball on every tick for loose balls.  Does nothing if
    the ball is possessed or has not passed the goal line.
    """
    if ball.possessed_by is not None:
        return

    r = params.ball_radius_m
    half_goal_w = pitch.goal_width_m / 2.0

    in_left = ball.position.x < -pitch.half_length
    in_right = ball.position.x > pitch.half_length
    if not (in_left or in_right):
        return

    # Only apply to balls within the goal mouth (with a small margin for the
    # ball radius).
    if abs(ball.position.y) > half_goal_w + r or ball.position.z > pitch.goal_height_m + r:
        return

    net_e = params.goal_net_restitution

    if in_left:
        back_wall_x = -(pitch.half_length + pitch.goal_depth_m)
        _reflect_axis(ball, "x", back_wall_x + r, net_e, ball.position.x <= back_wall_x + r and ball.velocity.x < 0.0)
    else:
        back_wall_x = pitch.half_length + pitch.goal_depth_m
        _reflect_axis(ball, "x", back_wall_x - r, net_e, ball.position.x >= back_wall_x - r and ball.velocity.x > 0.0)

    # Side posts (y): clamp and reflect.
    _reflect_axis(ball, "y", half_goal_w - r, net_e, ball.position.y > half_goal_w - r and ball.velocity.y > 0.0)
    _reflect_axis(ball, "y", -(half_goal_w - r), net_e, ball.position.y < -(half_goal_w - r) and ball.velocity.y < 0.0)

    # Crossbar (z): clamp and reflect.
    _reflect_axis(ball, "z", pitch.goal_height_m - r, net_e, ball.position.z > pitch.goal_height_m - r and ball.velocity.z > 0.0)