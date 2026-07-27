"""Ball physics: gravity, drag, Magnus effect, ground bounce, and rolling
friction. See engine/knowledge.md for the full derivation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from footballcoach.config import load_physics_config
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3


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

    @staticmethod
    def from_config() -> "BallPhysicsParams":
        world = load_physics_config()["world"]
        ball = load_physics_config()["ball"]
        bp = load_physics_config()["ball_physics"]
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
    BOUNCE_THRESHOLD_MPS = 0.5
    if new_position.z <= params.ball_radius_m:
        new_position = new_position.with_z(params.ball_radius_m)
        if new_velocity.z < -BOUNCE_THRESHOLD_MPS:
            new_velocity = Vector3(
                new_velocity.x * params.bounce_restitution_horizontal,
                new_velocity.y * params.bounce_restitution_horizontal,
                -new_velocity.z * params.bounce_restitution_vertical,
            )
            new_spin = new_spin * params.bounce_spin_retention
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
