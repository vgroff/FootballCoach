"""Kick trajectory simulation for the multi-phase kick UI preview.

Computes a forward-simulated ball path (deterministic, zero random noise)
given kick parameters.  Returns world-space 3D points that the renderer
colours based on height and vertical direction.

Spin convention (matches engine ball_physics Magnus force = spin × velocity):
  For a kick in the +x direction:
    spin.y > 0  →  topspin (ball curves downward)
    spin.y < 0  →  backspin (ball lifts)
    spin.z > 0  →  left sidespin (ball curves toward +y)
    spin.z < 0  →  right sidespin

The UI maps the mouse angle around the player (relative to the aim
direction) to a spin vector using the parametrisation:
    spin = magnitude * (cos(theta) * topspin_axis + sin(theta) * sidespin_axis)
where topspin_axis = 90°-CCW rotation of aim_dir in XY plane,
      sidespin_axis = (0, 0, 1).
This gives intuitive results: theta=0 (mouse ahead of player) = topspin,
theta=π (behind) = backspin, theta=±π/2 (left/right) = sidespin.
"""
from __future__ import annotations

import math

from footballcoach.engine.ball_physics import BallPhysicsParams, step_ball
from footballcoach.engine.kicking import (
    KickingParams,
    kick_sigma_rad,
    max_kick_speed_mps,
    running_power_multiplier,
    running_direction_precision_multiplier,
)
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3

# Simulation step: match the engine tick rate (30 Hz), but sub-step for
# smoother trajectory preview.
_TRAJ_DT_S = 1.0 / 60.0


def compute_speed_mps(
    kick_power_attr: float,
    power_fraction: float,
    player_velocity: Vector3 = Vector3.zero(),
    player_top_speed_mps: float = 0.0,
    aim_dir_x: float = 1.0,
    aim_dir_y: float = 0.0,
) -> float:
    """Launch speed accounting for running direction bonus/penalty.

    Mirrors the engine's kick_ball logic: sprinting toward the aim direction
    adds up to running_power_coefficient extra speed; running against it reduces
    it proportionally.
    """
    params = KickingParams.from_config()
    aim_dir = Vector3(aim_dir_x, aim_dir_y, 0.0)
    run_mult = running_power_multiplier(
        params.running_power_coefficient, player_velocity, aim_dir, player_top_speed_mps
    )
    return power_fraction * max_kick_speed_mps(params, kick_power_attr) * run_mult


def compute_launch_velocity(
    aim_dir_x: float,
    aim_dir_y: float,
    elevation_angle_rad: float,
    speed_mps: float,
) -> Vector3:
    """Compute launch velocity directly from aim direction, elevation angle, and speed.

    Uses elevation_angle_rad directly (not solve_launch_pitch_rad) so the
    visualised trajectory exactly matches the user's elevation input.  Phase 1
    (elevation=0) always produces a flat/rolling kick; Phase 2 lifts the ball
    at exactly the dialled angle.  solve_launch_pitch_rad is only used by the
    actual kick engine (which works from an aim_point); the preview trajectory
    uses this simpler, more predictable computation.
    """
    h_speed = speed_mps * math.cos(elevation_angle_rad)
    v_speed = speed_mps * math.sin(elevation_angle_rad)
    return Vector3(aim_dir_x * h_speed, aim_dir_y * h_speed, v_speed)


def simulate_trajectory(
    start_pos: Vector3,
    launch_velocity: Vector3,
    spin: Vector3,
    duration_s: float,
) -> list[Vector3]:
    """Forward-integrate ball physics from *start_pos* and return trajectory points.

    Stops early if the ball falls well below ground or slows to near-rest.
    """
    params = BallPhysicsParams.from_config()
    ball = Ball(
        position=start_pos,
        velocity=launch_velocity,
        spin=spin,
        radius_m=params.ball_radius_m,
        mass_kg=params.ball_mass_kg,
    )
    ball.possessed_by = None

    points: list[Vector3] = [start_pos]
    t = 0.0
    while t < duration_s:
        step_ball(ball, _TRAJ_DT_S, params)
        t += _TRAJ_DT_S
        points.append(Vector3(ball.position.x, ball.position.y, ball.position.z))
        # Stop well below ground
        if ball.position.z < -1.0:
            break
        # Stop if ball is basically at rest on the ground
        if ball.position.z <= 0.05 and ball.velocity.length() < 0.3:
            break
    return points


def build_cone_boundaries(
    player_pos: Vector3,
    aim_dir_x: float,
    aim_dir_y: float,
    elevation_angle_rad: float,
    speed_mps: float,
    sigma_rad: float,
    spin: Vector3,
    duration_s: float,
) -> tuple[list[Vector3], list[Vector3]]:
    """Return (+sigma, -sigma) yaw-offset trajectories for the 1-sigma XY error cone.

    Only yaw is perturbed (XY cone only, per design spec).  Spin is included so
    the cone correctly reflects Magnus-effect curvature at the 1-sigma boundaries.
    """
    launch_z = max(player_pos.z, 0.11)
    launch_pos = player_pos.with_z(launch_z)
    base_yaw = math.atan2(aim_dir_y, aim_dir_x)
    h_speed = speed_mps * math.cos(elevation_angle_rad)
    v_speed = speed_mps * math.sin(elevation_angle_rad)

    result = []
    for yaw_offset in (sigma_rad, -sigma_rad):
        yaw = base_yaw + yaw_offset
        vel = Vector3(
            math.cos(yaw) * h_speed,
            math.sin(yaw) * h_speed,
            v_speed,
        )
        result.append(simulate_trajectory(launch_pos, vel, spin, duration_s))
    return result[0], result[1]


def compute_error_sigma(
    kick_precision: float,
    power_fraction: float,
    player_velocity: Vector3 = Vector3.zero(),
    aim_dir_x: float = 1.0,
    aim_dir_y: float = 0.0,
) -> float:
    """1-sigma angular error in radians accounting for running direction penalty.

    Kicking against the running direction reduces effective precision, widening
    the cone.  Mirrors running_direction_precision_multiplier in the engine.
    """
    params = KickingParams.from_config()
    aim_dir = Vector3(aim_dir_x, aim_dir_y, 0.0)
    dir_precision_mult = running_direction_precision_multiplier(player_velocity, aim_dir, params)
    effective_precision = kick_precision * dir_precision_mult
    return kick_sigma_rad(params, effective_precision, power_fraction, rng_reduction=0.0)


def spin_from_mouse(
    aim_dir_x: float,
    aim_dir_y: float,
    mouse_dx_world: float,
    mouse_dy_world: float,
    max_spin_magnitude_rads: float,
) -> Vector3:
    """Convert mouse position relative to player into a spin vector.

    *mouse_dx_world*, *mouse_dy_world*: mouse offset from player in world metres.
    Returns a spin Vector3 using the convention described in the module docstring.
    """
    dist = math.hypot(mouse_dx_world, mouse_dy_world)
    if dist < 1e-6:
        return Vector3.zero()

    # Scale magnitude: capped at MAX_KICK_DRAG_M for full spin
    from footballcoach.ui.input import MAX_KICK_DRAG_M  # avoid circular at module level
    magnitude = min(1.0, dist / MAX_KICK_DRAG_M) * max_spin_magnitude_rads

    # Angle of mouse relative to aim direction
    mouse_angle = math.atan2(mouse_dy_world, mouse_dx_world)
    aim_angle = math.atan2(aim_dir_y, aim_dir_x)
    theta = mouse_angle - aim_angle  # 0 = ahead, π = behind

    # Topspin axis: 90° CCW rotation of aim_dir in XY plane
    topspin_x = -aim_dir_y
    topspin_y = aim_dir_x

    # spin = magnitude * (cos(theta) * topspin_axis + sin(theta) * sidespin_axis)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return Vector3(
        cos_t * topspin_x * magnitude,
        cos_t * topspin_y * magnitude,
        sin_t * magnitude,
    )
