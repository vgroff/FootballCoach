"""Kicking mechanics: power, direction, and spin, all with precision-scaled
Gaussian error. See engine/knowledge.md for the derivation of the error model
and how it was validated against the penalty-scoring balance targets.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from footballcoach.config import load_physics_config
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3
from footballcoach.mathutils.rng import reduced_sigma


@dataclass(frozen=True)
class KickingParams:
    angle_error_base_rad: float
    angle_error_scale_rad: float
    power_base_mps: float
    power_scale_mps: float
    firsttime_precision_weight: float
    running_power_coefficient: float

    @staticmethod
    def from_config() -> "KickingParams":
        d = load_physics_config()["kicking"]
        return KickingParams(
            angle_error_base_rad=d["angle_error_base_rad"],
            angle_error_scale_rad=d["angle_error_scale_rad"],
            power_base_mps=d["power_base_mps"],
            power_scale_mps=d["power_scale_mps"],
            firsttime_precision_weight=d["firsttime_precision_weight"],
            running_power_coefficient=d["running_power_coefficient"],
        )


@dataclass(frozen=True)
class PassingParams:
    """Constants for the dedicated "Pass" action (see `pass_ball`) - a
    grounded pass to a target point is a different technical skill to
    curling a shot into a corner, so it gets its own (more forgiving)
    angular error model and an auto-computed pace, rather than reusing
    `KickingParams` directly. See engine/knowledge.md for the derivation.
    """
    angle_error_base_rad: float
    angle_error_scale_rad: float
    power_overshoot_factor: float
    min_speed_mps: float
    max_speed_mps: float

    @staticmethod
    def from_config() -> "PassingParams":
        d = load_physics_config()["passing"]
        return PassingParams(
            angle_error_base_rad=d["angle_error_base_rad"],
            angle_error_scale_rad=d["angle_error_scale_rad"],
            power_overshoot_factor=d["power_overshoot_factor"],
            min_speed_mps=d["min_speed_mps"],
            max_speed_mps=d["max_speed_mps"],
        )


def angle_error_sigma_rad(params: KickingParams, kick_precision: float) -> float:
    """Angular error std-dev (radians), applied independently to yaw and
    pitch. Never zero even at precision=1.0, per the design spec that kicks
    are never perfectly exact."""
    return params.angle_error_base_rad + params.angle_error_scale_rad * (1.0 - kick_precision)


def pass_angle_error_sigma_rad(params: PassingParams, kick_precision: float) -> float:
    """Same shape as `angle_error_sigma_rad` but using the more forgiving
    passing-specific constants."""
    return params.angle_error_base_rad + params.angle_error_scale_rad * (1.0 - kick_precision)


def max_kick_speed_mps(params: KickingParams, kick_power_attr: float) -> float:
    return params.power_base_mps + params.power_scale_mps * kick_power_attr


def running_power_multiplier(
    running_power_coefficient: float,
    kicker_velocity: Vector3,
    aim_direction: Vector3,
    kicker_top_speed_mps: float,
) -> float:
    """Multiplier applied to kick/pass launch speed to model "running onto
    the ball": running towards the aim direction adds power (up to
    `running_power_coefficient` extra at a full sprint dead in line with the
    shot), running square-on has no effect, and running backwards away from
    the aim direction *reduces* power correspondingly - a simple cosine
    projection of the kicker's velocity onto the aim direction, scaled by how
    close to top speed they're currently running.

    Returns 1.0 (no effect) if the kicker isn't moving or has negligible top
    speed (e.g. callers that don't pass movement context at all). Shared by
    both `kick_ball` and `pass_ball` (each passes its own params' coefficient
    - currently only `KickingParams.running_power_coefficient` is used by
    default, but the function itself is params-agnostic).
    """
    run_speed = kicker_velocity.length_xy()
    if run_speed < 1e-6 or kicker_top_speed_mps < 1e-6 or aim_direction.length_xy() < 1e-9:
        return 1.0
    cos_angle = kicker_velocity.xy().normalized().dot(aim_direction.xy().normalized())
    run_speed_fraction = min(1.0, run_speed / kicker_top_speed_mps)
    return 1.0 + running_power_coefficient * cos_angle * run_speed_fraction


def pass_speed_mps(params: PassingParams, distance_m: float, gravity_mps2: float, rolling_friction_coefficient: float) -> float:
    """Auto-computes a sensible pace for a grounded pass to travel
    `distance_m`, so the passer (or a "Pass" action helper) doesn't need to
    manually pick a power fraction.

    Modelled as a rolling ball decelerating under (dominant) rolling
    friction from an initial speed: v = sqrt(2 * mu_roll * g * distance).
    This ignores the smaller, speed-dependent aerodynamic drag contribution
    that the full engine physics (see ball_physics.py) also applies, so a
    tunable `power_overshoot_factor` compensates for the resulting
    undershoot - see engine/knowledge.md for how this was calibrated against
    the pass-accuracy balance tests.
    """
    base_speed = math.sqrt(max(0.0, 2.0 * rolling_friction_coefficient * gravity_mps2 * distance_m))
    speed = base_speed * params.power_overshoot_factor
    return max(params.min_speed_mps, min(params.max_speed_mps, speed))


def firsttime_difficulty_multiplier(
    params: KickingParams, kick_precision: float, difficulty: float
) -> float:
    """Difficulty multiplier applied to angle_error_sigma for a first-time
    shot/pass taken directly off a difficult ball (see possession.py for how
    `difficulty` is computed from height/velocity)."""
    return 1.0 + (1.0 - params.firsttime_precision_weight * kick_precision) * difficulty


def solve_launch_pitch_rad(
    horizontal_distance_m: float,
    height_diff_m: float,
    speed_mps: float,
    gravity_mps2: float,
) -> float:
    """Solves for the launch pitch angle (radians above horizontal) needed
    for a projectile fired at `speed_mps` to pass through a point
    `horizontal_distance_m` away and `height_diff_m` above the launch point,
    given constant gravity (ignoring drag/Magnus for this aiming solve - the
    kicker "compensates" for those in the same way a real player does; drag
    and Magnus are still applied to the ball's actual flight afterwards).

    Standard projectile range equation, solved for tan(theta) as a quadratic:
        a*T^2 - dx*T + c = 0,  a = g*dx^2/(2v^2),  c = dz + a
    Of the two real roots, the flatter (smaller-angle) trajectory is chosen,
    matching how a real player would drive a ball rather than loop it,
    unless only the high-arc solution is reachable.

    Falls back to a direct straight-line angle if the target is out of range
    for the given speed (discriminant < 0), which only happens for
    unrealistically weak kicks relative to distance.
    """
    dx = horizontal_distance_m
    dz = height_diff_m

    if dx < 1e-6:
        return math.pi / 2 if dz > 0 else -math.pi / 2

    a = gravity_mps2 * dx * dx / (2.0 * speed_mps * speed_mps)
    b = -dx
    c = dz + a

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # Target unreachable at this speed/distance combo - fall back to a
        # direct straight-line aim (no gravity compensation possible).
        return math.atan2(dz, dx)

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    flatter_t = min(t1, t2)
    return math.atan(flatter_t)


def _launch_ball(
    ball: Ball,
    kicker_position: Vector3,
    aim_point: Vector3,
    speed: float,
    sigma: float,
    spin: Vector3,
    rng: random.Random,
    gravity_mps2: float,
) -> None:
    """Shared aim-then-launch logic used by both `kick_ball` and
    `pass_ball`: solves the ballistic angle to `aim_point` at the given
    `speed`, perturbs yaw/pitch by independent Gaussian noise of std-dev
    `sigma`, and releases the ball from possession with the resulting
    velocity."""
    launch_position = kicker_position.with_z(ball.position.z)
    delta = aim_point - launch_position
    horizontal_distance = delta.xy().length()
    yaw = delta.xy().angle_xy() if horizontal_distance > 1e-9 else 0.0

    pitch = solve_launch_pitch_rad(horizontal_distance, delta.z, speed, gravity_mps2)

    final_yaw = yaw + rng.gauss(0.0, sigma)
    final_pitch = pitch + rng.gauss(0.0, sigma)

    horizontal_speed = speed * math.cos(final_pitch)
    vertical_speed = speed * math.sin(final_pitch)

    velocity = Vector3(
        math.cos(final_yaw) * horizontal_speed,
        math.sin(final_yaw) * horizontal_speed,
        vertical_speed,
    )

    ball.possessed_by = None
    ball.position = launch_position
    ball.velocity = velocity
    ball.spin = spin


def kick_ball(
    ball: Ball,
    kicker_position: Vector3,
    aim_point: Vector3,
    power_fraction: float,
    kick_precision: float,
    kick_power_attr: float,
    spin: Vector3,
    rng_reduction: float,
    rng: random.Random | None = None,
    params: KickingParams | None = None,
    difficulty_multiplier: float = 1.0,
    gravity_mps2: float = 9.81,
    kicker_velocity: Vector3 = Vector3.zero(),
    kicker_top_speed_mps: float = 0.0,
) -> None:
    """Applies a kick to `ball`, releasing it from possession and giving it
    a velocity/spin aimed at `aim_point` (an absolute world position), with
    the launch angle solved ballistically (see solve_launch_pitch_rad) so
    the ball actually reaches the intended height at the intended distance
    under gravity. Gaussian error (scaled by kick_precision and
    rng_reduction) is then applied to both yaw and pitch.

    `power_fraction` in [0, 1] scales the kicker's max kick speed.
    `difficulty_multiplier` >= 1.0 inflates the error further for off-balance
    / first-time kicks (see kicking.firsttime_difficulty_multiplier).
    `kicker_velocity`/`kicker_top_speed_mps` (both optional, default to no
    effect) let a caller model "running onto the ball" - see
    `running_power_multiplier`: running towards the aim direction adds
    power, running away from it reduces power correspondingly.
    """
    params = params or KickingParams.from_config()
    r = rng or random

    sigma = reduced_sigma(angle_error_sigma_rad(params, kick_precision), rng_reduction) * difficulty_multiplier
    speed = max_kick_speed_mps(params, kick_power_attr) * max(0.0, min(1.0, power_fraction))
    speed *= running_power_multiplier(
        params.running_power_coefficient, kicker_velocity, aim_point - kicker_position, kicker_top_speed_mps
    )

    _launch_ball(ball, kicker_position, aim_point, speed, sigma, spin, r, gravity_mps2)


def pass_ball(
    ball: Ball,
    kicker_position: Vector3,
    target_position: Vector3,
    kick_precision: float,
    rng_reduction: float,
    rng: random.Random | None = None,
    params: PassingParams | None = None,
    gravity_mps2: float = 9.81,
    rolling_friction_coefficient: float = 0.06,
    power_fraction: float | None = None,
    max_speed_for_power: float | None = None,
    running_power_coefficient: float = 0.0,
    kicker_velocity: Vector3 = Vector3.zero(),
    kicker_top_speed_mps: float = 0.0,
) -> None:
    """Applies a grounded pass to `ball`, aimed at `target_position` (an
    absolute world position, height ignored - passes are aimed along the
    ground). Unlike `kick_ball`, the pace is auto-computed from distance
    (see `pass_speed_mps`) unless an explicit `power_fraction` is supplied,
    in which case it scales `max_speed_for_power` instead (letting a caller
    override the auto-pace with a manual power slider if desired, mirroring
    KickOrder's power_fraction).

    Uses the dedicated, more forgiving `PassingParams` error model rather
    than `KickingParams` - see engine/knowledge.md for the rationale.

    `running_power_coefficient`/`kicker_velocity`/`kicker_top_speed_mps`
    (all optional, default to no effect) apply the same "running onto the
    ball" power modifier as `kick_ball` - see `running_power_multiplier`.
    """
    params = params or PassingParams.from_config()
    r = rng or random

    aim_point = target_position.with_z(0.0)
    distance = kicker_position.xy().distance_to(aim_point.xy())

    if power_fraction is None:
        speed = pass_speed_mps(params, distance, gravity_mps2, rolling_friction_coefficient)
    else:
        max_speed = max_speed_for_power if max_speed_for_power is not None else params.max_speed_mps
        speed = max_speed * max(0.0, min(1.0, power_fraction))

    speed *= running_power_multiplier(
        running_power_coefficient, kicker_velocity, aim_point - kicker_position, kicker_top_speed_mps
    )

    sigma = reduced_sigma(pass_angle_error_sigma_rad(params, kick_precision), rng_reduction)

    _launch_ball(ball, kicker_position, aim_point, speed, sigma, Vector3.zero(), r, gravity_mps2)
