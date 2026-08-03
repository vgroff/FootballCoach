"""Kicking mechanics: power, direction, and spin, all with precision-scaled
Gaussian error. See engine/knowledge.md for the derivation of the error model
and how it was validated against the penalty-scoring balance targets.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3
from footballcoach.mathutils.interp import piecewise_lerp3

log = logging.getLogger("footballcoach.kicking")



@dataclass(frozen=True)
class KickingParams:
    angle_error_base_rad: float
    angle_error_scale_rad: float
    power_error_scale: float
    power_error_exponent: float
    precision_exponent: float
    power_base_mps: float
    power_scale_mps: float
    firsttime_precision_weight: float
    running_power_coefficient: float
    # Running-direction precision penalty breakpoints and magnitudes.
    # A kick aimed mostly in the same direction as the kicker is running incurs
    # no penalty; a kick aimed square-on or backward is penalised (up to -75%
    # precision at full backwards run). Only applied when kicker ground speed
    # exceeds running_direction_min_speed_mps.
    running_direction_precision_cos_high: float  # >= this → no penalty (default 0.35, ≈70° cone)
    running_direction_precision_cos_low: float   # < this → steep penalty zone (default -0.2, ≈102°)
    running_direction_precision_penalty_mid: float   # penalty at cos_low boundary (default 0.25 = -25%)
    running_direction_precision_penalty_max: float   # penalty at cos=-1 (default 0.75 = -75%)
    running_direction_min_speed_mps: float           # min kicker speed to apply penalty (default 1.0)

    @staticmethod
    def from_config() -> "KickingParams":
        d = require_section(load_physics_config(), "kicking")
        return KickingParams(
            angle_error_base_rad=d["angle_error_base_rad"],
            angle_error_scale_rad=d["angle_error_scale_rad"],
            power_error_scale=d["power_error_scale"],
            power_error_exponent=d["power_error_exponent"],
            precision_exponent=d["precision_exponent"],
            power_base_mps=d["power_base_mps"],
            power_scale_mps=d["power_scale_mps"],
            firsttime_precision_weight=d["firsttime_precision_weight"],
            running_power_coefficient=d["running_power_coefficient"],
            running_direction_precision_cos_high=d.get("running_direction_precision_cos_high", 0.35),
            running_direction_precision_cos_low=d.get("running_direction_precision_cos_low", -0.2),
            running_direction_precision_penalty_mid=d.get("running_direction_precision_penalty_mid", 0.25),
            running_direction_precision_penalty_max=d.get("running_direction_precision_penalty_max", 0.75),
            running_direction_min_speed_mps=d.get("running_direction_min_speed_mps", 1.0),
        )


@dataclass(frozen=True)
class PassingParams:
    """Auto-pace constants for ground passes (see `pass_ball`). The error model
    is now unified with `KickingParams` - all kicks use the same sigma formula
    regardless of whether they're shots, passes, or anything else. Passes tend
    to be lower power than shots, so they inherit naturally lower inaccuracy
    from the power-error coupling (see `kick_sigma_rad`).
    """
    power_overshoot_factor: float
    overshoot_drag_factor: float    # coefficient: total = base + drag_factor * distance^drag_exponent
    overshoot_drag_exponent: float  # exponent on distance; 1.0=linear, >1=superlinear boost at long range
    min_speed_mps: float
    max_speed_mps: float
    pass_aim_height_m: float  # height to aim at (and thus launch angle); 0.11=flat, higher=lofted

    @staticmethod
    def from_config() -> "PassingParams":
        d = require_section(load_physics_config(), "passing")
        return PassingParams(
            power_overshoot_factor=d["power_overshoot_factor"],
            overshoot_drag_factor=d.get("overshoot_drag_factor", 0.0),
            overshoot_drag_exponent=d.get("overshoot_drag_exponent", 1.0),
            min_speed_mps=d["min_speed_mps"],
            max_speed_mps=d["max_speed_mps"],
            pass_aim_height_m=d["pass_aim_height_m"],
        )


def angle_error_sigma_rad(params: KickingParams, kick_precision: float) -> float:
    """Base angular error std-dev before power coupling and rng_reduction.
    Never zero even at precision=1.0.
    ``precision_exponent`` < 1.0 compresses the precision curve: using
    ``1 - p^exponent`` reduces overall sensitivity to precision differences
    (lower sigma across the board) relative to ``(1-p)^exponent``."""
    return params.angle_error_base_rad + params.angle_error_scale_rad * (1.0 - kick_precision ** params.precision_exponent)


def kick_sigma_rad(
    params: KickingParams,
    kick_precision: float,
    effective_power: float,
    rng_reduction: float,
) -> float:
    """Unified angular error std-dev for all kicks (shots, passes, chips).

    sigma = base_sigma(precision) * (1 + power_error_scale * effective_power^exponent) * (1 - rng_reduction)

    ``effective_power`` is ``power_fraction * running_power_multiplier`` - it
    can exceed 1.0 when the kicker is sprinting toward the target at full
    pace (running_mult ≈ 1.3), producing proportionally more inaccuracy.

    ``power_error_exponent`` (default 2.0, tunable 1.0–2.5) controls the shape
    of the power-error curve. At exponent=2 the coupling is superlinear: a 25m
    auto-pass at eff_power≈0.20 gets barely any extra error (multiplier ≈ 1.01),
    while a running penalty at eff_power≈1.04 is nearly the same as linear
    (since 1.04^2 ≈ 1.04). The exponent therefore primarily buys accuracy at
    low-to-mid powers without touching the high-power penalty calibration.
    """
    base = angle_error_sigma_rad(params, kick_precision)
    return base * (1.0 + params.power_error_scale * (effective_power ** params.power_error_exponent)) * (1.0 - rng_reduction)


def max_kick_speed_mps(params: KickingParams, kick_power_attr: float) -> float:
    return params.power_base_mps + params.power_scale_mps * kick_power_attr


def _run_direction_geometry(kicker_velocity: Vector3, aim_direction: Vector3) -> tuple[float, float] | None:
    """Shared "how is the kicker running relative to the aim direction" geometry,
    used by both `running_power_multiplier` and
    `running_direction_precision_multiplier` (they're always called together).

    Returns ``(run_speed, cos_sim)`` where ``run_speed`` is the kicker's XY
    ground speed and ``cos_sim`` is the cosine similarity between the
    kicker's velocity direction and the aim direction, or ``None`` if either
    vector is degenerate (kicker stationary, or aim direction zero-length).
    """
    run_speed = kicker_velocity.length_xy()
    aim_len = aim_direction.length_xy()
    if run_speed < 1e-6 or aim_len < 1e-9:
        return None
    cos_sim = (kicker_velocity.xy() / run_speed).dot(aim_direction.xy() / aim_len)
    return run_speed, cos_sim


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
    geom = _run_direction_geometry(kicker_velocity, aim_direction)
    if geom is None or kicker_top_speed_mps < 1e-6:
        return 1.0
    run_speed, cos_angle = geom
    run_speed_fraction = min(1.0, run_speed / kicker_top_speed_mps)
    return 1.0 + running_power_coefficient * cos_angle * run_speed_fraction


def running_direction_precision_multiplier(
    kicker_velocity: Vector3,
    aim_direction: Vector3,
    params: KickingParams,
) -> float:
    """Precision multiplier (in [0.25, 1.0]) based on the angle between the
    kicker's running direction and the kick aim direction.

    Rationale: kicking in line with your momentum is biomechanically easiest
    (full precision); kicking square-on or backwards against your own momentum
    is significantly harder (reduced precision).

    Piecewise-linear in cos(angle):
      cos_sim >= cos_high (0.35)  → 1.0   (no penalty, ≈ 70° forward cone)
      cos_low (-0.2) <= cos_sim < cos_high → linear 1.0 → (1 - penalty_mid)
      cos_sim < cos_low           → linear (1 - penalty_mid) → (1 - penalty_max)

    Returns 1.0 (no effect) if the kicker's ground speed is below
    `running_direction_min_speed_mps`, or if either vector is degenerate.
    """
    geom = _run_direction_geometry(kicker_velocity, aim_direction)
    if geom is None:
        return 1.0
    run_speed, cos_sim = geom
    if run_speed < params.running_direction_min_speed_mps:
        return 1.0

    return piecewise_lerp3(
        cos_sim,
        x_low=-1.0,
        x_mid=params.running_direction_precision_cos_low,
        x_high=params.running_direction_precision_cos_high,
        y_low=1.0 - params.running_direction_precision_penalty_max,
        y_mid=1.0 - params.running_direction_precision_penalty_mid,
        y_high=1.0,
    )


def compensate_power_for_run_mult(power_fraction: float, run_mult: float) -> float:
    """Pre-compensates ``power_fraction`` so that ``adjusted * run_mult``
    delivers the originally-intended ``power_fraction``, up to the cap of 1.0.

    A player intending to kick at a given power naturally adjusts their effort
    based on running momentum: they ease off when sprinting in line with the
    kick (run_mult > 1) and try harder when running against it (run_mult < 1).
    The cap at 1.0 means a very unfavourable run direction still costs some
    pace, which is physically realistic.

    Called at the order-processing layer (match.py) for ShootOrder, KickOrder
    and PassOrder so all deliberate kicks deliver consistent intended power
    regardless of run direction, while the raw running-power physics in
    kick_ball/pass_ball remain unchanged.
    """
    return min(1.0, power_fraction / max(run_mult, 1e-6))


def pass_speed_mps(params: PassingParams, distance_m: float, gravity_mps2: float, rolling_friction_coefficient: float) -> float:
    """Auto-computes a sensible pace for a grounded pass to travel
    `distance_m`, so the passer (or a "Pass" action helper) doesn't need to
    manually pick a power fraction.

    Modelled as a rolling ball decelerating under (dominant) rolling
    friction from an initial speed: v = sqrt(2 * mu_roll * g * distance).
    Aerodynamic drag (ignored in the analytic model) removes more energy per
    metre from faster long passes, so the effective overshoot factor grows
    linearly with distance: overshoot = base + drag_factor * distance_m.
    This keeps short passes barely overshooting while long passes get enough
    extra pace to overcome drag - see engine/knowledge.md for calibration.
    """
    base_speed = math.sqrt(max(0.0, 2.0 * rolling_friction_coefficient * gravity_mps2 * distance_m))
    overshoot = params.power_overshoot_factor + params.overshoot_drag_factor * (distance_m ** params.overshoot_drag_exponent)
    speed = base_speed * overshoot
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


def _log_kick_debug(
    tag: str,
    kicker_velocity: Vector3,
    aim_dir: Vector3,
    run_mult: float,
    dir_precision_mult: float,
    kick_precision: float,
    effective_precision: float,
    effective_power: float,
    sigma: float,
    speed: float,
    rng_reduction: float,
    base_power_fraction: float | None = None,
) -> None:
    """Shared debug-log formatting for `kick_ball` and `pass_ball` (identical
    fields aside from `pass_ball`'s extra `base_power_fraction`)."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    kicker_speed_xy = kicker_velocity.length_xy()
    cos_sim = (
        (kicker_velocity.xy() / kicker_speed_xy).dot(aim_dir.xy().normalized())
        if kicker_speed_xy > 1e-6 and aim_dir.length_xy() > 1e-9
        else float('nan')
    )
    base_power_str = "" if base_power_fraction is None else f"base_power_frac={base_power_fraction:.3f} "
    log.debug(
        "[%s] kicker_vel=(%.2f,%.2f) kicker_speed=%.2f aim_dir=(%.2f,%.2f) "
        "cos_sim=%.3f run_mult=%.3f dir_precision_mult=%.3f "
        "kick_precision=%.3f -> effective_precision=%.3f "
        "%seffective_power=%.3f sigma=%.5f speed=%.2f rng_reduction=%.2f",
        tag,
        kicker_velocity.x, kicker_velocity.y, kicker_speed_xy,
        aim_dir.x, aim_dir.y,
        cos_sim, run_mult, dir_precision_mult,
        kick_precision, effective_precision,
        base_power_str,
        effective_power, sigma, speed, rng_reduction,
    )


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

    aim_dir = aim_point - kicker_position
    run_mult = running_power_multiplier(
        params.running_power_coefficient, kicker_velocity, aim_dir, kicker_top_speed_mps
    )
    dir_precision_mult = running_direction_precision_multiplier(kicker_velocity, aim_dir, params)
    effective_precision = kick_precision * dir_precision_mult
    effective_power = max(0.0, min(1.0, power_fraction)) * run_mult
    sigma = kick_sigma_rad(params, effective_precision, effective_power, rng_reduction) * difficulty_multiplier
    speed = max_kick_speed_mps(params, kick_power_attr) * max(0.0, min(1.0, power_fraction)) * run_mult

    _log_kick_debug(
        "kick_ball", kicker_velocity, aim_dir, run_mult, dir_precision_mult,
        kick_precision, effective_precision, effective_power, sigma, speed, rng_reduction,
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
    running_power_coefficient: float = 0.0,
    kicker_velocity: Vector3 = Vector3.zero(),
    kicker_top_speed_mps: float = 0.0,
    kick_power_attr: float = 0.5,
    kicking_params: "KickingParams | None" = None,
) -> None:
    """Applies a grounded pass to `ball`, aimed at `target_position`.
    Height component of target_position is ignored - passes travel along
    the ground.

    The pace is auto-computed from distance (see `pass_speed_mps`) unless
    ``power_fraction`` is supplied, in which case it scales
    ``params.max_speed_mps``.

    Error model: uses the same unified ``kick_sigma_rad`` as ``kick_ball``
    (see ``KickingParams``). The effective power for the sigma computation
    is derived from the actual launch speed as a fraction of the kicker's
    maximum kick speed, so a slow auto-pace short pass naturally yields a
    lower-sigma (more accurate) kick than a full-power shot without needing
    a separate error model.
    """
    params = params or PassingParams.from_config()
    kp = kicking_params or KickingParams.from_config()
    r = rng or random

    # Aim at pass_aim_height_m. At 0.11m (= ball radius) this gives pitch_angle=0
    # and a flat rolling pass. Higher values produce a slight loft, which can help
    # carries overcome friction on longer passes at the cost of a bounce on landing.
    # Aiming at z=0 (old behaviour) caused a perpetual micro-bounce loop - see
    # the comment in ball_physics.py's bounce threshold logic.
    aim_point = target_position.with_z(params.pass_aim_height_m)
    distance = kicker_position.xy().distance_to(aim_point.xy())

    if power_fraction is None:
        auto_speed = pass_speed_mps(params, distance, gravity_mps2, rolling_friction_coefficient)
        max_kick = max_kick_speed_mps(kp, kick_power_attr)
        base_power_fraction = auto_speed / max(max_kick, 1e-6)
    else:
        auto_speed = params.max_speed_mps * max(0.0, min(1.0, power_fraction))
        base_power_fraction = max(0.0, min(1.0, power_fraction))

    aim_dir = aim_point - kicker_position
    run_mult = running_power_multiplier(
        running_power_coefficient, kicker_velocity, aim_dir, kicker_top_speed_mps
    )
    dir_precision_mult = running_direction_precision_multiplier(kicker_velocity, aim_dir, kp)
    effective_precision = kick_precision * dir_precision_mult

    effective_power = base_power_fraction * run_mult
    speed = auto_speed * run_mult

    sigma = kick_sigma_rad(kp, effective_precision, effective_power, rng_reduction)

    _log_kick_debug(
        "pass_ball", kicker_velocity, aim_dir, run_mult, dir_precision_mult,
        kick_precision, effective_precision, effective_power, sigma, speed, rng_reduction,
        base_power_fraction=base_power_fraction,
    )

    _launch_ball(ball, kicker_position, aim_point, speed, sigma, Vector3.zero(), r, gravity_mps2)

# ShootOrder blocker detection: perpendicular-distance threshold from the shot
# line within which an opposition player is considered to be blocking the shot.
SHOT_BLOCKER_THRESHOLD_M: float = 1.0
# How far the shooter moves toward the goal when pausing due to a blocker.
SHOT_PAUSE_ADVANCE_M: float = 2.0


def has_blocker_on_shot_line(
    shooter_pos: Vector3,
    aim_point: Vector3,
    opposition: list,
    threshold_m: float = SHOT_BLOCKER_THRESHOLD_M,
) -> bool:
    """Return True if any active opposition player lies within *threshold_m*
    of the line segment from *shooter_pos* to *aim_point* (XY plane only)
    and is between the shooter and the aim point.

    ``opposition`` is a list of Player objects; inactive (tackled) players
    are excluded — they cannot intercept a shot.
    """
    import math as _math
    from footballcoach.entities.player import PlayerState
    sx, sy = shooter_pos.x, shooter_pos.y
    ax, ay = aim_point.x, aim_point.y
    dx, dy = ax - sx, ay - sy
    line_len_sq = dx * dx + dy * dy
    if line_len_sq < 1e-12:
        return False
    for opp in opposition:
        if opp.state == PlayerState.INACTIVE_TACKLED:
            continue
        ox, oy = opp.position.x, opp.position.y
        t = ((ox - sx) * dx + (oy - sy) * dy) / line_len_sq
        if t <= 0.0 or t >= 1.0:
            continue
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        if _math.hypot(ox - proj_x, oy - proj_y) < threshold_m:
            return True
    return False