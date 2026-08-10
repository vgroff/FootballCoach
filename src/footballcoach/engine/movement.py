"""Movement, stamina, and turning kinematics for players.

See engine/knowledge.md for the full derivation and justification of the
constants used here (they all live in config/physics.json).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3

log = logging.getLogger("footballcoach.movement")


class SpeedMode(Enum):
    """Three discrete target-speed modes for ``step_player_towards``.

    The order/AI layer picks a mode; the engine then uses its full
    kinematics (acceleration, turning, stamina) to reach that target.
    No order or action is ever permitted to write ``player.velocity``
    directly — ``SpeedMode`` is the sole control surface.  See
    ``src/footballcoach/knowledge.md`` for the full engine/AI boundary rule.
    """
    SPRINT = auto()      # target = v_top (full sprint pace)
    JOG = auto()         # target = v_top * 0.5 (half pace)
    STANDSTILL = auto()  # target = 0 m/s — decelerate to a stop


# Below this threshold in STANDSTILL mode, velocity is snapped to exactly zero
# to prevent floating-point drift keeping an "idle" player creeping forever.
# This is the *only* velocity snap in the codebase.
_STOP_SNAP_THRESHOLD_MPS: float = 0.02


@dataclass(frozen=True)
class MovementParams:
    top_speed_base_mps: float
    top_speed_scale_mps: float
    accel_base_mps2: float
    accel_scale_mps2: float
    stamina_speed_penalty_max: float
    stamina_drain_sprint_base_per_s: float
    stamina_drain_attr_lo: float
    stamina_drain_attr_hi: float
    stamina_regen_idle_base_per_s: float
    stamina_regen_attr_lo: float
    stamina_regen_attr_hi: float
    ball_carry_speed_mult_base: float
    ball_carry_speed_mult_scale: float
    lateral_accel_base_mps2: float
    lateral_accel_scale_mps2: float
    lateral_accel_ball_penalty_max: float
    min_speed_for_turn_mps: float
    goalkeeper_accel_multiplier: float
    goalkeeper_speed_multiplier: float
    standstill_decel_multiplier: float  # accel boost when decelerating to STANDSTILL
    turn_speed_penalty_max: float  # max fractional speed-cap reduction mid-turn (0.7 = up to 70%)
    control_speed_multiplier: float  # fraction of speed kept on first-touch contact (0.6 = 40% reduction)

    @staticmethod
    def from_config() -> "MovementParams":
        d = require_section(load_physics_config(), "movement")
        return MovementParams(
            top_speed_base_mps=d["top_speed_base_mps"],
            top_speed_scale_mps=d["top_speed_scale_mps"],
            accel_base_mps2=d["accel_base_mps2"],
            accel_scale_mps2=d["accel_scale_mps2"],
            stamina_speed_penalty_max=d["stamina_speed_penalty_max"],
            stamina_drain_sprint_base_per_s=d["stamina_drain_sprint_base_per_s"],
            stamina_drain_attr_lo=d["stamina_drain_attr_lo"],
            stamina_drain_attr_hi=d["stamina_drain_attr_hi"],
            stamina_regen_idle_base_per_s=d["stamina_regen_idle_base_per_s"],
            stamina_regen_attr_lo=d["stamina_regen_attr_lo"],
            stamina_regen_attr_hi=d["stamina_regen_attr_hi"],
            ball_carry_speed_mult_base=d["ball_carry_speed_mult_base"],
            ball_carry_speed_mult_scale=d["ball_carry_speed_mult_scale"],
            lateral_accel_base_mps2=d["lateral_accel_base_mps2"],
            lateral_accel_scale_mps2=d["lateral_accel_scale_mps2"],
            lateral_accel_ball_penalty_max=d["lateral_accel_ball_penalty_max"],
            min_speed_for_turn_mps=d["min_speed_for_turn_mps"],
            goalkeeper_accel_multiplier=d["goalkeeper_accel_multiplier"],
            goalkeeper_speed_multiplier=d["goalkeeper_speed_multiplier"],
            standstill_decel_multiplier=d.get("standstill_decel_multiplier", 1.5),
            turn_speed_penalty_max=d.get("turn_speed_penalty_max", 0.7),
            control_speed_multiplier=d.get("control_speed_multiplier", 0.6),
        )


def top_speed_mps(params: MovementParams, top_speed_attr: float) -> float:
    """Uncapped top speed for a given top_speed attribute (before stamina/ball penalties)."""
    return params.top_speed_base_mps + params.top_speed_scale_mps * top_speed_attr


def max_acceleration_mps2(params: MovementParams, acceleration_attr: float) -> float:
    return params.accel_base_mps2 + params.accel_scale_mps2 * acceleration_attr


def stamina_multiplier(params: MovementParams, stamina_fraction: float) -> float:
    """Multiplier applied to top speed & acceleration based on current stamina.

    At stamina_fraction=1.0 (fresh), multiplier=1.0 (no penalty).
    At stamina_fraction=0.0 (exhausted), multiplier=1 - stamina_speed_penalty_max
    (i.e. up to a 65% reduction, per the design spec).
    """
    return 1.0 - params.stamina_speed_penalty_max * (1.0 - stamina_fraction)


def ball_carry_speed_multiplier(params: MovementParams, ball_control_attr: float) -> float:
    """Multiplier applied to top speed while dribbling with the ball.

    Never reaches 1.0 even at ball_control=1.0 (base + scale = 0.75+0.22=0.97),
    per the design requirement that dribbling is always somewhat slower.
    """
    return params.ball_carry_speed_mult_base + params.ball_carry_speed_mult_scale * ball_control_attr


def effective_top_speed(
    params: MovementParams,
    top_speed_attr: float,
    stamina_fraction: float,
    has_ball: bool,
    ball_control_attr: float = 0.0,
    is_goalkeeper: bool = False,
) -> float:
    speed = top_speed_mps(params, top_speed_attr) * stamina_multiplier(params, stamina_fraction)
    if has_ball:
        speed *= ball_carry_speed_multiplier(params, ball_control_attr)
    if is_goalkeeper:
        speed *= params.goalkeeper_speed_multiplier
    return speed


def effective_acceleration(
    params: MovementParams,
    acceleration_attr: float,
    stamina_fraction: float,
    is_goalkeeper: bool = False,
) -> float:
    accel = max_acceleration_mps2(params, acceleration_attr) * stamina_multiplier(params, stamina_fraction)
    if is_goalkeeper:
        # Simulates diving reach: goalkeepers get a flat acceleration boost
        # on top of their attribute-driven acceleration, per the design
        # spec ("give goalkeeper like 1.5x acceleration to simulate
        # diving"). Applies to all goalkeeper movement, not just Save
        # orders, since a keeper's explosive first step matters generally.
        accel *= params.goalkeeper_accel_multiplier
    return accel


def sprint_eta(dist_m: float, v0_mps: float, v_top_mps: float, accel_mps2: float) -> float:
    """Estimate time (s) to sprint ``dist_m`` from initial speed ``v0_mps``.

    Uses a two-phase constant-acceleration model:
    - Phase 1: accelerate from ``v0_mps`` to ``v_top_mps`` at ``accel_mps2``.
    - Phase 2: cruise at ``v_top_mps``.

    If the distance is covered before reaching top speed, solves the quadratic
    ``v0*t + 0.5*a*t^2 = dist`` analytically instead.

    All arguments should be non-negative.  Returns 0 for zero (or negative)
    distance and a large sentinel (1e9) when speed and acceleration are both
    effectively zero.
    """
    if dist_m <= 0.0:
        return 0.0
    v_top_mps = max(v_top_mps, 1e-3)
    if accel_mps2 <= 0.0 or v0_mps >= v_top_mps:
        return dist_m / v_top_mps
    # Phase 1: time and distance to reach top speed
    t1 = (v_top_mps - v0_mps) / accel_mps2
    d1 = v0_mps * t1 + 0.5 * accel_mps2 * t1 * t1
    if d1 >= dist_m:
        # Distance covered entirely during acceleration phase
        disc = v0_mps * v0_mps + 2.0 * accel_mps2 * dist_m
        return (-v0_mps + math.sqrt(disc)) / accel_mps2
    # Phase 2: remaining distance at cruise speed
    return t1 + (dist_m - d1) / v_top_mps


def lateral_accel_capability(
    params: MovementParams,
    acceleration_attr: float,
    has_ball: bool,
    ball_control_attr: float = 0.0,
    is_goalkeeper: bool = False,
) -> float:
    """Max lateral (turning) acceleration available, in m/s^2.

    This governs turn rate: omega_max = a_lat / max(speed, eps). Carrying the
    ball reduces this unless ball_control is high (at ball_control=1.0 there
    is no penalty, per the design spec).

    Goalkeepers get the same `goalkeeper_accel_multiplier` boost applied
    here as `effective_acceleration` - without it, a keeper with boosted
    straight-line acceleration but unboosted turning would build up speed
    towards a save target faster than they can *correct* direction as the
    predicted crossing point shifts, causing them to overshoot/oscillate
    past a moving target instead of diving effectively (see
    tests/balance/test_save_balance.py's fast-vs-slow-keeper regression).
    """
    a_lat = params.lateral_accel_base_mps2 + params.lateral_accel_scale_mps2 * acceleration_attr
    if has_ball:
        a_lat *= 1.0 - params.lateral_accel_ball_penalty_max * (1.0 - ball_control_attr)
    if is_goalkeeper:
        a_lat *= params.goalkeeper_accel_multiplier
    return a_lat


def max_turn_rate_rad_s(
    params: MovementParams,
    acceleration_attr: float,
    speed_mps: float,
    has_ball: bool,
    ball_control_attr: float = 0.0,
    is_goalkeeper: bool = False,
) -> float:
    """omega_max = a_lat / max(speed, min_speed) -- turning is "free" (fast) at
    low speed and increasingly constrained at high speed, matching real
    running biomechanics (tight turns cost more the faster you're moving)."""
    a_lat = lateral_accel_capability(params, acceleration_attr, has_ball, ball_control_attr, is_goalkeeper)
    denom = max(speed_mps, params.min_speed_for_turn_mps)
    return a_lat / denom


def drain_stamina(
    params: MovementParams,
    stamina_fraction: float,
    stamina_attr: float,
    effort: float,
    dt_s: float,
) -> float:
    """Returns updated stamina fraction after `dt_s` of sprinting at `effort`
    in [0, 1] (0 = standing still/no drain, 1 = full sprint).

    Higher stamina_attr drains more slowly: drain_rate scales linearly from
    stamina_drain_attr_lo (attr=0) down to stamina_drain_attr_hi (attr=1).
    """
    drain_rate = params.stamina_drain_sprint_base_per_s * (
        params.stamina_drain_attr_lo - (params.stamina_drain_attr_lo - params.stamina_drain_attr_hi) * stamina_attr
    )
    new_stamina = stamina_fraction - drain_rate * effort * dt_s
    return max(0.0, min(1.0, new_stamina))


def regen_stamina(
    params: MovementParams,
    stamina_fraction: float,
    stamina_attr: float,
    dt_s: float,
) -> float:
    """Returns updated stamina fraction after `dt_s` of resting/jogging.

    Higher stamina_attr regenerates faster.
    """
    regen_rate = params.stamina_regen_idle_base_per_s * (
        params.stamina_regen_attr_lo + (params.stamina_regen_attr_hi - params.stamina_regen_attr_lo) * stamina_attr
    )
    new_stamina = stamina_fraction + regen_rate * dt_s
    return max(0.0, min(1.0, new_stamina))


def step_player_towards(
    player: Player,
    target_direction: Vector3,
    speed_mode: SpeedMode,
    dt_s: float,
    params: MovementParams | None = None,
    has_ball: bool = False,
) -> None:
    """Advances `player`'s velocity/heading/position one physics tick towards
    `target_direction` (need not be normalized; zero vector means decelerate
    to a stop). Mutates `player` in place.

    This is the *only* function permitted to write ``player.velocity`` —
    all callers must use ``SpeedMode`` to express intent rather than
    assigning velocity directly.  See ``src/footballcoach/knowledge.md``.
    """
    params = params or MovementParams.from_config()
    attrs = player.attributes

    v_top = effective_top_speed(params, attrs.top_speed, player.stamina, has_ball, attrs.ball_control, player.is_goalkeeper)
    a_max = effective_acceleration(params, attrs.acceleration, player.stamina, player.is_goalkeeper)

    current_speed = player.velocity.length_xy()
    desired_dir = target_direction.xy().normalized()

    if speed_mode is SpeedMode.SPRINT:
        desired_speed = v_top
    elif speed_mode is SpeedMode.JOG:
        desired_speed = v_top * 0.5
    else:  # STANDSTILL
        desired_speed = 0.0
        # Deceleration to standstill uses a boosted acceleration so stopping
        # feels snappier than accelerating (1.5× by default, config-tunable).
        a_max *= params.standstill_decel_multiplier

    if desired_dir.length() < 1e-9:
        # No target direction: decelerate to a stop, keeping current heading.
        desired_dir = player.velocity.xy().normalized()
        desired_speed = 0.0

    # Turn rate limits how fast heading can rotate towards the desired direction.
    current_heading = player.heading_rad
    desired_heading = desired_dir.angle_xy() if desired_dir.length() > 1e-9 else current_heading
    omega_max = max_turn_rate_rad_s(
        params, attrs.acceleration, max(current_speed, 0.5), has_ball, attrs.ball_control, player.is_goalkeeper
    )

    heading_diff = angle_diff(current_heading, desired_heading)
    max_turn_this_tick = omega_max * dt_s
    if abs(heading_diff) <= max_turn_this_tick:
        new_heading = desired_heading
    else:
        new_heading = current_heading + math.copysign(max_turn_this_tick, heading_diff)

    # Speed change is limited by acceleration, and further reduced the more
    # the player is turning (large heading changes cost more speed), per the
    # design requirement.
    turn_fraction = min(abs(heading_diff) / math.pi, 1.0) if desired_dir.length() > 1e-9 else 0.0
    turn_speed_penalty = 1.0 - params.turn_speed_penalty_max * turn_fraction
    target_speed = desired_speed * turn_speed_penalty

    speed_diff = target_speed - current_speed
    max_delta = a_max * dt_s
    if abs(speed_diff) <= max_delta:
        new_speed = target_speed
    else:
        new_speed = current_speed + math.copysign(max_delta, speed_diff)
    new_speed = max(0.0, new_speed)

    # Physics-level snap: the *only* velocity snap in the codebase.
    # Prevents floating-point drift keeping a player infinitely creeping
    # when the engine has already driven them to (near-)zero.
    if new_speed < _STOP_SNAP_THRESHOLD_MPS and desired_speed == 0.0:
        new_speed = 0.0

    player.heading_rad = new_heading
    player.velocity = Vector3.from_angle_xy(new_heading, new_speed)
    player.position = player.position + player.velocity * dt_s

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "[movement] pid=%s  heading: %.2f->%.2f (diff=%.2f)  "
            "speed: %.2f->%.2f (target=%.2f  a_max=%.2f)  "
            "turn_frac=%.2f  v_top=%.2f  pos: (%.3f,%.3f)->(%.3f,%.3f)",
            player.player_id,
            current_heading, new_heading, heading_diff,
            current_speed, new_speed, target_speed, a_max,
            turn_fraction, v_top,
            player.position.x - player.velocity.x * dt_s,
            player.position.y - player.velocity.y * dt_s,
            player.position.x, player.position.y,
        )


def angle_diff(a: float, b: float) -> float:
    """Returns b - a wrapped to [-pi, pi]."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return d


def drain_if_sprinting(params: MovementParams, player: Player, sprinting: bool, dt: float) -> float:
    """Drain stamina only if the player is actually sprinting; otherwise
    return current stamina unchanged.
    """
    if not sprinting:
        return player.stamina
    return drain_stamina(params, player.stamina, player.attributes.stamina, 1.0, dt)
