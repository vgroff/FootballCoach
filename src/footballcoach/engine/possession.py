"""Ball possession and first-touch control-time model.

See engine/knowledge.md for the full derivation of the control-time formula
and its constants (control points chosen to feel intuitive: a rolling ball
at the feet is a near-instant touch, a fast chest-high ball is a real
challenge, especially for low ball-control players).
"""
from __future__ import annotations

from dataclasses import dataclass

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.ball import Ball
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3


@dataclass(frozen=True)
class ControlTimeParams:
    t_base_s: float
    t_scale_s: float
    height_factor_knee: float
    height_factor_waist: float
    height_factor_head: float
    height_factor_above_head_extra: float
    height_factor_max: float
    k1_relative_velocity_s_per_mps: float
    k2_own_velocity_s_per_mps: float
    ball_control_alpha: float
    noise_sigma_fraction: float
    knee_height_m: float
    waist_height_m: float
    player_height_m: float
    gk_t_base_s: float
    gk_height_factor_scale: float
    gk_ball_control_alpha: float
    # Jump zone penalties (above player_height_m = 1.8m)
    # GK: flatter penalty, higher ceiling (gloves + specialised technique)
    gk_jump_scale_at_max_reach: float  # height_factor_scale at gk_max_reach_height_m
    gk_max_reach_height_m: float       # max height GK can credibly intercept
    # Outfield: steeper penalty, lower ceiling
    outfield_jump_scale_at_max_reach: float  # applied as multiplier on (height_factor-1) at max reach
    outfield_max_reach_height_m: float       # max height outfield can credibly jump to
    control_tackle_immune_height_m: float    # ball must be below this height for tackles to land on a CONTROLLING_BALL player

    @staticmethod
    def from_config() -> "ControlTimeParams":
        cfg = load_physics_config()
        d = require_section(cfg, "control_time")
        player_cfg = require_section(cfg, "player")
        gk = require_section(d, "goalkeeper", file_name="physics.json:control_time")
        return ControlTimeParams(
            t_base_s=d["t_base_s"],
            t_scale_s=d["t_scale_s"],
            height_factor_knee=d["height_factor_knee"],
            height_factor_waist=d["height_factor_waist"],
            height_factor_head=d["height_factor_head"],
            height_factor_above_head_extra=d["height_factor_above_head_extra"],
            height_factor_max=d["height_factor_max"],
            k1_relative_velocity_s_per_mps=d["k1_relative_velocity_s_per_mps"],
            k2_own_velocity_s_per_mps=d["k2_own_velocity_s_per_mps"],
            ball_control_alpha=d["ball_control_alpha"],
            noise_sigma_fraction=d["noise_sigma_fraction"],
            knee_height_m=player_cfg["knee_height_m"],
            waist_height_m=player_cfg["waist_height_m"],
            player_height_m=player_cfg["height_m"],
            gk_t_base_s=gk["t_base_s"],
            gk_height_factor_scale=gk["height_factor_scale"],
            gk_ball_control_alpha=gk["ball_control_alpha"],
            gk_jump_scale_at_max_reach=gk.get("jump_scale_at_max_reach", 1.2),
            gk_max_reach_height_m=gk.get("max_reach_height_m", 2.2),
            outfield_jump_scale_at_max_reach=d.get("outfield_jump_scale_at_max_reach", 2.0),
            outfield_max_reach_height_m=d.get("outfield_max_reach_height_m", 2.0),
            control_tackle_immune_height_m=d.get("control_tackle_immune_height_m", 0.95),
        )


@dataclass(frozen=True)
class BallPickupParams:
    """Config-driven constants for loose-ball pickup eligibility."""
    pickup_radius_m: float
    closing_speed_deadzone_mps: float  # below this relative speed, pickup is allowed even without closing

    @staticmethod
    def from_config() -> "BallPickupParams":
        d = require_section(load_physics_config(), "ball_pickup")
        return BallPickupParams(
            pickup_radius_m=d["pickup_radius_m"],
            closing_speed_deadzone_mps=d["closing_speed_deadzone_mps"],
        )


def can_pick_up_ball(
    player: Player,
    ball: Ball,
    params: BallPickupParams,
    ball_pre_tick_position: Vector3 | None = None,
    player_pre_tick_position: Vector3 | None = None,
) -> bool:
    """True if `player` is close enough to `ball`'s CURRENT position AND
    either closed on it at some point during this tick's motion, or is
    moving relative to it slowly enough that "closing" isn't a meaningful
    requirement (a resting/barely-rolling ball right next to a player
    should always be pickable).

    Replaces the old time-based "release grace" hack: a player can no
    longer instantly re-pick-up a ball they just kicked away at real speed,
    because it is moving away from them (not closing) — with no
    special-casing of who kicked it. It also means a ball rolling PAST a
    slow/stationary player is only picked up if they have a real closing
    velocity component toward it, not by proximity alone.

    `ball_pre_tick_position` / `player_pre_tick_position` (each entity's
    position before this tick's movement) are used to test the WHOLE swept
    path of ball-relative-to-player this tick, not just the end-of-tick
    separation — a fast ball (e.g. a hard shot, 10+ m per tick relative to
    the pickup radius) can tunnel straight past a player within one tick,
    and symmetrically a fast player can sweep straight past a slow/
    stationary ball (e.g. a sprinting player grazing the edge of the pickup
    radius) — both end up on the "far side" reading as separated/receding
    under a naive endpoint-only check even though the two were within
    pickup range at some instant during the tick. Passing either or both
    pre-tick positions enables the corresponding side of the sweep; the
    check is done in the ball-relative-to-player frame (see
    `_swept_min_separation`) so it is exactly symmetric and also correctly
    covers the case where BOTH ball and player move during the tick (e.g. a
    rolling ball crossed by a sprinting player) — a case neither one-sided
    check alone would catch. Omitting a side's pre-tick position treats
    that entity as stationary at its current (end-of-tick) position for the
    sweep, which exactly reproduces the historical ball-only-tunneling
    behaviour when `player_pre_tick_position` is omitted.

    The swept check ONLY applies when the two entities started the tick
    OUTSIDE the pickup radius of each other — this is what distinguishes
    genuine tunneling from a player who just released the ball themselves
    this same tick (whose pre-tick position is at their own feet, i.e.
    already inside the radius): a kick must not be treated as "swept past
    and back" just because its segment happens to start inside the radius.
    The same reasoning protects a player who was already standing next to
    the ball at the start of the tick — their own movement must not be
    treated as a fresh tunneling event either; they fall through to the
    ordinary closing-velocity/deadzone check below, same as always.

    Passing neither pre-tick position (both `None`, the default) skips the
    swept check entirely (endpoint-only). See engine/knowledge.md.
    """
    to_ball = ball.position.xy() - player.position.xy()
    distance = to_ball.length()
    within_radius = distance <= params.pickup_radius_m

    ball_pre_xy = ball_pre_tick_position.xy() if ball_pre_tick_position is not None else ball.position.xy()
    player_pre_xy = player_pre_tick_position.xy() if player_pre_tick_position is not None else player.position.xy()

    have_sweep_info = ball_pre_tick_position is not None or player_pre_tick_position is not None
    started_outside_radius = have_sweep_info and ball_pre_xy.distance_to(player_pre_xy) > params.pickup_radius_m

    swept_within_radius = started_outside_radius and _swept_min_separation(
        ball_pre_xy, ball.position.xy(), player_pre_xy, player.position.xy()
    ) <= params.pickup_radius_m

    if not within_radius and not swept_within_radius:
        return False

    relative_velocity = ball.velocity - player.velocity
    relative_speed = relative_velocity.length_xy()
    if relative_speed <= params.closing_speed_deadzone_mps:
        return True
    if swept_within_radius:
        # Ball-relative-to-player travelled through pickup range this tick
        # despite ending up outside/receding -- a genuine tunneling case,
        # not "kicker re-picking up their own kick" (that case never starts
        # outside the radius, see started_outside_radius above). Always
        # eligible regardless of endpoint closing direction: since both
        # entities are assumed to move in straight lines this tick, a
        # relative-position chord that dips inside the radius necessarily
        # means the two were closing on each other at some point during
        # the sweep, even if they are diverging again by the tick's end.
        return True
    if distance <= 1e-9:
        return True  # already coincident; direction of closing is undefined
    # Closing iff the relative velocity has a component reducing the
    # separation, i.e. pointing from the ball toward the player.
    closing_component = -(relative_velocity.x * to_ball.x + relative_velocity.y * to_ball.y)
    return closing_component > 0.0


def _swept_closest_distance(segment_start: Vector3, segment_end: Vector3, point: Vector3) -> float:
    """Closest distance from `point` to the line segment [segment_start, segment_end]."""
    seg = segment_end - segment_start
    seg_len_sq = seg.length_squared()
    if seg_len_sq <= 1e-12:
        return segment_start.distance_to(point)
    t = max(0.0, min(1.0, (point - segment_start).dot(seg) / seg_len_sq))
    closest = segment_start + seg * t
    return closest.distance_to(point)


def _swept_min_separation(
    ball_start: Vector3, ball_end: Vector3, player_start: Vector3, player_end: Vector3,
) -> float:
    """Minimum ball-to-player separation during this tick, assuming BOTH
    move in straight lines from their tick-start to tick-end positions.

    Computed by translating into the ball-relative-to-player frame: the
    relative position `ball(t) - player(t)` for `t` in [0, 1] traces a
    straight line (the difference of two linear motions is itself linear)
    from `ball_start - player_start` to `ball_end - player_end`, so the
    minimum separation over the tick is just the closest distance from that
    relative-position segment to the origin. This is exactly
    `_swept_closest_distance` applied in the relative frame -- passing
    `player_start == player_end` (a stationary player reference) reduces it
    to the original ball-only sweep, and passing `ball_start == ball_end`
    reduces it to the symmetric player-only sweep. Correctly handles both
    moving simultaneously too (e.g. a rolling ball crossed by a sprinting
    player), which neither one-sided check alone would catch.
    """
    return _swept_closest_distance(
        ball_start - player_start, ball_end - player_end, Vector3.zero(),
    )


def height_difficulty_factor(params: ControlTimeParams, ball_height_m: float) -> float:
    """Piecewise convex ramp: f=1.0 at/below knee height, rising to
    height_factor_waist at waist height, height_factor_head at head height,
    and increasing (capped) beyond that. Modeling choice: it's quadratic
    within each band so difficulty ramps up gently near the breakpoint and
    steeply as the ball gets awkwardly high (matches intuition that a
    shin-high ball is barely harder than a rolling one, but a shoulder-high
    ball is a real technical challenge).
    """
    knee = params.knee_height_m
    waist = params.waist_height_m
    head = params.player_height_m

    if ball_height_m <= knee:
        return params.height_factor_knee
    if ball_height_m <= waist:
        t = (ball_height_m - knee) / (waist - knee)
        return params.height_factor_knee + (params.height_factor_waist - params.height_factor_knee) * (t * t)
    if ball_height_m <= head:
        t = (ball_height_m - waist) / (head - waist)
        return params.height_factor_waist + (params.height_factor_head - params.height_factor_waist) * (t * t)

    extra = params.height_factor_above_head_extra * ((ball_height_m - head) / head)
    return min(params.height_factor_max, params.height_factor_head + extra)


def compute_difficulty(
    params: ControlTimeParams,
    ball_height_m: float,
    relative_speed_mps: float,
    player_speed_mps: float,
) -> float:
    """Returns a unitless "difficulty" score (0 = trivial rolling ball at
    rest at the feet, growing with height and velocity) used both for
    control-time and for inflating first-time shot error."""
    height_factor = height_difficulty_factor(params, ball_height_m)
    return (
        (height_factor - 1.0)
        + params.k1_relative_velocity_s_per_mps * relative_speed_mps
        + params.k2_own_velocity_s_per_mps * player_speed_mps
    )


def _jump_zone_scale(
    ball_height_m: float,
    player_height_m: float,
    max_reach_height_m: float,
    scale_at_player_height: float,
    scale_at_max_reach: float,
) -> float:
    """Interpolates the height-factor scale for balls above ``player_height_m``
    (the "jump zone"): ``scale_at_player_height`` at/below head height, rising
    linearly to ``scale_at_max_reach`` at ``max_reach_height_m`` and beyond.

    Shared by both the goalkeeper and outfield branches of `control_time_s`,
    which apply the same interpolation shape with different endpoints.
    """
    if ball_height_m <= player_height_m:
        return scale_at_player_height
    jump_range = max(max_reach_height_m - player_height_m, 1e-6)
    jump_frac = min(1.0, (ball_height_m - player_height_m) / jump_range)
    return scale_at_player_height + (scale_at_max_reach - scale_at_player_height) * jump_frac


def control_time_s(
    params: ControlTimeParams,
    ball_height_m: float,
    relative_speed_mps: float,
    player_speed_mps: float,
    ball_control_attr: float,
    is_goalkeeper_in_box: bool = False,
) -> float:
    """Returns the (deterministic, pre-RNG) time in seconds for a player to
    bring the ball under control on a first touch.

    t_control = t_base + t_scale * (1 - alpha*ball_control) * difficulty

    Goalkeepers in their own box use a flatter height penalty and a lower
    base time (catching is much easier/faster than an outfielder's touch),
    per the design spec.
    """
    if is_goalkeeper_in_box:
        height_factor = height_difficulty_factor(params, ball_height_m)
        # Below head height: existing flat-scale (gk_height_factor_scale).
        # Above head height (jump zone): interpolate the scale from
        # gk_height_factor_scale toward gk_jump_scale_at_max_reach as the
        # ball rises from player_height_m to gk_max_reach_height_m.
        # Beyond gk_max_reach_height_m the scale stays at max (ball is
        # unreachable — height_factor_max cap still applies).
        effective_scale = _jump_zone_scale(
            ball_height_m, params.player_height_m, params.gk_max_reach_height_m,
            params.gk_height_factor_scale, params.gk_jump_scale_at_max_reach,
        )
        scaled_height_factor = 1.0 + (height_factor - 1.0) * effective_scale
        difficulty = (
            (scaled_height_factor - 1.0)
            + params.k1_relative_velocity_s_per_mps * relative_speed_mps
            + params.k2_own_velocity_s_per_mps * player_speed_mps
        )
        alpha = params.gk_ball_control_alpha
        t_base = params.gk_t_base_s
    else:
        # Below head height: existing difficulty curve, unmodified (pure regression safety).
        # Above head height (outfield jump zone): apply extra difficulty scaling
        # on the height_factor term, interpolating from 1.0 at player_height_m
        # toward outfield_jump_scale_at_max_reach at outfield_max_reach_height_m.
        if ball_height_m > params.player_height_m:
            height_factor = height_difficulty_factor(params, ball_height_m)
            effective_scale = _jump_zone_scale(
                ball_height_m, params.player_height_m, params.outfield_max_reach_height_m,
                1.0, params.outfield_jump_scale_at_max_reach,
            )
            scaled_height_term = (height_factor - 1.0) * effective_scale
            difficulty = (
                scaled_height_term
                + params.k1_relative_velocity_s_per_mps * relative_speed_mps
                + params.k2_own_velocity_s_per_mps * player_speed_mps
            )
        else:
            difficulty = compute_difficulty(params, ball_height_m, relative_speed_mps, player_speed_mps)
        alpha = params.ball_control_alpha
        t_base = params.t_base_s

    extra = (1.0 - alpha * ball_control_attr) * difficulty
    return t_base + params.t_scale_s * extra
