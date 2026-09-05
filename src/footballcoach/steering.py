"""AI/order-layer steering: player repulsion during Move orders.

This module is deliberately NOT under ``engine/`` — it is an AI-layer
decision, analogous to ``actions.py``, that decides *what direction and
speed to request* from the engine, not a physics mechanic.  The engine
(``movement.py``, ``collision.py``) stays unaware of this module.

Only ``engine/match.py``'s MoveOrder handling calls into this module.
No other order type (ChaseTackle, GetPossession, Save, …) does so.

See ``ai/config/ai_config.json["repulsion"]`` for tuning notes.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from footballcoach.config import load_orders_config, require_section
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3

log = logging.getLogger("footballcoach.steering")


@dataclass(frozen=True)
class RepulsionParams:
    """Config for the repulsion steering mechanic — loaded from orders.json."""
    radius_m: float
    strength_base: float
    ball_carrier_repulsion_mult: float
    ball_carrier_speed_penalty_max: float
    speed_penalty_scale: float
    alignment_dot_threshold: float
    max_tangent_deg: float
    max_deflection_deg: float = 90.0
    behind_tolerance_m: float = 1.2
    velocity_lookahead_s: float = 0.4
    boundary_radius_m: float = 0.0
    boundary_strength_base: float = 0.0

    @staticmethod
    def from_config() -> "RepulsionParams":
        d = require_section(load_orders_config(), "repulsion", "orders.json")
        return RepulsionParams(
            radius_m=d["radius_m"],
            strength_base=d["strength_base"],
            ball_carrier_repulsion_mult=d["ball_carrier_repulsion_mult"],
            ball_carrier_speed_penalty_max=d["ball_carrier_speed_penalty_max"],
            speed_penalty_scale=d["speed_penalty_scale"],
            alignment_dot_threshold=d["alignment_dot_threshold"],
            max_tangent_deg=d["max_tangent_deg"],
            max_deflection_deg=d.get("max_deflection_deg", 90.0),
            behind_tolerance_m=d.get("behind_tolerance_m", 1.2),
            velocity_lookahead_s=d.get("velocity_lookahead_s", 0.4),
            boundary_radius_m=d.get("boundary_radius_m", 0.0),
            boundary_strength_base=d.get("boundary_strength_base", 0.0),
        )


def compute_repulsion(
    player: Player,
    desired_dir: Vector3,
    other_players: list[Player],
    ball_carrier_id: str | None,
    params: RepulsionParams,
    pitch: Pitch | None = None,
) -> tuple[Vector3, float]:
    """Compute a repulsion-adjusted movement direction and speed multiplier.

    Parameters
    ----------
    player:
        The player whose movement is being steered.
    desired_dir:
        The raw desired direction vector (need not be normalised; zero
        means the player wants to stop).
    other_players:
        All players in the match (including ``player`` itself — will be
        skipped internally).
    ball_carrier_id:
        ``player_id`` of the current ball carrier, or ``None`` if the ball
        is loose.  Ball carriers are **skipped as repulsion sources** (we
        don't nudge players away from a stationary carrier who isn't in
        their path — only the carrier gets extra push-force toward others).
    params:
        Repulsion config loaded from ``physics.json["repulsion"]``.
    pitch:
        Needed only to compute the boundary-repulsion term below (``None``
        skips it entirely, e.g. existing callers/tests that don't pass one).

    Returns
    -------
    adjusted_direction:
        A ``Vector3`` (z=0) giving the blended direction after repulsion
        (including its smooth per-obstacle tangential rotation -- see
        ``max_tangent_deg``).  If ``desired_dir`` is the zero vector,
        returns ``Vector3.zero()`` unchanged (nothing to blend).
    speed_multiplier:
        A value in ``[0, 1]``.  ``1.0`` means no speed change; lower
        values indicate the player should slow down (relevant when the
        player is a ball carrier near an obstacle, OR near the pitch
        boundary while carrying — see below).  Non-carrier players always
        receive ``1.0``.
    """
    has_ball = (ball_carrier_id is not None and ball_carrier_id == player.player_id)

    # Normalised desired direction — used for the final blend AND for the
    # ahead/behind gate below (desired_dir need not be pre-normalised).
    desired_len = (desired_dir.x * desired_dir.x + desired_dir.y * desired_dir.y) ** 0.5
    if desired_len < 1e-9:
        # Nowhere to go — nothing to blend into, no direction to gate against.
        return Vector3.zero(), 1.0
    dd_x = desired_dir.x / desired_len
    dd_y = desired_dir.y / desired_len

    # ── Accumulate repulsion from nearby non-ball-carrier neighbours ──────
    net_rep_x: float = 0.0
    net_rep_y: float = 0.0
    nearest_dist = float("inf")
    nearest_other: Player | None = None

    for other in other_players:
        if other.player_id == player.player_id:
            continue
        # Do NOT repel from the ball carrier — per the plan's explicit scoping.
        if other.player_id == ball_carrier_id:
            continue

        dx = player.position.x - other.position.x
        dy = player.position.y - other.position.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-9 or dist >= params.radius_m:
            continue

        # Ahead/behind gate: project the obstacle's *predicted* position
        # (current position plus a short velocity lookahead) onto the
        # player's direction of travel. Obstacles behind the player (beyond
        # behind_tolerance_m of slack) aren't in the way and are ignored,
        # regardless of straight-line distance. An obstacle moving away from
        # the path drifts further behind (more likely to be gated out); one
        # converging into the path drifts further ahead (stays gated in)
        # even if currently borderline.
        predicted_x = other.position.x + other.velocity.x * params.velocity_lookahead_s
        predicted_y = other.position.y + other.velocity.y * params.velocity_lookahead_s
        ox = predicted_x - player.position.x
        oy = predicted_y - player.position.y
        along = ox * dd_x + oy * dd_y
        if along < -params.behind_tolerance_m:
            continue

        # Repulsion: away from other, linear falloff.
        strength = params.strength_base * (1.0 - dist / params.radius_m)
        inv_dist = 1.0 / dist
        radial_x = dx * inv_dist
        radial_y = dy * inv_dist

        # dot = closing-velocity alignment: rel_vel projected onto the
        # radial (away-from-obstacle) direction, normalised. -1 = closing
        # dead head-on, 0 = neutral/tangential pass, +1 = retreating dead
        # straight away. Falls back to -1 (treated as fully closing) when
        # there's no relative-velocity signal at all (both stationary or
        # moving identically) -- personal-space separation still applies
        # there since there's no directional information to gate it by,
        # only the (separate, velocity-driven) rotation below skips in
        # that case, since there's no meaningful heading to rotate around.
        rvx = player.velocity.x - other.velocity.x
        rvy = player.velocity.y - other.velocity.y
        rv_len = (rvx * rvx + rvy * rvy) ** 0.5
        dot = (rvx * radial_x + rvy * radial_y) / rv_len if rv_len > 1e-9 else -1.0

        # Scale magnitude by closing velocity: full push while closing
        # head-on, tapering linearly to ZERO once we're moving away from
        # THIS obstacle (dot >= 0) -- a player already retreating from an
        # obstacle has no real collision risk left to react to, so pushing
        # them further is spurious deflection with no avoidance benefit.
        # Distance-only potential fields (the previous version of this
        # function) are the classic case that gets this wrong; velocity-
        # obstacle methods (RVO/ORCA) only ever constrain velocities that
        # would actually lead to a future collision, which this mirrors in
        # spirit without a full rewrite.
        closing_factor = -dot if dot < 0.0 else 0.0
        closing_factor = 1.0 if closing_factor > 1.0 else closing_factor
        strength *= closing_factor

        # Smooth tangential rotation ("spiral field"): as our closing
        # velocity toward THIS obstacle becomes more directly head-on, rotate
        # its radial repulsion vector toward tangential (curve around it)
        # instead of pushing straight backward. Replaces the old separate,
        # single-nearest-neighbour, fixed-magnitude "orthogonal nudge" that
        # snapped on/off at alignment_dot_threshold -- same threshold still
        # marks where the effect starts (dot == alignment_dot_threshold -> no
        # rotation), now ramping continuously to a full max_tangent_deg
        # rotation at dot == -1 (dead head-on), with no discontinuity at the
        # boundary. Applies individually to EVERY obstacle in range, not
        # just whichever one happens to be nearest.
        rx, ry = radial_x, radial_y
        if params.max_tangent_deg > 0.0 and rv_len > 1e-9:
            span = params.alignment_dot_threshold - (-1.0)
            t = (params.alignment_dot_threshold - dot) / span if span > 1e-9 else (
                1.0 if dot < params.alignment_dot_threshold else 0.0
            )
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            if t > 0.0:
                # Which side to curve toward -- same cross-product
                # convention the old orthogonal nudge used. This only
                # degenerates (near-zero, numerically unstable sign)
                # exactly when radial_* and the desired direction are
                # near-antiparallel -- an inherent ambiguity of any
                # symmetric head-on approach (a stateless function of
                # the current tick can't fully resolve which way a
                # dead-straight-on encounter should break either way),
                # not something this rotation introduces.
                cross_z = radial_x * dd_y - radial_y * dd_x
                sign = 1.0 if cross_z >= 0.0 else -1.0
                theta = math.radians(params.max_tangent_deg) * t * sign
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                rx = radial_x * cos_t - radial_y * sin_t
                ry = radial_x * sin_t + radial_y * cos_t

        net_rep_x += rx * strength
        net_rep_y += ry * strength

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_other = other

    # Ball carrier gets stronger directional repulsion toward others.
    if has_ball:
        net_rep_x *= params.ball_carrier_repulsion_mult
        net_rep_y *= params.ball_carrier_repulsion_mult

    # ── Boundary repulsion (ball carriers only) ────────────────────────────
    # Steers the CARRY path away from the pitch boundary before it's
    # crossed, the same way a nearby player steers it -- added directly
    # into net_rep so it rides the existing blend/orthogonal-nudge/
    # max_deflection_deg machinery below and the ball-carrier speed penalty
    # right after this block, rather than a bespoke mechanism.
    #
    # Real gap this closes: Match._run_get_possession_behaviour's boundary
    # braking (engine/match.py's boundary_braking_params) only ever runs
    # while CHASING a loose ball -- the instant a player actually has the
    # ball, movement goes through MoveOrder/this function instead, which
    # had zero boundary awareness at all. Confirmed via a real traced
    # episode: a player picked up the ball 2m from the corner and sprinted
    # straight through the touchline one second later, dead straight line,
    # despite plenty of room to have curved inward. Deliberately steering-
    # only (bend the direction + slow down), not a redirect kick -- run
    # differently, don't take an action to fix it.
    #
    # `params.boundary_radius_m <= 0.0` (the config default) disables this
    # outright, matching every pre-existing test/caller that never
    # anticipated boundary geometry mattering to repulsion.
    if has_ball and pitch is not None and params.boundary_radius_m > 0.0:
        margin_x = pitch.half_length - abs(player.position.x)
        margin_y = pitch.half_width - abs(player.position.y)
        if margin_x < params.boundary_radius_m:
            strength = params.boundary_strength_base * (1.0 - margin_x / params.boundary_radius_m)
            net_rep_x += (-1.0 if player.position.x > 0.0 else 1.0) * strength
        if margin_y < params.boundary_radius_m:
            strength = params.boundary_strength_base * (1.0 - margin_y / params.boundary_radius_m)
            net_rep_y += (-1.0 if player.position.y > 0.0 else 1.0) * strength

    # ── Speed multiplier (ball carrier only) ─────────────────────────────
    speed_multiplier = 1.0
    if has_ball:
        net_rep_len = (net_rep_x * net_rep_x + net_rep_y * net_rep_y) ** 0.5
        speed_penalty = min(
            params.ball_carrier_speed_penalty_max,
            net_rep_len * params.speed_penalty_scale,
        )
        speed_multiplier = 1.0 - speed_penalty

    # ── Blend: final_dir = normalise(desired_dir_norm + net_rep) ──────────
    # (net_rep already carries its own smooth tangential rotation per
    # obstacle, computed in the loop above -- no separate orthogonal term
    # to add here any more.)
    final_x = dd_x + net_rep_x
    final_y = dd_y + net_rep_y
    final_len = (final_x * final_x + final_y * final_y) ** 0.5
    if final_len < 1e-9:
        # Vectors cancelled completely — fall back to original desired dir.
        return Vector3(dd_x, dd_y, 0.0), speed_multiplier

    fn_x = final_x / final_len
    fn_y = final_y / final_len

    log.debug(
        "[repulsion] pid=%s  has_ball=%s  nearest=%.2fm  "
        "net_rep=(%.3f,%.3f)  "
        "raw_dir=(%.3f,%.3f)  blended=(%.3f,%.3f)  speed_mult=%.3f",
        player.player_id, has_ball,
        nearest_dist if nearest_other else float("inf"),
        net_rep_x, net_rep_y,
        dd_x, dd_y,
        fn_x, fn_y,
        speed_multiplier,
    )

    # ── Cap deflection so player never moves backwards ────────────────────
    if params.max_deflection_deg < 180.0:
        import math as _math
        fwd_dot = fn_x * dd_x + fn_y * dd_y
        min_cos = _math.cos(_math.radians(params.max_deflection_deg))
        if fwd_dot < min_cos:
            # Remove the excess backwards component; keep only the sideways part.
            side_x = fn_x - fwd_dot * dd_x
            side_y = fn_y - fwd_dot * dd_y
            side_len = (side_x * side_x + side_y * side_y) ** 0.5
            if side_len > 1e-9:
                sin_max = _math.sin(_math.radians(params.max_deflection_deg))
                fn_x = min_cos * dd_x + sin_max * (side_x / side_len)
                fn_y = min_cos * dd_y + sin_max * (side_y / side_len)
            else:
                fn_x, fn_y = dd_x, dd_y

    if log.isEnabledFor(logging.DEBUG):
        raw_dot = fn_x * dd_x + fn_y * dd_y
        defl_deg = math.degrees(math.acos(max(-1.0, min(1.0, raw_dot))))
        log.debug(
            "[repulsion] pid=%s  final_dir=(%.3f,%.3f)  deflection=%.1f°",
            player.player_id, fn_x, fn_y, defl_deg,
        )

    return Vector3(fn_x, fn_y, 0.0), speed_multiplier
