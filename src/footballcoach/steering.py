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

from footballcoach.ai.config import load_ai_config
from footballcoach.config import require_section
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3

log = logging.getLogger("footballcoach.steering")


@dataclass(frozen=True)
class RepulsionParams:
    """Config for the repulsion steering mechanic — loaded from ai_config.json."""
    radius_m: float
    strength_base: float
    ball_carrier_repulsion_mult: float
    ball_carrier_speed_penalty_max: float
    speed_penalty_scale: float
    alignment_dot_threshold: float
    min_orthogonal_adjust_mps: float
    max_deflection_deg: float = 90.0

    @staticmethod
    def from_config() -> "RepulsionParams":
        d = require_section(load_ai_config(), "repulsion")
        return RepulsionParams(
            radius_m=d["radius_m"],
            strength_base=d["strength_base"],
            ball_carrier_repulsion_mult=d["ball_carrier_repulsion_mult"],
            ball_carrier_speed_penalty_max=d["ball_carrier_speed_penalty_max"],
            speed_penalty_scale=d["speed_penalty_scale"],
            alignment_dot_threshold=d["alignment_dot_threshold"],
            min_orthogonal_adjust_mps=d["min_orthogonal_adjust_mps"],
            max_deflection_deg=d.get("max_deflection_deg", 90.0),
        )


def compute_repulsion(
    player: Player,
    desired_dir: Vector3,
    other_players: list[Player],
    ball_carrier_id: str | None,
    params: RepulsionParams,
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

    Returns
    -------
    adjusted_direction:
        A ``Vector3`` (z=0) giving the blended direction after repulsion
        and any orthogonal nudge.  If ``desired_dir`` is the zero vector,
        returns ``Vector3.zero()`` unchanged (nothing to blend).
    speed_multiplier:
        A value in ``[0, 1]``.  ``1.0`` means no speed change; lower
        values indicate the player should slow down (relevant when the
        player is a ball carrier near an obstacle).  Non-carrier players
        always receive ``1.0``.
    """
    has_ball = (ball_carrier_id is not None and ball_carrier_id == player.player_id)

    # Distance to the move target (desired_dir is the unnormalised vector to it).
    target_dist = (desired_dir.x * desired_dir.x + desired_dir.y * desired_dir.y) ** 0.5

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

        # If already closer to the target than to this obstacle, it is no
        # longer in the way — ignore its repulsion entirely.
        if target_dist < dist:
            continue

        # Repulsion: away from other, linear falloff.
        strength = params.strength_base * (1.0 - dist / params.radius_m)
        inv_dist = 1.0 / dist
        net_rep_x += dx * inv_dist * strength
        net_rep_y += dy * inv_dist * strength

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_other = other

    # Ball carrier gets stronger directional repulsion toward others.
    if has_ball:
        net_rep_x *= params.ball_carrier_repulsion_mult
        net_rep_y *= params.ball_carrier_repulsion_mult

    # ── Speed multiplier (ball carrier only) ─────────────────────────────
    speed_multiplier = 1.0
    if has_ball:
        net_rep_len = (net_rep_x * net_rep_x + net_rep_y * net_rep_y) ** 0.5
        speed_penalty = min(
            params.ball_carrier_speed_penalty_max,
            net_rep_len * params.speed_penalty_scale,
        )
        speed_multiplier = 1.0 - speed_penalty

    # ── Early-out: no desired direction ───────────────────────────────────
    desired_len = (desired_dir.x * desired_dir.x + desired_dir.y * desired_dir.y) ** 0.5
    if desired_len < 1e-9:
        # Nowhere to go — just return zero, no direction to blend into.
        return Vector3.zero(), speed_multiplier

    # ── Orthogonal nudge when heading nearly straight into nearest obstacle
    ortho_x: float = 0.0
    ortho_y: float = 0.0
    if nearest_other is not None and params.min_orthogonal_adjust_mps > 0.0:
        net_rep_len = (net_rep_x * net_rep_x + net_rep_y * net_rep_y) ** 0.5
        if net_rep_len > 1e-9:
            # rel_vel = player.velocity - nearest_other.velocity (xy only)
            rvx = player.velocity.x - nearest_other.velocity.x
            rvy = player.velocity.y - nearest_other.velocity.y
            rv_len = (rvx * rvx + rvy * rvy) ** 0.5
            if rv_len > 1e-9:
                # Dot of normalised rel_vel with normalised net_repulsion.
                # net_repulsion points AWAY from obstacle.
                # rel_vel pointing OPPOSITE to repulsion → heading into obstacle.
                dot = (rvx / rv_len) * (net_rep_x / net_rep_len) + (rvy / rv_len) * (net_rep_y / net_rep_len)
                if dot < params.alignment_dot_threshold:
                    # Orthogonal to repulsion direction; pick the side closer
                    # to the desired direction via the 2D cross product sign.
                    # cross_z(net_rep, desired_dir) = net_rep_x*desired_y - net_rep_y*desired_x
                    cross_z = net_rep_x * desired_dir.y - net_rep_y * desired_dir.x
                    if cross_z >= 0.0:
                        # Desired is to the left of repulsion → nudge left
                        # Left perpendicular of (x,y) = (-y, x)
                        perp_x = -net_rep_y / net_rep_len
                        perp_y = net_rep_x / net_rep_len
                    else:
                        # Desired is to the right → nudge right
                        # Right perpendicular of (x,y) = (y, -x)
                        perp_x = net_rep_y / net_rep_len
                        perp_y = -net_rep_x / net_rep_len
                    ortho_x = perp_x * params.min_orthogonal_adjust_mps
                    ortho_y = perp_y * params.min_orthogonal_adjust_mps

    # ── Blend: final_dir = normalise(desired_dir_norm + net_rep + ortho) ─
    dd_x = desired_dir.x / desired_len
    dd_y = desired_dir.y / desired_len

    final_x = dd_x + net_rep_x + ortho_x
    final_y = dd_y + net_rep_y + ortho_y
    final_len = (final_x * final_x + final_y * final_y) ** 0.5
    if final_len < 1e-9:
        # Vectors cancelled completely — fall back to original desired dir.
        return Vector3(dd_x, dd_y, 0.0), speed_multiplier

    fn_x = final_x / final_len
    fn_y = final_y / final_len

    log.debug(
        "[repulsion] pid=%s  has_ball=%s  nearest=%.2fm  "
        "net_rep=(%.3f,%.3f)  ortho=(%.3f,%.3f)  "
        "raw_dir=(%.3f,%.3f)  blended=(%.3f,%.3f)  speed_mult=%.3f",
        player.player_id, has_ball,
        nearest_dist if nearest_other else float("inf"),
        net_rep_x, net_rep_y,
        ortho_x, ortho_y,
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
