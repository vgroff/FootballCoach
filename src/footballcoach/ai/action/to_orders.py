"""Translate GatingResult + physical execution outputs -> engine orders.

This is the bridge between the AI layer's gating output and the engine's
``orders.py`` order types.  It also detects illegal action attempts
(preconditions not met) so the reward function can penalise them.

DESIGN NOTE (ai_design_doc.md section 9.7 / section 11):
  - The engine already guards illegal actions as safe no-ops (no state
    corruption). This module detects the attempt INDEPENDENTLY so the
    reward function receives an explicit signal.
  - Both protections coexist: engine safety + reward-shaping deterrent.

Slot-index -> player_id mapping: the ``slot_player_ids`` list (produced by
the obs encoder during the same tick - same random shuffle) maps the
categorical target slot index back to a concrete player_id.  The obs
encoder must produce and pass this mapping alongside the ObservationBatch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from footballcoach.ai.action.gating import GatingResult, SelectedAction
from footballcoach.engine.match import Match
from footballcoach.entities.player import Player, PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder,
    GetPossessionOrder,
    KickOrder,
    MarkOrder,
    MoveOrder,
    PassOrder,
    SaveOrder,
    StopOrder,
)


@dataclass
class OrderTranslationResult:
    """Result of translating one gating output to engine orders."""
    illegal_action: bool = False
    illegal_reason: str = ""


def apply_action_to_player(
    gating: GatingResult,
    player: Player,
    match: Match,
    slot_player_ids: list[Optional[str]],
    decision_physical: dict,
) -> OrderTranslationResult:
    """Translate a GatingResult into an engine order on ``player``.

    Args:
        gating: Output of ``select_action()``.
        player: The player receiving the order.
        match: Running match (needed for pitch geometry and other players).
        slot_player_ids: List of length MAX_OTHER_PLAYERS mapping slot index
            -> player_id (None for padded slots).  Must match the shuffle
            from the same obs encoding step.
        decision_physical: Dict of decoded physical values from the decision
            network (move_region_center_m, move_region_size_m,
            move_arrival_speed_mps).  All in physical units after squashing.

    Returns:
        OrderTranslationResult with illegal_action flag.
    """
    sel = gating.selected
    pitch = match.pitch

    # -----------------------------------------------------------------------
    # Execution-network motor outputs: always applied regardless of what the
    # decision network selected, UNLESS a specific high-level order already
    # encapsulates movement (e.g. ChaseTackleOrder runs its own chase logic).
    # For MOVE/HOLD_POSITION, the execution network's move_direction is used
    # as the preferred direction within the requested region.
    # -----------------------------------------------------------------------

    if sel == SelectedAction.SHOOT:
        return _apply_shoot(player, match, gating, decision_physical)

    elif sel == SelectedAction.PASS:
        return _apply_pass(player, match, gating, slot_player_ids, decision_physical)

    elif sel in (SelectedAction.MOVE, SelectedAction.HOLD_POSITION):
        return _apply_move(player, gating, decision_physical, hold_position=(sel == SelectedAction.HOLD_POSITION))

    elif sel == SelectedAction.TACKLE:
        return _apply_tackle(player, match, gating, slot_player_ids)

    elif sel == SelectedAction.GET_POSSESSION:
        return _apply_get_possession(player, match)

    elif sel == SelectedAction.MARK:
        return _apply_mark(player, match, gating, slot_player_ids)

    else:  # SelectedAction.NONE
        return _apply_none(player, gating)


# ---------------------------------------------------------------------------
# Per-action helpers
# ---------------------------------------------------------------------------

def _apply_shoot(player: Player, match: Match, gating: GatingResult, decision_physical: dict) -> OrderTranslationResult:
    """Issue a KickOrder toward goal (or the execution network's kick_direction)."""
    # Precondition: player must have the ball
    if match.ball.possessed_by != player.player_id:
        return OrderTranslationResult(illegal_action=True, illegal_reason="shoot_without_possession")

    # Use the execution network's kick_direction to determine aim_point.
    # Project direction forward to a reasonable aim distance, then use
    # actions.shoot's aim_point override for exact goal targeting.
    kick_dir = gating.kick_direction  # (2,) unit vector
    if kick_dir is None or np.linalg.norm(kick_dir) < 1e-6:
        # Fallback: aim at opponent goal centre
        from footballcoach.actions import opponent_goal_centre
        aim_pt = opponent_goal_centre(match.pitch, player.team).with_z(1.1)
    else:
        # Use kick_direction to pick the y-offset within the goal
        goal_half_w = match.pitch.goal_width_m / 2.0
        if player.team.name == "LEFT":
            goal_x = match.pitch.half_length
        else:
            goal_x = -match.pitch.half_length
        aim_y = float(kick_dir[1]) * goal_half_w
        aim_pt = Vector3(goal_x, aim_y, 1.1)

    player.current_order = KickOrder(
        aim_point=aim_pt,
        power_fraction=float(gating.kick_power_fraction) if gating.kick_power_fraction > 0 else 0.85,
        spin=Vector3(*gating.kick_spin) if gating.kick_spin is not None else Vector3.zero(),
    )
    return OrderTranslationResult()


def _apply_pass(
    player: Player,
    match: Match,
    gating: GatingResult,
    slot_player_ids: list[Optional[str]],
    decision_physical: dict,
) -> OrderTranslationResult:
    if match.ball.possessed_by != player.player_id:
        return OrderTranslationResult(illegal_action=True, illegal_reason="pass_without_possession")

    target_player = _resolve_target_player(gating.target_slot, slot_player_ids, match)
    if target_player is None:
        return OrderTranslationResult(illegal_action=True, illegal_reason="pass_no_valid_target")

    player.current_order = PassOrder(
        target_position=target_player.position,
        target_player_id=target_player.player_id,
    )
    return OrderTranslationResult()


def _apply_move(
    player: Player,
    gating: GatingResult,
    decision_physical: dict,
    hold_position: bool,
) -> OrderTranslationResult:
    """Issue a MoveOrder toward the decision network's move_region_center."""
    center = decision_physical.get("move_region_center_m")
    if center is None:
        # No region data; use player's current position (effectively hold)
        target = player.position
    else:
        target = Vector3(float(center[0]), float(center[1]), 0.0)

    arrival_speed = decision_physical.get("move_arrival_speed_mps")
    sprint = gating.sprint if gating.sprint is not None else True

    player.current_order = MoveOrder(
        target_position=target,
        sprint=sprint,
        max_speed_on_arrival_mps=arrival_speed,
    )
    return OrderTranslationResult()


def _apply_tackle(
    player: Player,
    match: Match,
    gating: GatingResult,
    slot_player_ids: list[Optional[str]],
) -> OrderTranslationResult:
    if player.state == PlayerState.INACTIVE_TACKLED:
        return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_while_inactive")

    target_player = _resolve_target_player(gating.target_slot, slot_player_ids, match)
    if target_player is None:
        return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_no_valid_target")

    # Don't tackle own teammates
    if target_player.team == player.team:
        return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_own_teammate")

    player.current_order = ChaseTackleOrder(target_player_id=target_player.player_id)
    return OrderTranslationResult()


def _apply_get_possession(player: Player, match: Match) -> OrderTranslationResult:
    if player.state == PlayerState.INACTIVE_TACKLED:
        return OrderTranslationResult(illegal_action=True, illegal_reason="get_possession_while_inactive")

    player.current_order = GetPossessionOrder()
    return OrderTranslationResult()


def _apply_mark(
    player: Player,
    match: Match,
    gating: GatingResult,
    slot_player_ids: list[Optional[str]],
) -> OrderTranslationResult:
    target_player = _resolve_target_player(gating.target_slot, slot_player_ids, match)
    if target_player is None:
        return OrderTranslationResult(illegal_action=True, illegal_reason="mark_no_valid_target")

    if target_player.team == player.team:
        return OrderTranslationResult(illegal_action=True, illegal_reason="mark_own_teammate")

    player.current_order = MarkOrder(target_player_id=target_player.player_id)
    return OrderTranslationResult()


def _apply_none(player: Player, gating: GatingResult) -> OrderTranslationResult:
    """All heads < 0.5: execution network drives movement only.

    Issue a MoveOrder in the execution network's move_direction if the player
    currently has no persistent order in progress; otherwise leave the
    current order alone (it will continue executing per the engine's
    tick-based order persistence).
    """
    from footballcoach.orders import OrderStatus
    current = player.current_order

    # If the player has an active in-progress order (not complete), leave it.
    if current is not None:
        status = getattr(current, "status", None)
        if status is not None and status != OrderStatus.COMPLETE:
            return OrderTranslationResult()

    # No active order: move in execution network direction (keep player alive)
    if gating.move_direction is not None:
        d = gating.move_direction
        # Take a step of 5m in the indicated direction as a provisional target
        step = 5.0
        target = Vector3(
            player.position.x + float(d[0]) * step,
            player.position.y + float(d[1]) * step,
            0.0,
        )
        player.current_order = MoveOrder(
            target_position=target,
            sprint=bool(gating.sprint),
        )

    return OrderTranslationResult()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _resolve_target_player(
    slot: Optional[int],
    slot_player_ids: list[Optional[str]],
    match: Match,
) -> Optional[Player]:
    """Look up the Player object for a categorical target slot index."""
    if slot is None or slot < 0 or slot >= len(slot_player_ids):
        return None
    pid = slot_player_ids[slot]
    if pid is None:
        return None
    try:
        return match.player_by_id(pid)
    except (KeyError, AttributeError):
        return None


def encode_slot_player_ids(
    match: Match,
    player_id: str,
    slot_indices: list[int],
    other_players: list[Player],
) -> list[Optional[str]]:
    """Build the slot_player_ids list that matches the slot assignment used
    during obs encoding.

    Call with the SAME ``slot_indices`` and ``other_players`` order that the
    obs encoder used so that slot_player_ids[i] always corresponds to
    other_feat[i] for any given observation.

    Returns a list of MAX_OTHER_PLAYERS entries, None for padded slots.
    """
    from footballcoach.ai.obs.encoder import MAX_OTHER_PLAYERS
    result: list[Optional[str]] = [None] * MAX_OTHER_PLAYERS
    for slot_idx, op in zip(slot_indices, other_players):
        result[slot_idx] = op.player_id
    return result
