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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from footballcoach.ai.action.gating import GatingResult, SelectedAction
from footballcoach.engine.match import Match
from footballcoach.entities.player import Player, PlayerState
from footballcoach.mathutils import Vector3
# Player.move_to() / stop() / kick() etc. construct Orders via lazy imports internally.


@dataclass
class OrderTranslationResult:
    """Result of translating one gating output to engine orders."""
    illegal_action: bool = False
    illegal_reason: str = ""


# Far-target distance used when setting movement direction via MoveOrder.
# 50m is beyond any braking horizon within a 0.5s decision interval at
# realistic top speeds (~8 m/s), so the player accelerates in the correct
# direction for the full interval without the engine's braking logic firing.
_FAR_TARGET_M: float = 50.0


def apply_movement_to_player(player: Player, gating: GatingResult) -> None:
    """Apply execution-network movement unconditionally every decision tick.

    Sets a MoveOrder with a far-away target in ``gating.move_direction`` so
    the engine's ``step_player_towards`` picks up the correct heading for all
    ~15 ticks in the decision interval.  Uses the execution network's
    ``sprint`` Bernoulli to choose SPRINT vs JOG speed mode.

    When the decision network selects NONE (all heads < 0.5, including move),
    the player decelerates to STANDSTILL instead.

    This function is always called first by ``apply_action_to_player``.  If a
    high-level order fires afterwards (shoot/pass/tackle/etc.), it overwrites
    the MoveOrder — those orders encapsulate their own movement logic.
    """
    # NONE means all Bernoulli heads < 0.5, including move → STANDSTILL.
    if gating.selected == SelectedAction.NONE:
        player.stop()
        return

    d = gating.move_direction
    if d is None or np.linalg.norm(d) < 1e-6:
        player.stop()
        return

    target = Vector3(
        player.position.x + float(d[0]) * _FAR_TARGET_M,
        player.position.y + float(d[1]) * _FAR_TARGET_M,
        0.0,
    )
    sprint = bool(gating.sprint) if gating.sprint is not None else False
    player.move_to(target, sprint=sprint)


def apply_action_to_player(
    gating: GatingResult,
    player: Player,
    match: Match,
    slot_player_ids: list[Optional[str]],
    decision_physical: dict,
) -> OrderTranslationResult:
    """Translate a GatingResult into engine orders on ``player``.

    Two-phase execution:
      1. ``apply_movement_to_player`` — always sets a MoveOrder (or StopOrder)
         based on the execution network's ``move_direction`` + ``sprint``.
      2. High-level order dispatch — if a decision head fired, call the
         corresponding player action method, which overwrites the MoveOrder.

    Args:
        gating: Output of ``select_action()``.
        player: The player receiving the order.
        match: Running match (needed for pitch geometry and other players).
        slot_player_ids: List of length MAX_OTHER_PLAYERS mapping slot index
            -> player_id (None for padded slots).
        decision_physical: Decoded physical values from the decision network
            (move_region_center_m etc.). Used only for shoot aim; movement
            is driven entirely by execution-network ``move_direction``.

    Returns:
        OrderTranslationResult with illegal_action flag.
    """
    # Phase 1: movement is unconditional.
    apply_movement_to_player(player, gating)

    # Phase 2: high-level order (overwrites movement order when it fires).
    sel = gating.selected

    if sel == SelectedAction.SHOOT:
        return _apply_shoot(player, match, gating, decision_physical)

    elif sel == SelectedAction.PASS:
        return _apply_pass(player, match, gating, slot_player_ids)

    elif sel == SelectedAction.TACKLE:
        return _apply_tackle(player, match, gating, slot_player_ids)

    elif sel == SelectedAction.GET_POSSESSION:
        return _apply_get_possession(player, match)

    elif sel == SelectedAction.MARK:
        return _apply_mark(player, match, gating, slot_player_ids)

    # MOVE, HOLD_POSITION, NONE: movement already applied in phase 1.
    return OrderTranslationResult()


# ---------------------------------------------------------------------------
# Per-action helpers
# ---------------------------------------------------------------------------

def _apply_shoot(player: Player, match: Match, gating: GatingResult, decision_physical: dict) -> OrderTranslationResult:
    """Issue a KickOrder toward goal using the execution network's kick_direction."""
    if match.ball.possessed_by != player.player_id:
        return OrderTranslationResult(illegal_action=True, illegal_reason="shoot_without_possession")

    kick_dir = gating.kick_direction  # (2,) unit vector
    if kick_dir is None or np.linalg.norm(kick_dir) < 1e-6:
        from footballcoach.actions import opponent_goal_centre
        aim_pt = opponent_goal_centre(match.pitch, player.team).with_z(1.1)
    else:
        goal_half_w = match.pitch.goal_width_m / 2.0
        goal_x = match.pitch.half_length if player.team.name == "LEFT" else -match.pitch.half_length
        aim_y = float(kick_dir[1]) * goal_half_w
        aim_pt = Vector3(goal_x, aim_y, 1.1)

    player.kick(
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
) -> OrderTranslationResult:
    if match.ball.possessed_by != player.player_id:
        return OrderTranslationResult(illegal_action=True, illegal_reason="pass_without_possession")

    target_player = _resolve_target_player(gating.target_slot, slot_player_ids, match)
    if target_player is None:
        return OrderTranslationResult(illegal_action=True, illegal_reason="pass_no_valid_target")

    player.pass_ball(
        target_position=target_player.position,
        target_player_id=target_player.player_id,
    )
    return OrderTranslationResult()


def _apply_tackle(
    player: Player,
    match: Match,
    gating: GatingResult,
    slot_player_ids: list[Optional[str]],
) -> OrderTranslationResult:
    if not player.is_available_to_tackle():
        return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_while_inactive")

    target_player = _resolve_target_player(gating.target_slot, slot_player_ids, match)
    if target_player is None:
        return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_no_valid_target")

    if target_player.team == player.team:
        return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_own_teammate")

    player.tackle_player(target_player_id=target_player.player_id)
    return OrderTranslationResult()


def _apply_get_possession(player: Player, match: Match) -> OrderTranslationResult:
    if not player.is_available_to_tackle():
        return OrderTranslationResult(illegal_action=True, illegal_reason="get_possession_while_inactive")

    player.get_possession()
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

    player.mark_player(target_player_id=target_player.player_id)
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
