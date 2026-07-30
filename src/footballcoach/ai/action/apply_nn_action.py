"""Apply neural network execution outputs directly to a player.

This module is the ONLY point where neural network outputs touch engine state.
It sets player.desired_direction, player.desired_speed_mode, calls
player.kick_direct(), and player.tackle_direct() directly.

No Orders are created here. Orders exist for the rules-based AI only.
The only connection between Orders and the neural network is that BC labels
read what order a rules AI would issue and translate that into equivalent
physical targets for imitation.

Slot-index -> player_id mapping: the ``slot_player_ids`` list (produced by
the obs encoder during the same tick - same random shuffle) maps the
categorical target slot index back to a concrete player_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from footballcoach.ai.action.gating import GatingResult
from footballcoach.engine.match import Match
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3


@dataclass
class OrderTranslationResult:
    """Result of applying neural network outputs to a player."""
    illegal_action: bool = False
    illegal_reason: str = ""


def apply_action_to_player(
    gating: GatingResult,
    player: Player,
    match: Match,
    slot_player_ids: list[Optional[str]],
    decision_physical: dict,
) -> OrderTranslationResult:
    """Apply execution-network outputs DIRECTLY to the player — no Orders.

    Movement: set player.desired_direction and player.desired_speed_mode.
    Kick: call player.kick_direct() if kick_this_tick is True.
    Tackle: call player.tackle_direct() if tackle_attempt is True.

    The decision network heads (shoot/pass/move/etc.) are INPUTS to the
    execution network and are used for BC label generation only. They do
    not trigger any Orders here.
    """
    from footballcoach.engine.movement import SpeedMode

    # --- Movement: exec_move decides standstill vs moving; sprint decides speed ---
    if gating.exec_move:
        d = gating.move_direction
        if d is not None and np.linalg.norm(d) > 1e-6:
            player.desired_direction = Vector3(float(d[0]), float(d[1]), 0.0)
        else:
            player.desired_direction = Vector3.zero()
        player.desired_speed_mode = SpeedMode.SPRINT if gating.sprint else SpeedMode.JOG
    else:
        player.desired_direction = Vector3.zero()
        player.desired_speed_mode = SpeedMode.STANDSTILL

    # --- Kick: immediate physics, no KickOrder ---
    if gating.kick_this_tick:
        if match.ball.possessed_by == player.player_id:
            kick_dir = gating.kick_direction
            if kick_dir is not None and np.linalg.norm(kick_dir) > 1e-6:
                goal_half_w = match.pitch.goal_width_m / 2.0
                goal_x = match.pitch.half_length if player.team.name == "LEFT" else -match.pitch.half_length
                aim_pt = Vector3(goal_x, float(kick_dir[1]) * goal_half_w, 1.1)
            else:
                from footballcoach.actions import opponent_goal_centre
                aim_pt = opponent_goal_centre(match.pitch, player.team).with_z(1.1)
            player.kick_direct(
                match,
                aim_pt,
                float(gating.kick_power_fraction) if gating.kick_power_fraction > 0 else 0.85,
                Vector3(*gating.kick_spin) if gating.kick_spin is not None else Vector3.zero(),
            )
        else:
            return OrderTranslationResult(illegal_action=True, illegal_reason="kick_without_possession")

    # --- Tackle: immediate physics if in contact range, no ChaseTackleOrder ---
    if gating.tackle_attempt:
        if not player.is_available_to_tackle():
            return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_while_inactive")
        target_player = _resolve_target_player(gating.target_slot, slot_player_ids, match)
        if target_player is None:
            return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_no_valid_target")
        if target_player.team == player.team:
            return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_own_teammate")
        player.tackle_direct(match, target_player.player_id)

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
