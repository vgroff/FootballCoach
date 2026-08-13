"""Apply neural network execution outputs directly to a player.

This module is the ONLY point where neural network outputs touch engine state.
It sets player.desired_direction, player.desired_speed_mode, calls
player.kick_with_direction(), and sets player.tackle_armed.

No Orders are created here. Orders exist for the rules-based AI only.
The only connection between Orders and the neural network is that BC labels
read what order a rules AI would issue and translate that into equivalent
physical targets for imitation.

Tackle intent is communicated via player.tackle_armed — the engine resolves
it on contact in _check_armed_tackles, firing on_tackle for BC recording.

Slot-index -> player_id mapping: the ``slot_player_ids`` list (produced by
the obs encoder during the same tick - same random shuffle) maps the
categorical target slot index back to a concrete player_id.

Gating's ``move_direction``/``kick_direction`` here are already in raw
world/engine-frame coordinates by the time this module sees them — any
canonical-AI-frame de-mirroring happens earlier, at the network-forward
boundary (see ``ai/obs/canonical.py``), not here.
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
    Tackle: set player.tackle_armed if tackle_attempt is True.

    The decision network heads (shoot/pass/move/etc.) are INPUTS to the
    execution network and are used for BC label generation only. They do
    not trigger any Orders here.
    """
    from footballcoach.engine.movement import SpeedMode

    # --- Movement: exec_move decides standstill vs moving; sprint decides speed ---
    # move_direction is set from gating.move_direction regardless of exec_move:
    # step_player_towards() turns a STANDSTILL player to face a nonzero
    # target_direction while decelerating to a stop (turn-in-place), same as
    # a rules-AI order sitting at SpeedMode.STANDSTILL with a real
    # desired_direction (e.g. settling into first-touch ball control). The BC
    # label already supervises move_direction unconditionally (phase1_labels()
    # captures it whenever player.desired_direction is nonzero post-execute(),
    # not gated on exec_move; bc_loss_from_tensor's cosine loss is gated only
    # on the label having a direction, never on exec_move) -- this was pure
    # inference-time information loss, discarding an already-trained signal.
    d = gating.move_direction
    if d is not None and np.linalg.norm(d) > 1e-6:
        player.desired_direction = Vector3(float(d[0]), float(d[1]), 0.0)
    else:
        player.desired_direction = Vector3.zero()
    player.desired_speed_mode = (
        (SpeedMode.SPRINT if gating.sprint else SpeedMode.JOG) if gating.exec_move else SpeedMode.STANDSTILL
    )

    # --- Kick: immediate physics via 3D direction, no ballistic solve ---
    # Fires only when the player actually has the ball; if not (e.g. cached gating
    # while chasing), the call is a silent no-op — kick_with_direction checks
    # possessed_by internally. First-touch difficulty is applied automatically
    # by kick_with_direction when the player is in CONTROLLING_BALL state.
    if gating.kick_this_tick:
        kick_dir = gating.kick_direction
        if kick_dir is not None and np.linalg.norm(kick_dir) > 1e-6:
            direction_3d = Vector3(float(kick_dir[0]), float(kick_dir[1]), float(kick_dir[2]) if len(kick_dir) > 2 else 0.0)
        else:
            direction_3d = Vector3(1.0, 0.0, 0.0)  # safe fallback, should not occur
        player.kick_with_direction(
            match,
            direction_3d,
            float(gating.kick_power_fraction) if gating.kick_power_fraction > 0 else 0.85,
            # Spin is disabled for the neural network for now -- see
            # agent_plans/spin_implementation_plan.md for the plan to re-enable it.
            Vector3.zero(),
        )

    # --- Tackle: arm intent; _check_armed_tackles resolves on contact ---
    # Target slot is not used — the engine finds the ball carrier directly.
    if gating.tackle_attempt:
        if not player.is_available_to_tackle():
            return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_while_inactive")
        if match.ball_carrier() is None or match.ball_carrier().team == player.team:
            return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_no_carrier")
        player.tackle_armed = True

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
