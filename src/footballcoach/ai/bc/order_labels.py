"""BC labels for ANY order a rules AI issues: which decision head it is, plus
the execution heads read from one counterfactual run of the order
(ai/ppo/bc._counterfactual_execution: nothing in the real match changes).

Decision heads come from the order's TYPE (the network's decision is "which
kind of action"), via DECISION_HEAD_OF_ORDER. Some orders have no exact head
and are mapped to the closest action the network can take; each such case is
listed (and its reason) in that table, so a new order type fails loudly
(invalid label) instead of being silently mislabelled.

Execution heads (move direction, sprint, standstill, kick fire/arm with its
direction / power / spin, tackle arm) are whatever the order's execute() set
on the player, kicks read from the order's kick intent before its random
error (engine/kick_intent.py). Target-slot heads (pass / tackle / mark
target) are not in the label vector and are not supervised.

Phase 1 keeps its own phase1_labels_for_player (which asks Phase1RulesAI for
the order); for the Move / GetPossession orders it issues this gives
identical labels (tests/ai_scenario/test_order_label_audit.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from footballcoach.ai.ppo.bc import BCLabel, _ai_type_of, _counterfactual_execution

if TYPE_CHECKING:
    from footballcoach.engine.match import Match
    from footballcoach.entities.player import Player


def _head_table() -> dict:
    from footballcoach.orders import (
        ChaseTackleOrder, GetPossessionOrder, JogOrder, KickOrder, MarkOrder, MoveOrder, PassOrder,
        SaveOrder, ShootOrder, StopOrder,
    )
    return {
        MoveOrder: "move",
        GetPossessionOrder: "get_possession_extra",
        ShootOrder: "shoot",
        PassOrder: "pass_",
        ChaseTackleOrder: "tackle",
        MarkOrder: "mark",
        StopOrder: "hold_position",   # stand still where you are
        # No head of its own: the network has no "save" action; a neural keeper
        # saves by moving to the ball (its reach comes from is_goalkeeper), so
        # the save run is a move.
        SaveOrder: "move",
        # A strike at an aim point (scenario fixtures: penalties, save drills):
        # the network's shot is its kick head, so it is a shoot.
        KickOrder: "shoot",
        # Immobile drill players only (a fixed-direction jog): a move.
        JogOrder: "move",
    }


DECISION_HEAD_OF_ORDER = None   # filled lazily (orders imports would be circular at module load)


def decision_head_of(order) -> str | None:
    global DECISION_HEAD_OF_ORDER
    if DECISION_HEAD_OF_ORDER is None:
        DECISION_HEAD_OF_ORDER = _head_table()
    return DECISION_HEAD_OF_ORDER.get(type(order))


def _opponent_ai_type(player, match) -> float:
    """As phase 1: the AI type of the first player of the other team (the
    first other player when there is no other team)."""
    others = [p for p in match.players if p.player_id != player.player_id]
    opp = next((p for p in others if p.team != player.team), others[0] if others else None)
    return _ai_type_of(opp.ai) if opp is not None else 1.0


def order_label(player: "Player", match: "Match", order) -> tuple[BCLabel, dict]:
    """(label, counterfactual result) for ``order`` issued to ``player`` now.
    The label is invalid for an order type with no decision head. The
    counterfactual result carries ``completed`` / ``order_after`` for callers
    that track the order's progress (ai/bc/role_experts.py)."""
    import math

    from footballcoach.orders import GetPossessionOrder, MoveOrder

    cf = _counterfactual_execution(player, match, order)
    # An order that handed over this tick (a ShootOrder pausing for a blocker
    # installs a MoveOrder and runs it) is labelled as what it became.
    labelled = cf["order_after"] if cf["order_after"] is not None else order
    head = decision_head_of(labelled)
    if head is None:
        return BCLabel.invalid(), cf
    region = None
    if isinstance(labelled, MoveOrder):
        region = np.array([labelled.target_position.x, labelled.target_position.y], dtype=np.float32)
    elif isinstance(labelled, GetPossessionOrder):
        region = np.array([match.ball.position.x, match.ball.position.y], dtype=np.float32)
    label = BCLabel(
        sprint=cf["sprint_label"],
        move_direction=cf["move_direction"],
        move_region_center_m=region,
        kick_this_tick=cf["kick_this_tick"],
        tackle_attempt=cf["tackle_attempt"],
        exec_move=cf["exec_move_label"],
        ai_type=_ai_type_of(player.ai),
        opponent_ai_type=_opponent_ai_type(player, match),
        kick_direction=cf["kick_direction"],
        kick_power_fraction=cf["kick_power_fraction"],
        kick_spin=cf["kick_spin"],
        heading_sin=math.sin(player.heading_rad),
        heading_cos=math.cos(player.heading_rad),
    )
    setattr(label, head, 1.0)
    return label, cf
