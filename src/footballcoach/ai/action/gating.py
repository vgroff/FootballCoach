"""Action gating: non-differentiable winner-take-all selection.

This module implements the post-hoc, non-differentiable gating rule that
translates the decision network's raw sigmoid probabilities into a single
selected action for the engine to execute.

CRITICAL SEPARATION (ai_design_doc.md section 2.6):
  - THIS MODULE IS NEVER IN THE GRADIENT GRAPH.  It is pure Python, no torch
    operations that propagate gradients, and is called AFTER sampling/log_prob
    have already been computed.
  - PPO's backprop works on the raw Bernoulli logits in the network.
  - Only AFTER gradients are computed does this module run to decide what
    actually happens in the engine this tick.

Rule (from ai_design_doc.md section 2.6):
  If any head's probability >= 0.5: select the single highest-probability
  head (treat it as 1.0, all others as 0.0) for the purpose of assigning
  engine orders.

  If ALL heads are < 0.5: no specific strategic action is triggered.  The
  execution network still drives physical movement (move direction, sprint/jog,
  kick, tackle attempt) via its own outputs.  No new order is assigned this
  tick; the player continues executing whatever order was previously assigned
  (orders persist tick-to-tick in the engine naturally).

Open item resolved here (ai_design_doc.md section 13): "all heads below 50%"
=> the player continues current order + execution-network motor outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class SelectedAction(Enum):
    """Which high-level action was selected by the gating rule."""
    NONE = auto()          # all heads < 0.5, continue current order
    SHOOT = auto()
    PASS = auto()
    MOVE = auto()
    TACKLE = auto()
    GET_POSSESSION = auto()
    MARK = auto()
    HOLD_POSITION = auto()


@dataclass
class GatingResult:
    """Output of select_action(); consumed by to_orders.py."""
    selected: SelectedAction
    # Target slot index (for PASS/TACKLE/MARK), or None
    target_slot: Optional[int] = None
    # Physical (squashed/rescaled) continuous outputs, always set regardless
    # of which action is selected (execution network always runs).
    exec_move: bool = False                          # True = move, False = standstill
    move_direction: Optional["np.ndarray"] = None   # 2D unit vector, (2,)
    sprint: bool = False
    kick_this_tick: bool = False
    kick_direction: Optional["np.ndarray"] = None   # 2D unit vector, (2,)
    kick_power_fraction: float = 0.0
    kick_spin: Optional["np.ndarray"] = None         # (3,)
    tackle_attempt: bool = False


# Head names and their sigmoid probabilities, in priority order for tie-breaking
# (tie-breaking order: shoot > pass > move > tackle > get_possession > mark >
# hold_position - the simplest resolution; adjust if needed).
_HEAD_ORDER: list[str] = [
    "shoot", "pass_", "move", "tackle", "get_possession", "mark", "hold_position"
]


@torch.no_grad()
def select_action(
    decision_probs: dict[str, float],
    execution_physical: dict,
    target_slots: dict[str, int],
) -> GatingResult:
    """Apply the winner-take-all gating rule.

    Args:
        decision_probs: Dict mapping head name to its sigmoid probability
            (Python float, unbatched).  Keys: 'shoot', 'pass_', 'move',
            'tackle', 'get_possession', 'mark', 'hold_position'.
        execution_physical: Dict of physical execution network outputs:
            'move_direction' (np.ndarray, 2), 'sprint' (bool),
            'kick_this_tick' (bool), 'kick_direction' (np.ndarray, 2),
            'kick_power_fraction' (float), 'kick_spin' (np.ndarray, 3),
            'tackle_attempt' (bool).
        target_slots: Dict mapping categorical head name to sampled slot
            index: 'pass', 'tackle', 'mark'.

    Returns:
        GatingResult with selected action and physical motor outputs.
    """
    # Find the highest-probability head above 0.5 threshold
    best_head = None
    best_prob = 0.5  # strictly > 0.5 required to fire

    for head_name in _HEAD_ORDER:
        p = decision_probs.get(head_name, 0.0)
        if p > best_prob:
            best_prob = p
            best_head = head_name

    if best_head is None:
        selected = SelectedAction.NONE
        target_slot = None
    else:
        selected = _HEAD_TO_ACTION[best_head]
        target_slot = target_slots.get(best_head)

    return GatingResult(
        selected=selected,
        target_slot=target_slot,
        exec_move=execution_physical.get("exec_move", False),
        move_direction=execution_physical.get("move_direction"),
        sprint=execution_physical.get("sprint", False),
        kick_this_tick=execution_physical.get("kick_this_tick", False),
        kick_direction=execution_physical.get("kick_direction"),
        kick_power_fraction=execution_physical.get("kick_power_fraction", 0.0),
        kick_spin=execution_physical.get("kick_spin"),
        tackle_attempt=execution_physical.get("tackle_attempt", False),
    )


_HEAD_TO_ACTION: dict[str, SelectedAction] = {
    "shoot": SelectedAction.SHOOT,
    "pass_": SelectedAction.PASS,
    "move": SelectedAction.MOVE,
    "tackle": SelectedAction.TACKLE,
    "get_possession": SelectedAction.GET_POSSESSION,
    "mark": SelectedAction.MARK,
    "hold_position": SelectedAction.HOLD_POSITION,
}
