"""Unit tests for action/gating.py - winner-take-all action selection.

CRITICAL: this is entirely outside the gradient graph (pure Python,
no autograd).  Tests verify the correct action is selected from probabilities,
and that the physical execution outputs pass through intact.

See ai_design_doc.md section 2.6 for the gating rule:
  "if any head's probability >= 0.5: select the single highest one"
  "if ALL heads are < 0.5: select NONE (continue current order)"
"""
import pytest

from footballcoach.ai.action.gating import GatingResult, SelectedAction, select_action

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EXEC_PHYSICAL = {
    "move_direction": __import__("numpy").array([1.0, 0.0]),
    "sprint": True,
    "kick_this_tick": False,
    "kick_direction": __import__("numpy").array([1.0, 0.0]),
    "kick_power_fraction": 0.8,
    "kick_spin": __import__("numpy").zeros(3),
    "tackle_attempt": False,
}

_EMPTY_PROBS = {
    "shoot": 0.0, "pass_": 0.0, "move": 0.0, "tackle": 0.0,
    "get_possession": 0.0, "mark": 0.0, "hold_position": 0.0,
}

_EMPTY_TARGETS = {"pass_": 0, "tackle": 0, "mark": 0}


def _select(probs, targets=None, exec_phys=None):
    return select_action(
        decision_probs={**_EMPTY_PROBS, **probs},
        execution_physical=exec_phys or _EXEC_PHYSICAL,
        target_slots=targets or _EMPTY_TARGETS,
    )


# ---------------------------------------------------------------------------
# All heads below 0.5 -> NONE
# ---------------------------------------------------------------------------

def test_all_below_threshold_selects_none():
    result = _select({"shoot": 0.49, "move": 0.48})
    assert result.selected == SelectedAction.NONE


def test_all_zero_probs_selects_none():
    result = _select({})
    assert result.selected == SelectedAction.NONE


def test_exactly_half_does_not_fire():
    """0.5 is NOT above threshold (strictly > 0.5 required)."""
    result = _select({"shoot": 0.5})
    assert result.selected == SelectedAction.NONE


# ---------------------------------------------------------------------------
# Single head above 0.5
# ---------------------------------------------------------------------------

def test_shoot_fires_when_above_threshold():
    result = _select({"shoot": 0.9})
    assert result.selected == SelectedAction.SHOOT


def test_pass_fires_when_above_threshold():
    result = _select({"pass_": 0.7})
    assert result.selected == SelectedAction.PASS


def test_move_fires_when_above_threshold():
    result = _select({"move": 0.6})
    assert result.selected == SelectedAction.MOVE


def test_tackle_fires_when_above_threshold():
    result = _select({"tackle": 0.8})
    assert result.selected == SelectedAction.TACKLE


def test_get_possession_fires_when_above_threshold():
    result = _select({"get_possession": 0.75})
    assert result.selected == SelectedAction.GET_POSSESSION


def test_mark_fires_when_above_threshold():
    result = _select({"mark": 0.55})
    assert result.selected == SelectedAction.MARK


def test_hold_position_fires_when_above_threshold():
    result = _select({"hold_position": 0.65})
    assert result.selected == SelectedAction.HOLD_POSITION


# ---------------------------------------------------------------------------
# Multiple heads above 0.5 -> highest wins
# ---------------------------------------------------------------------------

def test_highest_prob_wins_not_first_in_list():
    """Shoot has higher prob than move -> shoot wins."""
    result = _select({"shoot": 0.9, "move": 0.7})
    assert result.selected == SelectedAction.SHOOT


def test_highest_prob_wins_when_many_above():
    result = _select({"shoot": 0.6, "pass_": 0.7, "move": 0.8, "tackle": 0.51})
    assert result.selected == SelectedAction.MOVE


def test_winner_is_not_the_last_listed():
    """If move (last in priority) has highest prob, it still wins."""
    result = _select({"shoot": 0.51, "hold_position": 0.99})
    assert result.selected == SelectedAction.HOLD_POSITION


# ---------------------------------------------------------------------------
# Target slot propagation
# ---------------------------------------------------------------------------

def test_pass_target_slot_propagated():
    result = _select({"pass_": 0.9}, targets={"pass_": 5, "tackle": 0, "mark": 0})
    assert result.selected == SelectedAction.PASS
    assert result.target_slot == 5


def test_tackle_target_slot_propagated():
    result = _select({"tackle": 0.9}, targets={"pass_": 0, "tackle": 3, "mark": 0})
    assert result.selected == SelectedAction.TACKLE
    assert result.target_slot == 3


def test_none_has_no_target():
    result = _select({})
    assert result.target_slot is None


def test_shoot_has_no_target_slot():
    """SHOOT has no categorical target."""
    result = _select({"shoot": 0.9})
    assert result.target_slot is None


# ---------------------------------------------------------------------------
# Execution outputs always pass through
# ---------------------------------------------------------------------------

def test_move_direction_passed_through():
    import numpy as np
    exec_phys = {**_EXEC_PHYSICAL, "move_direction": np.array([0.5, 0.5])}
    result = _select({"shoot": 0.9}, exec_phys=exec_phys)
    assert result.move_direction is not None
    assert float(result.move_direction[0]) == pytest.approx(0.5)


def test_sprint_passed_through():
    exec_phys = {**_EXEC_PHYSICAL, "sprint": False}
    result = _select({"move": 0.9}, exec_phys=exec_phys)
    assert result.sprint is False


def test_kick_power_passed_through():
    exec_phys = {**_EXEC_PHYSICAL, "kick_power_fraction": 0.42}
    result = _select({"shoot": 0.9}, exec_phys=exec_phys)
    assert result.kick_power_fraction == pytest.approx(0.42)


def test_execution_outputs_passed_through_for_none():
    """Even when NONE is selected, execution outputs are available."""
    result = _select({})
    assert result.move_direction is not None
    assert result.sprint is True  # from _EXEC_PHYSICAL
