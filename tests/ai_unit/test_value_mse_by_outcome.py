"""Unit tests for ppo_trainer.value_mse_by_outcome() / format_outcome_rmse_breakdown()
-- the shared helper used by debug_value_network.py, PPOTrainer.pretrain_combined()'s
Phase 0, PPOTrainer.pretrain_value(), and PPOTrainer._ppo_update() to report value
loss split by episode outcome (see ai_trainer_knowledge.md / Idea2.md's "value loss
split by outcome" request). value_mse_by_outcome() itself still returns raw MSE
(squared-error units); format_outcome_rmse_breakdown() takes the sqrt at display
time so the printed numbers are in the same units as the value/return itself.
"""
from __future__ import annotations

import math

import pytest
import torch

from footballcoach.ai.ppo.ppo_trainer import format_outcome_rmse_breakdown, value_mse_by_outcome


class TestValueMseByOutcome:
    def test_splits_squared_error_by_outcome_label(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 10.0])
        target = torch.tensor([0.0, 2.0, 3.0, 0.0])
        outcomes = ["win", "win", "loss", "loss"]

        result = value_mse_by_outcome(pred, target, outcomes)

        assert set(result.keys()) == {"win", "loss"}
        win_mse, win_n = result["win"]
        loss_mse, loss_n = result["loss"]
        assert win_n == 2
        assert loss_n == 2
        # win: errors (1-0)^2=1, (2-2)^2=0 -> mean 0.5
        assert win_mse == pytest.approx(0.5)
        # loss: errors (3-3)^2=0, (10-0)^2=100 -> mean 50.0
        assert loss_mse == pytest.approx(50.0)

    def test_empty_outcomes_list_returns_empty_dict(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, 2.0])
        assert value_mse_by_outcome(pred, target, []) == {}

    def test_empty_string_outcome_grouped_as_unknown(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([0.0, 0.0])
        result = value_mse_by_outcome(pred, target, ["", "win"])
        assert "unknown" in result
        assert "win" in result
        assert result["unknown"][1] == 1
        assert result["win"][1] == 1

    def test_single_outcome_matches_plain_mse(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 5.0])
        result = value_mse_by_outcome(pred, target, ["win", "win", "win"])
        expected = float(torch.nn.functional.mse_loss(pred, target))
        mse, n = result["win"]
        assert n == 3
        assert mse == pytest.approx(expected)

    def test_does_not_require_grad_tracking_on_inputs(self):
        """pred is typically detached inside callers before this is invoked,
        but the function itself must not blow up if it isn't (mirrors
        .detach() being applied internally)."""
        pred = torch.tensor([1.0, 2.0], requires_grad=True)
        target = torch.tensor([1.5, 2.5])
        result = value_mse_by_outcome(pred, target, ["timeout", "timeout"])
        assert "timeout" in result


class TestFormatOutcomeRmseBreakdown:
    def test_empty_dict_returns_empty_string(self):
        assert format_outcome_rmse_breakdown({}) == ""

    def test_formats_sorted_by_outcome_name_as_rmse(self):
        by_outcome = {"win": (0.5, 10), "loss": (2.0, 3)}
        formatted = format_outcome_rmse_breakdown(by_outcome)
        expected_loss = math.sqrt(2.0)
        expected_win = math.sqrt(0.5)
        assert formatted == f"loss={expected_loss:.3f}(n=3)  win={expected_win:.3f}(n=10)"
