"""Unit tests for the shared physics_pretrain label-smoothing/floor helper
(``ai/physics_pretrain/event_head_smoothing.py``), used by both
``train_ball_dynamics.py`` and ``train_player_dynamics.py``. Direct port of
``ai/ppo/bc.py``'s pre-existing label-smoothing/floor mechanism -- see that
module's own tests for the analogous coverage on the BC/PPO side.
"""
from __future__ import annotations

import math

import pytest
import torch

from footballcoach.ai.physics_pretrain.event_head_smoothing import (
    bce_label_smoothing_floor,
    expected_bce_floor,
    smooth_target,
)


def test_smooth_target_is_noop_at_zero_smoothing():
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.equal(smooth_target(target, 0.0), target)


def test_smooth_target_hand_computed():
    target = torch.tensor([0.0, 1.0])
    smoothed = smooth_target(target, 0.1)
    # y' = y*(1-s) + 0.5*s
    assert smoothed[0].item() == pytest.approx(0.05, abs=1e-6)
    assert smoothed[1].item() == pytest.approx(0.95, abs=1e-6)


def test_bce_label_smoothing_floor_zero_at_zero_smoothing():
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    floor = bce_label_smoothing_floor(target, 0.0)
    assert torch.equal(floor, torch.zeros_like(target))


def test_bce_label_smoothing_floor_symmetric_in_hard_label():
    """H(y') is symmetric in the hard label -- H(0.5*s) == H(1-0.5*s) -- so
    a positive and a negative row get the identical floor when pos_weight=1."""
    target = torch.tensor([0.0, 1.0])
    floor = bce_label_smoothing_floor(target, 0.2)
    assert floor[0].item() == pytest.approx(floor[1].item(), abs=1e-6)


def test_bce_label_smoothing_floor_hand_computed():
    target = torch.tensor([0.0])
    smoothing = 0.1
    floor = bce_label_smoothing_floor(target, smoothing)
    y_prime = 0.05
    expected = -(y_prime * math.log(y_prime) + (1 - y_prime) * math.log(1 - y_prime))
    assert floor[0].item() == pytest.approx(expected, abs=1e-6)


def test_bce_label_smoothing_floor_pos_weight_scales_positive_rows_only():
    target = torch.tensor([0.0, 1.0])
    floor_unweighted = bce_label_smoothing_floor(target, 0.1, pos_weight=1.0)
    floor_weighted = bce_label_smoothing_floor(target, 0.1, pos_weight=3.0)
    assert floor_weighted[0].item() == pytest.approx(floor_unweighted[0].item(), abs=1e-6)
    assert floor_weighted[1].item() == pytest.approx(3.0 * floor_unweighted[1].item(), abs=1e-6)


def test_expected_bce_floor_zero_at_zero_smoothing():
    assert expected_bce_floor(0.0) == 0.0
    assert expected_bce_floor(0.0, pos_weight=2.0) == 0.0


def test_expected_bce_floor_matches_unweighted_row_floor():
    smoothing = 0.1
    row_floor = bce_label_smoothing_floor(torch.tensor([0.0]), smoothing)[0].item()
    assert expected_bce_floor(smoothing) == pytest.approx(row_floor, abs=1e-6)


def test_expected_bce_floor_matches_dataset_level_pos_weight_expectation():
    """expected_bce_floor(smoothing, pos_weight) should equal the actual
    per-row-mean floor over a batch whose positive fraction matches what
    pos_weight=n_neg/n_pos implies -- i.e. it's a real approximation of the
    batch-mean floor bce_label_smoothing_floor computes directly, not an
    unrelated formula."""
    smoothing = 0.15
    pos_weight = 4.0  # implies frac_pos = 1/(1+pos_weight) = 0.2
    n_pos, n_neg = 20, 80
    target = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
    row_floor = bce_label_smoothing_floor(target, smoothing, pos_weight=pos_weight)
    batch_mean = float(row_floor.mean().item())
    assert expected_bce_floor(smoothing, pos_weight) == pytest.approx(batch_mean, abs=1e-6)
