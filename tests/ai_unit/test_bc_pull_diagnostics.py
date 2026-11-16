"""BC diagnostics in the PPO update: _kick_agreement (the policy vs its BC
expert on the kick group) and _measure_bc_pull (BC's weighted gradient vs
everything else's, and their cosine)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from footballcoach.ai.ppo.bc import (
    BC_LABEL_DIM, _I_KICK_DIR_X, _I_KICK_DIR_Y, _I_KICK_DIR_Z, _I_KICK_POWER, _I_KICK_THIS_TICK, _I_VALID,
    _kick_power_physical,
)
from footballcoach.ai.ppo.ppo_update import _AGREE_KEYS, _UpdateStats, _kick_agreement


def test_kick_agreement_values():
    n = 4
    labels = torch.zeros(n, BC_LABEL_DIM)
    labels[:, _I_VALID] = 1.0
    labels[:2, _I_KICK_THIS_TICK] = 1.0                      # rows 0-1: "kick now"
    labels[:2, _I_KICK_POWER] = 0.9
    labels[:2, _I_KICK_DIR_X] = 1.0                          # +x
    kick_logit = torch.tensor([[0.0], [2.0], [-2.0], [0.0]])
    kick_power = torch.zeros(n, 1)                           # raw 0 -> a known physical fraction
    angle = math.radians(30.0)
    kick_direction = torch.tensor([[math.cos(angle), math.sin(angle), 0.0], [0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    heads = SimpleNamespace(kick_logit=kick_logit, kick_power=kick_power, kick_direction=kick_direction)
    rows = torch.tensor([True, True, True, False])           # row 3 excluded
    got = dict(zip(_AGREE_KEYS, _kick_agreement(labels, heads, rows)))
    sig = torch.sigmoid(kick_logit.reshape(-1))
    assert got["label_kick_rate"] == pytest.approx(2 / 3)
    assert got["p_kick_label_kick"] == pytest.approx(float(sig[:2].mean()))
    assert got["p_kick_label_no_kick"] == pytest.approx(float(sig[2]))
    assert got["power_err"] == pytest.approx(abs(float(_kick_power_physical(torch.zeros(1))) - 0.9))
    assert got["kick_dir_err_deg"] == pytest.approx((30.0 + 90.0) / 2, abs=1e-4)


def test_kick_agreement_with_no_kick_labels_is_nan_where_undefined():
    labels = torch.zeros(3, BC_LABEL_DIM)
    heads = SimpleNamespace(kick_logit=torch.zeros(3, 1), kick_power=torch.zeros(3, 1),
                            kick_direction=torch.tensor([[1.0, 0.0, 0.0]] * 3))
    got = dict(zip(_AGREE_KEYS, _kick_agreement(labels, heads, torch.ones(3, dtype=torch.bool))))
    assert got["label_kick_rate"] == 0.0 and got["p_kick_label_no_kick"] == pytest.approx(0.5)
    assert all(math.isnan(got[k]) for k in ("p_kick_label_kick", "power_err", "kick_dir_err_deg"))
    assert all(math.isnan(v) for v in _kick_agreement(labels, heads, torch.zeros(3, dtype=torch.bool)))


@pytest.fixture(scope="module")
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    return PPOTrainer.from_config(device=torch.device("cpu"))


def test_bc_pull_matches_hand_computed_gradients(trainer):
    """bc = 3 * sum(w^2) on one main-group weight, rest = sum(w) on it plus
    sum(d) on one direction weight: the norms and cosines are known exactly."""
    main, direction = trainer._main_and_direction_params()
    w = next(p for p in main if p.requires_grad)
    d = next(p for p in direction if p.requires_grad)
    bc = 3.0 * (w * w).sum()
    rest = w.sum() + d.sum()
    stats = _UpdateStats()
    trainer._measure_bc_pull(stats, rest, bc)
    bc_main, rest_main, cos_main, bc_dir, rest_dir, cos_dir = stats.bc_pull[0]
    g_bc = 6.0 * w.detach()
    assert bc_main == pytest.approx(float(g_bc.norm()), rel=1e-5)
    assert rest_main == pytest.approx(math.sqrt(w.numel()), rel=1e-5)
    assert cos_main == pytest.approx(float(g_bc.sum() / (g_bc.norm() * math.sqrt(w.numel()))), rel=1e-4)
    assert bc_dir == 0.0 and rest_dir == pytest.approx(math.sqrt(d.numel()), rel=1e-5)
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in (w, d))   # measurement only


def test_bc_pull_metrics_average_the_samples(trainer):
    stats = _UpdateStats()
    stats.bc_pull = [[1, 2, 0.5, 3, 4, -0.5], [3, 4, 0.1, 5, 6, 0.3]]
    m = trainer._bc_pull_metrics(stats)
    assert m == pytest.approx({"bc_main": 2, "rest_main": 3, "cos_main": 0.3, "bc_dir": 4, "rest_dir": 5, "cos_dir": -0.1})
    assert trainer._bc_pull_metrics(_UpdateStats()) == {}
