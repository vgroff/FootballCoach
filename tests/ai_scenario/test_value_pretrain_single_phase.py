"""Value pre-training is single-phase (phase 1) only for now: it refuses a
phase mix and role-based phases instead of silently warming the critic on
the wrong data (PretrainingMixin._require_single_phase)."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    return PPOTrainer.from_config(device=torch.device("cpu"))


@pytest.mark.parametrize("method", ["pretrain_value", "pretrain_combined"])
def test_a_phase_mix_is_refused(trainer, method):
    trainer.phase_mix = {1: 3.0, 2: 1.0}
    kwargs = dict(n_steps=10, n_epochs=1, lr=1e-3) if method == "pretrain_value" else dict(
        dataset=None, n_epochs=1, batch_size=8, bc_lr=1e-3, value_lr=1e-3, rollout_steps=10)
    with pytest.raises(NotImplementedError, match="phase mix"):
        getattr(trainer, method)(None, phase_id=1, **kwargs)


def test_a_role_based_phase_is_refused(trainer):
    with pytest.raises(NotImplementedError, match="role-based"):
        trainer.pretrain_value(None, n_steps=10, n_epochs=1, lr=1e-3, phase_id=2)


def test_phase_1_alone_passes_the_guard(trainer):
    trainer._require_single_phase("x", 1)
    trainer.phase_mix = {1: 1.0}
    trainer._require_single_phase("x", 1)
