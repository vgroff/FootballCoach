"""bc.ppg_mc_returns: ppg_value_refit asks its rollout collector for pure Monte
Carlo targets (use_gae=False) when set, and for GAE targets otherwise."""
from __future__ import annotations

import pytest
import torch


class _Stop(Exception):
    pass


@pytest.fixture(scope="module")
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    return PPOTrainer.from_config(device=torch.device("cpu"))


@pytest.mark.parametrize("mc,want_gae", [(True, False), (False, True)])
def test_refit_targets_follow_the_flag(trainer, monkeypatch, mc, want_gae):
    seen = {}

    def _collect(env, n_steps, phase_id, use_gae=False, **kw):
        seen["use_gae"] = use_gae
        raise _Stop

    monkeypatch.setattr(trainer, "separate_value_net", False)
    monkeypatch.setattr(trainer, "_ppg_mc_returns", mc)
    monkeypatch.setattr(trainer, "_spawn_value_pretrain_workers", lambda phase_id: None)
    monkeypatch.setattr(trainer, "_collect_value_pretrain_rollout", _collect)
    with pytest.raises(_Stop):
        trainer.ppg_value_refit(None, n_steps=100, phase_id=2, num_rollouts=1)
    assert seen["use_gae"] is want_gae


def test_flag_is_off_by_default():
    from footballcoach.ai.config import load_ai_config
    assert load_ai_config()["bc"]["ppg_mc_returns"] is False
