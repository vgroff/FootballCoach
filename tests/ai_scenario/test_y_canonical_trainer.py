"""PPOTrainer integration for the opt-in y-canonical frame (ppo.y_canonical / ai/obs/y_canonical.py)."""
from __future__ import annotations

import functools
import math
import random

import pytest
import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.obs.augment import augment_batch
from footballcoach.ai.obs.y_canonical import YCanonicalNetworkWrapper
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _ai_types
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

_PHASE = PHASES_BY_ID[1]


def _make_env(trainer) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="y_canonical_test", label="y canonical test", description="1v1 short episodes",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0),
        on_tick=phase1_training_on_tick,
    )
    env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=2.0)
    env.sample_action_fn = trainer._sample_action
    return env


def _trainer(y: bool, shuffles: int = 0) -> PPOTrainer:
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False, y_canonical=y)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    t.augment_n_slot_shuffles = shuffles
    return t


@pytest.fixture()
def trainer():
    return _trainer(True, shuffles=1)


def test_flag_wraps_the_networks_and_leaves_checkpoint_keys_unchanged():
    plain, y = _trainer(False), _trainer(True)
    assert not isinstance(plain.decision_net, YCanonicalNetworkWrapper)
    assert isinstance(y.decision_net, YCanonicalNetworkWrapper) and isinstance(y.execution_net, YCanonicalNetworkWrapper)
    assert set(plain.decision_net.state_dict()) == set(y.decision_net.state_dict())
    assert set(plain.execution_net.state_dict()) == set(y.execution_net.state_dict())


def test_default_follows_the_config_and_the_kwarg_overrides_it():
    from footballcoach.ai.config import load_ai_config

    configured = bool(load_ai_config()["ppo"].get("y_canonical", False))
    assert PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False).y_canonical is configured
    assert _trainer(True).y_canonical is True
    assert _trainer(False).y_canonical is False


def test_sampling_and_rollout_collection_work_under_the_wrapper(trainer):
    batch, stats = trainer._collect_value_pretrain_rollout(_make_env(trainer), 300, phase_id=None)
    assert len(batch["returns"]) > 0 and torch.isfinite(batch["returns"]).all()


def test_flip_copies_have_identical_log_probs_and_zero_flip_kl(trainer):
    batch, _ = trainer._collect_value_pretrain_rollout(_make_env(trainer), 400, phase_id=None)
    n = len(batch["returns"])
    aug = augment_batch(batch, 1, random.Random(0))
    aug.update(trainer._precompute_physics_full(aug, 256))
    pair: dict = {}
    lp, _ = trainer._recompute_old_log_probs_for_augmented_batch(aug, pair_out=pair)
    assert torch.allclose(lp[:n], lp[n:2 * n], atol=1e-3), float((lp[:n] - lp[n:2 * n]).abs().max())
    stats = trainer._flip_consistency_stats(aug, pair)
    assert stats["kl"] < 1e-4
    assert stats["metrics"]["exec_move_sign_disagree_pct"] == 0.0
    assert stats["metrics"]["move_dir_err_p90_deg"] < 0.1


def test_ppo_update_runs_with_augmentation_off():
    t = _trainer(True, shuffles=0)
    batch, _ = t._collect_value_pretrain_rollout(_make_env(t), 500, phase_id=None, use_gae=True)
    metrics = t._ppo_update(batch, progress=0.0)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            assert math.isfinite(v), k
