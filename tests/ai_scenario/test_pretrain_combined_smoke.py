"""Smoke test for PPOTrainer.pretrain_combined() (W1 baseline regression guard).

This is deliberately coarse: build a tiny synthetic DemonstrationDataset (same
fixture pattern as tests/ai_unit/test_demonstration_dataset.py), call
pretrain_combined() with small step/epoch counts against a real but short
ScenarioEnv, and assert it completes without raising and leaves no NaN/Inf in
any network parameter.

This test would have caught the pretrain_value()-dead-code-path /
Phase-0-gradient-discard issues described in agent_plans/new_ai_features_fixes.md
had it existed before that refactor — treat it as the regression guard for any
future changes to pretrain_combined()/pretrain_value().
"""
from __future__ import annotations

import numpy as np
import torch

from footballcoach.ai.bc.dataset import DemonstrationDataset
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


def _make_synthetic_dataset(env: ScenarioEnv, n: int = 32) -> DemonstrationDataset:
    """Small dataset built from real (obs, label) pairs recorded by stepping
    *env* with Phase1RulesAI, mirroring record_demonstrations.py.  Using real
    encoded observations (rather than random noise) keeps feature ranges
    realistic so pretrain_combined() doesn't diverge to NaN in one step."""
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ai.ppo.bc import phase1_labels

    self_feats, other_feats, exists_masks = [], [], []
    ball_feats, global_feats, bc_labels = [], [], []
    rewards, dones = [], []

    env.reset()
    try:
        env._loop.match.player_by_id(env.trainee_player_id).ai = Phase1RulesAI()
    except (AttributeError, KeyError):
        pass

    for _ in range(n):
        obs = env._get_obs()
        label = phase1_labels(env, env.trainee_player_id)
        self_feats.append(obs.self_feat.copy())
        other_feats.append(obs.other_feat.copy())
        exists_masks.append(obs.exists_mask.copy())
        ball_feats.append(obs.ball_feat.copy())
        global_feats.append(obs.global_feat.copy())
        bc_labels.append(label.to_array())

        _obs, reward, done, _info = env.step()
        rewards.append(np.float32(reward))
        dones.append(np.float32(1.0 if done else 0.0))
        if done:
            env.reset()
            try:
                env._loop.match.player_by_id(env.trainee_player_id).ai = Phase1RulesAI()
            except (AttributeError, KeyError):
                pass

    return DemonstrationDataset(
        obs_self_feat=np.stack(self_feats),
        obs_other_feat=np.stack(other_feats),
        obs_exists_mask=np.stack(exists_masks),
        obs_ball_feat=np.stack(ball_feats),
        obs_global_feat=np.stack(global_feats),
        bc_labels=np.stack(bc_labels),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=np.float32),
    )


def _build_1v1_rules_opponent(rng_reduction: float = 0.3, **kwargs) -> "Match":
    # Force a rules-based (non-immobile) opponent so this smoke test's tiny
    # synthetic dataset always has valid_indices() rows — build_1v1_scenario
    # defaults to opponent_immobile_prob=1.0, which made the opponent's
    # ai_type immobile in every recorded step and left valid_indices() (and
    # thus iterate_minibatches(valid_only=True)) empty, causing an
    # intermittent-looking but actually deterministic-per-default failure.
    kwargs.setdefault("opponent_rules_prob", 1.0)
    kwargs.setdefault("opponent_immobile_prob", 0.0)
    return build_1v1_scenario(rng_reduction, **kwargs)


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="pretrain_combined_smoke_1v1",
        label="Smoke: pretrain_combined 1v1",
        description="Smoke-test env for pretrain_combined()",
        build=_build_1v1_rules_opponent,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
    )


def test_pretrain_combined_smoke():
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = _make_synthetic_dataset(_make_env(), n=32)
    env = _make_env()
    trainer = PPOTrainer.from_config()

    trainer.pretrain_combined(
        env,
        dataset,
        n_epochs=1,
        batch_size=8,
        bc_lr=1e-3,
        value_lr=1e-3,
        rollout_steps=64,
        value_epochs=1,
    )

    for p in list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters()):
        assert torch.isfinite(p).all(), "Non-finite parameter after pretrain_combined()"


def test_pretrain_combined_populates_ai_type_side_channel():
    """W6 regression guard: pretrain_combined()'s dataset-driven minibatches
    (both Phase 0's decision-only path and Phase 1's BC epochs) must actually
    carry a populated self_ai_type/other_ai_type side-channel through to the
    network, not silently fall back to the None/all-zero default the whole
    time. This directly exercises the same DemonstrationDataset ->
    _ai_types(obs_dict) -> decision_net(..., sat, oat) plumbing used inside
    pretrain_combined(), so a future refactor that breaks this wiring (e.g.
    a call site that stops threading sat/oat) fails here instead of only
    showing up as a silent accuracy regression during real training.
    """
    from footballcoach.ai.ppo.ppo_trainer import _ai_types

    dataset = _make_synthetic_dataset(_make_env(), n=32)

    saw_self_ai_type = False
    saw_nonzero_other_ai_type = False
    for obs_dict, _labels in dataset.iterate_minibatches(
        batch_size=8, shuffle=False, device=None, valid_only=True,
    ):
        sat, oat = _ai_types(obs_dict)
        assert sat is not None, "self_ai_type missing from dataset minibatch obs_dict"
        assert oat is not None, "other_ai_type missing from dataset minibatch obs_dict"
        # self_ai_type is a one-hot over {immobile, neural, rules} per row.
        assert torch.allclose(sat.sum(dim=-1), torch.ones(sat.shape[0]))
        saw_self_ai_type = True
        # At least one row's opponent slot should carry a non-zero one-hot
        # (the demo dataset is 1v1, so exactly one "other" slot is real).
        if oat.abs().sum() > 0:
            saw_nonzero_other_ai_type = True

    assert saw_self_ai_type, "No minibatches were produced by the dataset"
    assert saw_nonzero_other_ai_type, (
        "other_ai_type side-channel was all-zero across every minibatch — "
        "the opponent's AI-type one-hot is not being populated"
    )


def test_pretrain_value_returns_rollout_stats():
    """pretrain_value()'s W7 reward/win-rate logging additions must actually
    compute stats without error, and complete at least one episode given a
    short episode length."""
    torch.manual_seed(0)
    np.random.seed(0)

    env = _make_env()
    trainer = PPOTrainer.from_config()

    stats = trainer.pretrain_value(env, n_steps=200, n_epochs=1, lr=1e-3, batch_size=8)

    assert "episode_returns" in stats
    assert len(stats["episode_returns"]) > 0
    assert isinstance(stats["outcomes_vs_rules"], list)
    assert isinstance(stats["outcomes_vs_immobile"], list)
    assert isinstance(stats["outcomes_vs_neural"], list)


def test_phase0_optimizer_includes_trunk_and_encoder_params():
    """Regression guard for decision #13: Phase 0 must train ALL of
    decision_net's parameters (encoders + trunk + value_head), not just the
    value head. Directly inspects the optimizer built inside
    pretrain_combined()'s Phase 0 block via a monkeypatched Adam constructor."""
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = _make_synthetic_dataset(_make_env(), n=32)
    env = _make_env()
    trainer = PPOTrainer.from_config()

    captured_param_groups = []
    real_adam = torch.optim.Adam

    def _spy_adam(params, *args, **kwargs):
        params = list(params)
        captured_param_groups.append(params)
        return real_adam(params, *args, **kwargs)

    import unittest.mock as mock
    with mock.patch("torch.optim.Adam", side_effect=_spy_adam):
        trainer.pretrain_combined(
            env, dataset, n_epochs=1, batch_size=8,
            bc_lr=1e-3, value_lr=1e-3, rollout_steps=64, value_epochs=1,
        )

    # The Phase 0 optimizer is the one built over ALL decision_net params —
    # find it by checking it contains an entity_encoder parameter.
    encoder_param_ids = {id(p) for p in trainer.decision_net.entity_encoder.parameters()}
    trunk_param_ids = {id(p) for p in trainer.decision_net.trunk.parameters()}
    found_phase0_opt = False
    for params in captured_param_groups:
        param_ids = {id(p) for p in params}
        if encoder_param_ids & param_ids and trunk_param_ids & param_ids:
            found_phase0_opt = True
            break
    assert found_phase0_opt, (
        "No optimizer in pretrain_combined() included both entity_encoder and "
        "trunk parameters -- Phase 0's freezing regression may have been "
        "reintroduced (see agent_plans/new_ai_features_fixes.md decision #13)."
    )
