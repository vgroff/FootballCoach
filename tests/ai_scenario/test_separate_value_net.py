"""Tests for PPOTrainer's permanent --separate-value-net mode.

See ai_trainer_knowledge.md / training_runs.log discussion: unlike the
throwaway --experiment-separate-value-net diagnostic (see
tests/ai_scenario/test_pretrain_combined_smoke.py and pretrain_value()'s
``experiment_separate_value_net`` docstring), ``separate_value_net=True`` is
a real, permanent architecture switch — a fully independent ExecutionNetwork
used as the ONLY critic for the entire training run (BC pre-training, value
warm-up, and PPO), never touched by BC gradients.

These tests guard:
  1. Construction: value_net/value_net_optimizer exist iff separate_value_net=True.
  2. Isolation: execution_net.value_head/value_ai_type_channel are frozen and
     never trained; value_net is a completely separate object (no weight sharing).
  3. GAE bootstrap (_get_value) and _sample_action route through value_net.
  4. One PPO update step trains value_net's params (and does NOT train
     execution_net.value_head's params).
  5. Checkpoint round-trip: value_net/value_net_optimizer persist and reload,
     and load_for_inference() auto-detects separate_value_net from the file.
  6. pretrain_value() runs to completion (not just one minibatch) with
     separate_value_net=True without crashing -- regression test for an
     UnboundLocalError where the trunk-unfreeze loop at the end of
     pretrain_value() referenced _freeze_params even when that branch was
     skipped (separate_value_net has nothing to freeze/unfreeze).
"""
from __future__ import annotations

import math

import pytest
import torch

from footballcoach.ai.action.schema import DecisionHeadsRaw
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import (
    PPOTrainer,
    _action_to_numpy,
    _scale_decision_heads_grad,
)
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS = 20


def _make_decision_heads_with(**overrides) -> DecisionHeadsRaw:
    """Minimal DecisionHeadsRaw for unit-testing _scale_decision_heads_grad
    in isolation, without a real DecisionNetwork forward pass. Every field
    not in ``overrides`` is a requires_grad=False zero tensor of a plausible
    shape -- irrelevant to the tests that use this (they only care about
    whichever field they pass in), and _scale_decision_heads_grad already
    skips cloning/hooking any field with requires_grad=False."""
    batch = 1
    max_other = 3
    defaults = dict(
        shoot_logit=torch.zeros(batch, 1),
        pass_logit=torch.zeros(batch, 1),
        move_logit=torch.zeros(batch, 1),
        tackle_logit=torch.zeros(batch, 1),
        get_possession_raw=torch.zeros(batch, 1),
        mark_logit=torch.zeros(batch, 1),
        hold_position_logit=torch.zeros(batch, 1),
        pass_target_logits=torch.zeros(batch, max_other),
        tackle_target_logits=torch.zeros(batch, max_other),
        mark_target_logits=torch.zeros(batch, max_other),
        move_region_center=torch.zeros(batch, 2),
        move_region_size=torch.zeros(batch, 1),
        move_arrival_speed=torch.zeros(batch, 1),
        region_of_play_center=torch.zeros(batch, 2),
        region_of_play_size=torch.zeros(batch, 1),
        attack_defence_raw=torch.zeros(batch, 1),
        latent_vector=torch.zeros(batch, 8),
        value=torch.zeros(batch, 1),
    )
    defaults.update(overrides)
    return DecisionHeadsRaw(**defaults)


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="smoke_1v1_sepvalue",
        label="Smoke: 1v1 (separate value net)",
        description="Smoke-test 1v1 environment for separate_value_net",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
    )


def _collect_rollout(env: ScenarioEnv, trainer: PPOTrainer, n: int):
    buffer = RolloutBuffer()
    env.sample_action_fn = trainer._sample_action
    env.reset()
    last_obs = None
    for _ in range(n):
        next_obs, reward, done, _info = env.step()
        tr = env.last_trainee_transition
        if tr is not None:
            buffer.add(
                obs=tr["obs"],
                action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=tr["log_prob"],
                value=tr["value"],
                reward=reward,
                done=1.0 if done else 0.0,
            )
            last_obs = next_obs
        if done:
            env.reset()
    if last_obs is None:
        last_obs = next_obs
    return buffer, last_obs


def _run_ppo_update(trainer: PPOTrainer, buffer: RolloutBuffer, last_obs):
    with torch.no_grad():
        last_obs_dict = {
            k: v.unsqueeze(0).to(trainer.device)
            for k, v in last_obs.to_torch_dict().items()
        }
        last_value = trainer._get_value(last_obs_dict)
    advantages, returns = buffer.compute_gae(trainer.gamma, trainer.lam, last_value)
    batch = buffer.as_tensors(advantages, returns)
    return trainer._ppo_update(batch, progress=0.0)


class TestConstruction:
    # Class-scoped: these 4 tests are all pure read-only assertions on a
    # freshly-constructed trainer (attribute/id/requires_grad checks) --
    # none of them train, step an optimizer, or otherwise mutate the
    # trainer, so building each config's PPOTrainer (a real, if now-
    # shrunk-for-tests, DecisionNetwork+ExecutionNetwork[+value_net])
    # exactly ONCE per class and sharing it across the 4 tests is safe,
    # instead of paying full construction cost 4 times over.
    @pytest.fixture(scope="class")
    def trainer_default(self) -> PPOTrainer:
        return PPOTrainer.from_config()

    @pytest.fixture(scope="class")
    def trainer_sep(self) -> PPOTrainer:
        return PPOTrainer.from_config(separate_value_net=True)

    def test_disabled_by_default(self, trainer_default):
        trainer = trainer_default
        assert trainer.separate_value_net is False
        assert trainer.value_net is None
        assert trainer.value_net_optimizer is None

    def test_enabled_constructs_independent_network(self, trainer_sep):
        trainer = trainer_sep
        assert trainer.separate_value_net is True
        assert trainer.value_net is not None
        assert trainer.value_net_optimizer is not None
        # Zero weight sharing with execution_net.
        assert trainer.value_net is not trainer.execution_net
        exec_ids = {id(p) for p in trainer.execution_net.parameters()}
        value_ids = {id(p) for p in trainer.value_net.parameters()}
        assert exec_ids.isdisjoint(value_ids)

    def test_execution_net_value_head_frozen_when_enabled(self, trainer_sep):
        trainer = trainer_sep
        for p in trainer.execution_net.value_head.parameters():
            assert p.requires_grad is False
        for p in trainer.execution_net.value_ai_type_channel.parameters():
            assert p.requires_grad is False
        # value_net itself IS trainable.
        assert all(p.requires_grad for p in trainer.value_net.parameters())

    def test_main_optimizer_excludes_value_net_params(self, trainer_sep):
        trainer = trainer_sep
        main_opt_param_ids = {
            id(p) for pg in trainer.optimizer.param_groups for p in pg["params"]
        }
        value_net_ids = {id(p) for p in trainer.value_net.parameters()}
        assert main_opt_param_ids.isdisjoint(value_net_ids)

    def test_value_net_trunk_hidden_override(self):
        """network.value_net_trunk_hidden, when set, must size ONLY
        value_net's trunk -- execution_net's trunk must stay at whatever
        exec_trunk_hidden (falling back to trunk_hidden) already resolves to."""
        import copy
        from footballcoach.ai.config import load_ai_config
        from footballcoach.ai.models.decision_network import DecisionNetwork
        from footballcoach.ai.models.execution_network import ExecutionNetwork

        cfg = copy.deepcopy(load_ai_config())
        # execution_net's own trunk sizing: exec_trunk_hidden falls back to
        # trunk_hidden when absent/null -- see ExecutionNetwork.from_config().
        expected_exec_trunk_hidden = cfg["network"].get("exec_trunk_hidden") or cfg["network"]["trunk_hidden"]
        override_trunk_hidden = expected_exec_trunk_hidden + 37  # deliberately different
        cfg["network"]["value_net_trunk_hidden"] = override_trunk_hidden

        trainer = PPOTrainer(
            decision_net=DecisionNetwork.from_config(),
            execution_net=ExecutionNetwork.from_config(),
            cfg=cfg,
            separate_value_net=True,
        )
        assert trainer.execution_net.trunk[-2].out_features == expected_exec_trunk_hidden
        assert trainer.value_net.trunk[-2].out_features == override_trunk_hidden

    def test_no_effect_when_separate_value_net_disabled(self):
        """value_net_trunk_hidden must have zero effect on execution_net's
        trunk when --separate-value-net is not passed."""
        import copy
        from footballcoach.ai.config import load_ai_config
        from footballcoach.ai.models.decision_network import DecisionNetwork
        from footballcoach.ai.models.execution_network import ExecutionNetwork

        cfg = copy.deepcopy(load_ai_config())
        expected_exec_trunk_hidden = cfg["network"].get("exec_trunk_hidden") or cfg["network"]["trunk_hidden"]
        cfg["network"]["value_net_trunk_hidden"] = expected_exec_trunk_hidden + 37

        trainer = PPOTrainer(
            decision_net=DecisionNetwork.from_config(),
            execution_net=ExecutionNetwork.from_config(),
            cfg=cfg,
            separate_value_net=False,
        )
        assert trainer.value_net is None
        assert trainer.execution_net.trunk[-2].out_features == expected_exec_trunk_hidden


class TestValueRouting:
    def test_get_value_uses_value_net_when_enabled(self):
        """_get_value() must differ between the two modes for the same obs,
        since value_net is randomly initialised independently of
        execution_net.value_head (extremely unlikely to coincide)."""
        env = _make_env()
        torch.manual_seed(0)
        trainer_shared = PPOTrainer.from_config(separate_value_net=False)
        torch.manual_seed(0)
        trainer_sep = PPOTrainer.from_config(separate_value_net=True)

        obs = env.reset()
        obs_dict = {k: v.unsqueeze(0) for k, v in obs.to_torch_dict().items()}
        v_shared = trainer_shared._get_value(obs_dict)
        v_sep = trainer_sep._get_value(obs_dict)
        assert math.isfinite(v_shared)
        assert math.isfinite(v_sep)
        # Same seed for decision_net/execution_net init, but value_net is an
        # extra independently-initialised network in the sep case -- the
        # value estimate should not coincide.
        assert v_shared != v_sep

    def test_sample_action_value_matches_get_value_path(self):
        """When separate_value_net is enabled, _sample_action()'s returned
        value must come from value_net, not execution_net.value_head."""
        env = _make_env()
        trainer = PPOTrainer.from_config(separate_value_net=True)
        obs = env.reset()
        obs_dict = obs.to_torch_dict()
        result = trainer._sample_action(obs_dict)
        _action, _log_prob, value, *_ = result
        assert math.isfinite(value)

        # Independently recompute value via value_net forward and compare.
        with torch.no_grad():
            batched = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
            sf, of, em = batched["self_feat"], batched["other_feat"], batched["exists_mask"]
            bf, gf = batched["ball_feat"], batched["global_feat"]
            sat = batched.get("self_ai_type")
            oat = batched.get("other_ai_type")
            d_heads = trainer.decision_net(sf, of, em, bf, gf, sat, oat)
            expected = float(trainer.value_net(sf, of, em, bf, gf, d_heads, sat, oat).value.mean())
        assert math.isclose(value, expected, rel_tol=1e-4, abs_tol=1e-5)


class TestPPOUpdateTrainsValueNet:
    def test_ppo_update_changes_value_net_params_not_execution_value_head(self):
        env = _make_env()
        trainer = PPOTrainer.from_config(separate_value_net=True)

        exec_value_head_before = [
            p.detach().clone() for p in trainer.execution_net.value_head.parameters()
        ]
        value_net_before = [p.detach().clone() for p in trainer.value_net.parameters()]

        buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
        metrics = _run_ppo_update(trainer, buffer, last_obs)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert math.isfinite(val), f"metrics['{key}']={val} not finite"

        exec_value_head_after = list(trainer.execution_net.value_head.parameters())
        value_net_after = list(trainer.value_net.parameters())

        # execution_net.value_head must be completely untouched (frozen + unused).
        for before, after in zip(exec_value_head_before, exec_value_head_after):
            assert torch.equal(before, after), (
                "execution_net.value_head changed even though it should be frozen "
                "and unused when separate_value_net=True"
            )
        # value_net's params should have moved (it just received a gradient step).
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(value_net_before, value_net_after)
        )
        assert changed, "value_net params did not change after a PPO update"


class TestPretrainValueCompletesWithSeparateValueNet:
    def test_pretrain_value_runs_to_completion(self):
        """Regression test: pretrain_value() must run its full loop (collect
        rollout, N epochs, trailing freeze/unfreeze bookkeeping) without
        crashing when separate_value_net=True. Previously crashed with
        UnboundLocalError on the trailing `for p in _freeze_params:` unfreeze
        loop, which assumed _freeze_params was always assigned even though it
        is only computed in the non-separate_value_net branch."""
        env = _make_env()
        trainer = PPOTrainer.from_config(separate_value_net=True)
        result = trainer.pretrain_value(
            env, n_steps=_ROLLOUT_STEPS, n_epochs=2, lr=1e-3, batch_size=8,
        )
        assert "episode_returns" in result


class TestCheckpointRoundTrip:
    def test_save_and_load_checkpoint_restores_value_net(self, tmp_path):
        trainer = PPOTrainer.from_config(
            separate_value_net=True, checkpoint_dir=tmp_path,
        )
        trainer._save_checkpoint(step=123)
        ckpt_path = tmp_path / "checkpoint1.pt"
        assert ckpt_path.exists()

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "value_net" in raw
        assert "value_net_optimizer" in raw

        trainer2 = PPOTrainer.from_config(separate_value_net=True)
        # Perturb value_net so we can tell load_checkpoint actually restored it.
        with torch.no_grad():
            for p in trainer2.value_net.parameters():
                p.add_(1.0)
        trainer2.load_checkpoint(ckpt_path)

        for p1, p2 in zip(trainer.value_net.parameters(), trainer2.value_net.parameters()):
            assert torch.equal(p1, p2)

    def test_load_for_inference_autodetects_separate_value_net(self, tmp_path):
        trainer = PPOTrainer.from_config(
            separate_value_net=True, checkpoint_dir=tmp_path,
        )
        trainer._save_checkpoint(step=1)
        ckpt_path = tmp_path / "checkpoint1.pt"

        loaded = PPOTrainer.load_for_inference(ckpt_path)
        assert loaded.separate_value_net is True
        assert loaded.value_net is not None

    def test_load_for_inference_shared_trunk_checkpoint_stays_disabled(self, tmp_path):
        trainer = PPOTrainer.from_config(
            separate_value_net=False, checkpoint_dir=tmp_path,
        )
        trainer._save_checkpoint(step=1)
        ckpt_path = tmp_path / "checkpoint1.pt"

        loaded = PPOTrainer.load_for_inference(ckpt_path)
        assert loaded.separate_value_net is False
        assert loaded.value_net is None


class TestShareValueGradWithDecision:
    """Coverage for --share-value-grad-with-decision / decision_value_coef
    (see ai_trainer_knowledge.md "Separate value network" for the full
    rationale). This is an opt-in, controlled exception to the isolation
    TestPPOUpdateTrainsValueNet above guards by default."""

    def test_disabled_by_default(self):
        trainer = PPOTrainer.from_config(separate_value_net=True)
        assert trainer.share_value_grad_with_decision is False

    def test_forced_off_without_separate_value_net(self):
        """Flag has no meaning outside separate_value_net mode -- must be
        coerced to False rather than silently doing nothing at some other
        call site."""
        trainer = PPOTrainer.from_config(
            separate_value_net=False, share_value_grad_with_decision=True,
        )
        assert trainer.share_value_grad_with_decision is False

    def test_enabled_when_both_set(self):
        trainer = PPOTrainer.from_config(
            separate_value_net=True, share_value_grad_with_decision=True,
            decision_value_coef=0.3,
        )
        assert trainer.share_value_grad_with_decision is True
        assert trainer.decision_value_coef == pytest.approx(0.3)

    def test_coef_one_is_identity_no_clone(self):
        d_heads = _make_decision_heads_with(
            shoot_logit=torch.randn(1, 1, requires_grad=True)
        )
        scaled = _scale_decision_heads_grad(d_heads, coef=1.0)
        assert scaled is d_heads

    def test_scale_decision_heads_grad_only_scales_the_leaked_path(self):
        """Core correctness property: coef must scale ONLY the gradient
        contribution flowing through the returned (cloned) tensor, leaving
        the ORIGINAL d_heads tensor's own gradient -- as used by
        execution_net's forward for the policy's own output, at the real
        _ppo_update() call site -- completely unscaled. If this were
        implemented by hooking the original tensor instead of a clone, this
        test would fail (the policy's own contribution would get scaled
        too)."""
        x = torch.randn(4, requires_grad=True)
        d_heads = _make_decision_heads_with(shoot_logit=x.view(1, 4))
        coef = 0.25
        scaled = _scale_decision_heads_grad(d_heads, coef)
        assert scaled.shoot_logit is not d_heads.shoot_logit

        # Simulates the real call site: the SAME d_heads feeds execution_net
        # (policy_loss, unscaled) AND self.value_net (value_loss, via the
        # coef-scaled clone) in one shared backward.
        policy_loss = (d_heads.shoot_logit ** 2).sum()
        value_loss = (scaled.shoot_logit ** 2).sum()
        (policy_loss + value_loss).backward()

        expected_grad = (2 * x.detach()) * (1.0 + coef)
        assert torch.allclose(x.grad, expected_grad, atol=1e-5)

    def test_ppo_update_decision_net_changes_more_when_enabled(self):
        """Integration check: given IDENTICAL starting weights, an identical
        rollout, and an identical minibatch-shuffle seed, decision_net's
        post-update parameters must differ between
        share_value_grad_with_decision=False (value gradient fully blocked
        from decision_net) and =True with a non-trivial coef -- proving the
        leaked gradient actually reaches self.optimizer.step() in a real
        _ppo_update() call, not just in the isolated helper test above."""
        env = _make_env()
        torch.manual_seed(0)
        trainer_off = PPOTrainer.from_config(
            separate_value_net=True, share_value_grad_with_decision=False,
        )
        torch.manual_seed(0)
        trainer_on = PPOTrainer.from_config(
            separate_value_net=True, share_value_grad_with_decision=True,
            decision_value_coef=1.0,
        )
        # Sanity-check the identical-seed assumption before relying on it.
        p_off0 = next(trainer_off.decision_net.parameters()).detach().clone()
        p_on0 = next(trainer_on.decision_net.parameters()).detach().clone()
        assert torch.equal(p_off0, p_on0)

        buffer, last_obs = _collect_rollout(env, trainer_on, _ROLLOUT_STEPS)

        torch.manual_seed(123)
        _run_ppo_update(trainer_off, buffer, last_obs)
        torch.manual_seed(123)
        _run_ppo_update(trainer_on, buffer, last_obs)

        changed = any(
            not torch.equal(a, b)
            for a, b in zip(
                trainer_off.decision_net.parameters(),
                trainer_on.decision_net.parameters(),
            )
        )
        assert changed, (
            "decision_net's parameters after one _ppo_update() were identical "
            "between share_value_grad_with_decision=False/True -- the leaked "
            "value gradient did not reach decision_net's optimizer step"
        )

    def test_execution_net_trunk_unaffected_when_enabled(self):
        """share_value_grad_with_decision must never let value_net's
        gradient reach execution_net -- value_net has zero weight sharing
        with it regardless of this flag (only decision_net is shared)."""
        env = _make_env()
        trainer = PPOTrainer.from_config(
            separate_value_net=True, share_value_grad_with_decision=True,
            decision_value_coef=1.0,
        )
        exec_before = [p.detach().clone() for p in trainer.execution_net.value_head.parameters()]

        buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
        _run_ppo_update(trainer, buffer, last_obs)

        for before, after in zip(exec_before, trainer.execution_net.value_head.parameters()):
            assert torch.equal(before, after), (
                "execution_net.value_head changed even though "
                "share_value_grad_with_decision must only affect decision_net"
            )
