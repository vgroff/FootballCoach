"""Tests for opt-in dual-clip PPO (ppo.dual_clip_c / PPOTrainer._apply_dual_clip
/ __init__'s dual_clip_c comment for the full rationale).

Vanilla PPO's clip only bounds the surrogate objective for POSITIVE
advantage. For NEGATIVE advantage, if ratio blows up (the network
drastically increases the probability of an action its OWN advantage
estimate says was bad), the unclipped side (ratio*adv) keeps decreasing
without bound as ratio grows, and min(surr1, surr2) picks it the moment
ratio exceeds (1+clip_range) -- giving an UNBOUNDED positive policy_loss
contribution from that one row. Confirmed live in this codebase's
training_runs.md: recurring "[ratio spike]" log entries with negative adv
imply per-sample policy_loss in the hundreds to (rarely) tens of thousands.

Dual-clip adds a second floor, ONLY for adv<0:
    max(min(surr1, surr2), dual_clip_c * adv)
bounding that row's worst case at dual_clip_c*|adv| without touching the
already-fine positive-advantage path at all. dual_clip_c<=0 (default) is a
true no-op -- byte-identical to vanilla PPO.

Two layers of coverage:
  1. TestApplyDualClipDisabled/Enabled -- pure tensor-math unit tests of
     _apply_dual_clip() in isolation (no network/env needed).
  2. TestDualClipIntegration -- a real rollout batch with one row's stored
     old_log_prob artificially offset to force an extreme ratio, run
     through the REAL _ppo_update(), confirming the aggregate
     metrics['policy_loss'] is bounded when enabled and is NOT (huge, or
     literally inf/nan from float32 overflow) when disabled.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS = 40


@pytest.fixture(scope="module")
def trainer() -> PPOTrainer:
    return PPOTrainer.from_config()


class TestApplyDualClipDisabled:
    def test_zero_is_a_true_no_op(self, trainer):
        trainer.dual_clip_c = 0.0
        min_surr = torch.tensor([-100.0, 5.0, -0.001, 3000.0])
        adv = torch.tensor([-2.0, 1.0, -0.5, -10.0])
        out = trainer._apply_dual_clip(min_surr, adv)
        assert torch.equal(out, min_surr)

    def test_negative_c_is_also_a_no_op(self, trainer):
        """Defensive: a stray negative config value must not accidentally
        do something (e.g. flip a comparison) -- treated the same as 0."""
        trainer.dual_clip_c = -1.0
        min_surr = torch.tensor([-100.0, 5.0])
        adv = torch.tensor([-2.0, 1.0])
        out = trainer._apply_dual_clip(min_surr, adv)
        assert torch.equal(out, min_surr)

    def test_default_config_value_is_disabled(self, trainer):
        # Regression guard on the DEFAULT, not a claim about any specific
        # value -- from_config() picks up whatever ai_config.json's live
        # dual_clip_c currently is; it must stay off (0/falsy) by default
        # so vanilla PPO is the out-of-the-box behavior.
        default_trainer = PPOTrainer.from_config()
        assert default_trainer.dual_clip_c <= 0


class TestApplyDualClipEnabled:
    def test_positive_advantage_rows_are_never_touched(self, trainer):
        trainer.dual_clip_c = 3.0
        min_surr = torch.tensor([-5.0, 1000.0, 0.0])
        adv = torch.tensor([2.0, 5.0, 0.0])  # all >= 0
        out = trainer._apply_dual_clip(min_surr, adv)
        assert torch.equal(out, min_surr), (
            "dual-clip must never modify rows with adv >= 0 -- that side "
            "is already bounded by vanilla PPO's own clip"
        )

    def test_negative_advantage_row_below_floor_gets_clamped_up(self, trainer):
        trainer.dual_clip_c = 3.0
        # adv=-2 -> floor = 3*-2 = -6. min_surr=-500 is far below the floor
        # -- exactly the unbounded-blowup case (huge ratio collapsed
        # min(surr1,surr2) to a hugely negative unclipped surr1).
        min_surr = torch.tensor([-500.0])
        adv = torch.tensor([-2.0])
        out = trainer._apply_dual_clip(min_surr, adv)
        assert out.item() == pytest.approx(-6.0)
        # -> policy_loss contribution -out = 6.0 = dual_clip_c*|adv|, not 500.

    def test_negative_advantage_row_above_floor_is_unchanged(self, trainer):
        trainer.dual_clip_c = 3.0
        # adv=-2 -> floor=-6. min_surr=-1 is ABOVE (less negative than) the
        # floor -- the normal, already-well-behaved clipped case; dual-clip
        # must leave it alone.
        min_surr = torch.tensor([-1.0])
        adv = torch.tensor([-2.0])
        out = trainer._apply_dual_clip(min_surr, adv)
        assert out.item() == pytest.approx(-1.0)

    def test_recovers_a_finite_value_from_literal_negative_infinity(self, trainer):
        """The real-world case this exists for: ratio=inf (float32
        overflow) with negative adv gives min_surr=-inf. Confirm the floor
        still recovers a finite, bounded value -- torch.max(-inf, finite)
        must equal the finite floor, not propagate -inf."""
        trainer.dual_clip_c = 3.0
        min_surr = torch.tensor([float("-inf")])
        adv = torch.tensor([-2.0])
        out = trainer._apply_dual_clip(min_surr, adv)
        assert math.isfinite(out.item())
        assert out.item() == pytest.approx(-6.0)

    def test_broadcasts_against_lower_rank_advantage(self, trainer):
        """The per-head counterfactual breakdown calls this with a
        (batch, 1) adv against a (batch, n_heads) min_surr."""
        trainer.dual_clip_c = 3.0
        min_surr = torch.tensor([[-500.0, -1.0, 10.0], [5.0, -0.1, -1000.0]])
        adv = torch.tensor([[-2.0], [-4.0]])
        out = trainer._apply_dual_clip(min_surr, adv)
        expected = torch.tensor([
            [max(-500.0, 3.0 * -2.0), max(-1.0, 3.0 * -2.0), 10.0],
            [5.0, max(-0.1, 3.0 * -4.0), max(-1000.0, 3.0 * -4.0)],
        ])
        assert torch.allclose(out, expected)


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="dual_clip_ppo_test", label="Dual-clip PPO test",
        description="Smoke-test 1v1 environment for the dual-clip PPO diagnostic",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=30.0)


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
                obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=tr["log_prob"], value=tr["value"], reward=reward,
                done=1.0 if done else 0.0,
            )
            last_obs = next_obs
        if done:
            env.reset()
    if last_obs is None:
        last_obs = next_obs
    return buffer, last_obs


def _make_fast_trainer(n_epochs: int, minibatch_size: int) -> PPOTrainer:
    """Manual construction (mirrors test_separate_value_net.py's
    test_value_net_trunk_hidden_override pattern) so hyperparameters are
    test-appropriate regardless of the live ai_config.json's current
    n_epochs (25+) -- n_epochs=1 and a minibatch_size covering the whole
    tiny rollout keeps this a single, easily-reasoned-about forward/
    backward pass instead of compounding across epochs/minibatches."""
    cfg = copy.deepcopy(load_ai_config())
    cfg["ppo"]["n_epochs"] = n_epochs
    cfg["ppo"]["minibatch_size"] = minibatch_size
    cfg["ppo"]["dual_clip_c"] = 0.0  # explicit baseline; tests set it themselves
    # augment_n_slot_shuffles>0 (the live default) would expand/reshuffle
    # the batch via geometric flips before _ppo_update's minibatch loop --
    # this test needs the one artificially-injected pathological row to
    # land in the batch exactly once, at a known state, not duplicated/
    # reordered by augmentation.
    cfg["ppo"]["augment_n_slot_shuffles"] = 0
    return PPOTrainer(
        decision_net=DecisionNetwork.from_config(),
        execution_net=ExecutionNetwork.from_config(),
        cfg=cfg,
    )


def _build_batch_with_forced_ratio_blowup(trainer: PPOTrainer, log_prob_offset: float):
    """Collects a real rollout, then artificially offsets ONE row's stored
    old_log_prob far into the past (and forces that row's advantage
    negative) so ratio=exp(new_lp - old_lp) is guaranteed to be extreme for
    that row -- deterministic, without needing many real epochs/rollouts to
    stumble into a genuine spike like training_runs.md's."""
    env = _make_env()
    buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
    with torch.no_grad():
        last_obs_dict = {
            k: v.unsqueeze(0).to(trainer.device) for k, v in last_obs.to_torch_dict().items()
        }
        last_value = trainer._get_value(last_obs_dict)
    advantages, returns = buffer.compute_gae(trainer.gamma, trainer.lam, last_value)
    batch = buffer.as_tensors(advantages, returns)

    target_row = 0
    batch["log_probs"] = batch["log_probs"].clone()
    batch["advantages"] = batch["advantages"].clone()
    batch["log_probs"][target_row] -= log_prob_offset
    batch["advantages"][target_row] = -5.0  # a real, moderate negative advantage
    return batch


class TestDualClipIntegration:
    def test_bounds_a_large_finite_ratio_blowup(self, trainer):
        """A ~45-nat old_log_prob deficit gives ratio ~= exp(45) ~= 3.5e19
        -- large but still finite in float32. Disabled: this single row's
        raw contribution (ratio*|adv| ~ 1.7e20) dominates the minibatch
        mean. Enabled (c=3): bounded at 3*5=15 regardless of how large the
        ratio actually got."""
        fast_trainer = _make_fast_trainer(n_epochs=1, minibatch_size=_ROLLOUT_STEPS + 5)
        batch = _build_batch_with_forced_ratio_blowup(fast_trainer, log_prob_offset=45.0)

        fast_trainer.dual_clip_c = 0.0
        metrics_off = fast_trainer._ppo_update(copy.deepcopy(batch), progress=0.0)
        assert metrics_off["policy_loss"] > 1e10, (
            "expected the disabled case to be dominated by the unbounded "
            f"blowup row, got policy_loss={metrics_off['policy_loss']}"
        )

        fast_trainer2 = _make_fast_trainer(n_epochs=1, minibatch_size=_ROLLOUT_STEPS + 5)
        fast_trainer2.dual_clip_c = 3.0
        metrics_on = fast_trainer2._ppo_update(copy.deepcopy(batch), progress=0.0)
        assert math.isfinite(metrics_on["policy_loss"])
        # Bounded by dual_clip_c*|adv| for the worst row, divided out over
        # the rest of the (small, otherwise-ordinary) minibatch -- generous
        # slack since other rows contribute some normal, non-zero amount too.
        assert abs(metrics_on["policy_loss"]) < 50.0, (
            f"expected a bounded policy_loss with dual-clip enabled, got {metrics_on['policy_loss']}"
        )

    def test_recovers_from_float32_overflow_to_inf(self, trainer):
        """A ~300-nat old_log_prob deficit guarantees ratio overflows to
        literal inf in float32 (exp(x) overflows past x~88.7). Disabled:
        the aggregate metric goes non-finite. Enabled: still finite and
        bounded, since torch.max(-inf, dual_clip_c*adv) recovers the floor."""
        fast_trainer = _make_fast_trainer(n_epochs=1, minibatch_size=_ROLLOUT_STEPS + 5)
        batch = _build_batch_with_forced_ratio_blowup(fast_trainer, log_prob_offset=300.0)

        fast_trainer.dual_clip_c = 0.0
        metrics_off = fast_trainer._ppo_update(copy.deepcopy(batch), progress=0.0)
        assert not math.isfinite(metrics_off["policy_loss"]), (
            f"expected inf/nan from float32 overflow with dual-clip disabled, "
            f"got a finite policy_loss={metrics_off['policy_loss']} -- test's "
            f"premise (guaranteed overflow) may not hold on this build/hardware"
        )

        fast_trainer2 = _make_fast_trainer(n_epochs=1, minibatch_size=_ROLLOUT_STEPS + 5)
        fast_trainer2.dual_clip_c = 3.0
        metrics_on = fast_trainer2._ppo_update(copy.deepcopy(batch), progress=0.0)
        assert math.isfinite(metrics_on["policy_loss"]), (
            "dual-clip should recover a finite value even from literal "
            "-inf inputs (torch.max(-inf, floor) == floor)"
        )
        assert abs(metrics_on["policy_loss"]) < 50.0
