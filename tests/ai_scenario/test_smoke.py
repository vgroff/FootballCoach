"""Smoke tests: a handful of PPO update steps on the Phase 1 env
don't crash and produce finite, changing losses.

These are the "catch gross implementation bugs before spending compute"
tests described in ai_design_doc.md section 12.  They are deliberately
small (20-step rollouts) so the full suite runs in a few seconds.

Run with:
    uv run pytest tests/ai_scenario/ -v
"""
from __future__ import annotations

import math

import torch

from footballcoach.ai.action.schema import ExecutionAction
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

# Small rollout so the test runs quickly (~20 decision steps = 10s sim time).
_ROLLOUT_STEPS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env() -> ScenarioEnv:
    """Minimal Phase 1 env with short episode length for speed."""
    defn = ScenarioDefinition(
        key="smoke_1v1",
        label="Smoke: 1v1",
        description="Smoke-test 1v1 environment",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,  # short to keep test fast
    )


def _collect_rollout(
    env: ScenarioEnv,
    trainer: PPOTrainer,
    n: int,
) -> tuple[RolloutBuffer, object]:
    """Collect ``n`` decision-interval steps and return ``(buffer, last_obs)``."""
    buffer = RolloutBuffer()
    obs = env.reset()
    last_obs = obs

    for _ in range(n):
        obs_dict = obs.to_torch_dict()
        (
            action,
            log_prob,
            value,
            decision_probs,
            execution_physical,
            decision_physical,
            target_slots,
        ) = trainer._sample_action(obs_dict)

        env_action = {
            "decision_probs": decision_probs,
            "execution_physical": execution_physical,
            "decision_physical": decision_physical,
            "target_slots": target_slots,
            "slot_player_ids": [None] * 21,
            "decision": action,
            "execution": ExecutionAction(),
        }
        next_obs, reward, done, _info = env.step(env_action)

        buffer.add(
            obs={k: v.numpy() for k, v in obs_dict.items()},
            action=_action_to_numpy(action),
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=1.0 if done else 0.0,
        )
        last_obs = next_obs
        obs = env.reset() if done else next_obs

    return buffer, last_obs


def _run_ppo_update(
    trainer: PPOTrainer,
    buffer: RolloutBuffer,
    last_obs: object,
) -> dict:
    """Bootstrap last value, compute GAE, run one PPO update, return metrics."""
    with torch.no_grad():
        last_obs_dict = {
            k: v.unsqueeze(0).to(trainer.device)
            for k, v in last_obs.to_torch_dict().items()
        }
        last_value = trainer._get_value(last_obs_dict)

    advantages, returns = buffer.compute_gae(trainer.gamma, trainer.lam, last_value)
    batch = buffer.as_tensors(advantages, returns)
    return trainer._ppo_update(batch, progress=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_env_reset_produces_finite_obs():
    """ScenarioEnv.reset() returns finite tensors with all expected keys."""
    env = _make_env()
    obs = env.reset()
    d = obs.to_torch_dict()
    for key in ("self_feat", "other_feat", "ball_feat", "global_feat", "exists_mask"):
        assert key in d, f"Missing observation key: {key}"
        assert torch.isfinite(d[key]).all(), f"obs[{key}] contains non-finite values"


def test_sample_action_returns_finite_log_prob():
    """PPOTrainer._sample_action() does not crash and returns a finite log_prob."""
    env = _make_env()
    trainer = PPOTrainer.from_config()
    obs = env.reset()
    result = trainer._sample_action(obs.to_torch_dict())
    assert len(result) == 7, "Expected 7-tuple from _sample_action"
    _action, log_prob, value, *_ = result
    assert math.isfinite(log_prob), f"log_prob={log_prob} is not finite"
    assert math.isfinite(value), f"value={value} is not finite"


def test_env_step_returns_finite_reward():
    """env.step() completes without crash; reward and next-obs are finite."""
    env = _make_env()
    trainer = PPOTrainer.from_config()
    obs = env.reset()
    (
        action,
        _log_prob,
        _value,
        decision_probs,
        execution_physical,
        decision_physical,
        target_slots,
    ) = trainer._sample_action(obs.to_torch_dict())

    env_action = {
        "decision_probs": decision_probs,
        "execution_physical": execution_physical,
        "decision_physical": decision_physical,
        "target_slots": target_slots,
        "slot_player_ids": [None] * 21,
        "decision": action,
        "execution": ExecutionAction(),
    }
    next_obs, reward, done, _info = env.step(env_action)

    assert math.isfinite(reward), f"reward={reward} is not finite"
    assert isinstance(done, bool)
    for k, v in next_obs.to_torch_dict().items():
        assert torch.isfinite(v).all(), f"next_obs[{k}] has non-finite values"


def test_ppo_update_produces_finite_losses():
    """Collect a 20-step rollout, run one PPO update; all loss metrics must
    be finite.  Catches crashes in log_prob computation, GAE, backward(), etc."""
    env = _make_env()
    trainer = PPOTrainer.from_config()
    buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
    metrics = _run_ppo_update(trainer, buffer, last_obs)

    for key, val in metrics.items():
        assert math.isfinite(val), f"PPO metrics['{key}']={val} is not finite"


def test_two_ppo_updates_change_policy():
    """Two sequential PPO updates should produce different loss values.

    If both updates return identical losses, backward()/optimizer.step() is
    not modifying the network weights - this is the primary signal that the
    full train loop (loss backward) is broken.
    """
    env = _make_env()
    trainer = PPOTrainer.from_config()

    buffer1, last_obs1 = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
    m1 = _run_ppo_update(trainer, buffer1, last_obs1)

    buffer2, last_obs2 = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
    m2 = _run_ppo_update(trainer, buffer2, last_obs2)

    changed = (
        m1["policy_loss"] != m2["policy_loss"]
        or m1["value_loss"] != m2["value_loss"]
    )
    assert changed, (
        "Two PPO updates produced identical losses - backward/optimizer may be broken.\n"
        f"  update 1: policy={m1['policy_loss']:.6f}, value={m1['value_loss']:.6f}\n"
        f"  update 2: policy={m2['policy_loss']:.6f}, value={m2['value_loss']:.6f}"
    )
