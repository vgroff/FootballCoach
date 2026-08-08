"""Tests for parallel PPO rollout collection (ai/ppo/rollout_worker.py,
PPOTrainer._train_parallel, PPOTrainer._merge_worker_batches).

See ai_trainer_knowledge.md "Parallel rollout collection" for the design.
"""
from __future__ import annotations

import numpy as np
import torch

from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _merge_worker_batches
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer


def _fake_batch(n: int, seed: int) -> dict:
    """Small hand-built batch matching RolloutBuffer.as_tensors()'s shape,
    with distinguishable values so concatenation order/correctness is checkable."""
    rng = np.random.default_rng(seed)
    return {
        "obs/self_feat": torch.from_numpy(rng.random((n, 4)).astype(np.float32)),
        "log_probs": torch.arange(seed * 100, seed * 100 + n, dtype=torch.float32),
        "values": torch.zeros(n, dtype=torch.float32),
        "rewards": torch.ones(n, dtype=torch.float32) * seed,
        "dones": torch.zeros(n, dtype=torch.float32),
        "advantages": torch.zeros(n, dtype=torch.float32),
        "returns": torch.zeros(n, dtype=torch.float32),
        "sample_weights": torch.ones(n, dtype=torch.float32),
        "reward_comps_raw": [{"appr": float(seed)} for _ in range(n)],
        "step_outcomes": [f"w{seed}"] * n,
    }


class TestMergeWorkerBatches:
    def test_concatenates_tensors_along_dim0(self):
        b1 = _fake_batch(3, seed=1)
        b2 = _fake_batch(5, seed=2)
        merged = _merge_worker_batches([b1, b2])
        assert merged["log_probs"].shape[0] == 8
        assert merged["obs/self_feat"].shape == (8, 4)
        # Order preserved: worker 1's rows first, then worker 2's.
        assert torch.equal(merged["log_probs"][:3], b1["log_probs"])
        assert torch.equal(merged["log_probs"][3:], b2["log_probs"])

    def test_concatenates_list_valued_keys(self):
        b1 = _fake_batch(2, seed=1)
        b2 = _fake_batch(2, seed=2)
        merged = _merge_worker_batches([b1, b2])
        assert merged["step_outcomes"] == ["w1", "w1", "w2", "w2"]
        assert merged["reward_comps_raw"] == [
            {"appr": 1.0}, {"appr": 1.0}, {"appr": 2.0}, {"appr": 2.0},
        ]

    def test_single_worker_is_passthrough(self):
        b1 = _fake_batch(4, seed=1)
        merged = _merge_worker_batches([b1])
        assert torch.equal(merged["log_probs"], b1["log_probs"])
        assert merged["step_outcomes"] == b1["step_outcomes"]


class TestTrainDispatch:
    def test_parallel_requires_phase_id(self):
        """train() must reject n_parallel_envs > 1 without phase_id -- each
        worker needs a phase id to rebuild its own env from."""
        import copy
        from footballcoach.ai.config import load_ai_config
        from footballcoach.ai.env.scenario_env import ScenarioEnv
        from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition

        cfg = copy.deepcopy(load_ai_config())
        cfg["ppo"]["n_parallel_envs"] = 2
        trainer = PPOTrainer.from_config()
        trainer.n_parallel_envs = 2  # override without rebuilding networks

        defn = ScenarioDefinition(
            key="dispatch_test", label="dispatch_test", description="",
            build=build_1v1_scenario,
        )
        env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=10.0)
        try:
            trainer.train(env, total_steps=1)
            assert False, "expected ValueError for missing phase_id"
        except ValueError as e:
            assert "phase_id" in str(e)


class TestParallelTrainingSmoke:
    def test_two_workers_complete_one_rollout(self, tmp_path):
        """End-to-end: spawn 2 real worker processes, collect exactly one
        small rollout, run one PPO update, and confirm the step counter and
        checkpoint both advance. Slower than the rest of the AI test suite
        because it spawns real OS processes -- keep rollout_steps tiny."""
        import copy
        from footballcoach.ai.config import load_ai_config

        cfg = copy.deepcopy(load_ai_config())
        cfg["ppo"]["n_parallel_envs"] = 2
        cfg["ppo"]["rollout_steps"] = 20
        cfg["ppo"]["rollout_eval_trials"] = 0  # skip the extra eval-vs-rules pass, keep it fast

        from footballcoach.ai.models.decision_network import DecisionNetwork
        from footballcoach.ai.models.execution_network import ExecutionNetwork

        trainer = PPOTrainer(
            decision_net=DecisionNetwork.from_config(),
            execution_net=ExecutionNetwork.from_config(),
            cfg=cfg,
            checkpoint_dir=tmp_path,
        )
        assert trainer.n_parallel_envs == 2

        trainer.train(env=None, total_steps=cfg["ppo"]["rollout_steps"], phase_id=1)

        assert trainer._total_steps >= cfg["ppo"]["rollout_steps"]
        assert (tmp_path / "checkpoint1.pt").exists()
