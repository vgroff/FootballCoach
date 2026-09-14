"""Coverage for ai/eval/seeded_eval.py's batched evaluation consumer
(run_seeded_evaluation_batched / run_seeded_evaluation_parallel_batched) --
the eval-side counterpart to test_batched_rollout_secondary_batching.py's
training-side coverage.

Deliberately does NOT compare full-episode rewards/outcomes between the
batched and unbatched paths (an earlier version of this file did, and it
was genuinely flaky -- not a false positive from ambient test-run state,
but a REAL property: running a batch-of-N>1 network call vs N separate
batch-of-1 calls is not bit-identical (floating-point matmul reduction
order depends on batch shape -- see test_sample_action_batch.py's own
atol=1e-4 for the exact same phenomenon at the network-output level), and
running that tiny per-decision difference through many ticks of physics
simulation amplifies it unpredictably -- sometimes zero episodes end up
different, sometimes one, sometimes two. There is no fixed "number of
outliers to tolerate" that makes this a stable test; tightening or
loosening the tolerance just moves which run happens to fail. What CAN be
tested exactly:
  1. Episode count/seed-enumeration math (pure arithmetic, no floats).
  2. That batching actually happens -- a real >1-sized _sample_action_batch
     call occurs when envs_per_process > 1 (proves the speed optimization
     is real, not a silent no-op looping batch-of-1 calls) -- via
     instrumentation/call-counting, not value comparison.
  3. run_seeded_evaluation_batched's own internal consistency (episode
     count actually collected, no crash) under secondary_trainer batching
     and uneven slot-refill timing.
The underlying batched network call's correctness (row alignment, no
cross-row leakage, batch-size independence) is already rigorously covered
by test_sample_action_batch.py -- not re-proven here through a much noisier
full-physics-episode lens.

run_seeded_evaluation_parallel_batched's actual subprocess spawning is
deliberately NOT covered here by a real multiprocessing.Pool (no test file
in this repo spawns real subprocesses -- pytest test modules aren't
reliably re-importable by a freshly spawned interpreter the way this
project's own top-level scripts/modules are); verified instead via a one-off
manual script during development, mirroring how ai/ppo/batched_rollout_worker.py's
own spawn_batched_workers plumbing has no direct test either (see that
module's test file's docstring).
"""
from __future__ import annotations

import functools

import pytest
import torch

from footballcoach.ai.eval.seeded_eval import (
    run_seeded_evaluation_batched,
)
from footballcoach.ai.ppo.ppo_trainer import (
    PPOTrainer,
    _build_eval_env_factory,
    _build_neural_vs_neural_env_factory,
)

_SEEDS = list(range(5_300_000, 5_300_006))  # 6 seeds, deliberately far from real eval_seed_base


@pytest.fixture(scope="module")
def trainer_a():
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


@pytest.fixture(scope="module")
def trainer_b():
    """A SECOND, independently-initialized trainer (different weights from
    trainer_a) -- the neural-vs-neural test's opponent snapshot."""
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


def _det(sample_action_fn):
    return functools.partial(
        sample_action_fn, deterministic=True, deterministic_decision=True, deterministic_direction=True,
    )


class TestBatchedEvalMatchesUnbatchedRulesOpponent:
    """No secondary_trainer -- the opponent is Phase1RulesAI, nothing to
    batch on that side (see _secondary_neural_candidates' duck-typed
    skip)."""

    def test_episode_count_matches_seed_enumeration(self, trainer_a):
        """Pure arithmetic, no floats involved: len(seeds) * repeats_per_seed
        * (2 if swap_sides else 1) episodes, exactly."""
        env_factory = _build_eval_env_factory(use_rules_ai=True, max_episode_s=12.0)
        result = run_seeded_evaluation_batched(
            env_factory, trainer_a, _SEEDS,
            repeats_per_seed=2, swap_sides=True, envs_per_process=4,
            deterministic=True, deterministic_decision=True, deterministic_direction=True,
        )
        assert result.n_episodes == len(_SEEDS) * 2 * 2
        assert len(result.rewards) == result.n_episodes
        assert len(result.outcomes_list) == result.n_episodes
        assert len(result.episode_lengths_s) == result.n_episodes

    def test_batching_actually_batches_the_network_call(self, trainer_a, monkeypatch):
        """Proves envs_per_process > 1 genuinely groups multiple envs into
        ONE _sample_action_batch call (the actual point of this module)
        rather than silently looping batch-of-1 calls -- a real call-count/
        call-size check, not a value comparison, so it can't be flaky."""
        env_factory = _build_eval_env_factory(use_rules_ai=True, max_episode_s=12.0)
        batch_sizes: list[int] = []
        orig = trainer_a._sample_action_batch

        def _spy(obs_dict_batch, **kwargs):
            batch_sizes.append(next(iter(obs_dict_batch.values())).shape[0])
            return orig(obs_dict_batch, **kwargs)

        monkeypatch.setattr(trainer_a, "_sample_action_batch", _spy)
        run_seeded_evaluation_batched(
            env_factory, trainer_a, _SEEDS,
            repeats_per_seed=1, swap_sides=False, envs_per_process=4,
            deterministic=True, deterministic_decision=True, deterministic_direction=True,
        )
        assert batch_sizes, "expected at least one _sample_action_batch call"
        assert max(batch_sizes) > 1, (
            f"expected at least one batch of >1 envs with envs_per_process=4, got sizes {batch_sizes}"
        )


class TestBatchedEvalMatchesUnbatchedNeuralOpponent:
    """secondary_trainer=trainer_b batches the opponent's decisions too --
    mirrors BatchedEnvGroup's real-self-play secondary_trainer path, just
    for eval instead of rollout collection."""

    def test_episode_count_and_secondary_batching(self, trainer_a, trainer_b, monkeypatch):
        env_factory = _build_neural_vs_neural_env_factory(
            max_episode_s=12.0, opponent_sample_action_fn=_det(trainer_b._sample_action),
        )
        secondary_batch_sizes: list[int] = []
        orig = trainer_b._sample_action_batch

        def _spy(obs_dict_batch, **kwargs):
            secondary_batch_sizes.append(next(iter(obs_dict_batch.values())).shape[0])
            return orig(obs_dict_batch, **kwargs)

        monkeypatch.setattr(trainer_b, "_sample_action_batch", _spy)
        result = run_seeded_evaluation_batched(
            env_factory, trainer_a, _SEEDS,
            repeats_per_seed=1, swap_sides=False, envs_per_process=3,
            secondary_trainer=trainer_b,
            deterministic=True, deterministic_decision=True, deterministic_direction=True,
        )

        assert result.n_episodes == len(_SEEDS)
        assert len(result.rewards) == len(_SEEDS)
        # Proves the OPPONENT's decisions really do go through
        # secondary_trainer's own _sample_action_batch (not silently falling
        # back to unbatched per-env sampling) -- exact/deterministic, no
        # value comparison.
        assert secondary_batch_sizes, "expected secondary_trainer._sample_action_batch to be called"

    def test_slot_refill_drains_uneven_episode_lengths(self, trainer_a, trainer_b):
        """envs_per_process=2 with 6 tasks forces at least one slot to
        finish early and get refilled from the remaining queue mid-run
        (episodes rarely finish in lockstep) -- guards the _new_slot()
        StopIteration/refill bookkeeping specifically, not just the
        steady-state batching math."""
        env_factory = _build_neural_vs_neural_env_factory(
            max_episode_s=12.0, opponent_sample_action_fn=_det(trainer_b._sample_action),
        )
        result = run_seeded_evaluation_batched(
            env_factory, trainer_a, _SEEDS,
            repeats_per_seed=1, swap_sides=False, envs_per_process=2,
            secondary_trainer=trainer_b,
            deterministic=True, deterministic_decision=True, deterministic_direction=True,
        )
        assert result.n_episodes == len(_SEEDS)
        assert len(result.rewards) == len(_SEEDS)