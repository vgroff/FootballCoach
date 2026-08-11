"""Regression coverage for RolloutBuffer's step_outcome backfilling.

record_demonstrations.py's on-episode-end trial_outcome string is only ever
stored (via RolloutBuffer.add()'s step_outcome param) on the terminal
done=1 row of an episode -- every other row defaults to "". Without
backfilling, ppo_trainer.value_mse_by_outcome() buckets every one of those
""  rows under a fabricated "unknown" outcome, which dwarfs the real
outcome buckets for any non-trivial episode length (the exact bug reported
live: PPO's value-pretrain val_mse-by-outcome log line showing
"unknown=...(n=21118)"). _backfill_step_outcomes() (used by
RolloutBuffer.as_tensors()) must eliminate this by propagating each
episode's real outcome to every one of its own rows.
"""
from __future__ import annotations

import numpy as np
import pytest

from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer, _backfill_step_outcomes


def _make_buffer_with_outcomes(rewards, dones, step_outcomes):
    """Build a RolloutBuffer with dummy obs/action but real dones/step_outcomes."""
    buf = RolloutBuffer()
    dummy_obs = {"x": np.zeros(1, dtype=np.float32)}
    dummy_act = {"a": np.zeros(1, dtype=np.float32)}
    for r, d, o in zip(rewards, dones, step_outcomes):
        buf.add(obs=dummy_obs, action=dummy_act, log_prob=0.0, value=0.0,
                reward=r, done=d, step_outcome=o)
    return buf


class TestBackfillStepOutcomes:
    def test_single_episode_backfills_terminal_outcome_to_all_rows(self):
        dones = [0.0, 0.0, 0.0, 1.0]
        step_outcomes = ["", "", "", "box_possession"]
        out = _backfill_step_outcomes(step_outcomes, dones)
        assert out == ["box_possession"] * 4

    def test_multiple_episodes_each_get_their_own_outcome(self):
        dones = [0.0, 1.0, 0.0, 0.0, 1.0]
        step_outcomes = ["", "box_possession", "", "", "timeout"]
        out = _backfill_step_outcomes(step_outcomes, dones)
        assert out == ["box_possession", "box_possession", "timeout", "timeout", "timeout"]

    def test_trailing_incomplete_episode_stays_empty_not_fabricated(self):
        """Rows after the last done=1 belong to an episode that hasn't
        ended yet -- must stay "" (meaning "no outcome yet"), never get
        assigned a fabricated outcome string."""
        dones = [0.0, 1.0, 0.0, 0.0]
        step_outcomes = ["", "box_possession", "", ""]
        out = _backfill_step_outcomes(step_outcomes, dones)
        assert out == ["box_possession", "box_possession", "", ""]

    def test_single_row_episode(self):
        dones = [1.0]
        step_outcomes = ["timeout"]
        out = _backfill_step_outcomes(step_outcomes, dones)
        assert out == ["timeout"]

    def test_empty_buffer_returns_empty_list(self):
        assert _backfill_step_outcomes([], []) == []

    def test_missing_terminal_outcome_string_leaves_episode_all_empty(self):
        """If the terminal row itself has an empty outcome (e.g. info was
        None at that exact step -- see RolloutBuffer.add()'s step_outcome
        default), there is nothing real to propagate; every row in that
        episode stays "" rather than fabricating a value."""
        dones = [0.0, 0.0, 1.0]
        step_outcomes = ["", "", ""]
        out = _backfill_step_outcomes(step_outcomes, dones)
        assert out == ["", "", ""]

    def test_does_not_mutate_input_list(self):
        dones = [0.0, 1.0]
        step_outcomes = ["", "box_possession"]
        original = list(step_outcomes)
        _backfill_step_outcomes(step_outcomes, dones)
        assert step_outcomes == original

    def test_five_consecutive_single_row_episodes(self):
        """Every row is itself a done=1 row (degenerate single-step
        episodes) -- each keeps exactly its own outcome, no leakage across
        episode boundaries."""
        dones = [1.0, 1.0, 1.0]
        step_outcomes = ["box_possession", "timeout", "invalid"]
        out = _backfill_step_outcomes(step_outcomes, dones)
        assert out == ["box_possession", "timeout", "invalid"]

    @pytest.mark.parametrize("seed", range(20))
    def test_fuzzed_random_episode_lengths_never_leak_across_boundaries(self, seed):
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(2, 8))
        outcomes_pool = ["box_possession", "opponent_box_possession", "timeout", "miss", "invalid"]
        dones: list[float] = []
        step_outcomes: list[str] = []
        expected: list[str] = []
        for _ in range(n_eps):
            ep_len = int(rng.integers(1, 6))
            outcome = str(rng.choice(outcomes_pool))
            dones.extend([0.0] * (ep_len - 1) + [1.0])
            step_outcomes.extend([""] * (ep_len - 1) + [outcome])
            expected.extend([outcome] * ep_len)
        out = _backfill_step_outcomes(step_outcomes, dones)
        assert out == expected


class TestAsTensorsAppliesBackfill:
    def test_as_tensors_step_outcomes_are_backfilled(self):
        buf = _make_buffer_with_outcomes(
            rewards=[0.0, 0.0, 1.0],
            dones=[0.0, 0.0, 1.0],
            step_outcomes=["", "", "box_possession"],
        )
        advantages, returns = buf.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
        batch = buf.as_tensors(advantages, returns)
        assert batch["step_outcomes"] == ["box_possession", "box_possession", "box_possession"]

    def test_as_tensors_does_not_fabricate_outcome_for_trailing_incomplete_episode(self):
        buf = _make_buffer_with_outcomes(
            rewards=[0.0, 1.0, 0.0, 0.0],
            dones=[0.0, 1.0, 0.0, 0.0],
            step_outcomes=["", "timeout", "", ""],
        )
        advantages, returns = buf.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
        batch = buf.as_tensors(advantages, returns)
        assert batch["step_outcomes"] == ["timeout", "timeout", "", ""]

    def test_value_mse_by_outcome_no_longer_dominated_by_unknown(self):
        """End-to-end: a rollout with several multi-step episodes must NOT
        classify the majority of rows as "unknown" once backfilled -- this
        is the exact live-reported symptom
        ("unknown=...(n=21118)" dwarfing every real bucket)."""
        import torch
        from footballcoach.ai.ppo.ppo_trainer import value_mse_by_outcome

        buf = _make_buffer_with_outcomes(
            rewards=[0.0] * 9 + [1.0] + [0.0] * 4 + [1.0],
            dones=[0.0] * 9 + [1.0] + [0.0] * 4 + [1.0],
            step_outcomes=[""] * 9 + ["box_possession"] + [""] * 4 + ["timeout"],
        )
        advantages, returns = buf.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
        batch = buf.as_tensors(advantages, returns)
        pred = torch.zeros(len(batch["step_outcomes"]))
        target = torch.zeros(len(batch["step_outcomes"]))
        by_outcome = value_mse_by_outcome(pred, target, batch["step_outcomes"])
        assert "unknown" not in by_outcome
        assert set(by_outcome) == {"box_possession", "timeout"}
        assert by_outcome["box_possession"][1] == 10
        assert by_outcome["timeout"][1] == 5
