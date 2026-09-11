"""Tests for PPOTrainer's batched action-sampling path (``_sample_action_batch``),
added to support batched rollout collection (see ai/ppo/batched_rollout_worker.py).

The refactor split the old single ``_sample_action`` into:
  ``_sample_action_networks``         -- batched network forward passes
                                          (decision_net/execution_net/value_net),
                                          the expensive part.
  ``_sample_action_from_heads``       -- the ORIGINAL per-row sampling/
                                          log_prob/decanonicalization body,
                                          unchanged. No longer called by
                                          ``_sample_action_batch`` (see
                                          below) -- kept as the ground-truth
                                          single-row reference implementation,
                                          used by this file and by
                                          test_sample_action_from_heads_batch.py.
  ``_sample_action_from_heads_batch`` -- vectorized replacement for the
                                          per-row tail: does the same
                                          sampling/log_prob/decanonicalization
                                          as ``_sample_action_from_heads``,
                                          but for all K rows in one batched
                                          pass instead of a Python loop of K
                                          calls (that loop was ~53% of
                                          decision time even after the
                                          network forward pass was batched --
                                          see test_sample_action_from_heads_batch.py's
                                          docstring for the full rationale
                                          and the masking-risk analysis that
                                          motivated it).
  ``_sample_action``                  -- thin wrapper: batch of 1 through the
                                          batched path.
  ``_sample_action_batch``            -- the batched entry point; calls
                                          ``_sample_action_networks`` once
                                          then ``_sample_action_from_heads_batch``
                                          once (not a loop).

Since ``_sample_action`` now literally calls
``_sample_action_batch(batch=1)``, those two can never diverge by
construction. This file's tests exercise the batched path end-to-end
through REAL network forward passes and compare against individual
batch=1 calls (which route through the exact same vectorized tail, just
with K=1) -- this mainly guards the NETWORK's own batched-vs-individual
forward pass (row misalignment, cross-row attention leakage, incorrect
masking would silently corrupt training data without ever raising an
exception). test_sample_action_from_heads_batch.py additionally compares
the vectorized tail against the ORIGINAL per-row ``_sample_action_from_heads``
using hand-constructed synthetic heads, which lets it force specific
gating/masking combinations directly instead of hoping real network outputs
happen to hit them.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from footballcoach.ai.obs.schema import (
    AI_TYPE_ONE_HOT_DIM,
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
)
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

_ATOL = 1e-4


def _make_obs_dict(seed: int) -> dict:
    """One random-but-valid, UNBATCHED observation dict -- matches the
    shapes ``ObservationBatch.to_torch_dict()`` produces, which is what
    ``_sample_action`` expects (it adds the batch dim itself)."""
    g = torch.Generator().manual_seed(seed)
    n_others = int(torch.randint(1, MAX_OTHER_PLAYERS + 1, (1,), generator=g).item())
    exists_mask = torch.zeros(MAX_OTHER_PLAYERS)
    exists_mask[:n_others] = 1.0
    return {
        "self_feat": torch.randn(PLAYER_FEATURE_DIM, generator=g),
        "other_feat": torch.randn(MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM, generator=g),
        "exists_mask": exists_mask,
        "ball_feat": torch.randn(BALL_FEATURE_DIM, generator=g),
        "global_feat": torch.randn(GLOBAL_FEATURE_DIM, generator=g),
        "self_ai_type": torch.zeros(AI_TYPE_ONE_HOT_DIM),
        "other_ai_type": torch.zeros(MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM),
    }


def _stack_obs(obs_list: list) -> dict:
    keys = obs_list[0].keys()
    return {k: torch.stack([o[k] for o in obs_list]) for k in keys}


@pytest.fixture(scope="module")
def trainer():
    # separate_value_net=True exercises the extra self.value_net(...,
    # value_only=True) batched path too, not just execution_net's own value.
    t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=True)
    # .eval() disables dropout (network.value_dropout, default 0.0 anyway,
    # but this makes the test robust regardless of config) so batched vs
    # individual forward passes are exactly comparable, not just "close
    # because dropout happened to roll the same mask twice".
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


def _assert_field_close(a, b, label: str) -> None:
    if isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, atol=_ATOL, err_msg=label)
    elif isinstance(a, bool):
        assert a == b, label
    elif isinstance(a, (int, float)):
        assert a == pytest.approx(b, abs=_ATOL), label
    else:
        assert a == b, label


class TestBatchMatchesIndividualDeterministic:
    """deterministic=True removes ALL sampling randomness (every head uses
    .mode()/mean) -- lets every returned field be compared for near-exact
    equality (not just shape/finiteness), which is the strongest possible
    check that batching didn't silently change the network's answer."""

    def test_all_fields_match_row_for_row(self, trainer):
        K = 4
        obs_list = [_make_obs_dict(seed=100 + i) for i in range(K)]

        individual = [trainer._sample_action(o, deterministic=True) for o in obs_list]
        batched = trainer._sample_action_batch(_stack_obs(obs_list), deterministic=True)

        assert len(batched) == K
        for i in range(K):
            (ind_action, ind_lp, ind_value, ind_probs, ind_exec,
             ind_dec, ind_tgt, ind_raw, ind_hlp) = individual[i]
            (bat_action, bat_lp, bat_value, bat_probs, bat_exec,
             bat_dec, bat_tgt, bat_raw, bat_hlp) = batched[i]

            for f in dataclasses.fields(ind_action):
                _assert_field_close(
                    getattr(ind_action, f.name), getattr(bat_action, f.name),
                    f"row {i} action.{f.name}",
                )
            assert ind_lp == pytest.approx(bat_lp, abs=_ATOL), f"row {i} log_prob"
            assert ind_value == pytest.approx(bat_value, abs=_ATOL), f"row {i} value"
            for k in ind_probs:
                assert ind_probs[k] == pytest.approx(bat_probs[k], abs=_ATOL), f"row {i} decision_probs[{k}]"
            for k in ind_tgt:
                assert ind_tgt[k] == bat_tgt[k], f"row {i} target_slots[{k}]"
            for k in ind_exec:
                _assert_field_close(ind_exec[k], bat_exec[k], f"row {i} execution_physical[{k}]")
            for k in ind_dec:
                _assert_field_close(ind_dec[k], bat_dec[k], f"row {i} decision_physical[{k}]")
            for k in ind_raw:
                np.testing.assert_allclose(ind_raw[k], bat_raw[k], atol=_ATOL, err_msg=f"row {i} raw_exec[{k}]")
            np.testing.assert_allclose(ind_hlp, bat_hlp, atol=_ATOL, err_msg=f"row {i} head_log_probs")

    def test_single_row_batch_matches_direct_call(self, trainer):
        """K=1 batch must exactly match a direct _sample_action call --
        this is _sample_action's own wrapper claim, verified directly."""
        obs = _make_obs_dict(seed=7)
        direct = trainer._sample_action(obs, deterministic=True)
        batched = trainer._sample_action_batch(_stack_obs([obs]), deterministic=True)
        assert len(batched) == 1
        assert direct[1] == pytest.approx(batched[0][1], abs=1e-6)  # log_prob
        assert direct[2] == pytest.approx(batched[0][2], abs=1e-6)  # value

    def test_no_cross_row_leakage(self, trainer):
        """Changing ONE row's observation must not change any OTHER row's
        result -- the sharpest possible test for attention/entity-encoder
        code accidentally mixing information across the batch dimension."""
        K = 4
        obs_list = [_make_obs_dict(seed=400 + i) for i in range(K)]
        batched_a = trainer._sample_action_batch(_stack_obs(obs_list), deterministic=True)

        obs_perturbed = list(obs_list)
        obs_perturbed[2] = _make_obs_dict(seed=999)  # totally different row 2
        batched_b = trainer._sample_action_batch(_stack_obs(obs_perturbed), deterministic=True)

        for i in (0, 1, 3):
            assert batched_a[i][1] == pytest.approx(batched_b[i][1], abs=1e-5), (
                f"row {i} log_prob changed when row 2 was perturbed -- cross-row leakage"
            )
            assert batched_a[i][2] == pytest.approx(batched_b[i][2], abs=1e-5), (
                f"row {i} value changed when row 2 was perturbed -- cross-row leakage"
            )
        # Row 2 itself is EXPECTED to differ (sanity check the perturbation
        # actually did something, so a broken test can't pass vacuously).
        assert batched_a[2][2] != pytest.approx(batched_b[2][2], abs=1e-5)

    def test_different_batch_sizes_dont_change_a_shared_row(self, trainer):
        """The same first 3 rows, called as a batch of 3 vs padded out to a
        batch of 6 with 3 extra rows appended, must give IDENTICAL results
        for those first 3 rows -- batch SIZE itself must not perturb
        per-row output (rules out any accidental batch-size-dependent
        behavior, e.g. a stray reduction over the wrong dimension)."""
        obs_list = [_make_obs_dict(seed=500 + i) for i in range(3)]
        obs_list_padded = obs_list + [_make_obs_dict(seed=600 + i) for i in range(3)]

        small = trainer._sample_action_batch(_stack_obs(obs_list), deterministic=True)
        padded = trainer._sample_action_batch(_stack_obs(obs_list_padded), deterministic=True)

        for i in range(3):
            assert small[i][1] == pytest.approx(padded[i][1], abs=1e-5), f"row {i} log_prob differs by batch size"
            assert small[i][2] == pytest.approx(padded[i][2], abs=1e-5), f"row {i} value differs by batch size"


class TestBatchStochasticMode:
    """Stochastic (default) mode: literal sampled draws differ between
    batched/individual calls (independent RNG draws), but the
    RNG-independent quantities -- decision_probs, value -- must still match
    exactly, and everything must be finite. This exercises the real
    .sample()/.rsample() code paths (VonMises rejection sampling included)
    that deterministic=True never touches, which is what real rollout
    collection actually uses."""

    def test_decision_probs_and_value_match_regardless_of_sampling(self, trainer):
        K = 4
        obs_list = [_make_obs_dict(seed=200 + i) for i in range(K)]
        individual = [trainer._sample_action(o) for o in obs_list]
        batched = trainer._sample_action_batch(_stack_obs(obs_list))
        for i in range(K):
            ind_probs, bat_probs = individual[i][3], batched[i][3]
            for k in ind_probs:
                assert ind_probs[k] == pytest.approx(bat_probs[k], abs=_ATOL), f"row {i} decision_probs[{k}]"
            assert individual[i][2] == pytest.approx(batched[i][2], abs=_ATOL), f"row {i} value"

    def test_batched_results_all_finite(self, trainer):
        K = 6
        obs_list = [_make_obs_dict(seed=300 + i) for i in range(K)]
        batched = trainer._sample_action_batch(_stack_obs(obs_list))
        for i, result in enumerate(batched):
            _, log_prob, value, _, exec_phys, dec_phys, _, raw_exec, head_lp = result
            assert math.isfinite(log_prob), f"row {i} log_prob"
            assert math.isfinite(value), f"row {i} value"
            assert np.all(np.isfinite(head_lp)), f"row {i} head_log_probs"
            for k, v in raw_exec.items():
                assert np.all(np.isfinite(v)), f"row {i} raw_exec[{k}]"
            for k, v in exec_phys.items():
                if isinstance(v, np.ndarray):
                    assert np.all(np.isfinite(v)), f"row {i} execution_physical[{k}]"


class TestSharedValueNetOff:
    """Same equivalence guarantee with separate_value_net=False (execution_net
    is its own critic) -- a different code path inside
    _sample_action_networks, deserves its own coverage."""

    @pytest.fixture(scope="class")
    def trainer_shared(self):
        t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=False)
        t.decision_net.eval()
        t.execution_net.eval()
        return t

    def test_batch_matches_individual(self, trainer_shared):
        K = 3
        obs_list = [_make_obs_dict(seed=700 + i) for i in range(K)]
        individual = [trainer_shared._sample_action(o, deterministic=True) for o in obs_list]
        batched = trainer_shared._sample_action_batch(_stack_obs(obs_list), deterministic=True)
        for i in range(K):
            assert individual[i][1] == pytest.approx(batched[i][1], abs=_ATOL), f"row {i} log_prob"
            assert individual[i][2] == pytest.approx(batched[i][2], abs=_ATOL), f"row {i} value"
