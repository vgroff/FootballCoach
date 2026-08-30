"""Unit tests for GAE(lambda) in ppo/rollout_buffer.py.

Uses hand-computed reference values so that any off-by-one or incorrect
episode-boundary handling is caught immediately, rather than silently
producing mediocre training without crashing.

See ai_design_doc.md section 9.3 for the formula and the warning about the
most common silent GAE bug (off-by-one on next_value / dones alignment).
"""
import math

import pytest

from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer


def _make_buffer(rewards, values, dones):
    """Build a RolloutBuffer from plain lists (no obs/action needed for GAE)."""
    buf = RolloutBuffer()
    dummy_obs = {"x": __import__("numpy").zeros(1, dtype="float32")}
    dummy_act = {"a": __import__("numpy").zeros(1, dtype="float32")}
    for r, v, d in zip(rewards, values, dones):
        buf.add(obs=dummy_obs, action=dummy_act, log_prob=0.0, value=v, reward=r, done=d)
    return buf


# ---------------------------------------------------------------------------
# Hand-computed reference cases
# ---------------------------------------------------------------------------

class TestGAEHandComputed:
    """Three-step episode, gamma=0.9, lam=0.8, computed by hand.

    rewards = [1.0, 0.0, 1.0]
    values  = [0.5, 0.4, 0.3]
    dones   = [0.0, 0.0, 1.0]   <- episode ends at step 2
    last_value = 0.2              <- not used since dones[2]=1

    Step 2 (t=2):
        nnt  = 1 - 1.0 = 0.0
        delta = 1.0 + 0.9*0.2*0.0 - 0.3 = 0.7
        gae  = 0.7
    Step 1 (t=1):
        nnt  = 1.0
        next_v = values[2] = 0.3
        delta = 0.0 + 0.9*0.3 - 0.4 = -0.13
        gae  = -0.13 + 0.9*0.8*1.0*0.7 = -0.13 + 0.504 = 0.374
    Step 0 (t=0):
        nnt  = 1.0
        next_v = values[1] = 0.4
        delta = 1.0 + 0.9*0.4 - 0.5 = 0.86
        gae  = 0.86 + 0.9*0.8*1.0*0.374 = 0.86 + 0.26928 = 1.12928
    returns = [1.62928, 0.774, 1.0]
    """
    GAMMA = 0.9
    LAM   = 0.8
    REWARDS = [1.0, 0.0, 1.0]
    VALUES  = [0.5, 0.4, 0.3]
    DONES   = [0.0, 0.0, 1.0]
    LAST_V  = 0.2

    EXPECTED_ADV = [1.12928, 0.374, 0.7]
    EXPECTED_RET = [1.62928, 0.774, 1.0]

    def setup_method(self):
        self.buf = _make_buffer(self.REWARDS, self.VALUES, self.DONES)
        self.adv, self.ret = self.buf.compute_gae(self.GAMMA, self.LAM, self.LAST_V)

    def test_advantages_correct(self):
        for i, expected in enumerate(self.EXPECTED_ADV):
            assert self.adv[i] == pytest.approx(expected, abs=1e-5), (
                f"advantages[{i}] expected {expected}, got {self.adv[i]}"
            )

    def test_returns_correct(self):
        for i, expected in enumerate(self.EXPECTED_RET):
            assert self.ret[i] == pytest.approx(expected, abs=1e-5), (
                f"returns[{i}] expected {expected}, got {self.ret[i]}"
            )

    def test_returns_equal_advantages_plus_values(self):
        for i, (a, v) in enumerate(zip(self.adv, self.VALUES)):
            assert self.ret[i] == pytest.approx(a + v, abs=1e-7)


class TestGAEMonteCarlo:
    """With gamma=1.0, lam=1.0 and V=0 everywhere, advantages = MC returns.

    rewards = [1.0, 2.0]
    values  = [0.0, 0.0]
    dones   = [0.0, 0.0]
    last_value = 0.0

    MC return at t=0: 1+2=3; at t=1: 2.
    """
    def test_matches_monte_carlo_returns(self):
        buf = _make_buffer([1.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        adv, ret = buf.compute_gae(gamma=1.0, lam=1.0, last_value=0.0)
        assert adv[0] == pytest.approx(3.0, abs=1e-6)
        assert adv[1] == pytest.approx(2.0, abs=1e-6)
        assert ret[0] == pytest.approx(3.0, abs=1e-6)
        assert ret[1] == pytest.approx(2.0, abs=1e-6)


class TestGAEEpisodeBoundary:
    """done=1 at step 1 must NOT let step 2's value bleed into step 0's advantage.

    rewards = [1.0, 5.0, 2.0]
    values  = [1.0, 2.0, 1.0]
    dones   = [0.0, 1.0, 0.0]
    gamma   = 1.0, lam = 1.0
    last_value = 0.0

    Step 2 (t=2): nnt=1.0, next_v=last_v=0.0
        delta = 2.0 + 0.0 - 1.0 = 1.0; gae = 1.0
    Step 1 (t=1): nnt=0.0 (done!)
        delta = 5.0 + 0.0 - 2.0 = 3.0; gae = 3.0 + 0.0 = 3.0
    Step 0 (t=0): nnt=1.0, next_v=values[1]=2.0
        delta = 1.0 + 2.0 - 1.0 = 2.0; gae = 2.0 + 1.0*1.0*1.0*3.0 = 5.0

    Key: step 2's gae (1.0) is NOT included in step 0's computation because
    the done at step 1 zeroed out the carry-forward from step 2.
    """
    def test_done_zeroes_cross_episode_gae(self):
        buf = _make_buffer([1.0, 5.0, 2.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0])
        adv, ret = buf.compute_gae(gamma=1.0, lam=1.0, last_value=0.0)
        assert adv[2] == pytest.approx(1.0, abs=1e-6)
        assert adv[1] == pytest.approx(3.0, abs=1e-6)
        assert adv[0] == pytest.approx(5.0, abs=1e-6)

    def test_cross_episode_contamination_absent(self):
        """Step 0 advantage must not include step 2's reward (2.0) contribution."""
        # With done at step 1, step 0's advantage should be purely based on
        # the return within its own episode (step 0 + step 1's terminal reward).
        # If the boundary was broken, adv[0] would be larger (it would include
        # step 2's reward 2.0 discounted through).
        buf = _make_buffer([1.0, 5.0, 100.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0])
        adv, ret = buf.compute_gae(gamma=1.0, lam=1.0, last_value=0.0)
        # With the done correctly applied, adv[0] should still be 5.0
        # regardless of the large step-2 reward (100.0 vs 2.0 from before).
        assert adv[0] == pytest.approx(5.0, abs=1e-6)


class TestGAELastValueBootstrap:
    """last_value is used when the episode is NOT done at the final step."""

    def test_last_value_bootstraps_when_not_done(self):
        """Single step, not done: advantage = r + gamma*last_v - V."""
        buf = _make_buffer([1.0], [0.5], [0.0])
        adv, _ = buf.compute_gae(gamma=0.9, lam=1.0, last_value=2.0)
        # delta = 1.0 + 0.9*2.0 - 0.5 = 2.3
        assert adv[0] == pytest.approx(2.3, abs=1e-6)

    def test_last_value_ignored_when_done(self):
        """Single step, done=1: last_value is bootstrapped against but
        multiplied by next_non_terminal=0, so it has no effect."""
        buf = _make_buffer([1.0], [0.5], [1.0])
        adv_large, _ = buf.compute_gae(gamma=0.9, lam=1.0, last_value=999.0)
        adv_zero, _  = buf.compute_gae(gamma=0.9, lam=1.0, last_value=0.0)
        assert adv_large[0] == pytest.approx(adv_zero[0], abs=1e-6)

    def test_last_value_effect_decays_with_gamma(self):
        """At t=0, last_value contributes gamma^1*last_v (one step away)."""
        buf = _make_buffer([0.0], [0.0], [0.0])
        # Single step r=0, V=0, not done: adv = gamma * last_v
        adv, _ = buf.compute_gae(gamma=0.5, lam=1.0, last_value=4.0)
        assert adv[0] == pytest.approx(0.5 * 4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_step_terminal():
    """Buffer of length 1, done=1: straightforward single TD step."""
    buf = _make_buffer([5.0], [1.0], [1.0])
    adv, ret = buf.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
    # delta = 5.0 + 0.99*0.0*0.0 - 1.0 = 4.0
    assert adv[0] == pytest.approx(4.0, abs=1e-6)
    assert ret[0] == pytest.approx(5.0, abs=1e-6)  # adv + value = 4 + 1


def test_all_zeros_produces_zero_advantages():
    buf = _make_buffer([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    adv, ret = buf.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
    for a in adv:
        assert abs(a) < 1e-7
    for r in ret:
        assert abs(r) < 1e-7


def test_advantages_length_matches_buffer():
    buf = _make_buffer([1.0, 2.0, 3.0, 4.0], [0.5]*4, [0.0]*3 + [1.0])
    adv, ret = buf.compute_gae(0.99, 0.95, 0.0)
    assert len(adv) == 4
    assert len(ret) == 4


# ---------------------------------------------------------------------------
# Rollout buffer housekeeping
# ---------------------------------------------------------------------------

def test_buffer_len_increments():
    buf = RolloutBuffer()
    assert len(buf) == 0
    dummy = {"x": __import__("numpy").zeros(1, dtype="float32")}
    buf.add(obs=dummy, action=dummy, log_prob=0.0, value=1.0, reward=1.0, done=0.0)
    assert len(buf) == 1
    buf.add(obs=dummy, action=dummy, log_prob=0.0, value=1.0, reward=1.0, done=0.0)
    assert len(buf) == 2


def test_buffer_clear_empties():
    buf = _make_buffer([1.0, 2.0], [0.5, 0.5], [0.0, 1.0])
    assert len(buf) == 2
    buf.clear()
    assert len(buf) == 0


def test_as_tensors_scalar_shapes():
    import numpy as np
    import torch
    buf = _make_buffer([1.0, 0.5], [0.3, 0.4], [0.0, 1.0])
    adv, ret = buf.compute_gae(0.99, 0.95, 0.0)
    tensors = buf.as_tensors(adv, ret)
    assert tensors["log_probs"].shape == (2,)
    assert tensors["rewards"].shape == (2,)
    assert tensors["dones"].shape == (2,)
    assert tensors["advantages"].shape == (2,)
    assert tensors["returns"].shape == (2,)


def test_as_tensors_obs_stacked_correctly():
    import numpy as np
    import torch
    obs_a = {"feat": np.array([1.0, 2.0], dtype=np.float32)}
    obs_b = {"feat": np.array([3.0, 4.0], dtype=np.float32)}
    buf = RolloutBuffer()
    dummy_act = {"a": np.zeros(1, dtype=np.float32)}
    buf.add(obs=obs_a, action=dummy_act, log_prob=0.0, value=1.0, reward=1.0, done=0.0)
    buf.add(obs=obs_b, action=dummy_act, log_prob=0.0, value=1.0, reward=1.0, done=1.0)
    adv, ret = buf.compute_gae(0.99, 0.95, 0.0)
    tensors = buf.as_tensors(adv, ret)
    stacked = tensors["obs/feat"]
    assert stacked.shape == (2, 2)
    assert stacked[0, 0].item() == pytest.approx(1.0)
    assert stacked[1, 0].item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Multi-track segmentation (trainee + secondary/opponent rows interleaved)
#
# Real PPO rollout collection (ppo_trainer.py's train()/rollout_worker.py's
# _collect()) appends one trainee row per tick immediately followed by that
# tick's secondary-player row(s) (only non-empty when the secondary player
# is neural-controlled, i.e. self-play) -- e.g.
# [trainee_t0, opponent_t0, trainee_t1, opponent_t1, ...]. A naive flat
# backward GAE scan over that layout reads the WRONG track's value/reward as
# "next step" for roughly half the rows (whichever track's row does not sit
# immediately before the next row of the SAME track). ``track_ids`` +
# ``_track_index_groups()`` fix this by segmenting the backward recursion
# per track before scattering results back into the flat output arrays.
# ---------------------------------------------------------------------------

def _add_track(buf, rewards, values, dones, track_id):
    import numpy as np
    dummy_obs = {"x": np.zeros(1, dtype="float32")}
    dummy_act = {"a": np.zeros(1, dtype="float32")}
    for r, v, d in zip(rewards, values, dones):
        buf.add(obs=dummy_obs, action=dummy_act, log_prob=0.0, value=v, reward=r,
                 done=d, track_id=track_id)


class TestGAEMultiTrack:
    # Same fixture as TestGAEEpisodeBoundary (expected adv = [5.0, 3.0, 1.0]).
    TRAINEE_R, TRAINEE_V, TRAINEE_D = [1.0, 5.0, 2.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]
    # Same fixture as TestGAEMonteCarlo-style (V=0 everywhere -> pure MC).
    OPP_R, OPP_V, OPP_D = [10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]

    def _interleaved_buffer(self):
        buf = RolloutBuffer()
        for i in range(3):
            _add_track(buf, [self.TRAINEE_R[i]], [self.TRAINEE_V[i]], [self.TRAINEE_D[i]], "trainee")
            _add_track(buf, [self.OPP_R[i]], [self.OPP_V[i]], [self.OPP_D[i]], "opponent")
        return buf

    def test_interleaved_matches_isolated_per_track(self):
        """The whole point of segmentation: interleaving two tracks in one
        buffer must give byte-identical results to running GAE on each
        track alone -- this is exactly what a flat backward scan over the
        interleaved rows would get wrong."""
        trainee_only = _make_buffer(self.TRAINEE_R, self.TRAINEE_V, self.TRAINEE_D)
        expected_trainee_adv, expected_trainee_ret = trainee_only.compute_gae(1.0, 1.0, 0.0)
        opp_only = _make_buffer(self.OPP_R, self.OPP_V, self.OPP_D)
        expected_opp_adv, expected_opp_ret = opp_only.compute_gae(1.0, 1.0, 0.0)

        buf = self._interleaved_buffer()
        adv, ret = buf.compute_gae(1.0, 1.0, {"trainee": 0.0, "opponent": 0.0})

        trainee_idx, opp_idx = [0, 2, 4], [1, 3, 5]
        for local_i, flat_i in enumerate(trainee_idx):
            assert adv[flat_i] == pytest.approx(expected_trainee_adv[local_i], abs=1e-6)
            assert ret[flat_i] == pytest.approx(expected_trainee_ret[local_i], abs=1e-6)
        for local_i, flat_i in enumerate(opp_idx):
            assert adv[flat_i] == pytest.approx(expected_opp_adv[local_i], abs=1e-6)
            assert ret[flat_i] == pytest.approx(expected_opp_ret[local_i], abs=1e-6)

    def test_cross_track_leakage_would_change_result(self):
        """Sanity check that the fixture actually exercises the bug this
        segmentation fixes -- i.e. this suite isn't accidentally vacuous.
        Manually replicate the OLD flat (unsegmented) scan and confirm it
        disagrees with the correctly-segmented result at the row where
        contamination would occur (trainee's step 0, whose flat "next" row
        is the opponent's step 0, not trainee's own step 1)."""
        buf = self._interleaved_buffer()
        adv, _ = buf.compute_gae(1.0, 1.0, {"trainee": 0.0, "opponent": 0.0})

        # Old algorithm: single running last_gae, all_values = flat values + [last_value].
        rewards = buf.rewards
        values = buf.values
        dones = buf.dones
        n = len(rewards)
        flat_adv = [0.0] * n
        last_gae = 0.0
        all_values = values + [0.0]
        for t in reversed(range(n)):
            next_value = all_values[t + 1]
            nnt = 1.0 - dones[t]
            delta = rewards[t] + 1.0 * next_value * nnt - values[t]
            last_gae = delta + 1.0 * 1.0 * nnt * last_gae
            flat_adv[t] = last_gae

        assert flat_adv[0] != pytest.approx(adv[0], abs=1e-6), (
            "fixture no longer exercises cross-track leakage -- flat and "
            "segmented results agree, so this test can no longer catch a "
            "regression back to an unsegmented scan"
        )

    def test_missing_track_in_last_value_dict_defaults_to_zero(self):
        buf = RolloutBuffer()
        _add_track(buf, [1.0], [0.5], [0.0], "opponent")
        adv_missing, _ = buf.compute_gae(0.9, 1.0, {})
        adv_explicit, _ = buf.compute_gae(0.9, 1.0, {"opponent": 0.0})
        assert adv_missing[0] == pytest.approx(adv_explicit[0], abs=1e-6)

    def test_scalar_last_value_applies_to_every_track(self):
        """Backward compat: a bare float last_value (every existing
        single-track call site/test) still applies uniformly to ALL tracks
        present in the buffer."""
        buf = RolloutBuffer()
        _add_track(buf, [1.0], [0.5], [0.0], "trainee")
        _add_track(buf, [2.0], [0.3], [0.0], "opponent")
        adv_scalar, ret_scalar = buf.compute_gae(0.9, 1.0, 5.0)
        adv_dict, ret_dict = buf.compute_gae(0.9, 1.0, {"trainee": 5.0, "opponent": 5.0})
        assert adv_scalar == pytest.approx(adv_dict, abs=1e-6)
        assert ret_scalar == pytest.approx(ret_dict, abs=1e-6)

    def test_track_ids_default_to_trainee(self):
        buf = RolloutBuffer()
        dummy = {"x": __import__("numpy").zeros(1, dtype="float32")}
        buf.add(obs=dummy, action=dummy, log_prob=0.0, value=0.0, reward=0.0, done=0.0)
        buf.add(obs=dummy, action=dummy, log_prob=0.0, value=0.0, reward=0.0, done=0.0, track_id="opponent")
        assert buf.track_ids == ["trainee", "opponent"]

    def test_truncate_to_last_episode_end_drops_track_ids_too(self):
        buf = self._interleaved_buffer()
        # Append one more trailing (incomplete) tick to both tracks.
        _add_track(buf, [1.0], [0.0], [0.0], "trainee")
        _add_track(buf, [1.0], [0.0], [0.0], "opponent")
        n_dropped = buf.truncate_to_last_episode_end()
        assert n_dropped == 2
        assert len(buf.track_ids) == len(buf.rewards)
        assert buf.track_ids == ["trainee", "opponent"] * 3


class TestMCReturnsMultiTrack:
    """compute_mc_returns() is currently only ever fed trainee-only buffers
    (value pretraining doesn't record secondary transitions), but is
    segmented the same way as compute_gae() defensively -- these tests
    guard that segmentation directly rather than relying on it never being
    exercised."""

    def test_interleaved_matches_isolated_per_track(self):
        trainee_r, trainee_d = [1.0, 5.0, 2.0], [0.0, 1.0, 0.0]
        opp_r, opp_d = [10.0, 20.0, 30.0], [0.0, 0.0, 1.0]

        trainee_only = RolloutBuffer()
        _add_track(trainee_only, trainee_r, [0.0] * 3, trainee_d, "trainee")
        expected_trainee = trainee_only.compute_mc_returns(0.9)

        opp_only = RolloutBuffer()
        _add_track(opp_only, opp_r, [0.0] * 3, opp_d, "opponent")
        expected_opp = opp_only.compute_mc_returns(0.9)

        buf = RolloutBuffer()
        for i in range(3):
            _add_track(buf, [trainee_r[i]], [0.0], [trainee_d[i]], "trainee")
            _add_track(buf, [opp_r[i]], [0.0], [opp_d[i]], "opponent")
        returns = buf.compute_mc_returns(0.9)

        trainee_idx, opp_idx = [0, 2, 4], [1, 3, 5]
        for local_i, flat_i in enumerate(trainee_idx):
            assert returns[flat_i] == pytest.approx(expected_trainee[local_i], abs=1e-6)
        for local_i, flat_i in enumerate(opp_idx):
            assert returns[flat_i] == pytest.approx(expected_opp[local_i], abs=1e-6)
