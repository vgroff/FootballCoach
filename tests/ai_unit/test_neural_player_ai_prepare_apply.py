"""Tests for NeuralPlayerAI's prepare()/apply() split (added to support
batched rollout collection -- see ai/ppo/batched_rollout_worker.py).

``act()`` now literally does ``prepare()`` then ``sample_action_fn()`` then
``apply()`` back-to-back -- so the real thing worth verifying isn't "does
act() still work" (existing tests already cover that extensively, e.g.
test_rules_ai_nn_replay_equivalence.py, test_phase1_labels_timing.py, both
still passing unmodified after this refactor) but "does calling prepare()
and apply() SEPARATELY, with something happening in between (simulating a
caller that batches the sample_action_fn() call across multiple players/
environments), produce the EXACT SAME result as calling act() directly?"
That's the actual new capability this split exists for, and the one most at
risk of a subtle bug (e.g. state that should have been captured by prepare()
but wasn't, and got silently re-read -- possibly already stale -- by apply()
instead).
"""
from __future__ import annotations

import functools
import random

import numpy as np
import pytest
import torch

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

TRAINEE_ID = "trainee"


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="test_prepare_apply", label="t", description="t", build=build_1v1_scenario,
    )
    return ScenarioEnv(definition=defn, trainee_player_id=TRAINEE_ID, phase=1, max_episode_s=30.0)


@pytest.fixture(scope="module")
def sample_fn():
    trainer = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True)
    trainer.decision_net.eval()
    trainer.execution_net.eval()
    # deterministic=True: .mode()/mean everywhere, zero RNG draws inside
    # _sample_action itself -- isolates this test to whether prepare()/
    # apply() reproduce act()'s behavior, not whether two independent
    # stochastic samples happen to match (they wouldn't, by design).
    return functools.partial(trainer._sample_action, deterministic=True)


def _fresh_seeded_player(sample_fn, seed: int):
    """A freshly-reset env/match/trainee, with the trainee's NeuralPlayerAI
    RNG (used by encode_observation for sensor noise) explicitly reseeded --
    env.reset(seed=...) seeds the scenario BUILD (positions/attributes/ball),
    not necessarily NeuralPlayerAI's own internal _rng, so this is forced
    separately for a fair, fully-reproducible comparison. reset() itself
    never touches _rng (only tick counters/last_transition/last_gating), so
    overwriting it right after reset() is safe."""
    env = _make_env()
    env.sample_action_fn = sample_fn
    env.reset(seed=seed)
    player = env.match.player_by_id(TRAINEE_ID)
    player.ai._rng = random.Random(999)
    return env, player


class TestPrepareApplyMatchesAct:
    def test_prepare_then_apply_matches_direct_act_call(self, sample_fn):
        env_a, player_a = _fresh_seeded_player(sample_fn, seed=123)
        env_b, player_b = _fresh_seeded_player(sample_fn, seed=123)

        # Path A: act() directly -- prepare()+sample_action_fn()+apply() all
        # happen internally, exactly as every non-batched caller already does.
        player_a.ai.act(player_a, env_a.match, 0)

        # Path B: prepare()/apply() called SEPARATELY (the batched-usage
        # pattern) -- something inert happens "in between" to simulate a
        # caller batching other players' observations before dispatching
        # ONE combined network call.
        obs_dict = player_b.ai.prepare(player_b, env_b.match, 0)
        assert obs_dict is not None, "first tick after reset must be a decision tick"
        _unrelated_work = sum(range(1000))  # stands in for "batch other envs here"
        result = sample_fn(obs_dict)
        player_b.ai.apply(player_b, env_b.match, result)

        tr_a, tr_b = player_a.ai.last_transition, player_b.ai.last_transition
        assert tr_a is not None and tr_b is not None

        assert tr_a["log_prob"] == pytest.approx(tr_b["log_prob"], abs=1e-6)
        assert tr_a["value"] == pytest.approx(tr_b["value"], abs=1e-6)
        assert tr_a["bc_label"] is None and tr_b["bc_label"] is None
        assert tr_a["illegal_action"] == tr_b["illegal_action"]

        for k in tr_a["obs"]:
            np.testing.assert_allclose(
                tr_a["obs"][k], tr_b["obs"][k], atol=1e-6,
                err_msg=f"obs[{k}] differs between act() and prepare()+apply()",
            )
        for k in tr_a["raw_exec"]:
            np.testing.assert_allclose(
                tr_a["raw_exec"][k], tr_b["raw_exec"][k], atol=1e-6,
                err_msg=f"raw_exec[{k}] differs between act() and prepare()+apply()",
            )
        np.testing.assert_allclose(tr_a["head_log_probs"], tr_b["head_log_probs"], atol=1e-6)

        # Both players must also end up with the SAME gating actually applied
        # to the engine this tick (desired_direction/speed_mode), not just
        # the SAME stored transition.
        assert player_a.desired_speed_mode == player_b.desired_speed_mode
        if player_a.desired_direction is not None:
            assert player_b.desired_direction is not None
            np.testing.assert_allclose(
                [player_a.desired_direction.x, player_a.desired_direction.y, player_a.desired_direction.z],
                [player_b.desired_direction.x, player_b.desired_direction.y, player_b.desired_direction.z],
                atol=1e-6,
            )

    def test_prepare_returns_none_and_reapplies_gating_between_decision_ticks(self, sample_fn):
        """Between decision ticks, prepare() must return None (no decision
        due) and re-apply the cached gating exactly like act() does -- the
        other half of the contract, not exercised by the decision-tick test
        above."""
        env, player = _fresh_seeded_player(sample_fn, seed=42)
        ai = player.ai
        assert ai.decision_interval_ticks > 1, "test needs a real multi-tick interval to exercise the gap"

        # First call is always a decision tick (reset() sets up "decide on
        # first tick") -- consume it via act() to populate _last_gating.
        ai.act(player, env.match, 0)
        cached_gating = ai._last_gating
        assert cached_gating is not None

        # The very next call must NOT be a decision tick.
        obs_dict = ai.prepare(player, env.match, 0)
        assert obs_dict is None, "second consecutive call should not be a fresh decision tick"
        # last_gating must be unchanged (prepare() only re-applies it, never replaces it).
        assert ai._last_gating is cached_gating

    def test_pending_state_cleared_after_apply(self, sample_fn):
        """apply() must clear _pending_obs_dict/_pending_bc_label after
        consuming them -- stale pending state left behind would silently
        leak into the NEXT decision's last_transition if prepare() were ever
        skipped by a caller bug."""
        env, player = _fresh_seeded_player(sample_fn, seed=7)
        ai = player.ai
        obs_dict = ai.prepare(player, env.match, 0)
        assert obs_dict is not None
        assert ai._pending_obs_dict is not None
        result = sample_fn(obs_dict)
        ai.apply(player, env.match, result)
        assert ai._pending_obs_dict is None
        assert ai._pending_bc_label is None


class TestPrecomputedResultHook:
    """The batching hook: an external caller (the batched rollout worker)
    calls prepare() itself, does its own batched network call, and stashes
    the result on _precomputed_result BEFORE act() runs this tick. act()
    must consume it via apply() directly and must NOT call prepare() again
    -- a second prepare() call for the same tick would double-increment
    _ticks_since_decision and silently desync the decision cadence."""

    def test_act_consumes_precomputed_result_without_calling_prepare_again(self, sample_fn):
        env_a, player_a = _fresh_seeded_player(sample_fn, seed=55)
        env_b, player_b = _fresh_seeded_player(sample_fn, seed=55)
        ai_a, ai_b = player_a.ai, player_b.ai

        # Path A: normal act() (prepare()+sample_action_fn()+apply() internally).
        ticks_before_a = ai_a._ticks_since_decision
        ai_a.act(player_a, env_a.match, 0)
        # _ticks_since_decision must have moved by exactly the normal amount
        # for ONE decision (reset to 0 on a decision tick) -- not
        # double-incremented.
        assert ai_a._ticks_since_decision == 0

        # Path B: simulate the batched worker -- call prepare() manually
        # (exactly what the batched worker does before its own network
        # call), then stash the result as "precomputed" and call act(),
        # which must use it WITHOUT calling prepare() a second time.
        ticks_before_b = ai_b._ticks_since_decision
        obs_dict = ai_b.prepare(player_b, env_b.match, 0)
        assert obs_dict is not None
        result = sample_fn(obs_dict)
        ai_b._precomputed_result = result
        ai_b.act(player_b, env_b.match, 0)

        assert ai_b._ticks_since_decision == 0, (
            "act() must not have called prepare() again -- _ticks_since_decision "
            "would be double-incremented (nonzero) if it had"
        )
        assert ai_b._precomputed_result is None, "act() must clear _precomputed_result after consuming it"

        tr_a, tr_b = ai_a.last_transition, ai_b.last_transition
        assert tr_a is not None and tr_b is not None
        assert tr_a["log_prob"] == pytest.approx(tr_b["log_prob"], abs=1e-6)
        assert tr_a["value"] == pytest.approx(tr_b["value"], abs=1e-6)
        assert player_a.desired_speed_mode == player_b.desired_speed_mode

    def test_precomputed_result_does_not_leak_into_next_tick(self, sample_fn):
        """Consuming a precomputed result must clear the slot -- a leftover
        value would silently hijack the NEXT tick's act() call too, skipping
        a prepare() call that should have happened normally."""
        env, player = _fresh_seeded_player(sample_fn, seed=88)
        ai = player.ai
        obs_dict = ai.prepare(player, env.match, 0)
        result = sample_fn(obs_dict)
        ai._precomputed_result = result
        ai.act(player, env.match, 0)
        assert ai._precomputed_result is None

        # Next tick (not a decision tick -- decision_interval_ticks > 1):
        # act() must fall through to the normal prepare() path, which
        # re-applies cached gating and returns without touching sample_action_fn.
        assert ai.decision_interval_ticks > 1
        ticks = ai._ticks_since_decision
        ai.act(player, env.match, 0)
        assert ai._ticks_since_decision == ticks + 1, "second tick should be a normal, non-decision prepare() call"


class TestIsDueForDecision:
    """is_due_for_decision() -- the peek ai/ppo/batched_rollout_worker.py
    uses to check due-ness WITHOUT calling prepare() (which increments
    _ticks_since_decision unconditionally on every call, so using it as the
    "check" would itself consume a tick and desync the cadence). Added after
    a real production crash: ScenarioEnv.step() has a legitimate early-exit
    that can end a decision interval several ticks short, leaving a
    NeuralPlayerAI genuinely not due at the start of what
    BatchedEnvGroup.collect() assumed was always a fresh decision boundary
    -- see that module's docstring for the full story. The one thing this
    method absolutely must get right is agreeing with prepare()'s OWN
    due/not-due branch for every possible counter value, since collect()
    trusts it to decide whether to call prepare() at all.
    """

    def test_agrees_with_prepare_across_the_full_counter_range(self, sample_fn):
        """For every _ticks_since_decision value from 0 to decision_interval_ticks
        (inclusive, covering one full cycle plus the reset boundary),
        is_due_for_decision()'s answer must exactly predict whether the
        immediately-following prepare() call returns a real obs_dict (due)
        or None (not due, cached-gating re-apply) -- checked by actually
        calling prepare() and comparing, not by re-deriving the same
        arithmetic a second time (which could just reproduce the same bug
        in two places at once)."""
        env, player = _fresh_seeded_player(sample_fn, seed=321)
        ai = player.ai

        for forced_ticks in range(0, ai.decision_interval_ticks + 1):
            ai._ticks_since_decision = forced_ticks
            predicted_due = ai.is_due_for_decision()

            obs_dict = ai.prepare(player, env.match, 0)
            actually_due = obs_dict is not None

            assert predicted_due == actually_due, (
                f"_ticks_since_decision={forced_ticks}: is_due_for_decision() "
                f"predicted {predicted_due} but prepare() said {actually_due}"
            )

    def test_true_right_before_a_decision_tick(self, sample_fn):
        env, player = _fresh_seeded_player(sample_fn, seed=322)
        ai = player.ai
        ai._ticks_since_decision = ai.decision_interval_ticks - 1
        assert ai.is_due_for_decision()

    def test_false_freshly_after_a_decision(self, sample_fn):
        """Right after prepare() resets the counter to 0 on a decision tick,
        the NEXT tick must not be due again immediately (assuming a real
        multi-tick interval)."""
        env, player = _fresh_seeded_player(sample_fn, seed=323)
        ai = player.ai
        assert ai.decision_interval_ticks > 1
        obs_dict = ai.prepare(player, env.match, 0)
        assert obs_dict is not None  # reset() primes "due on first tick"
        assert not ai.is_due_for_decision()

    def test_checking_due_ness_does_not_itself_advance_the_counter(self, sample_fn):
        """The whole reason this method exists instead of just calling
        prepare() and checking for None: prepare() mutates state on every
        call. is_due_for_decision() must not."""
        env, player = _fresh_seeded_player(sample_fn, seed=324)
        ai = player.ai
        ai._ticks_since_decision = 3
        for _ in range(5):
            ai.is_due_for_decision()
        assert ai._ticks_since_decision == 3, "is_due_for_decision() must be a pure read"
