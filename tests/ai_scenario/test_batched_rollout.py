"""Tests for BatchedEnvGroup (ai/ppo/batched_rollout_worker.py) -- the
multi-environment rollout collection core that batches the trainee's
decision network call across several ScenarioEnv instances into one
PPOTrainer._sample_action_batch() call per round.

The underlying batched network call itself is already rigorously covered
by test_sample_action_batch.py (bit-exact row equivalence, cross-row
leakage, batch-size independence). What THIS file needs to guard is the
env-stepping/bookkeeping ORCHESTRATION around that call: does
prepare()/precompute/env.step()/apply() correctly produce well-formed,
per-env-isolated rollout data with zero cross-contamination between
environments sharing the same round's batched call?
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import functools

from footballcoach.ai.curriculum.envs import bc_label_fn_for_phase_player
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.batched_rollout_worker import BatchedEnvGroup
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

_PHASE = PHASES_BY_ID[1]


@pytest.fixture(scope="module")
def trainer():
    t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=True)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    return t


def _make_envs(trainer, n: int):
    """Deliberately does NOT use curriculum.envs.build_env(_PHASE) -- that
    reads ai_config.json's CURRENT phase1_opponent_*_ratio at call time, and
    this file's tests need to stay stable regardless of what those ratios
    happen to be live (they were 0 neural when these tests were first
    written; ai_config.json's ratios have since changed to mostly-neural,
    i.e. self-play, which silently made an ambient-config-reading fixture
    here flaky -- see test_no_cross_env_contamination_in_group's own
    comment). Explicitly forces a non-neural (rules-based) opponent instead,
    matching test_secondary_opponent_gae.py's own stated rationale for
    avoiding ai_config.json dependence. Self-play + BatchedEnvGroup
    specifically is covered by test_batched_rollout_self_play.py, which
    forces the OPPOSITE explicitly (opponent_rules_prob=0) for the same
    config-independence reason -- together the two files cover both
    opponent regimes without either depending on the live config file.
    """
    envs = []
    for _ in range(n):
        defn = ScenarioDefinition(
            key="phase1_1v1_test", label="Phase 1: 1v1 (test, rules-based opponent)",
            description="1v1 scenario for BatchedEnvGroup orchestration tests -- non-neural opponent, config-independent",
            build=functools.partial(
                build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0,
            ),
            on_tick=phase1_training_on_tick,
        )
        env = ScenarioEnv(
            definition=defn, trainee_player_id="trainee", phase=1,
            secondary_player_ids=["opponent"], **_PHASE.env_kwargs,
        )
        # NOTE: BatchedEnvGroup never actually calls env.sample_action_fn for
        # the TRAINEE -- every trainee decision is always precomputed (via
        # trainer._sample_action_batch, see collect()'s own deterministic=
        # kwargs) before env.step() runs, so NeuralPlayerAI.act()'s
        # sample_action_fn branch is never reached for the trainee. Still
        # required: ScenarioEnv.reset() checks it to decide whether to wire
        # a NeuralPlayerAI onto the trainee at all (env.sample_action_fn
        # must be set before construction/reset regardless), and it WOULD
        # matter for a neural secondary player's own act() fallback -- moot
        # here since opponent_rules_prob=1.0 above means "opponent" is
        # never neural in this file.
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = bc_label_fn_for_phase_player(1)
        envs.append(env)
    return envs


class TestSmoke:
    def test_multiple_envs_complete_collection(self, trainer):
        envs = _make_envs(trainer, 3)
        group = BatchedEnvGroup(envs, trainer, seeds=[1000, 1001, 1002])
        results = group.collect(n_steps=150)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert len(r["buffer"]) > 0, f"env {i} buffer is empty"
            assert "trainee" in r["last_value"]
            assert math.isfinite(r["last_value"]["trainee"]), f"env {i} last_value not finite"
            rewards = r["buffer"].rewards
            assert all(math.isfinite(x) for x in rewards), f"env {i} has a non-finite reward"

    def test_total_steps_meets_budget(self, trainer):
        """The combined step count across all envs must reach (not
        necessarily exceed by much) the requested n_steps -- mirrors
        rollout_worker.py's own stopping-condition contract."""
        envs = _make_envs(trainer, 2)
        group = BatchedEnvGroup(envs, trainer, seeds=[2000, 2001])
        results = group.collect(n_steps=100)
        total = sum(len(r["buffer"]) for r in results)
        assert total >= 100


class TestPerEnvIsolation:
    def test_gae_computed_independently_per_env(self, trainer):
        """Each env's buffer is a separate RolloutBuffer object -- GAE on
        one must be computable/mutable without touching another's data."""
        envs = _make_envs(trainer, 3)
        group = BatchedEnvGroup(envs, trainer, seeds=[3000, 3001, 3002])
        results = group.collect(n_steps=150)

        buffers = [r["buffer"] for r in results]
        last_values = [r["last_value"] for r in results]
        # Snapshot original reward lists before mutating anything.
        original_rewards = [list(b.rewards) for b in buffers]

        advantages_0, returns_0 = buffers[0].compute_gae(0.98, 0.95, last_values[0])

        # Zero out env 1's and env 2's rewards directly and recompute THEIR
        # GAE -- must not have been able to affect env 0's already-computed
        # result, and env 0's underlying buffer object must be untouched.
        for b in buffers[1:]:
            for i in range(len(b.rewards)):
                b.rewards[i] = 0.0

        assert buffers[0].rewards == original_rewards[0], (
            "env 0's buffer was mutated by touching env 1/2's buffers -- shared state bug"
        )
        advantages_0_again, returns_0_again = buffers[0].compute_gae(0.98, 0.95, last_values[0])
        np.testing.assert_allclose(advantages_0, advantages_0_again)
        np.testing.assert_allclose(returns_0, returns_0_again)

    def test_no_cross_env_contamination_in_group(self, trainer):
        """Two otherwise-identical 3-env groups, differing only in env
        index 1's seed, must produce IDENTICAL buffer contents for envs 0
        and 2 -- proves the batched network call / env-stepping
        orchestration doesn't leak information between environments
        sharing the same round.

        Uses deterministic=True throughout: with real sampling, a
        DIFFERENTLY-seeded env 1 legitimately changes how many values it
        draws from the shared global torch RNG stream within the SAME
        batched call (e.g. VonMises rejection sampling's accept/reject loop
        count depends on env 1's own kappa), which can shift what env 0/2
        draw too -- a real, well-understood, harmless property of batched
        sampling (each row's distribution PARAMETERS are still exactly
        correct and independent -- already proven row-for-row in
        test_sample_action_batch.py's stochastic-mode tests -- only the
        literal draw can shift), NOT cross-env information leakage. Only
        deterministic mode (.mode()/mean, zero RNG draws) isolates the
        question this test actually asks: does the ORCHESTRATION leak
        env 1's ACTUAL DATA into env 0/2's results?
        """
        import random

        N = 20  # small: reduce (not eliminate) the chance of a mid-run reset

        def _seeded_envs(scenario_seeds, rng_seeds):
            envs = _make_envs(trainer, len(scenario_seeds))
            group = BatchedEnvGroup(envs, trainer, seeds=scenario_seeds)
            for env, rs in zip(envs, rng_seeds):
                env.match.player_by_id(env.trainee_player_id).ai._rng = random.Random(rs)
            return group

        group_a = _seeded_envs([5000, 5001, 5002], [111, 222, 333])
        results_a = group_a.collect(n_steps=N, deterministic=True)

        group_b = _seeded_envs([5000, 9999, 5002], [111, 222, 333])  # only index 1's scenario differs
        results_b = group_b.collect(n_steps=N, deterministic=True)

        def _first_done_index(buf) -> int:
            for idx, d in enumerate(buf.dones):
                if d >= 1.0:
                    return idx
            return len(buf) - 1

        for i in (0, 2):
            buf_a, buf_b = results_a[i]["buffer"], results_b[i]["buffer"]
            cutoff = min(_first_done_index(buf_a), _first_done_index(buf_b)) + 1
            np.testing.assert_allclose(
                buf_a.rewards[:cutoff], buf_b.rewards[:cutoff], atol=1e-5,
                err_msg=f"env {i} rewards differ when only env 1's seed changed -- cross-env contamination",
            )
            np.testing.assert_allclose(
                buf_a.log_probs[:cutoff], buf_b.log_probs[:cutoff], atol=1e-5,
                err_msg=f"env {i} log_probs differ when only env 1's seed changed -- cross-env contamination",
            )


class TestMatchesSingleEnvStepping:
    """A group of exactly 1 env must behave identically to stepping that
    SAME env directly (bypassing the group entirely) -- since
    _sample_action_batch(batch=1) is already proven identical to
    _sample_action (test_sample_action_batch.py), this isolates whether
    BatchedEnvGroup's own orchestration (prepare/precompute/step/apply,
    buffer bookkeeping) introduces any discrepancy for the trivial K=1 case."""

    def test_k1_group_matches_manual_stepping(self, trainer):
        import functools
        import random

        det_sample_fn = functools.partial(trainer._sample_action, deterministic=True)
        # Episodes can end after as few as ONE decision (observed min
        # episode length ~0.5s = exactly one decision interval), and after
        # ANY reset -- mid-collection or otherwise -- ScenarioEnv.reset()
        # builds a FRESH NeuralPlayerAI with its own unseeded _rng, so a
        # one-time _rng override before collection only controls the FIRST
        # episode. Comparing past the first reset in either path would be
        # comparing two independently-random episode-2-onward draws, not a
        # real equivalence check -- so this test only compares steps up to
        # (and including) whichever path's first `done` comes first.
        N = 30

        # _make_envs (module-level, see its own docstring) forces a
        # non-neural opponent -- deliberately NOT build_env(_PHASE), which
        # reads ai_config.json's LIVE phase1_opponent_neural_ratio at call
        # time (mostly self-play as of this session). If that rolled a
        # neural opponent, BatchedEnvGroup's buffer would pick up
        # interleaved trainee+secondary rows (via last_secondary_results)
        # that this test's manual stepping loop below never reads --
        # comparing the two buffers index-for-index would then silently
        # compare MISALIGNED rows (trainee vs. secondary), which looks like
        # "flaky float noise" but is actually a structural mismatch that
        # only surfaces when the ambient config happens to roll self-play.
        # Forcing a non-neural opponent makes this test fully independent
        # of that ambient config, matching what it's actually meant to
        # isolate (BatchedEnvGroup's own K=1 orchestration).

        # Path A: through BatchedEnvGroup with K=1. env.sample_action_fn
        # must still be set -- ScenarioEnv.reset() itself checks it to
        # decide whether/how to wire a NeuralPlayerAI onto the trainee at
        # all -- but note collect()'s own deterministic=True is what
        # actually controls sampling behavior; once wired, the assigned
        # NeuralPlayerAI's .sample_action_fn is never called by the group.
        env_a = _make_envs(trainer, 1)[0]
        env_a.sample_action_fn = det_sample_fn
        group = BatchedEnvGroup([env_a], trainer, seeds=[42])
        # Reseed the trainee's observation-noise RNG for a fully controlled comparison.
        env_a.match.player_by_id(env_a.trainee_player_id).ai._rng = random.Random(777)
        results = group.collect(n_steps=N, deterministic=True)
        buf_a = results[0]["buffer"]

        # Path B: manual stepping, mirroring rollout_worker.py's own loop,
        # with the exact same seed/RNG setup.
        env_b = _make_envs(trainer, 1)[0]
        env_b.sample_action_fn = det_sample_fn
        env_b.reset(seed=42)
        env_b.match.player_by_id(env_b.trainee_player_id).ai._rng = random.Random(777)

        from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
        from footballcoach.ai.ppo.ppo_trainer import _action_to_numpy
        buf_b = RolloutBuffer()
        collected = 0
        while collected < N:
            _next_obs, reward, done, info = env_b.step()
            tr = env_b.last_trainee_transition
            if tr is None:
                if done:
                    env_b.reset()
                continue
            buf_b.add(
                obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=float(tr["log_prob"]), value=float(tr["value"]), reward=reward,
                done=1.0 if done else 0.0, bc_label=tr.get("bc_label"),
                head_log_probs=tr.get("head_log_probs"),
                reward_comps=dict(getattr(env_b, "last_reward_components", {})),
                step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
            )
            collected += 1
            if done:
                env_b.reset()

        def _first_done_index(buf) -> int:
            for idx, d in enumerate(buf.dones):
                if d >= 1.0:
                    return idx
            return len(buf) - 1

        cutoff = min(_first_done_index(buf_a), _first_done_index(buf_b)) + 1
        assert cutoff >= 1, "first episode ended with zero recorded steps -- widen N or pick a different seed"

        # NOTE: K=1 here means BatchedEnvGroup's own _sample_action_batch
        # call is ALSO batch-of-1 (only one env in the group) -- and
        # trainer._sample_action(deterministic=True) (env_b's path)
        # literally calls _sample_action_batch(batch=1) internally too (see
        # test_sample_action_batch.py's docstring). Both paths are the
        # SAME batch shape, so unlike a real K>1-vs-K=1 comparison there's
        # no floating-point non-associativity excuse available here --
        # these should be bit-identical (tight tolerance only for float
        # repr slop, not batching-shape differences).
        np.testing.assert_allclose(buf_a.rewards[:cutoff], buf_b.rewards[:cutoff], atol=1e-5)
        np.testing.assert_allclose(buf_a.log_probs[:cutoff], buf_b.log_probs[:cutoff], atol=1e-5)
        np.testing.assert_allclose(buf_a.values[:cutoff], buf_b.values[:cutoff], atol=1e-5)
        np.testing.assert_allclose(buf_a.dones[:cutoff], buf_b.dones[:cutoff], atol=1e-6)


class TestTrainBatchedParallelStepBudget:
    """Regression test for a real bug found via end-to-end benchmarking:
    ``PPOTrainer._train_batched_parallel`` divided the per-round, per-WORKER
    step budget by ``total_envs`` (``n_parallel_envs``, the config's old,
    now-renamed single overloaded "total envs" knob) instead of
    ``n_processes`` (unlike ``_train_parallel``'s own ``steps_per_worker =
    rollout_steps // n_workers`` convention, which this was supposed to
    mirror). Since ``BatchedEnvGroup.collect(n_steps)`` treats ``n_steps`` as
    the budget for the WHOLE process (aggregated across all
    ``envs_per_process`` envs it owns), dividing by ``total_envs`` made that
    budget ``envs_per_process`` times too small -- forcing that many times
    more outer rounds (more weight broadcasts + PPO updates per unit of
    data) than intended, and diluting/reversing any throughput benefit from
    batching. ``n_processes``/``envs_per_process`` are now two independent
    config keys (no more derived-by-division "total envs" knob, precisely to
    avoid this class of confusion -- see ai_config.json's own comments), but
    the underlying arithmetic bug this guards against is the same shape. With
    ``envs_per_process=3`` this test's config needs 3 outer rounds under the
    bug vs. 1 after the fix -- a large enough gap to catch a regression
    reliably despite real run-to-run episode-length noise.
    """

    def test_one_round_suffices_when_evenly_divisible(self):
        import copy

        from footballcoach.ai.config import load_ai_config
        from footballcoach.ai.models.decision_network import DecisionNetwork
        from footballcoach.ai.models.execution_network import ExecutionNetwork

        n_processes = 2
        envs_per_process = 3
        total_envs = n_processes * envs_per_process  # 6
        rollout_steps = 60  # rollout_steps // n_processes == 30 (correct);
                             # rollout_steps // total_envs == 10 (buggy) --
                             # 3x fewer steps/round than intended either way,
                             # so the bug forces ~3 outer rounds here vs. 1.

        cfg = copy.deepcopy(load_ai_config())
        cfg["ppo"]["n_processes"] = n_processes
        cfg["ppo"]["batched_rollout"] = True
        cfg["ppo"]["envs_per_process"] = envs_per_process
        cfg["ppo"]["worker_torch_threads"] = 1
        cfg["ppo"]["rollout_steps"] = rollout_steps
        cfg["eval"]["eval_n_parallel_workers"] = 1
        trainer = PPOTrainer(
            decision_net=DecisionNetwork.from_config(),
            execution_net=ExecutionNetwork.from_config(),
            cfg=cfg,
            device=torch.device("cpu"),
            separate_value_net=True,
        )
        trainer.rollout_eval_trials = 0

        trainer.train(env=None, total_steps=rollout_steps, phase_id=1)

        # One correctly-sized round overshoots by at most ~total_envs (each
        # env's own round-completion adds one more transition than strictly
        # needed before the "collected >= n_steps" check trips). Three
        # buggy rounds compound that per round -- a generous-but-discriminating
        # bound that would have failed under the pre-fix arithmetic.
        assert trainer._total_steps < rollout_steps + total_envs + 1, (
            f"_total_steps={trainer._total_steps} overshot far more than one "
            f"round's worth of slack for rollout_steps={rollout_steps} -- "
            "suggests _train_batched_parallel needed multiple outer rounds "
            "instead of one, i.e. the steps_per_worker regression is back."
        )


class TestNotDueEnvHandledGracefully:
    """Regression test for a real crash found in PRODUCTION training (not a
    benchmark): ScenarioEnv.step() has a legitimate early-exit (trainee
    already in the opponent box with possession -- further ticks would only
    accumulate spurious negative progress) that can end a decision interval
    several ticks short. When that happens, the trainee's NeuralPlayerAI is
    genuinely not due for a decision at the start of collect()'s next round
    -- the ORIGINAL version of this module asserted every env must always be
    due and crashed the instant this fired for real. This directly forces
    that exact starting state (rather than hoping to trigger the rare
    in-box-with-possession condition naturally, which would make this test
    flaky-by-construction) and checks BOTH that collect() no longer crashes,
    AND that the not-due env's resulting transitions are IDENTICAL to
    stepping that same env unbatched from the same starting state -- the
    real correctness bar this fix has to clear, not just "doesn't raise."
    """

    @staticmethod
    def _force_not_due(trainer, env, ticks_since_decision: int) -> None:
        """Simulate ScenarioEnv.step()'s early-exit: a REAL decision already
        happened (so _last_gating is populated, exactly like production --
        the early-exit only ever fires partway THROUGH an interval that
        already had its decision tick), then _ticks_since_decision is wound
        back down to simulate the interval having ended several ticks short
        of the full decision_interval_ticks. Forcing "not due" on a freshly
        -reset env instead (skipping the real decision) would leave
        _last_gating=None, an impossible state in practice that just trips
        Match._apply_movement()'s "no movement intent this tick" guard for
        an unrelated reason -- not what this test is trying to exercise."""
        player = env.match.player_by_id(env.trainee_player_id)
        ai = player.ai
        obs_dict = ai.prepare(player, env.match, 0)
        assert obs_dict is not None, "test setup bug: expected a fresh reset to be due immediately"
        result = trainer._sample_action(obs_dict, deterministic=True)
        ai.apply(player, env.match, result)
        assert ai._last_gating is not None
        ai._ticks_since_decision = ticks_since_decision
        assert not ai.is_due_for_decision(), "test setup bug: forced state is already due"

    def test_collect_does_not_crash_when_one_env_is_not_due(self, trainer):
        envs = _make_envs(trainer, 3)
        group = BatchedEnvGroup(envs, trainer, seeds=[8000, 8001, 8002])
        self._force_not_due(trainer, envs[1], ticks_since_decision=3)

        results = group.collect(n_steps=60, deterministic=True)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert len(r["buffer"]) > 0, f"env {i} buffer is empty"
            assert all(math.isfinite(x) for x in r["buffer"].rewards), f"env {i} has a non-finite reward"

    def test_not_due_env_matches_unbatched_stepping(self, trainer):
        """The env that starts not-due must produce EXACTLY the transitions
        a normal, unbatched env.step() loop would from the same state --
        not just avoid crashing."""
        import functools
        import random

        from footballcoach.ai.ppo.ppo_trainer import _action_to_numpy
        from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer

        N = 40
        FORCED_TICKS_SINCE_DECISION = 3
        det_fn = functools.partial(trainer._sample_action, deterministic=True)

        # _make_envs forces a non-neural opponent -- deliberately NOT
        # build_env(_PHASE) (reads ai_config.json's LIVE
        # phase1_opponent_neural_ratio at call time; see
        # test_k1_group_matches_manual_stepping's identical note above for
        # why a neural-opponent roll would silently misalign these two
        # buffers via interleaved secondary rows).

        # Path A: through the group. seeds=[9000] resets it; _rng is seeded
        # BEFORE the priming decision so that decision (which consumes
        # encode_observation's sensor-noise RNG) is fully reproducible
        # against path B below. _force_not_due then runs ONE real decision
        # (populating _last_gating exactly like production) before winding
        # _ticks_since_decision back down to simulate the early-exit.
        env_a = _make_envs(trainer, 1)[0]
        env_a.sample_action_fn = det_fn
        group = BatchedEnvGroup([env_a], trainer, seeds=[9000])
        env_a.match.player_by_id(env_a.trainee_player_id).ai._rng = random.Random(555)
        self._force_not_due(trainer, env_a, FORCED_TICKS_SINCE_DECISION)
        results = group.collect(n_steps=N, deterministic=True)
        buf_a = results[0]["buffer"]

        # Path B: manual, unbatched stepping -- identical seed/_rng/priming
        # recipe, same ordering, so it reaches the IDENTICAL forced state.
        env_b = _make_envs(trainer, 1)[0]
        env_b.sample_action_fn = det_fn
        env_b.reset(seed=9000)
        env_b.match.player_by_id(env_b.trainee_player_id).ai._rng = random.Random(555)
        self._force_not_due(trainer, env_b, FORCED_TICKS_SINCE_DECISION)

        buf_b = RolloutBuffer()
        collected = 0
        while collected < N:
            _next_obs, reward, done, info = env_b.step()
            tr = env_b.last_trainee_transition
            if tr is None:
                if done:
                    env_b.reset()
                continue
            buf_b.add(
                obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=float(tr["log_prob"]), value=float(tr["value"]), reward=reward,
                done=1.0 if done else 0.0, bc_label=tr.get("bc_label"),
                head_log_probs=tr.get("head_log_probs"),
                reward_comps=dict(getattr(env_b, "last_reward_components", {})),
                step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
            )
            collected += 1
            if done:
                env_b.reset()

        def _first_done_index(buf) -> int:
            for idx, d in enumerate(buf.dones):
                if d >= 1.0:
                    return idx
            return len(buf) - 1

        cutoff = min(_first_done_index(buf_a), _first_done_index(buf_b)) + 1
        assert cutoff >= 1, "first episode ended with zero recorded steps -- widen N or pick a different seed"

        # NOTE: K=1 here means BatchedEnvGroup's own _sample_action_batch
        # call is ALSO batch-of-1 (only one env in the group) -- and
        # trainer._sample_action(deterministic=True) (env_b's path)
        # literally calls _sample_action_batch(batch=1) internally too (see
        # test_sample_action_batch.py's docstring). Both paths are the
        # SAME batch shape, so unlike a real K>1-vs-K=1 comparison there's
        # no floating-point non-associativity excuse available here --
        # these should be bit-identical (tight tolerance only for float
        # repr slop, not batching-shape differences).
        np.testing.assert_allclose(buf_a.rewards[:cutoff], buf_b.rewards[:cutoff], atol=1e-5)
        np.testing.assert_allclose(buf_a.log_probs[:cutoff], buf_b.log_probs[:cutoff], atol=1e-5)
        np.testing.assert_allclose(buf_a.values[:cutoff], buf_b.values[:cutoff], atol=1e-5)
        np.testing.assert_allclose(buf_a.dones[:cutoff], buf_b.dones[:cutoff], atol=1e-6)


class TestChunkedStreaming:
    """Coverage for collect(chunk_steps=..., on_chunk=...) -- fixes a real
    production MemoryError (ai/ppo/batched_rollout_worker.py's "Chunked
    streaming" docstring section): instead of one giant end-of-rollout
    pickle+send, results are periodically flushed in smaller pieces. The
    default (chunk_steps=None) path is exercised by every other test in
    this file already; these specifically guard the NEW opt-in behavior.
    """

    def test_on_chunk_called_multiple_times(self, trainer):
        # NOTE: on_chunk's dicts hold self._buffers[i] BY REFERENCE, not a
        # copy -- a later flush .clear()s the same object in place (see
        # collect()'s own docstring warning). A real consumer pickles
        # (conn.send) immediately; this test must extract the row COUNT
        # immediately too, rather than holding onto the dicts themselves.
        envs = _make_envs(trainer, 3)
        group = BatchedEnvGroup(envs, trainer, seeds=[9000, 9001, 9002])
        chunk_row_counts: list[int] = []
        result = group.collect(
            n_steps=300, chunk_steps=50,
            on_chunk=lambda chunk: chunk_row_counts.append(sum(len(r["buffer"]) for r in chunk)),
        )

        assert result == [], "chunked mode should return [] -- everything already delivered via on_chunk"
        assert len(chunk_row_counts) >= 2, "expected multiple flushes for n_steps=300, chunk_steps=50"
        assert sum(chunk_row_counts) >= 300

    def test_aggregate_stats_close_to_unchunked(self, trainer):
        """Same seeds, same total step budget, deterministic sampling --
        chunking must not change episode count/outcome/reward totals (only
        individual advantage values for episodes straddling a flush
        boundary are expected to differ, see the module docstring).

        Every episode AFTER the first draws its own fresh seed via
        collect()'s _next_episode_seed() (Python's GLOBAL random module,
        since replay_seeds isn't used here) -- the two collect() calls
        below must each start from the SAME global random state or their
        episodes beyond the first would legitimately diverge for reasons
        having nothing to do with chunking.
        """
        import random
        seeds = [9100, 9101, 9102]

        envs_unchunked = _make_envs(trainer, 3)
        group_unchunked = BatchedEnvGroup(envs_unchunked, trainer, seeds=seeds)
        random.seed(424242)
        result_unchunked = group_unchunked.collect(
            n_steps=400, deterministic=True, deterministic_decision=True, deterministic_direction=True,
        )

        envs_chunked = _make_envs(trainer, 3)
        group_chunked = BatchedEnvGroup(envs_chunked, trainer, seeds=seeds)
        random.seed(424242)
        # Extract episode_rewards immediately in the callback -- see the
        # note in test_on_chunk_called_multiple_times above (r["stats"] is
        # a freshly-reassigned dict per flush so it's actually safe to hold
        # onto here, but r["buffer"] is not; extracting immediately either
        # way is the correct pattern any real on_chunk consumer must follow).
        rewards_chunked: list[float] = []
        group_chunked.collect(
            n_steps=400, deterministic=True, deterministic_decision=True, deterministic_direction=True,
            chunk_steps=60,
            on_chunk=lambda chunk: rewards_chunked.extend(
                rew for r in chunk for rew in r["stats"]["episode_rewards"]
            ),
        )

        rewards_unchunked = sorted(
            rew for r in result_unchunked for rew in r["stats"]["episode_rewards"]
        )
        rewards_chunked = sorted(rewards_chunked)
        assert len(rewards_unchunked) > 0 and len(rewards_chunked) > 0
        # Not exact equality -- see module docstring on chunk-boundary GAE
        # truncation for straddling episodes; reward totals themselves are
        # computed independently of chunk boundaries (episode_reward_accum
        # persists across flushes) so should match very closely regardless.
        assert abs(sum(rewards_unchunked) - sum(rewards_chunked)) < 1e-6

    def test_chunk_steps_none_is_unchanged(self, trainer):
        """Regression guard: omitting chunk_steps must behave exactly as
        before this feature existed (on_chunk never called, full list
        returned)."""
        envs = _make_envs(trainer, 2)
        group = BatchedEnvGroup(envs, trainer, seeds=[9200, 9201])
        called = []
        result = group.collect(n_steps=100, on_chunk=called.append)
        assert called == [], "on_chunk must never be invoked when chunk_steps is None"
        assert len(result) == 2
        assert all(len(r["buffer"]) > 0 for r in result)

    def test_build_results_skips_empty_buffers(self, trainer):
        """Direct unit test of the empty-buffer guard: an env with zero
        rows collected since the last flush must be silently skipped, not
        crash RolloutBuffer.as_tensors()'s self.obs[0] indexing."""
        envs = _make_envs(trainer, 3)
        group = BatchedEnvGroup(envs, trainer, seeds=[9300, 9301, 9302])
        group.collect(n_steps=60)  # populate real data into all 3 buffers
        group._buffers[1].clear()  # simulate "nothing new for env 1 this flush"

        stats = [{"episode_rewards": []} for _ in range(3)]
        results = group._build_results(range(3), stats)

        result_buffer_ids = {id(r["buffer"]) for r in results}
        assert len(results) == 2, "the empty-buffer env must be skipped, not included"
        assert id(group._buffers[1]) not in result_buffer_ids
