"""Coverage for BatchedEnvGroup + a NEURAL secondary opponent (self-play)
together -- the exact combination that had ZERO test coverage before this
file: test_secondary_opponent_gae.py's own docstring only claims the
single-process PPOTrainer.train() path is covered, and doesn't even mention
ai/ppo/batched_rollout_worker.py -- it predates the batched-rollout
mechanism entirely. batched_rollout_worker.py's own module docstring notes
secondary neural players are "out of scope for batching" (they still decide
via their own NeuralPlayerAI.act() synchronously, one row at a time, NOT
through the vectorized/precomputed trainee path) -- this file is what
actually verifies that claim holds up: that a BatchedEnvGroup running SEVERAL
self-play envs at once produces well-formed, per-env-isolated, finite data
for BOTH tracks, and that PPOTrainer._ppo_update() (the real update loop,
not a stand-in) stays finite when fed a MERGED batch spanning several
self-play envs' trainee+opponent rows together -- mirroring exactly how
PPOTrainer._train_batched_parallel() itself merges multiple worker
processes' per-env buffers into one PPO update.

Mirrors test_secondary_opponent_gae.py's pattern (explicit
opponent_rules_prob=0.0/opponent_immobile_prob=0.0 scenario construction,
not ai_config.json's current ratios -- see that file's own docstring for
why: this repo's test suite runs under pytest-xdist, so mutating the shared
config file mid-run risks a concurrent test process picking up the wrong
curriculum ratios). Does not attempt to cover the subprocess-spawning layer
(spawn_batched_workers) with a forced self-play ratio, for the same reason
test_parallel_rollout.py/test_secondary_opponent_gae.py don't: each spawned
worker rebuilds its own env via curriculum.envs.build_env(), which reads
ai_config.json fresh in a fresh interpreter, so it can't be overridden
without touching the shared file. BatchedEnvGroup + _ppo_update together
(this file's actual scope) is the part that broke in production and the
part worth guarding directly; the Pipe/subprocess plumbing on top is
already covered generically by test_batched_rollout.py's existing smoke
tests (with a rules/immobile-only opponent).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

from footballcoach.ai.curriculum.envs import bc_label_fn_for_phase_player
from footballcoach.ai.ppo.batched_rollout_worker import BatchedEnvGroup
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _merge_worker_batches
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS_PER_ENV = 40


@pytest.fixture(scope="module")
def trainer():
    # NOT inference_only=True (unlike test_batched_rollout.py's own fixture)
    # -- this file's _ppo_update test needs a real optimizer, which
    # inference_only=True deliberately skips constructing.
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


def _make_self_play_envs(trainer, n: int, deterministic_secondary: bool = False) -> list[ScenarioEnv]:
    """n envs, each FORCED to a fully-neural secondary opponent every
    episode (opponent_rules_prob=0.0, opponent_immobile_prob=0.0) --
    matches test_secondary_opponent_gae.py's own fixture, just built n
    times for a real multi-env BatchedEnvGroup instead of one env.

    env.sample_action_fn MUST be set before BatchedEnvGroup's constructor
    calls env.reset() -- ScenarioEnv.reset() checks it to decide whether to
    wire a NeuralPlayerAI onto the trainee at all (see
    batched_rollout_worker.py's own module docstring); without it the
    trainee's .ai stays None and every prepare()/is_due_for_decision() call
    crashes with AttributeError.

    deterministic_secondary: collect()'s own deterministic= kwarg ONLY
    governs the group's precomputed TRAINEE batch calls (see its
    docstring) -- the secondary/opponent player decides via its own
    NeuralPlayerAI.act() -> env.sample_action_fn fallback path, entirely
    outside the group's control, and by default that samples stochastically
    from the shared GLOBAL torch RNG. That's fine for realistic/production-
    shaped tests, but makes any "only env N's seed differs" isolation test
    misleading: a differently-seeded env consumes a different number of
    global RNG draws for ITS OWN secondary player, which shifts what's left
    for every OTHER env's later rounds -- a real, harmless, already-
    documented property of shared-RNG-stream sampling (see
    test_batched_rollout.py's own no-cross-env-contamination test
    docstring), not data leakage, but it must be eliminated (not just
    tolerated) for a test whose entire point is proving isolation. Set True
    for that use case.
    """
    envs = []
    for _ in range(n):
        defn = ScenarioDefinition(
            key="batched_self_play_test", label="Batched self-play test",
            description="1v1 with a fully neural secondary opponent, for BatchedEnvGroup",
            build=functools.partial(
                build_1v1_scenario, opponent_rules_prob=0.0, opponent_immobile_prob=0.0,
            ),
        )
        env = ScenarioEnv(
            definition=defn, trainee_player_id="trainee", phase=1,
            secondary_player_ids=["opponent"], max_episode_s=30.0,
        )
        env.sample_action_fn = (
            functools.partial(trainer._sample_action, deterministic=True)
            if deterministic_secondary else trainer._sample_action
        )
        envs.append(env)
    return envs


class TestBatchedGroupWithSelfPlay:
    def test_every_env_produces_both_tracks(self, trainer):
        """Sanity check on the fixture itself: with a BatchedEnvGroup of
        several self-play envs, EVERY env's own buffer must end up with
        both "trainee" and "opponent" rows -- otherwise the rest of this
        file is vacuously testing the single-track (no-self-play) case."""
        envs = _make_self_play_envs(trainer, 4)
        for env in envs:
            env.bc_label_fn = bc_label_fn_for_phase_player(1)
        group = BatchedEnvGroup(envs, trainer, seeds=[11000, 11001, 11002, 11003])
        results = group.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        assert len(results) == 4
        for i, r in enumerate(results):
            track_ids = set(r["buffer"].track_ids)
            assert track_ids == {"trainee", "opponent"}, (
                f"env {i}: expected both tracks, got {track_ids} -- "
                "widen _ROLLOUT_STEPS_PER_ENV or check the fixture"
            )

    def test_no_cross_env_track_contamination(self, trainer):
        """Two groups differing only in env index 1's scenario seed must
        produce IDENTICAL data for envs 0/2's BOTH tracks (trainee AND
        opponent) -- the self-play-specific version of
        test_batched_rollout.py's own no-cross-env-contamination test.
        Batching interleaves every env's trainee decision into one network
        call per round; if that orchestration ever leaked a ROW from one
        env's opponent into another env's buffer, this would catch it."""
        import random

        def _seeded_envs(scenario_seeds, rng_seeds):
            envs = _make_self_play_envs(trainer, len(scenario_seeds), deterministic_secondary=True)
            for env in envs:
                env.bc_label_fn = bc_label_fn_for_phase_player(1)
            group = BatchedEnvGroup(envs, trainer, seeds=scenario_seeds)
            for env, rs in zip(envs, rng_seeds):
                env.match.player_by_id(env.trainee_player_id).ai._rng = random.Random(rs)
                env.match.player_by_id("opponent").ai._rng = random.Random(rs + 1)
            return group

        group_a = _seeded_envs([12000, 12001, 12002], [200, 300, 400])
        results_a = group_a.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        group_b = _seeded_envs([12000, 99999, 12002], [200, 300, 400])  # only index 1 differs
        results_b = group_b.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        def _first_done_index(buf) -> int:
            for idx, d in enumerate(buf.dones):
                if d >= 1.0:
                    return idx
            return len(buf) - 1

        for i in (0, 2):
            buf_a, buf_b = results_a[i]["buffer"], results_b[i]["buffer"]
            cutoff = min(_first_done_index(buf_a), _first_done_index(buf_b)) + 1
            assert buf_a.track_ids[:cutoff] == buf_b.track_ids[:cutoff], (
                f"env {i}: track_id sequence differs when only env 1 changed"
            )
            np.testing.assert_allclose(
                buf_a.rewards[:cutoff], buf_b.rewards[:cutoff], atol=1e-5,
                err_msg=f"env {i} rewards differ when only env 1's seed changed -- cross-env contamination",
            )
            np.testing.assert_allclose(
                buf_a.log_probs[:cutoff], buf_b.log_probs[:cutoff], atol=1e-5,
                err_msg=f"env {i} log_probs differ when only env 1's seed changed -- cross-env contamination",
            )

    def test_ppo_update_on_merged_multi_env_self_play_batch_produces_finite_losses(self, trainer):
        """The real regression target: mirrors exactly what
        _train_batched_parallel() does every rollout -- collect from several
        self-play envs via ONE BatchedEnvGroup, bootstrap + GAE each env's
        buffer independently, merge into ONE batch (_merge_worker_batches,
        the same helper the real training loop uses), then run a REAL
        trainer._ppo_update() over it. Would NOT have caught a rare,
        low-probability instability by construction (this is a single,
        deterministic-mode pass, not a fuzz/stress run) -- but DOES lock in
        that the ordinary, common-case path stays finite and shaped
        correctly, and gives future stress-testing something fast and
        targeted to run repeatedly instead of a full-scale multi-hour
        production reproduction.
        """
        envs = _make_self_play_envs(trainer, 5)
        for env in envs:
            env.bc_label_fn = bc_label_fn_for_phase_player(1)
        group = BatchedEnvGroup(envs, trainer, seeds=[13000, 13001, 13002, 13003, 13004])
        results = group.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        worker_batches = []
        for i, env in enumerate(envs):
            r = results[i]
            assert set(r["buffer"].track_ids) == {"trainee", "opponent"}, (
                f"env {i} fixture didn't produce both tracks this run -- "
                "rerun or widen _ROLLOUT_STEPS_PER_ENV"
            )
            advantages, returns = r["buffer"].compute_gae(trainer.gamma, trainer.lam, r["last_value"])
            for a, ret in zip(advantages, returns):
                assert math.isfinite(a), f"env {i}: non-finite advantage"
                assert math.isfinite(ret), f"env {i}: non-finite return"
            worker_batches.append(r["buffer"].as_tensors(advantages, returns))

        batch = _merge_worker_batches(worker_batches)
        assert "track_ids" in batch, "track_ids must survive _merge_worker_batches"
        assert len(batch["track_ids"]) == int(batch["rewards"].shape[0])
        assert set(batch["track_ids"]) == {"trainee", "opponent"}

        metrics = trainer._ppo_update(batch, progress=0.0)
        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            assert math.isfinite(val), f"PPO metrics['{key}']={val} is not finite"
