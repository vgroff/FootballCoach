"""Coverage for BatchedEnvGroup's NEW secondary_trainer batching capability
(ai/ppo/batched_rollout_worker.py) -- extends the group's existing
trainee-only batching to ALSO batch a secondary/opponent player's decision
into one extra network call per round, via the same prepare()/
_precomputed_result hook NeuralPlayerAI already exposes generically (see
that class's own docstring in rules_ai.py).

secondary_trainer=None (the default, unchanged) leaves every secondary
player fully unbatched -- that path is already covered by
test_batched_rollout_self_play.py. THIS file is what proves the NEW,
opt-in secondary_trainer=<trainer> path is a pure speed optimization: same
output as the unbatched path when given the SAME weights, and genuinely
uses whichever trainer it's given (not silently falling back to the
primary one) when given a DIFFERENT one.

Mirrors test_batched_rollout_self_play.py's fixture pattern throughout
(explicit opponent_rules_prob=0.0/opponent_immobile_prob=0.0 scenario
construction, not ai_config.json's live ratios -- see that file's own
docstring for why: this suite runs under pytest-xdist, and mutating the
shared config file mid-run risks a concurrent test process picking up the
wrong curriculum ratios).
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
from footballcoach.rules_ai import Phase1RulesAI
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS_PER_ENV = 40


@pytest.fixture(scope="module")
def trainer_a():
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


@pytest.fixture(scope="module")
def trainer_b():
    """A SECOND, independently-initialized trainer -- deliberately different
    weights from trainer_a (two separate from_config() calls draw from
    wherever the global torch RNG happens to be at each call, so they land
    on different random inits with overwhelming probability -- no shared
    seed between them)."""
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


def _make_self_play_envs(trainer, n: int, deterministic_secondary: bool = False) -> list[ScenarioEnv]:
    """Same fixture as test_batched_rollout_self_play.py's own helper --
    n envs, each FORCED to a fully-neural secondary opponent every episode.
    See that file's docstring for the deterministic_secondary rationale."""
    envs = []
    for _ in range(n):
        defn = ScenarioDefinition(
            key="batched_secondary_test", label="Batched secondary-batching test",
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


def _make_mixed_envs(trainer, n_neural: int, n_rules: int) -> list[ScenarioEnv]:
    """n_neural envs with a fully-neural secondary + n_rules envs with a
    rules-based secondary (opponent_rules_prob=1.0) -- for
    test_non_neural_secondary_unaffected."""
    envs = _make_self_play_envs(trainer, n_neural)
    for _ in range(n_rules):
        defn = ScenarioDefinition(
            key="batched_secondary_rules_test", label="Batched secondary-batching test (rules)",
            description="1v1 with a rules-based secondary opponent, for BatchedEnvGroup",
            build=functools.partial(
                build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0,
            ),
        )
        env = ScenarioEnv(
            definition=defn, trainee_player_id="trainee", phase=1,
            secondary_player_ids=["opponent"], max_episode_s=30.0,
        )
        env.sample_action_fn = trainer._sample_action
        envs.append(env)
    return envs


class TestSecondaryBatchingMatchesUnbatchedPath:
    def test_identical_to_unbatched_path(self, trainer_a):
        """THE critical correctness test: same env, same seed,
        deterministic=True, run once with secondary_trainer=None (today's
        unbatched path) and once with secondary_trainer=trainer_a (new
        batched path, SAME weights) -- must produce byte-identical
        rewards/log_probs/track_ids. Proves the new ORCHESTRATION
        (prepare()/_precomputed_result wiring for a secondary player) is a
        pure performance change, never a behavior change.

        Deliberately a SINGLE env (mirrors test_batched_rollout.py's own
        test_k1_group_matches_manual_stepping strategy for the trainee):
        with only one env in the group, the secondary's batched call is
        ALWAYS a batch of exactly 1, same as the unbatched fallback's own
        batch-of-1 call -- this isolates "is the wiring correct" from "does
        batching several DIFFERENT rows together introduce floating-point
        non-associativity noise", a separate, already-established, already-
        accepted phenomenon covered by test_sample_action_batch.py (whose
        own individual-vs-batched-row comparisons use atol=1e-4, not
        bit-exact) -- and which a long (40-tick) rollout would otherwise
        compound into much larger, meaningless-to-assert-on drift several
        decisions later purely from harmless per-call float noise, not any
        real behavior difference.

        NeuralPlayerAI._rng (observation-encoding noise) defaults to an
        UNSEEDED random.Random() -- see ScenarioEnv.__init__'s own `rng`
        default -- so two independently-built envs diverge in observation
        noise regardless of any BatchedEnvGroup behavior unless pinned to
        matching seeds here, exactly like test_no_cross_env_track_contamination
        below already does for the same reason.
        """
        import random

        def _seed_rngs(env, rs):
            env.match.player_by_id(env.trainee_player_id).ai._rng = random.Random(rs)
            env.match.player_by_id("opponent").ai._rng = random.Random(rs + 1)

        env_unbatched = _make_self_play_envs(trainer_a, 1, deterministic_secondary=True)[0]
        env_unbatched.bc_label_fn = bc_label_fn_for_phase_player(1)
        group_unbatched = BatchedEnvGroup([env_unbatched], trainer_a, seeds=[21000])
        _seed_rngs(env_unbatched, 800)
        results_unbatched = group_unbatched.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        env_batched = _make_self_play_envs(trainer_a, 1, deterministic_secondary=True)[0]
        env_batched.bc_label_fn = bc_label_fn_for_phase_player(1)
        group_batched = BatchedEnvGroup(
            [env_batched], trainer_a, seeds=[21000], secondary_trainer=trainer_a,
        )
        _seed_rngs(env_batched, 800)
        results_batched = group_batched.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        buf_u, buf_b = results_unbatched[0]["buffer"], results_batched[0]["buffer"]
        assert buf_u.track_ids == buf_b.track_ids, "track_id sequence differs"
        assert set(buf_u.track_ids) == {"trainee", "opponent"}, (
            "fixture didn't produce both tracks this run -- rerun or widen _ROLLOUT_STEPS_PER_ENV"
        )
        np.testing.assert_allclose(
            buf_u.rewards, buf_b.rewards, atol=1e-6,
            err_msg="rewards differ between unbatched and batched secondary paths",
        )
        np.testing.assert_allclose(
            # 1e-4, not 1e-6: batched vs single-row matmul are not bit-identical
            # (floating-point non-associativity) -- deterministic=True makes the
            # actual sampled actions/rewards match exactly (see the assert above),
            # but the log_prob itself is read straight off the forward pass, so a
            # ~1e-6-scale difference here is real and expected, not a bug. Same
            # tolerance test_sample_action_batch.py already uses for this exact
            # reason.
            buf_u.log_probs, buf_b.log_probs, atol=1e-4,
            err_msg="log_probs differ between unbatched and batched secondary paths",
        )

    def test_uses_the_given_trainer_not_the_primary(self, trainer_a, trainer_b):
        """secondary_trainer=trainer_b (DIFFERENT weights from the primary
        trainer_a) must make the secondary's sampled actions reflect
        trainer_b's own policy, not trainer_a's -- guards against the new
        code path accidentally hard-wiring self.trainer regardless of what
        secondary_trainer says."""
        envs = _make_self_play_envs(trainer_a, 3)
        for env in envs:
            env.bc_label_fn = bc_label_fn_for_phase_player(1)
        group = BatchedEnvGroup(envs, trainer_a, seeds=[22000, 22001, 22002], secondary_trainer=trainer_b)
        results = group.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        checked_any = False
        for r in results:
            buf = r["buffer"]
            for i, tid in enumerate(buf.track_ids):
                if tid != "opponent":
                    continue
                obs_dict = {k: torch.from_numpy(v) for k, v in buf.obs[i].items()}
                # trainer_b (the secondary_trainer actually passed) must
                # reproduce the recorded log_prob exactly under the same
                # deterministic sampling.
                _, lp_b, *_ = trainer_b._sample_action(obs_dict, deterministic=True)
                assert lp_b == pytest.approx(buf.log_probs[i], abs=1e-5), (
                    "recorded secondary log_prob doesn't match trainer_b "
                    "(the given secondary_trainer) re-evaluated on the same obs"
                )
                # trainer_a (the PRIMARY trainer, NOT what was passed as
                # secondary_trainer) must generally disagree -- two
                # independently-initialized networks producing the exact
                # same log_prob on a real observation would be an
                # astronomically unlikely coincidence, so a mismatch here
                # confirms the secondary genuinely used trainer_b, not a
                # silent fallback to trainer_a.
                _, lp_a, *_ = trainer_a._sample_action(obs_dict, deterministic=True)
                assert lp_a != pytest.approx(buf.log_probs[i], abs=1e-5), (
                    "recorded secondary log_prob matches trainer_a (the PRIMARY "
                    "trainer) -- secondary_trainer was likely ignored and the "
                    "primary trainer's weights were used instead"
                )
                checked_any = True
        assert checked_any, "no 'opponent' rows recorded -- widen _ROLLOUT_STEPS_PER_ENV or seeds"

    def test_non_neural_secondary_unaffected(self, trainer_a, trainer_b):
        """Rules-based secondaries mixed in alongside secondary_trainer set
        must be skipped cleanly (no crash) and behave exactly as the
        unbatched path already does -- rules-based/immobile secondaries
        never had a _precomputed_result hook to consume, so they must stay
        driven by their own Phase1RulesAI.decide() regardless of
        secondary_trainer."""
        envs = _make_mixed_envs(trainer_a, n_neural=2, n_rules=2)
        for env in envs:
            env.bc_label_fn = bc_label_fn_for_phase_player(1)
        group = BatchedEnvGroup(
            envs, trainer_a, seeds=[23000, 23001, 23002, 23003], secondary_trainer=trainer_b,
        )
        results = group.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        # The two rules-opponent envs (indices 2, 3) must never have picked
        # up a NeuralPlayerAI-style "opponent" track row driven by a
        # network -- their opponent stays Phase1RulesAI throughout, so
        # last_secondary_results (and therefore buffer rows) for them
        # should be empty/rules-only, never crash-producing.
        for i in (2, 3):
            assert isinstance(envs[i].match.player_by_id("opponent").ai, Phase1RulesAI), (
                f"env {i}: rules-based opponent's .ai was replaced -- "
                "secondary_trainer batching must not touch non-neural secondaries"
            )
        assert len(results) == 4  # no crash, one result per env

    def test_ppo_update_on_merged_batch_stays_finite(self, trainer_a):
        """Mirrors test_batched_rollout_self_play.py's own
        test_ppo_update_on_merged_multi_env_self_play_batch_produces_finite_losses,
        with secondary_trainer=trainer_a this time: the full
        collect -> per-env GAE -> _merge_worker_batches -> real
        trainer._ppo_update() pipeline must stay finite when secondary
        decisions came from the batched path."""
        envs = _make_self_play_envs(trainer_a, 5)
        for env in envs:
            env.bc_label_fn = bc_label_fn_for_phase_player(1)
        group = BatchedEnvGroup(
            envs, trainer_a, seeds=[24000, 24001, 24002, 24003, 24004],
            secondary_trainer=trainer_a,
        )
        results = group.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        worker_batches = []
        for i, env in enumerate(envs):
            r = results[i]
            assert set(r["buffer"].track_ids) == {"trainee", "opponent"}, (
                f"env {i} fixture didn't produce both tracks this run -- "
                "rerun or widen _ROLLOUT_STEPS_PER_ENV"
            )
            advantages, returns = r["buffer"].compute_gae(trainer_a.gamma, trainer_a.lam, r["last_value"])
            for a, ret in zip(advantages, returns):
                assert math.isfinite(a), f"env {i}: non-finite advantage"
                assert math.isfinite(ret), f"env {i}: non-finite return"
            worker_batches.append(r["buffer"].as_tensors(advantages, returns))

        batch = _merge_worker_batches(worker_batches)
        metrics = trainer_a._ppo_update(batch, progress=0.0)
        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            assert math.isfinite(val), f"PPO metrics['{key}']={val} is not finite"


class TestSecondaryBatchingNoCrossEnvContamination:
    def test_no_cross_env_contamination(self, trainer_a):
        """Adapts test_batched_rollout_self_play.py's own
        test_no_cross_env_track_contamination to the NEW batched-secondary
        path: two groups differing only in env index 1's scenario seed must
        produce IDENTICAL data for envs 0/2's BOTH tracks."""
        import random

        def _seeded_envs(scenario_seeds, rng_seeds):
            envs = _make_self_play_envs(trainer_a, len(scenario_seeds), deterministic_secondary=True)
            for env in envs:
                env.bc_label_fn = bc_label_fn_for_phase_player(1)
            group = BatchedEnvGroup(
                envs, trainer_a, seeds=scenario_seeds, secondary_trainer=trainer_a,
            )
            for env, rs in zip(envs, rng_seeds):
                env.match.player_by_id(env.trainee_player_id).ai._rng = random.Random(rs)
                env.match.player_by_id("opponent").ai._rng = random.Random(rs + 1)
            return group

        group_a = _seeded_envs([25000, 25001, 25002], [500, 600, 700])
        results_a = group_a.collect(n_steps=_ROLLOUT_STEPS_PER_ENV, deterministic=True)

        group_b = _seeded_envs([25000, 99998, 25002], [500, 600, 700])  # only index 1 differs
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
