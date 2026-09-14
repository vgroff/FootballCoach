"""Coverage for episode-seed replay (ai/ppo/batched_rollout_worker.py's
"Episode-seed replay" docstring section, and
PPOTrainer._update_episode_replay/_trainee_episode_abs_adv_means in
ppo_trainer.py) -- the Prioritized-Level-Replay-style feature that re-queues
the seeds of the highest-mean-|advantage| episodes for the very next
rollout.

Two independent pieces are tested here:
1. BatchedEnvGroup.collect(replay_seeds=...) -- does a queued seed actually
   get consumed by the next episode reset, and is every episode's seed
   correctly recorded in stats["episode_seeds"] (1:1 with
   stats["episode_rewards"])?
2. _trainee_episode_abs_adv_means -- the pure per-episode advantage
   segmentation helper, including that it correctly IGNORES an interleaved
   secondary/opponent track (unlike the track-naive '[advantage |.| by
   episode]' diagnostic log line elsewhere in ppo_trainer.py).

Mirrors test_batched_rollout.py's fixture pattern (_make_envs: explicit
opponent_rules_prob/opponent_immobile_prob, not ai_config.json's live
ratios -- see that file's own docstring for why: this suite runs under
pytest-xdist and mutating the shared config file mid-run is unsafe).
"""
from __future__ import annotations

import functools

import pytest
import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.batched_rollout_worker import BatchedEnvGroup
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _trainee_episode_abs_adv_means
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

_PHASE = PHASES_BY_ID[1]
# Short on purpose: episodes must end (via timeout, at worst) within a
# handful of decisions so a modest collect(n_steps=...) budget reliably
# produces several completed episodes per env.
_SHORT_MAX_EPISODE_S = 2.0


@pytest.fixture(scope="module")
def trainer():
    t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=True)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    return t


def _make_short_envs(trainer, n: int) -> list[ScenarioEnv]:
    envs = []
    for _ in range(n):
        defn = ScenarioDefinition(
            key="episode_replay_test", label="Episode-replay test scenario",
            description="1v1, short episodes, rules-based opponent -- config-independent",
            build=functools.partial(
                build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0,
            ),
            on_tick=phase1_training_on_tick,
        )
        env = ScenarioEnv(
            definition=defn, trainee_player_id="trainee", phase=1,
            secondary_player_ids=["opponent"], max_episode_s=_SHORT_MAX_EPISODE_S,
        )
        env.sample_action_fn = trainer._sample_action
        envs.append(env)
    return envs


class TestReplaySeedConsumption:
    def test_replay_seed_is_consumed_by_next_reset(self, trainer):
        """Single env (no race between slots) -- the FIRST episode (from
        __init__'s own reset) is unseeded/unrecorded; the queued replay
        seed must become the SECOND episode's recorded seed."""
        env = _make_short_envs(trainer, 1)[0]
        group = BatchedEnvGroup([env], trainer)
        results = group.collect(n_steps=300, replay_seeds=[987654321])
        seeds = results[0]["stats"]["episode_seeds"]
        assert len(seeds) >= 2, "need at least 2 completed episodes for this test to be meaningful"
        assert seeds[0] is None, "first-ever episode (from __init__) should be unseeded"
        assert seeds[1] == 987654321, "the queued replay seed should land on the very next episode"
        # Once the queue is exhausted, later episodes fall back to fresh
        # random seeds -- still real ints, just not the replay value again.
        if len(seeds) > 2:
            assert all(isinstance(s, int) for s in seeds[2:])

    def test_episode_seeds_aligned_with_episode_rewards(self, trainer):
        env = _make_short_envs(trainer, 1)[0]
        group = BatchedEnvGroup([env], trainer)
        results = group.collect(n_steps=300)  # no replay_seeds -- default None
        stats = results[0]["stats"]
        assert len(stats["episode_seeds"]) == len(stats["episode_rewards"])
        assert stats["episode_seeds"][0] is None
        assert len(stats["episode_seeds"]) >= 2
        assert all(isinstance(s, int) for s in stats["episode_seeds"][1:])

    def test_no_replay_seeds_is_unchanged_behavior(self, trainer):
        """Regression guard: collect() with replay_seeds omitted/None must
        still work exactly as it did before this feature existed."""
        envs = _make_short_envs(trainer, 3)
        group = BatchedEnvGroup(envs, trainer)
        results = group.collect(n_steps=200)
        assert len(results) == 3
        for r in results:
            assert "episode_seeds" in r["stats"]
            assert r["buffer"] is not None


class TestTraineeEpisodeAbsAdvMeans:
    def test_simple_single_track(self):
        track_ids = ["trainee"] * 6
        dones = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        advantages = [1.0, -2.0, 3.0, 0.0, 0.0, 0.0]
        # episode 1: rows 0-2 -> mean(|1|,|-2|,|3|) = 2.0
        # episode 2: rows 3-5 -> mean(0,0,0) = 0.0
        means = _trainee_episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([2.0, 0.0])

    def test_ignores_interleaved_secondary_track(self):
        # trainee_t0, secondary_t0, trainee_t1(done), secondary_t1(done), trainee_t2, trainee_t3(done)
        track_ids = ["trainee", "opponent", "trainee", "opponent", "trainee", "trainee"]
        dones = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        advantages = [10.0, 999.0, 2.0, 999.0, 4.0, 6.0]
        # trainee-only rows (indices 0, 2, 4, 5): values [10.0, 2.0, 4.0, 6.0], dones [0, 1, 0, 1]
        # episode 1: rows (0, 2) -> mean(10, 2) = 6.0
        # episode 2: rows (4, 5) -> mean(4, 6) = 5.0
        means = _trainee_episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([6.0, 5.0])

    def test_trailing_incomplete_episode_dropped(self):
        track_ids = ["trainee"] * 4
        dones = [1.0, 0.0, 0.0, 0.0]  # only the first row completes an episode
        advantages = [5.0, 100.0, 100.0, 100.0]
        means = _trainee_episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([5.0])

    def test_no_complete_episodes_returns_empty(self):
        means = _trainee_episode_abs_adv_means(["trainee", "trainee"], [0.0, 0.0], [1.0, 2.0])
        assert means == []
