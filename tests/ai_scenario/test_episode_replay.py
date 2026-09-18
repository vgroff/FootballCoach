"""Coverage for episode-seed replay (ai/ppo/batched_rollout_worker.py's
"Episode-seed replay" docstring section, and
PPOTrainer._update_episode_replay/_episode_abs_adv_means in
ppo_trainer.py) -- the Prioritized-Level-Replay-style feature that re-queues
the seeds of the highest-mean-|advantage| episodes for the very next
rollout.

Two independent pieces are tested here:
1. BatchedEnvGroup.collect(replay_seeds=...) -- does a queued seed actually
   get consumed by the next episode reset, and is every episode's seed
   correctly recorded in stats["episode_seeds"] (1:1 with
   stats["episode_rewards"])?
2. _episode_abs_adv_means -- the pure per-episode advantage segmentation
   helper. Segments strictly on the TRAINEE's own done events (so it lines
   up 1:1 with stats["episode_seeds"]/stats["episode_rewards"]), but each
   episode's mean now INCLUDES any interleaved secondary/opponent-track
   rows for that same episode -- see the function's own docstring for why
   that's safe today (a non-"trainee" row only ever exists when the
   secondary player is the current live network playing itself; rules/
   immobile opponents never produce a buffer row at all).

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
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _episode_abs_adv_means
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


class TestEpisodeAbsAdvMeans:
    def test_simple_single_track(self):
        track_ids = ["trainee"] * 6
        dones = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        advantages = [1.0, -2.0, 3.0, 0.0, 0.0, 0.0]
        # episode 1: rows 0-2 -> mean(|1|,|-2|,|3|) = 2.0
        # episode 2: rows 3-5 -> mean(0,0,0) = 0.0
        means = _episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([2.0, 0.0])

    def test_includes_secondary_row_at_the_terminal_tick(self):
        # Real buffer.add() ordering: the trainee's row for a tick is always
        # added BEFORE that same tick's secondary row(s) -- so the terminal
        # tick of episode 1 looks like [trainee(done=1), opponent(done=1)],
        # with the opponent row arriving AFTER the trainee done=1 row but
        # still belonging to episode 1, not episode 2.
        track_ids = ["trainee", "opponent", "trainee", "opponent", "trainee", "trainee"]
        dones = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        advantages = [10.0, 20.0, 2.0, 40.0, 4.0, 6.0]
        # episode 1: rows (0, 1, 2, 3) -> mean(10, 20, 2, 40) = 18.0
        # episode 2: rows (4, 5) -> mean(4, 6) = 5.0 (no secondary this time)
        means = _episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([18.0, 5.0])

    def test_includes_secondary_rows_spread_through_the_whole_episode(self):
        # Secondary rows interleaved at every tick of a 3-tick episode, not
        # just the terminal one -- all of them should count.
        track_ids = ["trainee", "opponent", "trainee", "opponent", "trainee", "opponent"]
        dones = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        advantages = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        means = _episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([(1 + 2 + 3 + 4 + 5 + 6) / 6])

    def test_mixed_rollout_some_episodes_with_secondary_some_without(self):
        # episode 1: vs a rules/immobile opponent -- no secondary rows at
        # all (they never reach the buffer), trainee-only mean.
        # episode 2: self-play -- secondary rows present, included.
        track_ids = ["trainee", "trainee", "trainee", "opponent", "trainee"]
        dones = [0.0, 1.0, 0.0, 1.0, 1.0]
        advantages = [1.0, 3.0, 10.0, 20.0, 30.0]
        # episode 1: rows (0, 1) -> mean(1, 3) = 2.0
        # episode 2: rows (2, 3, 4) -> mean(10, 20, 30) = 20.0
        means = _episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([2.0, 20.0])

    def test_trailing_incomplete_episode_dropped(self):
        track_ids = ["trainee"] * 4
        dones = [1.0, 0.0, 0.0, 0.0]  # only the first row completes an episode
        advantages = [5.0, 100.0, 100.0, 100.0]
        means = _episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([5.0])

    def test_trailing_incomplete_episode_with_secondary_rows_dropped(self):
        # Same as above, but the dangling trailing episode also has
        # secondary rows mixed in -- still fully dropped, not partially
        # counted.
        track_ids = ["trainee", "trainee", "opponent", "trainee", "opponent"]
        dones = [1.0, 0.0, 0.0, 0.0, 0.0]
        advantages = [5.0, 100.0, 100.0, 100.0, 100.0]
        means = _episode_abs_adv_means(track_ids, dones, advantages)
        assert means == pytest.approx([5.0])

    def test_no_complete_episodes_returns_empty(self):
        means = _episode_abs_adv_means(["trainee", "trainee"], [0.0, 0.0], [1.0, 2.0])
        assert means == []
