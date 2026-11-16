"""Mixed-phase rollouts: phase-1 and phase-2 envs in one BatchedEnvGroup, built
the way a rollout worker builds them (build_worker_envs). Checks rows come
from both phases, phase-2 outcomes are tallied per episode tag (phase 1 keeps
its vs_* lists), seeds are phase-tagged, replayed seeds only go to an env of
their own phase, and the non-batched training paths refuse a mix."""
from __future__ import annotations

import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.mix import PHASE_SEED_STRIDE, tag_seed, untag_seed
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.role_env import RoleScenarioEnv
from footballcoach.ai.ppo.batched_rollout_worker import BatchedEnvGroup, build_worker_envs
from footballcoach.ai.ppo.rollout_logging import format_tag_outcomes

TAG = "phase2_1v2[attacker=neural,defender=neural,goalkeeper=rules]"


@pytest.fixture(scope="module")
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True)
    t.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    return t


@pytest.fixture
def neural_phase2(monkeypatch):
    monkeypatch.setitem(load_ai_config()["curriculum"]["phase2"], "controllers", {
        "attacker": {"neural": 1}, "defender": {"neural": 1}, "goalkeeper": {"rules": 1},
    })


def _group(trainer, phases, seeds=None):
    envs = build_worker_envs(phases, trainer, base_seed=7)
    return envs, BatchedEnvGroup(envs, trainer, seeds=seeds, secondary_trainer=trainer)


def test_worker_envs_follow_their_phase(trainer, neural_phase2):
    envs = build_worker_envs([1, 2, 2], trainer, base_seed=1)
    assert [e.phase for e in envs] == [1, 2, 2]
    assert not isinstance(envs[0], RoleScenarioEnv)
    assert all(isinstance(e, RoleScenarioEnv) and e.require_neural_player for e in envs[1:])


def test_role_env_without_a_neural_role_cannot_be_built_for_training(trainer, monkeypatch):
    monkeypatch.setitem(load_ai_config()["curriculum"]["phase2"], "controllers",
                        {r: {"rules": 1, "neural": 0} for r in ("attacker", "defender", "goalkeeper")})
    envs = build_worker_envs([2], trainer, base_seed=1)
    with pytest.raises(ValueError, match="no role can be neural"):
        envs[0].reset(seed=0)


def test_mixed_collection_tallies_each_phase_separately(trainer, neural_phase2):
    envs, group = _group(trainer, [1, 2, 2])
    results = group.collect(900)
    phase1, *phase2 = results
    assert len(phase1["buffer"]) > 0 and all(len(r["buffer"]) > 0 for r in phase2)
    for r in phase2:
        st = r["stats"]
        assert set(st["episode_outcomes_by_tag"]) == {TAG}, st["episode_outcomes_by_tag"]
        assert len(st["episode_outcomes_by_tag"][TAG]) == len(st["episode_rewards"])
        assert st["episode_rewards_by_tag"][TAG] == st["episode_rewards"]
        assert not (st["episode_outcomes_vs_rules"] or st["episode_outcomes_vs_neural"] or st["episode_outcomes_vs_immobile"])
        assert all(s is None or untag_seed(s)[1] == 2 for s in st["episode_seeds"])
        # Two neural players per phase-2 episode: the trainee's rows (track
        # "trainee", whichever player it is, as in phase 1) plus the secondary's.
        assert set(r["buffer"].track_ids) == {"trainee", "defender"}
    st1 = phase1["stats"]
    assert st1["episode_outcomes_by_tag"] == {}
    assert all(s is None or s < PHASE_SEED_STRIDE for s in st1["episode_seeds"])


def test_replayed_seeds_only_reach_envs_of_their_phase(trainer, neural_phase2):
    envs, group = _group(trainer, [1, 2])
    reset_seeds = {0: [], 1: []}
    for i, env in enumerate(envs):
        orig = env.reset

        def _spy(*, seed=None, _i=i, _orig=orig):
            reset_seeds[_i].append(seed)
            return _orig(seed=seed)
        env.reset = _spy
    queued = [tag_seed(4242, 2), tag_seed(4343, 2), 777]
    results = group.collect(600, replay_seeds=list(queued))
    p2_seeds = results[1]["stats"]["episode_seeds"]
    # The phase-2 env consumes its own queued seeds first, in order; the
    # phase-1 env never sees them, and gets the phase-1 seed if it resets.
    assert p2_seeds[1:3] == queued[:2]
    assert reset_seeds[1][:2] == [4242, 4343]
    assert 4242 not in reset_seeds[0] and 4343 not in reset_seeds[0]
    if reset_seeds[0]:
        assert reset_seeds[0][0] == 777


def test_non_batched_paths_refuse_a_mix():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    t.phase_mix = {1: 3.0, 2: 1.0}
    t.batched_rollout = False
    with pytest.raises(NotImplementedError, match="batched rollout"):
        t.train(None, total_steps=10, phase_id=1)


def test_format_tag_outcomes_uses_the_phase_vocabulary_order():
    line = format_tag_outcomes(TAG, ["saved", "goal", "goal", "timeout"], [4.0, 4.0, 1.0, -1.0])
    assert line == (f"{TAG}(4): goal=50.0% saved=25.0% defence_possession=0.0% "
                    "shot_off_target=0.0% attacker_out=0.0% timeout=25.0%  trainee_rew=+2.000")
