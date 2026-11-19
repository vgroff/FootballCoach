"""eval.seed_network_sampling (ai/eval/seeded_eval.py): with it on, an eval's
network sampling is seeded per episode (seed, side, repeat), so the same
checkpoint gives the same eval every time, the batched path gives the same
episodes as the sequential one, repeats of a seed still differ, and the
caller's own torch RNG stream is left untouched."""
from __future__ import annotations

import random

import pytest
import torch

from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.eval.seeded_eval import (
    eval_episode_sampling_seed, run_seeded_evaluation, run_seeded_evaluation_batched,
)
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

PHASE2 = PHASES_BY_ID[2]
CTRL = {"attacker": "neural", "defender": "rules", "goalkeeper": "rules"}
SEEDS = [2200001, 2200002, 2200003]


@pytest.fixture(scope="module")
def trainer():
    torch.manual_seed(0)
    return PPOTrainer.from_config(device=torch.device("cpu"))


def _factory(seed: int, _swap: bool):
    env = build_role_env(PHASE2, controller_weights={r: {c: 1} for r, c in CTRL.items()})
    env.controller_overrides = CTRL
    env.scenario_kwargs = {**env.scenario_kwargs, "no_keeper_prob": 0.0, "seed": seed}
    return env


def _episodes(result):
    return sorted(zip(result.outcomes_list, [round(r, 9) for r in result.rewards],
                      [round(t, 9) for t in result.episode_lengths_s]))


def test_the_same_checkpoint_gives_the_same_eval_twice(trainer):
    a = run_seeded_evaluation(_factory, trainer._sample_action, SEEDS, repeats_per_seed=2,
                              win_outcome="goal", seed_network_sampling=True)
    torch.manual_seed(12345); random.seed(12345)              # unrelated global state must not matter
    b = run_seeded_evaluation(_factory, trainer._sample_action, SEEDS, repeats_per_seed=2,
                              win_outcome="goal", seed_network_sampling=True)
    assert a.outcomes_list == b.outcomes_list
    assert a.rewards == b.rewards
    assert a.episode_lengths_s == b.episode_lengths_s
    assert a.value_preds == b.value_preds


def test_batched_matches_sequential(trainer):
    seq = run_seeded_evaluation(_factory, trainer._sample_action, SEEDS, repeats_per_seed=2,
                                win_outcome="goal", seed_network_sampling=True)
    bat = run_seeded_evaluation_batched(_factory, trainer, SEEDS, repeats_per_seed=2, win_outcome="goal",
                                        envs_per_process=4, seed_network_sampling=True)
    assert _episodes(seq) == _episodes(bat)


def test_repeats_of_a_seed_get_their_own_sampling_seed():
    assert len({eval_episode_sampling_seed(5, s, r) for s in (False, True) for r in range(4)}) == 8


def test_the_callers_torch_rng_is_untouched(trainer):
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    run_seeded_evaluation(_factory, trainer._sample_action, SEEDS[:1], repeats_per_seed=1,
                          win_outcome="goal", seed_network_sampling=True)
    assert torch.equal(torch.rand(3), expected)


def test_batched_with_a_neural_secondary_reproduces(trainer):
    """The snapshot-eval shape: a second network batched for the secondary
    player (here the defender); its draws use the same per-slot state."""
    ctrl = {"attacker": "neural", "defender": "neural", "goalkeeper": "rules"}

    def factory(seed, _swap):
        env = build_role_env(PHASE2, controller_weights={r: {c: 1} for r, c in ctrl.items()})
        env.controller_overrides = ctrl
        env.scenario_kwargs = {**env.scenario_kwargs, "no_keeper_prob": 0.0, "seed": seed}
        return env

    runs = [run_seeded_evaluation_batched(factory, trainer, SEEDS, repeats_per_seed=2, win_outcome="goal",
                                          envs_per_process=4, secondary_trainer=trainer,
                                          seed_network_sampling=True) for _ in range(2)]
    assert _episodes(runs[0]) == _episodes(runs[1])
