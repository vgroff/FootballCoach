"""What a seed does and doesn't pin in seeded evaluation.

Pinned: the scenario setup and, with a deterministic policy, the whole episode.
Not pinned: a stochastic policy's own samples, which come from torch's global
RNG. The same seed list therefore gives different episodes on every stochastic
eval unless the caller reseeds torch first.
"""
from __future__ import annotations

import functools

import pytest
import torch

from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _build_eval_env_factory

_SEEDS = list(range(5_400_000, 5_400_006))
_OTHER_SEEDS = list(range(5_500_000, 5_500_006))


@pytest.fixture(scope="module")
def trainer():
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


def _deterministic(trainer):
    return functools.partial(
        trainer._sample_action, deterministic=True, deterministic_decision=True, deterministic_direction=True,
    )


def _run(sample_action_fn, seeds=_SEEDS, swap_sides=False):
    env_factory = _build_eval_env_factory(use_rules_ai=True, max_episode_s=12.0)
    return run_seeded_evaluation(env_factory, sample_action_fn, seeds, repeats_per_seed=1, swap_sides=swap_sides)


def _episodes(result):
    return result.rewards, result.outcomes_list, result.episode_lengths_s


def _setup(env):
    players = [(p.player_id, p.position, p.velocity, p.heading_rad, p.attributes) for p in env.match.players]
    return players, env.match.ball.position, env.match.ball.velocity


def test_deterministic_policy_reproduces_every_episode_exactly(trainer):
    assert _episodes(_run(_deterministic(trainer))) == _episodes(_run(_deterministic(trainer)))


def test_deterministic_policy_with_swapped_sides_reproduces_exactly(trainer):
    first = _run(_deterministic(trainer), swap_sides=True)
    assert first.n_episodes == 2 * len(_SEEDS)
    assert _episodes(first) == _episodes(_run(_deterministic(trainer), swap_sides=True))


def test_different_seeds_give_different_episodes(trainer):
    assert _run(_deterministic(trainer)).rewards != _run(_deterministic(trainer), seeds=_OTHER_SEEDS).rewards


def test_stochastic_policy_reproduces_only_after_reseeding_torch(trainer):
    torch.manual_seed(1)
    first = _run(trainer._sample_action)
    torch.manual_seed(1)
    assert _episodes(first) == _episodes(_run(trainer._sample_action))


def test_stochastic_policy_varies_without_reseeding_torch(trainer):
    torch.manual_seed(1)
    assert _run(trainer._sample_action).rewards != _run(trainer._sample_action).rewards


def test_scenario_setup_is_pinned_by_the_seed(trainer):
    factory = _build_eval_env_factory(use_rules_ai=True, max_episode_s=12.0)
    for seed in _SEEDS:
        first, second = factory(seed, False), factory(seed, False)
        first.sample_action_fn = second.sample_action_fn = _deterministic(trainer)
        first.reset()
        second.reset()
        assert _setup(first) == _setup(second)
