"""Episode-seed replay per phase (curriculum/mix.py): the top fraction is taken
within each phase, and seeds are only routed to workers running their phase.
With one phase both reduce exactly to the previous behaviour."""
from __future__ import annotations

import math
import random

import pytest

from footballcoach.ai.curriculum.mix import route_seeds_to_workers, tag_seed, top_fraction_per_phase


def _old_top(items, fraction):
    n_top = max(1, math.ceil(fraction * len(items)))
    return sorted(items, key=lambda t: t[2], reverse=True)[:n_top]


@pytest.mark.parametrize("n,fraction", [(1, 0.05), (7, 0.05), (40, 0.05), (40, 0.25), (100, 0.1)])
def test_one_phase_is_exactly_the_old_top_fraction(n, fraction):
    rng = random.Random(n)
    items = [(rng.randrange(2 ** 31), rng.random(), rng.choice([0.1, 0.2, rng.random()])) for _ in range(n)]
    got = top_fraction_per_phase(items, fraction, seed_of=lambda t: t[0], score_of=lambda t: t[2])
    assert got == _old_top(items, fraction)


def test_each_phase_gets_its_own_top_fraction():
    # Phase 2's scores are 100x phase 1's: ranked together it would take every slot.
    p1 = [(tag_seed(i, 1), 0.0, 0.01 * i) for i in range(20)]
    p2 = [(tag_seed(i, 2), 0.0, 1.0 + i) for i in range(20)]
    got = top_fraction_per_phase(p1 + p2, 0.1, seed_of=lambda t: t[0], score_of=lambda t: t[2])
    assert [t for t in got if t in p1] == sorted(p1, key=lambda t: t[2], reverse=True)[:2]
    assert [t for t in got if t in p2] == sorted(p2, key=lambda t: t[2], reverse=True)[:2]
    assert len(got) == 4


def test_a_small_phase_still_gets_one():
    p1 = [(tag_seed(i, 1), 0.0, float(i)) for i in range(100)]
    p2 = [(tag_seed(7, 2), 0.0, 0.0)]
    got = top_fraction_per_phase(p1 + p2, 0.05, seed_of=lambda t: t[0], score_of=lambda t: t[2])
    assert p2[0] in got and sum(t in p1 for t in got) == 5


@pytest.mark.parametrize("n_seeds,n_workers", [(0, 3), (2, 4), (9, 4), (13, 5)])
def test_one_phase_routing_is_the_old_round_robin(n_seeds, n_workers):
    seeds = list(range(100, 100 + n_seeds))
    got = route_seeds_to_workers(seeds, [{1}] * n_workers)
    assert got == [seeds[i::n_workers] for i in range(n_workers)]


def test_seeds_only_go_to_workers_running_their_phase():
    seeds = [tag_seed(i, 1) for i in range(4)] + [tag_seed(i, 2) for i in range(3)] + [tag_seed(0, 3)]
    workers = [{1}, {1, 2}, {2}, {1}]
    got = route_seeds_to_workers(seeds, workers)
    for w, chunk in enumerate(got):
        assert all((s // 2 ** 31) + 1 in workers[w] for s in chunk)
    routed = sorted(s for c in got for s in c)
    assert routed == sorted(seeds[:-1])          # the phase-3 seed has no worker: dropped
    # round-robin within a phase across its eligible workers
    assert [sum(tag_seed(i, 2) in got[w] for i in range(3)) for w in (1, 2)] == [2, 1]


def test_ppo_selection_uses_the_per_phase_top(monkeypatch):
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    import torch
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    t._episode_replay_top_fraction = 0.1
    t._pending_replay_before = {}
    p1 = [(tag_seed(i, 1), 0.0, 0.01 * i) for i in range(20)]
    p2 = [(tag_seed(i, 2), 0.0, 1.0 + i) for i in range(20)]
    seeds = t._update_episode_replay(p1 + p2)
    assert sum(s in {x[0] for x in p1} for s in seeds) == 2
    assert sum(s in {x[0] for x in p2} for s in seeds) == 2
