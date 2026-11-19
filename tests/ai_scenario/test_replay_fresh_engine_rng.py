"""Episode replay with fresh engine dice (ppo.episode_replay_fresh_engine_rng):
reset(seed, fresh_engine_rng=True) rebuilds the seed's situation but reseeds
the match's own rng; without it a seed fixes the engine's draws."""
from __future__ import annotations

import random

from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID

PHASE2 = PHASES_BY_ID[2]
RULES = {r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")}


def _start(fresh: bool, seed=1000020):
    env = build_role_env(PHASE2, controller_weights=RULES)
    env.reset(seed=seed, fresh_engine_rng=fresh)
    m = env.match
    state = {p.player_id: (p.position, p.velocity, p.stamina, p.attributes) for p in m.players}
    return env, state, m.rng.random()


def test_same_seed_same_situation_and_same_dice_by_default():
    _, s1, r1 = _start(False)
    _, s2, r2 = _start(False)
    assert s1 == s2 and r1 == r2


def test_fresh_engine_rng_keeps_the_situation_but_changes_the_dice():
    _, s0, r0 = _start(False)
    rolls = set()
    for _ in range(5):
        _, s, r = _start(True)
        assert s == s0                    # same positions / velocities / stamina / attributes
        rolls.add(r)
    assert r0 not in rolls and len(rolls) == 5


def test_fresh_engine_rng_episodes_still_run():
    env = build_role_env(PHASE2, controller_weights=RULES)
    for seed in range(5):
        env.rng = random.Random(seed)
        env.reset(seed=seed, fresh_engine_rng=True)
        done = False
        while not done:
            _, _, done, info = env.step()
        assert info.trial_outcome in env.outcome_keys
