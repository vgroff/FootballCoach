"""Coverage for ScenarioEnv.max_episode_s_variability_s (added 2026-10-14) --
per-episode randomized episode length, so a losing player can't reliably
time a "retreat and force a timeout" strategy against a fixed, known
episode length. See that constructor arg's own docstring in scenario_env.py.

variability_s=0.0 (the live config value as of this writing) must be fully
inert: every episode gets exactly the configured floor, bit-identical to
before this knob existed, and -- importantly -- costs no rng draw at all
(see test_zero_variability_consumes_no_rng_draw), so existing seeded
reproducibility is untouched for every caller that doesn't opt in.
"""
from __future__ import annotations

import random

import pytest

from footballcoach.ai.curriculum.envs import build_env
from footballcoach.ai.curriculum.phases import CurriculumPhase


def _env(*, max_episode_s=10.0, variability_s=0.0, rng=None):
    phase = CurriculumPhase(
        name="p1", phase_id=1, scenario_key="phase1_1v1",
        env_kwargs={"max_episode_s": max_episode_s, "max_episode_s_variability_s": variability_s},
    )
    env = build_env(phase)
    if rng is not None:
        env.rng = rng
    return env


class TestZeroVariabilityIsInert:

    def test_effective_duration_equals_base_every_reset(self):
        env = _env(max_episode_s=12.5, variability_s=0.0)
        for _ in range(5):
            env.reset()
            assert env.max_episode_s == pytest.approx(12.5)

    def test_max_episode_ticks_matches_base_duration(self):
        env = _env(max_episode_s=12.5, variability_s=0.0)
        env.reset()
        assert env._max_episode_ticks == max(1, int(12.5 / env._dt_s))

    def test_zero_variability_never_calls_rng_uniform(self):
        """Confirms the implementation guards the rng.uniform() call behind
        `variability_s > 0.0` rather than always drawing -- env.rng is also
        used for other, unrelated purposes during reset() (obs slot
        shuffling, etc.), so this specifically spies on .uniform() rather
        than asserting "no rng state change at all" (which would be false
        regardless of this feature)."""
        class _SpyRandom(random.Random):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.uniform_calls = 0

            def uniform(self, a, b):
                self.uniform_calls += 1
                return super().uniform(a, b)

        rng = _SpyRandom(42)
        env = _env(max_episode_s=12.5, variability_s=0.0, rng=rng)
        for _ in range(5):
            env.reset()
        assert rng.uniform_calls == 0


class TestNonzeroVariability:

    def test_effective_duration_never_below_base(self):
        rng = random.Random(7)
        env = _env(max_episode_s=10.0, variability_s=5.0, rng=rng)
        for _ in range(200):
            env.reset()
            assert env.max_episode_s >= 10.0

    def test_effective_duration_never_exceeds_base_plus_variability(self):
        rng = random.Random(7)
        env = _env(max_episode_s=10.0, variability_s=5.0, rng=rng)
        for _ in range(200):
            env.reset()
            assert env.max_episode_s <= 15.0 + 1e-9

    def test_duration_actually_varies_across_resets(self):
        """Not just 'bounded' -- genuinely different episode to episode, not
        silently collapsing to a constant somewhere in the range."""
        rng = random.Random(7)
        env = _env(max_episode_s=10.0, variability_s=5.0, rng=rng)
        seen = set()
        for _ in range(50):
            env.reset()
            seen.add(round(env.max_episode_s, 6))
        assert len(seen) > 10

    def test_max_episode_ticks_tracks_the_resampled_duration_each_episode(self):
        rng = random.Random(7)
        env = _env(max_episode_s=10.0, variability_s=5.0, rng=rng)
        for _ in range(10):
            env.reset()
            assert env._max_episode_ticks == max(1, int(env.max_episode_s / env._dt_s))

    def test_base_floor_is_never_mutated(self):
        """self._base_max_episode_s must stay fixed across resets -- only
        self.max_episode_s (the effective, per-episode value) changes."""
        rng = random.Random(7)
        env = _env(max_episode_s=10.0, variability_s=5.0, rng=rng)
        for _ in range(20):
            env.reset()
            assert env._base_max_episode_s == pytest.approx(10.0)

    def test_positive_control_rng_uniform_is_actually_called(self):
        """Proves the spy mechanism used by
        test_zero_variability_never_calls_rng_uniform would actually catch a
        real call -- i.e. that test isn't vacuously passing."""
        class _SpyRandom(random.Random):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.uniform_calls = 0

            def uniform(self, a, b):
                self.uniform_calls += 1
                return super().uniform(a, b)

        rng = _SpyRandom(7)
        env = _env(max_episode_s=10.0, variability_s=5.0, rng=rng)
        env.reset()
        assert rng.uniform_calls >= 1
