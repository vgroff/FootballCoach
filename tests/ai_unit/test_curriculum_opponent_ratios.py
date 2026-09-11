"""Coverage for curriculum/envs.py's opponent_type_probs() -- the arithmetic
that converts ai_config.json's phase1_opponent_{rules,immobile,neural}_ratio
into the probabilities build_1v1_scenario actually rolls against. This is
the single piece of code that determines what fraction of REAL phase-1
training is self-play (currently configured 1:0:3, meant to be 75% neural),
and until now nothing had ever exercised it: every self-play test in this
repo (test_secondary_opponent_gae.py, test_batched_rollout_self_play.py,
test_self_play_reward_attribution.py, ...) deliberately bypasses
_build_phase1_env()/load_ai_config() entirely and hardcodes
opponent_rules_prob=0.0/opponent_immobile_prob=0.0 directly instead, for a
documented and legitimate reason (pytest-xdist workers share the config
file, so mutating it mid-run is unsafe). That left this exact conversion
with zero coverage.

This isn't a hypothetical risk: the same function's docstring/the code
right below it documents a PRIOR silent bug of exactly this shape --
ball_max_speed_mps was hardcoded and had "zero effect on real training
regardless of what it was set to" until 2026-09-03, found only by manually
debugging suspiciously clean scores, not by a test.

Two layers of coverage:
- TestOpponentTypeProbs: pure arithmetic, no engine, every edge case.
- TestRealConfigRatioProducesExpectedMix: a statistical test that feeds the
  ACTUAL ai_config.json ratios through the real opponent_type_probs() AND
  through build_1v1_scenario()'s own roll (match._opponent_use_rules_ai /
  match._opponent_is_immobile), across many seeds, and checks the resulting
  empirical mix is really ~25/0/75 -- closing the loop from "the arithmetic
  is right" to "the roll consumer actually interprets it the way we assume"
  in one test, without needing to touch the shared config file at all
  (build_1v1_scenario is called directly with explicit probs, seeded).
"""
from __future__ import annotations

import pytest

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import opponent_type_probs
from footballcoach.ui.scenarios import build_1v1_scenario

_N_SAMPLES = 4000
_TOLERANCE = 0.03  # +/- 3 percentage points on a 4000-sample binomial draw


class TestOpponentTypeProbs:
    def test_real_production_ratio_is_25_0_75(self):
        """The actual configured ai_config.json ratio (1 : 0 : 3) -- locks
        in that this really does mean 25% rules / 0% immobile / 75% neural,
        the number the whole self-play curriculum design rests on."""
        cfg = load_ai_config()["curriculum"]
        rules_prob, immobile_prob = opponent_type_probs(
            float(cfg["phase1_opponent_rules_ratio"]),
            float(cfg["phase1_opponent_immobile_ratio"]),
            float(cfg["phase1_opponent_neural_ratio"]),
        )
        assert rules_prob == pytest.approx(0.25)
        assert immobile_prob == pytest.approx(0.0)
        neural_prob = 1.0 - rules_prob - immobile_prob
        assert neural_prob == pytest.approx(0.75)

    def test_all_rules(self):
        assert opponent_type_probs(1.0, 0.0, 0.0) == (1.0, 0.0)

    def test_all_immobile(self):
        assert opponent_type_probs(0.0, 1.0, 0.0) == (0.0, 1.0)

    def test_all_neural(self):
        rules_prob, immobile_prob = opponent_type_probs(0.0, 0.0, 1.0)
        assert rules_prob == 0.0
        assert immobile_prob == 0.0  # neural = 1 - 0 - 0 = 1.0

    def test_equal_thirds(self):
        rules_prob, immobile_prob = opponent_type_probs(1.0, 1.0, 1.0)
        assert rules_prob == pytest.approx(1.0 / 3.0)
        assert immobile_prob == pytest.approx(1.0 / 3.0)
        neural_prob = 1.0 - rules_prob - immobile_prob
        assert neural_prob == pytest.approx(1.0 / 3.0)

    def test_all_zero_falls_back_to_always_immobile(self):
        """Matches phase1_opponent_immobile_ratio's own config default of
        1.0 -- if every ratio were somehow 0 (misconfiguration), this must
        not divide by zero or silently produce all-neural/all-rules."""
        assert opponent_type_probs(0.0, 0.0, 0.0) == (0.0, 1.0)

    def test_probabilities_never_exceed_one(self):
        """rules_prob + immobile_prob must stay <= 1.0 for any nonnegative
        input -- build_1v1_scenario's roll (`elif _r < rules_prob +
        immobile_prob`) silently breaks (immobile absorbs part of what
        should be neural's share, and neural could even become
        unreachable) if this ever drifts above 1.0."""
        for rules, immobile, neural in [
            (1.0, 0.0, 3.0), (5.0, 3.0, 0.0), (0.1, 0.1, 0.1), (2.0, 2.0, 2.0),
        ]:
            rules_prob, immobile_prob = opponent_type_probs(rules, immobile, neural)
            assert rules_prob + immobile_prob <= 1.0 + 1e-9

    def test_zero_neural_ratio_never_produces_self_play(self):
        rules_prob, immobile_prob = opponent_type_probs(1.0, 1.0, 0.0)
        assert (1.0 - rules_prob - immobile_prob) == pytest.approx(0.0)


class TestRealConfigRatioProducesExpectedMix:
    def test_build_1v1_scenario_roll_matches_25_0_75_empirically(self):
        cfg = load_ai_config()["curriculum"]
        rules_prob, immobile_prob = opponent_type_probs(
            float(cfg["phase1_opponent_rules_ratio"]),
            float(cfg["phase1_opponent_immobile_ratio"]),
            float(cfg["phase1_opponent_neural_ratio"]),
        )

        n_rules = n_immobile = n_neural = 0
        for seed in range(_N_SAMPLES):
            match = build_1v1_scenario(
                opponent_rules_prob=rules_prob, opponent_immobile_prob=immobile_prob,
                seed=100_000 + seed,
            )
            if match._opponent_use_rules_ai:
                n_rules += 1
            elif match._opponent_is_immobile:
                n_immobile += 1
            else:
                n_neural += 1

        assert n_rules + n_immobile + n_neural == _N_SAMPLES
        rules_frac = n_rules / _N_SAMPLES
        immobile_frac = n_immobile / _N_SAMPLES
        neural_frac = n_neural / _N_SAMPLES

        assert rules_frac == pytest.approx(0.25, abs=_TOLERANCE), (
            f"rules fraction {rules_frac:.3f} over {_N_SAMPLES} seeds is not "
            f"within {_TOLERANCE} of the expected 0.25 -- either the ratio "
            "arithmetic or build_1v1_scenario's roll has drifted from the "
            "configured 1:0:3 ai_config.json ratio"
        )
        assert immobile_frac == pytest.approx(0.0, abs=_TOLERANCE)
        assert neural_frac == pytest.approx(0.75, abs=_TOLERANCE), (
            f"neural (self-play) fraction {neural_frac:.3f} over "
            f"{_N_SAMPLES} seeds is not within {_TOLERANCE} of the expected "
            "0.75 -- this is the number the entire self-play curriculum "
            "design rests on"
        )
