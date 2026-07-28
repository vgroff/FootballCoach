"""Balance tests for the tackle angle modifier.

The modifier scales the tackler's effective boost based on which direction
they approach from, relative to the dribbler's heading:

  Frontal (tackler in front of dribbler, angle=0°):  +10% to boost
  Side-on (angle=90°):                               -5%  to boost
  From behind (angle=180°):                          -65% to boost

At rng_reduction=1.0 (deterministic), the roll equals the attribute * effective
boost, so we can verify the angle modifier with closed-form assertions. At
rng_reduction=0.3 (balance setting) we verify win rates follow the expected
ordering: frontal > side > from-behind.
"""
from __future__ import annotations

import math
import random

from footballcoach.engine.tackling import TacklingParams, attempt_tackle, tackle_angle_modifier
from footballcoach.mathutils import Vector3

RNG_REDUCTION = 0.3
N_TRIALS = 2000


# ---------------------------------------------------------------------------
# Deterministic angle modifier value tests (rng_reduction=1.0)
# ---------------------------------------------------------------------------

def test_frontal_modifier_is_positive():
    params = TacklingParams.from_config()
    # Dribbler faces +x (heading=0), tackler is at +x (directly in front).
    mod = tackle_angle_modifier(0.0, Vector3(0, 0, 0), Vector3(1, 0, 0), params)
    assert mod > 0.0, f"Frontal modifier should be positive, got {mod}"
    assert abs(mod - params.angle_modifier_frontal) < 1e-9


def test_behind_modifier_is_most_negative():
    params = TacklingParams.from_config()
    # Dribbler faces +x, tackler is at -x (directly behind).
    mod = tackle_angle_modifier(0.0, Vector3(0, 0, 0), Vector3(-1, 0, 0), params)
    assert mod <= -0.5, f"Behind modifier should be heavily negative, got {mod}"
    assert abs(mod - params.angle_modifier_behind) < 1e-9


def test_side_modifier_is_small_negative():
    params = TacklingParams.from_config()
    # Dribbler faces +x, tackler is at +y (side-on, 90°).
    mod = tackle_angle_modifier(0.0, Vector3(0, 0, 0), Vector3(0, 1, 0), params)
    assert abs(mod - params.angle_modifier_side) < 1e-9


def test_modifier_ordering():
    """frontal > side > behind."""
    params = TacklingParams.from_config()
    frontal = tackle_angle_modifier(0.0, Vector3(0, 0, 0), Vector3(1, 0, 0), params)
    side = tackle_angle_modifier(0.0, Vector3(0, 0, 0), Vector3(0, 1, 0), params)
    behind = tackle_angle_modifier(0.0, Vector3(0, 0, 0), Vector3(-1, 0, 0), params)
    assert frontal > side > behind


def test_frontal_tackle_deterministic_easier(balance_recorder):
    """At rng_reduction=1.0, a frontal tackle with tackling=0.45 vs dribbling=0.5
    should succeed (1.2 * 1.1 * 0.45 = 0.594 > 0.5), while without the angle
    modifier it would fail (1.2 * 0.45 = 0.54 > 0.5, actually still succeeds at
    these values). Use a tighter matchup: tackling=0.43.
      Without modifier: 1.2 * 0.43 = 0.516 > 0.5 -> wins.
    Let's use behind angle instead to verify that a behind tackle fails.
    """
    params = TacklingParams.from_config()
    # Behind tackle: tackling=0.5, dribbling=0.5
    # effective_boost = 1.2 * (1 + (-0.65)) = 1.2 * 0.35 = 0.42
    # tackler_roll = 0.42 * 0.5 = 0.21 < 0.5 -> dribbler wins
    result = attempt_tackle(0.5, 0.5, rng_reduction=1.0, rng=random.Random(0), params=params,
                            angle_modifier=params.angle_modifier_behind)
    assert not result.tackler_won, "Behind tackle with equal attrs should fail deterministically"
    # Same attrs but frontal: 1.2 * 1.1 * 0.5 = 0.66 > 0.5 -> tackler wins
    result_frontal = attempt_tackle(0.5, 0.5, rng_reduction=1.0, rng=random.Random(0), params=params,
                                    angle_modifier=params.angle_modifier_frontal)
    assert result_frontal.tackler_won, "Frontal tackle with equal attrs should win deterministically"


# ---------------------------------------------------------------------------
# Statistical win-rate tests (rng_reduction=0.3)
# ---------------------------------------------------------------------------

def _win_rate(tackling: float, dribbling: float, angle_mod: float, n: int, params: TacklingParams) -> float:
    wins = sum(
        1 for seed in range(n)
        if attempt_tackle(tackling, dribbling, RNG_REDUCTION, random.Random(seed), params,
                          angle_modifier=angle_mod).tackler_won
    )
    return wins / n


def test_win_rate_frontal_greater_than_behind(balance_recorder):
    params = TacklingParams.from_config()
    frontal_rate = _win_rate(0.5, 0.5, params.angle_modifier_frontal, N_TRIALS, params)
    side_rate = _win_rate(0.5, 0.5, params.angle_modifier_side, N_TRIALS, params)
    behind_rate = _win_rate(0.5, 0.5, params.angle_modifier_behind, N_TRIALS, params)
    stats = {
        "n": N_TRIALS,
        "frontal_win_pct": round(100 * frontal_rate, 1),
        "side_win_pct": round(100 * side_rate, 1),
        "behind_win_pct": round(100 * behind_rate, 1),
    }
    balance_recorder.report("tackle_angle_win_rates_equal_attrs", stats)
    assert frontal_rate > side_rate > behind_rate, (
        f"Expected frontal > side > behind, got {stats}"
    )


def test_behind_tackle_rarely_wins_equal_attrs(balance_recorder):
    """Tackle from behind with equal attributes should be a big underdog."""
    params = TacklingParams.from_config()
    behind_rate = _win_rate(0.5, 0.5, params.angle_modifier_behind, N_TRIALS, params)
    stats = {"n": N_TRIALS, "behind_win_pct": round(100 * behind_rate, 1)}
    balance_recorder.report("tackle_angle_behind_equal_attrs", stats)
    assert behind_rate < 0.35, f"Behind tackle with equal attrs should win <35%, got {behind_rate:.2%}"


def test_frontal_tackle_favours_tackler_equal_attrs(balance_recorder):
    """Frontal tackle with equal attributes should win more often than not."""
    params = TacklingParams.from_config()
    frontal_rate = _win_rate(0.5, 0.5, params.angle_modifier_frontal, N_TRIALS, params)
    stats = {"n": N_TRIALS, "frontal_win_pct": round(100 * frontal_rate, 1)}
    balance_recorder.report("tackle_angle_frontal_equal_attrs", stats)
    assert frontal_rate > 0.60, f"Frontal tackle with equal attrs should win >60%, got {frontal_rate:.2%}"
