"""Balance tests for the dribble-past-tackle speed penalty.

When a dribbler beats a tackle by a narrow margin they should be noticeably
slowed; when they beat it convincingly (>=35% relative margin on the rolls)
they should be barely slowed at all.

Design targets:
- Narrow-win trials (tackle barely beaten): average speed multiplier clearly
  below 0.7 (i.e. the dribbler is significantly slowed most of the time).
- Convincing-win trials (tackle beaten by >=35% margin): average speed
  multiplier is above 0.9 (mostly full speed retained).
"""
from __future__ import annotations

import random

from footballcoach.engine.tackling import TacklingParams, attempt_tackle

RNG_REDUCTION = 0.3
N_TRIALS = 3000


def _run_trials_collect_multipliers(
    tackling: float,
    dribbling: float,
    rng_reduction: float,
    n: int,
    params: TacklingParams,
) -> list[float]:
    """Returns tacklee_speed_mult for every trial where the dribbler
    wins (tackler_won == False)."""
    results = []
    for seed in range(n):
        r = attempt_tackle(tackling, dribbling, rng_reduction, random.Random(seed), params)
        if not r.tackler_won:
            results.append(r.tacklee_speed_mult)
    return results


def test_dribble_penalty_average_multiplier_reported(balance_recorder):
    """Reports the average speed multiplier across the whole distribution of
    dribbler-wins for a mid-attribute matchup. Not a hard pass/fail test -
    just useful for manual tuning."""
    params = TacklingParams.from_config()
    multipliers = _run_trials_collect_multipliers(0.5, 0.5, RNG_REDUCTION, N_TRIALS, params)
    if not multipliers:
        balance_recorder.report("dribble_speed_penalty_avg", {"note": "no dribbler wins in sample"})
        return
    avg = sum(multipliers) / len(multipliers)
    stats = {
        "n_dribbler_wins": len(multipliers),
        "avg_speed_multiplier": round(avg, 4),
        "min_speed_multiplier": round(min(multipliers), 4),
        "max_speed_multiplier": round(max(multipliers), 4),
    }
    balance_recorder.report("dribble_speed_penalty_avg", stats)


def test_large_margin_wins_barely_slow_dribbler(balance_recorder):
    """A much stronger dribbler (0.9) vs a much weaker tackler (0.1) should
    win by large margins almost every time, meaning the speed multiplier is
    near 1.0 (barely slowed) in the vast majority of dribbler-win trials."""
    params = TacklingParams.from_config()
    multipliers = _run_trials_collect_multipliers(0.1, 0.9, RNG_REDUCTION, N_TRIALS, params)
    avg = sum(multipliers) / max(len(multipliers), 1)
    stats = {
        "n_dribbler_wins": len(multipliers),
        "avg_speed_multiplier": round(avg, 4),
    }
    balance_recorder.report("dribble_penalty_large_margin", stats)
    # Strong dribbler vs weak tackler: wins are convincing, so they should be
    # barely slowed.
    assert avg > 0.85, f"Expected avg speed_mult > 0.85 for dominant dribbler, got {avg:.4f}"


def test_narrow_margin_wins_slow_dribbler_significantly(balance_recorder):
    """A slightly stronger dribbler (0.55) vs a near-equal tackler (0.5) will
    often win by small margins, so the average speed multiplier should be
    noticeably below 1.0 when they do win."""
    params = TacklingParams.from_config()
    multipliers = _run_trials_collect_multipliers(0.5, 0.55, RNG_REDUCTION, N_TRIALS, params)
    avg = sum(multipliers) / max(len(multipliers), 1)
    stats = {
        "n_dribbler_wins": len(multipliers),
        "avg_speed_multiplier": round(avg, 4),
    }
    balance_recorder.report("dribble_penalty_narrow_margin", stats)
    # Near-equal matchup: wins are marginal, so the dribbler should be
    # meaningfully slowed on average.
    assert avg < 0.80, f"Expected avg speed_mult < 0.80 for narrow dribbler wins, got {avg:.4f}"


def test_speed_penalty_zero_at_threshold(balance_recorder):
    """Analytically: at rng_reduction=1.0 the rolls are deterministic
    (= the attribute values scaled by boost). With rng_reduction=1.0:
      tackler_roll = 1.0 * 1.2 * tackling
      dribbler_roll = 1.0 * dribbling
    So with tackling=0.4, dribbling=0.7:
      tackler_roll = 0.48, dribbler_roll = 0.70
      relative_margin = (0.70 - 0.48) / 0.48 = 0.458 >= 0.35 -> mult = 1.0
    """
    params = TacklingParams.from_config()
    result = attempt_tackle(0.4, 0.7, rng_reduction=1.0, rng=random.Random(0), params=params)
    assert not result.tackler_won
    assert result.tacklee_speed_mult == 1.0, (
        f"Expected no slowdown for large margin win, got {result.tacklee_speed_mult}"
    )


def test_speed_penalty_max_at_zero_margin(balance_recorder):
    """At rng_reduction=1.0, with matched attributes where dribbler just
    barely beats the tackle: tackling=0.58, dribbling=0.7:
      tackler_roll = 1.0 * 1.2 * 0.58 = 0.696, dribbler_roll = 0.7
      relative_margin = (0.7 - 0.696) / 0.696 ≈ 0.0057 << 0.35
    Dribbler wins but should be heavily slowed (multiplier near 0.20).
    """
    params = TacklingParams.from_config()
    # tackling=0.555 → tackler_roll = 1.0 * 1.25 * 0.555 = 0.694, dribbler_roll = 0.7
    # relative_margin = (0.7 - 0.694) / 0.694 ≈ 0.009 << 0.35 threshold → heavy slowdown
    result = attempt_tackle(0.555, 0.7, rng_reduction=1.0, rng=random.Random(0), params=params)
    assert not result.tackler_won
    balance_recorder.report("dribble_penalty_near_zero_margin", {"speed_multiplier": result.tacklee_speed_mult})
    # The multiplier should be close to the minimum (1 - 0.80 = 0.20).
    assert result.tacklee_speed_mult < 0.40, (
        f"Expected heavy slowdown near zero margin, got {result.tacklee_speed_mult:.4f}"
    )
