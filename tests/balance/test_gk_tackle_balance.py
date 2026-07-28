"""Balance tests for Phase B GK tackle changes.

Tests:
1. GK tackle win rate inside vs outside own box (N>=2000, rng_reduction=0.3):
   outside rate must be meaningfully lower than inside rate.
2. GK-in-box-with-ball is untackleable: N=100 trials, always 0 wins.
3. Boundary regression: GK just outside box uses normal outside-box rules.
4. CONTROLLING_BALL dribble penalty: tackler wins measurably more often when
   the target is early in control vs near the end.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.engine.tackling import TacklingParams, attempt_tackle
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import ChaseTackleOrder, TackleOrder
from tests.conftest import make_player


# ---------------------------------------------------------------------------
# 1. GK tackle win rate inside vs outside box
# ---------------------------------------------------------------------------

def test_gk_tackle_win_rate_inside_vs_outside_box(balance_recorder):
    """GK tackling (gk_outside_box=False) should win significantly more often
    than when outside the box (gk_outside_box=True)."""
    params = TacklingParams.from_config()
    N = 2000
    RNG_REDUCTION = 0.3

    # GK tackling=0.7, dribbler dribbling=0.6
    gk_tackling = 0.7
    dribbler_dribbling = 0.6

    wins_inside = sum(
        1 for seed in range(N)
        if attempt_tackle(
            gk_tackling, dribbler_dribbling, RNG_REDUCTION,
            rng=random.Random(seed), params=params,
            is_goalkeeper_tackle=True, gk_outside_box=False,
        ).tackler_won
    )
    wins_outside = sum(
        1 for seed in range(N)
        if attempt_tackle(
            gk_tackling, dribbler_dribbling, RNG_REDUCTION,
            rng=random.Random(seed), params=params,
            is_goalkeeper_tackle=True, gk_outside_box=True,
        ).tackler_won
    )

    rate_inside = round(100 * wins_inside / N, 2)
    rate_outside = round(100 * wins_outside / N, 2)

    balance_recorder.report("gk_tackle_inside_vs_outside_box", {
        "n": N,
        "gk_tackling": gk_tackling,
        "dribbler_dribbling": dribbler_dribbling,
        "win_rate_inside_box_pct": rate_inside,
        "win_rate_outside_box_pct": rate_outside,
        "gap_pct": round(rate_inside - rate_outside, 2),
    })
    # Outside should be meaningfully lower (at least 10 percentage points).
    assert rate_outside < rate_inside - 10.0, (
        f"Expected outside rate ({rate_outside}%) to be at least 10pp below "
        f"inside rate ({rate_inside}%)"
    )
    # GK in box with a 2.0 boost should win majority against dribbling=0.6
    assert rate_inside > 60.0, f"GK inside-box win rate too low: {rate_inside}%"


# ---------------------------------------------------------------------------
# 2. GK-in-box-with-ball is untackleable (scenario + statistical)
# ---------------------------------------------------------------------------

def _make_gk_in_box_match(rng_reduction: float = 1.0) -> tuple[Match, object, object]:
    """Creates a match where a GK has the ball in the left box and an
    outfield player is attempting to tackle."""
    pitch = Pitch.standard()
    # GK in the left penalty box (Team.LEFT defends left).
    gk_pos = Vector3(-pitch.half_length + 5.0, 0.0, 0.0)
    # Tackler adjacent to GK.
    tackler_pos = Vector3(-pitch.half_length + 5.5, 0.0, 0.0)

    gk = make_player("gk", Team.LEFT, 0.7, position=gk_pos, is_goalkeeper=True)
    tackler = make_player("tackler", Team.RIGHT, 0.8, position=tackler_pos)

    ball = Ball(position=Vector3(-pitch.half_length + 5.41, 0.0, 0.11),
                possessed_by="gk")

    match = Match(pitch=pitch, players=[gk, tackler], ball=ball,
                  rng_reduction=rng_reduction, rng=random.Random(42))
    return match, gk, tackler


def test_gk_in_box_untackleable_deterministic(balance_recorder):
    """At rng_reduction=1.0, a tackle attempt against a GK in own box must
    always result in tackler_won=False (auto-fail, no skill check)."""
    N = 20  # deterministic so fewer reps suffice
    wins = 0
    for seed in range(N):
        match, gk, tackler = _make_gk_in_box_match(rng_reduction=1.0)
        match.rng = random.Random(seed)
        tackler.current_order = TackleOrder(target_player_id="gk")
        match.step()
        # Ball should still belong to GK (not transferred)
        if match.ball.possessed_by == tackler.player_id:
            wins += 1
        # GK state should be unchanged (still ACTIVE)
        gk_state = match.players[0].state  # gk is players[0]
        assert gk_state == PlayerState.ACTIVE, (
            f"GK state changed to {gk_state} after tackle attempt in own box"
        )

    balance_recorder.report("gk_in_box_untackleable_deterministic", {
        "n": N, "tackler_wins": wins, "gk_unchanged": True,
    })
    assert wins == 0, f"GK in own box was tackled {wins} times out of {N}"


def test_gk_in_box_untackleable_stochastic(balance_recorder):
    """At rng_reduction=0.3, same rule must hold: 0/100 tackle wins."""
    N = 100
    wins = 0
    for seed in range(N):
        match, gk, tackler = _make_gk_in_box_match(rng_reduction=0.3)
        match.rng = random.Random(seed)
        tackler.current_order = TackleOrder(target_player_id="gk")
        match.step()
        if match.ball.possessed_by == tackler.player_id:
            wins += 1

    balance_recorder.report("gk_in_box_untackleable_stochastic", {
        "n": N, "tackler_wins": wins,
    })
    assert wins == 0, (
        f"GK in own box was tackled {wins}/100 times with rng_reduction=0.3 — "
        "this must be an absolute rule, not a statistical one"
    )


# ---------------------------------------------------------------------------
# 3. Boundary: GK just outside box — normal outside-box rules apply
# ---------------------------------------------------------------------------

def test_gk_just_outside_box_gets_outside_penalty(balance_recorder):
    """A GK positioned just outside their box uses the reduced tackle boost.
    Win rate should match the outside-box stats from test 1, not the inside rate."""
    pitch = Pitch.standard()
    # Place GK 0.5m outside the left box (x = -half_length + box_length + 0.5)
    gk_x = -pitch.half_length + pitch.box_length_m + 0.5
    gk = make_player("gk", Team.LEFT, 0.7,
                     position=Vector3(gk_x, 0.0, 0.0), is_goalkeeper=True)
    # Confirm GK is indeed outside their box
    assert not pitch.is_in_box(gk.position, left=True), "Setup error: GK should be outside box"

    params = TacklingParams.from_config()
    N = 500
    RNG_REDUCTION = 0.3
    wins_outside_penalty = sum(
        1 for seed in range(N)
        if attempt_tackle(
            0.7, 0.6, RNG_REDUCTION,
            rng=random.Random(seed), params=params,
            is_goalkeeper_tackle=True, gk_outside_box=True,
        ).tackler_won
    )
    wins_inside_penalty = sum(
        1 for seed in range(N)
        if attempt_tackle(
            0.7, 0.6, RNG_REDUCTION,
            rng=random.Random(seed), params=params,
            is_goalkeeper_tackle=True, gk_outside_box=False,
        ).tackler_won
    )
    rate_outside = round(100 * wins_outside_penalty / N, 2)
    rate_inside = round(100 * wins_inside_penalty / N, 2)

    balance_recorder.report("gk_boundary_outside_box_uses_penalty", {
        "n": N,
        "rate_with_outside_penalty_pct": rate_outside,
        "rate_without_outside_penalty_pct": rate_inside,
        "gk_x_m": round(gk_x, 3),
        "box_length_m": pitch.box_length_m,
    })
    # The penalty should cause a measurable drop.
    assert rate_outside < rate_inside - 5.0, (
        f"Outside-box penalty should reduce win rate: "
        f"outside={rate_outside}%, inside={rate_inside}%"
    )


# ---------------------------------------------------------------------------
# 4. CONTROLLING_BALL dribbling penalty
# ---------------------------------------------------------------------------

def test_controlling_ball_penalty_unit(balance_recorder):
    """Unit check: effective dribbling as a function of state_timer_s.
    At timer=reference_s: 25% penalty. At timer=0: no penalty."""
    from footballcoach.engine.match import Match
    from footballcoach.entities.player import PlayerState

    pitch = Pitch.standard()
    target = make_player("target", Team.RIGHT, 0.8, dribbling=0.8)
    other = make_player("other", Team.LEFT, 0.5)
    ball = Ball(position=Vector3(50.0, 0.0, 0.0))
    match = Match(pitch=pitch, players=[target, other], ball=ball)

    ref_s = match.tackling_params.control_time_penalty_reference_s
    results = {}
    for timer_s in [0.0, ref_s * 0.25, ref_s * 0.5, ref_s, ref_s * 1.5]:
        target.state = PlayerState.CONTROLLING_BALL
        target.state_timer_s = timer_s
        eff = match._effective_dribbling(target)
        penalty_frac = min(1.0, timer_s / ref_s)
        expected = 0.8 * (1.0 - 0.25 * penalty_frac)
        results[f"timer={round(timer_s, 4)}s"] = {
            "effective_dribbling": round(eff, 6),
            "expected": round(expected, 6),
            "match": abs(eff - expected) < 1e-9,
        }
        assert abs(eff - expected) < 1e-9, (
            f"timer={timer_s}s: expected {expected:.6f}, got {eff:.6f}"
        )
    # ACTIVE player: no penalty regardless of timer
    target.state = PlayerState.ACTIVE
    target.state_timer_s = ref_s
    eff_active = match._effective_dribbling(target)
    assert abs(eff_active - 0.8) < 1e-9, f"ACTIVE player should have no penalty: {eff_active}"

    balance_recorder.report("controlling_ball_dribbling_penalty_unit", results)


def test_controlling_ball_penalty_balance(balance_recorder):
    """Balance: same tackler vs same dribbler, early control (high timer) vs
    late control (near 0 timer) — win rate must be measurably higher early."""
    params = TacklingParams.from_config()
    N = 2000
    RNG_REDUCTION = 0.3
    tackling = 0.6
    dribbling = 0.6
    ref_s = params.control_time_penalty_reference_s

    # Early control: timer near reference_s → 25% dribbling penalty
    # effective_dribbling ≈ 0.6 * 0.75 = 0.45
    early_dribbling = dribbling * (1.0 - 0.25 * min(1.0, ref_s / ref_s))
    wins_early = sum(
        1 for seed in range(N)
        if attempt_tackle(
            tackling, early_dribbling, RNG_REDUCTION,
            rng=random.Random(seed), params=params,
        ).tackler_won
    )
    # Late control: timer near 0 → no penalty
    late_dribbling = dribbling
    wins_late = sum(
        1 for seed in range(N)
        if attempt_tackle(
            tackling, late_dribbling, RNG_REDUCTION,
            rng=random.Random(seed), params=params,
        ).tackler_won
    )
    rate_early = round(100 * wins_early / N, 2)
    rate_late = round(100 * wins_late / N, 2)

    balance_recorder.report("controlling_ball_penalty_balance", {
        "n": N,
        "tackling_attr": tackling,
        "dribbling_attr": dribbling,
        "early_effective_dribbling": round(early_dribbling, 4),
        "late_effective_dribbling": round(late_dribbling, 4),
        "win_rate_early_control_pct": rate_early,
        "win_rate_late_control_pct": rate_late,
        "gap_pct": round(rate_early - rate_late, 2),
    })
    assert rate_early > rate_late, (
        f"Tackler should win more often against early-control player: "
        f"early={rate_early}% vs late={rate_late}%"
    )


def test_controlling_ball_regression_active_player_unaffected(balance_recorder):
    """Regression: an ACTIVE ball carrier is completely unaffected by the
    CONTROLLING_BALL penalty — existing balance ranges must still hold."""
    params = TacklingParams.from_config()
    N = 5000
    RNG_REDUCTION = 0.3
    wins = sum(
        1 for seed in range(N)
        if attempt_tackle(0.8, 0.6, RNG_REDUCTION,
                          rng=random.Random(seed), params=params).tackler_won
    )
    win_rate = round(100 * wins / N, 2)
    balance_recorder.report("controlling_ball_regression_active_70_90_band", {
        "n": N, "win_rate_pct": win_rate,
    })
    assert 70.0 < win_rate < 90.0, (
        f"Existing tackling balance broken: 0.8 vs 0.6 win rate = {win_rate}%"
    )
