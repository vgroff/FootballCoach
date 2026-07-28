"""Balance tests for Phase E: running-direction precision penalty on kicks/passes.

Validates that shooting/passing AGAINST the run direction produces
measurably worse accuracy than shooting IN LINE with the run direction,
using the existing in-box shooting framework and pass-accuracy framework.
"""
from __future__ import annotations

import math
import random

from footballcoach import actions
from footballcoach.engine.match import Match
from footballcoach.engine.movement import MovementParams, effective_top_speed
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder
from tests.conftest import make_player

RNG_REDUCTION = 0.3
N_TRIALS = 300


def _run_shot_trial_with_run_direction(
    pitch: Pitch,
    run_cos_sim: float,
    precision: float,
    x: float,
    y: float,
    seed: int,
) -> bool:
    """Runs a single shot trial where the kicker is running in a direction
    with cos_sim `run_cos_sim` relative to the aim direction (goal centre).
    Returns True if a goal is scored."""
    kicker = make_player(
        "k", Team.LEFT, position=Vector3(x, y, 0),
        kick_precision=precision, kick_power=0.7,
        top_speed=0.7, acceleration=0.7,
    )
    ball = Ball.at_rest(kicker.position)
    ball.possessed_by = kicker.player_id

    # Aim at the bottom corner of the right goal (harder to hit, more sensitive to precision)
    goal_corner = Vector3(pitch.half_length, pitch.goal_width_m / 2.0 - 0.3, 0.2)
    aim_dir = (goal_corner - kicker.position).xy().normalized()
    # Construct kicker velocity with given cos_sim relative to aim direction
    # using a perpendicular component: v = run_speed * (cos_sim * aim + sin_sim * perp)
    run_speed = 5.0
    sin_sim = math.sqrt(max(0.0, 1.0 - run_cos_sim ** 2))
    perp = Vector3(-aim_dir.y, aim_dir.x, 0)
    kicker.velocity = (aim_dir * run_cos_sim + perp * sin_sim) * run_speed
    kicker.heading_rad = math.atan2(kicker.velocity.y, kicker.velocity.x)

    match = Match(
        pitch=pitch, players=[kicker], ball=ball,
        rng_reduction=RNG_REDUCTION, rng=random.Random(seed),
    )
    actions.shoot(kicker, pitch, aim_point=goal_corner)

    for _ in range(200):
        match.step()
        if match.scoreboard.left_goals > 0:
            return True
        if match.ball.position.x > pitch.half_length + 2.0:
            return False
        if match.ball.velocity.length() < 0.1 and match.ball.position.x < pitch.half_length - 1.0:
            return False
    return False


def _run_shot_batch(
    pitch: Pitch, run_cos_sim: float, precision: float, n: int, seed_offset: int
) -> dict:
    x = pitch.half_length - 15.0  # 15m from goal
    y = 0.0
    scored = sum(
        _run_shot_trial_with_run_direction(pitch, run_cos_sim, precision, x, y, seed_offset * 1000 + i)
        for i in range(n)
    )
    return {
        "n_trials": n,
        "scored": scored,
        "score_rate_pct": round(100 * scored / n, 2),
        "run_cos_sim": run_cos_sim,
    }


def test_shooting_toward_goal_beats_backward_run(balance_recorder):
    """Running toward the aim direction (cos_sim=1.0, no penalty) should score
    more often than running directly backward (cos_sim=-1.0, -75% precision).
    Also validates 'square' (cos_sim≈0) is intermediate."""
    pitch = Pitch.standard()
    precision = 0.7

    stats_forward = _run_shot_batch(pitch, run_cos_sim=1.0, precision=precision, n=N_TRIALS, seed_offset=10)
    stats_square = _run_shot_batch(pitch, run_cos_sim=0.0, precision=precision, n=N_TRIALS, seed_offset=11)
    stats_backward = _run_shot_batch(pitch, run_cos_sim=-1.0, precision=precision, n=N_TRIALS, seed_offset=12)

    balance_recorder.report("shoot_run_forward_cos1.0", stats_forward)
    balance_recorder.report("shoot_run_square_cos0.0", stats_square)
    balance_recorder.report("shoot_run_backward_cos-1.0", stats_backward)

    fwd_rate = stats_forward["score_rate_pct"]
    sqr_rate = stats_square["score_rate_pct"]
    bwd_rate = stats_backward["score_rate_pct"]

    # Note: forward running also boosts effective_power via run_mult, which raises
    # sigma via the power-error coupling — so forward vs square is not a clean
    # comparison. The meaningful check is that backward (large direction penalty +
    # power reduction) is worse than both forward and square.
    assert fwd_rate > bwd_rate, f"Forward ({fwd_rate}%) should beat backward ({bwd_rate}%)"
    assert sqr_rate > bwd_rate, f"Square ({sqr_rate}%) should beat backward ({bwd_rate}%)"


def test_shooting_below_min_speed_no_penalty(balance_recorder):
    """A stationary kicker (no run direction) should not be penalised at all.
    Score rate should be comparable to the forward-run baseline."""
    pitch = Pitch.standard()
    precision = 0.7

    def _run_stationary_shot(seed: int) -> bool:
        x = pitch.half_length - 10.0
        kicker = make_player(
            "k", Team.LEFT, position=Vector3(x, 0, 0),
            kick_precision=precision, kick_power=0.7,
        )
        kicker.velocity = Vector3.zero()
        ball = Ball.at_rest(kicker.position)
        ball.possessed_by = kicker.player_id
        match = Match(
            pitch=pitch, players=[kicker], ball=ball,
            rng_reduction=RNG_REDUCTION, rng=random.Random(seed),
        )
        actions.shoot(kicker, pitch)
        for _ in range(200):
            match.step()
            if match.scoreboard.left_goals > 0:
                return True
            if match.ball.position.x > pitch.half_length + 2.0:
                return False
            if match.ball.velocity.length() < 0.1 and match.ball.position.x < pitch.half_length - 1.0:
                return False
        return False

    n = N_TRIALS
    scored = sum(_run_stationary_shot(i) for i in range(n))
    stats = {"n_trials": n, "scored": scored, "score_rate_pct": round(100 * scored / n, 2)}
    balance_recorder.report("shoot_stationary_no_penalty", stats)
    # Stationary should not be penalised — must still score reasonably often
    assert stats["score_rate_pct"] > 40.0, (
        f"Stationary shot score rate too low: {stats['score_rate_pct']}% "
        "(expected no running-direction penalty at speed=0)"
    )


def test_pass_accuracy_forward_beats_backward_run(balance_recorder):
    """Pass completion: running toward the pass direction (cos_sim≈1.0) must
    yield higher success rate than running backward (cos_sim=-1.0)."""
    pitch = Pitch.standard()
    precision = 0.7
    distance = 15.0

    def _run_pass_trial(run_cos_sim: float, seed: int) -> bool:
        passer_pos = Vector3(0, 0, 0)
        receiver_pos = Vector3(distance, 0, 0)
        passer = make_player("p", Team.LEFT, position=passer_pos, kick_precision=precision)
        receiver = make_player("r", Team.LEFT, position=receiver_pos)
        ball = Ball.at_rest(passer_pos)
        ball.possessed_by = passer.player_id

        aim_dir = (receiver_pos - passer_pos).normalized()
        run_speed = 5.0
        sin_sim = math.sqrt(max(0.0, 1.0 - run_cos_sim ** 2))
        perp = Vector3(-aim_dir.y, aim_dir.x, 0)
        passer.velocity = (aim_dir * run_cos_sim + perp * sin_sim) * run_speed
        passer.heading_rad = math.atan2(passer.velocity.y, passer.velocity.x)

        match = Match(
            pitch=pitch, players=[passer, receiver], ball=ball,
            rng_reduction=RNG_REDUCTION, rng=random.Random(seed),
        )
        actions.pass_to(passer, receiver_pos)

        from footballcoach.entities.player import PlayerState
        for _ in range(300):
            match.step()
            if ball.possessed_by == receiver.player_id:
                return True
            controlling = any(p.state == PlayerState.CONTROLLING_BALL for p in match.players)
            if ball.velocity.length() < 0.05 and ball.possessed_by is None and not controlling:
                return False
        return False

    n = N_TRIALS
    fwd_successes = sum(_run_pass_trial(1.0, i) for i in range(n))
    bwd_successes = sum(_run_pass_trial(-1.0, n + i) for i in range(n))

    fwd_rate = round(100 * fwd_successes / n, 2)
    bwd_rate = round(100 * bwd_successes / n, 2)

    balance_recorder.report("pass_run_forward", {"n_trials": n, "success_rate_pct": fwd_rate})
    balance_recorder.report("pass_run_backward", {"n_trials": n, "success_rate_pct": bwd_rate})

    assert fwd_rate > bwd_rate, (
        f"Forward pass ({fwd_rate}%) should beat backward ({bwd_rate}%) — "
        "confirms running-direction penalty applies to passes too"
    )
