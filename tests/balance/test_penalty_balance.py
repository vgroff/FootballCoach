"""Balance tests: penalty kick scoring rates at various precision levels and
aim points, run at rng_reduction=0.3 (the default game setting) with a large
number of trials. These directly encode the user's explicit design targets:

- precision 0.5, aim dead centre: >95% scored
- precision 0.5, aim bottom corner: 50-80% scored
- precision 0.8, aim bottom corner: 85-95% scored

Every test reports full statistics via `balance_recorder`, not just pass/fail.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder
from tests.conftest import make_player

N_TRIALS = 2000
RNG_REDUCTION = 0.3


def _run_penalty_trials(precision: float, aim_offset_y: float, aim_offset_z: float, n_trials: int) -> dict:
    pitch = Pitch.standard()
    penalty_spot = pitch.penalty_spot(left=False)
    goal_centre = pitch.right_goal_centre

    scored = 0
    for seed in range(n_trials):
        kicker = make_player(
            "p1", team=Team.LEFT, position=penalty_spot,
            kick_precision=precision, kick_power=0.7,
        )
        ball = Ball.at_rest(penalty_spot)
        ball.possessed_by = "p1"
        rng = random.Random(seed)
        match = Match(pitch=pitch, players=[kicker], ball=ball, rng_reduction=RNG_REDUCTION, rng=rng)

        aim_point = goal_centre + Vector3(0, aim_offset_y, aim_offset_z)
        kicker.current_order = KickOrder(aim_point=aim_point, power_fraction=0.65, spin=Vector3.zero())

        for _ in range(150):
            match.step()
            if match.scoreboard.left_goals > 0:
                scored += 1
                break
            if match.ball.position.x > pitch.half_length + 2.0:
                break  # ball flew past goal line wide/high - clearly a miss
            if match.ball.velocity.length() < 0.1 and match.ball.position.x < pitch.half_length - 1.0:
                break  # ball stopped well short - miss (blocked/rolled dead)

    return {
        "n_trials": n_trials,
        "scored": scored,
        "score_rate_pct": round(100 * scored / n_trials, 2),
    }


def test_penalty_precision_05_centre_aim_scores_over_95_percent(balance_recorder):
    stats = _run_penalty_trials(precision=0.5, aim_offset_y=0.0, aim_offset_z=1.1, n_trials=N_TRIALS)
    balance_recorder.report("penalty_precision_0.5_centre_aim", stats)
    assert stats["score_rate_pct"] > 95.0


def test_penalty_precision_05_corner_aim_scores_50_to_80_percent(balance_recorder):
    pitch = Pitch.standard()
    corner_offset_y = pitch.goal_width_m / 2.0 - 0.475
    stats = _run_penalty_trials(precision=0.5, aim_offset_y=corner_offset_y, aim_offset_z=0.475, n_trials=N_TRIALS)
    balance_recorder.report("penalty_precision_0.5_corner_aim", stats)
    assert 50.0 <= stats["score_rate_pct"] <= 80.0


def test_penalty_precision_08_corner_aim_scores_85_to_95_percent(balance_recorder):
    pitch = Pitch.standard()
    corner_offset_y = pitch.goal_width_m / 2.0 - 0.475
    stats = _run_penalty_trials(precision=0.8, aim_offset_y=corner_offset_y, aim_offset_z=0.475, n_trials=N_TRIALS)
    balance_recorder.report("penalty_precision_0.8_corner_aim", stats)
    assert 85.0 <= stats["score_rate_pct"] <= 95.0
