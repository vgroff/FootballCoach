"""Balance tests for the goalkeeper 'Save' action: predicting the incoming
shot's goal-line crossing point and moving there. Mirrors the shape of the
user's other balance-test requests (good vs bad players, randomly generated
but reasonable scenarios), applied to goalkeeping since there's no single
literal numeric target given for saves (unlike shoot/pass/tackle) - instead
we validate that save rate responds sensibly to goalkeeper attributes and
shot difficulty.
"""
from __future__ import annotations

import random

from footballcoach import actions
from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder
from tests.conftest import make_player

RNG_REDUCTION = 0.3
MAX_TICKS = 300


def _run_save_trial(
    pitch: Pitch,
    gk_top_speed: float,
    gk_acceleration: float,
    gk_ball_control: float,
    gk_start_y: float,
    shot_x: float,
    aim_y: float,
    aim_z: float,
    power: float,
    precision: float,
    seed: int,
) -> str:
    """Returns 'saved', 'scored', 'miss' (ball went dead without a goal or
    save - e.g. wide/over), or 'timeout'."""
    gk_position = pitch.left_goal_centre + Vector3(0, gk_start_y, 0)
    gk = make_player(
        "gk", Team.LEFT, position=gk_position, is_goalkeeper=True,
        top_speed=gk_top_speed, acceleration=gk_acceleration, ball_control=gk_ball_control,
    )
    shooter = make_player(
        "s", Team.RIGHT, position=Vector3(shot_x, 0, 0),
        kick_precision=precision, kick_power=power,
    )
    ball = Ball.at_rest(shooter.position)
    ball.possessed_by = "s"
    match = Match(pitch=pitch, players=[gk, shooter], ball=ball, rng_reduction=RNG_REDUCTION, rng=random.Random(seed))

    actions.save(gk)
    aim_point = pitch.left_goal_centre + Vector3(0, aim_y, aim_z)
    shooter.current_order = KickOrder(aim_point=aim_point, power_fraction=power, spin=Vector3.zero())

    for _ in range(MAX_TICKS):
        match.step()
        if ball.possessed_by == "gk":
            return "saved"
        if match.scoreboard.right_goals > 0:
            return "scored"
        if (
            ball.velocity.length() < 0.1
            and ball.possessed_by is None
            and gk.state.name != "CONTROLLING_BALL"
            and match.time_s > 1.0
        ):
            return "miss"
    return "timeout"


def test_fast_goalkeeper_saves_far_post_shot_much_more_than_slow_one(balance_recorder):
    """Goalkeeper starts pinned to the near post, shot is precisely placed
    in the far corner - a scenario that specifically requires the keeper to
    travel across goal in time, isolating movement attributes from the
    (separately balance-tested) penalty/shot-accuracy mechanics."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    gk_start_y = -half_goal_w + 0.3
    aim_y = half_goal_w - 0.3
    aim_z = 0.3
    shot_x = -11.0
    power = 0.9
    n = 300

    def run_for(gk_attr: float) -> dict:
        outcomes = [
            _run_save_trial(pitch, gk_attr, gk_attr, 0.6, gk_start_y, shot_x, aim_y, aim_z, power, 0.95, seed)
            for seed in range(n)
        ]
        saved = outcomes.count("saved")
        return {"n_trials": n, "saved": saved, "save_rate_pct": round(100 * saved / n, 2)}

    fast_stats = run_for(1.0)
    slow_stats = run_for(0.0)
    balance_recorder.report(
        "save_far_post_fast_vs_slow_gk",
        {"fast_gk": fast_stats, "slow_gk": slow_stats},
    )
    assert fast_stats["save_rate_pct"] > slow_stats["save_rate_pct"] + 30.0
    assert fast_stats["save_rate_pct"] > 80.0
    assert slow_stats["save_rate_pct"] < 30.0


def test_centrally_placed_goalkeeper_saves_centre_aimed_shot_almost_always(balance_recorder):
    """A shot aimed dead centre with the keeper already positioned there
    (the common case - see save_target_position's default behaviour) should
    be saved essentially every time, regardless of attribute level, since
    minimal movement is required."""
    pitch = Pitch.standard()
    n = 200

    def run_for(gk_attr: float) -> dict:
        outcomes = [
            _run_save_trial(pitch, gk_attr, gk_attr, 0.5, 0.0, -20.0, 0.0, 1.1, 0.6, 0.6, seed)
            for seed in range(n)
        ]
        saved = outcomes.count("saved")
        return {"n_trials": n, "saved": saved, "save_rate_pct": round(100 * saved / n, 2)}

    stats = {"good_gk": run_for(0.8), "bad_gk": run_for(0.1)}
    balance_recorder.report("save_centre_aim_easy_case", stats)
    assert stats["good_gk"]["save_rate_pct"] > 95.0
    assert stats["bad_gk"]["save_rate_pct"] > 95.0


def test_save_rate_table_across_gk_speed_and_shot_placement(balance_recorder):
    """Not a hard-target test - a table across a few goalkeeper speed levels
    and shot placements, for visual balance inspection. The keeper always
    starts pinned to the near post here (rather than centred), so shots
    placed at the far post genuinely require covering ground - a centred
    keeper facing a close-range, precisely-placed shot saves it regardless
    of speed (see test_centrally_placed_goalkeeper_saves_centre_aimed_shot_
    almost_always), which would make this table's "does speed matter"
    sanity check meaningless."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    gk_start_y = -half_goal_w + 0.3  # pinned to near post
    table = {}
    for gk_speed in (0.1, 0.5, 0.9):
        for label, aim_y in (
            ("near_post", -half_goal_w + 0.4),
            ("centre", 0.0),
            ("far_post", half_goal_w - 0.3),
        ):
            n = 150
            outcomes = [
                _run_save_trial(pitch, gk_speed, gk_speed, 0.5, gk_start_y, -11.0, aim_y, 0.3, 0.9, 0.9, seed)
                for seed in range(n)
            ]
            saved = outcomes.count("saved")
            table[f"gk_speed={gk_speed}_{label}"] = round(100 * saved / n, 1)
    balance_recorder.report("save_rate_grid_pct", table)
    # Sanity: for the far-post (hardest, requires most travel from a
    # near-post start) shot, higher GK speed should save more often than lower.
    assert table["gk_speed=0.9_far_post"] > table["gk_speed=0.1_far_post"]


def test_random_scenario_batch_good_vs_bad_goalkeeper(balance_recorder):
    """Randomly generates (reasonable) shot distances, placements, and
    goalkeeper starting offsets, comparing an overall save-rate for a
    consistently good vs consistently bad goalkeeper across the same random
    scenario set. Goalkeeper starting offset is deliberately biased away
    from the shot's target side (so a real save typically requires covering
    some ground, rather than the shot already being aimed near wherever the
    keeper happens to be standing) - otherwise both a good and bad keeper
    trivially save most attempts and the comparison is meaningless (as an
    earlier version of this test found: both ends of the attribute range
    saved ~97% when the keeper's start position and shot placement were
    independently random)."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    rng = random.Random(99)
    n = 150
    scenarios = []
    for _ in range(n):
        shot_x = -rng.uniform(9.0, 16.0)
        aim_side = rng.choice([-1.0, 1.0])
        aim_y = aim_side * rng.uniform(half_goal_w * 0.5, half_goal_w - 0.3)
        aim_z = rng.uniform(0.1, 0.6)
        gk_start_y = -aim_side * rng.uniform(half_goal_w * 0.5, half_goal_w - 0.3)  # opposite side
        power = rng.uniform(0.7, 0.95)
        scenarios.append((shot_x, aim_y, aim_z, gk_start_y, power))

    def run_for(gk_attr: float) -> int:
        saved = 0
        for i, (shot_x, aim_y, aim_z, gk_start_y, power) in enumerate(scenarios):
            outcome = _run_save_trial(pitch, gk_attr, gk_attr, gk_attr, gk_start_y, shot_x, aim_y, aim_z, power, 0.9, seed=i)
            if outcome == "saved":
                saved += 1
        return saved

    good_saved = run_for(0.85)
    bad_saved = run_for(0.15)

    stats = {
        "n_scenarios": n,
        "good_gk_save_rate_pct": round(100 * good_saved / n, 1),
        "bad_gk_save_rate_pct": round(100 * bad_saved / n, 1),
    }
    balance_recorder.report("save_random_scenarios_good_vs_bad", stats)
    assert stats["good_gk_save_rate_pct"] > stats["bad_gk_save_rate_pct"]
