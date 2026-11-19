"""Keeper save-target shaping (reward.phase2.shaping.goalkeeper.save_target_coef,
ai/env/role_env._keeper_shaping_tick): a pure potential while the ball is
loose and heading for the keeper's goal, zero while the ball is held."""
from __future__ import annotations

import pytest

from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.engine.goalkeeping import ball_aimed_at_goal, keeper_save_target
from footballcoach.mathutils import Vector3

PHASE2 = PHASES_BY_ID[2]
RULES = {r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")}


def _env(coef=0.1):
    env = build_role_env(PHASE2, controller_weights=RULES)
    env._gk_shaping_coef = coef
    env.reset(seed=5)
    return env


def _shoot(match, speed=20.0):
    """A loose ball flying at the keeper's goal from 20 m out."""
    gk = match.player_by_id("keeper")
    goal = match.pitch.right_goal_centre if gk.team.name == "RIGHT" else match.pitch.left_goal_centre
    sign = 1.0 if goal.x > 0 else -1.0
    match.ball.possessed_by = None
    match.ball.position = Vector3(goal.x - sign * 20.0, 1.5, 0.3)
    match.ball.velocity = Vector3(sign * speed, -0.1, 0.0)
    return gk


def test_no_shaping_while_the_ball_is_held():
    env = _env()
    m = env.match
    assert m.ball.possessed_by is not None
    env._keeper_shaping_tick(m)
    env._keeper_shaping_tick(m)
    assert env._gk_shaping_step == 0.0 and env._gk_prev_dist is None


def test_closing_on_the_save_target_pays_and_the_sum_telescopes():
    env = _env(coef=0.1)
    m = env.match
    gk = _shoot(m)
    assert ball_aimed_at_goal(m, gk.team)
    target = keeper_save_target(gk, m)
    d0 = gk.position.xy().distance_to(target.xy())
    gk.position = Vector3(gk.position.x, gk.position.y + 3.0, 0.0)     # 3 m off the line of the shot
    start = gk.position.xy().distance_to(keeper_save_target(gk, m).xy())
    env._keeper_shaping_tick(m)                                         # first gated tick: sets the reference
    assert env._gk_shaping_step == 0.0
    path = []
    for k in range(1, 4):                                               # walk towards the target
        tgt = keeper_save_target(gk, m)
        step = (tgt.xy() - gk.position.xy()) * 0.25
        gk.position = Vector3(gk.position.x + step.x, gk.position.y + step.y, 0.0)
        env._keeper_shaping_tick(m)
        path.append(gk.position.xy().distance_to(keeper_save_target(gk, m).xy()))
    assert env._gk_shaping_step > 0.0
    assert env._gk_shaping_step == pytest.approx(0.1 * (start - path[-1]))
    assert d0 >= 0.0


def test_moving_away_costs_the_same_rate():
    env = _env(coef=0.1)
    m = env.match
    gk = _shoot(m)
    env._keeper_shaping_tick(m)
    d1 = gk.position.xy().distance_to(keeper_save_target(gk, m).xy())
    gk.position = Vector3(gk.position.x, gk.position.y + 2.0, 0.0)
    env._keeper_shaping_tick(m)
    d2 = gk.position.xy().distance_to(keeper_save_target(gk, m).xy())
    assert env._gk_shaping_step == pytest.approx(0.1 * (d1 - d2)) and env._gk_shaping_step < 0


def test_off_by_default_coefficient_zero_adds_nothing():
    env = build_role_env(PHASE2, controller_weights=RULES)
    env._gk_shaping_coef = 0.0
    env.reset(seed=5)
    done = False
    while not done:
        _, r, done, info = env.step()
    assert "gk_save_target" not in env.last_reward_components


def test_a_neural_keeper_trainee_receives_the_shaping_in_real_episodes():
    import random
    import torch
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    env = build_role_env(PHASE2, controller_weights={"attacker": {"rules": 1}, "defender": {"rules": 1},
                                                    "goalkeeper": {"neural": 1}})
    env.sample_action_fn = trainer._sample_action
    env._gk_shaping_coef = 0.1
    seen = []
    for seed in range(6):
        env.rng = random.Random(seed)
        env.reset(seed=seed)
        if env.trainee_player_id != "keeper":
            continue
        done = False
        while not done:
            _, r, done, _ = env.step()
            if "gk_save_target" in env.last_reward_components and env.last_reward_components["gk_save_target"] != 0.0:
                seen.append(env.last_reward_components["gk_save_target"])
    assert seen, "no non-zero keeper shaping in 6 keeper-trainee episodes"
