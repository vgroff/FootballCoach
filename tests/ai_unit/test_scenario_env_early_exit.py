"""ScenarioEnv.step() (phase 1) ends a step early exactly when the episode ends
on a box terminal, never otherwise, and never reports a stale transition.

Until 2026-11-16 the early exit tested "the trainee held the ball at the end of
the PREVIOUS tick and the ball is in the opponent box now", read from a flag
carried across steps that the exit path never refreshed. A ball pushed loose
into the box therefore cut the step short without ending the episode, every
following step then lasted one tick while the ball stayed there, and since the
trainee decides every _ticks_per_decision ticks those steps re-reported its
previous decision as new rows. The exit now uses the same detector as `done`
(detect_phase1_box_terminal), on the same tick, for both sides.
"""
from __future__ import annotations

import functools

import pytest
import torch

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import NeuralPlayerAI, StopWhenIdleAI
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


def _env(seed: int = 5, **kw) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="early_exit_test", label="early exit", description="early exit",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=0.0, opponent_immobile_prob=1.0),
    )
    env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1,
                      secondary_player_ids=[], max_episode_s=30.0, **kw)
    env.reset(seed=seed)
    return env


def _place(env: ScenarioEnv, *, holder: str | None, ball_in: str, holder_dist_m: float = 0.0) -> None:
    """Both players stationary with controllers; the ball stationary in the
    opponent's box ("opp_box", the box the trainee attacks), the trainee's own
    box ("own_box") or the centre spot. ``holder`` gets possession (None =
    loose ball, with the trainee ``holder_dist_m`` away from it)."""
    m = env._loop.match
    tr = m.player_by_id("trainee")
    op = m.player_by_id("opponent")
    attack_sign = -1.0 if tr.team == Team.RIGHT else 1.0   # the box step() checks is left=(team == RIGHT)
    bx = {"opp_box": attack_sign * (m.pitch.half_length - 5.0), "own_box": -attack_sign * (m.pitch.half_length - 5.0),
          "centre": 0.0}[ball_in]
    for p in (tr, op):
        p.ai = StopWhenIdleAI()
        p.current_order = None
        p.velocity = Vector3.zero()
    far = Vector3(0.0, 30.0, 0.0)
    if holder == "trainee":
        tr.position, op.position = Vector3(bx, 0.0, 0.0), far
    elif holder == "opponent":
        op.position, tr.position = Vector3(bx, 0.0, 0.0), far
    else:
        tr.position, op.position = Vector3(bx, holder_dist_m, 0.0), far
    m.ball.position = Vector3(bx, 0.0, m.ball.radius_m)
    m.ball.velocity = Vector3.zero()
    if holder is not None:
        m._set_possession(holder)
        m._sync_possessed_ball()
    else:
        m.ball.possessed_by = None


def test_holding_the_ball_in_the_opponent_box_ends_the_episode_on_the_first_tick():
    env = _env()
    _place(env, holder="trainee", ball_in="opp_box")
    _, _, done, info = env.step()
    assert done and info.trial_outcome == "box_possession" and info.ticks_elapsed == 1


def test_the_opponent_holding_the_ball_in_the_trainee_box_ends_it_on_the_first_tick():
    # Before 2026-11-16 this side was only checked at the end of a full step.
    env = _env()
    _place(env, holder="opponent", ball_in="own_box")
    _, _, done, info = env.step()
    assert done and info.trial_outcome == "opponent_box_possession" and info.ticks_elapsed == 1


def test_a_loose_ball_in_the_opponent_box_does_not_cut_the_step_short():
    # The old bug: the trainee had the ball last step, now it is loose in the box.
    env = _env()
    _place(env, holder="trainee", ball_in="centre")
    env.step()
    _place(env, holder=None, ball_in="opp_box", holder_dist_m=10.0)
    for _ in range(3):
        before = env._episode_ticks
        _, _, done, _ = env.step()
        assert not done
        assert env._episode_ticks - before == env._ticks_per_decision


def test_no_early_exit_outside_the_box():
    env = _env()
    _place(env, holder="trainee", ball_in="centre")
    _, _, done, info = env.step()
    assert not done and info.ticks_elapsed == env._ticks_per_decision


def test_an_uncontrolled_holder_in_a_box_is_not_a_terminal():
    # can_score_box_terminal: a player with no controller cannot win a box terminal.
    from footballcoach.orders import JogOrder
    env = _env()
    _place(env, holder="trainee", ball_in="opp_box")
    tr = env._loop.match.player_by_id("trainee")
    tr.ai = None
    tr.current_order = JogOrder(direction=Vector3.zero())   # the immobile-player build path's order
    _, _, done, info = env.step()
    assert not done and info.ticks_elapsed == env._ticks_per_decision


# --- the stale-transition guard ----------------------------------------------

@pytest.fixture(scope="module")
def sampler():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    return PPOTrainer.from_config(device=torch.device("cpu"))._sample_action


def test_a_step_without_a_new_decision_reports_no_row(sampler):
    """Force the trainee to decide every 2nd step (its interval is twice the
    env's): every other step has no new decision, and must report None
    instead of re-reporting the previous decision."""
    env = _env()
    env.sample_action_fn = sampler
    env.reset(seed=7)
    player = env._loop.match.player_by_id("trainee")
    ai = player.ai
    assert isinstance(ai, NeuralPlayerAI)
    ai.decision_interval_ticks = 2 * env._ticks_per_decision
    ai._ticks_since_decision = ai.decision_interval_ticks     # due on the first tick, as after reset
    reported = []
    for _ in range(6):
        _, _, done, _ = env.step()
        reported.append(env.last_trainee_transition)
        if done:
            break
    fresh = [r for r in reported if r is not None]
    assert len(fresh) == len({id(r) for r in fresh})          # no row reported twice
    assert reported[1] is None and env.stale_transition_steps >= 1


def test_ordinary_steps_always_carry_a_fresh_decision(sampler):
    env = _env()
    env.sample_action_fn = sampler
    for seed in range(3):
        env.reset(seed=seed)
        done, prev = False, None
        while not done:
            _, _, done, _ = env.step()
            tr = env.last_trainee_transition
            assert tr is not None and tr is not prev
            prev = tr
    assert env.stale_transition_steps == 0
