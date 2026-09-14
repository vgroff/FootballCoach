"""Regression test: ScenarioEnv._compute_phase1_reward_for_player() must
forward heading_cos_sim/player_speed_mps/stamina_used into phase1_reward().

These three kwargs were computed via _player_speed_and_heading_cos() and
start_stamina but never passed through to the phase1_reward() call, so the
'hdg' (heading), 'appr_sq' (approach_speed), and 'stam' (stamina_penalty)
reward components were always silently zero regardless of actual player
motion -- see ai_trainer_knowledge.md for the incident writeup. This test
exercises the real ScenarioEnv wiring end-to-end (not just phase1_reward()
in isolation, which was already covered by tests/ai_unit/test_reward.py and
would not have caught this class of bug).
"""
from __future__ import annotations

import copy
import math

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.mathutils.vector3 import Vector3
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_CFG1 = load_ai_config()["reward"]["phase1"]


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="reward_wiring_1v1",
        label="Reward wiring test: 1v1",
        description="Regression test for heading/appr_sq/stamina reward wiring",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
    )


class TestPhase1RewardWiring:
    def test_heading_and_approach_speed_nonzero_when_moving_away_fast(self):
        """Directly reproduces the bug: set the trainee's velocity to move
        FAST directly AWAY from the ball, then call the real
        _compute_phase1_reward_for_player() wiring (not phase1_reward()
        directly) and assert 'hdg'/'appr_sq' are nonzero -- prior to the
        fix, both were always exactly 0.0 regardless of this setup because
        heading_cos_sim/player_speed_mps were never forwarded.
        """
        # heading_penalty_coef is 0.0 in the live config (disabled); override
        # locally on the env's own reward-cfg dict so the wiring under test
        # still exercises the penalty formula.
        env = _make_env()
        env._reward_cfg["phase1"] = {**env._reward_cfg["phase1"], "heading_penalty_coef": 0.15}
        env.reset()
        match = env._loop.match
        player = match.player_by_id("trainee")
        ball = match.ball

        # Point the player's velocity directly AWAY from the ball, at a
        # speed comfortably above heading_penalty_min_speed_mps.
        min_speed = float(_CFG1.get("heading_penalty_min_speed_mps", 0.5))
        speed = min_speed + 3.0
        dx = player.position.x - ball.position.x
        dy = player.position.y - ball.position.y
        dist = math.hypot(dx, dy)
        assert dist > 1e-3, "player spawned on top of the ball -- flaky fixture, adjust scenario"
        ux, uy = dx / dist, dy / dist
        player.velocity = Vector3(x=ux * speed, y=uy * speed, z=0.0)
        # heading_rad must match velocity direction (see repo convention:
        # turning mechanics fight the initial velocity otherwise).
        player.heading_rad = math.atan2(uy, ux)

        reward, comps, _ = env._compute_phase1_reward_for_player(
            player_id="trainee",
            player_obj=player,
            ball_pos=ball.position,
            start_stamina=env._trainee_start_stamina,
            prev_ball_dist=dist,
            curr_ball_dist=dist,
            has_possession_now=False,
            gained_possession_this_step=False,
            lost_possession_this_step=False,
            ball_progress_toward_goal_m=0.0,
            ball_went_out_after_touch=False,
            illegal_action_attempted=False,
            reached_opponent_box_with_possession=False,
            opponent_reached_trainee_box=False,
            timed_out=False,
            episode_done=False,
        )

        assert comps["hdg"] < 0.0, (
            f"'hdg' (heading) component is {comps['hdg']} -- expected a nonzero "
            "penalty for moving directly away from the ball at speed. If this is "
            "0.0, heading_cos_sim/player_speed_mps are not being forwarded from "
            "_compute_phase1_reward_for_player() into phase1_reward()."
        )
        retreat_sq_coef = _CFG1.get(
            "ball_retreat_speed_penalty", _CFG1.get("ball_approach_speed_bonus", 0.0)
        )
        if retreat_sq_coef > 0.0:
            assert comps["appr_sq"] < 0.0, (
                f"'appr_sq' (approach_speed) component is {comps['appr_sq']} -- "
                "expected a nonzero retreat-speed penalty. If this is 0.0, "
                "player_speed_mps/heading_cos_sim are not being forwarded into "
                "phase1_reward()."
            )

    def test_tackle_armed_forwarded_into_tack_component(self):
        """player_obj.tackle_armed must be forwarded into phase1_reward()'s
        'tack' component -- direct regression coverage for the same class of
        bug as the heading/stamina ones above (a param computed in
        _compute_phase1_reward_for_player but never passed through)."""
        env = _make_env()
        env._reward_cfg["phase1"] = {
            **env._reward_cfg["phase1"],
            "tackle_armed_penalty_per_second": -0.2,
        }
        env.reset()
        match = env._loop.match
        player = match.player_by_id("trainee")
        player.tackle_armed = True

        _, comps, _ = env._compute_phase1_reward_for_player(
            player_id="trainee", player_obj=player, ball_pos=match.ball.position,
            start_stamina=env._trainee_start_stamina,
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            lost_possession_this_step=False, ball_progress_toward_goal_m=0.0,
            ball_went_out_after_touch=False, illegal_action_attempted=False,
            reached_opponent_box_with_possession=False,
            opponent_reached_trainee_box=False, timed_out=False, episode_done=False,
        )
        assert comps["tack"] < 0.0, (
            f"'tack' component is {comps['tack']} -- expected a nonzero penalty "
            "with player_obj.tackle_armed=True. If this is 0.0, tackle_armed is "
            "not being forwarded from _compute_phase1_reward_for_player() into "
            "phase1_reward()."
        )

    def test_tackle_attempted_this_step_forwarded_into_tatt_component(self):
        """tackle_attempted_this_step must be forwarded into phase1_reward()'s
        'tatt' component."""
        env = _make_env()
        env._reward_cfg["phase1"] = {
            **env._reward_cfg["phase1"],
            "tackle_armed_penalty_per_second": -0.2,
            "tackle_attempt_bonus_multiplier": 1.0,
        }
        env.reset()
        match = env._loop.match
        player = match.player_by_id("trainee")

        _, comps, _ = env._compute_phase1_reward_for_player(
            player_id="trainee", player_obj=player, ball_pos=match.ball.position,
            start_stamina=env._trainee_start_stamina,
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            lost_possession_this_step=False, ball_progress_toward_goal_m=0.0,
            ball_went_out_after_touch=False, illegal_action_attempted=False,
            tackle_attempted_this_step=True,
            reached_opponent_box_with_possession=False,
            opponent_reached_trainee_box=False, timed_out=False, episode_done=False,
        )
        assert comps["tatt"] > 0.0, (
            f"'tatt' component is {comps['tatt']} -- expected a nonzero bonus "
            "with tackle_attempted_this_step=True. If this is 0.0, that flag is "
            "not being forwarded from _compute_phase1_reward_for_player() into "
            "phase1_reward()."
        )

    def test_on_tackle_callback_wired_and_counts_attempts(self):
        """Match fires player.on_tackle synchronously the instant an armed
        tackle resolves contact (see Match._attempt_tackle_contact) --
        reset() must wire this to a callback that increments
        self._trainee_tackle_attempt_count, which step() later reads into
        tackle_attempted_this_step. Calling the callback directly here
        (rather than engineering a real physical contact) isolates the
        wiring itself from phase1_reward()'s own arithmetic, already covered
        by tests/ai_unit/test_reward.py."""
        env = _make_env()
        env.reset()
        match = env._loop.match
        player = match.player_by_id("trainee")

        assert callable(player.on_tackle), (
            "player.on_tackle was not wired in ScenarioEnv.reset() -- "
            "tackle_attempted_this_step can never become true"
        )
        assert env._trainee_tackle_attempt_count == 0
        player.on_tackle(player)
        assert env._trainee_tackle_attempt_count == 1
        player.on_tackle(player)
        assert env._trainee_tackle_attempt_count == 2

    def test_stamina_penalty_nonzero_on_episode_done_when_stamina_used(self):
        """'stam' must reflect actual stamina drop on episode_done=True --
        prior to the fix, stamina_used was never forwarded so 'stam' was
        always exactly 0.0 regardless of how much stamina was spent.

        This is purely a wiring test -- it doesn't matter what
        stamina_sprint_penalty's value actually is, only that phase1_reward()
        sees a nonzero one and produces a nonzero 'stam' component. Pin a
        local copy of the reward config with it forced on, rather than
        asserting on (or depending on) whatever ai_config.json currently has
        live -- that value is tuned for training, not for this test, and
        leaving it at 0.0 (as it currently is) shouldn't fail this check."""
        env = _make_env()
        env.reset()
        env._reward_cfg = copy.deepcopy(env._reward_cfg)
        env._reward_cfg["phase1"]["stamina_sprint_penalty"] = 0.5
        match = env._loop.match
        player = match.player_by_id("trainee")
        ball = match.ball

        start_stamina = env._trainee_start_stamina
        # Simulate stamina having been spent since episode start.
        player.stamina = max(0.0, start_stamina - 0.3)

        _, comps, _ = env._compute_phase1_reward_for_player(
            player_id="trainee",
            player_obj=player,
            ball_pos=ball.position,
            start_stamina=start_stamina,
            prev_ball_dist=5.0,
            curr_ball_dist=5.0,
            has_possession_now=False,
            gained_possession_this_step=False,
            lost_possession_this_step=False,
            ball_progress_toward_goal_m=0.0,
            ball_went_out_after_touch=False,
            illegal_action_attempted=False,
            reached_opponent_box_with_possession=False,
            opponent_reached_trainee_box=False,
            timed_out=False,
            episode_done=True,
        )

        assert comps["stam"] < 0.0, (
            f"'stam' (stamina_penalty) component is {comps['stam']} -- expected a "
            "nonzero penalty since episode_done=True and stamina was spent. If "
            "this is 0.0, stamina_used is not being forwarded into phase1_reward()."
        )
