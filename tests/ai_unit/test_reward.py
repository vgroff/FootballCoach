"""Unit tests for env/reward.py - reward shaping functions and EMAFilter.

These test the arithmetic of each reward component in isolation so that
coefficient changes in ai_config.json produce predictably scaled effects.

Also tests the EMAFilter (attack/defence smoothing) for correct latency
behaviour under normal play vs. post-goal windows.
"""
import pytest

from footballcoach.ai.env.reward import EMAFilter, phase1_reward, phase2_reward

# Load the actual coefficients from config so the tests stay in sync
from footballcoach.ai.config import load_ai_config

_CFG1 = load_ai_config()["reward"]["phase1"]
_CFG2 = load_ai_config()["reward"]["phase2"]


# ---------------------------------------------------------------------------
# phase1_reward
# ---------------------------------------------------------------------------

class TestPhase1Reward:

    def test_zero_when_nothing_happens(self):
        r = phase1_reward(
            prev_ball_dist=5.0, curr_ball_dist=5.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r == pytest.approx(0.0, abs=1e-7)

    def test_closing_distance_positive(self):
        r = phase1_reward(
            prev_ball_dist=10.0, curr_ball_dist=5.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        expected = _CFG1["ball_distance_shaping"] * (10.0 - 5.0)
        assert r == pytest.approx(expected, rel=1e-5)

    def test_moving_away_from_ball_negative(self):
        r = phase1_reward(
            prev_ball_dist=5.0, curr_ball_dist=10.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r < 0.0

    def test_gaining_possession_bonus(self):
        r = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=True, gained_possession_this_step=True,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r >= _CFG1["gain_possession_bonus"]

    def test_ball_progress_when_possessed(self):
        progress_m = 3.0
        r = phase1_reward(
            prev_ball_dist=0.5, curr_ball_dist=0.5,
            has_possession_now=True, gained_possession_this_step=False,
            ball_progress_toward_goal_m=progress_m, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r == pytest.approx(_CFG1["ball_progress_scale"] * progress_m, rel=1e-5)

    def test_no_ball_progress_reward_without_possession(self):
        """ball_progress is only rewarded if has_possession_now."""
        r_with = phase1_reward(
            prev_ball_dist=0.5, curr_ball_dist=0.5,
            has_possession_now=True, gained_possession_this_step=False,
            ball_progress_toward_goal_m=5.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        r_without = phase1_reward(
            prev_ball_dist=0.5, curr_ball_dist=0.5,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=5.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r_with > r_without

    def test_ball_out_penalty(self):
        r = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=True,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r <= _CFG1["ball_out_penalty"]

    def test_illegal_action_penalty(self):
        r = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=True, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r <= _CFG1["illegal_action_penalty"]

    def test_box_possession_terminal_large_bonus(self):
        r = phase1_reward(
            prev_ball_dist=0.5, curr_ball_dist=0.5,
            has_possession_now=True, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=True,
            cfg=_CFG1,
        )
        assert r >= _CFG1["box_possession_terminal"]

    def test_all_penalties_stack(self):
        """Multiple bad things happening should stack (all negative)."""
        r = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=True,
            illegal_action_attempted=True, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        assert r < _CFG1["ball_out_penalty"]  # worse than just ball_out alone

    def test_reward_is_additive(self):
        """Total reward should equal sum of individual components."""
        r_total = phase1_reward(
            prev_ball_dist=10.0, curr_ball_dist=5.0,
            has_possession_now=True, gained_possession_this_step=True,
            ball_progress_toward_goal_m=2.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        expected = (
            _CFG1["ball_distance_shaping"] * 5.0
            + _CFG1["gain_possession_bonus"]
            + _CFG1["ball_progress_scale"] * 2.0
        )
        assert r_total == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# phase2_reward
# ---------------------------------------------------------------------------

class TestPhase2Reward:

    def test_zero_when_nothing_happens(self):
        r = phase2_reward(
            shot_taken_this_step=False, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        assert r == pytest.approx(0.0, abs=1e-7)

    def test_shot_taken_gives_positive_reward(self):
        r = phase2_reward(
            shot_taken_this_step=True, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        assert r > 0.0

    def test_faster_shot_rewarded_more(self):
        r_fast = phase2_reward(
            shot_taken_this_step=True, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        r_slow = phase2_reward(
            shot_taken_this_step=True, ticks_since_episode_start=500,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        assert r_fast > r_slow

    def test_shot_on_target_extra_bonus(self):
        r_off = phase2_reward(
            shot_taken_this_step=True, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        r_on = phase2_reward(
            shot_taken_this_step=True, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=True,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        assert r_on > r_off
        assert (r_on - r_off) == pytest.approx(_CFG2["shot_on_target_bonus"], rel=1e-5)

    def test_goal_gives_max_reward(self):
        r = phase2_reward(
            shot_taken_this_step=True, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=True,
            goal_scored=True, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        assert r >= _CFG2["goal_terminal"]

    def test_no_shot_no_shot_bonus(self):
        """Without shot_taken=True, shot-related bonuses must not appear."""
        r_goal_no_shot = phase2_reward(
            shot_taken_this_step=False, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=True,
            goal_scored=True, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        assert r_goal_no_shot == pytest.approx(0.0, abs=1e-7)

    def test_illegal_action_penalty(self):
        r = phase2_reward(
            shot_taken_this_step=False, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=True,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        assert r == pytest.approx(_CFG2["illegal_action_penalty"], rel=1e-5)

    def test_possession_lost_penalty(self):
        r = phase2_reward(
            shot_taken_this_step=False, ticks_since_episode_start=0,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=True, cfg=_CFG2,
        )
        assert r == pytest.approx(_CFG2["possession_lost_to_keeper_penalty"], rel=1e-5)

    def test_very_late_shot_minimal_speed_bonus(self):
        """Shot at the last tick: time bonus should be ≈ 0 (capped at 0)."""
        r = phase2_reward(
            shot_taken_this_step=True, ticks_since_episode_start=999,
            max_episode_ticks=1000, shot_on_target=False,
            goal_scored=False, illegal_action_attempted=False,
            possession_lost_to_keeper=False, cfg=_CFG2,
        )
        # Only shot_on_target bonus is absent; time bonus ≈ 0
        # But should still be >= 0 (max(0, ...) in formula)
        assert r >= 0.0


# ---------------------------------------------------------------------------
# EMAFilter (attack/defence smoothing)
# ---------------------------------------------------------------------------

class TestEMAFilter:

    def test_initial_value_is_half(self):
        ema = EMAFilter()
        assert ema.smoothed == pytest.approx(0.5, abs=1e-6)

    def test_slow_alpha_changes_little_per_step(self):
        ema = EMAFilter(alpha_normal=0.99, alpha_post_goal=0.5, post_goal_window_s=10.0)
        prev = ema.smoothed
        ema.update(raw_value=1.0, dt_s=0.5)
        # alpha=0.99 -> smoothed = 0.99*0.5 + 0.01*1.0 = 0.505
        assert abs(ema.smoothed - prev) < 0.02

    def test_fast_alpha_changes_more(self):
        ema_slow = EMAFilter(alpha_normal=0.99, alpha_post_goal=0.5, post_goal_window_s=10.0)
        ema_fast = EMAFilter(alpha_normal=0.99, alpha_post_goal=0.5, post_goal_window_s=10.0)

        ema_fast.on_goal()

        ema_slow.update(1.0, dt_s=0.5)
        ema_fast.update(1.0, dt_s=0.5)

        # Fast EMA (post-goal window) should move more toward 1.0
        assert ema_fast.smoothed > ema_slow.smoothed

    def test_on_goal_enables_fast_window(self):
        ema = EMAFilter(alpha_normal=0.995, alpha_post_goal=0.5, post_goal_window_s=5.0)
        ema.on_goal()
        before = ema.smoothed
        ema.update(1.0, dt_s=0.5)
        change_post_goal = abs(ema.smoothed - before)

        ema2 = EMAFilter(alpha_normal=0.995, alpha_post_goal=0.5, post_goal_window_s=5.0)
        before2 = ema2.smoothed
        ema2.update(1.0, dt_s=0.5)
        change_normal = abs(ema2.smoothed - before2)

        assert change_post_goal > change_normal * 5

    def test_post_goal_window_expires(self):
        """After post_goal_window_s has elapsed, should revert to slow alpha."""
        ema = EMAFilter(alpha_normal=0.995, alpha_post_goal=0.5, post_goal_window_s=2.0)
        ema.on_goal()
        # Advance dt past the window
        ema.update(1.0, dt_s=3.0)  # 3s > 2s window

        # Now updates should use slow alpha again
        ema_before = ema.smoothed
        ema.update(0.0, dt_s=0.5)
        change = abs(ema.smoothed - ema_before)
        # slow alpha=0.995: change = (1 - 0.995) * |raw - smoothed| which is tiny
        assert change < 0.01 * abs(1.0 - ema_before) + 0.01

    def test_reset_returns_to_initial(self):
        ema = EMAFilter()
        ema.on_goal()
        for _ in range(10):
            ema.update(0.9, dt_s=0.5)
        ema.reset()
        assert ema.smoothed == pytest.approx(0.5, abs=1e-6)

    def test_reset_cancels_post_goal_window(self):
        ema = EMAFilter(alpha_normal=0.995, alpha_post_goal=0.5, post_goal_window_s=10.0)
        ema.on_goal()
        ema.reset()
        before = ema.smoothed
        ema.update(1.0, dt_s=0.5)
        change = abs(ema.smoothed - before)
        # Should use slow alpha (0.005 * |1.0 - 0.5| = 0.0025)
        assert change < 0.01

    def test_smoothed_converges_to_raw_given_many_updates(self):
        """With fast alpha (0.5) and many updates toward 1.0, smoothed -> 1.0."""
        ema = EMAFilter(alpha_normal=0.5, alpha_post_goal=0.5, post_goal_window_s=0.0)
        for _ in range(30):
            ema.update(1.0, dt_s=0.1)
        assert ema.smoothed > 0.99

    def test_from_config_constructs(self):
        ema = EMAFilter.from_config()
        assert 0.0 <= ema.smoothed <= 1.0
        assert ema.alpha_normal > 0.9  # should be slow (close to 1)
        assert ema.alpha_post_goal < ema.alpha_normal  # faster after goal
