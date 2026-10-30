"""Unit tests for env/reward.py - reward shaping functions and EMAFilter.

These test the arithmetic of each reward component in isolation so that
coefficient changes in ai_config.json produce predictably scaled effects.

Also tests the EMAFilter (attack/defence smoothing) for correct latency
behaviour under normal play vs. post-goal windows.
"""
import pytest

from footballcoach.ai.env.reward import (
    EMAFilter,
    phase1_reward,
    phase1_terminal_reward_only,
    phase2_reward,
)

# Load the actual coefficients from config so the tests stay in sync
from footballcoach.ai.config import load_ai_config

_CFG1 = load_ai_config()["reward"]["phase1"]
_CFG2 = load_ai_config()["reward"]["phase2"]


# ---------------------------------------------------------------------------
# phase1_reward
# ---------------------------------------------------------------------------

class TestPhase1Reward:
    """Tests for phase1_reward(), which returns a (float, dict[str, float]) tuple.

    The float is the total scalar reward; the dict breaks it down by
    component. 6 terms survived the 2026-10-14 cleanup (out, box, spd,
    lterm, tout, sprint) -- a much larger set of debugging-era terms was
    removed because every one of them was either coefficient 0.0 in the
    live config or a multiplier/clamp gating a term that was itself 0.0.
    "appr"/"prog" were re-introduced the same day in a cleaned-up,
    single-coefficient, provably-telescoping form (still 0.0 in the live
    config) -- see TestApprAndProgShaping below.
    """

    def _call(self, **kwargs) -> tuple[float, dict]:
        defaults = dict(
            ball_went_out_after_touch=False,
            reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        defaults.update(kwargs)
        return phase1_reward(**defaults)

    def test_returns_tuple_of_float_and_dict(self):
        total, comps = self._call()
        assert isinstance(total, float)
        assert isinstance(comps, dict)

    def test_components_dict_has_all_keys(self):
        _, comps = self._call()
        assert set(comps.keys()) == {"out", "box", "spd", "lterm", "tout", "sprint", "appr", "prog"}

    def test_every_component_key_has_a_display_label(self):
        """Regression guard: PPOTrainer.REWARD_COMP_LABELS (used by both the
        per-episode 'component mean/std/min/max' table and the per-step
        'rew/step' table in _log_rollout_summary) is a SEPARATE, hand-
        maintained list from phase1_reward's own comps dict -- adding a new
        component here without also adding it to REWARD_COMP_LABELS means
        it silently never appears in either diagnostic table, with no
        error anywhere."""
        from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS
        _, comps = self._call()
        labeled_keys = {k for k, _ in REWARD_COMP_LABELS}
        missing = set(comps.keys()) - labeled_keys
        assert not missing, (
            f"phase1_reward returns component(s) {missing} with no entry in "
            "REWARD_COMP_LABELS -- they will never show up in the rollout "
            "summary tables. Add them to REWARD_COMP_LABELS in ppo_trainer.py."
        )

    def test_total_always_equals_sum_of_components(self):
        total, comps = self._call(
            reached_opponent_box_with_possession=True, time_fraction_remaining=0.5,
        )
        assert total == pytest.approx(sum(comps.values()), rel=1e-5)

    def test_zero_when_nothing_happens(self):
        total, _ = self._call()
        assert total == pytest.approx(0.0, abs=1e-7)

    def test_ball_out_penalty(self):
        total, comps = self._call(ball_went_out_after_touch=True)
        assert comps["out"] == pytest.approx(_CFG1["ball_out_penalty"], rel=1e-5)
        assert total == pytest.approx(_CFG1["ball_out_penalty"], rel=1e-5)
        assert total < 0.0

    def test_box_possession_terminal_large_bonus(self):
        total, comps = self._call(
            reached_opponent_box_with_possession=True, time_fraction_remaining=0.0,
        )
        assert comps["box"] == pytest.approx(_CFG1["box_possession_terminal"], rel=1e-5)
        assert total >= _CFG1["box_possession_terminal"]

    def test_speed_bonus_scales_with_time_remaining(self):
        _cfg = {**_CFG1, "speed_bonus_scale": 0.2}
        total_fast, comps_fast = self._call(
            reached_opponent_box_with_possession=True, time_fraction_remaining=1.0, cfg=_cfg,
        )
        total_slow, comps_slow = self._call(
            reached_opponent_box_with_possession=True, time_fraction_remaining=0.0, cfg=_cfg,
        )
        assert comps_fast["spd"] == pytest.approx(0.2)
        assert comps_slow["spd"] == pytest.approx(0.0)
        assert total_fast > total_slow

    def test_no_speed_bonus_without_box_terminal(self):
        _cfg = {**_CFG1, "speed_bonus_scale": 0.2}
        _, comps = self._call(
            reached_opponent_box_with_possession=False, time_fraction_remaining=1.0, cfg=_cfg,
        )
        assert comps["spd"] == pytest.approx(0.0)

    def test_opponent_reached_box_loss_terminal(self):
        total, comps = self._call(opponent_reached_trainee_box=True)
        assert comps["lterm"] == pytest.approx(_CFG1.get("loss_terminal", 0.0), rel=1e-5)
        assert total < 0.0

    def test_timeout_penalty(self):
        total, comps = self._call(timed_out=True)
        assert comps["tout"] == pytest.approx(_CFG1.get("timeout_penalty", 0.0), rel=1e-5)
        assert total < 0.0

    def test_no_timeout_penalty_when_not_timed_out(self):
        _, comps = self._call(timed_out=False)
        assert comps["tout"] == pytest.approx(0.0, abs=1e-7)

    def test_all_penalties_stack(self):
        """ball_out + loss terminal together should be worse than either alone."""
        r_both, _ = self._call(ball_went_out_after_touch=True, opponent_reached_trainee_box=True)
        r_out, _ = self._call(ball_went_out_after_touch=True)
        r_lterm, _ = self._call(opponent_reached_trainee_box=True)
        assert r_both < r_out
        assert r_both < r_lterm

    def test_sprint_penalty_fires_only_when_sprinting_and_coef_nonzero(self):
        _cfg = {**_CFG1, "sprint_penalty_per_s": 0.1}
        _, comps_sprinting = self._call(is_sprinting=True, decision_interval_s=0.5, cfg=_cfg)
        _, comps_not = self._call(is_sprinting=False, decision_interval_s=0.5, cfg=_cfg)
        assert comps_sprinting["sprint"] == pytest.approx(-0.1 * 0.5)
        assert comps_not["sprint"] == pytest.approx(0.0, abs=1e-7)

    def test_sprint_penalty_zero_when_coef_disabled(self):
        _cfg = {**_CFG1, "sprint_penalty_per_s": 0.0}
        _, comps = self._call(is_sprinting=True, decision_interval_s=1.0, cfg=_cfg)
        assert comps["sprint"] == pytest.approx(0.0, abs=1e-7)

    def test_sprint_penalty_scales_with_decision_interval(self):
        _cfg = {**_CFG1, "sprint_penalty_per_s": 0.2}
        _, comps = self._call(is_sprinting=True, decision_interval_s=0.3, cfg=_cfg)
        assert comps["sprint"] == pytest.approx(-0.2 * 0.3)


class TestApprAndProgShaping:
    """'appr' (ball-approach) and 'prog' (ball-progress) -- re-introduced
    2026-10-14 as pure, single-coefficient potential-based shaping
    (Phi(s) = -coef * distance(s), F = Phi(s) - Phi(s') each step, gamma=1),
    re-gated 2026-10-15 onto ball.last_touched_by_player_id (see reward.py's
    'last toucher' gating note) instead of live possession. Live config
    (_CFG1): both 0.005 -- every test here still overrides the coefficient
    locally so the arithmetic checks stay independent of whatever the
    live-tuned value happens to be.

    phase1_reward() itself takes the ALREADY per-tick-accumulated,
    ALREADY last-toucher-gated delta as a single scalar
    (player_ball_approach_delta_m / ball_progress_toward_goal_m) -- the
    gating itself is ScenarioEnv's responsibility (see its accumulator-init
    comment in step()), not tested here. What IS tested here: given a
    sequence of such already-gated per-step deltas, the formula (a) always
    telescopes to exactly coef*(sum of deltas) regardless of path, and (b)
    can never be farmed by shuttling back and forth, since any closed round
    trip must then net EXACTLY zero. TestApprRoundTripNetsZero /
    TestProgRoundTripNetsZero assert (b) directly rather than trusting the
    arithmetic by eye.
    """

    def _call(self, **kwargs) -> tuple[float, dict]:
        defaults = dict(
            ball_went_out_after_touch=False,
            reached_opponent_box_with_possession=False,
            cfg=_CFG1,
        )
        defaults.update(kwargs)
        return phase1_reward(**defaults)

    # -- appr: basic arithmetic / disabled-by-default --------------------

    def test_appr_zero_when_coef_disabled(self):
        _cfg = {**_CFG1, "ball_approach_bonus": 0.0}
        _, comps = self._call(player_ball_approach_delta_m=5.0, cfg=_cfg)
        assert comps["appr"] == pytest.approx(0.0, abs=1e-7)

    def test_appr_positive_when_closing_distance(self):
        _cfg = {**_CFG1, "ball_approach_bonus": 0.1}
        _, comps = self._call(player_ball_approach_delta_m=3.0, cfg=_cfg)
        assert comps["appr"] == pytest.approx(0.1 * 3.0)

    def test_appr_negative_when_opening_distance(self):
        """Opening the gap by the SAME amount as closing it must produce the
        exact negative of the closing case -- i.e. one coefficient, not an
        independently-tunable 'retreat' rate (that asymmetry is exactly what
        broke telescoping in the pre-removal version -- see reward.py's
        2026-10-14 re-introduction note)."""
        _cfg = {**_CFG1, "ball_approach_bonus": 0.1}
        _, comps_close = self._call(player_ball_approach_delta_m=3.0, cfg=_cfg)
        _, comps_open = self._call(player_ball_approach_delta_m=-3.0, cfg=_cfg)
        assert comps_open["appr"] == pytest.approx(-comps_close["appr"])

    def test_appr_zero_when_distance_unchanged(self):
        _cfg = {**_CFG1, "ball_approach_bonus": 0.1}
        _, comps = self._call(player_ball_approach_delta_m=0.0, cfg=_cfg)
        assert comps["appr"] == pytest.approx(0.0, abs=1e-7)

    def test_appr_has_no_separate_retreat_coefficient(self):
        """ball_retreat_penalty must not exist as a distinct knob any more --
        setting it has no effect on 'appr' (only ball_approach_bonus does)."""
        _cfg = {**_CFG1, "ball_approach_bonus": 0.1, "ball_retreat_penalty": 999.0}
        _, comps = self._call(player_ball_approach_delta_m=-3.0, cfg=_cfg)
        assert comps["appr"] == pytest.approx(0.1 * -3.0)

    # -- appr: telescoping / non-exploitability ---------------------------

    def test_appr_telescopes_over_a_monotonic_trajectory(self):
        """Sum of per-step 'appr' over N steps of CONTINUOUS one-directional
        closing must equal coef*(dist_start - dist_end) exactly, regardless
        of how many steps or how uneven the per-step deltas are."""
        _cfg = {**_CFG1, "ball_approach_bonus": 0.37}
        dists = [20.0, 17.0, 17.0, 12.5, 9.0, 9.0, 9.0, 3.0]  # uneven steps, some flat
        total_appr = 0.0
        for prev_d, curr_d in zip(dists[:-1], dists[1:]):
            _, comps = self._call(player_ball_approach_delta_m=(prev_d - curr_d), cfg=_cfg)
            total_appr += comps["appr"]
        assert total_appr == pytest.approx(0.37 * (dists[0] - dists[-1]), rel=1e-9)

    def test_appr_round_trip_nets_exactly_zero(self):
        """The core non-exploitability guarantee: closing distance and then
        opening it back up to the SAME starting distance (any path, any
        number of steps) must net to exactly 0 -- this is what makes 'appr'
        impossible to farm by shuttling back and forth near the ball."""
        _cfg = {**_CFG1, "ball_approach_bonus": 1.25}
        # A wandering round trip: 20 -> 12 -> 18 -> 4 -> 20 (ends where it started)
        dists = [20.0, 12.0, 18.0, 4.0, 20.0]
        total_appr = 0.0
        for prev_d, curr_d in zip(dists[:-1], dists[1:]):
            _, comps = self._call(player_ball_approach_delta_m=(prev_d - curr_d), cfg=_cfg)
            total_appr += comps["appr"]
        assert total_appr == pytest.approx(0.0, abs=1e-9)

    def test_appr_asymmetric_coefficients_WOULD_break_round_trip(self):
        """Negative control: prove the round-trip-nets-zero property is NOT
        trivially true for any shaping scheme -- an asymmetric approach/
        retreat split (the pre-removal design) does NOT net zero on the same
        round trip, which is exactly why that version was abuseable and why
        this re-introduction deliberately uses one coefficient."""
        approach_coef, retreat_coef = 0.1, 0.9  # retreat penalized harder (old design)
        total = 0.0
        for prev_d, curr_d in [(20.0, 12.0), (12.0, 18.0), (18.0, 4.0), (4.0, 20.0)]:
            delta = prev_d - curr_d
            total += approach_coef * delta if delta >= 0 else retreat_coef * delta
        assert total != pytest.approx(0.0, abs=1e-9)

    # -- prog: basic arithmetic / disabled-by-default --------------------

    def test_prog_zero_when_coef_disabled(self):
        _cfg = {**_CFG1, "ball_progress_scale": 0.0}
        _, comps = self._call(ball_progress_toward_goal_m=5.0, cfg=_cfg)
        assert comps["prog"] == pytest.approx(0.0, abs=1e-7)

    def test_prog_linear_in_progress(self):
        _cfg = {**_CFG1, "ball_progress_scale": 0.2}
        _, comps = self._call(ball_progress_toward_goal_m=3.0, cfg=_cfg)
        assert comps["prog"] == pytest.approx(0.2 * 3.0)

    def test_prog_negative_when_pushed_backward(self):
        _cfg = {**_CFG1, "ball_progress_scale": 0.2}
        _, comps = self._call(ball_progress_toward_goal_m=-3.0, cfg=_cfg)
        assert comps["prog"] == pytest.approx(0.2 * -3.0)

    def test_prog_not_double_gated_on_possession(self):
        """2026-10-14 fix: the caller (ScenarioEnv) already gates
        ball_progress_toward_goal_m per-TICK on possession before accumulating
        it for the decision interval (see reward.py's 'prog' docstring) --
        phase1_reward itself must NOT apply a second, interval-level
        possession gate on top of that, since the old version's extra gate
        silently zeroed out real progress whenever the interval ended with a
        mid-interval loss of possession, breaking telescoping. There is no
        has_possession_now parameter any more; passing a nonzero
        ball_progress_toward_goal_m must always count."""
        _cfg = {**_CFG1, "ball_progress_scale": 0.5}
        _, comps = self._call(ball_progress_toward_goal_m=2.0, cfg=_cfg)
        assert comps["prog"] == pytest.approx(1.0)

    # -- prog: telescoping / non-exploitability ---------------------------

    def test_prog_telescopes_over_a_possession_spell(self):
        """Sum of per-decision-step 'prog' over a possession spell must equal
        coef*(box_dist_at_gain - box_dist_at_loss), matching how
        ScenarioEnv's trainee_prog_accum/sec_prog_accum already accumulate
        PRE-tick-possession-gated box-distance deltas internally -- this
        test exercises phase1_reward's own side of that contract with the
        already-accumulated per-decision-step deltas a caller would pass."""
        _cfg = {**_CFG1, "ball_progress_scale": 0.6}
        box_dists = [30.0, 25.0, 25.0, 18.0, 10.0]  # closing on the box while possessing
        total_prog = 0.0
        for prev_d, curr_d in zip(box_dists[:-1], box_dists[1:]):
            _, comps = self._call(ball_progress_toward_goal_m=(prev_d - curr_d), cfg=_cfg)
            total_prog += comps["prog"]
        assert total_prog == pytest.approx(0.6 * (box_dists[0] - box_dists[-1]), rel=1e-9)

    def test_prog_round_trip_nets_exactly_zero(self):
        """Dribbling the ball away from the box and back to the SAME
        box-distance (without ever losing possession) must net exactly 0."""
        _cfg = {**_CFG1, "ball_progress_scale": 2.0}
        box_dists = [30.0, 22.0, 28.0, 12.0, 30.0]  # ends where it started
        total_prog = 0.0
        for prev_d, curr_d in zip(box_dists[:-1], box_dists[1:]):
            _, comps = self._call(ball_progress_toward_goal_m=(prev_d - curr_d), cfg=_cfg)
            total_prog += comps["prog"]
        assert total_prog == pytest.approx(0.0, abs=1e-9)

    # -- appr/prog: additive with the rest of the terms -------------------

    def test_appr_and_prog_included_in_total(self):
        _cfg = {**_CFG1, "ball_approach_bonus": 0.3, "ball_progress_scale": 0.4}
        total, comps = self._call(
            player_ball_approach_delta_m=4.0, ball_progress_toward_goal_m=2.0, cfg=_cfg,
        )
        assert total == pytest.approx(sum(comps.values()), rel=1e-9)
        assert total == pytest.approx(0.3 * 4.0 + 0.4 * 2.0, rel=1e-9)


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


class TestPhase1TerminalRewardOnly:
    """phase1_terminal_reward_only() -- the UI's live-scenario terminal-only
    reward approximation (see ScenarioLoop._compute_terminal_rewards). Must
    agree EXACTLY with phase1_reward() itself for the terminal components
    (it's a thin wrapper, not a reimplementation), and must always zero out
    the "sprint" shaping term regardless of what the terminal flags are."""

    _NEUTRAL = dict(
        reached_opponent_box_with_possession=False,
        opponent_reached_trainee_box=False,
        timed_out=False,
        ball_went_out_after_touch=False,
        cfg=_CFG1,
    )

    def test_agrees_with_phase1_reward_for_box_possession(self):
        total_only, comps_only = phase1_terminal_reward_only(
            **{**self._NEUTRAL, "reached_opponent_box_with_possession": True},
            time_fraction_remaining=0.4,
        )
        total_full, comps_full = phase1_reward(
            ball_went_out_after_touch=False,
            reached_opponent_box_with_possession=True, cfg=_CFG1,
            time_fraction_remaining=0.4,
        )
        assert total_only == pytest.approx(total_full)
        assert comps_only["box"] == pytest.approx(comps_full["box"])
        assert comps_only["spd"] == pytest.approx(comps_full["spd"])

    def test_no_terminal_flags_leaves_everything_zero(self):
        total, comps = phase1_terminal_reward_only(**self._NEUTRAL)
        for key in ("out", "box", "spd", "lterm", "tout", "sprint", "appr", "prog"):
            assert comps[key] == pytest.approx(0.0), key
        assert total == pytest.approx(0.0)

    def test_loss_terminal_fires_on_opponent_reached_trainee_box(self):
        cfg = {**_CFG1, "loss_terminal": -1.0}
        _total, comps = phase1_terminal_reward_only(
            **{**self._NEUTRAL, "cfg": cfg, "opponent_reached_trainee_box": True},
        )
        assert comps["lterm"] == pytest.approx(-1.0)
        assert comps["box"] == pytest.approx(0.0)

    def test_timeout_penalty(self):
        cfg = {**_CFG1, "timeout_penalty": -0.5}
        _total, comps = phase1_terminal_reward_only(
            **{**self._NEUTRAL, "cfg": cfg, "timed_out": True},
        )
        assert comps["tout"] == pytest.approx(-0.5)

    def test_ball_out_fires_independently_of_other_flags(self):
        cfg = {**_CFG1, "ball_out_penalty": -0.2}
        _total, comps = phase1_terminal_reward_only(
            **{**self._NEUTRAL, "cfg": cfg, "ball_went_out_after_touch": True},
        )
        assert comps["out"] == pytest.approx(-0.2)
