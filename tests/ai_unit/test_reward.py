"""Unit tests for env/reward.py - reward shaping functions and EMAFilter.

These test the arithmetic of each reward component in isolation so that
coefficient changes in ai_config.json produce predictably scaled effects.

Also tests the EMAFilter (attack/defence smoothing) for correct latency
behaviour under normal play vs. post-goal windows.
"""
import pytest

from footballcoach.ai.env.reward import (
    EMAFilter,
    cumulative_clamped_delta,
    phase1_reward,
    phase2_reward,
    symmetric_clamp,
)

# Load the actual coefficients from config so the tests stay in sync
from footballcoach.ai.config import load_ai_config

_CFG1 = load_ai_config()["reward"]["phase1"]
_CFG2 = load_ai_config()["reward"]["phase2"]


# ---------------------------------------------------------------------------
# cumulative_clamped_delta / symmetric_clamp
# ---------------------------------------------------------------------------

class TestSymmetricClamp:

    def test_none_passthrough(self):
        assert symmetric_clamp(None) == (None, None)

    def test_returns_max_then_min_order(self):
        """Must match cumulative_clamped_delta()'s (clamp_max, clamp_min)
        param order — regression test for a bug where this returned
        (-clamp, +clamp) i.e. (min, max), which made clamp_max < clamp_min
        and silently collapsed every clamped term (e.g. 'prog') to 0."""
        clamp_max, clamp_min = symmetric_clamp(0.5)
        assert clamp_max == pytest.approx(0.5)
        assert clamp_min == pytest.approx(-0.5)
        assert clamp_max > clamp_min


class TestCumulativeClampedDelta:

    def test_no_clamp_returns_raw_delta_unchanged(self):
        payout, cum_after = cumulative_clamped_delta(0.37, 1.0)
        assert payout == pytest.approx(0.37)
        assert cum_after == pytest.approx(1.37)

    def test_within_bounds_payout_equals_raw_delta(self):
        payout, cum_after = cumulative_clamped_delta(0.1, 0.0, clamp_max=1.0, clamp_min=-1.0)
        assert payout == pytest.approx(0.1)
        assert cum_after == pytest.approx(0.1)

    def test_saturates_at_upper_bound(self):
        """Once the running total exceeds clamp_max, further positive
        deltas pay out only the remaining headroom, then 0 once saturated."""
        payout1, cum1 = cumulative_clamped_delta(0.6, 0.0, clamp_max=1.0, clamp_min=-1.0)
        assert payout1 == pytest.approx(0.6)
        # unclamped running total 0.6 -> 1.2, but clamped total can only move 0.6 -> 1.0
        payout2, cum2 = cumulative_clamped_delta(0.6, cum1, clamp_max=1.0, clamp_min=-1.0)
        assert payout2 == pytest.approx(0.4)
        assert cum2 == pytest.approx(1.2)
        # now fully saturated -- further positive deltas pay out 0
        payout3, cum3 = cumulative_clamped_delta(0.6, cum2, clamp_max=1.0, clamp_min=-1.0)
        assert payout3 == pytest.approx(0.0, abs=1e-9)
        assert cum3 == pytest.approx(1.8)

    def test_saturates_at_lower_bound(self):
        payout1, cum1 = cumulative_clamped_delta(-0.6, 0.0, clamp_max=1.0, clamp_min=-1.0)
        assert payout1 == pytest.approx(-0.6)
        payout2, cum2 = cumulative_clamped_delta(-0.6, cum1, clamp_max=1.0, clamp_min=-1.0)
        assert payout2 == pytest.approx(-0.4)
        assert cum2 == pytest.approx(-1.2)
        payout3, cum3 = cumulative_clamped_delta(-0.6, cum2, clamp_max=1.0, clamp_min=-1.0)
        assert payout3 == pytest.approx(0.0, abs=1e-9)
        assert cum3 == pytest.approx(-1.8)

    def test_reversal_after_saturation_pays_out_again(self):
        """After saturating at the upper bound, moving back the other way
        (past the bound) should resume paying out (episode total still
        bounded)."""
        _, cum1 = cumulative_clamped_delta(2.0, 0.0, clamp_max=1.0, clamp_min=-1.0)  # unclamped=2.0, clamped total saturated at +1.0
        # unclamped total moves from 2.0 -> 0.5 (below the +1.0 clamp), so the
        # clamped total moves from 1.0 -> 0.5 -> payout is the full delta
        payout2, cum2 = cumulative_clamped_delta(-1.5, cum1, clamp_max=1.0, clamp_min=-1.0)
        assert payout2 == pytest.approx(-0.5)
        assert cum2 == pytest.approx(0.5)

    def test_episode_sum_of_payouts_never_exceeds_clamp_max(self):
        """Many small positive deltas over a long 'episode' must never sum
        past clamp_max — the exact scenario this primitive exists for."""
        cum = 0.0
        total_payout = 0.0
        for _ in range(1000):
            payout, cum = cumulative_clamped_delta(0.01, cum, clamp_max=0.5, clamp_min=-0.5)
            total_payout += payout
        assert total_payout <= 0.5 + 1e-9
        assert total_payout == pytest.approx(0.5, rel=1e-3)

    def test_asymmetric_bounds_independent(self):
        """clamp_max/clamp_min can differ (e.g. appr_sq's independently
        tunable approach/retreat coefficients)."""
        cum = 0.0
        total_payout = 0.0
        for _ in range(1000):
            payout, cum = cumulative_clamped_delta(0.01, cum, clamp_max=0.2, clamp_min=-0.9)
            total_payout += payout
        assert total_payout == pytest.approx(0.2, rel=1e-3)

        cum = 0.0
        total_payout = 0.0
        for _ in range(1000):
            payout, cum = cumulative_clamped_delta(-0.01, cum, clamp_max=0.2, clamp_min=-0.9)
            total_payout += payout
        assert total_payout == pytest.approx(-0.9, rel=1e-3)

    def test_only_clamp_max_set(self):
        cum = 0.0
        total_payout = 0.0
        for _ in range(1000):
            payout, cum = cumulative_clamped_delta(-0.01, cum, clamp_max=0.2, clamp_min=None)
            total_payout += payout
        # No lower bound -> unbounded negative accumulation, all raw deltas pass through
        assert total_payout == pytest.approx(-10.0, rel=1e-3)


# ---------------------------------------------------------------------------
# phase1_reward
# ---------------------------------------------------------------------------

class TestPhase1Reward:
    """Tests for phase1_reward(), which returns a (float, dict[str, float]) tuple.

    The float is the total scalar reward; the dict breaks it down by component.
    Config now uses asymmetric ball-distance shaping: ball_approach_bonus and
    ball_retreat_penalty (retreat coefficient is larger than approach).
    """

    def _call(self, **kwargs) -> tuple[float, dict]:
        """Call phase1_reward with no-op defaults; kwargs override specific fields.

        Drops the 3rd return value (prog_cumulative_after) so existing
        callers below can keep unpacking (total, comps) — see
        cumulative_clamped_delta()/phase1_reward's docstring for why prog
        clamping is now episode-cumulative, not per-step.
        """
        defaults = dict(
            prev_ball_dist=5.0, curr_ball_dist=5.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg={**_CFG1, "step_penalty": 0.0},
        )
        defaults.update(kwargs)
        total, comps, _prog_cum_after = phase1_reward(**defaults)
        return total, comps

    def test_returns_tuple_of_float_and_dict(self):
        result = self._call()
        total, comps = result
        assert isinstance(total, float)
        assert isinstance(comps, dict)

    def test_components_dict_has_all_keys(self):
        _, comps = self._call()
        assert set(comps.keys()) == {
            "appr", "retr", "appr_sq", "hdg", "poss", "prog", "out", "ill",
            "box", "spd", "lpos", "lterm", "tout", "prox", "stam", "step",
        }

    def test_total_always_equals_sum_of_components(self):
        """Invariant: total == sum(comps.values()) for any input."""
        total, comps = self._call(
            prev_ball_dist=10.0, curr_ball_dist=5.0,
            has_possession_now=True, gained_possession_this_step=True,
            ball_progress_toward_goal_m=2.0,
        )
        assert total == pytest.approx(sum(comps.values()), rel=1e-5)

    def test_zero_when_nothing_happens(self):
        total, _ = self._call()
        assert total == pytest.approx(0.0, abs=1e-7)

    def test_closing_distance_gives_approach_reward(self):
        """appr is driven by ball-distance delta; appr_sq is driven
        SEPARATELY by the player's own speed toward the ball
        (player_speed_mps * heading_cos_sim), not by the distance delta —
        pass a matching player_speed_mps/heading_cos_sim to exercise it.
        """
        total, comps = self._call(
            prev_ball_dist=10.0, curr_ball_dist=5.0,
            player_speed_mps=5.0, heading_cos_sim=1.0,
        )
        expected_appr = _CFG1["ball_approach_bonus"] * 5.0
        expected_appr_sq = _CFG1.get("ball_approach_speed_bonus", 0.0) * (5.0 ** 2)
        expected = expected_appr + expected_appr_sq
        assert total == pytest.approx(expected, rel=1e-5)
        assert comps["appr"] == pytest.approx(expected_appr, rel=1e-5)
        assert comps["appr_sq"] == pytest.approx(expected_appr_sq, rel=1e-5)
        assert comps["retr"] == pytest.approx(0.0, abs=1e-7)

    def test_moving_away_from_ball_positional_retreat(self):
        # ball_retreat_penalty is 0.0 (positional retreat disabled; heading penalty used instead).
        total, comps = self._call(prev_ball_dist=5.0, curr_ball_dist=10.0)
        expected = _CFG1["ball_retreat_penalty"] * (-5.0)
        assert comps["retr"] == pytest.approx(expected, rel=1e-5)
        assert comps["appr"] == pytest.approx(0.0, abs=1e-7)

    def test_heading_penalty_fires_when_moving_away_fast(self):
        """Cosine heading penalty: running directly away (cos_sim=-1) at speed > threshold."""
        coef = _CFG1.get("heading_penalty_coef", 0.0)
        exp = _CFG1.get("heading_penalty_exponent", 2.0)
        if coef == 0.0:
            pytest.skip("heading_penalty_coef is 0 — penalty disabled")
        _, comps = self._call(heading_cos_sim=-1.0, player_speed_mps=5.0)
        expected = -coef * (1.0 - (-1.0)) ** exp  # max penalty
        assert comps["hdg"] == pytest.approx(expected, rel=1e-5)
        assert comps["hdg"] < 0.0

    def test_heading_penalty_zero_when_running_toward_ball(self):
        """No heading penalty when aimed directly at the ball (cos_sim=1)."""
        _, comps = self._call(heading_cos_sim=1.0, player_speed_mps=5.0)
        assert comps["hdg"] == pytest.approx(0.0, abs=1e-7)

    def test_heading_penalty_zero_when_stationary(self):
        """No heading penalty when player speed is below the threshold."""
        _, comps = self._call(heading_cos_sim=-1.0, player_speed_mps=0.1)
        assert comps["hdg"] == pytest.approx(0.0, abs=1e-7)

    def test_gaining_possession_bonus(self):
        total, comps = self._call(
            has_possession_now=True, gained_possession_this_step=True,
            prev_ball_dist=1.0, curr_ball_dist=1.0,
        )
        assert comps["poss"] == pytest.approx(_CFG1["gain_possession_bonus"], rel=1e-5)
        assert total == pytest.approx(_CFG1["gain_possession_bonus"], rel=1e-5)

    def test_ball_progress_when_possessed(self):
        progress_m = 3.0
        total, comps = self._call(
            has_possession_now=True, ball_progress_toward_goal_m=progress_m,
            prev_ball_dist=0.5, curr_ball_dist=0.5,
        )
        assert comps["prog"] == pytest.approx(_CFG1["ball_progress_scale"] * progress_m, rel=1e-5)
        assert total == pytest.approx(_CFG1["ball_progress_scale"] * progress_m, rel=1e-5)

    def test_no_ball_progress_reward_without_possession(self):
        """ball_progress is only rewarded when has_possession_now is True."""
        # ball_progress_scale may be 0.0 in the live config; force a nonzero
        # value locally so this test still exercises the with/without-possession gate.
        _cfg_prog = {**_CFG1, "ball_progress_scale": 0.1}
        r_with, _ = self._call(
            has_possession_now=True, ball_progress_toward_goal_m=5.0,
            prev_ball_dist=0.5, curr_ball_dist=0.5, cfg=_cfg_prog,
        )
        r_without, _ = self._call(
            has_possession_now=False, ball_progress_toward_goal_m=5.0,
            prev_ball_dist=0.5, curr_ball_dist=0.5, cfg=_cfg_prog,
        )
        assert r_with > r_without

    def test_ball_out_penalty(self):
        total, comps = self._call(ball_went_out_after_touch=True, prev_ball_dist=1.0, curr_ball_dist=1.0)
        assert comps["out"] == pytest.approx(_CFG1["ball_out_penalty"], rel=1e-5)
        assert total == pytest.approx(_CFG1["ball_out_penalty"], rel=1e-5)
        assert total < 0.0

    def test_illegal_action_penalty(self):
        # illegal_action_penalty is 0.0 in the live config (disabled); override
        # locally so this test still exercises the penalty formula.
        _cfg_ill = {**_CFG1, "illegal_action_penalty": -0.4, "step_penalty": 0.0}
        total, comps, _ = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=True, reached_opponent_box_with_possession=False,
            cfg=_cfg_ill,
        )
        assert comps["ill"] == pytest.approx(_cfg_ill["illegal_action_penalty"], rel=1e-5)
        assert total == pytest.approx(_cfg_ill["illegal_action_penalty"], rel=1e-5)
        assert total < 0.0

    def test_box_possession_terminal_large_bonus(self):
        total, comps = self._call(
            has_possession_now=True, reached_opponent_box_with_possession=True,
            prev_ball_dist=0.5, curr_ball_dist=0.5,
        )
        assert comps["box"] == pytest.approx(_CFG1["box_possession_terminal"], rel=1e-5)
        assert total >= _CFG1["box_possession_terminal"]

    def test_all_penalties_stack(self):
        """ball_out + illegal action together should be worse than either alone."""
        # illegal_action_penalty is 0.0 in the live config (disabled); override
        # locally so illegal-action alone is distinguishable from "both".
        _cfg_ill = {**_CFG1, "illegal_action_penalty": -0.4}

        def _call_ill(**kwargs) -> tuple[float, dict]:
            defaults = dict(
                prev_ball_dist=5.0, curr_ball_dist=5.0,
                has_possession_now=False, gained_possession_this_step=False,
                ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
                illegal_action_attempted=False, reached_opponent_box_with_possession=False,
                cfg=_cfg_ill,
            )
            defaults.update(kwargs)
            total, comps, _ = phase1_reward(**defaults)
            return total, comps

        r_both, _ = _call_ill(
            ball_went_out_after_touch=True, illegal_action_attempted=True,
            prev_ball_dist=1.0, curr_ball_dist=1.0,
        )
        r_out, _ = _call_ill(ball_went_out_after_touch=True, prev_ball_dist=1.0, curr_ball_dist=1.0)
        r_ill, _ = _call_ill(illegal_action_attempted=True, prev_ball_dist=1.0, curr_ball_dist=1.0)
        assert r_both < r_out
        assert r_both < r_ill

    def test_loss_of_possession_penalty(self):
        total, comps = self._call(lost_possession_this_step=True, prev_ball_dist=1.0, curr_ball_dist=1.0)
        assert comps["lpos"] == pytest.approx(_CFG1.get("loss_of_possession_penalty", 0.0), rel=1e-5)
        assert total < 0.0

    def test_opponent_reached_box_loss_terminal(self):
        total, comps = self._call(opponent_reached_trainee_box=True, prev_ball_dist=1.0, curr_ball_dist=1.0)
        assert comps["lterm"] == pytest.approx(_CFG1.get("loss_terminal", 0.0), rel=1e-5)
        assert total < 0.0

    def test_timeout_penalty_no_proximity(self):
        """Ball far from box on timeout: only the timeout penalty fires, prox=0."""
        total, comps = self._call(timed_out=True, ball_dist_to_opponent_box_m=9999.0,
                                  prev_ball_dist=1.0, curr_ball_dist=1.0)
        assert comps["tout"] == pytest.approx(_CFG1.get("timeout_penalty", 0.0), rel=1e-5)
        assert comps["prox"] == pytest.approx(0.0, abs=1e-7)
        assert total < 0.0

    def test_proximity_bonus_on_timeout_increases_with_closeness(self):
        """Ball near the box on timeout earns a larger prox bonus than ball far away.

        prox is normalized by start_ball_to_box_dist_m (the episode's own
        ball-to-box distance at the start), so this must be passed as a
        realistic value — leaving it at the 1.0 default would make "near"
        (dist=1.0) and "far" (dist=9999.0) both saturate to comps["prox"]=0.0.
        proximity_bonus_scale is 0.0 in the live config (disabled), so this
        test overrides it locally to a non-zero value to exercise the formula.
        """
        _cfg_prox = {**_CFG1, "proximity_bonus_scale": 0.65}
        _, comps_near, _ = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_cfg_prox, timed_out=True, ball_dist_to_opponent_box_m=1.0,
            start_ball_to_box_dist_m=30.0,
        )
        _, comps_far, _ = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_cfg_prox, timed_out=True, ball_dist_to_opponent_box_m=9999.0,
            start_ball_to_box_dist_m=30.0,
        )
        assert comps_near["prox"] > comps_far["prox"]

    def test_reward_is_additive(self):
        """Total matches sum of components and matches expected arithmetic.

        appr_sq is driven by the player's own speed toward the ball
        (player_speed_mps * heading_cos_sim), independent of the ball-gap
        delta — pass player_speed_mps=5.0, heading_cos_sim=1.0 to exercise it
        with the same 5.0 magnitude the old _delta-based test used.
        """
        _cfg_additive = {**_CFG1, "step_penalty": 0.0}
        total, comps, _ = phase1_reward(
            prev_ball_dist=10.0, curr_ball_dist=5.0,
            has_possession_now=True, gained_possession_this_step=True,
            ball_progress_toward_goal_m=2.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_cfg_additive, player_speed_mps=5.0, heading_cos_sim=1.0,
        )
        assert total == pytest.approx(sum(comps.values()), rel=1e-5)
        expected = (
            _CFG1["ball_approach_bonus"] * 5.0
            + _CFG1.get("ball_approach_speed_bonus", 0.0) * (5.0 ** 2)
            + _CFG1["gain_possession_bonus"]
            + _CFG1["ball_progress_scale"] * 2.0
        )
        assert total == pytest.approx(expected, rel=1e-5)

    def test_prog_reward_clamp_bounds_episode_total(self):
        """Regression test: prog_reward_clamp must actually bound the
        episode-cumulative 'prog' payout over many steps, and must NOT
        collapse it to 0 (see symmetric_clamp() param-order bug)."""
        cum_state: dict = {}
        total_prog_payout = 0.0
        for _ in range(500):
            _, comps, cum_state = phase1_reward(
                prev_ball_dist=1.0, curr_ball_dist=1.0,
                has_possession_now=True, gained_possession_this_step=False,
                ball_progress_toward_goal_m=1.0, ball_went_out_after_touch=False,
                illegal_action_attempted=False, reached_opponent_box_with_possession=False,
                cfg={**_CFG1, "ball_progress_scale": 1.0},
                start_ball_to_box_dist_m=1.0,
                prog_reward_clamp=0.5,
                cumulative_state=cum_state,
            )
            total_prog_payout += comps["prog"]
        assert total_prog_payout > 0.0, "prog reward clamp must not zero out the reward entirely"
        assert total_prog_payout <= 0.5 + 1e-6
        assert total_prog_payout == pytest.approx(0.5, rel=1e-3)

    def test_appr_sq_clamps_bound_episode_total_independently(self):
        """appr_sq_approach_reward_clamp/appr_sq_retreat_reward_clamp bound
        the positive/negative cumulative appr_sq totals independently."""
        cum_state: dict = {}
        total_approach_payout = 0.0
        for _ in range(500):
            _, comps, cum_state = phase1_reward(
                prev_ball_dist=1.0, curr_ball_dist=1.0,
                has_possession_now=False, gained_possession_this_step=False,
                ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
                illegal_action_attempted=False, reached_opponent_box_with_possession=False,
                cfg={**_CFG1, "ball_approach_speed_bonus": 1.0},
                start_ball_dist_m=1.0, player_speed_mps=1.0, heading_cos_sim=1.0,
                appr_sq_approach_reward_clamp=0.3,
                cumulative_state=cum_state,
            )
            total_approach_payout += comps["appr_sq"]
        assert total_approach_payout > 0.0
        assert total_approach_payout <= 0.3 + 1e-6
        assert total_approach_payout == pytest.approx(0.3, rel=1e-3)

        cum_state = {}
        total_retreat_payout = 0.0
        for _ in range(500):
            _, comps, cum_state = phase1_reward(
                prev_ball_dist=1.0, curr_ball_dist=1.0,
                has_possession_now=False, gained_possession_this_step=False,
                ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
                illegal_action_attempted=False, reached_opponent_box_with_possession=False,
                cfg={**_CFG1, "ball_retreat_speed_penalty": 1.0},
                start_ball_dist_m=1.0, player_speed_mps=1.0, heading_cos_sim=-1.0,
                appr_sq_retreat_reward_clamp=0.3,
                cumulative_state=cum_state,
            )
            total_retreat_payout += comps["appr_sq"]
        assert total_retreat_payout < 0.0
        assert total_retreat_payout >= -0.3 - 1e-6
        assert total_retreat_payout == pytest.approx(-0.3, rel=1e-3)

    def test_step_penalty_fires_every_step(self):
        """step_penalty is subtracted on every call including no-op steps."""
        _cfg_step = {**_CFG1, "step_penalty": 0.05}
        total, comps, _ = phase1_reward(
            prev_ball_dist=5.0, curr_ball_dist=5.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_cfg_step,
        )
        assert comps["step"] == pytest.approx(-0.05, rel=1e-5)
        assert total == pytest.approx(-0.05, rel=1e-5)

    def test_step_penalty_zero_when_disabled(self):
        _cfg_no_step = {**_CFG1, "step_penalty": 0.0}
        _, comps, _ = phase1_reward(
            prev_ball_dist=5.0, curr_ball_dist=5.0,
            has_possession_now=False, gained_possession_this_step=False,
            ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_cfg_no_step,
        )
        assert comps["step"] == pytest.approx(0.0, abs=1e-9)

    def test_step_penalty_accumulates_over_episode(self):
        """Over N steps the sum of step payouts equals -step_penalty * N."""
        _cfg_step = {**_CFG1, "step_penalty": 0.01}
        total_step = 0.0
        cum_state: dict = {}
        for _ in range(50):
            _, comps, cum_state = phase1_reward(
                prev_ball_dist=5.0, curr_ball_dist=5.0,
                has_possession_now=False, gained_possession_this_step=False,
                ball_progress_toward_goal_m=0.0, ball_went_out_after_touch=False,
                illegal_action_attempted=False, reached_opponent_box_with_possession=False,
                cfg=_cfg_step, cumulative_state=cum_state,
            )
            total_step += comps["step"]
        assert total_step == pytest.approx(-0.5, rel=1e-5)

    def test_cumulative_state_missing_keys_default_to_zero(self):
        """Passing an empty/partial cumulative_state dict must not crash and
        must treat missing terms as starting from 0.0."""
        total, comps, cum_after = phase1_reward(
            prev_ball_dist=1.0, curr_ball_dist=1.0,
            has_possession_now=True, gained_possession_this_step=False,
            ball_progress_toward_goal_m=1.0, ball_went_out_after_touch=False,
            illegal_action_attempted=False, reached_opponent_box_with_possession=False,
            cfg=_CFG1, cumulative_state={},
        )
        assert isinstance(cum_after, dict)
        assert "prog" in cum_after
        assert "appr_sq" in cum_after


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
