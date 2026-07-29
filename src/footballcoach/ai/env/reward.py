"""Per-head reward shaping functions for each curriculum phase.

See ai_design_doc.md sections 10.1 and 10.2 for the concrete starting
formulas.  All coefficients are loaded from ai_config.json so they can
be tuned without touching code.

These functions are called by the env wrappers at the end of each
decision-interval step (after ~15 engine ticks) to compute the scalar
reward for that step.

Convention: reward is always a Python float, accumulated over the engine
ticks within one decision interval and returned as one scalar per step.
"""
from __future__ import annotations


def phase1_reward(
    prev_ball_dist: float,
    curr_ball_dist: float,
    has_possession_now: bool,
    gained_possession_this_step: bool,
    ball_progress_toward_goal_m: float,
    ball_went_out_after_touch: bool,
    illegal_action_attempted: bool,
    reached_opponent_box_with_possession: bool,
    cfg: dict,
) -> float:
    """GetPossession/Move experiment reward (curriculum phase 1).

    See ai_design_doc.md section 10.1.

    Args:
        prev_ball_dist: Distance to ball at start of this decision step.
        curr_ball_dist: Distance to ball at end of this decision step.
        has_possession_now: Player currently has ball.
        gained_possession_this_step: Player won the ball during this step.
        ball_progress_toward_goal_m: Metres the ball advanced toward the
            opponent's goal during this step (negative = moved away).
        ball_went_out_after_touch: Ball went out of bounds after the
            trainee player touched it.
        illegal_action_attempted: reward function received an illegal-action
            flag from to_orders.py.
        reached_opponent_box_with_possession: Terminal bonus condition.
        cfg: The 'reward.phase1' section of ai_config.json.
    """
    r = 0.0
    r += cfg["ball_distance_shaping"] * (prev_ball_dist - curr_ball_dist)
    if gained_possession_this_step:
        r += cfg["gain_possession_bonus"]
    if has_possession_now:
        r += cfg["ball_progress_scale"] * ball_progress_toward_goal_m
    if ball_went_out_after_touch:
        r += cfg["ball_out_penalty"]
    if illegal_action_attempted:
        r += cfg["illegal_action_penalty"]
    if reached_opponent_box_with_possession:
        r += cfg["box_possession_terminal"]
    return r


def phase2_reward(
    shot_taken_this_step: bool,
    ticks_since_episode_start: int,
    max_episode_ticks: int,
    shot_on_target: bool,
    goal_scored: bool,
    illegal_action_attempted: bool,
    possession_lost_to_keeper: bool,
    cfg: dict,
) -> float:
    """Shoot experiment reward (curriculum phase 2).

    See ai_design_doc.md section 10.2.

    Args:
        shot_taken_this_step: Player kicked toward goal this step.
        ticks_since_episode_start: Engine ticks elapsed since episode start
            (used for the "faster shot = better" decay).
        max_episode_ticks: Total episode length in ticks.
        shot_on_target: Shot would reach the goal (not blocked/wide).
        goal_scored: Ball crossed the goal line this step.
        illegal_action_attempted: Illegal action flag from to_orders.py.
        possession_lost_to_keeper: Keeper caught/saved the ball.
        cfg: The 'reward.phase2' section of ai_config.json.
    """
    r = 0.0
    if shot_taken_this_step:
        # Faster shots rewarded more - linearly decaying bonus
        time_fraction = ticks_since_episode_start / max(max_episode_ticks, 1)
        r += cfg["shot_speed_bonus_max"] * max(0.0, 1.0 - time_fraction)
        if shot_on_target:
            r += cfg["shot_on_target_bonus"]
        if goal_scored:
            r += cfg["goal_terminal"]
    if illegal_action_attempted:
        r += cfg["illegal_action_penalty"]
    if possession_lost_to_keeper:
        r += cfg["possession_lost_to_keeper_penalty"]
    return r


class EMAFilter:
    """Exponential moving average for the attack/defence weighting (section 2.7).

    Args:
        alpha_normal: Smoothing factor during normal play (close to 1 = slow).
        alpha_post_goal: Smoothing factor right after a goal (further from 1).
        post_goal_window_s: How many seconds after a goal the fast-alpha window
            lasts before reverting to the normal alpha.
    """

    def __init__(
        self,
        alpha_normal: float = 0.995,
        alpha_post_goal: float = 0.95,
        post_goal_window_s: float = 10.0,
    ):
        self.alpha_normal = alpha_normal
        self.alpha_post_goal = alpha_post_goal
        self.post_goal_window_s = post_goal_window_s
        self._smoothed: float = 0.5
        self._post_goal_remaining_s: float = 0.0

    @property
    def smoothed(self) -> float:
        return self._smoothed

    def update(self, raw_value: float, dt_s: float) -> float:
        """Update and return the smoothed value.

        Args:
            raw_value: The instantaneous target (decision network's sigmoid
                output for attack_defence_raw).
            dt_s: Elapsed time since last update (in sim-seconds, i.e.
                typically one decision interval = 0.5s).
        """
        if self._post_goal_remaining_s > 0:
            alpha = self.alpha_post_goal
            self._post_goal_remaining_s = max(0.0, self._post_goal_remaining_s - dt_s)
        else:
            alpha = self.alpha_normal
        self._smoothed = alpha * self._smoothed + (1.0 - alpha) * raw_value
        return self._smoothed

    def on_goal(self) -> None:
        """Call when a goal is scored to enter the fast-change window."""
        self._post_goal_remaining_s = self.post_goal_window_s

    def reset(self, value: float = 0.5) -> None:
        """Reset the filter (e.g. on episode reset)."""
        self._smoothed = value
        self._post_goal_remaining_s = 0.0

    @classmethod
    def from_config(cls) -> "EMAFilter":
        from footballcoach.ai.config import load_ai_config
        cfg = load_ai_config()["ema"]
        return cls(
            alpha_normal=cfg["attack_defence_alpha_normal"],
            alpha_post_goal=cfg["attack_defence_alpha_post_goal"],
            post_goal_window_s=cfg["post_goal_window_s"],
        )
