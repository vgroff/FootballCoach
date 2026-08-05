"""Per-head reward shaping functions for each curriculum phase.

See ai_design_doc.md sections 10.1 and 10.2 for the concrete starting
formulas.  All coefficients are loaded from ai_config.json so they can
be tuned without touching code.

These functions are called by the env wrappers at the end of each
decision-interval step (after ~15 engine ticks) to compute the scalar
reward for that step.

Convention: reward is always a Python float, accumulated over the engine
ticks within one decision interval and returned as one scalar per step.

phase1_reward() returns a (float, dict[str, float]) tuple where the dict
breaks the total down by source key:
  appr  — linear ball approach bonus (potential-based; telescopes over an
          episode to net distance closed, NOT sensitive to how fast — see
          appr_sq below for a term that actually rewards speed). Normalized
          by start_ball_dist_m (the ball-to-player distance at episode
          start) so total achievable reward from this term is ~coef
          regardless of how far the ball happened to spawn — otherwise a
          scenario with a larger spawn distance hands out more "free"
          closing-distance reward purely from randomness, not skill.
  retr  — linear ball retreat penalty (symmetric formula to appr, but
          DELIBERATELY NOT normalized — a penalty's absolute weight should
          be scenario-independent: normalizing it would make a wasted/wrong-
          direction metre cheaper in large-spawn-distance episodes than in
          small ones, which is backwards for a deterrent).
  appr_sq — squared bonus/penalty for the PLAYER's own speed toward/away
          from the ball (asymmetric coefficients: ball_approach_speed_bonus /
          ball_retreat_speed_penalty). Uses player_speed_mps * heading_cos_sim
          (the scalar projection of the player's own velocity onto the
          direction-to-ball axis) — NOT prev_ball_dist - curr_ball_dist (the
          gap-closing rate), which is contaminated by the ball's own motion
          (e.g. a hard clearance kick moves the ball away fast independent of
          anything the player did, producing large uncontrollable swings that
          have nothing to do with player skill). Unlike appr/retr, this is
          NOT a linear potential-based term, so it does NOT telescope:
          closing the same net distance in fewer/faster steps yields
          strictly more total reward than doing it slowly, because
          sum(x_i^2) >= (sum(x_i))^2 / n with equality only when every step
          closes the same amount. Use this (not appr) when you want to
          reward genuine speed rather than just final position. Same
          normalization asymmetry as appr/retr above: the approach
          (closing-fast) side is normalized by start_ball_dist_m before
          squaring, the retreat (retreating-fast) side is not.
  hdg   — heading penalty (moving away from the ball)
  poss  — gain possession bonus
  prog  — ball progress while in possession (normalized by
          start_ball_to_box_dist_m — same reference distance already used by
          spd/prox below; NOT split into an asymmetric forward/backward pair
          like appr/retr, since progress is a single signed quantity with
          one coefficient, not two)
  out   — ball out of bounds penalty
  ill   — illegal action penalty
  box   — box possession terminal
  spd   — speed bonus (fast finish)
  lpos  — loss of possession penalty
  lterm — loss terminal (opponent reaches trainee box)
  tout  — timeout penalty
  prox  — proximity bonus on timeout. Normalized by start_ball_to_box_dist_m
          (previously a fixed 30m constant, inconsistent with spd's use of
          the actual per-episode start distance)
  stam  — stamina usage penalty (episode-end only)
"""
from __future__ import annotations


def phase1_reward(
    prev_ball_dist: float,
    curr_ball_dist: float,
    has_possession_now: bool,
    gained_possession_this_step: bool | int,
    ball_progress_toward_goal_m: float,
    ball_went_out_after_touch: bool,
    illegal_action_attempted: bool,
    reached_opponent_box_with_possession: bool,
    cfg: dict,
    time_fraction_remaining: float = 0.0,
    start_ball_to_box_dist_m: float = 1.0,
    start_ball_dist_m: float = 1.0,
    opponent_reached_trainee_box: bool = False,
    lost_possession_this_step: bool | int = False,
    timed_out: bool = False,
    ball_dist_to_opponent_box_m: float = 9999.0,
    heading_cos_sim: float = 1.0,
    player_speed_mps: float = 0.0,
    stamina_used: float = 0.0,
    episode_done: bool = False,
) -> tuple[float, dict[str, float]]:
    """GetPossession/Move experiment reward (curriculum phase 1).

    Returns a (total_reward, components) tuple where components maps short
    source keys to their individual contributions for this step.
    See module docstring for key definitions.

    See ai_design_doc.md section 10.1.
    """
    r = 0.0
    comps: dict[str, float] = {}

    # Floor the normalizing distance so a ball spawning ~on top of the player
    # doesn't create a divide-by-near-zero blow-up in the normalized terms.
    _norm_ball_dist = max(start_ball_dist_m, 1.0)

    _delta = prev_ball_dist - curr_ball_dist  # positive = closing, negative = retreating
    if _delta >= 0:
        # Approach (reward): normalized by start_ball_dist_m so the total
        # achievable reward from closing distance is ~coef regardless of the
        # episode's spawn distance (see module docstring).
        appr_r = cfg.get("ball_approach_bonus", cfg.get("ball_distance_shaping", 0.002)) * (_delta / _norm_ball_dist)
        retr_r = 0.0
    else:
        appr_r = 0.0
        # Retreat (penalty): deliberately NOT normalized — a deterrent's
        # absolute weight should be scenario-independent (see module docstring).
        retr_r = cfg.get("ball_retreat_penalty", cfg.get("ball_distance_shaping", 0.002)) * _delta
    r += appr_r + retr_r
    comps["appr"] = appr_r
    comps["retr"] = retr_r

    # Squared closing-speed bonus/penalty: rewards CLOSING FAST, penalises
    # RETREATING FAST — symmetric in shape but with independently tunable
    # coefficients (mirrors the appr/retr asymmetric-coefficient pattern
    # above). Unlike appr/retr (linear in _delta, potential-based, telescopes
    # over an episode to coef * (start_dist - end_dist) — a player who
    # closes 10m in 2 fast steps and one who closes the same 10m in 20 slow
    # steps earn IDENTICAL total appr/retr reward), squaring breaks that
    # telescoping: concentrating the same net distance into fewer/faster
    # steps yields strictly more total reward
    # (sum(x_i^2) >= (sum(x_i))^2/n, equality only when every step is equal).
    #
    # IMPORTANT: uses the PLAYER's own speed toward the ball
    # (player_speed_mps * heading_cos_sim — the scalar projection of the
    # player's velocity onto the direction-to-ball axis), NOT _delta
    # (prev_ball_dist - curr_ball_dist). _delta measures the closing rate of
    # the GAP, i.e. the *relative* velocity between player and ball, and is
    # contaminated by the ball's own motion (e.g. a hard clearance kick moves
    # the ball away fast independent of anything the player did, producing
    # large uncontrollable _delta swings — and once squared, huge outlier
    # penalties — that have nothing to do with player skill).
    # player_speed_mps * heading_cos_sim depends only on the player's own
    # velocity, so ball motion no longer affects this term. Coefficients 0.0
    # (default) disable each side independently. Approach side normalizes by
    # start_ball_dist_m BEFORE squaring (dividing first then squaring gives
    # the dimensionally-correct v^2/start_dist^2 scaling, not v^2/start_dist).
    # Retreat side is deliberately left unnormalized — same asymmetric
    # rationale as appr/retr above.
    _player_speed_toward_ball = player_speed_mps * heading_cos_sim
    _appr_sq_coef = float(cfg.get("ball_approach_speed_bonus", 0.0))
    _retr_sq_coef = float(cfg.get("ball_retreat_speed_penalty", _appr_sq_coef))
    if _player_speed_toward_ball > 0.0:
        appr_sq_r = _appr_sq_coef * ((_player_speed_toward_ball / _norm_ball_dist) ** 2)
    elif _player_speed_toward_ball < 0.0:
        appr_sq_r = -_retr_sq_coef * (_player_speed_toward_ball ** 2)
    else:
        appr_sq_r = 0.0
    r += appr_sq_r
    comps["appr_sq"] = appr_sq_r

    # Cosine heading penalty: penalise moving away from the ball based on velocity direction.
    # Only fires when the player is actually moving (speed > threshold).
    # Formula: -coef * max(0, 1 - cos_sim) ** exp
    #   cos_sim = 1  (toward ball): penalty = 0
    #   cos_sim = 0  (perpendicular): penalty = -coef
    #   cos_sim = -1 (directly away): penalty = -coef * 2^exp
    # Set heading_penalty_coef=0.0 (default) to disable entirely.
    _hdg_coef = float(cfg.get("heading_penalty_coef", 0.0))
    hdg_r = 0.0
    if _hdg_coef > 0.0 and player_speed_mps > float(cfg.get("heading_penalty_min_speed_mps", 0.5)):
        _exp = float(cfg.get("heading_penalty_exponent", 2.0))
        hdg_r = -_hdg_coef * max(0.0, 1.0 - heading_cos_sim) ** _exp
    r += hdg_r
    comps["hdg"] = hdg_r

    # gained_possession_this_step may be an int count (>1 if possession was
    # gained more than once within a single decision interval, e.g. gain ->
    # lose -> regain via tackles mid-tick-loop) -- multiply, don't just gate.
    poss_r = cfg["gain_possession_bonus"] * gained_possession_this_step
    r += poss_r
    comps["poss"] = poss_r

    # Normalized by start_ball_to_box_dist_m (same reference distance used
    # by spd/prox below) so total achievable progress reward over an episode
    # is ~ball_progress_scale regardless of how far the ball started from
    # the opponent box. Single symmetric term (forward and backward progress
    # share one coefficient) — not split into an asymmetric pair like appr/retr.
    _norm_box_dist = max(start_ball_to_box_dist_m, 1.0)
    prog_r = cfg["ball_progress_scale"] * (ball_progress_toward_goal_m / _norm_box_dist) if has_possession_now else 0.0
    r += prog_r
    comps["prog"] = prog_r

    out_r = cfg["ball_out_penalty"] if ball_went_out_after_touch else 0.0
    r += out_r
    comps["out"] = out_r

    ill_r = cfg["illegal_action_penalty"] if illegal_action_attempted else 0.0
    r += ill_r
    comps["ill"] = ill_r

    box_r = 0.0
    spd_r = 0.0
    if reached_opponent_box_with_possession:
        box_r = cfg["box_possession_terminal"]
        # Speed bonus: reward finishing faster, scaled by how far the ball
        # had to travel so a nearby start doesn't trivially earn a large bonus.
        # bonus = scale × time_remaining_fraction × clamp(start_dist / 30m, 0, 1.5)
        # Max possible bonus: scale × 1.0 × 1.5  (ball starts 45m+ from box)
        speed_scale = float(cfg.get("speed_bonus_scale", 0.0))
        if speed_scale > 0.0:
            dist_weight = min(start_ball_to_box_dist_m / 30.0, 1.5)
            spd_r = speed_scale * time_fraction_remaining * dist_weight
    r += box_r + spd_r
    comps["box"] = box_r
    comps["spd"] = spd_r

    # Same rationale as gained_possession_this_step above -- multiply by count.
    lpos_r = cfg.get("loss_of_possession_penalty", 0.0) * lost_possession_this_step
    r += lpos_r
    comps["lpos"] = lpos_r

    lterm_r = cfg.get("loss_terminal", 0.0) if opponent_reached_trainee_box else 0.0
    r += lterm_r
    comps["lterm"] = lterm_r

    tout_r = 0.0
    prox_r = 0.0
    if timed_out:
        tout_r = cfg.get("timeout_penalty", 0.0)
        # Proximity bonus on timeout: consolation for how close you got with
        # the ball. Normalized by start_ball_to_box_dist_m (the episode's own
        # ball-to-box distance at the start) rather than a fixed 30m constant
        # — consistent with spd's dist_weight above, which already
        # self-references the same per-episode start distance. A fixed 30m
        # denominator would always read a large-spawn-distance episode as
        # "far" even relative to its own start.
        # scale * max(0, 1 - dist_to_box / start_ball_to_box_dist_m) → 0 at
        # start distance or farther, scale at the box edge.
        prox_scale = float(cfg.get("proximity_bonus_scale", 0.0))
        if prox_scale > 0.0:
            prox = max(0.0, 1.0 - ball_dist_to_opponent_box_m / _norm_box_dist)
            prox_r = prox_scale * prox
    r += tout_r + prox_r
    comps["tout"] = tout_r
    comps["prox"] = prox_r

    # Stamina usage penalty — only applied at episode end (begin-to-end stamina drain).
    # Discourages unnecessary sprinting without punishing individual steps.
    # stamina_used = start_stamina - final_stamina (0.0 on non-terminal steps).
    _stam_coef = float(cfg.get("stamina_sprint_penalty", 0.0))
    stam_r = -_stam_coef * stamina_used if (episode_done and _stam_coef > 0.0) else 0.0
    r += stam_r
    comps["stam"] = stam_r

    return r, comps


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
