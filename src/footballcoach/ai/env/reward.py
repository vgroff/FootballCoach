"""Per-head reward shaping functions for each curriculum phase.

See ai_design_doc.md sections 10.1 and 10.2 for the concrete starting
formulas.  All coefficients are loaded from ai_config.json so they can
be tuned without touching code.

These functions are called by the env wrappers at the end of each
decision-interval step (after ~15 engine ticks) to compute the scalar
reward for that step.

Convention: reward is always a Python float, accumulated over the engine
ticks within one decision interval and returned as one scalar per step.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! CRITICAL, STANDING INVARIANT: NO REWARD TERM MAY DEPEND ON            !!
!! INITIAL-EPISODE STATE. NOT EVEN AS A NORMALISATION DIVISOR.           !!
!!                                                                      !!
!! Normalising or scaling a reward term by a quantity measured at        !!
!! episode start (e.g. start_ball_dist_m, start_ball_to_box_dist_m,     !!
!! start_stamina) is a MISTAKE in PPO.  Reason:                          !!
!!                                                                      !!
!!  1. The value network does NOT observe the initial-state normaliser. !!
!!     It sees only the current observation. So the same observed state !!
!!     produces different return signals across episodes purely because !!
!!     the spawn was different — the value function literally cannot    !!
!!     predict this variation, breaking its training signal.            !!
!!                                                                      !!
!!  2. PPO + GAE already handles cross-episode baseline variation via   !!
!!     advantage estimation. That is what the advantage IS for. You do  !!
!!     not need to normalise manually; doing so fights the baseline.    !!
!!                                                                      !!
!! If a large spawn distance produces "too much free reward", the fix   !!
!! is to tighten the spawn distribution, add episode-cumulative clamps  !!
!! (see cumulative_clamped_delta()), or tune the coefficient — NOT to   !!
!! divide by an unobservable initial quantity.                          !!
!!                                                                      !!
!! Every term below is written against this invariant: appr/appr_sq use !!
!! raw per-step deltas, prog uses raw ball_progress_toward_goal_m, prox  !!
!! uses a FIXED 40m pitch-scale reference (not a per-episode start       !!
!! distance), spd is speed_scale * time_remaining_fraction only, and     !!
!! stam uses (1 - final_stamina), never (start_stamina - final_stamina). !!
!! When adding a new term or reviewing a PR, check it against this list  !!
!! — do not reintroduce a start_*_m / start_stamina style parameter.     !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

phase1_reward() returns a (float, dict[str, float]) tuple where the dict
breaks the total down by source key:
  appr  — linear ball approach bonus (potential-based; telescopes over an
          episode to net distance closed, NOT sensitive to how fast — see
          appr_sq below for a term that actually rewards speed). Raw
          per-step distance delta, no initial-state normalisation (see
          module invariant above).
  retr  — linear ball retreat penalty (symmetric formula to appr). Also raw,
          also not normalized — a penalty's absolute weight should be
          scenario-independent regardless of spawn distance.
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
          reward genuine speed rather than just final position. Raw
          player_speed_toward_ball^2, no initial-state normalisation.
          appr_sq_approach_reward_clamp/appr_sq_retreat_reward_clamp (both
          null = uncapped) independently bound the EPISODE-CUMULATIVE total
          of each side via cumulative_clamped_delta() -- two separate bounds
          (not one symmetric clamp) to match the independently-tunable
          bonus/penalty coefficients above.
  hdg   — heading penalty (moving away from the ball)
  poss  — gain possession bonus
  prog  — ball progress toward the opponent BOX while in possession (delta
          of _ball_dist_to_opponent_box(), i.e. box-distance closed this
          step — NOT raw goal-line x movement, so lateral movement into the
          box counts). Raw ball_progress_toward_goal_m (m/step), no
          initial-state normalisation; NOT split into an asymmetric
          forward/backward pair like appr/retr, since progress is a single
          signed quantity with one coefficient, not two.
          prog_reward_clamp (when set) bounds the EPISODE-CUMULATIVE total
          of this term to +/-prog_reward_clamp via cumulative_clamped_delta()
          — NOT each step independently, since that alone wouldn't stop many
          small steps from summing past the clamp over a long/wandering
          episode. Caller must track and pass back prog_cumulative_before/
          prog_cumulative_after across steps for the same player/episode.
  out   — ball out of bounds penalty
  ill   — illegal action penalty
  box   — box possession terminal
  spd   — speed bonus (fast finish). speed_scale * time_remaining_fraction only.
  lpos  — loss of possession penalty
  lterm — loss terminal (opponent reaches trainee box)
  tout  — timeout penalty
  prox  — proximity bonus on timeout. Uses a FIXED 40m pitch-scale reference
          (scale * max(0, 1 - dist_to_box / 40m)), NOT a per-episode start
          distance — see module invariant above.
  step  — flat per-step penalty (encourages finishing faster; every step incl. terminal)
  stam  — stamina usage penalty (episode-end only). (1 - final_stamina), never
          (start_stamina - final_stamina) — see module invariant above.
"""
from __future__ import annotations


def cumulative_clamped_delta(
    raw_delta: float,
    cumulative_before: float,
    clamp_max: float | None = None,
    clamp_min: float | None = None,
) -> tuple[float, float]:
    """Clamp a running EPISODE TOTAL to [clamp_min, clamp_max], returning this
    step's payout.

    Clamping each step's raw value independently only bounds that one step —
    many small steps under the per-step clamp can still sum to an unbounded
    episode total (e.g. a wandering trajectory). This instead tracks the
    UNCLAMPED running sum, clamps it to [clamp_min, clamp_max], and returns
    the marginal change vs the previous step's clamped value as the actual
    payout — so the episode-wide sum of payouts can never exceed those
    bounds, regardless of how many steps it takes. Once the running total
    saturates at a bound, further movement in the same direction pays 0
    until the total reverses back inside the bound.

    clamp_max/clamp_min are independent so asymmetric terms (e.g. appr_sq's
    separately-tunable approach/retreat coefficients) can bound each side by
    a different amount. Use symmetric_clamp() for the common single-bound
    case (e.g. "prog").

    Args:
        raw_delta: This step's unclamped contribution (e.g. a reward term
            before clamping).
        cumulative_before: The running UNCLAMPED sum from all previous steps
            this episode (0.0 at episode start).
        clamp_max: Upper bound on the cumulative CLAMPED total. None = no
            upper bound.
        clamp_min: Lower bound on the cumulative CLAMPED total. None = no
            lower bound.

    Returns:
        (payout, cumulative_after) — payout is this step's actual (possibly
        clamped) contribution; cumulative_after is the new unclamped running
        sum to pass back in as cumulative_before on the next step.
    """
    cumulative_after = cumulative_before + raw_delta
    if clamp_max is None and clamp_min is None:
        return raw_delta, cumulative_after
    lo = clamp_min if clamp_min is not None else float("-inf")
    hi = clamp_max if clamp_max is not None else float("inf")
    clamped_before = max(lo, min(hi, cumulative_before))
    clamped_after = max(lo, min(hi, cumulative_after))
    return clamped_after - clamped_before, cumulative_after


def symmetric_clamp(clamp: float | None) -> tuple[float | None, float | None]:
    """Convenience for the common single-bound case: clamp -> (clamp_max, clamp_min)
    i.e. (+clamp, -clamp), matching cumulative_clamped_delta()'s param order."""
    if clamp is None:
        return None, None
    return clamp, -clamp


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
    opponent_reached_trainee_box: bool = False,
    lost_possession_this_step: bool | int = False,
    timed_out: bool = False,
    ball_dist_to_opponent_box_m: float = 9999.0,
    heading_cos_sim: float = 1.0,
    player_speed_mps: float = 0.0,
    stamina_used: float = 0.0,
    episode_done: bool = False,
    prog_reward_clamp: float | None = None,
    appr_sq_approach_reward_clamp: float | None = None,
    appr_sq_retreat_reward_clamp: float | None = None,
    cumulative_state: dict[str, float] | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """GetPossession/Move experiment reward (curriculum phase 1).

    Returns a (total_reward, components, cumulative_state_after) tuple where
    components maps short source keys to their individual contributions for
    this step, and cumulative_state_after maps term name (e.g. "prog",
    "appr_sq") to that term's running UNCLAMPED sum for this episode -- pass
    it back in as cumulative_state on the next call for the SAME
    player/episode (see cumulative_clamped_delta() and the "prog"/"appr_sq"
    entries in the module docstring). Missing keys in cumulative_state
    default to 0.0, so adding a newly-clamped term never requires updating
    existing callers' dict contents.

    See ai_design_doc.md section 10.1.
    """
    r = 0.0
    comps: dict[str, float] = {}
    cumulative_state = cumulative_state or {}
    cumulative_state_after: dict[str, float] = {}

    _delta = prev_ball_dist - curr_ball_dist  # positive = closing, negative = retreating
    if _delta >= 0:
        appr_r = cfg.get("ball_approach_bonus", cfg.get("ball_distance_shaping", 0.002)) * _delta
        retr_r = 0.0
    else:
        appr_r = 0.0
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
    # (default) disable each side independently.
    _player_speed_toward_ball = player_speed_mps * heading_cos_sim
    _appr_sq_coef = float(cfg.get("ball_approach_speed_bonus", 0.0))
    _retr_sq_coef = float(cfg.get("ball_retreat_speed_penalty", _appr_sq_coef))
    if _player_speed_toward_ball > 0.0:
        _appr_sq_raw = _appr_sq_coef * (_player_speed_toward_ball ** 2)
    elif _player_speed_toward_ball < 0.0:
        _appr_sq_raw = -_retr_sq_coef * (_player_speed_toward_ball ** 2)
    else:
        _appr_sq_raw = 0.0
    # Independent per-side clamps (not symmetric_clamp()) since approach/
    # retreat already have independently tunable coefficients above.
    appr_sq_r, cumulative_state_after["appr_sq"] = cumulative_clamped_delta(
        _appr_sq_raw,
        cumulative_state.get("appr_sq", 0.0),
        clamp_max=appr_sq_approach_reward_clamp,
        clamp_min=-appr_sq_retreat_reward_clamp if appr_sq_retreat_reward_clamp is not None else None,
    )
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

    # Ball progress toward box while in possession. prog_reward_clamp bounds
    # the EPISODE-CUMULATIVE total — a per-step clamp alone doesn't stop many
    # small steps summing past the clamp over a long/wandering episode.
    _prog_raw = cfg["ball_progress_scale"] * ball_progress_toward_goal_m if has_possession_now else 0.0
    prog_r, cumulative_state_after["prog"] = cumulative_clamped_delta(
        _prog_raw, cumulative_state.get("prog", 0.0), *symmetric_clamp(prog_reward_clamp)
    )
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
        # Speed bonus: reward finishing faster. Range [0, speed_bonus_scale].
        speed_scale = float(cfg.get("speed_bonus_scale", 0.0))
        if speed_scale > 0.0:
            spd_r = speed_scale * time_fraction_remaining
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
        # Proximity bonus on timeout: consolation for how close the ball is
        # to the opponent box. Uses a fixed 40m pitch-scale reference so the
        # formula is fully observable (no initial-state dependency).
        # scale * max(0, 1 - dist_to_box / 40m) → 0 at 40m+, scale at box edge.
        prox_scale = float(cfg.get("proximity_bonus_scale", 0.0))
        if prox_scale > 0.0:
            prox = max(0.0, 1.0 - ball_dist_to_opponent_box_m / 40.0)
            prox_r = prox_scale * prox
    r += tout_r + prox_r
    comps["tout"] = tout_r
    comps["prox"] = prox_r

    # Flat per-step penalty to discourage dawdling. Applied every step (including
    # terminal steps). Equivalent to a time limit that "costs" something.
    step_r = -float(cfg.get("step_penalty", 0.0))
    r += step_r
    comps["step"] = step_r

    # Stamina penalty — only at episode end. stamina_used = 1 - final_stamina
    # (0.0 on non-terminal steps), so -coef * (1 - full) penalises exhaustion.
    _stam_coef = float(cfg.get("stamina_sprint_penalty", 0.0))
    stam_r = -_stam_coef * stamina_used if (episode_done and _stam_coef > 0.0) else 0.0
    r += stam_r
    comps["stam"] = stam_r

    return r, comps, cumulative_state_after


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
