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
!! is to tighten the spawn distribution, add an episode-cumulative      !!
!! running-total clamp, or tune the coefficient — NOT to divide by an   !!
!! unobservable initial quantity.                                       !!
!!                                                                      !!
!! Every term below is written against this invariant: "spd" is         !!
!! speed_scale * time_remaining_fraction only (no start-of-episode      !!
!! reference at all). When adding a new term or reviewing a PR, check   !!
!! it against this list — do not reintroduce a start_*_m / start_stamina!!
!! style parameter.                                                      !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! STANDING GUIDELINE: BE VERY CAREFUL WITH ONE-OFF (EVENT-PAID) REWARDS !!
!! AND PENALTIES -- AVOID THEM UNLESS THEY ARE TERMINAL. PREFER          !!
!! PER-STEP OR POTENTIAL-BASED TERMS WHEREVER POSSIBLE.                  !!
!!                                                                      !!
!! Why (all measured 2026-09-20 when gain_possession_bonus +1.0 and     !!
!! loss_of_possession_penalty -0.9 were removed):                       !!
!!  1. The ground-truth return-to-go is a sum of FUTURE rewards, so it   !!
!!     rises toward a one-off payment and DROPS by ~the payment the      !!
!!     instant it is made -- right after the agent did something good.   !!
!!     Measured: mean step -0.86 in return-to-go after a possession      !!
!!     gain. The value net must forecast WHEN such events happen (mostly !!
!!     unpredictable: tackles/turnovers): they added ~14% to the return  !!
!!     variance and value error near them was 1.2x-6.8x higher (partly   !!
!!     confounded with turnovers genuinely mattering).                   !!
!!  2. A non-potential-based event payment changes the optimal policy,   !!
!!     and gain +1.0 / loss -0.9 netted +0.1 per cycle (farmable).       !!
!!  3. It was redundant: the win/lose terminals already separate having  !!
!!     vs not having the ball by ~3.5 of return.                         !!
!!  After removal the value fit improved (R2 ~0.77 -> ~0.81 on the new   !!
!!  targets; not strictly comparable) and timeout-episode return std     !!
!!  fell from ~0.46 to ~0.07 (it was mostly possession-event noise).     !!
!!                                                                      !!
!! Rules:                                                                !!
!!  * Terminal one-offs are fine (box, lterm, tout, out): the episode    !!
!!    ends, there is no future for the return to drop into, and they     !!
!!    ARE the objective.                                                 !!
!!  * Otherwise prefer a bounded PER-STEP term, or POTENTIAL-BASED       !!
!!    shaping F = gamma*Phi(s') - Phi(s) with Phi(terminal) = 0 (it      !!
!!    telescopes and keeps the optimal policy). A potential term still   !!
!!    shifts the value by -Phi(s), but that offset is an observable      !!
!!    function of state, not of future event timing.                    !!
!!  * Any per-step term must be bounded over the maximum episode length: !!
!!    per_step * max_steps must stay well below the terminal win reward, !!
!!    or holding/dawdling beats winning (bound it with an episode-       !!
!!    cumulative running-total clamp).                                   !!
!!  * Never pair a bonus with a near-equal offsetting penalty that nets  !!
!!    non-zero.                                                          !!
!!  * If you must add a non-terminal event term: justify it, default it  !!
!!    to 0.0, and measure its effect first (recompute MC returns with    !!
!!    and without it: return variance, value R2, per-outcome loss).      !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

phase1_reward() returns a (float, dict[str, float]) tuple where the dict
breaks the total down by source key:
  out   — ball out of bounds penalty
  box   — box possession terminal
  spd   — speed bonus (fast finish). speed_scale * time_remaining_fraction only.
  lterm — loss terminal (opponent reaches trainee box)
  tout  — timeout penalty
  sprint — sprint penalty. -sprint_penalty_per_s * decision_interval_s
          whenever the player was sprinting this step, 0 otherwise. Applied
          every DECISION step (this function's own per-call granularity --
          see phase1_reward's decision_interval_s arg), scaled by the
          decision interval's real duration in seconds (not a flat
          per-step cost) so the total per-second cost of sprinting stays
          the same regardless of decision_interval_s tuning.
  appr  — ball-approach shaping: ball_approach_bonus * player_ball_approach_delta_m,
          i.e. a potential Phi(s) = -ball_approach_bonus * player_to_ball_dist(s),
          with F = Phi(s) - Phi(s') (gamma=1) per tick, gated on
          ball.last_touched_by_player_id != this player (see 2026-10-15 note
          below) -- active while chasing a ball someone ELSE last touched,
          frozen while this player is the one who last touched it (including
          the flight of their OWN kick). Deliberately a SINGLE coefficient
          for both closing and opening the gap (no separate "retreat" rate)
          -- see 2026-10-14 re-introduction note below for why that matters.
  prog  — ball-progress shaping: ball_progress_scale * ball_progress_toward_goal_m,
          i.e. the same potential shape as "appr" but
          Phi(s) = -ball_progress_scale * ball_dist_to_opponent_box(s), gated
          on ball.last_touched_by_player_id == this player (see 2026-10-15
          note below) -- active while possessing AND during the flight of
          this player's own kick (a kick clears ball.possessed_by but never
          touches last_touched_by_player_id -- see Ball's own docstring), so
          a deliberate forward kick correctly earns progress credit for the
          ground it covers. Phi is constant (contributes nothing) while
          last_touched_by_player_id is someone else's, so no reward is paid
          or charged purely for gaining/losing the ball itself; see the
          STANDING GUIDELINE above re one-off event payments.
          player_ball_approach_delta_m / ball_progress_toward_goal_m are the
          caller's ALREADY per-tick-accumulated, PRE-tick-last_touched_by-
          gated deltas for this decision interval (ScenarioEnv's
          trainee_appr_accum/trainee_prog_accum and their sec_* equivalents)
          -- phase1_reward itself applies no further gate on top of that.

2026-10-14 re-introduction note ("appr"/"prog", after being 0.0 since
2026-09 and removed entirely alongside a larger dead-knob cleanup earlier
the same day): the ORIGINAL pre-removal version of "appr" used TWO
independently-tunable coefficients (ball_approach_bonus for closing,
ball_retreat_penalty for opening, with retreat penalized more) -- that is
NOT a potential-based term. A true potential Phi(s) = -c*dist(s) requires
ONE c for both directions; using c1 going one way and c2 the other breaks
the telescoping identity itself (a closed round-trip no longer nets exactly
zero -- it nets (c2-c1)*(detour distance), which IS farmable by shuttling
back and forth, and the episode total depends on PATH, not just start/end
state). This re-introduction is deliberately back to a single coefficient
for exactly that reason -- see TestApprTelescopes/TestApprRoundTripNetsZero
in test_reward.py, which assert the telescoping identity and the zero-net
round-trip directly rather than trusting the arithmetic by eye. The squared
"appr_sq" speed-reward variant from the same removed generation is NOT
reintroduced -- it was explicitly non-telescoping by design (rewards
concentrating the same net distance into fewer/faster steps), which is
exactly the property being guarded against here.

2026-10-15 "last toucher" gating note: the first cut of this
re-introduction gated "prog" on live possession and left "appr" totally
unconditional. That reintroduced the EXACT problem the OLD (pre-removal)
"appr_sq" design doc already warned about for raw ball-distance deltas --
contamination by the ball's OWN motion, independent of player skill -- just
via a more direct path: a player's own kick ALWAYS separates them from the
ball (at least momentarily, often for many ticks on a hard kick), so
unconditional "appr" charged a self-inflicted penalty for the mere act of
kicking, not recoverable within the episode if the player (or anyone) never
closed the gap again before episode end. Separately, gating "prog" on LIVE
possession meant a deliberate forward kick -- often the fastest way to make
real progress -- earned ZERO progress credit for the ball's flight, since
kicking immediately clears ball.possessed_by. Both are fixed by gating on
ball.last_touched_by_player_id instead of live possession/distance: that
field is written ONLY by Match._set_possession (a genuine gain), NEVER by a
kick itself, so it already means exactly "whose play should this ball
movement be attributed to" -- surviving through a player's own kick's
flight until someone else actually takes the ball over. Both terms keep
telescoping exactly within an unbroken "last touched by the same
player_id" stretch (same proof as before, just a better-chosen gate), and
remain non-exploitable for the same reason: a round trip that never changes
who last touched the ball nets exactly zero.
"""
from __future__ import annotations


def phase1_reward(
    ball_went_out_after_touch: bool,
    reached_opponent_box_with_possession: bool,
    cfg: dict,
    time_fraction_remaining: float = 0.0,
    opponent_reached_trainee_box: bool = False,
    timed_out: bool = False,
    is_sprinting: bool = False,
    decision_interval_s: float = 0.0,
    player_ball_approach_delta_m: float = 0.0,
    ball_progress_toward_goal_m: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """GetPossession/Move experiment reward (curriculum phase 1).

    Returns a (total_reward, components) tuple where components maps short
    source keys to their individual contributions for this step.

    See ai_design_doc.md section 10.1.
    """
    r = 0.0
    comps: dict[str, float] = {}

    out_r = cfg["ball_out_penalty"] if ball_went_out_after_touch else 0.0
    r += out_r
    comps["out"] = out_r

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

    lterm_r = cfg.get("loss_terminal", 0.0) if opponent_reached_trainee_box else 0.0
    r += lterm_r
    comps["lterm"] = lterm_r

    tout_r = cfg.get("timeout_penalty", 0.0) if timed_out else 0.0
    r += tout_r
    comps["tout"] = tout_r

    # Sprint penalty — every DECISION step (this function's own call
    # granularity, not every raw engine tick), scaled by the real elapsed
    # decision-interval duration so the total per-second cost of sprinting
    # doesn't silently change if decision_interval_s is retuned later.
    _sprint_coef = float(cfg.get("sprint_penalty_per_s", 0.0))
    sprint_r = -_sprint_coef * decision_interval_s if (is_sprinting and _sprint_coef > 0.0) else 0.0
    r += sprint_r
    comps["sprint"] = sprint_r

    # Ball-approach shaping -- pure potential Phi(s) = -ball_approach_bonus *
    # player_to_ball_dist(s), ONE coefficient for both closing and opening
    # the gap (see module docstring's 2026-10-14 re-introduction note for why
    # a second, asymmetric "retreat" coefficient is deliberately NOT exposed
    # here -- it would break the telescoping/no-exploit guarantee). The
    # caller's player_ball_approach_delta_m is ALREADY gated per-tick on
    # last_touched_by_player_id != this player (see module docstring) -- no
    # further gate here.
    _appr_coef = float(cfg.get("ball_approach_bonus", 0.0))
    appr_r = _appr_coef * player_ball_approach_delta_m
    r += appr_r
    comps["appr"] = appr_r

    # Ball-progress shaping -- same pure-potential shape, Phi(s) =
    # -ball_progress_scale * ball_dist_to_opponent_box(s). The caller's
    # ball_progress_toward_goal_m is ALREADY gated per-tick on
    # last_touched_by_player_id == this player (see module docstring) -- no
    # further gate here.
    _prog_coef = float(cfg.get("ball_progress_scale", 0.0))
    prog_r = _prog_coef * ball_progress_toward_goal_m
    r += prog_r
    comps["prog"] = prog_r

    return r, comps


def phase1_terminal_reward_only(
    *,
    reached_opponent_box_with_possession: bool,
    opponent_reached_trainee_box: bool,
    timed_out: bool,
    ball_went_out_after_touch: bool,
    cfg: dict,
    time_fraction_remaining: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Approximate a phase-1 episode's reward using ONLY phase1_reward()'s
    TERMINAL-gated components (box/spd, lterm, tout, out) -- for callers
    that only have the final-tick state, not the continuous per-tick
    running state (is_sprinting, decision_interval_s,
    player_ball_approach_delta_m, ball_progress_toward_goal_m, ...) the
    "sprint"/"appr"/"prog" terms need. Those are fed neutral inputs below so
    they always contribute exactly 0.

    Intended for the UI's live scenario view (``ScenarioLoop``, which never
    routes through ``ScenarioEnv`` and so never tracks that per-tick state --
    see ``ai/knowledge.md``), to show a rough "what did this episode's
    ending actually pay out" number during the post-trial linger. This is
    NOT the real per-episode training reward (the sum of every per-tick
    shaping term across the whole episode) -- just this one terminal
    snapshot's contribution, useful for sanity-checking the terminal/outcome
    coefficients by eye, not for comparing against training logs.
    """
    return phase1_reward(
        ball_went_out_after_touch=ball_went_out_after_touch,
        reached_opponent_box_with_possession=reached_opponent_box_with_possession,
        cfg=cfg,
        time_fraction_remaining=time_fraction_remaining,
        opponent_reached_trainee_box=opponent_reached_trainee_box,
        timed_out=timed_out,
        is_sprinting=False,
        decision_interval_s=0.0,
    )


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
        illegal_action_attempted: Illegal action flag from apply_nn_action.py.
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
