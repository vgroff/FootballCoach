"""ScenarioEnv: Gym-like wrapper around ScenarioDefinition + ScenarioLoop.

See ai_design_doc.md section 9.2 for the design rationale.

All players (trainee, secondary, rules-based) carry their AI in ``player.ai``.
``Match.step()`` calls ``ai.act()`` automatically every physics tick.

For neural players, ``ScenarioEnv`` assigns ``NeuralPlayerAI`` on ``reset()``.
After each ``step()``, ``env.last_trainee_transition`` holds the PPO rollout
data for the trainee; ``env.last_secondary_results`` holds the same for
any secondary neural players.

API:
    env = ScenarioEnv(definition, trainee_player_id="kicker", phase=1, max_episode_s=120.0)
    obs = env.reset()
    obs, reward, done, info = env.step()
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from footballcoach.ai.action.gating import select_action
from footballcoach.ai.action.apply_nn_action import (
    OrderTranslationResult,
    apply_action_to_player,
    encode_slot_player_ids,
)
from footballcoach.ai.config import load_ai_config
from footballcoach.ai.env.outcome import (
    can_score_box_terminal,
    detect_phase1_box_terminal,
    remap_phase1_outcome,
)
from footballcoach.ai.env.reward import EMAFilter, phase1_reward, phase2_reward
from footballcoach.ai.obs.encoder import MAX_OTHER_PLAYERS, encode_observation
from footballcoach.ai.obs.schema import ObservationBatch
from footballcoach.engine.match import Match
from footballcoach.engine.match_logger import MatchLogger
from footballcoach.engine.movement import SpeedMode
from footballcoach.entities.player import Team
from footballcoach.ui.scenarios import ScenarioDefinition, ScenarioLoop


@dataclass
class StepInfo:
    """Diagnostic information returned alongside each step's (obs, reward, done)."""
    illegal_action: bool = False
    illegal_reason: str = ""
    trial_outcome: Optional[str] = None  # "goal", "miss", "dispossessed", etc.
    ticks_elapsed: int = 0
    is_rules_episode: bool = False    # True when opponent is rules-based
    is_immobile_episode: bool = False  # True when opponent is immobile
    # Real, physics-executed trainee kicks (Player.kicked_this_tick firing),
    # scanned every physics tick of this decision interval inside step()'s own
    # loop -- NOT read off the player once after the whole interval elapses,
    # since kicked_this_tick resets every physics tick and one step() call
    # spans self._ticks_per_decision (>1) of them. Same rationale as the
    # trainee_gained_count/trainee_lost_count possession scan below.
    trainee_kicks_this_step: int = 0
    # Subset of trainee_kicks_this_step that fired via the armed path (no ball at
    # decision time, kick executed at pickup -- the "first touch" kicks).
    trainee_armed_kicks_this_step: int = 0


class ScenarioEnv:
    """Training environment wrapping a ScenarioDefinition + ScenarioLoop.

    One ``ScenarioEnv`` per trainee player.  Other players are rules-based,
    *unless* ``secondary_player_ids`` is provided — those players are also
    driven by the shared neural network and their transitions are added to the
    rollout buffer via ``last_secondary_results`` (drained by the trainer).

    Args:
        definition: A ScenarioDefinition from ui/scenarios.py.
        trainee_player_id: The player_id of the AI player being trained.
        phase: Curriculum phase (1 or 2) - selects the reward function.
        rng_reduction: Physics randomness level (updated by the trainer).
        max_episode_s: Maximum episode duration in sim-seconds. REQUIRED, no
            default -- this used to default to 120.0, which silently diverged
            from the real curriculum value (ai_config.json
            curriculum.phase1_max_episode_s, 22s as of this writing) at any
            call site that forgot to pass it explicitly. That exact gap
            caused two separate, real bugs this session: NeuralPlayerAI
            instances (rules_ai.py) built with the stale 120.0 default fed a
            wrong "time remaining" observation feature from tick 0, and a
            standalone comparison script silently ran 5.4x longer episodes
            than real training/eval ever does, invalidating its outcome
            distribution (near-zero timeouts vs a real few-percent rate).
            Forcing every caller to choose this value explicitly (usually by
            threading through whatever produced it, e.g. an existing env's
            own ``max_episode_s``, or curriculum/phases.py's
            ``_phase1_max_episode_s()``) is the real fix -- not a bigger
            default, which would just move the silent-mismatch risk
            elsewhere.
        max_episode_s_variability_s: Extra seconds, uniformly resampled in
            [0, this] at the START of every reset() and ADDED to
            max_episode_s for that one episode only (never subtracted, so an
            episode is never shorter than max_episode_s). 0.0 (default) means
            every episode gets exactly max_episode_s, bit-identical to not
            having this knob at all -- it costs nothing when unused (no rng
            draw happens; see reset()). Exists so a player that has learned
            to predict a fixed episode length can't reliably retreat and run
            out the clock to force the cheaper timeout_penalty over a worse
            loss_terminal -- randomizing (and/or raising) the ceiling makes
            that bet less certain without changing how a timeout itself is
            scored. self.max_episode_s is OVERWRITTEN each reset() with the
            resampled effective value for that episode (read by
            _max_episode_ticks, the time-remaining observation feature, and
            NeuralPlayerAI); the originally-passed floor is kept separately
            as self._base_max_episode_s.
        linger_s: How long to wait after trial ends before resetting.
        rng: Random for obs slot shuffling.
        secondary_player_ids: Additional player IDs to drive with the neural
            network when not in rules-based-AI mode.  Their transitions are
            collected each step and exposed via ``last_secondary_results``.
        scenario_kwargs: Extra kwargs forwarded to definition.build().
    """

    def __init__(
        self,
        definition: ScenarioDefinition,
        trainee_player_id: str,
        *,
        max_episode_s: float,
        max_episode_s_variability_s: float = 0.0,
        phase: int = 1,
        rng_reduction: float = 0.3,
        linger_s: float = 0.0,
        rng: Optional[random.Random] = None,
        secondary_player_ids: Optional[list] = None,
        **scenario_kwargs,
    ):
        self.definition = definition
        self.trainee_player_id = trainee_player_id
        self.phase = phase
        self.rng_reduction = rng_reduction
        # self.max_episode_s is the EFFECTIVE duration, resampled every
        # reset() (see that method) -- _base_max_episode_s is the configured
        # floor passed in here, never mutated after __init__.
        self._base_max_episode_s = max_episode_s
        self.max_episode_s = max_episode_s
        self._max_episode_s_variability_s = max_episode_s_variability_s
        self.linger_s = linger_s
        self.scenario_kwargs = scenario_kwargs
        self.rng = rng or random.Random()
        self.secondary_player_ids: list = secondary_player_ids or []

        # Set by the trainer after construction: trainer._sample_action.
        # When set, neural actions are applied to secondary players each step
        # and their transitions are stored in last_secondary_results.
        self.sample_action_fn = None
        # Optional Callable[[Player, Match], BCLabel] (e.g.
        # footballcoach.ai.ppo.bc.phase1_labels_for_player), threaded into
        # the TRAINEE's NeuralPlayerAI only (see reset() below) -- NOT
        # secondary players, matching existing behaviour (secondary-player
        # transitions have never carried a bc_label; see
        # always_compute_secondary_reward's own docstring on the adjacent
        # asymmetry between trainee/secondary handling). Must be set (like
        # sample_action_fn) BEFORE reset(), since that's where it gets
        # passed into NeuralPlayerAI's constructor. See NeuralPlayerAI.
        # bc_label_fn's own docstring for why this has to be computed
        # SYNCHRONOUSLY inside NeuralPlayerAI.act(), not by a caller after
        # env.step() returns.
        self.bc_label_fn = None
        # Opt-in, off by default: when True, last_secondary_results gets an
        # entry for EVERY configured secondary player every step, computed via
        # the same _compute_phase1_reward_for_player() the trainee/neural
        # secondary players use -- regardless of whether that player is
        # neural-driven. Normally (False) a secondary player only gets an
        # entry when it has a live NeuralPlayerAI transition (last_transition
        # is not None), because PPOTrainer._train() drains this list via
        # buffer.add(obs=sec["obs"], action=..., log_prob=sec["log_prob"], ...)
        # -- fields that ONLY exist on a real neural transition. Flipping this
        # on for that path would KeyError the instant a rules-AI/immobile
        # opponent episode hit this loop (50%+ of Phase 1 episodes). Entries
        # added because of this flag (no live transition) omit those neural-
        # only fields -- only {"reward", "done"} -- so this must stay False
        # unless the caller (record_demonstrations.py, which reads reward/done
        # only, never drains this into a PPO rollout buffer) explicitly opts
        # in by setting env.always_compute_secondary_reward = True.
        self.always_compute_secondary_reward = False

        cfg = load_ai_config()
        self._obs_cfg = cfg["observation"]
        self._reward_cfg = cfg["reward"]
        # Per-decision "could the kick / armed tackle have taken effect" flags
        # (entities/action_opportunity.py; agent_plans/masked_action_training_plan.md). Attached to
        # every neural transition's raw_exec (=> the batch's action/opp_* tensors) only when
        # something consumes them, so a run with both flags off is bit-identical to before.
        _ppo_cfg = cfg.get("ppo", {})
        self._opp_flags_enabled = bool(
            _ppo_cfg.get("log_action_opportunity_stats", False) or _ppo_cfg.get("mask_untriggered_actions", False)
        )
        self._opp_cross_head_cutoff = bool(_ppo_cfg.get("action_opportunity_cross_head_cutoff", True))
        self._dt_s = float(self._obs_cfg.get("sim_dt_s", 1.0 / 30.0))
        self._decision_interval_s = float(self._obs_cfg["decision_interval_s"])
        self._ticks_per_decision = max(1, round(self._decision_interval_s / self._dt_s))

        self._ema = EMAFilter.from_config()
        self._loop: Optional[ScenarioLoop] = None
        self._trial_done: bool = False
        self._episode_ticks: int = 0
        # Cache of per-team attacking-box bounds (rx_min, rx_max, half_box_w) for
        # _ball_dist_to_opponent_box() -- these depend only on pitch dims + team,
        # both fixed for the lifetime of one episode's Match, but that function is
        # called several times per step() for both trainee and secondary players.
        # Cleared in reset() since a fresh Match/pitch is built each episode.
        self._box_bounds_cache: dict = {}
        self._max_episode_ticks: int = 1
        # Who last possessed or kicked the ball, for attributing the
        # ball_out_penalty / "miss" outcome to the ONE player who actually put
        # the ball out. Read via self._loop.last_completed_trial_toucher_id
        # (ScenarioLoop is the single source of truth for this -- it tracks
        # it correctly-ordered, i.e. AFTER each match.step(), including the
        # trial-ending tick; this env used to keep its own separate, less
        # correctly-ordered copy, now removed).
        self._trainee_had_possession_last_step: bool = False
        # True while the ball is loose after the trainee lost it and no OTHER
        # player has taken possession yet (e.g. a push-kick dribble touch, or
        # a knock-loose the trainee immediately re-collects). The eventual
        # lost/gained counting is deferred until this resolves — see
        # _possession_transition_step() and ai/knowledge.md "Possession
        # gain/loss reward: real turnovers only" note.
        self._trainee_pending_loss: bool = False
        # Last player who actually settled possession of the ball (ignores
        # momentary loose-ball gaps, e.g. a player kicking the ball to
        # themselves). Used to distinguish a real turnover/steal (possession
        # resolves to a DIFFERENT player) from a harmless self-pass (ball goes
        # loose then comes back to the same player who kicked it) — see
        # ai/knowledge.md "Possession gain/loss reward" note.
        self._last_settled_ball_owner: Optional[str] = None
        self._prev_goal_count: tuple[int, int] = (0, 0)
        self._trainee_start_stamina: float = 1.0

        # Per-secondary-player state
        self._sec_start_stamina: dict = {}
        self._sec_ema: dict = {}
        self._sec_had_possession_last_step: dict = {}
        # Per-secondary-player pending-loss state, mirrors _trainee_pending_loss.
        self._sec_pending_loss: dict = {}
        # Populated after each step(); drained by PPOTrainer for the rollout buffer.
        # last_trainee_transition: dict with obs/action/log_prob/value/bc_label/raw_exec/illegal_action
        # last_secondary_results: list of same dicts for secondary neural players
        self.last_trainee_transition: Optional[dict] = None
        self.last_secondary_results: list = []
        self.last_reward_components: dict[str, float] = {}

        # Set by the trainer to enable saving match logs on bad episodes.
        # e.g. env.match_log_dir = trainer.checkpoint_dir / "match_logs"
        self.match_log_dir: Optional[Path] = None
        _log_cfg = cfg.get("logging", {})
        self._match_logger: MatchLogger = MatchLogger(
            consistency_interval_s=float(_log_cfg.get("consistency_interval_s", 2.5))
        )
        # Monotonically increasing episode counter used for log file naming.
        self._episode_index: int = 0
        # Dict of named trigger flags; each fires at most once per env lifetime.
        self._match_log_triggers_fired: dict[str, bool] = {}

    # -----------------------------------------------------------------------
    # Gym-like API
    # -----------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> ObservationBatch:
        """Start a new trial and return the initial observation.

        ``seed``, when given, is forwarded as the scenario builder's own
        ``seed`` kwarg (e.g. ``build_1v1_scenario(seed=...)``) -- makes the
        WHOLE trial (positions, attributes, ball state, and every subsequent
        physics RNG draw the returned Match's own ``rng`` makes) exactly
        reproducible, for builders that accept it. Has no effect on builders
        that don't accept a ``seed`` kwarg at all (e.g. phase 2's), same as
        ``sim_dt_s`` below. ``None`` (default) -- the scenario builder draws
        its own unseeded (OS-entropy) rng, matching prior behaviour exactly.
        """
        from footballcoach.rules_ai import NeuralPlayerAI, maybe_assign_neural_opponent
        # Inject sim_dt_s so build functions can pass it to Match.
        # This has no effect on builds that don't accept it (e.g. phase 2).
        build_kwargs = {**self.scenario_kwargs, "sim_dt_s": self._dt_s}
        if seed is not None:
            build_kwargs["seed"] = seed
        # terminal_outcomes: phase 1 only allows "miss"/"goal" to terminate
        # the loop; everything else (dispossessed, saved, box_possession,
        # timeout) is suppressed so the episode runs until the env's own
        # timeout or box-possession check fires. timeout_ticks=10**9 so the
        # loop's internal timeout never races the env's authoritative one.
        _terminal = frozenset({"miss", "goal"}) if self.phase == 1 else None
        self._loop = ScenarioLoop(
            definition=self.definition,
            max_trials=0,
            rng_reduction=self.rng_reduction,
            linger_s=self.linger_s,
            kwargs=build_kwargs,
            timeout_ticks=10**9,
            terminal_outcomes=_terminal,
        )
        # Resample this episode's EFFECTIVE duration before anything below
        # reads self.max_episode_s (_max_episode_ticks, the NeuralPlayerAI
        # construction further down, the time-remaining observation feature).
        # No rng draw at all when variability is 0 (the live default) -- see
        # max_episode_s_variability_s's own docstring.
        if self._max_episode_s_variability_s > 0.0:
            self.max_episode_s = self._base_max_episode_s + self.rng.uniform(
                0.0, self._max_episode_s_variability_s
            )
        else:
            self.max_episode_s = self._base_max_episode_s

        self._ema.reset()
        self._episode_ticks = 0
        self._trial_done = False
        self._box_bounds_cache = {}
        self._prev_goal_count = (
            self._loop.match.scoreboard.left_goals,
            self._loop.match.scoreboard.right_goals,
        )
        self._max_episode_ticks = max(1, int(self.max_episode_s / self._dt_s))
        self._trainee_had_possession_last_step = False
        self._trainee_pending_loss = False
        self._sec_had_possession_last_step = {pid: False for pid in self.secondary_player_ids}
        self._sec_pending_loss = {pid: False for pid in self.secondary_player_ids}
        self._last_settled_ball_owner = None
        # Record stamina at episode start for end-of-episode stamina penalty.
        # Also recorded per-secondary-player below (same treatment — see
        # _compute_phase1_reward_for_player()).
        # Per-decision-step tackle-attempt counter (see "tatt" in reward.py's
        # module docstring) -- Match.on_tackle fires synchronously inside
        # Match.step() (called from self._loop.step() in the tick loop
        # below), so this must be a persistent instance attribute the
        # callback can mutate, not a local var -- reset at the top of step()
        # itself (mirrors trainee_kicks_this_step's per-step reset, just as
        # an instance attribute since that one polls player.kicked_this_tick
        # directly instead of needing a callback).
        self._trainee_tackle_attempt_count = 0
        try:
            _trainee = self._loop.match.player_by_id(self.trainee_player_id)
            self._trainee_start_stamina = _trainee.stamina
            _trainee.on_tackle = self._on_trainee_tackle
        except KeyError:
            self._trainee_start_stamina = 1.0
        self.last_trainee_transition = None
        self.last_secondary_results = []
        self.last_terminal_match = None
        self.last_reward_components: dict[str, float] = {}

        # Assign NeuralPlayerAI to trainee (and secondary players) when a
        # sampling function is available.  Rules-based and immobile players
        # already have their ai set in the scenario build function.
        if self.sample_action_fn is not None:
            match = self._loop.match
            try:
                trainee = match.player_by_id(self.trainee_player_id)
                trainee.ai = NeuralPlayerAI(
                    self.sample_action_fn,
                    decision_interval_ticks=self._ticks_per_decision,
                    max_episode_s=self.max_episode_s,
                    ema_smoothed=self._ema.smoothed,
                    rng=self.rng,
                    bc_label_fn=self.bc_label_fn,
                )
            except KeyError:
                pass

            for pid in self.secondary_player_ids:
                if pid not in self._sec_ema:
                    self._sec_ema[pid] = EMAFilter.from_config()
                maybe_assign_neural_opponent(
                    match, pid, self.sample_action_fn,
                    decision_interval_ticks=self._ticks_per_decision,
                    max_episode_s=self.max_episode_s,
                    ema_smoothed=self._sec_ema[pid].smoothed,
                    rng=self.rng,
                )

        # Initialise per-secondary-player state
        self._sec_tackle_attempt_count: dict[str, int] = {pid: 0 for pid in self.secondary_player_ids}
        for pid in self.secondary_player_ids:
            if pid not in self._sec_ema:
                self._sec_ema[pid] = EMAFilter.from_config()
            self._sec_ema[pid].reset()
            # Record stamina at episode start for THIS secondary player, same
            # treatment as the trainee above — see
            # _compute_phase1_reward_for_player().
            try:
                _sec_player = self._loop.match.player_by_id(pid)
                self._sec_start_stamina[pid] = _sec_player.stamina
                _sec_player.on_tackle = self._make_on_sec_tackle(pid)
            except KeyError:
                self._sec_start_stamina[pid] = 1.0
        # Attach fresh logger to the match and record initial state.
        self._episode_index += 1
        self._match_logger.reset()
        self._loop.match.match_logger = self._match_logger
        self._match_logger.record_start(self._loop.match)
        return self._get_obs()

    def step(self, *, timers_already_advanced: bool = False) -> tuple[ObservationBatch, float, bool, StepInfo]:
        """Advance one decision interval (DECISION_INTERVAL_S sim-seconds).

        All player AI (neural and rules-based) fires automatically inside
        Match.step() via player.ai.act().  This method just ticks the sim,
        computes reward, and collects transition data from player.ai.

        ``timers_already_advanced``: pass ``True`` only when the caller
        already called ``match.advance_state_timers()`` for this env's
        upcoming tick before calling ``step()`` -- see
        ``Match.step(skip_state_timers=...)``'s own docstring. This env's
        FIRST internal tick then skips its own (would-be-duplicate)
        state-timer update; every later tick this call runs is unaffected
        (``Match.step()`` runs its own timer update normally for those).
        Only ``BatchedEnvGroup`` (ai/ppo/batched_rollout_worker.py) sets
        this -- it must encode a due decision's observation BEFORE calling
        ``step()``, which would otherwise make that decision from
        one-tick-stale timer state (e.g. stamina) relative to the identical
        unbatched path, where a decision is only ever made from INSIDE
        ``Match.step()``, after timers have already advanced for that tick.
        Default ``False``: unchanged behavior for every other caller.

        Returns:
            (observation, reward, done, info)
        """
        if self._loop is None:
            raise RuntimeError("Call reset() before step()")

        match = self._loop.match
        player = self._find_trainee(match)
        info = StepInfo()
        self.last_trainee_transition = None

        # Snapshot pre-tick state for secondary player transitions
        sec_pre: dict = {}
        for pid in self.secondary_player_ids:
            sec_pre[pid] = {
                "prev_ball_x": match.ball.position.x,
                "prev_box_dist": self._ball_dist_to_opponent_box(pid),
            }

        # --- Snapshot state before ticking ---
        prev_ball_x = match.ball.position.x
        prev_box_dist = self._ball_dist_to_opponent_box(self.trainee_player_id)
        prev_goal_count = self._prev_goal_count
        initial_scoreboard = (match.scoreboard.left_goals, match.scoreboard.right_goals)

        # --- Advance simulation for DECISION_INTERVAL_S ---
        trial_ended_this_step = False
        outcome_this_step: Optional[str] = None
        shot_taken = False
        shot_on_target = False
        goal_scored = False
        possession_lost_to_keeper = False

        # Per-tick possession-transition counters. Comparing only the state
        # before vs. after the whole decision interval misses events that
        # happen and reverse within the same interval (e.g. gain via tackle,
        # then immediately lose it again to the opponent before the next
        # decision tick) — scan every engine tick instead.
        _trainee_poss_prev = self._trainee_had_possession_last_step
        _trainee_pending_loss = self._trainee_pending_loss
        trainee_gained_count = 0
        trainee_lost_count = 0
        # Per-tick kick scan -- see StepInfo.trainee_kicks_this_step's own
        # comment for why a single post-loop read would silently miss kicks
        # that land on any tick but the last one of this decision interval.
        trainee_kicks_this_step = 0
        # Of those, how many fired via the ARMED path (Player.kick_armed was
        # still True the instant kick_with_direction/kicked_this_tick fired
        # -- see Player._finish_kick, which never touches kick_armed itself,
        # so it still reflects whichever branch of apply_nn_action.py's
        # if/else set it this exact tick: the armed branch leaves it True,
        # the instant-in-possession branch never touches it, so it's already
        # False from that tick's own reset). Gates "katt" (see reward.py's
        # own docstring) so an ordinary in-possession kick -- which never
        # paid any "karm" cost in the first place -- doesn't also collect
        # the bonus meant to offset that cost for a genuine, uncertain chase.
        trainee_armed_kicks_this_step = 0
        # Reset the tackle-attempt counters THIS on_tackle callback(s)
        # mutate synchronously inside the tick loop below (see
        # _on_trainee_tackle/_make_on_sec_tackle) -- instance attributes, not
        # locals, since the callback runs inside Match.step() with no direct
        # access to this method's stack.
        self._trainee_tackle_attempt_count = 0
        for pid in sec_pre:
            self._sec_tackle_attempt_count[pid] = 0
        _sec_poss_prev = {pid: self._sec_had_possession_last_step.get(pid, False) for pid in sec_pre}
        _sec_pending_loss = {pid: self._sec_pending_loss.get(pid, False) for pid in sec_pre}
        sec_gained_count = {pid: 0 for pid in sec_pre}
        sec_lost_count = {pid: 0 for pid in sec_pre}
        # Secondary-player kick scan -- same rationale/shape as the
        # trainee's own trainee_kicks_this_step/trainee_armed_kicks_this_step
        # above, just per-pid. Player objects fetched once here (not inside
        # the tick loop) to avoid a repeated match.player_by_id() lookup
        # every physics tick; a pid whose player has since left the match
        # (shouldn't happen mid-episode, but mirrors every other secondary
        # lookup's defensive try/except in this file) is simply skipped.
        sec_players: dict = {}
        for pid in sec_pre:
            try:
                sec_players[pid] = match.player_by_id(pid)
            except KeyError:
                pass
        sec_kicks_this_step = {pid: 0 for pid in sec_pre}
        sec_armed_kicks_this_step = {pid: 0 for pid in sec_pre}

        # Per-tick box-distance progress accumulators, and player<->ball
        # approach accumulators (2026-10-15 redesign). Both are gated on
        # PRE-tick last_touched_by_player_id -- the ball's OWN "whose play is
        # this" field (Match._set_possession's write, NEVER touched by a kick
        # itself -- see Ball.last_touched_by_player_id's own docstring) --
        # not live possession, and not at the end of the whole (multi-tick)
        # decision interval. Using the PRE-tick value (not end-of-interval)
        # matters for the same reason either way: a mid-interval tackle must
        # not attribute the OPPONENT's prior ball-carrying movement to the
        # trainee's "progress" the instant it steals the ball.
        #
        # "progress" (ball_dist_to_opponent_box shrinking) counts while
        # last_touched_by_player_id == this player -- true for live possession
        # AND for the flight of this player's OWN kick (a kick clears
        # possessed_by but never touches last_touched_by_player_id), so a
        # deliberate forward kick correctly earns progress credit for the
        # ground it covers, not just continuous dribble-carry.
        #
        # "approach" (player<->ball distance shrinking) counts while
        # last_touched_by_player_id != this player -- i.e. only while chasing
        # a ball someone ELSE last touched. This is the fix for the mirror-
        # image problem: with the OLD unconditional formula, a player's own
        # kick (which necessarily separates them from the ball, at least
        # momentarily) registered as a self-inflicted "approach" PENALTY, not
        # recoverable within the episode if they never fully closed the gap
        # again before episode end. Freezing "approach" while in control
        # (including mid-kick-flight) means a deliberate kick costs nothing
        # here and only "progress" (correctly) accounts for it.
        _last_toucher_prev = match.ball.last_touched_by_player_id
        _trainee_box_dist_prev = prev_box_dist
        trainee_prog_accum = 0.0
        _trainee_ball_dist_prev = self._ball_dist_for_player(self.trainee_player_id, match)
        trainee_appr_accum = 0.0
        _sec_box_dist_prev = {pid: pre["prev_box_dist"] for pid, pre in sec_pre.items()}
        sec_prog_accum = {pid: 0.0 for pid in sec_pre}
        _sec_ball_dist_prev = {pid: self._ball_dist_for_player(pid, match) for pid in sec_pre}
        sec_appr_accum = {pid: 0.0 for pid in sec_pre}

        for _tick_idx in range(self._ticks_per_decision):
            # Track shot events (KickOrder completing toward goal)
            if self._detect_shot_this_tick(match, player):
                shot_taken = True

            # Only the FIRST tick of this call is ever the one a caller
            # could have pre-advanced timers for (that's the only tick a
            # precomputed decision is waiting on) -- every later tick here
            # runs Match.step()'s own state-timer update normally.
            tick_done = self._loop.step(
                skip_state_timers=(timers_already_advanced and _tick_idx == 0)
            )
            self._episode_ticks += 1
            if player.kicked_this_tick:
                trainee_kicks_this_step += 1
                if player.kick_armed:
                    trainee_armed_kicks_this_step += 1
            for _pid, _sec_p in sec_players.items():
                if _sec_p.kicked_this_tick:
                    sec_kicks_this_step[_pid] += 1
                    if _sec_p.kick_armed:
                        sec_armed_kicks_this_step[_pid] += 1

            if tick_done:
                # Break BEFORE reading any match state. When linger_s=0
                # (training), _loop.step() rebuilds the match before returning
                # True — self._loop.match is already the next episode here.
                outcome_this_step = self._latest_outcome()
                if outcome_this_step == "goal":
                    goal_scored = True
                    if self.phase != 1:
                        self._ema.on_goal()
                    else:
                        # Phase 1's only win condition is box_possession
                        # (dribble into the opponent's box) -- there is no
                        # scoring reward term at all. Normalize "goal" to
                        # "miss" HERE, before anything downstream branches on
                        # outcome_this_step, so a goal is treated exactly
                        # like any other ball-left-the-pitch event: the
                        # reward's toucher-based fault check (ball_went_out /
                        # sec_ball_went_out below) and the invalid-vs-miss
                        # split (remap_phase1_outcome, no toucher -> nobody's
                        # fault) both apply uniformly, with no separate
                        # special-casing needed. Previously this normalization
                        # only happened much later, inside
                        # remap_phase1_outcome() (label only) -- the reward
                        # calculation below read the un-normalized "goal"
                        # directly, so a goal never triggered the ball-out
                        # penalty even though the episode was labelled
                        # "ball_out" downstream. See ai_trainer_knowledge.md.
                        outcome_this_step = "miss"
                trial_ended_this_step = True
                break

            # Early-exit when the trainee is already in the opponent box with possession —
            # any further ticks only accumulate spurious negative progress as box_dist=0.
            if (_trainee_poss_prev and match.pitch.is_in_box(
                match.ball.position, left=(player.team == Team.RIGHT)
            )):
                break

            # Accumulate progress/approach using PRE-tick last_touched_by (the
            # snapshot from BEFORE this tick's Match.step() call above) --
            # see the accumulator-init comment above for why this is gated on
            # last_touched_by rather than live possession, and why PRE-tick.
            # Pass local `match` to all distance helpers — never self._loop.match.
            _curr_trainee_box_dist = self._ball_dist_to_opponent_box(self.trainee_player_id, match)
            _curr_trainee_ball_dist = self._ball_dist_for_player(self.trainee_player_id, match)
            if _last_toucher_prev == self.trainee_player_id:
                trainee_prog_accum += _trainee_box_dist_prev - _curr_trainee_box_dist
            else:
                trainee_appr_accum += _trainee_ball_dist_prev - _curr_trainee_ball_dist
            _trainee_box_dist_prev = _curr_trainee_box_dist
            _trainee_ball_dist_prev = _curr_trainee_ball_dist
            for pid in sec_pre:
                _curr_sec_box_dist = self._ball_dist_to_opponent_box(pid, match)
                _curr_sec_ball_dist = self._ball_dist_for_player(pid, match)
                if _last_toucher_prev == pid:
                    sec_prog_accum[pid] += _sec_box_dist_prev[pid] - _curr_sec_box_dist
                else:
                    sec_appr_accum[pid] += _sec_ball_dist_prev[pid] - _curr_sec_ball_dist
                _sec_box_dist_prev[pid] = _curr_sec_box_dist
                _sec_ball_dist_prev[pid] = _curr_sec_ball_dist
            # Refresh for the NEXT tick's gating -- AFTER this tick's
            # accumulation used the pre-tick value above. Single shared read
            # (not a per-player derived flag like _trainee_poss_prev), since
            # last_touched_by_player_id is one ball-level field every
            # player_id is just compared against.
            _last_toucher_prev = match.ball.last_touched_by_player_id

            # An armed-kick auto-fire (Match._update_loose_ball_pickup)
            # grants possession then immediately kicks it away again WITHIN
            # this same tick (see Player.kick_armed_direction's docstring) --
            # a real, if momentary, possession event that the single
            # end-of-tick possessed_by sample below can never observe on its
            # own (it reads None both before and after this tick). Detected
            # via the same kicked_this_tick+kick_armed combination
            # trainee_armed_kicks_this_step already checks above:
            # _finish_kick never touches kick_armed, so it still reads True
            # here iff THIS tick's kick came from the armed path (never true
            # for an already-possessing instant kick, since
            # apply_nn_action.py's kick branch is exclusive -- that case
            # needs no special handling, since poss_prev was already True
            # from an earlier tick and the real end-of-tick sample below
            # correctly sees the release). Feeds a synthetic
            # "possessed_by=this player" sample through
            # _possession_transition_step FIRST (registering the gain),
            # before the real end-of-tick sample below (registering the
            # subsequent loss/pending-loss) -- reuses the state machine
            # completely unchanged, just calls it one extra time for the
            # tick a touch happened that the final poll would otherwise miss.
            if player.kicked_this_tick and player.kick_armed:
                (
                    _trainee_poss_prev, _trainee_pending_loss,
                    trainee_gained_count, trainee_lost_count,
                ) = self._possession_transition_step(
                    self.trainee_player_id, self.trainee_player_id, _trainee_poss_prev,
                    _trainee_pending_loss, trainee_gained_count, trainee_lost_count,
                )
            for pid in sec_pre:
                _sec_p = sec_players.get(pid)
                if _sec_p is not None and _sec_p.kicked_this_tick and _sec_p.kick_armed:
                    (
                        _sec_poss_prev[pid], _sec_pending_loss[pid],
                        sec_gained_count[pid], sec_lost_count[pid],
                    ) = self._possession_transition_step(
                        pid, pid, _sec_poss_prev[pid],
                        _sec_pending_loss[pid], sec_gained_count[pid], sec_lost_count[pid],
                    )

            # Update possession transitions AFTER progress, so gain-tick
            # motion counts on the NEXT tick, not the tick possession is taken.
            _possessed_by = match.ball.possessed_by
            (
                _trainee_poss_prev,
                _trainee_pending_loss,
                trainee_gained_count,
                trainee_lost_count,
            ) = self._possession_transition_step(
                self.trainee_player_id, _possessed_by, _trainee_poss_prev,
                _trainee_pending_loss, trainee_gained_count, trainee_lost_count,
            )

            for pid in sec_pre:
                (
                    _sec_poss_prev[pid],
                    _sec_pending_loss[pid],
                    sec_gained_count[pid],
                    sec_lost_count[pid],
                ) = self._possession_transition_step(
                    pid, _possessed_by, _sec_poss_prev[pid],
                    _sec_pending_loss[pid], sec_gained_count[pid], sec_lost_count[pid],
                )

        # Update EMA with the trainee's attack/defence output (from NeuralPlayerAI transition)
        if self.last_trainee_transition is not None:
            decision = self.last_trainee_transition.get("action")
            if decision is not None and hasattr(decision, "attack_defence_raw"):
                self._ema.update(float(decision.attack_defence_raw), self._decision_interval_s)
            info.illegal_action = self.last_trainee_transition.get("illegal_action", False)

        # --- Compute reward ---
        new_goal_count = (match.scoreboard.left_goals, match.scoreboard.right_goals)
        self._prev_goal_count = new_goal_count

        # Ball went out and the trainee was the last player to touch it
        # (possession or kick) — only the last toucher is ever penalised.
        # self._loop.last_completed_trial_toucher_id is correctly-ordered
        # (updated after every match.step(), including the trial-ending
        # tick) and safe to read here since trial_ended_this_step short-
        # circuits this whole expression to exactly the case where
        # ScenarioLoop.step() just called _start_trial(), which stashes it.
        ball_went_out = (
            trial_ended_this_step
            and outcome_this_step == "miss"
            and self._loop.last_completed_trial_toucher_id == self.trainee_player_id
        )

        # Ball progress toward opponent BOX (not raw goal-line x) — counts
        # lateral movement into the box. Per-tick accumulated (see
        # trainee_prog_accum init comment above), NOT a single
        # start-vs-end-of-interval delta.
        ball_progress = trainee_prog_accum

        # Reached opponent box with possession (phase 1 terminal — trainee
        # wins) / opponent reached trainee's box with possession (trainee
        # loses). Single shared detector (ai/env/outcome.py's
        # detect_phase1_box_terminal, gated via can_score_box_terminal so a
        # player with no real controller — e.g. an immobile opponent sitting
        # in its own box by pure spawn coincidence — can never trigger this)
        # -- also used by ScenarioLoop/detect_trial_outcome for the UI/debug-
        # script path (see ScenarioLoop._trial_outcome), so the two can never
        # independently drift apart again. This used to be duplicated inline
        # here with a SEPARATE, ungated, trainee-unaware version living in
        # detect_trial_outcome's generic box-possession check -- confirmed to
        # genuinely disagree on real episodes (a UI/debug-script playback
        # could declare a winner before the event eval's own check correctly
        # waited for had even happened).
        _phase1_terminal = detect_phase1_box_terminal(match, trainee_player_id=self.trainee_player_id)
        box_terminal = _phase1_terminal == "box_possession"
        opponent_box_terminal = _phase1_terminal == "opponent_box_possession"
        # Use the per-tick scan (trainee_gained_count/trainee_lost_count) instead
        # of a simple before/after comparison, so gain/lose transitions that
        # both happen within this single decision interval (e.g. tackle then
        # immediately re-tackled) are not silently dropped from the reward.
        trainee_has_possession_now = _trainee_poss_prev  # final state after the tick loop
        gained_possession = trainee_gained_count  # int count, not just bool
        lost_possession = trainee_lost_count
        self._trainee_had_possession_last_step = trainee_has_possession_now
        self._trainee_pending_loss = _trainee_pending_loss

        timeout = self._episode_ticks >= int(self.max_episode_s / self._dt_s)
        if self.phase == 1:
            # Routes through the SINGLE shared _compute_phase1_reward_for_player()
            # method (see its docstring) — trainee and every secondary player
            # call the exact same code path.
            reward, self.last_reward_components = self._compute_phase1_reward_for_player(
                player_obj=player,
                ball_went_out_after_touch=ball_went_out,
                reached_opponent_box_with_possession=box_terminal,
                opponent_reached_trainee_box=opponent_box_terminal,
                timed_out=timeout and not box_terminal and not opponent_box_terminal,
                player_ball_approach_delta_m=trainee_appr_accum,
                ball_progress_toward_goal_m=ball_progress,
            )
        else:
            reward = phase2_reward(
                shot_taken_this_step=shot_taken,
                ticks_since_episode_start=self._episode_ticks,
                max_episode_ticks=int(self.max_episode_s * 30),
                shot_on_target=shot_on_target,
                goal_scored=goal_scored,
                illegal_action_attempted=info.illegal_action,
                possession_lost_to_keeper=possession_lost_to_keeper,
                cfg=self._reward_cfg["phase2"],
            )

        # --- Done? ---
        any_box_terminal = box_terminal or opponent_box_terminal
        done = trial_ended_this_step or timeout or any_box_terminal

        # Stashed BEFORE the secondary-player loop below (which additively
        # merges secondary components into self.last_reward_components under
        # the same short keys) -- see _check_match_log_triggers()'s
        # "ball_out_nonneg" trigger, which needs the TRAINEE's own "out"
        # component in isolation, not summed with any secondary player's.
        _trainee_out_component = self.last_reward_components.get("out", 0.0)

        if done:
            if self.phase == 1:
                outcome_label = remap_phase1_outcome(
                    outcome_this_step or "",
                    last_ball_toucher_id=self._loop.last_completed_trial_toucher_id,
                    box_terminal=box_terminal,
                    opponent_box_terminal=opponent_box_terminal,
                    timeout=timeout,
                )
            else:
                outcome_label = outcome_this_step or ""
            info.trial_outcome = outcome_label

        info.is_rules_episode = getattr(match, "_opponent_use_rules_ai", False)
        info.is_immobile_episode = getattr(match, "_opponent_is_immobile", False)
        info.ticks_elapsed = self._episode_ticks
        info.trainee_kicks_this_step = trainee_kicks_this_step
        info.trainee_armed_kicks_this_step = trainee_armed_kicks_this_step

        # --- Collect trainee transition from NeuralPlayerAI ---
        if hasattr(player, "ai") and player.ai is not None and hasattr(player.ai, "last_transition"):
            self.last_trainee_transition = player.ai.last_transition
            if self._opp_flags_enabled and self.last_trainee_transition is not None:
                self.last_trainee_transition = self._with_opportunity_flags(
                    self.last_trainee_transition, player, done,
                )

        # --- Collect secondary player transitions from their NeuralPlayerAI ---
        self.last_secondary_results = []
        for pid, pre in sec_pre.items():
            try:
                sec_player = match.player_by_id(pid)
            except KeyError:
                continue
            _sec_has_transition = (
                hasattr(sec_player, "ai") and sec_player.ai is not None
                and hasattr(sec_player.ai, "last_transition")
                and sec_player.ai.last_transition is not None
            )
            if not _sec_has_transition and not self.always_compute_secondary_reward:
                continue

            # Box-distance closed / player<->ball distance closed this step,
            # matching the trainee's ball_progress/trainee_appr_accum above --
            # both per-tick accumulated and last_touched_by-gated (see the
            # accumulator-init comment above), not a single start-vs-end-of-
            # interval delta.
            sec_ball_prog = sec_prog_accum.get(pid, 0.0)
            sec_ball_appr = sec_appr_accum.get(pid, 0.0)

            sec_ball_went_out = (
                trial_ended_this_step
                and outcome_this_step == "miss"
                and self._loop.last_completed_trial_toucher_id == pid
            )
            # Per-tick scan (sec_gained_count/sec_lost_count) instead of a
            # simple before/after comparison — see trainee note above.
            sec_has_poss_now = _sec_poss_prev[pid]  # final state after the tick loop
            sec_gained_poss = sec_gained_count[pid]  # int count
            sec_lost_poss = sec_lost_count[pid]
            self._sec_had_possession_last_step[pid] = sec_has_poss_now
            self._sec_pending_loss[pid] = _sec_pending_loss[pid]
            sec_in_atk_box = match.pitch.is_in_box(
                match.ball.position,
                left=(sec_player.team == Team.RIGHT),
            )
            # Gated via the SAME shared can_score_box_terminal() used by
            # box_terminal/opponent_box_terminal above -- note
            # sec_box_terminal does NOT feed into the real episode-ending
            # `done` below, so leaving it ungated (as it was before this
            # helper existed) let the one-time box_possession_terminal/
            # speed_bonus reward re-fire every tick for as long as an
            # immobile secondary player happened to be sitting where the
            # ball settled (confirmed: 64 consecutive ticks in one real
            # recorded episode) instead of once.
            sec_box_terminal = (
                sec_in_atk_box
                and match.ball.possessed_by == pid
                and can_score_box_terminal(sec_player)
            )

            if self.phase == 1:
                # Routes through the SAME shared _compute_phase1_reward_for_player()
                # method used by the trainee above — see its docstring. This
                # is what guarantees secondary players get identical
                # treatment to the trainee; do not revert to calling
                # phase1_reward() directly here.
                sec_reward, _sec_comps = self._compute_phase1_reward_for_player(
                    player_obj=sec_player,
                    ball_went_out_after_touch=sec_ball_went_out,
                    reached_opponent_box_with_possession=sec_box_terminal,
                    opponent_reached_trainee_box=box_terminal,  # from sec's POV, trainee winning = sec losing
                    timed_out=timeout and not sec_box_terminal and not box_terminal,
                    player_ball_approach_delta_m=sec_ball_appr,
                    ball_progress_toward_goal_m=sec_ball_prog,
                )
            else:
                sec_reward = 0.0
                _sec_comps = {}

            _sec_transition = sec_player.ai.last_transition if _sec_has_transition else {}
            if _sec_has_transition and self._opp_flags_enabled:
                _sec_transition = self._with_opportunity_flags(_sec_transition, sec_player, done)
            self.last_secondary_results.append({
                **_sec_transition,
                "player_id": pid,
                "reward": sec_reward,
                "done": 1.0 if done else 0.0,
                # The episode-ending outcome is a single shared, env-level
                # fact (one trial ends the same way for every player on the
                # pitch at once) -- NOT player-specific, so this is just
                # info.trial_outcome (already computed above, "" when this
                # tick isn't terminal). Without this, every secondary row's
                # step_outcome silently defaulted to "" via
                # RolloutBuffer.add()'s default (only the trainee's own
                # buffer.add() call threaded step_outcome through), and
                # _backfill_step_outcomes -- which operates on the flat,
                # track-unaware dones/step_outcomes lists -- can never
                # recover a done=1 row whose OWN outcome slot is empty. Every
                # secondary player's terminal-tick row therefore fell into
                # value_mse_by_outcome()'s "unknown" bucket forever, despite
                # the backfill correctly covering every earlier row of the
                # same episode. See ai/knowledge.md "unknown outcome bucket".
                "step_outcome": info.trial_outcome or "",
                # This player's OWN component breakdown -- kept separate
                # from self.last_reward_components (the trainee's own),
                # NOT merged into it. Merging used to make e.g. a "win"
                # episode's trainee row carry BOTH the trainee's own
                # box_possession_terminal bonus AND the losing secondary
                # player's own loss_terminal penalty added together,
                # silently corrupting every per-component diagnostic (and,
                # for self-play PPO training with a neural secondary
                # player, the trainee's own stored rollout-buffer
                # reward_comps too -- see ppo_trainer.py's `reward_comps=
                # dict(getattr(env, "last_reward_components", {}))`).
                # Callers that want each player's own row-level components
                # (e.g. record_demonstrations.py) must read them from here.
                "reward_components": dict(_sec_comps),
            })
        _match_time_s = match.time_s
        _ball_pos = match.ball.position
        if done:
            self._match_logger.notify_episode_end(
                _match_time_s, _ball_pos,
                info.trial_outcome or "unknown",
                reward, self.last_reward_components,
            )
        else:
            self._match_logger.accumulate_reward(reward, self.last_reward_components)
        if self.phase == 1 and self.match_log_dir is not None:
            self._check_match_log_triggers(
                done=done,
                outcome_label=info.trial_outcome if done else None,
                trainee_out_component=_trainee_out_component,
            )
        # The Match exactly as it stood on the TRUE final tick of the trial
        # that just ended. Two different done-paths need two different
        # sources here:
        # - trial_ended_this_step=True: ScenarioLoop._trial_outcome() itself
        #   detected the end (miss/goal/box_possession/loop-level timeout)
        #   and already rebuilt self._loop.match for the NEXT trial before
        #   step() returns -- last_completed_trial_match is the only place
        #   that final state survives (see its own docstring).
        # - trial_ended_this_step=False but done=True anyway (this env's OWN
        #   independent box_terminal/opponent_box_terminal/timeout checks,
        #   computed above from THIS tick's match state, separately from
        #   ScenarioLoop's own outcome detection): nothing rebuilt anything
        #   this step, so self._loop.match is still exactly the terminal
        #   state itself -- last_completed_trial_match would be stale (the
        #   PREVIOUS trial's end, or None on this env's very first episode).
        if not done:
            self.last_terminal_match = None
        elif trial_ended_this_step:
            self.last_terminal_match = self._loop.last_completed_trial_match
        else:
            self.last_terminal_match = self._loop.match
        return self._get_obs(), reward, done, info

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _check_match_log_triggers(
        self, done: bool = False, outcome_label: Optional[str] = None,
        trainee_out_component: float = 0.0,
    ) -> None:
        """Save a named log file the first time each trigger condition fires."""
        triggers: dict[str, bool] = {
            # Diagnostic for the "ball_out episode with a non-negative
            # trainee reward" investigation (see ai_trainer_knowledge.md) --
            # outcome_label == "miss" is the PRE-remap-to-"invalid" raw label
            # (i.e. classify_outcome() would report this episode as
            # "ball_out"), but the trainee's own "out" reward component only
            # fires when the trainee was recorded as the LAST BALL TOUCHER
            # (see ball_went_out in this file) -- those two can disagree
            # (e.g. the ball passes within pickup_radius_m of the opponent,
            # silently reassigning last_completed_trial_toucher_id, right
            # before the ball leaves the pitch). Saving the full match log
            # whenever they disagree lets us inspect exactly what happened
            # on the terminal tick instead of guessing from aggregate stats.
            "ball_out_nonneg": (
                done and outcome_label == "miss" and trainee_out_component >= 0.0
            ),
        }
        for name, fired in triggers.items():
            if fired and not self._match_log_triggers_fired.get(name):
                self._match_log_triggers_fired[name] = True
                self._match_logger.save(
                    self.match_log_dir / f"episode_{self._episode_index:06d}_{name}.json"
                )

    def _get_obs(self, player_id: str | None = None, match: "Match | None" = None) -> ObservationBatch:
        """Encode the observation for *player_id* (default: the trainee).

        Passing an explicit ``player_id`` lets callers (e.g.
        record_demonstrations.py) encode observations for other players
        (the opponent) without disturbing any of the trainee-specific
        internal call sites, which all call this with no arguments.

        Passing an explicit ``match`` (default: ``self._loop.match``, the
        live match) lets a caller encode against a DIFFERENT match object --
        specifically ``self.last_terminal_match`` (see its own docstring),
        the true final tick of a just-ended trial, which ``self._loop.match``
        itself can no longer represent by the time ``step()`` has returned.
        """
        time_remaining = max(
            0.0,
            self.max_episode_s - self._episode_ticks * self._dt_s
        )
        return encode_observation(
            match=match if match is not None else self._loop.match,
            player_id=player_id if player_id is not None else self.trainee_player_id,
            time_remaining_s=time_remaining,
            attack_defence_smoothed=self._ema.smoothed,
            rng=self.rng,
            phase=self.phase,
        )

    def _ball_dist_to_trainee(self, match=None) -> float:
        return self._ball_dist_for_player(self.trainee_player_id, match)

    # can_score_box_terminal moved to ai/env/outcome.py (shared with
    # ScenarioLoop/detect_trial_outcome, see that module's docstring) --
    # imported at module level above.

    def _compute_phase1_reward_for_player(
        self,
        *,
        player_obj,
        ball_went_out_after_touch: bool,
        reached_opponent_box_with_possession: bool,
        opponent_reached_trainee_box: bool,
        timed_out: bool,
        player_ball_approach_delta_m: float = 0.0,
        ball_progress_toward_goal_m: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        """SINGLE call site for phase1_reward(), used for the trainee AND
        every secondary player -- route through this method instead of
        calling phase1_reward() directly, so a future per-player-derived
        term can't silently diverge between the trainee and secondary
        players again (see ai/knowledge.md "Reward parity" note). This is
        also what guarantees "appr"/"prog" treat the trainee and a flipped-
        side opponent identically: both route through this one method, and
        player_ball_approach_delta_m/ball_progress_toward_goal_m are always
        resolved and last_touched_by-gated per player_id by the caller (see
        the accumulator-init comment in step()), never hardcoded to one side
        of the pitch.
        """
        # last_desired_speed_mode (NOT desired_speed_mode, which Match._apply_movement
        # clears every tick once consumed) is the "what did they actually just do"
        # signal used elsewhere for the same reason -- see ai/obs/encoder.py's
        # identical convention for the live desired-speed-mode observation feature.
        _is_sprinting = player_obj.last_desired_speed_mode is SpeedMode.SPRINT
        return phase1_reward(
            ball_went_out_after_touch=ball_went_out_after_touch,
            reached_opponent_box_with_possession=reached_opponent_box_with_possession,
            cfg=self._reward_cfg["phase1"],
            time_fraction_remaining=1.0 - self._episode_ticks / self._max_episode_ticks,
            opponent_reached_trainee_box=opponent_reached_trainee_box,
            timed_out=timed_out,
            is_sprinting=_is_sprinting,
            decision_interval_s=self._decision_interval_s,
            player_ball_approach_delta_m=player_ball_approach_delta_m,
            ball_progress_toward_goal_m=ball_progress_toward_goal_m,
        )

    def _ball_dist_to_opponent_box(self, player_id: Optional[str] = None, match=None) -> float:
        """2D shortest distance from the ball to *player_id*'s attacking box.

        Returns 0.0 when the ball is inside the box. On the wings the
        distance to the nearest box corner/edge is used rather than x-only,
        so the normalization constant and progress reward are consistent with
        the actual path the player needs to travel.

        Always pass `match` explicitly inside step() — self._loop.match is
        replaced by the next episode before step() returns on terminal ticks.
        """
        try:
            m = match if match is not None else self._loop.match
            ball = m.ball.position
            player = m.player_by_id(player_id if player_id is not None else self.trainee_player_id)
            bounds = self._box_bounds_cache.get(player.team)
            if bounds is None:
                # Depends only on pitch dims + team, both fixed for this episode's
                # Match -- computed once per team per episode, not once per call.
                pitch = m.pitch
                half_box_w = pitch.box_width_m / 2.0
                if player.team == Team.LEFT:
                    rx_min = pitch.half_length - pitch.box_length_m
                    rx_max = pitch.half_length
                else:
                    rx_min = -pitch.half_length
                    rx_max = -pitch.half_length + pitch.box_length_m
                bounds = (rx_min, rx_max, half_box_w)
                self._box_bounds_cache[player.team] = bounds
            rx_min, rx_max, half_box_w = bounds
            # Shortest distance from ball to axis-aligned rectangle.
            dx = max(rx_min - ball.x, 0.0, ball.x - rx_max)
            dy = max(-half_box_w - ball.y, 0.0, ball.y - half_box_w)
            return float(math.hypot(dx, dy))
        except (KeyError, AttributeError):
            return 1.0

    def _ball_dist_for_player(self, player_id: str, match=None) -> float:
        try:
            m = match if match is not None else self._loop.match
            player = m.player_by_id(player_id)
            ball = m.ball
            return math.hypot(
                ball.position.x - player.position.x,
                ball.position.y - player.position.y,
            )
        except (KeyError, AttributeError):
            return 0.0

    def _encode_obs_for_player(self, player_id: str) -> "ObservationBatch":
        time_remaining = max(0.0, self.max_episode_s - self._episode_ticks * self._dt_s)
        sec_ema = self._sec_ema.get(player_id, self._ema)
        return encode_observation(
            match=self._loop.match,
            player_id=player_id,
            time_remaining_s=time_remaining,
            attack_defence_smoothed=sec_ema.smoothed,
            rng=self.rng,
            phase=self.phase,
        )

    @staticmethod
    def _possession_transition_step(
        pid: str,
        possessed_by: Optional[str],
        poss_prev: bool,
        pending_loss: bool,
        gained_count: int,
        lost_count: int,
    ) -> tuple[bool, bool, int, int]:
        """Advance one engine tick of possession-transition tracking for *pid*.

        Distinguishes a "real" turnover (possession is confirmed settled onto
        a DIFFERENT player) from a harmless momentary loose ball (e.g. a
        push-kick dribble touch, or *pid* knocking the ball loose and
        immediately re-collecting it themselves) — see ai/knowledge.md
        "Possession gain/loss reward: real turnovers only".

        Returns (poss_prev, pending_loss, gained_count, lost_count) updated
        for this tick.
        """
        now = possessed_by == pid
        if now:
            if pending_loss:
                # Regained directly with no OTHER player ever taking it in
                # between -> the ball was never really lost; cancel silently.
                pending_loss = False
            elif not poss_prev:
                gained_count += 1
            poss_prev = True
        else:
            if poss_prev:
                # Just lost it this tick.
                if possessed_by is not None:
                    # Someone else already holds it this very tick -> confirmed turnover.
                    lost_count += 1
                    pending_loss = False
                else:
                    # Ball is loose; defer counting until it's resolved.
                    pending_loss = True
                poss_prev = False
            elif pending_loss and possessed_by is not None:
                # Ball was loose and has now settled onto someone else -> confirmed turnover.
                lost_count += 1
                pending_loss = False
        return poss_prev, pending_loss, gained_count, lost_count

    def _find_trainee(self, match: Match):
        return match.player_by_id(self.trainee_player_id)

    def _with_opportunity_flags(self, transition: dict, player, done: bool) -> dict:
        """Copy of a NeuralPlayerAI ``last_transition`` whose ``raw_exec`` also carries the
        decision interval's opportunity flags (``entities/action_opportunity.OPP_ACTION_KEYS``),
        so they ride through ``_action_to_numpy`` -> ``RolloutBuffer`` -> the batch's
        ``action/*`` tensors exactly like the sampled action does (merge / chunk flush / replay /
        augment tiling all handle ``action/*`` generically).

        The flags describe the interval opened by this transition's decision: read here, at the
        end of ``step()``, which normally spans exactly that interval. ``opp_partial`` marks the
        exceptions -- an interval cut short by an early exit that is NOT the end of the episode
        (the next ``step()`` starts mid-interval), whose flags are therefore incomplete; counted
        so the size of that exception is visible instead of assumed. A copy, never an in-place
        edit: on such a not-due step ``last_transition`` is still the previous decision's dict.
        """
        f = player.opportunity.flags(player.kick_count, player.tackle_fire_count, self._opp_cross_head_cutoff)
        ai = player.ai
        partial = (not done) and hasattr(ai, "is_due_for_decision") and not ai.is_due_for_decision()

        def _arr(x) -> np.ndarray:
            return np.array([1.0 if x else 0.0], dtype=np.float32)

        raw_exec = dict(transition["raw_exec"])
        raw_exec.update(
            opp_kick=_arr(f.kick_opp), opp_kick_raw=_arr(f.kick_opp_raw), fired_kick=_arr(f.kick_fired),
            opp_tack=_arr(f.tack_opp), opp_tack_raw=_arr(f.tack_opp_raw), fired_tack=_arr(f.tack_fired),
            opp_partial=_arr(partial),
        )
        out = dict(transition)
        out["raw_exec"] = raw_exec
        return out

    def _on_trainee_tackle(self, player) -> None:  # noqa: ANN001
        """Player.on_tackle callback -- fires synchronously inside
        Match.step() the instant an INTENTIONAL (armed) tackle resolves
        contact, before the win/lose result is known (see
        Match._attempt_tackle_contact). Never fires for the collision-based
        auto-tackle fallback (Match._check_head_on_tackles uses
        on_auto_tackle_result instead) -- see reward.py's "tatt" docstring."""
        self._trainee_tackle_attempt_count += 1

    def _make_on_sec_tackle(self, pid: str):
        """Same as _on_trainee_tackle, for a secondary player -- a closure
        per pid since Player.on_tackle takes no player_id argument of its
        own (the callback IS bound to one specific Player object already,
        but this env tracks counts per pid in a shared dict)."""
        def _cb(player) -> None:  # noqa: ANN001
            self._sec_tackle_attempt_count[pid] = self._sec_tackle_attempt_count.get(pid, 0) + 1
        return _cb

    def _latest_outcome(self) -> Optional[str]:
        if self._loop is None:
            return None
        outcomes = self._loop.outcomes
        # The most recent outcome is whichever count just incremented
        # (ScenarioLoop stores cumulative counts)
        return max(outcomes, key=outcomes.get) if any(outcomes.values()) else None

    def _detect_shot_this_tick(self, match: Match, player) -> bool:
        """Rough heuristic: player just kicked the ball (it's now loose and
        moving in the opponent's goal direction)."""
        from footballcoach.orders import KickOrder, ShootOrder
        order = player.current_order
        if isinstance(order, (KickOrder, ShootOrder)):
            from footballcoach.orders import OrderStatus
            if getattr(order, "status", None) == OrderStatus.COMPLETE:
                return True
        return False

    @property
    def match(self) -> Optional[Match]:
        return self._loop.match if self._loop else None
