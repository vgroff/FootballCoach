"""ScenarioEnv: Gym-like wrapper around ScenarioDefinition + ScenarioLoop.

See ai_design_doc.md section 9.2 for the design rationale.

All players (trainee, secondary, rules-based) carry their AI in ``player.ai``.
``Match.step()`` calls ``ai.act()`` automatically every physics tick.

For neural players, ``ScenarioEnv`` assigns ``NeuralPlayerAI`` on ``reset()``.
After each ``step()``, ``env.last_trainee_transition`` holds the PPO rollout
data for the trainee; ``env.last_secondary_results`` holds the same for
any secondary neural players.

API:
    env = ScenarioEnv(definition, trainee_player_id="kicker", phase=1)
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
from footballcoach.ai.env.outcome import remap_phase1_outcome
from footballcoach.ai.env.reward import EMAFilter, phase1_reward, phase2_reward
from footballcoach.ai.obs.encoder import MAX_OTHER_PLAYERS, encode_observation
from footballcoach.ai.obs.schema import ObservationBatch
from footballcoach.engine.match import Match
from footballcoach.engine.match_logger import MatchLogger
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
        max_episode_s: Maximum episode duration in sim-seconds.
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
        phase: int = 1,
        rng_reduction: float = 0.3,
        max_episode_s: float = 120.0,
        linger_s: float = 0.0,
        rng: Optional[random.Random] = None,
        secondary_player_ids: Optional[list] = None,
        **scenario_kwargs,
    ):
        self.definition = definition
        self.trainee_player_id = trainee_player_id
        self.phase = phase
        self.rng_reduction = rng_reduction
        self.max_episode_s = max_episode_s
        self.linger_s = linger_s
        self.scenario_kwargs = scenario_kwargs
        self.rng = rng or random.Random()
        self.secondary_player_ids: list = secondary_player_ids or []

        # Set by the trainer after construction: trainer._sample_action.
        # When set, neural actions are applied to secondary players each step
        # and their transitions are stored in last_secondary_results.
        self.sample_action_fn = None
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
        self._dt_s = float(self._obs_cfg.get("sim_dt_s", 1.0 / 30.0))
        self._decision_interval_s = float(self._obs_cfg["decision_interval_s"])
        self._ticks_per_decision = max(1, round(self._decision_interval_s / self._dt_s))

        self._ema = EMAFilter.from_config()
        self._loop: Optional[ScenarioLoop] = None
        self._trial_done: bool = False
        self._episode_ticks: int = 0
        self._last_ball_dist: float = 0.0
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
        self._sec_last_ball_dist: dict = {}
        self._sec_start_stamina: dict = {}
        # Per-secondary-player running UNCLAMPED cumulative-term state, see
        # reward.cumulative_clamped_delta() / self._trainee_cumulative_state.
        # dict[pid] -> dict[term_name] -> running unclamped sum (e.g. "prog", "appr_sq").
        self._sec_cumulative_state: dict = {}
        self._sec_ema: dict = {}
        self._sec_had_possession_last_step: dict = {}
        # Per-secondary-player pending-loss state, mirrors _trainee_pending_loss.
        self._sec_pending_loss: dict = {}
        # Populated after each step(); drained by PPOTrainer for the rollout buffer.
        # last_trainee_transition: dict with obs/action/log_prob/value/raw_exec/illegal_action
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

    def reset(self) -> ObservationBatch:
        """Start a new trial and return the initial observation."""
        from footballcoach.rules_ai import NeuralPlayerAI
        # Inject sim_dt_s so build functions can pass it to Match.
        # This has no effect on builds that don't accept it (e.g. phase 2).
        build_kwargs = {**self.scenario_kwargs, "sim_dt_s": self._dt_s}
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
        self._ema.reset()
        self._episode_ticks = 0
        self._trial_done = False
        self._box_bounds_cache = {}
        self._prev_goal_count = (
            self._loop.match.scoreboard.left_goals,
            self._loop.match.scoreboard.right_goals,
        )
        self._last_ball_dist = self._ball_dist_to_trainee()
        self._max_episode_ticks = max(1, int(self.max_episode_s / self._dt_s))
        self._trainee_had_possession_last_step = False
        self._trainee_pending_loss = False
        self._sec_had_possession_last_step = {pid: False for pid in self.secondary_player_ids}
        self._sec_pending_loss = {pid: False for pid in self.secondary_player_ids}
        self._last_settled_ball_owner = None
        # Record stamina at episode start for end-of-episode stamina penalty.
        # Also recorded per-secondary-player below (same treatment — see
        # _compute_phase1_reward_for_player()).
        try:
            _trainee = self._loop.match.player_by_id(self.trainee_player_id)
            self._trainee_start_stamina = _trainee.stamina
        except KeyError:
            self._trainee_start_stamina = 1.0
        # Running UNCLAMPED cumulative-term state for this episode (see
        # reward.cumulative_clamped_delta()) — reset every episode. Maps
        # term name (e.g. "prog", "appr_sq") to its running unclamped sum.
        self._trainee_cumulative_state: dict[str, float] = {}
        self.last_trainee_transition = None
        self.last_secondary_results = []
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
                )
            except KeyError:
                pass

            is_rules_episode = getattr(match, "_opponent_use_rules_ai", False)
            is_immobile_episode = getattr(match, "_opponent_is_immobile", False)
            if not is_rules_episode and not is_immobile_episode:
                for pid in self.secondary_player_ids:
                    try:
                        sec_player = match.player_by_id(pid)
                        if pid not in self._sec_ema:
                            self._sec_ema[pid] = EMAFilter.from_config()
                        sec_player.ai = NeuralPlayerAI(
                            self.sample_action_fn,
                            decision_interval_ticks=self._ticks_per_decision,
                            max_episode_s=self.max_episode_s,
                            ema_smoothed=self._sec_ema[pid].smoothed,
                            rng=self.rng,
                        )
                    except KeyError:
                        pass

        # Initialise per-secondary-player state
        for pid in self.secondary_player_ids:
            if pid not in self._sec_ema:
                self._sec_ema[pid] = EMAFilter.from_config()
            self._sec_ema[pid].reset()
            self._sec_last_ball_dist[pid] = self._ball_dist_for_player(pid)
            # Record stamina at episode start for THIS secondary player, same
            # treatment as the trainee above — see
            # _compute_phase1_reward_for_player().
            try:
                self._sec_start_stamina[pid] = self._loop.match.player_by_id(pid).stamina
            except KeyError:
                self._sec_start_stamina[pid] = 1.0
            # Running UNCLAMPED cumulative-term state for this secondary
            # player's episode, same treatment as self._trainee_cumulative_state above.
            self._sec_cumulative_state[pid] = {}
        # Attach fresh logger to the match and record initial state.
        self._episode_index += 1
        self._match_logger.reset()
        self._loop.match.match_logger = self._match_logger
        self._match_logger.record_start(self._loop.match)
        return self._get_obs()

    def step(self) -> tuple[ObservationBatch, float, bool, StepInfo]:
        """Advance one decision interval (DECISION_INTERVAL_S sim-seconds).

        All player AI (neural and rules-based) fires automatically inside
        Match.step() via player.ai.act().  This method just ticks the sim,
        computes reward, and collects transition data from player.ai.

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
                "prev_ball_dist": self._sec_last_ball_dist.get(pid, 0.0),
                "prev_ball_x": match.ball.position.x,
                "prev_box_dist": self._ball_dist_to_opponent_box(pid),
            }

        # --- Snapshot state before ticking ---
        prev_ball_dist = self._last_ball_dist
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
        _sec_poss_prev = {pid: self._sec_had_possession_last_step.get(pid, False) for pid in sec_pre}
        _sec_pending_loss = {pid: self._sec_pending_loss.get(pid, False) for pid in sec_pre}
        sec_gained_count = {pid: 0 for pid in sec_pre}
        sec_lost_count = {pid: 0 for pid in sec_pre}

        # Per-tick box-distance progress accumulators. Must be gated on
        # possession at the SAME tick the box-distance moved, not possession
        # at the end of the whole (multi-tick) decision interval -- otherwise
        # a mid-interval tackle attributes the OPPONENT's prior ball-carrying
        # movement (typically away from the trainee's target box) to the
        # trainee's "progress" the instant it steals the ball, producing
        # large spurious negative progress on winning steps.
        _trainee_box_dist_prev = prev_box_dist
        trainee_prog_accum = 0.0
        _sec_box_dist_prev = {pid: pre["prev_box_dist"] for pid, pre in sec_pre.items()}
        sec_prog_accum = {pid: 0.0 for pid in sec_pre}

        for _ in range(self._ticks_per_decision):
            # Track shot events (KickOrder completing toward goal)
            if self._detect_shot_this_tick(match, player):
                shot_taken = True

            tick_done = self._loop.step()
            self._episode_ticks += 1

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

            # Accumulate progress using PRE-tick possession (before transition
            # below). Only ticks where the player already held the ball count;
            # the gain tick itself is excluded so the ball's pre-possession
            # trajectory doesn't pollute the player's progress reward.
            # Pass local `match` to all distance helpers — never self._loop.match.
            _curr_trainee_box_dist = self._ball_dist_to_opponent_box(self.trainee_player_id, match)
            if _trainee_poss_prev:
                trainee_prog_accum += _trainee_box_dist_prev - _curr_trainee_box_dist
            _trainee_box_dist_prev = _curr_trainee_box_dist
            for pid in sec_pre:
                _curr_sec_box_dist = self._ball_dist_to_opponent_box(pid, match)
                if _sec_poss_prev[pid]:
                    sec_prog_accum[pid] += _sec_box_dist_prev[pid] - _curr_sec_box_dist
                _sec_box_dist_prev[pid] = _curr_sec_box_dist

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
        # All geometry reads use local `match` (captured at step() entry).
        curr_ball_dist = self._ball_dist_to_trainee(match)
        self._last_ball_dist = curr_ball_dist
        _trainee_box_dist_now = self._ball_dist_to_opponent_box(self.trainee_player_id, match)

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

        # Reached opponent box with possession (phase 1 terminal — trainee wins)
        in_opponent_box = match.pitch.is_in_box(
            match.ball.position,
            left=(player.team == Team.RIGHT),  # opponent's box
        )
        # Use the per-tick scan (trainee_gained_count/trainee_lost_count) instead
        # of a simple before/after comparison, so gain/lose transitions that
        # both happen within this single decision interval (e.g. tackle then
        # immediately re-tackled) are not silently dropped from the reward.
        trainee_has_possession_now = _trainee_poss_prev  # final state after the tick loop
        gained_possession = trainee_gained_count  # int count, not just bool
        lost_possession = trainee_lost_count
        self._trainee_had_possession_last_step = trainee_has_possession_now
        self._trainee_pending_loss = _trainee_pending_loss
        box_terminal = (
            in_opponent_box
            and trainee_has_possession_now
            and self._can_score_box_terminal(player)
        )

        # Opponent reached trainee's box with possession (phase 1 terminal — trainee loses).
        # Excluded when the ball carrier has no real controller (see
        # _can_score_box_terminal): it never chases or holds a defensive
        # line, so the only way it could ever satisfy this otherwise would
        # be a pure coincidence of spawn position (and/or incidental
        # collision push-apart nudging it there — see ui/scenarios.py's
        # build_1v1_scenario, which also re-rolls the immobile opponent's
        # spawn away from its own box with a clearance margin,
        # belt-and-braces). Gating the terminal condition itself here is
        # the actually-robust fix: it holds regardless of how the immobile
        # opponent's position ever got wherever it is, rather than trying
        # to prevent every possible geometric path to this outcome.
        #
        # Looked up generically by whoever is actually carrying the ball
        # (not hardcoded to "the opponent") so this holds even once a
        # scenario has more than one non-trainee player.
        in_trainee_box = match.pitch.is_in_box(
            match.ball.position,
            left=(player.team == Team.LEFT),  # trainee's own box
        )
        _carrier_id = match.ball.possessed_by
        _carrier_can_score = False
        if _carrier_id is not None and _carrier_id != self.trainee_player_id:
            try:
                _carrier_can_score = self._can_score_box_terminal(match.player_by_id(_carrier_id))
            except KeyError:
                _carrier_can_score = False
        opponent_box_terminal = (
            in_trainee_box
            and _carrier_id is not None
            and _carrier_id != self.trainee_player_id
            and _carrier_can_score
        )

        timeout = self._episode_ticks >= int(self.max_episode_s / self._dt_s)
        if self.phase == 1:
            # Routes through the SINGLE shared _compute_phase1_reward_for_player()
            # method (see its docstring) — trainee and every secondary player
            # call the exact same code path, so per-player inputs (speed,
            # heading cosine, stamina used) can never silently diverge between
            # the two again.
            reward, self.last_reward_components, self._trainee_cumulative_state = self._compute_phase1_reward_for_player(
                player_id=self.trainee_player_id,
                player_obj=player,
                ball_pos=match.ball.position,
                start_stamina=self._trainee_start_stamina,
                prev_ball_dist=prev_ball_dist,
                curr_ball_dist=curr_ball_dist,
                has_possession_now=trainee_has_possession_now,
                gained_possession_this_step=gained_possession,
                lost_possession_this_step=lost_possession,
                ball_progress_toward_goal_m=ball_progress,
                ball_went_out_after_touch=ball_went_out,
                illegal_action_attempted=info.illegal_action,
                reached_opponent_box_with_possession=box_terminal,
                opponent_reached_trainee_box=opponent_box_terminal,
                timed_out=timeout and not box_terminal and not opponent_box_terminal,
                episode_done=box_terminal or opponent_box_terminal or timeout,
                cumulative_state=self._trainee_cumulative_state,
                ball_dist_to_opponent_box_m=_trainee_box_dist_now,
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

        # --- Collect trainee transition from NeuralPlayerAI ---
        if hasattr(player, "ai") and player.ai is not None and hasattr(player.ai, "last_transition"):
            self.last_trainee_transition = player.ai.last_transition

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

            sec_curr_ball_dist = self._ball_dist_for_player(pid, match)
            self._sec_last_ball_dist[pid] = sec_curr_ball_dist
            _sec_box_dist_now = self._ball_dist_to_opponent_box(pid, match)

            # Box-distance closed this step (own attacking box), matching the
            # trainee's ball_progress above — not raw goal-line x movement,
            # and per-tick accumulated/possession-gated (see sec_prog_accum
            # init comment above), not a single start-vs-end-of-interval delta.
            sec_ball_prog = sec_prog_accum.get(pid, 0.0)

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
            # Gated via the SAME shared _can_score_box_terminal() used by
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
                and self._can_score_box_terminal(sec_player)
            )

            if self.phase == 1:
                # Routes through the SAME shared _compute_phase1_reward_for_player()
                # method used by the trainee above — see its docstring. This
                # is what guarantees secondary players get identical
                # heading/appr_sq/stamina treatment to the trainee; do not
                # revert to calling phase1_reward() directly here.
                sec_episode_done = sec_box_terminal or box_terminal or timeout
                sec_reward, _sec_comps, self._sec_cumulative_state[pid] = self._compute_phase1_reward_for_player(
                    player_id=pid,
                    player_obj=sec_player,
                    ball_pos=match.ball.position,
                    start_stamina=self._sec_start_stamina.get(pid, 1.0),
                    prev_ball_dist=pre["prev_ball_dist"],
                    curr_ball_dist=sec_curr_ball_dist,
                    has_possession_now=sec_has_poss_now,
                    gained_possession_this_step=sec_gained_poss,
                    lost_possession_this_step=sec_lost_poss,
                    ball_progress_toward_goal_m=sec_ball_prog,
                    ball_went_out_after_touch=sec_ball_went_out,
                    illegal_action_attempted=(
                        sec_player.ai.last_transition.get("illegal_action", False)
                        if _sec_has_transition else False
                    ),
                    reached_opponent_box_with_possession=sec_box_terminal,
                    opponent_reached_trainee_box=box_terminal,  # from sec's POV, trainee winning = sec losing
                    timed_out=timeout and not sec_box_terminal and not box_terminal,
                    episode_done=sec_episode_done,
                    cumulative_state=self._sec_cumulative_state.get(pid, {}),
                    ball_dist_to_opponent_box_m=_sec_box_dist_now,
                )
            else:
                sec_reward = 0.0
                _sec_comps = {}

            self.last_secondary_results.append({
                **(sec_player.ai.last_transition if _sec_has_transition else {}),
                "player_id": pid,
                "reward": sec_reward,
                "done": 1.0 if done else 0.0,
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
            "prog_negative_clamp": (
                (clamp := self._reward_cfg["phase1"].get("ball_progress_reward_clamp")) is not None
                and self._trainee_cumulative_state.get("prog", 0.0) <= -clamp
            ),
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

    def _get_obs(self, player_id: str | None = None) -> ObservationBatch:
        """Encode the observation for *player_id* (default: the trainee).

        Passing an explicit ``player_id`` lets callers (e.g.
        record_demonstrations.py) encode observations for other players
        (the opponent) without disturbing any of the trainee-specific
        internal call sites, which all call this with no arguments.
        """
        time_remaining = max(
            0.0,
            self.max_episode_s - self._episode_ticks * self._dt_s
        )
        return encode_observation(
            match=self._loop.match,
            player_id=player_id if player_id is not None else self.trainee_player_id,
            time_remaining_s=time_remaining,
            attack_defence_smoothed=self._ema.smoothed,
            rng=self.rng,
            phase=self.phase,
        )

    def _ball_dist_to_trainee(self, match=None) -> float:
        return self._ball_dist_for_player(self.trainee_player_id, match)

    @staticmethod
    def _player_speed_and_heading_cos(player_obj, ball_pos) -> tuple[float, float]:
        """Player speed (m/s) and cos-sim of velocity vs direction-to-ball.

        Shared by ALL phase-1 reward computations (trainee AND every
        secondary player) via _compute_phase1_reward_for_player() below —
        do not inline this logic at a second call site. Having two separate
        inline copies of this is exactly how the heading/appr_sq/stamina
        terms silently went missing for secondary players previously (see
        ai/knowledge.md "Reward parity" note) — a single shared helper makes
        that class of bug structurally impossible to reintroduce.
        """
        _vel = player_obj.velocity
        _to_ball = ball_pos - player_obj.position
        _speed = math.sqrt(_vel.x ** 2 + _vel.y ** 2)
        _to_ball_len = math.sqrt(_to_ball.x ** 2 + _to_ball.y ** 2)
        if _speed > 1e-3 and _to_ball_len > 1e-3:
            _cos = (_vel.x * _to_ball.x + _vel.y * _to_ball.y) / (_speed * _to_ball_len)
        else:
            _cos = 1.0  # neutral: no penalty when stationary or at ball
        return _speed, _cos

    @staticmethod
    def _can_score_box_terminal(player_obj) -> bool:
        """True iff `player_obj` has a real controller and can therefore
        legitimately trigger a box-terminal event (reaching a scoring box
        with possession).

        Shared by EVERY phase-1 box-terminal check below (`box_terminal` --
        trainee win, `opponent_box_terminal` -- trainee loss, and
        `sec_box_terminal` -- each secondary player's own reward) -- do not
        re-derive this gate at a second call site. Before this helper
        existed, each of those three sites independently decided whether to
        exclude an immobile player (via a single match-level
        `_opponent_is_immobile` flag, checked at some call sites and not
        others) instead of asking a genuinely player-scoped question --
        `sec_box_terminal` missed the gate entirely, letting a ball that
        settled near an immobile demo-recording opponent by physics
        coincidence re-fire the one-time box/speed-bonus reward every tick
        for as long as the fluke held (confirmed: 64 consecutive ticks in
        one real recorded episode) instead of once.

        `player_obj.ai is None` means no controller was ever assigned (the
        immobile-opponent build path) -- a player with no controller can
        never intentionally "reach" anything, so the ball settling near
        them isn't a real terminal event. This is per-player rather than a
        single global match flag, so it generalises correctly to however
        many non-trainee players a scenario ever has, unlike
        `_opponent_is_immobile` (see ai/knowledge.md's "hardcoded to
        exactly 2 players" note).
        """
        return player_obj.ai is not None

    def _compute_phase1_reward_for_player(
        self,
        *,
        player_id: str,
        player_obj,
        ball_pos,
        start_stamina: float,
        prev_ball_dist: float,
        curr_ball_dist: float,
        has_possession_now: bool,
        gained_possession_this_step,
        lost_possession_this_step,
        ball_progress_toward_goal_m: float,
        ball_went_out_after_touch: bool,
        illegal_action_attempted: bool,
        reached_opponent_box_with_possession: bool,
        opponent_reached_trainee_box: bool,
        timed_out: bool,
        episode_done: bool,
        cumulative_state: dict[str, float] | None = None,
        ball_dist_to_opponent_box_m: float | None = None,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """SINGLE call site for phase1_reward(), used for the trainee AND
        every secondary player. All per-player-derived inputs (speed,
        heading cosine, stamina used, box-distance) are computed HERE, once,
        so trainee and secondary players are structurally guaranteed to
        receive identical treatment — see ai/knowledge.md "Reward parity"
        note. Do not call phase1_reward() directly from anywhere else in
        this file; route through this method instead, even if it means
        passing a few extra already-known values as kwargs. In particular,
        every caller MUST pass player_id explicitly — the trainee's own
        attacking box and a secondary player's own attacking box are
        generally on OPPOSITE ends of the pitch, so reusing one player's
        box-distance for another would compute prog/prox against the wrong
        goal (see ball_dist_to_opponent_box_m below, resolved per player_id).

        Returns (reward, components, cumulative_state_after) — callers must
        store cumulative_state_after (per player) and pass it back in as
        cumulative_state on the next call for that same player, so the
        per-term cumulative clamps (see phase1_reward/cumulative_clamped_delta)
        track correctly across the whole episode.
        """
        _speed, _hdg_cos = self._player_speed_and_heading_cos(player_obj, ball_pos)
        _stamina_used = (1.0 - player_obj.stamina) if episode_done else 0.0
        return phase1_reward(
            prev_ball_dist=prev_ball_dist,
            curr_ball_dist=curr_ball_dist,
            has_possession_now=has_possession_now,
            gained_possession_this_step=gained_possession_this_step,
            lost_possession_this_step=lost_possession_this_step,
            ball_progress_toward_goal_m=ball_progress_toward_goal_m,
            ball_went_out_after_touch=ball_went_out_after_touch,
            illegal_action_attempted=illegal_action_attempted,
            reached_opponent_box_with_possession=reached_opponent_box_with_possession,
            cfg=self._reward_cfg["phase1"],
            time_fraction_remaining=1.0 - self._episode_ticks / self._max_episode_ticks,
            opponent_reached_trainee_box=opponent_reached_trainee_box,
            timed_out=timed_out,
            ball_dist_to_opponent_box_m=(
                ball_dist_to_opponent_box_m
                if ball_dist_to_opponent_box_m is not None
                else self._ball_dist_to_opponent_box(player_id)
            ),
            episode_done=episode_done,
            heading_cos_sim=_hdg_cos,
            player_speed_mps=_speed,
            stamina_used=_stamina_used,
            prog_reward_clamp=self._reward_cfg["phase1"].get("ball_progress_reward_clamp"),
            appr_sq_approach_reward_clamp=self._reward_cfg["phase1"].get("ball_approach_speed_reward_clamp"),
            appr_sq_retreat_reward_clamp=self._reward_cfg["phase1"].get("ball_retreat_speed_reward_clamp"),
            cumulative_state=cumulative_state,
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
