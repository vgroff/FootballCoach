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
from typing import Optional

import numpy as np

from footballcoach.ai.action.gating import select_action
from footballcoach.ai.action.apply_nn_action import (
    OrderTranslationResult,
    apply_action_to_player,
    encode_slot_player_ids,
)
from footballcoach.ai.config import load_ai_config
from footballcoach.ai.env.reward import EMAFilter, phase1_reward, phase2_reward
from footballcoach.ai.obs.encoder import MAX_OTHER_PLAYERS, encode_observation
from footballcoach.ai.obs.schema import ObservationBatch
from footballcoach.engine.match import Match
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
        self._start_ball_to_box_dist_m: float = 1.0
        self._max_episode_ticks: int = 1
        self._ball_touched_by_trainee: bool = False
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
        self._sec_ball_touched: dict = {}
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

    # -----------------------------------------------------------------------
    # Gym-like API
    # -----------------------------------------------------------------------

    def reset(self) -> ObservationBatch:
        """Start a new trial and return the initial observation."""
        from footballcoach.rules_ai import NeuralPlayerAI
        # Inject sim_dt_s so build functions can pass it to Match.
        # This has no effect on builds that don't accept it (e.g. phase 2).
        build_kwargs = {**self.scenario_kwargs, "sim_dt_s": self._dt_s}
        self._loop = ScenarioLoop(
            definition=self.definition,
            max_trials=0,
            rng_reduction=self.rng_reduction,
            linger_s=self.linger_s,
            kwargs=build_kwargs,
            timeout_ticks=int(self.max_episode_s / self._dt_s),
        )
        self._ema.reset()
        self._episode_ticks = 0
        self._trial_done = False
        self._ball_touched_by_trainee = False
        self._prev_goal_count = (
            self._loop.match.scoreboard.left_goals,
            self._loop.match.scoreboard.right_goals,
        )
        self._last_ball_dist = self._ball_dist_to_trainee()
        self._max_episode_ticks = max(1, int(self.max_episode_s / self._dt_s))
        self._start_ball_to_box_dist_m = self._ball_dist_to_opponent_box()
        self._trainee_had_possession_last_step = False
        self._trainee_pending_loss = False
        self._sec_had_possession_last_step = {pid: False for pid in self.secondary_player_ids}
        self._sec_pending_loss = {pid: False for pid in self.secondary_player_ids}
        self._last_settled_ball_owner = None
        # Record stamina at episode start for end-of-episode stamina penalty.
        try:
            _trainee = self._loop.match.player_by_id(self.trainee_player_id)
            self._trainee_start_stamina = _trainee.stamina
        except KeyError:
            self._trainee_start_stamina = 1.0
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
            self._sec_ball_touched[pid] = False
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
            }

        # --- Snapshot state before ticking ---
        prev_ball_dist = self._last_ball_dist
        prev_ball_x = match.ball.position.x
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

        for _ in range(self._ticks_per_decision):
            # Track ball touches by trainee
            if match.ball.possessed_by == self.trainee_player_id:
                self._ball_touched_by_trainee = True

            # Track ball touches by secondary players
            for pid in sec_pre:
                if match.ball.possessed_by == pid:
                    self._sec_ball_touched[pid] = True

            # Track shot events (KickOrder completing toward goal)
            if self._detect_shot_this_tick(match, player):
                shot_taken = True

            tick_done = self._loop.step()
            self._episode_ticks += 1

            # Detect possession gain/loss transitions THIS tick (trainee).
            # Uses the shared state machine so a "kick to yourself" / brief
            # loose-ball self-repossession is NOT counted as a turnover (only
            # a possession settling onto a DIFFERENT player counts).
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

            # Same for secondary players sharing this tick loop.
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

            if tick_done:
                outcome_this_step = self._latest_outcome()
                if outcome_this_step == "goal":
                    goal_scored = True
                    # Phase 1 doesn't reward/track goals; only box possession matters.
                    if self.phase != 1:
                        self._ema.on_goal()
                trial_ended_this_step = True
                break

        # Update EMA with the trainee's attack/defence output (from NeuralPlayerAI transition)
        if self.last_trainee_transition is not None:
            decision = self.last_trainee_transition.get("action")
            if decision is not None and hasattr(decision, "attack_defence_raw"):
                self._ema.update(float(decision.attack_defence_raw), self._decision_interval_s)
            info.illegal_action = self.last_trainee_transition.get("illegal_action", False)

        # --- Compute reward ---
        curr_ball_dist = self._ball_dist_to_trainee()
        self._last_ball_dist = curr_ball_dist

        new_goal_count = (match.scoreboard.left_goals, match.scoreboard.right_goals)
        self._prev_goal_count = new_goal_count

        # Ball went out after trainee touched it
        ball_went_out = trial_ended_this_step and outcome_this_step == "miss" and self._ball_touched_by_trainee

        # Ball progress toward opponent goal
        if player.team == Team.LEFT:
            ball_progress = match.ball.position.x - prev_ball_x
        else:
            ball_progress = prev_ball_x - match.ball.position.x

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
        box_terminal = in_opponent_box and trainee_has_possession_now

        # Opponent reached trainee's box with possession (phase 1 terminal — trainee loses)
        in_trainee_box = match.pitch.is_in_box(
            match.ball.position,
            left=(player.team == Team.LEFT),  # trainee's own box
        )
        opponent_box_terminal = (
            in_trainee_box
            and match.ball.possessed_by is not None
            and match.ball.possessed_by != self.trainee_player_id
        )

        timeout = self._episode_ticks >= int(self.max_episode_s / self._dt_s)
        if self.phase == 1:
            # Cosine similarity between player velocity direction and direction to ball.
            # Used for heading penalty: penalises running away from the ball.
            _vel = player.velocity
            _to_ball = match.ball.position - player.position
            _speed = math.sqrt(_vel.x ** 2 + _vel.y ** 2)
            _to_ball_len = math.sqrt(_to_ball.x ** 2 + _to_ball.y ** 2)
            if _speed > 1e-3 and _to_ball_len > 1e-3:
                _hdg_cos = (_vel.x * _to_ball.x + _vel.y * _to_ball.y) / (_speed * _to_ball_len)
            else:
                _hdg_cos = 1.0  # neutral: no penalty when stationary or at ball

            _episode_done = done if 'done' in dir() else False
            _stamina_used = max(0.0, self._trainee_start_stamina - player.stamina) if (box_terminal or opponent_box_terminal or timeout) else 0.0

            reward, self.last_reward_components = phase1_reward(
                prev_ball_dist=prev_ball_dist,
                curr_ball_dist=curr_ball_dist,
                has_possession_now=trainee_has_possession_now,
                gained_possession_this_step=gained_possession,
                lost_possession_this_step=lost_possession,
                ball_progress_toward_goal_m=ball_progress,
                ball_went_out_after_touch=ball_went_out,
                illegal_action_attempted=info.illegal_action,
                reached_opponent_box_with_possession=box_terminal,
                cfg=self._reward_cfg["phase1"],
                time_fraction_remaining=1.0 - self._episode_ticks / self._max_episode_ticks,
                start_ball_to_box_dist_m=self._start_ball_to_box_dist_m,
                opponent_reached_trainee_box=opponent_box_terminal,
                timed_out=timeout and not box_terminal and not opponent_box_terminal and not trial_ended_this_step,
                ball_dist_to_opponent_box_m=self._ball_dist_to_opponent_box(),
                heading_cos_sim=_hdg_cos,
                player_speed_mps=_speed,
                stamina_used=_stamina_used,
                episode_done=box_terminal or opponent_box_terminal or timeout,
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

        if done:
            if box_terminal:
                outcome_label = "box_possession"
            elif opponent_box_terminal:
                outcome_label = "opponent_box_possession"
            elif timeout:
                outcome_label = "timeout"
            else:
                outcome_label = outcome_this_step
                # Phase 1 has no goal condition: treat it the same as the ball
                # going out of bounds (neither player wins by shooting).
                if self.phase == 1 and outcome_label == "goal":
                    outcome_label = "miss"
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
            if not (hasattr(sec_player, "ai") and sec_player.ai is not None
                    and hasattr(sec_player.ai, "last_transition")
                    and sec_player.ai.last_transition is not None):
                continue

            sec_curr_ball_dist = self._ball_dist_for_player(pid)
            self._sec_last_ball_dist[pid] = sec_curr_ball_dist

            if sec_player.team == Team.LEFT:
                sec_ball_prog = match.ball.position.x - pre["prev_ball_x"]
            else:
                sec_ball_prog = pre["prev_ball_x"] - match.ball.position.x

            sec_ball_went_out = (
                trial_ended_this_step
                and outcome_this_step == "miss"
                and self._sec_ball_touched.get(pid, False)
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
            sec_box_terminal = sec_in_atk_box and match.ball.possessed_by == pid

            if self.phase == 1:
                sec_reward, _sec_comps = phase1_reward(
                    prev_ball_dist=pre["prev_ball_dist"],
                    curr_ball_dist=sec_curr_ball_dist,
                    has_possession_now=sec_has_poss_now,
                    gained_possession_this_step=sec_gained_poss,
                    lost_possession_this_step=sec_lost_poss,
                    ball_progress_toward_goal_m=sec_ball_prog,
                    ball_went_out_after_touch=sec_ball_went_out,
                    illegal_action_attempted=sec_player.ai.last_transition.get("illegal_action", False),
                    reached_opponent_box_with_possession=sec_box_terminal,
                    cfg=self._reward_cfg["phase1"],
                    time_fraction_remaining=1.0 - self._episode_ticks / self._max_episode_ticks,
                    start_ball_to_box_dist_m=self._start_ball_to_box_dist_m,
                    opponent_reached_trainee_box=box_terminal,  # from sec's POV, trainee winning = sec losing
                    timed_out=timeout and not sec_box_terminal and not box_terminal and not trial_ended_this_step,
                    ball_dist_to_opponent_box_m=self._ball_dist_to_opponent_box(),
                )
            else:
                sec_reward = 0.0
                _sec_comps = {}

            self.last_secondary_results.append({
                **sec_player.ai.last_transition,
                "reward": sec_reward,
                "done": 1.0 if done else 0.0,
            })
            # Accumulate secondary components into last_reward_components
            for _k, _v in _sec_comps.items():
                self.last_reward_components[_k] = self.last_reward_components.get(_k, 0.0) + _v

        return self._get_obs(), reward, done, info

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

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

    def _ball_dist_to_trainee(self) -> float:
        return self._ball_dist_for_player(self.trainee_player_id)

    def _ball_dist_to_opponent_box(self) -> float:
        """Distance from the ball to the near edge of the opponent's box at call time."""
        try:
            match = self._loop.match
            ball = match.ball.position
            player = match.player_by_id(self.trainee_player_id)
            pitch = match.pitch
            # Opponent box near edge: half_length - box_length_m from centre
            box_edge_x = pitch.half_length - pitch.box_length_m
            if player.team == Team.LEFT:
                dist_x = max(0.0, box_edge_x - ball.x)
            else:
                dist_x = max(0.0, box_edge_x - (-ball.x))
            return float(dist_x)
        except (KeyError, AttributeError):
            return 1.0

    def _ball_dist_for_player(self, player_id: str) -> float:
        try:
            player = self._loop.match.player_by_id(player_id)
            ball = self._loop.match.ball
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
