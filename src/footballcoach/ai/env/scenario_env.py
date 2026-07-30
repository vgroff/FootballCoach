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
    is_rules_episode: bool = False  # True when opponent is rules-based (50% of Phase 1 episodes)


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
        self._ball_touched_by_trainee: bool = False
        self._prev_goal_count: tuple[int, int] = (0, 0)

        # Per-secondary-player state
        self._sec_last_ball_dist: dict = {}
        self._sec_ball_touched: dict = {}
        self._sec_ema: dict = {}
        # Populated after each step(); drained by PPOTrainer for the rollout buffer.
        # last_trainee_transition: dict with obs/action/log_prob/value/raw_exec/illegal_action
        # last_secondary_results: list of same dicts for secondary neural players
        self.last_trainee_transition: Optional[dict] = None
        self.last_secondary_results: list = []

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
        self.last_trainee_transition = None
        self.last_secondary_results = []

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
            if not is_rules_episode:
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
        gained_possession = (
            match.ball.possessed_by == self.trainee_player_id
            and prev_ball_dist > 0.5
        )
        box_terminal = in_opponent_box and match.ball.possessed_by == self.trainee_player_id

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

        if self.phase == 1:
            reward = phase1_reward(
                prev_ball_dist=prev_ball_dist,
                curr_ball_dist=curr_ball_dist,
                has_possession_now=(match.ball.possessed_by == self.trainee_player_id),
                gained_possession_this_step=gained_possession,
                ball_progress_toward_goal_m=ball_progress,
                ball_went_out_after_touch=ball_went_out,
                illegal_action_attempted=info.illegal_action,
                reached_opponent_box_with_possession=box_terminal,
                cfg=self._reward_cfg["phase1"],
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
        timeout = self._episode_ticks >= int(self.max_episode_s / self._dt_s)
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
            sec_gained_poss = (
                match.ball.possessed_by == pid
                and pre["prev_ball_dist"] > 0.5
            )
            sec_in_atk_box = match.pitch.is_in_box(
                match.ball.position,
                left=(sec_player.team == Team.RIGHT),
            )
            sec_box_terminal = sec_in_atk_box and match.ball.possessed_by == pid

            if self.phase == 1:
                sec_reward = phase1_reward(
                    prev_ball_dist=pre["prev_ball_dist"],
                    curr_ball_dist=sec_curr_ball_dist,
                    has_possession_now=(match.ball.possessed_by == pid),
                    gained_possession_this_step=sec_gained_poss,
                    ball_progress_toward_goal_m=sec_ball_prog,
                    ball_went_out_after_touch=sec_ball_went_out,
                    illegal_action_attempted=sec_player.ai.last_transition.get("illegal_action", False),
                    reached_opponent_box_with_possession=sec_box_terminal,
                    cfg=self._reward_cfg["phase1"],
                )
            else:
                sec_reward = 0.0

            self.last_secondary_results.append({
                **sec_player.ai.last_transition,
                "reward": sec_reward,
                "done": 1.0 if done else 0.0,
            })

        return self._get_obs(), reward, done, info

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_obs(self) -> ObservationBatch:
        time_remaining = max(
            0.0,
            self.max_episode_s - self._episode_ticks * self._dt_s
        )
        return encode_observation(
            match=self._loop.match,
            player_id=self.trainee_player_id,
            time_remaining_s=time_remaining,
            attack_defence_smoothed=self._ema.smoothed,
            rng=self.rng,
        )

    def _ball_dist_to_trainee(self) -> float:
        return self._ball_dist_for_player(self.trainee_player_id)

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
        )

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
