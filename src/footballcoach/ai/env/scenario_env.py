"""ScenarioEnv: Gym-like wrapper around ScenarioDefinition + ScenarioLoop.

See ai_design_doc.md section 9.2 for the design rationale (reuse existing
UI scenarios rather than reimplementing from scratch).

The env wraps ONE trainee player in a scenario.  Other players are driven by
whatever rules-based orders the ScenarioDefinition already assigns them
(SaveOrder for GKs, ChaseTackleOrder for defenders, etc.) - no special
scripted-opponent logic is needed here; it already exists in ui/scenarios.py.

API:
    env = ScenarioEnv(definition, trainee_player_id="kicker", phase=1)
    obs = env.reset()
    obs, reward, done, info = env.step(action_dict)

``action_dict`` is the output of the policy's sample() call:
    {
      "decision": DecisionAction (dataclass with .shoot, .pass_, .move, etc.),
      "execution": ExecutionAction (dataclass with .move_direction_raw, .sprint, etc.),
      "decision_probs": dict of sigmoid probs,
      "target_slots": dict of categorical target slot indices,
      "execution_physical": dict of physical execution outputs,
      "decision_physical": dict of decoded physical decision outputs,
      "slot_player_ids": list of player_id per slot,
    }
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from footballcoach.ai.action.gating import select_action
from footballcoach.ai.action.to_orders import (
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


class ScenarioEnv:
    """Training environment wrapping a ScenarioDefinition + ScenarioLoop.

    One ``ScenarioEnv`` per trainee player.  Other players are rules-based.

    Args:
        definition: A ScenarioDefinition from ui/scenarios.py.
        trainee_player_id: The player_id of the AI player being trained.
        phase: Curriculum phase (1 or 2) - selects the reward function.
        rng_reduction: Physics randomness level (updated by the trainer).
        max_episode_s: Maximum episode duration in sim-seconds.
        linger_s: How long to wait after trial ends before resetting.
        rng: Random for obs slot shuffling.
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

        cfg = load_ai_config()
        self._obs_cfg = cfg["observation"]
        self._reward_cfg = cfg["reward"]
        self._dt_s = 1.0 / 30.0  # engine tick rate
        self._decision_interval_s = float(self._obs_cfg["decision_interval_s"])
        self._ticks_per_decision = max(1, round(self._decision_interval_s / self._dt_s))

        self._ema = EMAFilter.from_config()
        self._loop: Optional[ScenarioLoop] = None
        self._trial_done: bool = False
        self._episode_ticks: int = 0
        self._last_ball_dist: float = 0.0
        self._ball_touched_by_trainee: bool = False
        self._prev_goal_count: tuple[int, int] = (0, 0)

    # -----------------------------------------------------------------------
    # Gym-like API
    # -----------------------------------------------------------------------

    def reset(self) -> ObservationBatch:
        """Start a new trial and return the initial observation."""
        self._loop = ScenarioLoop(
            definition=self.definition,
            max_trials=0,
            rng_reduction=self.rng_reduction,
            linger_s=self.linger_s,
            kwargs=self.scenario_kwargs,
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
        return self._get_obs()

    def step(self, action: dict) -> tuple[ObservationBatch, float, bool, StepInfo]:
        """Advance one decision interval (DECISION_INTERVAL_S sim-seconds).

        Args:
            action: Dict with keys described in the module docstring.

        Returns:
            (observation, reward, done, info)
        """
        if self._loop is None:
            raise RuntimeError("Call reset() before step()")

        match = self._loop.match
        player = self._find_trainee(match)
        info = StepInfo()

        # --- Apply the action as engine orders ---
        gating = select_action(
            decision_probs=action.get("decision_probs", {}),
            execution_physical=action.get("execution_physical", {}),
            target_slots=action.get("target_slots", {}),
        )
        result: OrderTranslationResult = apply_action_to_player(
            gating=gating,
            player=player,
            match=match,
            slot_player_ids=action.get("slot_player_ids", [None] * MAX_OTHER_PLAYERS),
            decision_physical=action.get("decision_physical", {}),
        )
        info.illegal_action = result.illegal_action
        info.illegal_reason = result.illegal_reason

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

            # Track shot events (KickOrder completing toward goal)
            if self._detect_shot_this_tick(match, player):
                shot_taken = True

            tick_done = self._loop.step()
            self._episode_ticks += 1

            if tick_done:
                outcome_this_step = self._latest_outcome()
                if outcome_this_step == "goal":
                    goal_scored = True
                    self._ema.on_goal()
                trial_ended_this_step = True
                break

        # Update EMA with the decision network's attack/defence raw output
        if "decision" in action and hasattr(action.get("decision"), "attack_defence_raw"):
            raw_ad = float(action["decision"].attack_defence_raw)
            self._ema.update(raw_ad, self._decision_interval_s)

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

        # Reached opponent box with possession (phase 1 terminal)
        in_box = match.pitch.is_in_box(
            match.ball.position,
            left=(player.team == Team.RIGHT),  # opponent's box
        )
        gained_possession = (
            match.ball.possessed_by == self.trainee_player_id
            and prev_ball_dist > 0.5
        )
        box_terminal = in_box and match.ball.possessed_by == self.trainee_player_id

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
        done = trial_ended_this_step or timeout or box_terminal

        if done:
            info.trial_outcome = outcome_this_step or ("timeout" if timeout else "box_possession")

        info.ticks_elapsed = self._episode_ticks

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
        try:
            player = self._find_trainee(self._loop.match)
            ball = self._loop.match.ball
            return math.hypot(
                ball.position.x - player.position.x,
                ball.position.y - player.position.y,
            )
        except (KeyError, AttributeError):
            return 0.0

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
