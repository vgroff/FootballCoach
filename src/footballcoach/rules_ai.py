"""Rules-based AI controllers.

Each class is a ``PlayerAI`` subclass — assign one to ``player.ai`` and
``Match.step()`` will call ``ai.act(player, match, tick)`` automatically
every physics tick.

Classes:
  Phase1RulesAI           — chase ball; when possessed, sprint to opponent box entry
  StagedGoalkeeperAI      — wait until goal-centre MoveOrder completes, then SaveOrder
  BallCarrierAttackerAI   — move to goal; switch to ShootOrder when progress stalls
  BallReceiverThenShootAI — wait for ball receipt, set up shot/run, then delegate
  SprintWaypointAI        — issue sequential MoveOrders along a waypoint list
  NeuralPlayerAI          — wraps a PPO network; samples every N ticks, exposes
                            last_transition for the rollout buffer
"""
from __future__ import annotations

import random

from footballcoach.entities.player import Player, PlayerAI, Team
from footballcoach.engine.match import Match
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    GetPossessionOrder,
    MoveOrder,
    SaveOrder,
    ShootOrder,
)


class Phase1RulesAI(PlayerAI):
    """Chase ball; when possessed, sprint toward a random point inside the
    opponent's box (in the half of the box nearest to the player's current
    y-position).  Team-aware via player.team."""

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if match.ball.possessed_by == player.player_id:
            if not isinstance(player.current_order, MoveOrder):
                pitch = match.pitch
                half_box_w = pitch.box_width_m / 2.0
                # x: random within the full box depth (inner edge → goal line)
                if player.team == Team.LEFT:
                    box_inner_x = pitch.half_length - pitch.box_length_m
                    target_x = random.uniform(box_inner_x, pitch.half_length)
                else:
                    box_inner_x = -(pitch.half_length - pitch.box_length_m)
                    target_x = random.uniform(-pitch.half_length, box_inner_x)
                # y: random in the nearest half of the box to the player
                if player.position.y >= 0.0:
                    target_y = random.uniform(0.0, half_box_w)
                else:
                    target_y = random.uniform(-half_box_w, 0.0)
                player.current_order = MoveOrder(
                    target_position=Vector3(target_x, target_y, 0.0),
                    sprint=True,
                )
        else:
            if player.current_order is None:
                player.current_order = GetPossessionOrder(
                    sprint=random.random() >= 0.25
                )


class StagedGoalkeeperAI(PlayerAI):
    """Once the GK's initial MoveOrder (to goal centre) completes and the GK
    does not have the ball, issue a SaveOrder."""

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if player.current_order is None and match.ball.possessed_by != player.player_id:
            player.current_order = SaveOrder()


class BallCarrierAttackerAI(PlayerAI):
    """Ball carrier runs toward goal; if their MoveOrder progress stalls
    (distance to target starts increasing) or the order completes, switches
    to a ShootOrder at the configured aim point."""

    def __init__(self, aim_point: Vector3, power_fraction: float = 0.9) -> None:
        self.aim_point = aim_point
        self.power_fraction = power_fraction
        self._prev_dist_to_target: float | None = None

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if match.ball.possessed_by != player.player_id:
            self._prev_dist_to_target = None
            return
        order = player.current_order
        if isinstance(order, MoveOrder):
            dist = player.position.xy().distance_to(order.target_position.xy())
            prev = self._prev_dist_to_target
            self._prev_dist_to_target = dist
            if prev is not None and dist > prev:
                player.current_order = ShootOrder(
                    aim_point=self.aim_point, power_fraction=self.power_fraction
                )
        elif order is None:
            self._prev_dist_to_target = None
            player.current_order = ShootOrder(
                aim_point=self.aim_point, power_fraction=self.power_fraction
            )


class BallReceiverThenShootAI(PlayerAI):
    """AI for a player waiting to receive a pass, then shooting or running.

    Phase 1: waits (initial MoveOrder already set at build time).
    Phase 2: once possession is gained, either shoots immediately or runs
    ``run_fraction`` of the way toward the goal first.
    Phase 3: delegates to ``BallCarrierAttackerAI`` for the rest.
    """

    def __init__(
        self,
        goal_aim_point: Vector3,
        shoot_immediately: bool,
        run_fraction: float = 0.3,
        power_fraction: float = 0.85,
    ) -> None:
        self.goal_aim_point = goal_aim_point
        self._shoot_immediately = shoot_immediately
        self._run_fraction = run_fraction
        self._received = False
        self._carrier_ai = BallCarrierAttackerAI(goal_aim_point, power_fraction=power_fraction)

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if not self._received:
            if match.ball.possessed_by == player.player_id:
                self._received = True
                if self._shoot_immediately:
                    player.current_order = ShootOrder(
                        aim_point=self.goal_aim_point,
                        power_fraction=self._carrier_ai.power_fraction,
                    )
                else:
                    run_target = Vector3(
                        player.position.x
                        + (self.goal_aim_point.x - player.position.x) * self._run_fraction,
                        player.position.y,
                        0.0,
                    )
                    player.current_order = MoveOrder(target_position=run_target, sprint=True)
        self._carrier_ai.act(player, match, trial_tick)


class SprintWaypointAI(PlayerAI):
    """Issue sequential MoveOrders along a pre-computed waypoint list.
    The first waypoint and ``start_idx`` should be set during scenario build::

        player.current_order = MoveOrder(target_position=waypoints[0], sprint=True)
        player.ai = SprintWaypointAI(waypoints, start_idx=1)
    """

    def __init__(self, waypoints: list[Vector3], start_idx: int = 1) -> None:
        self.waypoints = waypoints
        self._next_idx = start_idx

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if self._next_idx >= len(self.waypoints):
            return
        if player.current_order is None:
            player.current_order = MoveOrder(
                target_position=self.waypoints[self._next_idx], sprint=True
            )
            self._next_idx += 1


class NeuralPlayerAI(PlayerAI):
    """Drives a player with a PPO neural network.

    Assign to ``player.ai`` — ``Match.step()`` calls ``act()`` every physics
    tick.  Only samples a new action every ``decision_interval_ticks`` ticks;
    the rest of the time the current order persists unchanged.

    After each decision tick, ``last_transition`` holds the data needed to
    fill the PPO rollout buffer::

        player.ai = NeuralPlayerAI(trainer._sample_action, max_episode_s=120.0)
        ...
        match.step()
        if player.ai.last_transition:
            buffer.add(**player.ai.last_transition, reward=..., done=...)

    ``reset()`` must be called at the start of each episode (done by
    ``ScenarioEnv.reset()`` automatically).
    """

    def __init__(
        self,
        sample_action_fn,
        decision_interval_ticks: int = 15,
        max_episode_s: float = 120.0,
        ema_smoothed: float = 0.0,
        rng=None,
    ) -> None:
        self.sample_action_fn = sample_action_fn
        self.decision_interval_ticks = decision_interval_ticks
        self.max_episode_s = max_episode_s
        self.ema_smoothed = ema_smoothed
        self._rng = rng or random.Random()
        self._ticks_since_decision: int = decision_interval_ticks  # act on first tick
        self._episode_ticks: int = 0
        self.last_transition = None

    def reset(self) -> None:
        self._ticks_since_decision = self.decision_interval_ticks
        self._episode_ticks = 0
        self.last_transition = None

    def act(self, player: "Player", match: "Match", trial_tick: int) -> None:
        from footballcoach.ai.obs.encoder import encode_observation, MAX_OTHER_PLAYERS
        from footballcoach.ai.action.to_orders import apply_action_to_player
        from footballcoach.ai.action.gating import select_action

        self._episode_ticks += 1
        self._ticks_since_decision += 1

        if self._ticks_since_decision < self.decision_interval_ticks:
            return
        # New decision interval — clear stale transition, then sample.
        self.last_transition = None
        self._ticks_since_decision = 0

        time_remaining = max(0.0, self.max_episode_s - self._episode_ticks / 30.0)
        obs = encode_observation(
            match=match,
            player_id=player.player_id,
            time_remaining_s=time_remaining,
            attack_defence_smoothed=self.ema_smoothed,
            rng=self._rng,
        )
        obs_dict = obs.to_torch_dict()

        result = self.sample_action_fn(obs_dict)
        (action, log_prob, value, decision_probs, exec_phys,
         dec_phys, target_slots, raw_exec) = result

        slot_player_ids = [None] * MAX_OTHER_PLAYERS  # safe default; NeuralPlayerAI does not need target resolution
        gating = select_action(decision_probs, exec_phys, target_slots)
        translation = apply_action_to_player(
            gating=gating,
            player=player,
            match=match,
            slot_player_ids=slot_player_ids,
            decision_physical=dec_phys,
        )

        self.last_transition = {
            "obs": {k: v.numpy() for k, v in obs_dict.items()},
            "action": action,
            "log_prob": float(log_prob),
            "value": float(value),
            "raw_exec": raw_exec,
            "illegal_action": translation.illegal_action,
        }
