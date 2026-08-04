"""Rules-based AI controllers.

Each class is a ``PlayerAI`` subclass — assign one to ``player.ai`` and
``Match.step()`` will call ``ai.act(player, match, tick)`` automatically
every physics tick.

Classes:
  Phase1RulesAI           — chase ball; when possessed, sprint to opponent box entry
  StagedGoalkeeperAI      — wait until goal-centre MoveOrder completes, then SaveOrder
  BallCarrierAttackerAI   — move to goal; switch to ShootOrder when progress stalls
  PassReceiverAI          — continue on current order until ball within radius, then GetPossession
  BallReceiverThenShootAI — wait for ball receipt, set up shot/run, then delegate
  SprintWaypointAI        — issue sequential MoveOrders along a waypoint list
  NeuralPlayerAI          — wraps a PPO network; samples every N ticks, exposes
                            last_transition for the rollout buffer
"""
from __future__ import annotations

import random

from footballcoach.entities.player import Player, PlayerAI, PlayerState, Team
from footballcoach.engine.match import Match
from footballcoach.engine.movement import effective_top_speed
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
            # Have the ball — run toward a random point inside the opponent box.
            if not isinstance(player.current_order, MoveOrder):
                pitch = match.pitch
                half_box_w = pitch.box_width_m / 2.0
                # x: random within the full box depth (inner edge → goal line)
                if player.team == Team.LEFT:
                    box_inner_x = pitch.half_length - pitch.box_length_m
                    target_x = match.rng.uniform(box_inner_x, pitch.half_length)
                else:
                    box_inner_x = -(pitch.half_length - pitch.box_length_m)
                    target_x = match.rng.uniform(-pitch.half_length, box_inner_x)
                # y: random in the nearest half of the box to the player
                if player.position.y >= 0.0:
                    target_y = match.rng.uniform(0.0, half_box_w)
                else:
                    target_y = match.rng.uniform(-half_box_w, 0.0)
                player.current_order = MoveOrder(
                    target_position=Vector3(target_x, target_y, 0.0),
                    sprint=True,
                    push_kick_enabled=True,
                )
                match._log_debug(f"[AI] {player.player_id}: MoveOrder (box run)")
        else:
            # Don't have the ball — chase it.  Recalculate sprint every tick
            # so the decision tracks changing distances; only re-issue the
            # order when the flag actually flips to avoid resetting its state.
            should_sprint = _should_sprint_to_ball(player, match)
            if not isinstance(player.current_order, GetPossessionOrder):
                player.current_order = GetPossessionOrder(sprint=should_sprint)
                match._log_info(f"[AI] {player.player_id}: GetPossession")
            elif player.current_order.sprint != should_sprint:
                player.current_order = GetPossessionOrder(sprint=should_sprint)


def _should_sprint_to_ball(player: Player, match: Match) -> bool:
    """Return True if any active opponent would reach the ball before us at jog pace.

    Compares each opponent's sprint ETA to the ball against our own jog ETA.
    If even one opponent can beat us while they sprint and we jog, we sprint.
    Falls back to sprinting when no opponents are present (safe default).
    """
    ball_pos = match.ball.position
    dist_self = (ball_pos - player.position).length()
    if dist_self < 0.1:
        return False  # already at the ball

    jog_speed = effective_top_speed(
        match.movement_params,
        player.attributes.top_speed,
        player.stamina,
        has_ball=False,
    ) * 0.5
    jog_speed = max(jog_speed, 0.1)
    eta_self_jog = dist_self / jog_speed

    has_opponents = False
    for opp in match.players:
        if opp.player_id == player.player_id:
            continue
        if opp.team == player.team:
            continue
        if opp.state == PlayerState.INACTIVE_TACKLED:
            continue
        has_opponents = True
        dist_opp = (ball_pos - opp.position).length()
        opp_sprint_speed = max(
            effective_top_speed(
                match.movement_params,
                opp.attributes.top_speed,
                opp.stamina,
                has_ball=False,
            ),
            0.1,
        )
        if dist_opp / opp_sprint_speed < eta_self_jog:
            return True  # opponent wins the race if we jog

    # No opponents present — default to sprinting.
    return has_opponents is False


class StagedGoalkeeperAI(PlayerAI):
    """GK AI: jogs to goal centre, then reacts to shots.

    Enters SaveOrder only when the ball's trajectory is aimed at the GK's goal
    (linear projection to the goal-line x).  Exits SaveOrder as soon as the ball
    becomes possessed by anyone — even an attacker receiving a pass — and jogs
    back to goal centre.
    """

    def __init__(self, jog_to_centre: bool = True) -> None:
        self._jog_to_centre = jog_to_centre  # kept for API compat; behaviour unchanged

    def _goal_centre(self, player: Player, match: Match) -> Vector3:
        from footballcoach.entities.player import Team
        if player.team == Team.LEFT:
            return match.pitch.left_goal_centre
        return match.pitch.right_goal_centre

    def _ball_aimed_at_goal(self, player: Player, match: Match) -> bool:
        """True if the loose ball's straight-line path projects into the goal mouth."""
        from footballcoach.entities.player import Team
        if match.ball.possessed_by is not None:
            return False
        ball_speed = match.ball.velocity.length()
        if ball_speed < 1.0:
            return False

        pitch = match.pitch
        vx = match.ball.velocity.x
        vy = match.ball.velocity.y
        bx = match.ball.position.x
        by = match.ball.position.y

        if player.team == Team.LEFT:
            goal_x = -pitch.half_length
            if vx >= 0.0:
                return False  # ball moving away from left goal
        else:
            goal_x = pitch.half_length
            if vx <= 0.0:
                return False  # ball moving away from right goal

        t = (goal_x - bx) / vx
        if t < 0.0:
            return False
        proj_y = by + vy * t
        half_goal_w = pitch.goal_width_m / 2.0 * 1.3  # 30% margin for early reaction
        return abs(proj_y) < half_goal_w

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        ball = match.ball

        # Ball is held by anyone → cease SaveOrder and jog back to goal centre.
        if ball.possessed_by is not None:
            if isinstance(player.current_order, SaveOrder):
                target = self._goal_centre(player, match)
                player.current_order = MoveOrder(
                    target_position=target, sprint=False, max_speed_on_arrival_mps=0.0,
                )
                match._log_debug(f"[AI] {player.player_id}: back to goal centre (ball possessed)")
            return

        # Ball is loose — enter SaveOrder if aimed at our goal.
        if self._ball_aimed_at_goal(player, match):
            if not isinstance(player.current_order, SaveOrder):
                player.current_order = SaveOrder(auto_sprint=True)
                match._log_info(f"[AI] {player.player_id}: SaveOrder")
            return

        # Ball is loose but not threatening — fill None with goal-centre jog.
        if player.current_order is None:
            target = self._goal_centre(player, match)
            player.current_order = MoveOrder(
                target_position=target, sprint=False, max_speed_on_arrival_mps=0.0,
            )


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
                match._log_info(f"[AI] {player.player_id}: ShootOrder (stalled)")
        elif order is None:
            self._prev_dist_to_target = None
            player.current_order = ShootOrder(
                aim_point=self.aim_point, power_fraction=self.power_fraction
            )
            match._log_info(f"[AI] {player.player_id}: ShootOrder")


class PassReceiverAI(PlayerAI):
    """Continues on whatever order the player currently holds until the loose
    ball comes within ``get_possession_radius_m``, then switches to
    ``GetPossessionOrder``.  Once possession is gained, delegates to
    ``after_receipt_ai`` (if provided) for all subsequent ticks.

    Typical usage — pair with an initial ``MoveOrder`` set at scenario-build
    time so the receiver runs toward a useful position and only commits to
    chasing the ball when it is realistically catchable::

        receiver.current_order = MoveOrder(target_position=..., sprint=True)
        receiver.ai = PassReceiverAI(get_possession_radius_m=8.0,
                                     after_receipt_ai=BallCarrierAttackerAI(aim))
    """

    def __init__(
        self,
        get_possession_radius_m: float = 8.0,
        after_receipt_ai: "PlayerAI | None" = None,
    ) -> None:
        self.get_possession_radius_m = get_possession_radius_m
        self.after_receipt_ai = after_receipt_ai
        self._switched_to_gp = False
        self._received = False

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if self._received:
            if self.after_receipt_ai is not None:
                self.after_receipt_ai.act(player, match, trial_tick)
            return

        if match.ball.possessed_by == player.player_id:
            self._received = True
            return

        if not self._switched_to_gp:
            ball_dist = (match.ball.position - player.position).length()
            if ball_dist <= self.get_possession_radius_m:
                player.current_order = GetPossessionOrder(sprint=True)
                self._switched_to_gp = True


class BallReceiverThenShootAI(PlayerAI):
    """AI for a player waiting to receive a pass, then shooting or running.

    Phase 1: continues on whatever order the player holds at build time;
    switches to ``GetPossessionOrder`` when the ball comes within
    ``get_possession_radius_m`` (default 8 m).
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
        get_possession_radius_m: float = 8.0,
    ) -> None:
        self.goal_aim_point = goal_aim_point
        self._shoot_immediately = shoot_immediately
        self._run_fraction = run_fraction
        self._get_possession_radius_m = get_possession_radius_m
        self._received = False
        self._switched_to_gp = False
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
                    match._log_info(f"[AI] {player.player_id}: ShootOrder (immediate)")
                else:
                    run_target = Vector3(
                        player.position.x
                        + (self.goal_aim_point.x - player.position.x) * self._run_fraction,
                        player.position.y,
                        0.0,
                    )
                    player.current_order = MoveOrder(target_position=run_target, sprint=True)
                    match._log_debug(f"[AI] {player.player_id}: MoveOrder (pre-shot run)")
            elif not self._switched_to_gp:
                ball_dist = (match.ball.position - player.position).length()
                if ball_dist <= self._get_possession_radius_m:
                    player.current_order = GetPossessionOrder(sprint=True)
                    self._switched_to_gp = True
                    match._log_info(f"[AI] {player.player_id}: GetPossession (ball nearby)")
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
        self._last_gating = None  # cached gating result; re-applied every tick

    def reset(self) -> None:
        self._ticks_since_decision = self.decision_interval_ticks
        self._episode_ticks = 0
        self.last_transition = None
        self._last_gating = None

    def act(self, player: "Player", match: "Match", trial_tick: int) -> None:
        from footballcoach.ai.obs.encoder import encode_observation, MAX_OTHER_PLAYERS
        from footballcoach.ai.action.apply_nn_action import apply_action_to_player
        from footballcoach.ai.action.gating import select_action

        self._episode_ticks += 1
        self._ticks_since_decision += 1

        if self._ticks_since_decision < self.decision_interval_ticks:
            # Re-apply last cached gating so desired_direction/speed_mode
            # are set every tick (not just on decision ticks).
            if self._last_gating is not None:
                apply_action_to_player(
                    gating=self._last_gating,
                    player=player,
                    match=match,
                    slot_player_ids=[None] * 21,
                    decision_physical={},
                )
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
         dec_phys, target_slots, raw_exec, head_log_probs) = result

        slot_player_ids = [None] * MAX_OTHER_PLAYERS  # safe default; NeuralPlayerAI does not need target resolution
        gating = select_action(decision_probs, exec_phys, target_slots)
        self._last_gating = gating  # cache so between-decision ticks can re-apply direction/speed
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
            "head_log_probs": head_log_probs,
            "illegal_action": translation.illegal_action,
        }
