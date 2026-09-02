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
  StopWhenIdleAI          — issue StopOrder whenever there's no active order; reusable
                            fallback so a player with a single fire-and-forget order
                            (KickOrder/PassOrder/...) decelerates to a stop afterward
                            instead of coasting forever on stale velocity
  NeuralPlayerAI          — wraps a PPO network; samples every N ticks, exposes
                            last_transition for the rollout buffer
  HybridPlayerAI          — NeuralPlayerAI + two independent human-override
                            channels: order override (issue_order) and
                            decision-neuron override (set_decision_override)
"""
from __future__ import annotations

import math
import random

from footballcoach.entities.player import Player, PlayerAI, PlayerState, Team
from footballcoach.engine.match import Match
from footballcoach.engine.movement import effective_acceleration, effective_top_speed, sprint_eta
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    GetPossessionOrder,
    MoveOrder,
    SaveOrder,
    ShootOrder,
    StopOrder,
)


def _default_decision_interval_ticks() -> int:
    """Ticks between decisions for rules-based AI, defaulting to MATCH the
    real trained policy's own decision cadence -- ai_config.json's
    observation.decision_interval_s / observation.sim_dt_s, computed exactly
    the way ScenarioEnv itself derives _ticks_per_decision for NeuralPlayerAI
    -- rather than a separately maintained number in some other config file
    that could silently drift out of sync with it. (0.239s / 0.06s -> 4
    ticks as of this writing.)"""
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["observation"]
    return max(1, round(float(cfg["decision_interval_s"]) / float(cfg["sim_dt_s"])))


class _RulesBasedAI(PlayerAI):
    """Shared base for every rules-based AI in this module.

    Subclasses implement ``decide(player, match, trial_tick)`` (NOT
    ``act()`` -- see PlayerAI's own docstring for why) and get decision-
    cadence throttling for free, with no per-subclass boilerplate: ``act()``
    itself lives once, here, inherited unchanged by everyone below.

    ``decision_interval_ticks=None`` (the default for every rules-based AI
    constructor in this file) resolves to ``_default_decision_interval_
    ticks()`` -- the real trained policy's own cadence -- rather than
    "every tick" the way a bare PlayerAI defaults; pass an explicit value
    to override. This throttles WHICH order/target/sprint-flag the AI
    holds, not the continuous steering underneath it -- Match._process_
    orders calls order.execute() unconditionally every tick regardless of
    this cadence, so movement itself never freezes (see PlayerAI's own
    docstring for the full reasoning)."""

    def __init__(self, decision_interval_ticks: int | None = None) -> None:
        super().__init__(
            decision_interval_ticks=(
                decision_interval_ticks if decision_interval_ticks is not None
                else _default_decision_interval_ticks()
            )
        )

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if player.current_order is None:
            # No order in effect at all -- never something to wait out the
            # decision interval for. Throttling is only meant to gate
            # "should I RECONSIDER my current order" (see this class's own
            # docstring); "should I even HAVE an order" is not optional --
            # a rules-based AI with current_order=None just sits there
            # (Match._process_orders skips execute() entirely when order is
            # None, leaving desired_speed_mode unset -- the player coasts
            # on stale velocity, same as a human letting go of the
            # controls). Confirmed a real bug via direct trace: an order
            # that self-completes mid-interval (e.g. GetPossessionOrder
            # reporting done the instant has_ball is confirmed, which can
            # land on any tick, not just a decision tick) left the player
            # coasting for up to decision_interval_ticks-1 ticks before the
            # next SCHEDULED decision, even though "I have no order" is
            # exactly the situation this AI exists to never leave
            # unresolved. Force a decision now, and resync the cadence
            # counter to it (not the original schedule) so the next
            # ROUTINE reconsideration is a full interval from this forced
            # one, not from wherever the old schedule happened to land.
            self._ticks_since_decision = 0
            self.decide(player, match, trial_tick)
            return
        super().act(player, match, trial_tick)


class Phase1RulesAI(_RulesBasedAI):
    """Chase ball; when possessed, sprint toward the closest point on/in the
    opponent box (deterministic -- the true nearest point on the box
    rectangle to wherever we are, not a random point and not always the
    near edge).  Team-aware via player.team."""

    def decide(self, player: Player, match: Match, trial_tick: int) -> None:
        # Always recompute target_position fresh on every call -- decide()
        # is deliberately memoryless/idempotent given the current player/
        # match state, rather than locking a target in once and holding it
        # for the order's whole lifetime. A prior version only recomputed
        # the target when the order TYPE changed (or, for GetPossessionOrder,
        # when `sprint` flipped) -- confirmed via
        # tests/ai_scenario/test_rules_ai_fresh_vs_persistent_equivalence.py
        # to make a genuinely continuous, persistent Phase1RulesAI diverge
        # measurably (within a few seconds) from bc.py's phase1_labels(),
        # which asks a BRAND NEW Phase1RulesAI instance what it would do
        # once per real decision, with no memory of any previously-locked
        # target -- exactly the mechanism DAgger's rollout labelling and
        # PPO's on-policy BC-aux-loss both rely on. Always recomputing
        # removes the discrepancy at the source (this shared decision
        # function) instead of adding statefulness to every label-generation
        # call site that queries it. Order objects are still only
        # RECREATED (and the transition logged) when the order TYPE
        # actually changes -- recomputing target_position on an unchanged
        # order type is not itself a state transition worth logging.
        if match.ball.possessed_by == player.player_id:
            # Have the ball — run toward the closest point on the box
            # rectangle (deterministic: no rng draw here -- see
            # rng_state_before_kick_noise's docstring in
            # test_rules_ai_nn_replay_equivalence.py for why an order-layer
            # rng draw here was a real source of real/shadow replay
            # divergence whenever it landed on the same tick as another
            # rng-consuming physics event, e.g. a tackle roll).
            if not isinstance(player.current_order, MoveOrder):
                match._log_debug(f"[AI] {player.player_id}: MoveOrder (box run)")
            player.current_order = MoveOrder(
                target_position=_nearest_box_point(player, match),
                sprint=True,
                push_kick_enabled=True,
            )
        else:
            # Don't have the ball — chase it.  Recalculate sprint every tick
            # so the decision tracks changing distances.
            #
            # Against an immobile opponent (can never move or contest the
            # ball), _should_sprint_to_ball's opponent-relative race check
            # is structurally always "no threat, jog is fine" -- but that's
            # wrong here: it also throttles the chase back to a ball THIS
            # player just push-kicked away, which is exactly when speed
            # matters most. Confirmed empirically: one kick, then ~16.6s of
            # JOG-paced chasing back to the player's own kicked ball before
            # recatching it, in a real traced episode. Skip the race check
            # entirely and always sprint when the opponent can't threaten.
            opponent_is_immobile = any(
                opp.player_id != player.player_id and opp.team != player.team and opp.ai is None
                for opp in match.players
            )
            should_sprint = opponent_is_immobile or _should_sprint_to_ball(player, match)
            # Push-kick target: same nearest-point-on-box logic as the
            # has-ball branch above, and the SAME field GetPossessionOrder
            # shares with MoveOrder (orders.py's _try_push_kick).
            if not isinstance(player.current_order, GetPossessionOrder):
                match._log_info(f"[AI] {player.player_id}: GetPossession")
            player.current_order = GetPossessionOrder(
                sprint=should_sprint,
                target_position=_nearest_box_point(player, match),
                push_kick_enabled=True,
            )


class StopWhenIdleAI(PlayerAI):
    """Reusable fallback: whenever the player has no active order, issues a
    ``StopOrder`` so they decelerate to a standstill through their normal
    braking physics, rather than coasting forever on stale velocity.

    ``Match._apply_movement`` deliberately leaves velocity/heading untouched
    on any tick with no movement intent set (``desired_speed_mode is None``)
    — that's a real signal used elsewhere (e.g. BC label generation) to mean
    "no order was in effect this tick", so the engine can't just always
    auto-brake there. For a player driven by a single fire-and-forget order
    (``KickOrder``, ``PassOrder``, a self-cancelling ``SaveOrder``, ...) with
    no AI to decide what's next, that gap is permanent rather than momentary
    — assign this class as ``player.ai`` (instead of leaving ``ai=None``) to
    close it. Any richer AI wanting the same "stop when there's nothing else
    to do" fallback (e.g. after a one-off task completes) can reuse the same
    one-line pattern directly in its own ``act()`` — see ``PassReceiverAI``.
    """

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        if player.current_order is None:
            player.current_order = StopOrder()


_BOX_AIM_SHRINK_FRAC = 0.05  # aim at a box 5% smaller than the real one, not
# right on its true edges -- a corner target (esp. the end-line edge, which
# IS the pitch boundary) is delicate: small overshoot/timing variance can
# carry the ball past the real line. Shrinking the box uniformly around its
# own center pulls every edge a bit inward, trading a negligible amount of
# box depth/width for a real margin of error. Simple deliberately -- not a
# per-axis "nudge 1m in whichever direction" special case.


def _nearest_box_point(player: Player, match: Match) -> Vector3:
    """Nearest point on/in a box 5% smaller than the real opponent box
    (see _BOX_AIM_SHRINK_FRAC) to the player's current position --
    independently clamp x and y to that shrunk box's own range. Clamping x
    to ONLY the near edge is wrong whenever the player already happens to
    be within the box's x-range (e.g. gained possession already deep
    upfield) -- that would send them back out to the shallow edge first, a
    visibly backwards detour instead of the true shortest path in.
    Team-aware via player.team. Used as the shared target for both the
    has-ball box-run MoveOrder and the pre-possession GetPossessionOrder
    push-kick (see Phase1RulesAI.act())."""
    pitch = match.pitch
    half_box_w = (pitch.box_width_m / 2.0) * (1.0 - _BOX_AIM_SHRINK_FRAC)
    if player.team == Team.LEFT:
        box_x_min = pitch.half_length - pitch.box_length_m
        box_x_max = pitch.half_length
    else:
        box_x_min = -pitch.half_length
        box_x_max = -(pitch.half_length - pitch.box_length_m)
    box_center_x = (box_x_min + box_x_max) / 2.0
    half_box_length = (box_x_max - box_x_min) / 2.0 * (1.0 - _BOX_AIM_SHRINK_FRAC)
    box_x_min = box_center_x - half_box_length
    box_x_max = box_center_x + half_box_length
    target_x = max(box_x_min, min(box_x_max, player.position.x))
    target_y = max(-half_box_w, min(half_box_w, player.position.y))
    return Vector3(target_x, target_y, 0.0)


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
    # Self ETA at jog: already moving (use current speed, capped to jog target)
    self_v0 = min(player.speed_mps, jog_speed)
    # No accel shortcut for jog — treat as cruising at jog_speed from v0
    eta_self_jog = dist_self / jog_speed if self_v0 >= jog_speed else sprint_eta(
        dist_self, self_v0, jog_speed,
        effective_acceleration(match.movement_params, player.attributes.acceleration, player.stamina),
    )

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
        opp_v_top = max(
            effective_top_speed(
                match.movement_params,
                opp.attributes.top_speed,
                opp.stamina,
                has_ball=False,
            ),
            0.1,
        )
        opp_accel = effective_acceleration(
            match.movement_params, opp.attributes.acceleration, opp.stamina,
        )
        opp_eta = sprint_eta(dist_opp, opp.speed_mps, opp_v_top, opp_accel)
        if opp_eta * match.order_params.sprint_to_ball_clearance_margin < eta_self_jog:
            return True  # opponent wins the race if we jog

    # Sprint if the ball is heading out of bounds before we can jog there.
    ball_vel_xy = match.ball.velocity.xy()
    ball_speed_xy = ball_vel_xy.length()
    if ball_speed_xy > 0.5 and match.ball.possessed_by is None:
        pitch = match.pitch
        t_out = float("inf")
        bx, by = match.ball.position.x, match.ball.position.y
        vx, vy = ball_vel_xy.x, ball_vel_xy.y
        if abs(vx) > 1e-6:
            tx = ((pitch.half_length if vx > 0 else -pitch.half_length) - bx) / vx
            if tx > 0:
                t_out = min(t_out, tx)
        if abs(vy) > 1e-6:
            ty = ((pitch.half_width if vy > 0 else -pitch.half_width) - by) / vy
            if ty > 0:
                t_out = min(t_out, ty)
        if t_out * match.order_params.sprint_to_ball_clearance_margin < eta_self_jog:
            return True  # ball leaves play before we can jog there

    # No opponents present — default to sprinting.
    return has_opponents is False


_GK_PARKED_TOLERANCE_M = 0.5  # matches MoveOrder's widened arrival tolerance for max_speed_on_arrival_mps=0.0


class StagedGoalkeeperAI(_RulesBasedAI):
    """GK AI: jogs to goal centre, then reacts to shots.

    Enters SaveOrder only when the ball's trajectory is aimed at the GK's goal
    (linear projection to the goal-line x).  Exits SaveOrder as soon as the ball
    becomes possessed by anyone — even an attacker receiving a pass — and jogs
    back to goal centre.  Once parked at goal centre it faces outfield (see
    ``_outfield_heading``) rather than whatever heading the final approach
    step happened to leave it at.
    """

    def __init__(self, jog_to_centre: bool = True, decision_interval_ticks: int | None = None) -> None:
        super().__init__(decision_interval_ticks=decision_interval_ticks)
        self._jog_to_centre = jog_to_centre  # kept for API compat; behaviour unchanged

    def _goal_centre(self, player: Player, match: Match) -> Vector3:
        from footballcoach.entities.player import Team
        if player.team == Team.LEFT:
            return match.pitch.left_goal_centre
        return match.pitch.right_goal_centre

    def _outfield_heading(self, player: Player) -> float:
        """Heading facing away from the GK's own goal line, into the pitch."""
        from footballcoach.entities.player import Team
        return 0.0 if player.team == Team.LEFT else math.pi

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

    def _face_outfield_if_parked(self, player: Player, match: Match) -> None:
        """If already stationary at goal centre, snap heading to face outfield.

        Physics never turns a stationary player (no movement intent to turn
        towards), so without this the GK keeps whatever heading its last
        approach step happened to leave it at -- including, by default,
        heading_rad=0.0 (facing +x), which for a Team.RIGHT GK means facing
        straight into their own net.
        """
        target = self._goal_centre(player, match)
        dist = player.position.xy().distance_to(target.xy())
        if dist <= _GK_PARKED_TOLERANCE_M and player.speed_mps < 0.05:
            player.heading_rad = self._outfield_heading(player)

    def decide(self, player: Player, match: Match, trial_tick: int) -> None:
        ball = match.ball

        # Ball is held by anyone → cease SaveOrder and jog back to goal centre.
        if ball.possessed_by is not None:
            if isinstance(player.current_order, SaveOrder):
                target = self._goal_centre(player, match)
                player.current_order = MoveOrder(
                    target_position=target, sprint=False, max_speed_on_arrival_mps=0.0,
                )
                match._log_debug(f"[AI] {player.player_id}: back to goal centre (ball possessed)")
            elif player.current_order is None:
                # Idle while someone else has the ball (e.g. the attacker's
                # build-up run) -- this is the common case, so make sure
                # heading gets fixed here too, not just in the "ball loose"
                # branch below.
                self._face_outfield_if_parked(player, match)
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
            dist = player.position.xy().distance_to(target.xy())
            if dist <= _GK_PARKED_TOLERANCE_M and player.speed_mps < 0.05:
                self._face_outfield_if_parked(player, match)
            else:
                player.current_order = MoveOrder(
                    target_position=target, sprint=False, max_speed_on_arrival_mps=0.0,
                )


class BallCarrierAttackerAI(_RulesBasedAI):
    """Ball carrier runs toward goal; if their MoveOrder progress stalls
    (distance to target starts increasing), the order completes, or the
    player is already at least as close to goal as the MoveOrder's own
    target (see below), switches to a ShootOrder.

    The shot target is resolved lazily via ``resolve_aim_point()`` -- see
    its docstring: pass a fixed ``aim_point`` for deterministic behaviour
    (tests, or scenarios that don't want smart aiming), or leave it ``None``
    (default) to pick a live, goalkeeper-aware corner at the moment of
    shooting via ``shot_selection.choose_shot_target()``.
    """

    def __init__(
        self, aim_point: Vector3 | None = None, power_fraction: float = 0.9,
        decision_interval_ticks: int | None = None,
    ) -> None:
        super().__init__(decision_interval_ticks=decision_interval_ticks)
        self.aim_point = aim_point
        self.power_fraction = power_fraction
        self._prev_dist_to_target: float | None = None

    def _goal_centre(self, player: Player, match: Match) -> Vector3:
        return match.pitch.right_goal_centre if player.team == Team.LEFT else match.pitch.left_goal_centre

    def resolve_aim_point(self, player: Player, match: Match) -> Vector3:
        """Returns the fixed ``aim_point`` given at construction, if any;
        otherwise computes a smart, goalkeeper-aware corner live (see
        ``shot_selection.choose_shot_target``) using the shooter's and
        opposing goalkeeper's CURRENT positions -- called right at the
        moment of shooting, not baked in at scenario-build time, so it
        reacts to how the play has actually developed."""
        if self.aim_point is not None:
            return self.aim_point
        from footballcoach.shot_selection import choose_shot_target
        gk = next((p for p in match.players if p.team != player.team and p.is_goalkeeper), None)
        return choose_shot_target(player, self.power_fraction, gk, match.pitch, match.rng)

    def decide(self, player: Player, match: Match, trial_tick: int) -> None:
        if match.ball.possessed_by != player.player_id:
            self._prev_dist_to_target = None
            return
        order = player.current_order
        if isinstance(order, MoveOrder):
            # Repulsion steering (see steering.py) can push the ball carrier
            # around an obstacle in a way that carries them closer to goal
            # than the MoveOrder's own target -- e.g. dodging wide of a
            # defender overshoots the intended run. Continuing to chase the
            # now-relatively-behind target would mean moving BACKWARDS
            # (away from goal) just to satisfy the literal waypoint. Once
            # that happens there's no reason to keep running: shoot from
            # here instead, same as a normal arrival/stall. Measured against
            # true goal centre (not the eventual shot target, which may be
            # an off-centre corner picked only once shooting is decided).
            goal_centre = self._goal_centre(player, match)
            goal_dist = player.position.xy().distance_to(goal_centre.xy())
            target_goal_dist = order.target_position.xy().distance_to(goal_centre.xy())
            if goal_dist <= target_goal_dist:
                player.current_order = ShootOrder(
                    aim_point=self.resolve_aim_point(player, match), power_fraction=self.power_fraction,
                    compensate_for_run=False,
                )
                match._log_info(f"[AI] {player.player_id}: ShootOrder (overtook move target)")
                return
            dist = player.position.xy().distance_to(order.target_position.xy())
            prev = self._prev_dist_to_target
            self._prev_dist_to_target = dist
            if prev is not None and dist > prev:
                player.current_order = ShootOrder(
                    aim_point=self.resolve_aim_point(player, match), power_fraction=self.power_fraction,
                    compensate_for_run=False,
                )
                match._log_info(f"[AI] {player.player_id}: ShootOrder (stalled)")
        elif order is None:
            self._prev_dist_to_target = None
            player.current_order = ShootOrder(
                aim_point=self.resolve_aim_point(player, match), power_fraction=self.power_fraction,
                compensate_for_run=False,
            )
            match._log_info(f"[AI] {player.player_id}: ShootOrder")


class PassReceiverAI(PlayerAI):
    """Continues on whatever order the player currently holds until the loose
    ball comes within ``get_possession_radius_m``, then switches to
    ``GetPossessionOrder``.  Once possession is gained, delegates to
    ``after_receipt_ai`` (if provided) for all subsequent ticks; if no
    ``after_receipt_ai`` was given, falls back to ``StopWhenIdleAI``'s
    behaviour (decelerate to a standstill via ``StopOrder``) so the receiver
    doesn't coast forever on stale velocity once ``GetPossessionOrder``
    completes.

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
            elif player.current_order is None:
                player.current_order = StopOrder()
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
        goal_aim_point: Vector3 | None = None,
        shoot_immediately: bool = False,
        run_fraction: float = 0.3,
        power_fraction: float = 0.85,
        get_possession_radius_m: float = 8.0,
    ) -> None:
        """``goal_aim_point``: fixed shot target, or ``None`` (default) for
        a live, goalkeeper-aware corner pick at the moment of shooting --
        see ``BallCarrierAttackerAI.resolve_aim_point()``, which this class
        delegates to via its internal ``_carrier_ai``."""
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
                        aim_point=self._carrier_ai.resolve_aim_point(player, match),
                        power_fraction=self._carrier_ai.power_fraction,
                        compensate_for_run=False,
                    )
                    match._log_info(f"[AI] {player.player_id}: ShootOrder (immediate)")
                else:
                    # goal centre (not the eventual shot target -- see
                    # resolve_aim_point) just to gauge how far to run first.
                    goal_x = self._carrier_ai._goal_centre(player, match).x
                    run_target = Vector3(
                        player.position.x
                        + (goal_x - player.position.x) * self._run_fraction,
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


class SprintWaypointAI(_RulesBasedAI):
    """Issue sequential MoveOrders along a pre-computed waypoint list.
    The first waypoint and ``start_idx`` should be set during scenario build::

        player.current_order = MoveOrder(target_position=waypoints[0], sprint=True)
        player.ai = SprintWaypointAI(waypoints, start_idx=1)
    """

    def __init__(
        self, waypoints: list[Vector3], start_idx: int = 1,
        decision_interval_ticks: int | None = None,
    ) -> None:
        super().__init__(decision_interval_ticks=decision_interval_ticks)
        self.waypoints = waypoints
        self._next_idx = start_idx

    def course_complete(self, player: Player) -> bool:
        """True once the runner has arrived at the final waypoint: no more
        waypoints left to issue AND the last MoveOrder has finished
        (current_order cleared by the engine on arrival)."""
        return self._next_idx >= len(self.waypoints) and player.current_order is None

    def decide(self, player: Player, match: Match, trial_tick: int) -> None:
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
        bc_label_fn=None,
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
        # Optional Callable[[Player, Match], BCLabel] (e.g.
        # footballcoach.ai.ppo.bc.phase1_labels_for_player), called from
        # WITHIN act() -- see its call site below for why: it must run at
        # the exact same instant the observation is encoded, before
        # Match._apply_movement() advances the player for this tick, or the
        # resulting label describes a state one physics tick later than the
        # observation it's meant to accompany (see phase1_labels_for_player's
        # own "CRITICAL -- CALLER-SIDE TIMING" docstring section for the
        # full story -- this is exactly the fix for that bug). None
        # (default) = no label computed, preserving prior behaviour for
        # every caller that doesn't need one (pure PPO fine-tuning phases,
        # eval, etc).
        self.bc_label_fn = bc_label_fn

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

        # Compute the BC label (if configured) RIGHT HERE, before anything
        # else this tick touches the player -- this is the exact instant
        # `obs` was encoded from, still inside Match._process_orders(),
        # BEFORE Match._apply_movement() advances position/velocity/heading
        # for this tick. See bc_label_fn's own docstring (__init__ above)
        # and phase1_labels_for_player's "CRITICAL -- CALLER-SIDE TIMING"
        # section for why this specific placement (not "any time before
        # act() returns", and definitely not from a caller after a whole
        # env.step() has already resolved) is the actual fix, not just a
        # convenient one. sample_action_fn/apply_action_to_player below only
        # ever set desired_direction/desired_speed_mode/kicked_this_tick/
        # kick_armed/tackle_armed -- never position/velocity/heading -- and
        # phase1_labels_for_player()'s own snapshot/reset/restore already
        # protects against those specific flags regardless of call order, so
        # computing the label before vs. after sample_action_fn runs is
        # equivalent; before is simplest to reason about.
        bc_label_arr = None
        if self.bc_label_fn is not None:
            bc_label_arr = self.bc_label_fn(player, match).to_array()

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
            "bc_label": bc_label_arr,
            "raw_exec": raw_exec,
            "head_log_probs": head_log_probs,
            "illegal_action": translation.illegal_action,
        }


class HybridPlayerAI(NeuralPlayerAI):
    """``NeuralPlayerAI`` plus two independent human/rules-based override
    channels, so a single player can be a mix of neural network control and
    direct human/rules intervention.  Both channels are opt-in and orthogonal
    — using one does not disable the other::

        player.ai = HybridPlayerAI(trainer._sample_action)

        # Channel 1 — order override: bypass the neural net entirely and run
        # a real Order (MoveOrder, ShootOrder, ...) through the normal engine
        # order machinery, exactly like a rules-based AI would.  Useful for
        # "take direct control" style human intervention.
        player.ai.issue_order(MoveOrder(target_position=..., sprint=True))

        # Channel 2 — decision-neuron override: force one or more decision
        # heads' probabilities before the winner-take-all gating rule runs,
        # while the execution network still supplies the physical motor
        # output (move_direction, sprint, kick, tackle).  This is "give the
        # neural net an order via its own decision neurons" rather than
        # bypassing it — e.g. force the 'move' head to fire this tick:
        player.ai.set_decision_override("move", 1.0)
        # Clear a single override, or all of them:
        player.ai.set_decision_override("move", None)
        player.ai.clear_decision_overrides()

    **Order override** (channel 1) takes priority over the decision-neuron
    override and over the network's own decision sampling for as long as the
    override order is in progress — ``act()`` assigns it to
    ``player.current_order`` and returns without touching the network at all
    that tick (mirrors how ``Phase1RulesAI``/other rules AIs work — the
    engine's own ``_process_orders`` then calls ``order.execute()`` right
    after ``act()`` returns).  Once the order completes (engine clears
    ``player.current_order`` back to ``None``), control reverts to the
    neural network automatically on the following tick — no manual
    "hand back control" step needed.

    **Decision-neuron override** (channel 2) only affects ticks where the
    network actually samples a fresh decision (every
    ``decision_interval_ticks`` ticks) and only when no order override is
    in progress.  Forcing a head's probability to ``1.0`` guarantees it wins
    ``select_action``'s winner-take-all rule (barring another forced head
    with higher `_HEAD_ORDER` priority); forcing to ``0.0`` guarantees it
    cannot fire.  Valid head names: ``shoot``, ``pass_``, ``move``,
    ``tackle``, ``get_possession``, ``mark``, ``hold_position`` — see
    ``ai/action/gating.py``'s ``_HEAD_ORDER``.

    This class is deliberately generic/extensible: it does not know or care
    whether the override values come from a human clicking in the UI, from a
    rules-based `PlayerAI`, or from a scripted test — anything that can call
    ``issue_order``/``set_decision_override`` can drive it.  See
    ``ai/knowledge.md``'s "HybridPlayerAI" section for the full design
    rationale and the Orders-vs-execution-network boundary this respects.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._order_override: object | None = None
        self._order_override_assigned: bool = False
        self._decision_overrides: dict[str, float] = {}

    def reset(self) -> None:
        super().reset()
        self._order_override = None
        self._order_override_assigned = False
        self._decision_overrides = {}

    # -- channel 1: order override -------------------------------------------

    def issue_order(self, order) -> None:
        """Take direct control this tick: assign *order* and let the engine's
        normal order machinery execute it, bypassing the neural network
        entirely until the order completes."""
        self._order_override = order
        self._order_override_assigned = False

    def clear_order_override(self) -> None:
        """Cancel any in-progress order override and return control to the
        neural network immediately (the player's current order, if any, is
        left as-is; it will simply no longer be refreshed by this AI)."""
        self._order_override = None
        self._order_override_assigned = False

    @property
    def order_override_active(self) -> bool:
        return self._order_override is not None

    # -- channel 2: decision-neuron override ---------------------------------

    def set_decision_override(self, head_name: str, value: float | None) -> None:
        """Force decision head *head_name*'s probability to *value* (e.g. 1.0
        to guarantee it fires, 0.0 to suppress it) on the next decision tick.
        Pass ``value=None`` to clear the override for that head."""
        if value is None:
            self._decision_overrides.pop(head_name, None)
        else:
            self._decision_overrides[head_name] = float(value)

    def clear_decision_overrides(self) -> None:
        self._decision_overrides = {}

    # -- act() ----------------------------------------------------------------

    def act(self, player: "Player", match: "Match", trial_tick: int) -> None:
        # Channel 1 takes priority: while an order override is active, skip
        # the neural network entirely (no sampling, no last_transition) and
        # let the engine's normal order-execution machinery run it.
        if self._order_override is not None:
            if not self._order_override_assigned:
                player.current_order = self._order_override
                self._order_override_assigned = True
                return
            if player.current_order is None:
                # The override order completed (or was cleared externally) —
                # hand control back to the neural network from here on.
                self._order_override = None
                self._order_override_assigned = False
            else:
                return  # still in progress; do nothing further this tick

        if not self._decision_overrides:
            super().act(player, match, trial_tick)
            return

        # Channel 2: sample normally, but patch decision_probs before gating
        # on decision ticks only (between-decision ticks just re-apply the
        # last cached gating, same as NeuralPlayerAI).
        from footballcoach.ai.obs.encoder import encode_observation, MAX_OTHER_PLAYERS
        from footballcoach.ai.action.apply_nn_action import apply_action_to_player
        from footballcoach.ai.action.gating import select_action

        self._episode_ticks += 1
        self._ticks_since_decision += 1

        if self._ticks_since_decision < self.decision_interval_ticks:
            if self._last_gating is not None:
                apply_action_to_player(
                    gating=self._last_gating,
                    player=player,
                    match=match,
                    slot_player_ids=[None] * 21,
                    decision_physical={},
                )
            return

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

        decision_probs = dict(decision_probs)
        decision_probs.update(self._decision_overrides)

        slot_player_ids = [None] * MAX_OTHER_PLAYERS
        gating = select_action(decision_probs, exec_phys, target_slots)
        self._last_gating = gating
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
