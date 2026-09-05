"""Match: the top-level engine tying together movement, ball physics,
collision, kicking, tackling, possession, offside, and scoring into a single
steppable simulation. This is the main entry point other code (tests, a
future UI, a future RL training loop) should use.

See engine/knowledge.md for the overall tick order and design rationale.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from footballcoach.ui.gamelog import LogLevel

from footballcoach.config import load_orders_config, load_physics_config, require_section
from footballcoach.engine.ball_physics import BallPhysicsParams, step_ball, resolve_goal_boundary
from footballcoach.engine.collision import (
    CollisionParams,
    are_touching,
    can_tackle,
    resolve_all_overlaps,
    resolve_ball_block_by_inactive_players,
)
from footballcoach.engine.goalkeeping import GoalkeepingParams, early_intercept_target, save_target_position
from footballcoach.engine.interception import intercept_target
from footballcoach.engine.kicking import KickingParams, PassingParams, kick_ball, pass_ball, pass_speed_mps, compensate_power_for_run_mult, running_power_multiplier
from footballcoach.engine.movement import (
    MovementParams,
    SpeedMode,
    angle_diff,
    effective_acceleration,
    effective_top_speed,
    regen_stamina,
    step_player_towards,
)
from footballcoach.engine.possession import BallPickupParams, ControlTimeParams, can_pick_up_ball, control_time_s, compute_difficulty
from footballcoach.engine.scoring import Scoreboard, check_goal
from footballcoach.engine.tackling import TacklingParams, apply_tackle_result, attempt_tackle, tackle_angle_modifier
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder,
    GetPossessionOrder,
    JockeyParams,
    KickOrder,
    MarkOrder,
    MoveOrder,
    OrderLayerParams,
    OrderStatus,
    PassOrder,
    SaveOrder,
    ShootOrder,
    StopOrder,
    TackleApproachParams,
)
from footballcoach.steering import RepulsionParams, compute_repulsion

import logging
log = logging.getLogger("footballcoach.match")

from footballcoach.engine.match_logger import MatchLogger

@dataclass(frozen=True)
class MarkingParams:
    """Config for the ``MarkOrder`` AI behaviour — loaded from
    ``ai_config.json["marking"]``."""
    mark_intercept_radius_m: float = 4.0
    mark_standoff_m: float = 1.5

    @staticmethod
    def from_config() -> "MarkingParams":
        d = require_section(load_orders_config(), "marking", "orders.json")
        return MarkingParams(
            mark_intercept_radius_m=d.get("mark_intercept_radius_m", 4.0),
            mark_standoff_m=d.get("mark_standoff_m", 1.5),
        )


@dataclass(frozen=True)
class BoundaryBrakingParams:
    """Config for ``Match._run_get_possession_behaviour``'s loose-ball
    boundary-safety braking (and its own-team-touched-last abandonment) —
    loaded from ``orders.json["boundary_braking"]``. See that section's own
    ``_comment``/``_comment_brake_buffer``/``_comment_abandon_margin`` for
    the full reasoning; kept here as one config-backed dataclass rather than
    local magic-number constants so both are independently tunable."""
    brake_buffer_m: float = 0.6
    abandon_margin_m: float = -1.0

    @staticmethod
    def from_config() -> "BoundaryBrakingParams":
        d = require_section(load_orders_config(), "boundary_braking", "orders.json")
        return BoundaryBrakingParams(
            brake_buffer_m=d.get("brake_buffer_m", 0.6),
            abandon_margin_m=d.get("abandon_margin_m", -1.0),
        )


@dataclass(frozen=True)
class InterceptionParams:
    """Config for the deceleration-aware loose-ball intercept solve (see
    ``engine.interception.intercept_target``'s own ``target_decel_mps2``
    docstring for the full reasoning) — loaded from
    ``orders.json["interception"]``.

    ``assumed_ball_decel_mps2`` defaults to the ball's own REAL configured
    ground-friction deceleration (``ball_physics.rolling_friction_coefficient
    * world.gravity_mps2`` — 0.05 * 9.81 = 0.4905 m/s^2 by default), but is
    kept as its own independently-tunable value here rather than reading
    physics.json directly at call time: the intercept solve only needs a
    reasonable ESTIMATE of how fast the ball is slowing down, not an exact
    match to the live ground-physics constant (a kicked ball spends part of
    its flight airborne, where drag behaves differently from rolling
    friction) — separating the two lets this be retuned for the solve's own
    behaviour without touching real ball physics, or vice versa.
    ``0.0`` disables the deceleration-aware solve entirely, reproducing the
    exact prior (constant-velocity-only) behaviour."""
    assumed_ball_decel_mps2: float = 0.4905

    @staticmethod
    def from_config() -> "InterceptionParams":
        d = require_section(load_orders_config(), "interception", "orders.json")
        return InterceptionParams(
            assumed_ball_decel_mps2=d.get("assumed_ball_decel_mps2", 0.4905),
        )


@dataclass
class Match:
    pitch: Pitch
    players: list[Player]
    ball: Ball
    rng_reduction: float = 0.3
    dt_s: float = 1.0 / 30.0
    offside_enabled: bool = True
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    rng: random.Random = field(default_factory=random.Random)

    movement_params: MovementParams = field(default_factory=MovementParams.from_config)
    order_params: OrderLayerParams = field(default_factory=OrderLayerParams.from_config)
    ball_physics_params: BallPhysicsParams = field(default_factory=BallPhysicsParams.from_config)
    kicking_params: KickingParams = field(default_factory=KickingParams.from_config)
    passing_params: PassingParams = field(default_factory=PassingParams.from_config)
    control_time_params: ControlTimeParams = field(default_factory=ControlTimeParams.from_config)
    tackling_params: TacklingParams = field(default_factory=TacklingParams.from_config)
    goalkeeping_params: GoalkeepingParams = field(default_factory=GoalkeepingParams.from_config)
    repulsion_params: RepulsionParams = field(default_factory=RepulsionParams.from_config)
    collision_params: CollisionParams = field(default_factory=CollisionParams.from_config)
    marking_params: MarkingParams = field(default_factory=MarkingParams.from_config)
    ball_pickup_params: BallPickupParams = field(default_factory=BallPickupParams.from_config)
    boundary_braking_params: BoundaryBrakingParams = field(default_factory=BoundaryBrakingParams.from_config)
    tackle_approach_params: TackleApproachParams = field(default_factory=TackleApproachParams.from_config)
    jockey_params: JockeyParams = field(default_factory=JockeyParams.from_config)
    interception_params: InterceptionParams = field(default_factory=InterceptionParams.from_config)

    paused: bool = False
    time_s: float = 0.0

    # Pickup radius: how close a loose ball must be to a player before that
    # player begins the first-touch control-time countdown. Synced from
    # ball_pickup_params.pickup_radius_m in __post_init__ (single source of
    # truth in config) - kept as its own field since it's read as a simple
    # attribute in several places (e.g. tests, GetPossession order homing
    # logic) that predate ball_pickup_params. Do not set this independently
    # of ball_pickup_params; override ball_pickup_params instead.
    pickup_radius_m: float = 0.4

    # Sim-seconds to keep the ball in the net after a goal before resetting
    # to centre.  0.0 = immediate reset (tests / headless use).  Set to
    # physics.json["ui"]["goal_linger_s"] when creating a match for the UI.
    goal_linger_s: float = 0.0
    _goal_linger_remaining_s: float = field(default=0.0, init=False, repr=False)

    # Optional callback invoked with (level, message) for UI game-log display.
    # None by default so tests / headless use incur zero cost.
    log_callback: Callable[["LogLevel", str], None] | None = field(default=None, repr=False)

    # Optional match event logger for post-hoc inspection.  None by default
    # so training/tests incur zero cost when logging is not needed.
    match_logger: MatchLogger | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.pickup_radius_m = self.ball_pickup_params.pickup_radius_m
        self._validate_heading_velocity_alignment()

    def _validate_heading_velocity_alignment(self) -> None:
        """Fail loudly if any player is constructed facing a direction other
        than the one they're moving in -- a real scenario-construction bug
        (confirmed: ui/scenarios.py used to draw heading_rad and velocity's
        direction as two INDEPENDENT random values, so a moving player's
        heading only matched its velocity by a ~1.4% coincidence). Skipped
        for near-stationary players (speed below _min_speed_mps), whose
        heading isn't physically constrained by anything.

        A player facing a direction other than the one they're moving in is
        never physically valid -- there is no legitimate case for it, so
        this is a hard raise, not a warning. Known casualty: tests/scenario/
        test_control_behaviour.py's aerial-ball-control/head-on-tackle-
        immunity fixture constructs a receiver facing backward relative to
        its velocity for reasons unrelated to what it's testing -- fix that
        fixture's heading_rad rather than relaxing this check.

        Only checked ONCE, here at construction -- not re-validated every
        tick. After t=0, step_player_towards turns heading toward whatever
        direction the player is actually moving/steering at a bounded rate
        (see its own turn-rate-limited rotation), so real, gradual heading/
        velocity divergence mid-turn during normal play is expected and
        correct, not a bug to flag.
        """
        _min_speed_mps = 0.05
        _max_misalignment_deg = 5.0
        for p in self.players:
            speed = p.velocity.length_xy()
            if speed < _min_speed_mps:
                continue
            vel_heading = math.atan2(p.velocity.y, p.velocity.x)
            diff_deg = math.degrees(abs(angle_diff(p.heading_rad, vel_heading)))
            if diff_deg > _max_misalignment_deg:
                raise ValueError(
                    f"Player {p.player_id!r} constructed with heading_rad={p.heading_rad:.3f} "
                    f"but velocity {p.velocity!r} points at {vel_heading:.3f} rad "
                    f"({diff_deg:.1f} deg apart, speed={speed:.2f} m/s) -- heading and "
                    f"velocity must agree at match construction time (derive heading "
                    f"from velocity's direction, not an independent random draw)."
                )

    def player_by_id(self, player_id: str) -> Player:
        for p in self.players:
            if p.player_id == player_id:
                return p
        raise KeyError(f"no player with id {player_id}")

    def ball_carrier(self) -> Player | None:
        if self.ball.possessed_by is None:
            return None
        return self.player_by_id(self.ball.possessed_by)

    def step(self) -> None:
        if self.paused:
            return
        dt = self.dt_s

        # Snapshot pre-movement positions for the swept ball-pickup check
        # (_update_loose_ball_pickup) -- must be captured before
        # _process_orders/_apply_movement actually move anyone this tick.
        # Captured for all players (not just currently-ACTIVE ones) so a
        # player whose INACTIVE_TACKLED/CONTROLLING_BALL timer expires
        # later this same tick (in _update_state_timers, right below) still
        # has a valid pre-tick position on record.
        pre_tick_player_positions = {p.player_id: p.position for p in self.players}

        self._update_state_timers(dt)
        self._process_orders(dt)
        self._apply_movement(dt)
        self._sync_possessed_ball()
        if self.match_logger is not None:
            self.match_logger.check_consistency(self.time_s, self.ball.position, self.ball.possessed_by)

        # Advance a loose ball's free-flight physics *before* checking for
        # pickup, so a ball that was just kicked this tick has already moved
        # away from the kicker's feet before we check whether anyone is close
        # enough to start controlling it (also matters for can_pick_up_ball's
        # closing-velocity check below, which needs the ball's post-kick
        # trajectory to correctly read as "moving away" from the kicker).
        #
        # Ball mid-control (CONTROLLING_BALL) is now possessed immediately on
        # contact, so step_ball() is already a no-op for it (possessed_by is set).
        pre_flight_position = self.ball.position
        if self.ball.possessed_by is None:
            step_ball(self.ball, dt, self.ball_physics_params)
            resolve_ball_block_by_inactive_players(
                self.ball, self.players, pre_flight_position, self.ball_physics_params.block_restitution
            )
            resolve_goal_boundary(self.ball, self.pitch, self.ball_physics_params)

        self._update_loose_ball_pickup(dt, pre_flight_position, pre_tick_player_positions)

        self._check_armed_tackles()
        self._check_head_on_tackles()
        resolve_all_overlaps(self.players, collision_params=self.collision_params)

        # Goal linger: if a goal was recently scored, count down before
        # resetting the ball.  Skip further goal detection during the linger.
        if self._goal_linger_remaining_s > 0.0:
            self._goal_linger_remaining_s -= dt
            if self._goal_linger_remaining_s <= 0.0:
                self._goal_linger_remaining_s = 0.0
                self._reset_after_goal()
        else:
            self._check_goal()

        self.time_s += dt

    # -- logging helpers (zero cost when log_callback is None) ---------------

    def _log_info(self, msg: str) -> None:
        if self.log_callback is not None:
            from footballcoach.ui.gamelog import LogLevel  # local import: only paid when UI is live
            self.log_callback(LogLevel.INFO, msg)

    def _log_debug(self, msg: str) -> None:
        if self.log_callback is not None:
            from footballcoach.ui.gamelog import LogLevel
            self.log_callback(LogLevel.DEBUG, msg)

    def _complete_order(self, order) -> None:
        """Transition *order* to COMPLETE and fire its on_complete callback if set."""
        order.status = OrderStatus.COMPLETE
        if order.on_complete is not None:
            order.on_complete()

    def _apply_gk_immune_penalty(self, player: Player) -> None:
        """Apply the auto-fail penalty to a tackler who charged into a GK protected in their own box.
        The GK is untouched; only the tackler is penalised.
        """
        player.velocity = player.velocity * self.tackling_params.tackle_attempt_tackler_speed_mult
        player.state = PlayerState.INACTIVE_TACKLED
        player.state_timer_s = self.tackling_params.tackle_cooldown_s

    def _apply_movement(self, dt: float) -> None:
        """Apply deferred movement intent set by orders/AI during _process_orders.

        Orders and AI set ``player.desired_direction`` and ``player.desired_speed_mode`` each tick.
        This is the ONLY place ``step_player_towards`` and stamina drain are called for locomotion.

        Every player must have an active AI or Order setting movement intent every tick -- either
        via a persistent Order's execute() (called unconditionally every tick regardless of AI
        decision cadence, see _RulesBasedAI/_process_orders) or via a neural AI re-applying its
        cached gating on non-decision ticks (see NeuralPlayerAI.act()). A player that should just
        stand there needs an explicit AI/Order that says so (e.g. StopWhenIdleAI, or a persistent
        StopOrder) -- NOT the absence of one. ``desired_speed_mode is None`` here means NOTHING set
        an intent this tick: always a real bug (a player with no AI and no order, or an AI/order
        that silently failed to re-assert intent) -- fail loudly rather than silently defaulting to
        any particular behaviour (coasting on stale velocity, or an implicit brake-to-standstill),
        which would just as easily paper over a real AI/order bug as it would a deliberately-idle
        test player. Mirrors _process_orders' own "every order must explicitly decide movement
        intent... fail loudly instead of coasting" check for the order-executed-but-forgot-to-set-
        intent case -- this is the analogous check for the "nothing ran at all" case.
        ``desired_speed_mode`` is cleared to ``None`` after application so each tick is independent.
        ``player.last_desired_speed_mode`` is set to whatever was just consumed and is NOT cleared --
        see its docstring on ``Player`` (used by ai/obs/encoder.py to read "current movement intent").
        """
        for player in self.players:
            if player.desired_speed_mode is None:
                raise RuntimeError(
                    f"{player.player_id!r} has no movement intent this tick (desired_speed_mode "
                    f"is None) -- every player must have an active AI or order setting this every "
                    f"tick (ai={player.ai!r}, current_order={player.current_order!r}). A player "
                    f"that should just stand still needs an explicit AI/Order that says so (e.g. "
                    f"StopWhenIdleAI), not the absence of one."
                )
            has_ball = (
                self.ball.possessed_by == player.player_id
                or player.state == PlayerState.CONTROLLING_BALL
            )
            speed_mode = player.desired_speed_mode
            step_player_towards(player, player.desired_direction, speed_mode, dt, self.movement_params, has_ball)
            player.stamina = _drain_if_sprinting(self.movement_params, player, speed_mode is SpeedMode.SPRINT, dt)
            player.last_desired_speed_mode = speed_mode  # NOT cleared -- see Player.last_desired_speed_mode
            player.desired_speed_mode = None  # consumed; reset for next tick

    def _set_possession(self, player_id: str | None) -> None:
        """Single write-path for ball possession.

        Every place that changes ``ball.possessed_by`` must go through here so
        that ``on_possession_gained`` callbacks fire reliably.  Direct writes to
        ``self.ball.possessed_by`` outside this method are forbidden.
        """
        old = self.ball.possessed_by
        self.ball.possessed_by = player_id
        if player_id is not None and player_id != old:
            self.ball.last_touched_by_player_id = player_id
            p = self.player_by_id(player_id)
            if p.on_possession_gained is not None:
                p.on_possession_gained(p)
            if self.match_logger is not None:
                self.match_logger.notify_possession_change(
                    self.time_s, self.ball.position, p, old
                )

    # -------------------------------------------------------------------------

    def _update_state_timers(self, dt: float) -> None:
        for player in self.players:
            if player.state == PlayerState.ACTIVE:
                player.stamina = regen_stamina(
                    self.movement_params, player.stamina, player.attributes.stamina, dt * 0.3
                )
                continue
            player.state_timer_s -= dt
            if player.state_timer_s <= 0.0:
                if player.state == PlayerState.CONTROLLING_BALL:
                    self._complete_control(player)
                player.state = PlayerState.ACTIVE
                player.state_timer_s = 0.0

    def _process_orders(self, dt: float) -> None:
        for player in self.players:
            # Reset before this tick's AI/order execution so kick_direct()
            # (called from ANY order type, or directly by the neural net) can
            # set it fresh — see Player.kicked_this_tick docstring.
            player.kicked_this_tick = False
            player.tackle_armed = False
            player.kick_armed = False
            player.kick_armed_direction = None
            player.last_kick_direction = None
            player.last_kick_power_fraction = None
            player.last_kick_spin = None
            if player.ai is not None:
                player.ai.act(player, self, 0)
            order = player.current_order
            if order is None:
                # Neural players kick directly (no Order), so log here before skipping.
                if player.kicked_this_tick:
                    self._log_info(f"{player.player_id} kicked the ball at {self.ball.velocity.length():.1f} m/s")
                if self.match_logger is not None and player.kicked_this_tick:
                    self.match_logger.notify_kick(self.time_s, self.ball.position, player)
                continue
            completed = order.execute(player, self, dt)
            if player.desired_speed_mode is None:
                # Every order must explicitly decide movement intent (even
                # STANDSTILL) on every tick it executes -- leaving it unset
                # is read by _apply_movement as inertial "coasting", a state
                # the neural decode path can never produce, which silently
                # diverges live play from the NN's action-replay/BC-label
                # view of the same tick. Fail loudly instead of coasting.
                raise RuntimeError(
                    f"{type(order).__name__}.execute() for player {player.player_id!r} "
                    f"did not set desired_direction/desired_speed_mode this tick"
                )
            if completed:
                self._complete_order(order)
                player.current_order = None
            if player.kicked_this_tick:
                self._log_info(f"{player.player_id} kicked the ball at {self.ball.velocity.length():.1f} m/s")
            if self.match_logger is not None and player.kicked_this_tick:
                self.match_logger.notify_kick(self.time_s, self.ball.position, player)

    def _run_get_possession_behaviour(self, player: Player, dt: float) -> bool:
        """Runs one tick of 'acquire the ball' behaviour, shared by
        ``GetPossessionOrder`` and ``MarkOrder``'s intercept/tackle fallback.

        - Returns ``True`` once possession is actually confirmed
          (``self.ball.possessed_by == player.player_id``).
        - If another player has the ball: chase them and, on contact, attempt
          a tackle.
        - If the ball is loose: sprint to intercept, continuing to close the
          distance and set movement intent every tick — including while
          already within pickup radius — until possession is confirmed
          above. Previously this branch returned ``True`` purely from
          geometric proximity ("at the ball — pickup will complete next
          tick"), assuming pickup always succeeds once close enough. That's
          not guaranteed: `possession.can_pick_up_ball`'s closing-velocity
          release-grace check can legitimately block pickup on that tick
          (e.g. a player who just push-kicked the ball away, now standing
          right next to a ball still moving away from them). When that
          happened, the order completed and cleared itself without actually
          gaining possession and without ever setting movement intent that
          tick — `act()` would then immediately re-issue a fresh
          ``GetPossessionOrder`` next tick, which repeated the same hollow
          "complete" with no motion, in a loop, until the ball drifted far
          enough away for the proximity check to stop firing (often 1-2
          ticks of the player doing nothing productive after every
          push-kick). Gating completion on the ACTUAL confirmed possession
          (`has_ball`, computed below but previously never used for this)
          fixes this at the source — no test-side workaround needed.
        """
        order = player.current_order
        sprint = getattr(order, 'sprint', True) if order is not None else True

        has_ball = self.ball.possessed_by == player.player_id
        if has_ball:
            # Already have the ball -- carry momentum forward rather than
            # freezing (e.g. MarkOrder delegating in here while its marker
            # already holds the ball).
            from footballcoach.orders import _continue_current_motion
            adj_dir, sm = _continue_current_motion(player, self, sprint=sprint)
            player.desired_direction = adj_dir
            player.desired_speed_mode = sm
            return True
        carrier = self.ball_carrier()

        if carrier is not None and carrier.player_id != player.player_id:
            # Arm the tackle; _check_armed_tackles resolves it when in contact range.
            player.tackle_armed = True
            intercept = self._intercept_target(player, carrier.position, carrier.velocity)
            tap = self.tackle_approach_params
            if tap.intercept_ahead_s > 0.0:
                # Match._intercept_target solves for the EARLIEST meeting
                # point -- when catching up from behind, that sits right on
                # the carrier's heels, leaving repulsion below no room to
                # curve the approach before contact. This pushes the aim
                # point further along the carrier's CURRENT velocity beyond
                # that earliest point (extra lead distance, not a different/
                # later solve), aiming at open ground ahead of the carrier
                # instead of their exact backside.
                intercept = intercept + carrier.velocity * tap.intercept_ahead_s
            direction = intercept - player.position

            # Tackle-angle-aware approach: same geometric convention as
            # engine/tackling.py's tackle_angle_modifier() -- angle between
            # the CARRIER's facing direction and the vector from the carrier
            # to US. cos ~ +1 means we're positioned where they're facing
            # (frontal, good angle); cos ~ -1 means directly behind them
            # (worst tackle_angle_modifier outcome). While this angle is
            # still bad, blend in repulsion steering -- WITHOUT excluding
            # the carrier (ball_carrier_id=None below), unlike every other
            # repulsion call site -- so the chase actually curves onto a
            # better angle instead of running straight up the carrier's
            # back. Once the angle is good enough, repulsion switches off
            # and we close in on the same intercept point directly.
            dribbler_dir = carrier.velocity.xy()
            if dribbler_dir.length() < 1e-9:
                dribbler_dir = Vector3.from_angle_xy(carrier.heading_rad, 1.0).xy()
            d_to_t = (player.position - carrier.position).xy()
            d_to_t_len = d_to_t.length()
            cos_angle = dribbler_dir.normalized().dot(d_to_t.normalized()) if d_to_t_len > 1e-9 else 1.0

            if cos_angle < tap.good_angle_cos_threshold:
                adj_dir, speed_mult = compute_repulsion(
                    player, direction, self.players, None, self.repulsion_params, self.pitch,
                )
                sm = SpeedMode.SPRINT if sprint else SpeedMode.JOG
                # Same "repulsion strong enough -> don't sprint through it"
                # rule _compute_movement_intent applies, for consistency
                # with how every other order treats a heavy push.
                if sm is SpeedMode.SPRINT and speed_mult < 0.75:
                    sm = SpeedMode.JOG
                adj_dir = adj_dir.xy().normalized() if adj_dir.length_xy() > 1e-9 else Vector3.zero()
                # Brake-to-turn: _compute_movement_intent applies this for
                # every other movement path (see its own use_brake_to_turn
                # docstring); this branch bypasses that helper entirely
                # (needed for the carrier-NOT-excluded repulsion above,
                # which _compute_movement_intent's built-in repulsion can't
                # do), so it must replicate the check itself. Without it, a
                # sharp heading change straight into this branch (e.g. right
                # after jockeying in a very different direction) never
                # brakes to reorient -- turn rate is capped by lateral_
                # accel/speed (engine/movement.py's max_turn_rate_rad_s), so
                # at full speed it just carves a slow, wide arc instead.
                op = self.order_params
                if (sm is not SpeedMode.STANDSTILL and adj_dir.length() > 1e-9
                        and player.speed_mps > op.brake_min_speed_mps):
                    heading_error = abs(angle_diff(player.heading_rad, adj_dir.angle_xy()))
                    if heading_error > op.brake_turn_angle_rad:
                        sm = SpeedMode.STANDSTILL
            else:
                from footballcoach.orders import _compute_movement_intent
                adj_dir, sm = _compute_movement_intent(
                    player, direction, self,
                    sprint=sprint, arrival_dist=None,
                    use_repulsion=False, use_brake_to_turn=True,
                )
            player.desired_direction = adj_dir
            player.desired_speed_mode = sm
            if are_touching(player, carrier):
                # Contact will be resolved this tick by _check_armed_tackles;
                # keep tracking the carrier's motion rather than freezing
                # mid-duel.
                return True
            return False
        else:
            # Ball is loose — run to intercept; pickup via _update_loose_ball_pickup.
            # Deceleration-aware: see engine.interception.intercept_target's
            # own target_decel_mps2 docstring -- without this, the solve
            # assumes the ball holds its CURRENT speed forever, which has a
            # real, confirmed failure mode (not just a directional bias): a
            # numerical singularity whenever the ball's speed crosses the
            # chaser's own speed, producing a predicted intercept point
            # hundreds of metres off the pitch for a real multi-tick window
            # (confirmed on a real traced "invalid" episode -- the chaser
            # aimed at that nonsense point instead of anywhere near the
            # actual ball, and missed by 1.89m at the line as a direct
            # result). Confirmed fix via the same traced episode: with this
            # enabled, the chaser gets genuine possession instead.
            intercept = self._intercept_target(
                player, self.ball.position, self.ball.velocity,
                self.interception_params.assumed_ball_decel_mps2,
            )
            # Overshoot margin: aim past the intercept in the direction of ball travel so
            # the player arrives slightly before the ball and has time to set their feet.
            # Proportional to the player's current distance to intercept so the offset
            # shrinks naturally as they close in (no overshoot on the final approach).
            overshoot_frac = self.order_params.intercept_overshoot_frac
            if overshoot_frac > 0.0:
                ball_vel_xy = self.ball.velocity.xy()
                ball_speed_xy = ball_vel_xy.length()
                if ball_speed_xy > 0.1:
                    dist_to_intercept = (intercept - player.position).length_xy()
                    intercept = intercept + ball_vel_xy * (dist_to_intercept * overshoot_frac / ball_speed_xy)
            direction = intercept - player.position

            # Degenerate-intercept fallback: intercept_target() solves
            # assuming the ball holds its CURRENT velocity for the whole
            # chase, so whenever the ball's recorded position already
            # coincides with the player's own (confirmed real case: the
            # tick right after THIS player kicks the ball, its velocity
            # jumps immediately but its position only integrates forward on
            # the NEXT physics tick -- so for that one tick the ball is
            # still recorded exactly where the player is standing, despite
            # already moving at real speed), the solve's "time to meet" is
            # trivially 0 and it hands back the player's OWN position as
            # the "intercept point" -- direction collapses to the zero
            # vector. _compute_movement_intent/step_player_towards treats a
            # zero direction as "no target, decelerate to a stop" (a
            # perfectly reasonable behaviour when there's genuinely nowhere
            # to go), which here actively BRAKES the player every single
            # tick this happens, although the ball is plainly still moving
            # away at real speed the whole time. Confirmed via direct
            # instrumentation (traced push-kick repetition: this fired on
            # literally every other tick, capping speed around 5.3 m/s
            # against a 7+ m/s ceiling, for 87 kicks straight). Falling back
            # to the ball's OWN direction of travel here -- not standing
            # still -- is the correct read of "the target is right where I
            # am AND already moving": follow it.
            if direction.length_xy() < 1e-6:
                ball_vel_xy = self.ball.velocity.xy()
                if ball_vel_xy.length() > 1e-6:
                    direction = ball_vel_xy

            # Boundary-aware braking: don't sprint blindly onto a ball
            # sitting near the touchline/goal line -- a player who arrives
            # too fast to decelerate before the boundary just carries a
            # newly-possessed ball out with them the instant they pick it
            # up (confirmed real bug -- see test_ball_out_overrun.py).
            #
            # Compute the fastest speed the player could be moving at the
            # intercept point and still be able to brake to a stop before
            # crossing the NEAREST boundary (same v^2=2*a*d kinematics
            # already used by braking_speed_mode for a normal arrival
            # stop), then feed it in as arrival_speed via the same
            # arrival-distance/braking machinery every other order uses.
            # When the ball is nowhere near a boundary this naturally
            # computes a safe speed at or above sprint speed, so
            # braking_speed_mode's own "no braking needed" branch leaves
            # ordinary mid-pitch chases at full sprint, unchanged.
            #
            # This doesn't guarantee a stop exactly at the line (discrete
            # 30Hz ticks + the tiny pickup radius mean some residual
            # overrun is still possible) -- it's a real reduction, not a
            # hard guarantee, per the tradeoff of reusing the existing
            # braking curve rather than a bespoke boundary solver. When
            # there's essentially NO room left to decelerate into at all
            # (raw_margin <= boundary_braking_params.abandon_margin_m,
            # below), the chase is abandoned -- braking to a stop if that
            # can actually clear the pickup radius in time, or steering
            # away from the ball instead if it can't -- rather than
            # crawling in at whatever near-zero speed the formula computes
            # -- see that branch's own comment.
            #
            # Both this buffer and the abandon threshold are configurable
            # via orders.json["boundary_braking"] (self.boundary_braking_
            # params) rather than hardcoded, so they can be tuned without a
            # code change.
            #
            # Skipped entirely when the player's OWN TEAM touched this ball
            # last (Ball.last_touched_by_player_id): the reward model only
            # scores a clean "ball_out" penalty once SOMEONE has touched the
            # ball (an untouched exit is "invalid", 0 reward, not -4 -- see
            # ai/env/reward.py's ball_went_out_after_touch / ai/env/
            # outcome.py's "miss" vs "invalid" split). Once our own team has
            # already touched it, that untouched-exit floor is gone for us
            # either way, so cautious braking can only ever cost a shot at
            # recovering it (stop-and-redirect) -- there's no longer a
            # worse outcome it's protecting against. Braking still applies
            # normally on the FIRST chase of a possession sequence (nobody
            # has touched it yet) and whenever the OPPONENT touched it last
            # (their team's fault, not ours, if it rolls out) -- both cases
            # where a careless carry-out would make things worse than a
            # clean miss.
            own_team_touched_last = False
            _last_toucher_id = self.ball.last_touched_by_player_id
            if _last_toucher_id is not None:
                own_team_touched_last = self.player_by_id(_last_toucher_id).team == player.team

            if own_team_touched_last:
                # No braking at all -- arrival_dist=None is what disables
                # the braking curve entirely in _compute_movement_intent
                # (arrival_speed=None alone would NOT do this -- it only
                # resolves the arrival TARGET to jog speed, still braking
                # to get there; see that function's own docstring). This is
                # the exact pre-boundary-fix behaviour, deliberately
                # reinstated for this specific case.
                arrival_dist_for_call = None
                arrival_speed_for_call = None
            else:
                from footballcoach.engine.movement import effective_acceleration
                bp = self.boundary_braking_params
                # Unclamped: how much room is left to decelerate into,
                # which can go negative (there's NO room at all, not just
                # "zero room") -- used below to decide whether to abandon
                # the chase entirely, not just cap its speed. The clamped
                # (>= 0) version is what actually feeds the kinematic
                # v^2=2*a*d safe-speed formula, matching braking_speed_
                # mode's own arrival_speed convention (never negative).
                #
                # Deliberately measured against the ball's CURRENT position,
                # not `intercept` (the predicted future meeting point). This
                # was a real, confirmed bug: `intercept` extrapolates the
                # ball forward by however long it'd take THIS player to
                # close the gap, so a player far from a ball that's merely
                # heading toward a boundary got an already-near-boundary
                # predicted margin from tick one -- abandoning (see below)
                # immediately and permanently, long before it was a live
                # decision, even from the opposite side of the pitch.
                # Confirmed via two real traced episodes: player frozen at
                # velocity zero for the entire episode while a loose ball
                # 15-25m away drifted untouched out of bounds. Using the
                # ball's actual position instead means this only goes
                # sharply negative once the ball is genuinely near the
                # line, and it re-evaluates fresh every tick as both the
                # player and the ball actually move.
                raw_margin = min(
                    self.pitch.half_length - abs(self.ball.position.x),
                    self.pitch.half_width - abs(self.ball.position.y),
                ) - self.pickup_radius_m - bp.brake_buffer_m

                if raw_margin <= bp.abandon_margin_m:
                    # No meaningful room to decelerate into, and it isn't
                    # our team's ball to lose -- crawling toward it at the
                    # near-zero safe speed the formula below would compute
                    # still reliably ends up carrying it out anyway
                    # (confirmed: a real traced episode overran by 0.54m
                    # despite max_safe_speed already computing to 0 there).
                    #
                    # Braking to a dead stop (SpeedMode.STANDSTILL, which
                    # already applies a boosted decel -- movement_params.
                    # standstill_decel_multiplier, 1.5x by default) sounds
                    # like it should avoid touching the ball, but by the
                    # time raw_margin crosses this threshold the player is
                    # typically already CLOSER to the ball than that boosted
                    # brake's own stopping distance needs: confirmed on a
                    # real traced episode (v0=5.98 m/s, boosted-brake
                    # stopping distance 2.42m, actual gap to the ball only
                    # 1.54m) where full braking still let momentum carry the
                    # player into the ball's pickup radius. That's a real
                    # bug on its own (possession.can_pick_up_ball has no
                    # notion of ball-in-bounds, only proximity, so this
                    # incidental touch flips a free "invalid" (0 reward)
                    # into a penalised "ball_out", -4) -- but even setting
                    # that aside, braking alone can't reliably prevent the
                    # touch in the first place here, so predict whether it
                    # actually can before committing to it.
                    #
                    # If braking (that same boosted decel) IS predicted to
                    # stop the player short of the pickup radius, do that --
                    # unchanged from before. If not, steer directly AWAY
                    # from the ball's current position instead, at full
                    # sprint with no arrival braking: this changes the
                    # velocity VECTOR rather than trying to kill speed the
                    # player physically can't shed in time, so distance-to-
                    # ball stops closing without needing to stop moving
                    # first. Confirmed on the same traced episode: closest
                    # approach becomes 0.78m (pickup radius is 0.55m)
                    # instead of an incidental touch, flipping that
                    # episode's outcome from ball_out (-4) to invalid (0)
                    # with zero effect on any other episode -- verified via
                    # a 500-episode outcome_baseline.py run before and after
                    # this change (identical counts for every OTHER outcome
                    # bucket). Tackle_armed was already set above (still
                    # relevant if this is actually a carrier chase reusing
                    # this branch via a race-condition edge case); nothing
                    # else needs undoing.
                    dist_to_ball = player.position.xy().distance_to(self.ball.position.xy())
                    current_speed = player.velocity.length_xy()
                    a_max = effective_acceleration(
                        self.movement_params, player.attributes.acceleration,
                        player.stamina, player.is_goalkeeper,
                    )
                    a_eff_brake = a_max * self.movement_params.standstill_decel_multiplier
                    stop_dist = (
                        (current_speed ** 2) / (2.0 * a_eff_brake) if a_eff_brake > 0 else float("inf")
                    )
                    can_brake_in_time = stop_dist <= max(0.0, dist_to_ball - self.pickup_radius_m)

                    if can_brake_in_time:
                        player.desired_direction = Vector3.zero()
                        player.desired_speed_mode = SpeedMode.STANDSTILL
                        return False

                    away_xy = player.position.xy() - self.ball.position.xy()
                    if away_xy.length() < 1e-6:
                        # Degenerate: standing exactly on the ball's (x,y) --
                        # fall back to the reverse of current heading rather
                        # than a zero direction (which would just re-trigger
                        # the "no target: decelerate" path in
                        # step_player_towards).
                        away_xy = player.velocity.xy() * -1.0
                    away_dir = Vector3(away_xy.x, away_xy.y, 0.0)
                    from footballcoach.orders import _compute_movement_intent
                    adj_dir, sm = _compute_movement_intent(
                        player, away_dir, self, sprint=True, arrival_dist=None, arrival_speed=None,
                        use_repulsion=False, use_brake_to_turn=True,
                    )
                    player.desired_direction = adj_dir
                    player.desired_speed_mode = sm
                    return False

                boundary_margin = max(0.0, raw_margin)
                a_max = effective_acceleration(
                    self.movement_params, player.attributes.acceleration,
                    player.stamina, player.is_goalkeeper,
                )
                a_eff = a_max * self.movement_params.standstill_decel_multiplier
                max_safe_speed = (2.0 * a_eff * boundary_margin) ** 0.5
                arrival_dist_for_call = direction.length_xy()
                arrival_speed_for_call = max_safe_speed

            from footballcoach.orders import _compute_movement_intent
            adj_dir, sm = _compute_movement_intent(
                player, direction, self,
                sprint=sprint, arrival_dist=arrival_dist_for_call, arrival_speed=arrival_speed_for_call,
                use_repulsion=False, use_brake_to_turn=True,
            )
            player.desired_direction = adj_dir
            player.desired_speed_mode = sm
            return False

    def _sync_possessed_ball(self) -> None:
        carrier = self.ball_carrier()
        if carrier is None:
            return
        offset = Vector3.from_angle_xy(carrier.heading_rad, carrier.radius_m + self.ball.radius_m)
        self.ball.position = carrier.position + offset
        self.ball.position = self.ball.position.with_z(self.ball.radius_m)
        self.ball.velocity = carrier.velocity

    def _update_loose_ball_pickup(
        self, dt: float, pre_flight_position: Vector3, pre_tick_player_positions: dict[str, Vector3],
    ) -> None:
        if self.ball.possessed_by is not None:
            return

        # Collect every ACTIVE player eligible to pick up the ball this tick
        # (including via the swept-tunneling exception -- see
        # possession.can_pick_up_ball), then resolve contention by picking
        # whoever is CURRENTLY closest to the ball, rather than the first
        # eligible player in self.players' (essentially arbitrary, team-
        # assignment-order) iteration order. This also naturally prefers a
        # player who is genuinely standing on the ball right now over one
        # who is only eligible because they swept past it earlier this tick
        # and has since moved further away -- a swept-only candidate is by
        # definition currently outside pickup_radius_m, so a within-radius
        # candidate (if any) always wins the distance comparison.
        candidates: list[Player] = []
        for player in self.players:
            if player.state != PlayerState.ACTIVE:
                continue
            player_pre_tick_position = pre_tick_player_positions.get(player.player_id)
            if not can_pick_up_ball(
                player, self.ball, self.ball_pickup_params, pre_flight_position, player_pre_tick_position,
            ):
                continue
            candidates.append(player)

        if not candidates:
            return

        player = min(
            candidates,
            key=lambda p: (p.position.xy().distance_to(self.ball.position.xy()), p.player_id),
        )

        # Armed kick (see orders.py's _try_push_kick / apply_nn_action.py /
        # Player.kick_armed): the player already committed to redirecting
        # this ball the instant it's reachable, so skip CONTROLLING_BALL
        # possession entirely -- no control-time delay, no
        # control_speed_multiplier slowdown. This is the actual one-touch
        # path; without it, "arming" a kick had no effect on physics at all
        # (every pickup went through the normal control-then-kick sequence
        # below regardless of kick_armed). Flat, direction-only kick (no
        # ballistic solve) -- see Player.kick_armed_direction's docstring.
        #
        # ball_settled (real bug fix, confirmed via direct instrumentation):
        # an armed redirect must NOT fire while the ball still has real
        # vertical velocity (mid-bounce, e.g. from this SAME player's own
        # prior kick landing with a touch of angle noise -- kicks are
        # nominally flat but still draw the same pitch noise every other
        # kick does, occasionally crossing the ground-contact engine's own
        # real-bounce threshold). Confirmed by direct trace: a redirect
        # consumed the instant the ball re-entered pickup range regardless
        # of vz, so a bounce that brought the ball's speed down to near-
        # match the chaser's own sprint speed got treated as pickable
        # immediately (can_pick_up_ball's own closing-speed deadzone,
        # unrelated to this fix and untouched), re-firing another armed
        # kick before the player ever settled it -- a self-sustaining loop
        # confirmed on one real seed to repeat 90 times in a row. Falling
        # through to the NORMAL control-time grant below when unsettled
        # fixes this at the actual point of failure: _sync_possessed_ball
        # zeroes the ball's vz every tick it's genuinely held, so a forced
        # real possession here reliably breaks the loop, whereas the armed
        # path never touches vz at all. Confirmed via the same seed: real
        # kicks dropped from 91 to 28 with this change alone (arming itself,
        # and every other mechanic here, is unchanged).
        #
        # Threshold is config-backed (ball_pickup_params.armed_redirect_
        # settle_vz_mps, physics.json), not "exactly zero" -- an initial
        # 1e-6 cutoff turned out to be a second, connected bug of its own:
        # a push-kick's own kick-angle noise routinely leaves it with some
        # small vz (~0.5-0.6 m/s is typical), which almost never decays to
        # 1e-6 before a fast-sprinting player recatches it -- so EVERY touch
        # was falling through to the slow control-time grant regardless,
        # defeating the entire point of arming (confirmed on a real traced
        # episode: 40 short, speed-losing hops instead of ~7 long full-speed
        # ones). Retuned empirically across 500 real episodes -- see
        # physics.json's own _comment_armed_redirect_settle_vz for the full
        # sweep (0.01/0.5/0.7/1.0/1.5) and why 1.0 was chosen.
        ball_settled = abs(self.ball.velocity.z) < self.ball_pickup_params.armed_redirect_settle_vz_mps
        if player.kick_armed and player.kick_armed_direction is not None and ball_settled:
            self._set_possession(player.player_id)
            # kick_armed_power_fraction is already the final value (run-
            # compensated at arm time for push-kicks -- see its docstring)
            # -- compensate_for_run defaults False here, i.e. used as-is.
            player.kick_with_direction(
                self, player.kick_armed_direction, player.kick_armed_power_fraction,
                player.kick_armed_spin or Vector3.zero(),
            )
            return

        relative_speed = (self.ball.velocity - player.velocity).length()
        player.firsttime_difficulty = compute_difficulty(
            self.control_time_params, self.ball.height_m, relative_speed, player.speed_mps
        )
        t_control = control_time_s(
            self.control_time_params,
            self.ball.height_m,
            relative_speed,
            player.speed_mps,
            player.attributes.ball_control,
            is_goalkeeper_in_box=player.is_goalkeeper and self.pitch.is_in_either_box(player.position),
        )
        noise_sigma = t_control * self.control_time_params.noise_sigma_fraction * (1.0 - self.rng_reduction)
        t_control = max(0.01, t_control + self.rng.gauss(0.0, noise_sigma))

        player.state = PlayerState.CONTROLLING_BALL
        player.state_timer_s = t_control
        # Grant possession immediately so the ball is glued to the player
        # via _sync_possessed_ball each tick (rather than frozen in space).
        # Speed is snapped down by control_speed_multiplier on contact;
        # the player then coasts at that reduced speed until control ends.
        self._set_possession(player.player_id)
        player.velocity = player.velocity * self.movement_params.control_speed_multiplier
        # Display hint for the UI action-icon system (consumed by renderer, not engine logic).
        player.action_icon = "🧤" if player.is_goalkeeper else "✋"


    def _leading_pass_target(self, passer: Player, order: "PassOrder") -> Vector3:
        """If the PassOrder has a ``target_player_id``, leads the pass to
        where that player will be when the ball arrives.

        The estimate: t_arrive ≈ distance / ball_speed (from auto-pace model),
        capped at 2 s. If the target is standing still (speed < 0.3 m/s) or
        doesn't exist, falls back to ``order.target_position``.
        """
        from footballcoach.orders import PassOrder  # local import to avoid circular
        if order.target_player_id is None:
            return order.target_position
        try:
            target = self.player_by_id(order.target_player_id)
        except KeyError:
            return order.target_position

        if target.speed_mps < 0.3:
            return order.target_position  # stationary - no lead needed

        dist = passer.position.xy().distance_to(order.target_position.xy())
        ball_speed = pass_speed_mps(
            self.passing_params, dist,
            self.ball_physics_params.gravity_mps2,
            self.ball_physics_params.rolling_friction_coefficient,
        ) * getattr(order, "power_multiplier", 1.0)
        t_arrive = min(dist / max(ball_speed, 1.0), 2.0)
        predicted = order.target_position + target.velocity.xy() * t_arrive
        return predicted.with_z(0.0)

    def _intercept_target(
        self, player: Player, target_pos: Vector3, target_vel: Vector3,
        target_decel_mps2: float = 0.0,
    ) -> Vector3:
        """Returns the world position this player should sprint toward in order
        to intercept the target (ball or carrier) in the shortest possible time.

        See `engine.interception.intercept_target` for the underlying math,
        including `target_decel_mps2` -- callers chasing the loose BALL
        should pass `self.interception_params.assumed_ball_decel_mps2` (see
        `_run_get_possession_behaviour`'s own call); a chased carrier isn't
        decelerating the same way, so the carrier-chase call site leaves
        this at its 0.0 default.
        """
        v_p = effective_top_speed(
            self.movement_params,
            player.attributes.top_speed,
            player.stamina,
            has_ball=False,
            is_goalkeeper=player.is_goalkeeper,
        )
        return intercept_target(player.position, v_p, target_pos, target_vel, target_decel_mps2)

    def _check_armed_tackles(self) -> None:
        """Resolve tackles for any player who armed tackle_armed this tick
        and is now in contact range.  Fires on_tackle so BC records the event.
        Runs after movement so players that sprinted into range are caught here
        before _check_head_on_tackles (the collision-only fallback) can fire."""
        carrier = self.ball_carrier()
        if carrier is None or carrier.is_inactive:
            return
        for player in self.players:
            if not player.tackle_armed:
                continue
            if player.player_id == carrier.player_id:
                continue
            if player.team == carrier.team:
                continue
            # Use the same overlap radius as auto-tackle so tunnelled approaches
            # are caught here; skip can_tackle's are_touching to avoid re-checking
            # the tighter 0.65m threshold that was already passed during movement.
            dist = player.position.xy().distance_to(carrier.position.xy())
            overlap_threshold = self.tackling_params.auto_tackle_overlap_factor * (player.radius_m + carrier.radius_m)
            if dist < overlap_threshold and player.is_available_to_tackle() and carrier.is_available_to_tackle():
                self._attempt_tackle_contact(player, carrier)
                return  # one tackle per tick

    def _check_head_on_tackles(self) -> None:
        """Automatically triggers a tackle when two players from opposite teams
        are in serious overlap (distance < auto_tackle_overlap_factor * combined_radius)
        and have a net closing velocity above auto_tackle_min_closing_mps.

        This replaces the old requirement that BOTH players charge head-on at
        >= 1 m/s, which was too restrictive and never fired when one player was
        slower or approaching from an angle.  The new condition fires whenever
        the players are deeply overlapping and moving toward each other at all.

        Only one such tackle is resolved per tick (the first qualifying pair).
        The resulting tackle uses the same skill-check, inactivity penalties,
        and dribble-speed-penalty model as regular tackles.
        """
        carrier = self.ball_carrier()
        if carrier is None or carrier.is_inactive:
            return

        overlap_factor = self.tackling_params.auto_tackle_overlap_factor
        min_closing = self.tackling_params.auto_tackle_min_closing_mps

        for other in self.players:
            if other.player_id == carrier.player_id:
                continue
            if other.is_inactive or other.team == carrier.team:
                continue

            # Direction from carrier to other player.
            delta = other.position.xy() - carrier.position.xy()
            dist = delta.length()
            if dist < 1e-9:
                continue

            # Trigger only on serious overlap (beyond mere touching).
            min_distance = carrier.radius_m + other.radius_m
            if dist >= overlap_factor * min_distance:
                continue

            direction = delta / dist  # unit vector carrier -> other

            # Net closing speed: sum of each player's velocity component toward
            # the other.  Positive = getting closer together.
            carrier_speed_towards = carrier.velocity.xy().dot(direction)
            other_speed_towards = -(other.velocity.xy().dot(direction))
            closing_speed = carrier_speed_towards + other_speed_towards

            if closing_speed < min_closing:
                continue

            # Head-on: resolve as a tackle (other player is the tackler).
            # Aerial control: if the carrier is mid first-touch on a high ball,
            # the tackle is blocked (nothing to poke away at foot level yet).
            if (
                carrier.state == PlayerState.CONTROLLING_BALL
                and self.ball.position.z > self.control_time_params.control_tackle_immune_height_m
            ):
                self._log_info(
                    f"{other.player_id} head-on tackle on {carrier.player_id} blocked [aerial control]"
                )
                return
            if self._gk_immune_from_tackle(carrier):
                # Phase B: GK in own box with ball is untackleable —
                # the onrushing player is penalised; carrier is untouched.
                other.velocity = other.velocity * self.tackling_params.tackle_attempt_tackler_speed_mult
                other.state = PlayerState.INACTIVE_TACKLED
                other.state_timer_s = self.tackling_params.tackle_cooldown_s
                self._log_info(f"{other.player_id} head-on tackle on {carrier.player_id} auto-failed [GK in own box]")
                if self.match_logger is not None:
                    self.match_logger.notify_tackle_attempt(self.time_s, self.ball.position, other, carrier)
                return
            result = attempt_tackle(
                other.attributes.tackling,
                carrier.attributes.dribbling,
                self.rng_reduction,
                self.rng,
                self.tackling_params,
                is_goalkeeper_tackle=other.is_goalkeeper,
                angle_modifier=tackle_angle_modifier(
                    carrier.heading_rad, carrier.position, other.position, self.tackling_params
                ),
                gk_outside_box=self._gk_outside_own_box(other),  # Phase B: GK penalty
            )
            self._log_tackle_result(other.player_id, carrier.player_id, result, "head-on")
            if result.tackler_won:
                self._set_possession(other.player_id)
            apply_tackle_result(result, other, carrier, self.tackling_params)
            # NOTE: on_tackle/on_tackle_result deliberately do NOT fire here,
            # mirroring on_tackle's existing scope — this is the collision-
            # based auto-tackle fallback, not a recordable AI-intended attempt
            # (see test_tackle.py's auto-tackle-vs-armed-path tests). Firing
            # on_tackle_result here with no matching on_tackle "attempt" would
            # corrupt success/fail stats derived from on_tackle counts.
            # on_auto_tackle_result is the separate, dedicated callback for
            # callers that DO want to see/count these events.
            if other.on_auto_tackle_result is not None:
                other.on_auto_tackle_result(other, result.tackler_won, True)
            if carrier.on_auto_tackle_result is not None:
                carrier.on_auto_tackle_result(carrier, result.tackler_won, False)
            return  # only one head-on tackle per tick

    def _complete_control(self, player: Player) -> None:
        # Possession was already granted at the moment of contact; just log completion.
        self._log_info(f"{player.player_id} controlled the ball")

    def _log_tackle_result(
        self,
        tackler_id: str,
        dribbler_id: str,
        result: "TackleResult",  # type: ignore[name-defined]  # noqa: F821
        modifier_notes: str = "",
    ) -> None:
        """Emit tackle outcome to log_callback (if set)."""
        if self.log_callback is None:
            return
        outcome = "tackled" if result.tackler_won else "failed tackle on"
        self._log_info(f"{tackler_id} {outcome} {dribbler_id}")
        notes = f"  tackler_roll={result.tackler_roll:.3f}  dribbler_roll={result.dribbler_roll:.3f}"
        if modifier_notes:
            notes += f"  [{modifier_notes}]"
        self._log_debug(notes)

    # ── Phase B helpers ────────────────────────────────────────────────────

    def _defends_left(self, player: Player) -> bool:
        """True if this player's team defends the left goal (Team.LEFT)."""
        return player.team == Team.LEFT

    def _gk_immune_from_tackle(self, gk_candidate: Player) -> bool:
        """True when a goalkeeper is in possession inside their own penalty
        box — in that case all tackle attempts auto-fail (GK is legally
        handling the ball in a protected area)."""
        return (
            gk_candidate.is_goalkeeper
            and self.ball.possessed_by == gk_candidate.player_id
            and self.pitch.is_in_box(
                gk_candidate.position, left=self._defends_left(gk_candidate)
            )
        )

    def _gk_outside_own_box(self, tackler: Player) -> bool:
        """True when the tackler is a goalkeeper making a tackle from outside
        their own penalty box (triggers the -40% tackle boost penalty)."""
        return (
            tackler.is_goalkeeper
            and not self.pitch.is_in_box(
                tackler.position, left=self._defends_left(tackler)
            )
        )

    def _effective_dribbling(self, target: Player) -> float:
        """Returns the target's dribbling attribute, with up to 25% penalty
        if they are mid first-touch control (CONTROLLING_BALL state).

        penalty_frac = min(1, state_timer_s / control_time_penalty_reference_s)
        dribbling_eff = dribbling_attr * (1 - 0.25 * penalty_frac)

        A player who just picked the ball up (near max timer) has the full
        25% penalty; one about to finish control (near 0 s remaining) has
        essentially no penalty.
        """
        if target.state != PlayerState.CONTROLLING_BALL:
            return target.attributes.dribbling
        ref_s = self.tackling_params.control_time_penalty_reference_s
        penalty_frac = min(1.0, target.state_timer_s / ref_s)
        return target.attributes.dribbling * (1.0 - 0.25 * penalty_frac)

    def _target_has_or_controls_ball(self, target: Player) -> bool:
        """True if the target has possession OR is mid first-touch control
        (ball frozen at their feet). Used to decide ball transfer on a
        tackle win against a CONTROLLING_BALL player."""
        return (
            self.ball.possessed_by == target.player_id
            or target.state == PlayerState.CONTROLLING_BALL
        )

    def _attempt_tackle_contact(self, player: Player, target: Player) -> None:
        """Resolve one tackle attempt between *player* (tackler) and *target*
        (ball carrier).  Caller must have already confirmed the two players are
        touching and that ``target.is_available_to_tackle()`` is True.

        Fires ``player.on_tackle`` before physics resolution so BC demonstration
        recording captures the event regardless of which order type triggered
        the tackle (``ChaseTackleOrder`` or ``GetPossessionOrder``).
        """
        if player.on_tackle is not None:
            player.on_tackle(player)
        if self.match_logger is not None:
            self.match_logger.notify_tackle_attempt(self.time_s, self.ball.position, player, target)
        # A player controlling an aerial ball (above waist height) cannot be tackled —
        # the ball isn't on the ground yet so there's nothing to poke away.
        if (
            target.state == PlayerState.CONTROLLING_BALL
            and self.ball.position.z > self.control_time_params.control_tackle_immune_height_m
        ):
            self._log_info(
                f"{player.player_id} tackle on {target.player_id} blocked [aerial control]"
            )
            return
        if self._gk_immune_from_tackle(target):
            self._apply_gk_immune_penalty(player)
            self._log_info(
                f"{player.player_id} tackle on {target.player_id} auto-failed [GK in own box]"
            )
        else:
            result = attempt_tackle(
                player.attributes.tackling,
                self._effective_dribbling(target),
                self.rng_reduction,
                self.rng,
                self.tackling_params,
                is_goalkeeper_tackle=player.is_goalkeeper,
                angle_modifier=tackle_angle_modifier(
                    target.heading_rad, target.position, player.position, self.tackling_params
                ),
                gk_outside_box=self._gk_outside_own_box(player),
            )
            self._log_tackle_result(player.player_id, target.player_id, result)
            if result.tackler_won and self._target_has_or_controls_ball(target):
                self._set_possession(player.player_id)
            apply_tackle_result(result, player, target, self.tackling_params)
            if player.on_tackle_result is not None:
                player.on_tackle_result(player, result.tackler_won, True)
            if target.on_tackle_result is not None:
                target.on_tackle_result(target, result.tackler_won, False)

    def _check_goal(self) -> None:
        side = check_goal(self.ball, self.pitch)
        if side is not None:
            self.scoreboard.score_for(side)
            self._log_info(f"GOAL for {'LEFT' if side == 'left' else 'RIGHT'} — score {self.scoreboard.left_goals}:{self.scoreboard.right_goals}")
            if self.match_logger is not None:
                self.match_logger.notify_goal(self.time_s, self.ball.position, side)
            if self.goal_linger_s > 0.0:
                self._goal_linger_remaining_s = self.goal_linger_s
            else:
                self._reset_after_goal()

    def _reset_after_goal(self) -> None:
        self.ball.position = Vector3.zero()
        self.ball.velocity = Vector3.zero()
        self.ball.spin = Vector3.zero()
        self._set_possession(None)

    def notify_ball_out(self) -> None:
        """Called by the scenario loop when the ball goes out of play.
        Forwards to the match logger if one is attached."""
        if self.match_logger is not None:
            self.match_logger.notify_ball_out(self.time_s, self.ball.position)


def _drain_if_sprinting(params: MovementParams, player: Player, sprinting: bool, dt: float) -> float:
    from footballcoach.engine.movement import drain_stamina

    if not sprinting:
        return player.stamina
    return drain_stamina(params, player.stamina, player.attributes.stamina, 1.0, dt)
