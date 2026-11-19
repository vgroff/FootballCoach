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

from footballcoach.config import load_orders_config, require_section
from footballcoach.engine.ball_physics import BallPhysicsParams, step_ball, resolve_goal_boundary
from footballcoach.engine.collision import CollisionParams, resolve_all_overlaps, resolve_ball_block_by_inactive_players
from footballcoach.engine.goalkeeping import GoalkeepingParams
from footballcoach.engine.kicking import KickingParams, PassingParams
from footballcoach.engine.movement import MovementParams, SpeedMode, angle_diff, regen_stamina, step_player_towards
from footballcoach.engine.possession import BallPickupParams, ControlTimeParams, can_pick_up_ball, control_time_s, compute_difficulty
from footballcoach.engine.scoring import Scoreboard, check_goal
from footballcoach.engine.tackling import TacklingParams
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import JockeyParams, OrderLayerParams, OrderStatus, TackleApproachParams
from footballcoach.steering import RepulsionParams

import logging
log = logging.getLogger("footballcoach.match")

from footballcoach.engine.match_logger import MatchLogger
from footballcoach.engine.match_tackles import MatchTacklesMixin
from footballcoach.engine.match_get_possession import MatchGetPossessionMixin

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
class Match(MatchTacklesMixin, MatchGetPossessionMixin):
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
    # Number of completed physics ticks (increments at the end of step()). The clock
    # ActionOpportunity events are stamped with (entities/action_opportunity.py).
    tick_index: int = field(default=0, init=False, repr=False)

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

    # Optional callback invoked with (level, message, detail) for UI game-log
    # display -- detail is an optional "\n"-separated hover breakdown (see
    # ui/gamelog.py's LogEntry.detail), None for the common no-detail case.
    # None by default so tests / headless use incur zero cost.
    log_callback: Callable[["LogLevel", str, str | None], None] | None = field(default=None, repr=False)

    # Optional match event logger for post-hoc inspection.  None by default
    # so training/tests incur zero cost when logging is not needed.
    match_logger: MatchLogger | None = field(default=None, repr=False)

    _players_by_id: dict[str, Player] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # reversed(): with a duplicate id the first player in the list wins.
        self._players_by_id = {p.player_id: p for p in reversed(self.players)}
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
        player = self._players_by_id.get(player_id)
        if player is None:
            raise KeyError(f"no player with id {player_id}")
        return player

    def player_by_id_or_none(self, player_id: str) -> Player | None:
        return self._players_by_id.get(player_id)

    def ball_carrier(self) -> Player | None:
        if self.ball.possessed_by is None:
            return None
        return self.player_by_id(self.ball.possessed_by)

    def advance_state_timers(self) -> None:
        """Public entry point for a caller that needs to run THIS tick's
        state-timer update (stamina drain/regen, inactive-tackled/
        controlling-ball countdowns, etc.) before ``step()`` itself runs --
        see ``step(skip_state_timers=...)``'s own docstring for why this
        exists: an external batching caller (``ai/ppo/batched_rollout_worker.py``'s
        ``BatchedEnvGroup``) needs decisions to see the SAME up-to-date
        timer state a normal ``step()`` call would give them, even though it
        must encode the observation (hence make the decision) BEFORE calling
        ``step()`` for this tick."""
        self._update_state_timers(self.dt_s)

    def step(self, *, skip_state_timers: bool = False) -> None:
        """Advance the match by one physics tick.

        ``skip_state_timers``: pass ``True`` only when the caller already
        called ``advance_state_timers()`` for this exact tick before calling
        ``step()`` -- see that method's docstring. Needed because
        ``BatchedEnvGroup`` (ai/ppo/batched_rollout_worker.py) encodes a
        batched decision's observation BEFORE calling ``env.step()`` for
        that round, i.e. before this method's own state-timer update would
        normally run (below, ahead of ``_process_orders``) -- without this,
        every batched decision would see one tick's worth of STALE timer
        state (e.g. stamina) relative to what the identical unbatched
        ``env.step()`` path sees, since there a decision is only ever made
        from INSIDE this method, after timers have already advanced for
        that tick. Confirmed as the root cause of a real (if tiny -- one
        tick's stamina delta, ~1e-4 scale) discrepancy between
        BatchedEnvGroup's K=1 case and plain unbatched stepping. Default
        ``False`` (unchanged behavior) for every other caller.
        """
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

        if not skip_state_timers:
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
        self.tick_index += 1

    # -- logging helpers (zero cost when log_callback is None) ---------------

    def _log_info(self, msg: str, detail: str | None = None) -> None:
        if self.log_callback is not None:
            from footballcoach.ui.gamelog import LogLevel  # local import: only paid when UI is live
            self.log_callback(LogLevel.INFO, msg, detail)

    def _log_debug(self, msg: str, detail: str | None = None) -> None:
        if self.log_callback is not None:
            from footballcoach.ui.gamelog import LogLevel
            self.log_callback(LogLevel.DEBUG, msg, detail)

    def _complete_order(self, order) -> None:
        """Transition *order* to COMPLETE and fire its on_complete callback if set."""
        order.status = OrderStatus.COMPLETE
        if order.on_complete is not None:
            order.on_complete()


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
            gk_boost = None
            if player.is_goalkeeper and self.movement_params.goalkeeper_dive_gating:
                speed_mode, gk_boost = self._gate_goalkeeper_dive(player, speed_mode, dt)
            step_player_towards(player, player.desired_direction, speed_mode, dt, self.movement_params, has_ball,
                                goalkeeper_boost=gk_boost)
            player.stamina = _drain_if_sprinting(self.movement_params, player, speed_mode is SpeedMode.SPRINT, dt)
            player.last_desired_speed_mode = speed_mode  # NOT cleared -- see Player.last_desired_speed_mode
            player.desired_speed_mode = None  # consumed; reset for next tick

    def _gate_goalkeeper_dive(self, player: Player, speed_mode: SpeedMode, dt: float) -> tuple[SpeedMode, bool]:
        """Gated keeper dive (physics.json movement._comment_goalkeeper_dive):
        a keeper's sprint is its dive. Returns (the speed mode to apply this
        tick, whether the keeper speed/accel boost applies), updating
        player.gk_dive_remaining_s / gk_dive_recovery_s.

        - Diving and still asking to sprint: keep diving until the window runs
          out; the tick that exhausts it is the last boosted one.
        - Diving and not asking to sprint: the dive ends now, recovery starts.
        - Not diving, asking to sprint, no recovery pending: a new dive starts.
        - Recovering: a sprint request is forced to a jog; the recovery
          countdown keeps running either way.
        """
        mp = self.movement_params
        wants_sprint = speed_mode is SpeedMode.SPRINT
        if player.gk_dive_remaining_s > 0.0:
            if wants_sprint:
                player.gk_dive_remaining_s = max(0.0, player.gk_dive_remaining_s - dt)
                if player.gk_dive_remaining_s == 0.0:
                    player.gk_dive_recovery_s = mp.goalkeeper_dive_recovery_s
                return speed_mode, True
            player.gk_dive_remaining_s = 0.0
            player.gk_dive_recovery_s = mp.goalkeeper_dive_recovery_s
            return speed_mode, False
        if player.gk_dive_recovery_s > 0.0:
            player.gk_dive_recovery_s = max(0.0, player.gk_dive_recovery_s - dt)
            return (SpeedMode.JOG if wants_sprint else speed_mode), False
        if wants_sprint:
            player.gk_dive_remaining_s = max(0.0, mp.goalkeeper_dive_duration_s - dt)
            if player.gk_dive_remaining_s == 0.0:
                player.gk_dive_recovery_s = mp.goalkeeper_dive_recovery_s
            return speed_mode, True
        return speed_mode, False

    def _pickup_params_for(self, player: Player):
        """Ball-pickup params for ``player``: a goalkeeper inside his own box
        uses goalkeeper_pickup_radius_m (hands) when it is set."""
        bp = self.ball_pickup_params
        r = bp.goalkeeper_pickup_radius_m
        if r is None or not player.is_goalkeeper:
            return bp
        if not self.pitch.is_in_box(player.position, left=self._defends_left(player)):
            return bp
        import dataclasses
        return dataclasses.replace(bp, pickup_radius_m=r)

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
        # Pass 1 (plan): every player decides using ONLY tick-start state.
        # See PlayerAI.plan()'s own docstring (entities/player.py) for the
        # full rationale -- in short, an AI whose act()/decide() also
        # mutates shared/other-player-visible state immediately (e.g.
        # NeuralPlayerAI's apply(), which can kick the ball right now if
        # this player already possesses it) must defer that mutation to
        # commit() below, or a LATER-PROCESSED player's plan() this SAME
        # tick would see the earlier player's already-resolved action
        # (confirmed a real bug this way: two players' observations of
        # each other differed on the same tick when they should not have).
        # Flag resets happen here, before this tick's own plan() call,
        # exactly as before this split.
        for player in self.players:
            # Reset before this tick's AI/order resolution so kick_direct()
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
                player.ai.plan(player, self, 0)

        # Pass 2 (commit/resolve): every player's decision takes effect, now
        # that everyone has planned -- no player's plan() above could have
        # observed another's resolution, since none had happened yet this
        # tick. order.execute() is the resolution step for any Order-driven
        # (rules-based) AI; commit() is the equivalent hook for an AI that
        # applies its own effects directly (see PlayerAI.commit()'s
        # docstring).
        for player in self.players:
            if player.ai is not None:
                player.ai.commit(player, self, 0)
            order = player.current_order
            if order is None:
                # Neural players kick directly (no Order), so log here before skipping.
                if player.kicked_this_tick:
                    self._log_info(f"{player.player_id} kicked the ball at {self.ball.velocity.length():.1f} m/s")
                if self.match_logger is not None and player.kicked_this_tick:
                    self.match_logger.notify_kick(self.time_s, self.ball.position, player)
                continue
            completed = order.execute(player, self, dt)
            if player.pending_kick is not None:
                # The order's kick, carried out here, at the same point in the
                # tick it used to happen inside execute() (engine/kick_intent.py).
                from footballcoach.engine.kick_intent import apply_kick_intent
                _intent, player.pending_kick = player.pending_kick, None
                apply_kick_intent(player, self, _intent)
            if player.pending_tackle is not None:
                # The order's tackle, resolved here at the same point in the
                # tick it used to be resolved inside execute().
                _target_id, player.pending_tackle = player.pending_tackle, None
                # Counted like an armed tackle's (_check_armed_tackles), so every
                # tackle a player makes is in tackle_fire_count.
                player.tackle_fire_count += 1
                self._attempt_tackle_contact(player, self.player_by_id(_target_id))
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
                player, self.ball, self._pickup_params_for(player), pre_flight_position, player_pre_tick_position,
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
        # Loose-ball pickup = a kick opportunity for this player this tick, identical whether or
        # not it armed a kick (the armed and the normal path below both grant possession here).
        player.opportunity.note_kick_opp(self.tick_index)
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







    def _complete_control(self, player: Player) -> None:
        # Possession was already granted at the moment of contact; just log completion.
        self._log_info(f"{player.player_id} controlled the ball")








    def _check_goal(self) -> None:
        conceding_team = check_goal(self.ball, self.pitch)
        if conceding_team is not None:
            self.scoreboard.record_goal_against(conceding_team)
            scoring_team = conceding_team.opponent
            self._log_info(f"GOAL for {scoring_team.name} — score {self.scoreboard.left_goals}:{self.scoreboard.right_goals}")
            if self.match_logger is not None:
                self.match_logger.notify_goal(self.time_s, self.ball.position, scoring_team.name.lower())
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
