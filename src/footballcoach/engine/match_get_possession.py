"""Match's get-possession behaviour (chasing a loose ball or a carrier) and the intercept / lead-pass target math."""
from __future__ import annotations

from footballcoach.engine.collision import are_touching
from footballcoach.engine.interception import intercept_target
from footballcoach.engine.kicking import pass_speed_mps
from footballcoach.engine.movement import SpeedMode, angle_diff, effective_acceleration, effective_top_speed
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3
from footballcoach.steering import compute_repulsion
from footballcoach.mathutils.tolerances import GEOM_ZERO_EPS



class MatchGetPossessionMixin:
    """Get-possession and interception methods of Match (see engine/match.py for the tick order that calls them)."""

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
            if dribbler_dir.length() < GEOM_ZERO_EPS:
                dribbler_dir = Vector3.from_angle_xy(carrier.heading_rad, 1.0).xy()
            d_to_t = (player.position - carrier.position).xy()
            d_to_t_len = d_to_t.length()
            cos_angle = dribbler_dir.normalized().dot(d_to_t.normalized()) if d_to_t_len > GEOM_ZERO_EPS else 1.0

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
                adj_dir = adj_dir.xy().normalized() if adj_dir.length_xy() > GEOM_ZERO_EPS else Vector3.zero()
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
                if (sm is not SpeedMode.STANDSTILL and adj_dir.length() > GEOM_ZERO_EPS
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
