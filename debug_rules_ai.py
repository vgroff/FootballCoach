"""Experimental rules-AI tackle-approach-angle A/B test.

Tests whether steering the chaser AROUND the ball carrier (instead of
straight-line intercepting them) produces more frontal/side tackles instead
of tackles from behind -- see engine/tackling.py's tackle_angle_modifier()
for why this matters (+0.10 frontal, -0.05 side, -0.65 behind: roughly a
2x+ swing in effective tackling attribute from angle alone).

EVERYTHING experimental lives in THIS file. Nothing in src/footballcoach/ is
touched -- TackleAngleChaseOrder and TackleAngleAwareRulesAI are throwaway
classes for this test only, not wired into rules_ai.py/orders.py.

Why a straight-line intercept can't just be re-aimed at a "better" point:
Phase1RulesAI's existing chase logic (Match._run_get_possession_behaviour)
already uses a real pursuit solver (engine/interception.py's
intercept_target(), given the carrier's actual velocity) -- it's not naive.
But it walks a STRAIGHT LINE to that one computed point, every tick, with
repulsion disabled (use_repulsion=False at every call site in
_run_get_possession_behaviour). Simply biasing the target point doesn't
produce a curved approach -- the straight line from the chaser's current
position to any point near the carrier converges with the carrier's own
path from whichever side is geometrically shortest, which is very often
behind/the side when catching up to a moving carrier. There's no
"go around" behaviour without an actual lateral force.

Why plain repulsion (steering.py's compute_repulsion) can't be flipped on
as-is: it deliberately EXCLUDES the ball carrier as a repulsion source
("Do NOT repel from the ball carrier -- per the plan's explicit scoping",
steering.py). So merely passing use_repulsion=True on a chase order would
have ZERO effect on how the chaser approaches the actual tackle target --
it would only account for OTHER (non-carrier) players in the way.

This experiment's fix: call compute_repulsion() with ball_carrier_id=None
instead of the real carrier's id -- this is the ONE thing that disables the
carrier-exclusion (see steering.py: `if other.player_id == ball_carrier_id:
continue`), so the carrier becomes a completely normal repulsion-emitting
neighbour for this purpose, with no other side effects (compute_repulsion's
OWN extra ball-carrier speed-penalty logic keys off whether the CALLING
player has the ball, not off this id, so it's unaffected). Blended with the
existing intercept_target() lead point, this should curve the approach
around the carrier instead of running straight into their back -- switched
OFF again (not delegated to a different code path -- the SAME order keeps
driving movement throughout) once the chaser reaches a safe tackle angle
(frontal/side, not behind -- same cos(angle) convention tackle_angle_
modifier() itself uses). Actual tackle contact/resolution is untouched
production code either way (Match._check_armed_tackles, via
player.tackle_armed) -- only the approach path is experimental.

Usage:
    uv run python debug_rules_ai.py --n-episodes 300
    uv run python debug_rules_ai.py --n-episodes 300 --angle-threshold 0.3
    uv run python debug_rules_ai.py --n-episodes 300 --n-workers 6
"""
from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from dataclasses import replace as _dc_replace

from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder, OrderStatus
from footballcoach.rules_ai import Phase1RulesAI, _nearest_box_point, _should_sprint_to_ball
from footballcoach.steering import RepulsionParams

log = logging.getLogger("footballcoach.debug_rules_ai")


# Trigger-rate instrumentation for _should_sprint_to_ball_min_dt -- module-
# level so it accumulates across a whole eval run; reset/report around each
# run in main(). "calls" = decision points where an opponent was present to
# check at all (has_opponents True); stock/margin = how many of those calls
# had that condition true for at least one opponent (or the OOB check);
# margin_only = margin true but stock false -- the ONLY calls where this
# experimental change actually flips the sprint decision vs stock alone.
_trigger_stats = {"calls": 0, "stock": 0, "margin": 0, "margin_only": 0}


def _reset_trigger_stats() -> None:
    _trigger_stats.update(calls=0, stock=0, margin=0, margin_only=0)


def _should_sprint_to_ball_min_dt(player, match, min_dt_s: float) -> bool:
    """EXPERIMENTAL variant of rules_ai.py's _should_sprint_to_ball(): sprint
    if EITHER the stock relative-margin condition triggers OR an absolute-
    time-buffer condition does -- an extra reason to sprint, on top of stock,
    never a replacement that could sprint LESS than stock would. (An earlier
    version of this function fully replaced the stock trigger instead of
    OR-ing it in -- with sprint_to_ball_clearance_margin=0.4 (very permissive:
    `opp_eta < eta_self_jog * 2.5`, stock sprints almost unconditionally at
    ordinary distances), a straight replacement made the AI sprint LESS than
    stock at nearly every realistic distance -- exactly backwards. Confirmed
    empirically: that version alone cost ~17pp win rate.)

    Stock trigger: `opp_eta * sprint_to_ball_clearance_margin < eta_self_jog`.
    Extra trigger here: `opp_eta < eta_self_jog + min_dt_s` -- sprint if the
    opponent could arrive within min_dt_s of (or before) us, i.e. there's a
    real CHANCE they beat us, with a safety margin -- not only once they're
    already certain to. The two conditions cross over at roughly
    eta_self_jog = min_dt_s / (1/margin - 1) (~0.53s at min_dt_s=0.8,
    margin=0.4): below that, the extra trigger is the wider net (exactly the
    close-race edge case this was meant to patch); above it, stock's own
    margin already covers more ground and this term rarely changes anything.
    Same OR pattern applied to the "ball heading out of bounds" check too,
    for consistency (both checks share the same margin pattern in stock).
    """
    from footballcoach.engine.movement import effective_acceleration, effective_top_speed, sprint_eta
    from footballcoach.entities.player import PlayerState

    ball_pos = match.ball.position
    dist_self = (ball_pos - player.position).length()
    if dist_self < 0.1:
        return False  # already at the ball

    jog_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina, has_ball=False,
    ) * 0.5
    jog_speed = max(jog_speed, 0.1)
    self_v0 = min(player.speed_mps, jog_speed)
    eta_self_jog = dist_self / jog_speed if self_v0 >= jog_speed else sprint_eta(
        dist_self, self_v0, jog_speed,
        effective_acceleration(match.movement_params, player.attributes.acceleration, player.stamina),
    )

    has_opponents = False
    any_stock = False
    any_margin = False
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
            effective_top_speed(match.movement_params, opp.attributes.top_speed, opp.stamina, has_ball=False),
            0.1,
        )
        opp_accel = effective_acceleration(match.movement_params, opp.attributes.acceleration, opp.stamina)
        opp_eta = sprint_eta(dist_opp, opp.speed_mps, opp_v_top, opp_accel)
        any_stock = any_stock or (opp_eta * match.order_params.sprint_to_ball_clearance_margin < eta_self_jog)
        any_margin = any_margin or (opp_eta < eta_self_jog + min_dt_s)

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
        any_stock = any_stock or (t_out * match.order_params.sprint_to_ball_clearance_margin < eta_self_jog)
        any_margin = any_margin or (t_out < eta_self_jog + min_dt_s)

    if has_opponents:
        _trigger_stats["calls"] += 1
        if any_stock:
            _trigger_stats["stock"] += 1
        if any_margin:
            _trigger_stats["margin"] += 1
        if any_margin and not any_stock:
            _trigger_stats["margin_only"] += 1

    if any_stock or any_margin:
        return True  # opponent/ball might beat us (stock margin), or within min_dt_s of us
    return has_opponents is False


# ---------------------------------------------------------------------------
# Experimental order + AI (throwaway -- see module docstring)
# ---------------------------------------------------------------------------

@dataclass
class TackleAngleChaseOrder:
    """Chase the ball. If an opposing player currently carries it, always
    compute the same pursuit lead point production uses (Match.
    _intercept_target), but blend it with repulsion for as long as the
    tackle angle is still bad -- and, unlike every production call site,
    this repulsion NEVER excludes the ball carrier (see module docstring)
    -- so it actually steers around them instead of being a no-op. Once the
    angle is good enough, repulsion is simply switched off (not delegated
    to a different code path) and the same order keeps closing in on the
    same intercept point directly. Real tackle resolution (contact/roll) is
    untouched production code either way (Match._check_armed_tackles, via
    player.tackle_armed set below) -- only the APPROACH is experimental.

    Re-checks the angle every physics tick (not just every decision tick),
    since _process_orders calls execute() unconditionally every tick
    regardless of AI decision cadence -- this keeps it correctly reactive
    to the carrier's own movement without needing extra state.
    """
    sprint: bool = True
    good_angle_cos_threshold: float = 0.0  # cos(angle); >=0 = frontal-to-side, <0 = behind
    intercept_ahead_s: float = 0.0
    repulsion_params: Optional[object] = None  # RepulsionParams override; None = match.repulsion_params (orders.json)
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Optional[Callable[[], None]] = field(default=None, repr=False, compare=False)

    def execute(self, player, match, dt: float) -> bool:
        from footballcoach.engine.movement import SpeedMode, angle_diff
        from footballcoach.steering import compute_repulsion

        carrier = match.ball_carrier()
        if carrier is None or carrier.player_id == player.player_id or carrier.team == player.team:
            # No opposing carrier at all (loose ball, we have it, or a
            # teammate has it) -- nothing to steer around, stock chase/pickup.
            return match._run_get_possession_behaviour(player, dt)

        player.tackle_armed = True
        # Match._intercept_target solves for the EARLIEST meeting point --
        # exactly where production's own GetPossessionOrder aims. When
        # catching up to a carrier from behind, that earliest point sits
        # right on their heels, leaving repulsion no room to curve the
        # approach before contact. intercept_ahead_s pushes the aim point
        # further along the carrier's CURRENT velocity beyond that earliest
        # point -- not a different (later) solve, just extra lead distance
        # tacked on to the same point -- so the chaser is steering toward
        # open ground ahead of the carrier instead of their exact backside,
        # giving repulsion (below) an actual lateral gap to act on.
        intercept = match._intercept_target(player, carrier.position, carrier.velocity)
        if self.intercept_ahead_s > 0.0:
            intercept = intercept + carrier.velocity * self.intercept_ahead_s
        desired = intercept - player.position

        # Same geometric convention as engine/tackling.py's
        # tackle_angle_modifier(): angle between the CARRIER's facing
        # direction and the vector from the carrier to US. cos ~ +1 means
        # we're positioned where they're facing (frontal); cos ~ -1 means
        # we're directly behind them. Repulsion is APPLIED (not delegated
        # away to a different code path) for as long as this angle is still
        # bad -- once it's good enough, repulsion is simply switched off and
        # we close in on the same intercept point directly, still via this
        # same order/movement path throughout.
        dribbler_dir = Vector3.from_angle_xy(carrier.heading_rad, 1.0).xy()
        d_to_t = (player.position - carrier.position).xy()
        d_to_t_len = d_to_t.length()
        cos_angle = dribbler_dir.dot(d_to_t.normalized()) if d_to_t_len > 1e-9 else 1.0
        still_need_repulsion = cos_angle < self.good_angle_cos_threshold

        if still_need_repulsion:
            # ball_carrier_id=None is what disables compute_repulsion's
            # built-in "never repel from the carrier" exclusion (steering.py)
            # -- the carrier is NEVER excluded here, full stop, unlike every
            # production call site.
            adj_dir, speed_mult = compute_repulsion(
                player, desired, match.players, None,
                self.repulsion_params or match.repulsion_params, match.pitch,
            )
        else:
            adj_dir, speed_mult = desired, 1.0

        speed_mode = SpeedMode.SPRINT if self.sprint else SpeedMode.JOG
        # Same "repulsion strong enough -> don't sprint through it" rule
        # _compute_movement_intent() itself applies (orders.py), for
        # consistency with how every other order treats a heavy push.
        if speed_mode is SpeedMode.SPRINT and speed_mult < 0.75:
            speed_mode = SpeedMode.JOG

        adj_dir_xy = adj_dir.xy().normalized() if adj_dir.length_xy() > 1e-9 else Vector3.zero()

        # Brake-to-turn: the same heuristic every other chase/intercept
        # order gets for free via _compute_movement_intent (orders.py) --
        # turn rate is capped by lateral_accel/speed (engine/movement.py's
        # max_turn_rate_rad_s), so a large heading change at full speed just
        # carves a slow, wide arc instead of actually reorienting unless
        # speed is shed first. This order hand-rolls its own movement intent
        # (needed for the carrier-NOT-excluded repulsion above, which
        # _compute_movement_intent can't do) instead of calling that shared
        # helper, so it has to replicate this check itself -- without it, a
        # fresh TackleAngleChaseOrder issued right after the player was
        # heading a very different way (e.g. coming out of jockeying) never
        # brakes to reorient, confirmed via direct UI observation.
        op = match.order_params
        if (speed_mode is not SpeedMode.STANDSTILL and adj_dir_xy.length() > 1e-9
                and player.speed_mps > op.brake_min_speed_mps):
            heading_error = abs(angle_diff(player.heading_rad, adj_dir_xy.angle_xy()))
            if heading_error > op.brake_turn_angle_rad:
                speed_mode = SpeedMode.STANDSTILL

        player.desired_direction = adj_dir_xy
        player.desired_speed_mode = speed_mode
        return False


def _opponent_clearly_wins_loose_ball_race(player, match, give_up_margin: float):
    """True (and the winning opponent) iff a real (non-immobile) opponent's
    sprint-ETA to the loose ball is COMFORTABLY better than our own -- i.e.
    self_eta > opp_eta * give_up_margin. give_up_margin > 1.0 means the
    opponent must be at least that much faster than merely tying us before
    we treat the race as lost.

    Deliberately a STRICTER bar than _should_sprint_to_ball's own relative
    margin (0.4 in the live config -- confirmed this session to make stock's
    sprint trigger fire almost unconditionally, i.e. it's built to say
    "sprint" liberally, not "give up" liberally). Jockeying away from a
    genuinely contestable ball instead of contesting it would be a real net
    loss, not a gain, so this must only fire when the race is clearly lost,
    not merely close -- see the design discussion this was built from.

    Immobile opponents (opp.ai is None) are skipped entirely -- they can
    never truly "win" a race in the sense that matters here (they'd just
    coast to a stop near the ball, not actually control it purposefully),
    mirroring Phase1RulesAI's own opponent_is_immobile special-casing
    elsewhere.

    Returns (won, winning_opponent, winning_opponent_eta_to_ball) --
    (False, None, None) if no opponent clearly wins. The ETA is returned
    (not just recomputed by the caller) so _dynamic_jockey_target can reuse
    the exact same "opponent's ETA to the loose ball" figure as its
    reachability bar, rather than a second, potentially-diverging solve.
    """
    from footballcoach.engine.movement import effective_acceleration, effective_top_speed, sprint_eta

    ball_pos = match.ball.position
    dist_self = (ball_pos - player.position).length()
    self_top = effective_top_speed(match.movement_params, player.attributes.top_speed, player.stamina, has_ball=False)
    self_accel = effective_acceleration(match.movement_params, player.attributes.acceleration, player.stamina)
    self_eta = sprint_eta(dist_self, player.speed_mps, self_top, self_accel)

    for opp in match.players:
        if opp.player_id == player.player_id or opp.team == player.team or opp.ai is None:
            continue
        dist_opp = (ball_pos - opp.position).length()
        opp_top = effective_top_speed(match.movement_params, opp.attributes.top_speed, opp.stamina, has_ball=False)
        opp_accel = effective_acceleration(match.movement_params, opp.attributes.acceleration, opp.stamina)
        opp_eta = sprint_eta(dist_opp, opp.speed_mps, opp_top, opp_accel)
        if self_eta > opp_eta * give_up_margin:
            return True, opp, opp_eta
    return False, None, None


def _jockey_target_position(player, match, blend_factor: float):
    """Loose ball, opponent clearly wins the race to it (see
    _opponent_clearly_wins_loose_ball_race): defensive position between the
    ball and where the opponent will run once they have it -- approximated
    as their own scoring-box target point (_nearest_box_point, the SAME
    target Phase1RulesAI itself aims for once IT has the ball), since the
    opponent in every matchup here is stock Phase1RulesAI and behaves
    deterministically on pickup (runs straight for its own box). This
    assumption is specific to testing against Phase1RulesAI -- would need
    rethinking against a neural or otherwise unpredictable opponent.

    blend_factor=0.5 ("halfway between the ball and where they're headed")
    is tunable -- 0.0 = stand on the ball, 1.0 = stand on the opponent's own
    box target. Returns None if no opponent found (shouldn't happen in 1v1).

    Called fresh from decide() every decision tick (NOT a custom per-physics-
    tick order recomputing this itself -- there used to be one here, but
    that just reinvents MoveOrder's own steering/arrival/braking for no
    reason: every other branch in this AI already re-issues a fresh order
    each decision tick, e.g. GetPossessionOrder/TackleAngleChaseOrder, so a
    moving target is already handled by that existing cadence -- see
    _default_decision_interval_ticks, ~4 ticks/0.24s, same reactivity every
    other order here gets).
    """
    opponent = next(
        (p for p in match.players if p.player_id != player.player_id and p.team != player.team), None,
    )
    if opponent is None:
        return None
    ball_pt = match.ball.position
    opp_target_pt = _nearest_box_point(opponent, match)
    return ball_pt + (opp_target_pt - ball_pt) * blend_factor


# Ascending (most attacking/closest-to-ball tried first) to match the live
# production default (orders.json["jockey"]["blend_candidates"]) -- see that
# config's own comment for why ascending was chosen over descending.
_JOCKEY_DYNAMIC_BLEND_CANDIDATES: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85)


def _dynamic_jockey_target(player, match, opp, opp_eta_to_ball: float, blend_candidates=_JOCKEY_DYNAMIC_BLEND_CANDIDATES):
    """Alternative to a fixed jockey_blend_factor: try each candidate blend
    (see _jockey_target_position -- 0.0 = stand on the ball, 1.0 = stand on
    the opponent's own box target), highest (most defensive) first, and pick
    the first one the player can reach in sprint-ETA <= opp_eta_to_ball --
    i.e. the most defensive position still reachable BEFORE the opponent
    actually gets the ball and starts running with it. Falls back to
    whichever candidate has the LOWEST sprint-ETA (i.e. is actually closest
    to the player right now) if none qualify -- NOT simply the smallest
    blend value: blend interpolates between the ball and the opponent's box
    target, so which candidate is geometrically nearest the player depends
    on where the player is currently standing relative to that line, not on
    blend order. A middle blend value can easily be closer than the
    smallest one.

    blend_candidates must be given highest-first. opp/opp_eta_to_ball are
    the SAME winning opponent and ETA _opponent_clearly_wins_loose_ball_race
    already computed for the trigger check -- reused here rather than
    resolved a second time, so this can't diverge from the decision that
    caused jockeying to fire in the first place.
    """
    from footballcoach.engine.movement import effective_acceleration, effective_top_speed, sprint_eta

    self_top = effective_top_speed(match.movement_params, player.attributes.top_speed, player.stamina, has_ball=False)
    self_accel = effective_acceleration(match.movement_params, player.attributes.acceleration, player.stamina)
    ball_pt = match.ball.position
    opp_target_pt = _nearest_box_point(opp, match)

    closest_target = ball_pt
    closest_eta = float("inf")
    for blend in blend_candidates:
        target = ball_pt + (opp_target_pt - ball_pt) * blend
        dist = (target - player.position).length()
        self_eta = sprint_eta(dist, player.speed_mps, self_top, self_accel)
        if self_eta <= opp_eta_to_ball:
            return target
        if self_eta < closest_eta:
            closest_eta = self_eta
            closest_target = target
    return closest_target


# Trigger-rate instrumentation for the jockey feature -- module-level so it
# accumulates across a whole eval run; reset/report around each run in the
# sweep below. "loose_ball_decisions" = every decide() call where the ball
# was loose (jockey COULD have fired); "jockey_triggered" = how many of
# those actually issued a jockey-target MoveOrder. Lets us tell "genuinely no
# effect" apart from "the trigger barely ever fires" -- exactly the
# distinction that mattered for the earlier sprint-timing experiment.
_jockey_trigger_stats = {"loose_ball_decisions": 0, "jockey_triggered": 0}


def _reset_jockey_trigger_stats() -> None:
    _jockey_trigger_stats.update(loose_ball_decisions=0, jockey_triggered=0)


class TackleAngleAwareRulesAI(Phase1RulesAI):
    """Same as Phase1RulesAI, except the "don't have the ball, opponent
    does" branch issues TackleAngleChaseOrder instead of GetPossessionOrder.
    Everything else (own-ball box run, loose-ball chase, sprint/jog
    decision) is identical to stock Phase1RulesAI -- see that class's own
    decide() this mirrors, and _should_sprint_to_ball's docstring for the
    sprint heuristic reused here unchanged, so the ONLY variable being
    tested is the tackle-approach angle, not a different sprint policy.
    """

    def __init__(
        self, good_angle_cos_threshold: float = 0.0, sprint_min_dt_s: float | None = None,
        intercept_ahead_s: float = 0.0, repulsion_params=None, decision_interval_ticks=None,
        jockey_give_up_margin: float | None = None, jockey_blend_factor: float = 0.5,
        jockey_dynamic_blend_candidates: "tuple[float, ...] | None" = None,
    ) -> None:
        super().__init__(decision_interval_ticks=decision_interval_ticks)
        self.good_angle_cos_threshold = good_angle_cos_threshold
        # None (default) = stock _should_sprint_to_ball (relative margin).
        # A float = _should_sprint_to_ball_min_dt (absolute time buffer)
        # instead -- see that function's own docstring.
        self.sprint_min_dt_s = sprint_min_dt_s
        self.intercept_ahead_s = intercept_ahead_s
        # RepulsionParams override (None = match.repulsion_params, i.e.
        # orders.json unmodified) -- lets a param sweep vary radius_m/
        # strength_base/etc. per run without ever touching that file.
        self.repulsion_params = repulsion_params
        # None (default) = never jockey, stock loose-ball chase always.
        # A float = give-up margin for _opponent_clearly_wins_loose_ball_race
        # -- see that function's own docstring for what the value means.
        self.jockey_give_up_margin = jockey_give_up_margin
        self.jockey_blend_factor = jockey_blend_factor
        # None (default) = always use the fixed jockey_blend_factor above. A
        # tuple (highest blend first) = use _dynamic_jockey_target instead,
        # picking the most defensive reachable-in-time blend each decision
        # tick -- see that function's own docstring.
        self.jockey_dynamic_blend_candidates = jockey_dynamic_blend_candidates
        self._jockeying_active = False  # log-once-per-entry state, see decide()

    def decide(self, player, match, trial_tick: int) -> None:
        was_jockeying = self._jockeying_active
        self._jockeying_active = False
        if match.ball.possessed_by == player.player_id:
            super().decide(player, match, trial_tick)
            return

        carrier = match.ball_carrier()
        if carrier is None:
            # Loose ball -- check whether the opponent clearly wins the race
            # before committing to a likely-hopeless chase. Restricted to
            # the OPENING scramble only (mirrors production's rules_ai.py):
            # once the ball has been touched by anyone this match
            # (Ball.last_touched_by_player_id != None, set forever after the
            # first touch), jockeying never fires again for the rest of the
            # match -- giving up on a live-in-play loose ball isn't the same
            # "clearly hopeless" situation the opening scramble is. NOTE:
            # every jockey sweep result recorded earlier this session
            # predates this restriction (jockeying could fire on ANY loose
            # ball, any time) -- those numbers reflect the more liberal
            # behaviour, not this one; re-sweep if this needs re-validating.
            if self.jockey_give_up_margin is not None and match.ball.last_touched_by_player_id is None:
                _jockey_trigger_stats["loose_ball_decisions"] += 1
                opponent_wins, _opp, _opp_eta = _opponent_clearly_wins_loose_ball_race(
                    player, match, self.jockey_give_up_margin,
                )
                if opponent_wins:
                    _jockey_trigger_stats["jockey_triggered"] += 1
                    if self.jockey_dynamic_blend_candidates is not None:
                        target = _dynamic_jockey_target(
                            player, match, _opp, _opp_eta, self.jockey_dynamic_blend_candidates,
                        )
                    else:
                        target = _jockey_target_position(player, match, self.jockey_blend_factor)
                    if target is not None:
                        if not was_jockeying:
                            match._log_info(f"[AI] {player.player_id}: jockeying (MoveOrder)")
                        self._jockeying_active = True
                        player.current_order = MoveOrder(
                            target_position=target, sprint=True, max_speed_on_arrival_mps=0.0,
                        )
                        return
            super().decide(player, match, trial_tick)
            return
        if carrier.team == player.team:
            # Teammate has it, N/A in 1v1 -- stock behaviour.
            super().decide(player, match, trial_tick)
            return

        # Opposing carrier -- same sprint heuristic Phase1RulesAI itself
        # uses (see its own decide() for the identical block/rationale),
        # except the ETA-vs-ball race check swaps to the min-dt variant
        # when sprint_min_dt_s is set.
        opponent_is_immobile = any(
            opp.player_id != player.player_id and opp.team != player.team and opp.ai is None
            for opp in match.players
        )
        if self.sprint_min_dt_s is not None:
            race_sprint = _should_sprint_to_ball_min_dt(player, match, self.sprint_min_dt_s)
        else:
            race_sprint = _should_sprint_to_ball(player, match)
        should_sprint = opponent_is_immobile or race_sprint
        if not isinstance(player.current_order, TackleAngleChaseOrder):
            match._log_info(f"[AI] {player.player_id}: TackleAngleChaseOrder")
        player.current_order = TackleAngleChaseOrder(
            sprint=should_sprint,
            good_angle_cos_threshold=self.good_angle_cos_threshold,
            intercept_ahead_s=self.intercept_ahead_s,
            repulsion_params=self.repulsion_params,
        )


def run_jockey_sweep(n_episodes: int, seed_base: int) -> None:
    """One-at-a-time sweep over jockey_give_up_margin and jockey_blend_factor,
    reporting win%/reward AND trigger rate (loose_ball_decisions vs
    jockey_triggered) for every point -- so a flat win-rate result can be
    told apart from "the trigger barely ever fires" (the same distinction
    that mattered for the earlier sprint-timing experiment)."""
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)

    def _run_jockey(label: str, margin: float, blend: float):
        _reset_jockey_trigger_stats()
        result = _run_eval(
            label,
            lambda: TackleAngleAwareRulesAI(jockey_give_up_margin=margin, jockey_blend_factor=blend),
            n_episodes, seed_base, 1,
        )
        ts = _jockey_trigger_stats
        trigger_rate = ts["jockey_triggered"] / ts["loose_ball_decisions"] if ts["loose_ball_decisions"] else 0.0
        log.info(
            f"{label:<45}win={result['win_rate_pct']:>5.1f}%  vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}  vs_base={result['mean_reward'] - baseline['mean_reward']:+.3f}  "
            f"triggered={ts['jockey_triggered']}/{ts['loose_ball_decisions']} ({trigger_rate:.1%})"
        )

    log.info("\n-- jockey_give_up_margin (blend_factor=0.5 fixed) --")
    for margin in (1.1, 1.2, 1.3, 1.5, 2.0):
        _run_jockey(f"margin={margin}", margin, 0.5)

    log.info("\n-- jockey_blend_factor (give_up_margin=1.3 fixed) --")
    for blend in (0.2, 0.35, 0.5, 0.65, 0.8):
        _run_jockey(f"blend={blend}", 1.3, blend)


def run_jockey_sweep2(n_episodes: int, seed_base: int) -> None:
    """Round 2, informed by round 1: both axes trended toward LESS harm as
    margin/blend increased (stricter trigger, position closer to the
    opponent's target than the ball), never clearly beating baseline. This
    pushes further in that same direction to see if it keeps improving or
    has already plateaued."""
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)

    def _run_jockey(label: str, margin: float, blend: float):
        _reset_jockey_trigger_stats()
        result = _run_eval(
            label,
            lambda: TackleAngleAwareRulesAI(jockey_give_up_margin=margin, jockey_blend_factor=blend),
            n_episodes, seed_base, 1,
        )
        ts = _jockey_trigger_stats
        trigger_rate = ts["jockey_triggered"] / ts["loose_ball_decisions"] if ts["loose_ball_decisions"] else 0.0
        log.info(
            f"{label:<45}win={result['win_rate_pct']:>5.1f}%  vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}  vs_base={result['mean_reward'] - baseline['mean_reward']:+.3f}  "
            f"triggered={ts['jockey_triggered']}/{ts['loose_ball_decisions']} ({trigger_rate:.1%})"
        )
        return result

    log.info("\n-- jockey_blend_factor (give_up_margin=1.5 fixed, round 1's best margin) --")
    for blend in (0.7, 0.8, 0.9, 0.95):
        _run_jockey(f"blend={blend}", 1.5, blend)

    log.info("\n-- jockey_give_up_margin (blend_factor=0.8 fixed, round 1's best blend) --")
    for margin in (1.4, 1.6, 1.8):
        _run_jockey(f"margin={margin}", margin, 0.8)


def run_jockey_sweep3(n_episodes: int, seed_base: int) -> None:
    """Round 3: a genuine 2D grid, unlike rounds 1-2 (both one-at-a-time,
    which can miss the true joint optimum). Rounds 1-2 found both axes
    plateauing around margin=1.4-1.5, blend=0.5-0.8, with every point in
    that region landing in a narrow ~0.7-1.0pp band -- this crosses the
    top candidates from both to see if a genuinely better JOINT combination
    exists there, or whether it's flat everywhere in the region. Logs the
    single best (margin, blend, win_rate_pct) combo found at the end."""
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)

    def _run_jockey(label: str, margin: float, blend: float):
        _reset_jockey_trigger_stats()
        result = _run_eval(
            label,
            lambda: TackleAngleAwareRulesAI(jockey_give_up_margin=margin, jockey_blend_factor=blend),
            n_episodes, seed_base, 1,
        )
        ts = _jockey_trigger_stats
        trigger_rate = ts["jockey_triggered"] / ts["loose_ball_decisions"] if ts["loose_ball_decisions"] else 0.0
        log.info(
            f"{label:<45}win={result['win_rate_pct']:>5.1f}%  vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}  vs_base={result['mean_reward'] - baseline['mean_reward']:+.3f}  "
            f"triggered={ts['jockey_triggered']}/{ts['loose_ball_decisions']} ({trigger_rate:.1%})"
        )
        return result

    log.info("\n-- 2D grid: margin x blend, N={} --".format(n_episodes))
    best = None
    for margin in (1.3, 1.4, 1.5):
        for blend in (0.6, 0.7, 0.8):
            result = _run_jockey(f"margin={margin} blend={blend}", margin, blend)
            if best is None or result["win_rate_pct"] > best[2]:
                best = (margin, blend, result["win_rate_pct"], result["mean_reward"])
    log.info(
        f"\nBest grid point: margin={best[0]}  blend={best[1]}  "
        f"win={best[2]:.1f}%  mean_rew={best[3]:.3f}  (vs baseline win={baseline['win_rate_pct']:.1f}%  "
        f"mean_rew={baseline['mean_reward']:.3f})"
    )


def run_jockey_tackle_stats_sweep(n_episodes: int, seed_base: int) -> None:
    """N=2000-scale sweep of jockey_give_up_margin, with front/back tackle
    breakdown per point (via run_tackle_angle_stats -- works with ANY
    trainee_ai_factory, jockey-enabled or not, since it just hooks tackle
    events regardless of which order led to them).

    The (margin=1.5, blend=0.8) config found "best" across three small-N
    sweeps (N=300-500) turned out to be a clear NET NEGATIVE at N=2000
    (-3.2pp) -- so this isn't re-testing values already disproven. It tests
    a genuinely different, still-open question: as the trigger is made
    progressively STRICTER (fires less and less often), does the effect
    converge toward baseline (0pp, i.e. "harmless once it barely fires"),
    or stay net-negative even at a low trigger rate (i.e. "harmful whenever
    it fires at all, regardless of how rarely")? blend_factor fixed at 0.8
    (the best-performing value found across every prior round)."""
    baseline = run_tackle_angle_stats(
        "baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base,
    )
    for margin in (1.5, 1.8, 2.0, 2.5, 3.0):
        _reset_jockey_trigger_stats()
        run_tackle_angle_stats(
            f"jockey margin={margin} blend=0.8",
            lambda margin=margin: TackleAngleAwareRulesAI(jockey_give_up_margin=margin, jockey_blend_factor=0.8),
            n_episodes, seed_base,
        )
        ts = _jockey_trigger_stats
        trigger_rate = ts["jockey_triggered"] / ts["loose_ball_decisions"] if ts["loose_ball_decisions"] else 0.0
        log.info(
            f"  [margin={margin}] jockey triggered {ts['jockey_triggered']}/{ts['loose_ball_decisions']} "
            f"({trigger_rate:.1%}) of loose-ball decisions"
        )


def run_jockey_grid_final(
    n_episodes: int, seed_base: int, tackling_params=None,
    margins=(3.0, 5.0), blends=(0.8, 0.95),
) -> None:
    """margin x blend grid (default (3.0,5.0) x (0.8,0.95)), N=2000-scale,
    same win/reward/outcomes + front/back tackle breakdown + trigger rate as
    run_jockey_tackle_stats_sweep. tackling_params: optional TacklingParams
    override (None = live physics.json), applied to BOTH baseline and every
    jockey variant equally -- see _rules_vs_rules_env_worker_factory."""
    run_tackle_angle_stats(
        "baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, tackling_params=tackling_params,
    )
    for margin in margins:
        for blend in blends:
            _reset_jockey_trigger_stats()
            run_tackle_angle_stats(
                f"jockey margin={margin} blend={blend}",
                lambda margin=margin, blend=blend: TackleAngleAwareRulesAI(
                    jockey_give_up_margin=margin, jockey_blend_factor=blend,
                ),
                n_episodes, seed_base, tackling_params=tackling_params,
            )
            ts = _jockey_trigger_stats
            trigger_rate = ts["jockey_triggered"] / ts["loose_ball_decisions"] if ts["loose_ball_decisions"] else 0.0
            log.info(
                f"  [margin={margin} blend={blend}] jockey triggered "
                f"{ts['jockey_triggered']}/{ts['loose_ball_decisions']} ({trigger_rate:.1%}) of loose-ball decisions"
            )


def run_combined_final(
    n_episodes: int, seed_base: int, tackling_params=None,
    good_angle_cos_threshold: float = -0.25, jockey_give_up_margin: float = 4.0, jockey_blend_factor: float = 0.8,
) -> None:
    """Both proven features together vs stock baseline: tackle-angle curving
    (good_angle_cos_threshold, best/most-robust value from the curving
    sweeps -- -0.25 and -0.05 were nearly tied under harsh rules) AND
    jockeying (jockey_give_up_margin/jockey_blend_factor, margin=4.0/
    blend=0.8 was the single best point in the most robust grid, and the
    whole margin 4-8 x blend 0.7-0.9 region was flat/robust around
    +1.1-1.7pp). Neither feature helps in isolation under LIVE tackling
    rules -- both only became real, replicating wins once paired with the
    harsh TacklingParams override, so tackling_params should normally be
    that override here too (caller decides; None = live physics.json)."""
    run_tackle_angle_stats(
        "baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, tackling_params=tackling_params,
    )
    _reset_jockey_trigger_stats()
    run_tackle_angle_stats(
        f"combined (angle={good_angle_cos_threshold} margin={jockey_give_up_margin} blend={jockey_blend_factor})",
        lambda: TackleAngleAwareRulesAI(
            good_angle_cos_threshold=good_angle_cos_threshold,
            sprint_min_dt_s=None,
            repulsion_params=None,
            jockey_give_up_margin=jockey_give_up_margin,
            jockey_blend_factor=jockey_blend_factor,
        ),
        n_episodes, seed_base, tackling_params=tackling_params,
    )
    ts = _jockey_trigger_stats
    trigger_rate = ts["jockey_triggered"] / ts["loose_ball_decisions"] if ts["loose_ball_decisions"] else 0.0
    log.info(
        f"  [combined] jockey triggered {ts['jockey_triggered']}/{ts['loose_ball_decisions']} "
        f"({trigger_rate:.1%}) of loose-ball decisions"
    )


# ---------------------------------------------------------------------------
# Eval plumbing (mirrors ai/scripts/evaluate.py's _baseline_env_worker_factory
# / _run_baseline_evaluation -- same shape, just able to swap in a different
# trainee AI class + kwargs instead of always using stock Phase1RulesAI)
# ---------------------------------------------------------------------------

def _rules_vs_rules_env_worker_factory(
    trainee_ai_factory: Callable[[], object],
    tackling_params=None,
) -> tuple:
    """Module-level-picklable-shaped factory (actually a closure -- fine for
    n_workers<=1 sequential use; see note in main() for why parallel isn't
    wired up here). Opponent is always stock Phase1RulesAI; trainee is
    whatever trainee_ai_factory() returns.

    tackling_params: optional TacklingParams override (None = live
    physics.json, unmodified) -- applied to match.tackling_params (a plain
    settable Match field, same pattern as match.repulsion_params/match.rng)
    post-construction, for both players equally, so it's a fair test of
    "how do these two AIs compare under harsher/different tackle rules",
    not a per-AI advantage."""
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.ai.env.scenario_env import ScenarioEnv

    def _build(*args, **kwargs):
        match = build_1v1_scenario(*args, **kwargs)
        # Deterministic per-episode reseed (NOT evaluate.py's usual fresh
        # random.Random() entropy) -- we always run repeats_per_seed=1 here,
        # so there's no need for independent repeats of the same seed; this
        # way baseline and every experimental variant see BYTE-IDENTICAL
        # physics rolls (control-time noise, tackle rolls, ...) for the same
        # seed, isolating the comparison to genuine AI-behaviour differences
        # instead of also comparing two different random physics realizations.
        match.rng = random.Random(kwargs.get("seed"))
        if tackling_params is not None:
            match.tackling_params = tackling_params
        match.player_by_id("trainee").ai = trainee_ai_factory()
        match.player_by_id("opponent").ai = Phase1RulesAI()
        return match

    def _env_factory(seed: int) -> ScenarioEnv:
        def _b(*_a, **_kw):
            return _build(*_a, seed=seed, **_kw)

        defn = ScenarioDefinition(
            key="debug_rules_ai_1v1",
            label="debug_rules_ai.py A/B",
            description="rules vs rules, trainee AI swappable for tackle-angle experiment",
            build=_b,
        )
        return ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, secondary_player_ids=[])

    return _env_factory, None


def _classify_tackle_angle(target, tackler) -> str:
    """'front' (frontal-to-side, cos_angle >= 0) or 'back' (side-to-behind,
    cos_angle < 0) -- the EXACT same geometry and split point engine/
    tackling.py's tackle_angle_modifier() itself uses (angle between the
    dribbler's facing direction and the vector from dribbler to tackler),
    just binarized instead of interpolated into a modifier value."""
    dribbler_dir = Vector3.from_angle_xy(target.heading_rad, 1.0).xy()
    d_to_t = (tackler.position - target.position).xy()
    d_to_t_len = d_to_t.length()
    if d_to_t_len < 1e-9:
        return "front"  # degenerate (same position) -- tackle_angle_modifier itself returns 0.0 (neutral) here
    cos_angle = dribbler_dir.dot(d_to_t.normalized())
    return "front" if cos_angle >= 0.0 else "back"


def _install_tackle_stats_hooks(match, stats: dict, trainee_player_id: str = "trainee") -> None:
    """Wires Player.on_tackle/on_tackle_result (the same hooks the engine
    already fires for BC recording -- see Match._attempt_tackle_contact)
    onto every player in `match` to tally the TRAINEE's own attempted/
    successful tackles by approach half. Only counts the trainee's tackles
    (not the opponent tackling the trainee back) since the opponent is
    identical stock Phase1RulesAI in every comparison run -- its own tackle
    stats would just be noise, not something the experimental AI affects.

    Must be called fresh after each trial's Match is (re)built (hooks live
    on Player instances, which get recreated every episode).

    NOTE: on_tackle/on_tackle_result only fire for the ARMED tackle path
    (Match._check_armed_tackles, i.e. tackle_armed was set by an order like
    TackleAngleChaseOrder/GetPossessionOrder) -- the separate collision-only
    auto-tackle fallback (_check_head_on_tackles) deliberately never fires
    these (see Player.on_tackle's own field comment), so a rare incidental
    collision-tackle wouldn't be counted here. In this 1v1 chase scenario
    the armed path should cover the vast majority of real tackles.

    Attempt classification is stashed and reused for the matching result
    callback (rather than recomputed) because by the time on_tackle_result
    fires, possession may have ALREADY changed hands if the tackle
    succeeded -- match.ball_carrier() at that point could be the tackler,
    not the original dribbler, corrupting a fresh angle recomputation.
    Safe because _check_armed_tackles resolves at most one tackle per tick
    (`return` right after), so there's no interleaving between an attempt
    and its own result.
    """
    pending: dict[str, str | None] = {"half": None}

    def _on_tackle(p) -> None:
        if p.player_id != trainee_player_id:
            pending["half"] = None
            return
        target = match.ball_carrier()
        if target is None or target.player_id == p.player_id:
            pending["half"] = None
            return
        half = _classify_tackle_angle(target, p)
        pending["half"] = half
        stats["attempted"][half] += 1

    def _on_tackle_result(p, tackler_won: bool, is_tackler: bool) -> None:
        if not is_tackler or p.player_id != trainee_player_id:
            return
        half = pending["half"]
        pending["half"] = None
        if half is not None and tackler_won:
            stats["successful"][half] += 1

    for player in match.players:
        player.on_tackle = _on_tackle
        player.on_tackle_result = _on_tackle_result


def run_tackle_angle_stats(
    label: str, trainee_ai_factory: Callable[[], object], n_episodes: int, seed_base: int, tackling_params=None,
) -> dict:
    """Runs n_episodes episodes (same paired-seed harness as _run_eval),
    tallying the trainee's tackle attempts/successes by approach half
    alongside the usual win/reward/outcome numbers. tackling_params: see
    _rules_vs_rules_env_worker_factory (None = live physics.json)."""
    env_factory, sample_action_fn = _rules_vs_rules_env_worker_factory(trainee_ai_factory, tackling_params=tackling_params)
    stats = {"attempted": {"front": 0, "back": 0}, "successful": {"front": 0, "back": 0}}
    seeds = list(range(seed_base, seed_base + n_episodes))
    win_count = 0
    total_reward = 0.0
    outcomes: dict[str, int] = {}

    for seed in seeds:
        env = env_factory(seed)
        env.sample_action_fn = sample_action_fn
        env.reset()
        _install_tackle_stats_hooks(env.match, stats)
        done = False
        ep_reward = 0.0
        info = None
        while not done:
            _, reward, done, info = env.step()
            ep_reward += reward
        total_reward += ep_reward
        outcome = info.trial_outcome if info else "unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == "box_possession":
            win_count += 1

    win_pct = 100.0 * win_count / n_episodes
    mean_rew = total_reward / n_episodes
    log.info(f"[{label}] win={win_pct:.1f}%  mean_rew={mean_rew:.3f}  outcomes={outcomes}")
    att, succ = stats["attempted"], stats["successful"]
    log.info(
        f"[{label}] tackles (trainee only) -- "
        f"attempted: front={att['front']}  back={att['back']}  (total={att['front'] + att['back']})  |  "
        f"successful: front={succ['front']}  back={succ['back']}  (total={succ['front'] + succ['back']})"
    )
    return stats


def watch_seed(
    seed: int, trainee_ai_factory: Callable[[], object], tackling_params=None, paused: bool = False,
) -> None:
    """Launch the match UI with this exact seed, trainee playing
    trainee_ai_factory() and opponent playing stock Phase1RulesAI -- the SAME
    1v1_phase1 scenario/seed pairing _rules_vs_rules_env_worker_factory uses
    for eval, just visual instead of aggregated. Reuses App._start_scenario
    exactly like ai/scripts/replay_episode.py's run_ui() does -- this jumps
    straight to Screen.MATCH, the main menu / scenario-params screens never
    render at all.

    tackling_params: optional TacklingParams override (None = live
    physics.json), same override mechanism as _rules_vs_rules_env_worker_
    factory's own tackling_params param -- needed to watch a seed that was
    found/scanned under a harsh-tackling override, since the live config
    alone would simulate a DIFFERENT trajectory for the same seed.

    paused: if False (default), the match auto-plays from the start --
    appropriate here since ScenarioLoop's own max_trials defaults to 0
    (infinite), so once a trial ends (linger_s later) it automatically
    rebuilds and replays the SAME seed again, forever -- i.e. this function
    already loops the exact same episode on repeat with no extra plumbing,
    as long as every rebuild re-applies the seed/tackling_params/AI
    overrides too (see the custom ScenarioDefinition.build closure below --
    reusing SCENARIOS's static "1v1_phase1" entry + one-shot post-hoc
    attribute overrides, as the old version of this function did, would only
    apply those overrides to the FIRST play-through: every subsequent
    auto-replay calls definition.build() fresh via ScenarioLoop._start_trial()
    and would silently fall back to default rng/tackling_params/AI).
    """
    from footballcoach.ui.app import App
    from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario
    from footballcoach.ai.scripts.replay_episode import _real_sim_dt_s

    def _build(rng_reduction: float = 0.3, **kwargs) -> object:
        match = build_1v1_scenario(rng_reduction, **kwargs)
        match.rng = random.Random(seed)
        if tackling_params is not None:
            match.tackling_params = tackling_params
        match.player_by_id("trainee").ai = trainee_ai_factory()
        match.player_by_id("opponent").ai = Phase1RulesAI()
        return match

    defn = ScenarioDefinition(
        key="debug_watch_seed",
        label="debug_rules_ai.py watch_seed",
        description="Deterministic single-seed replay for reviewing one specific trial; loops automatically.",
        build=_build,
        phase1_trainee_player_id="trainee",
    )
    app = App()
    # sim_dt_s MUST match what eval actually runs at (ai_config.json's
    # observation.sim_dt_s, ~0.06s/16.7Hz) -- the UI's own interactive
    # default is a fixed 30Hz. Same seed, different timestep size, still
    # gives a genuinely different simulated trajectory (discrete physics
    # isn't invariant to step size) -- confirmed the hard way: watching a
    # seed found via the eval harness's own env_factory at the UI's default
    # 30Hz showed the OPPOSITE outcome from what eval recorded for it.
    app._start_scenario(defn, kwargs={"seed": seed, "sim_dt_s": _real_sim_dt_s()})
    # The sim_dt_s kwarg above only reaches the Match constructor's initial
    # value -- App._start_scenario() (app.py) unconditionally overwrites
    # match.dt_s right afterward using the UI's own physics_tick_hz config
    # (default 30), discarding whatever the builder set. Confirmed by direct
    # comparison: a ScenarioEnv-based headless replay of this exact
    # construction path DID match eval's recorded outcome for a known seed;
    # only the live App path (this override) diverged. Must re-force it
    # AFTER _start_scenario, not just pass it as a kwarg.
    app.match.dt_s = _real_sim_dt_s()
    # app.py's own frame-pacing (_step_match: steps = physics_tick_hz *
    # sim_speed / target_fps) assumes match.dt_s is ALWAYS 1/physics_tick_hz
    # -- it has no idea we just changed dt_s above. Left alone, it keeps
    # advancing the SAME number of physics ticks per rendered frame, each
    # now covering MORE simulated time than before -- confirmed the hard
    # way: played back visibly faster than real-time at sim_speed=1. Keep
    # physics_tick_hz in sync with the real dt_s so "steps per frame" still
    # produces correct real-time pacing.
    app._physics_tick_hz = 1.0 / _real_sim_dt_s()
    # Syncing physics_tick_hz alone isn't enough: steps is `max(1, round(...))`
    # -- a hard floor of 1 physics tick per RENDERED frame. Real sim_dt_s
    # (~16.7-20Hz) is slower than the UI's normal 60fps render rate, so
    # `physics_tick_hz * sim_speed / target_fps` rounds toward 0 and gets
    # clamped back up to 1 -- advancing a full (larger) dt_s-sized tick every
    # frame at 60fps still overshoots real-time by roughly target_fps /
    # physics_tick_hz (~3x). Cap the render rate at physics_tick_hz too so
    # `steps` naturally lands at 1 with no clamp-induced overshoot.
    app._target_fps = max(1, round(app._physics_tick_hz))
    # The real bug behind "UI shows the trainee winning on a seed eval
    # recorded as a loss" wasn't timing/patience -- it was that the
    # "1v1_phase1" ScenarioDefinition's outcome detection didn't know who
    # was trainee vs opponent at all (generic "any player, any box" check),
    # so an OPPONENT reaching ITS OWN win condition still just got logged as
    # the same undifferentiated "box_possession" string, which read exactly
    # like a trainee win. Fixed at the source (ai/env/outcome.py's
    # detect_phase1_box_terminal, wired in via ScenarioDefinition.
    # phase1_trainee_player_id="trainee" on "1v1_phase1" itself) -- this
    # scenario now reports box_possession/opponent_box_possession correctly
    # no matter how it's driven, so no special-casing is needed here.
    # rng/tackling_params/AI are all applied inside _build() above, which
    # ScenarioLoop re-invokes on every automatic trial rebuild (max_trials
    # defaults to 0 = infinite) -- so nothing further needed here for that;
    # re-setting app.match.* post-hoc, as the old version of this function
    # did, would only ever affect the very first trial.
    app.match.paused = paused
    log.info(
        f"Watching seed={seed}{'' if tackling_params is None else ' (custom tackling_params)'} -- "
        f"{'space to unpause/pause, ' if paused else 'auto-playing, space to pause, '}"
        f"loops the same seed automatically -- see in-UI help for other controls."
    )
    app.run()


def _run_eval(label: str, trainee_ai_factory: Callable[[], object], n_episodes: int, seed_base: int, repeats: int):
    from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation

    env_factory, sample_action_fn = _rules_vs_rules_env_worker_factory(trainee_ai_factory)
    seeds = list(range(seed_base, seed_base + n_episodes))
    result = run_seeded_evaluation(env_factory, sample_action_fn, seeds, repeats)
    d = result.as_dict()
    log.info(
        f"[{label}] win={d['win_rate_pct']:.1f}%  mean_rew={d['mean_reward']:.3f}"
        f"±{d['std_reward']:.3f}  outcomes={d['outcomes']}"
    )
    return d


# One-at-a-time sweep spec: (param_name, is_repulsion_field, [non-default
# values to try]). Baseline/current-default rows are computed once and
# reused across every param's block -- see run_full_param_sweep().
_DEFAULT_REPULSION = RepulsionParams.from_config()
_DEFAULT_GOOD_ANGLE_THRESHOLD = 0.0

_SWEEP_SPECS: list[tuple[str, bool, list[float]]] = [
    ("radius_m", True, [5.0, 7.0, 12.0, 16.0]),
    ("strength_base", True, [0.25, 0.35, 0.75, 1.0]),
    ("alignment_dot_threshold", True, [-0.9, -0.6, -0.4, -0.2]),
    ("max_tangent_deg", True, [30.0, 60.0, 90.0]),
    ("max_deflection_deg", True, [40.0, 60.0, 110.0, 150.0]),
    ("velocity_lookahead_s", True, [0.0, 0.2, 0.6, 0.8]),
    ("good_angle_cos_threshold", False, [-0.3, -0.15, 0.15, 0.3]),
]


def _make_trainee_factory(
    *, good_angle_cos_threshold: float, repulsion_params, intercept_ahead_s: float = 0.0,
) -> Callable[[], object]:
    return lambda: TackleAngleAwareRulesAI(
        good_angle_cos_threshold=good_angle_cos_threshold,
        sprint_min_dt_s=None,  # stock sprint -- isolate repulsion/angle params only
        repulsion_params=repulsion_params,
        intercept_ahead_s=intercept_ahead_s,
    )


def run_full_param_sweep(n_episodes: int, seed_base: int) -> None:
    """One-at-a-time sweep over every repulsion/angle param, holding every
    OTHER param at its current default -- NOT a combinatorial grid (7 params
    at ~5 values each would be thousands of runs). Same seed_base/n_episodes
    (and therefore the same paired seeds) for every single run in this
    function, so every row is directly comparable to every other."""
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)
    current_defaults = _run_eval(
        "current defaults (angle_threshold=0.0, stock repulsion params)",
        _make_trainee_factory(
            good_angle_cos_threshold=_DEFAULT_GOOD_ANGLE_THRESHOLD, repulsion_params=None,
        ),
        n_episodes, seed_base, 1,
    )

    log.info(
        f"\n{'param':<28}{'value':>10}{'win%':>8}{'vs base':>10}{'vs default':>12}{'mean_rew':>11}"
    )
    log.info(
        f"{'baseline (no repulsion)':<28}{'':>10}{baseline['win_rate_pct']:>7.1f}%{'':>10}{'':>12}"
        f"{baseline['mean_reward']:>11.3f}"
    )
    log.info(
        f"{'current defaults':<28}{'':>10}{current_defaults['win_rate_pct']:>7.1f}%"
        f"{current_defaults['win_rate_pct'] - baseline['win_rate_pct']:>+9.1f}pp{'(ref)':>12}"
        f"{current_defaults['mean_reward']:>11.3f}"
    )

    for param_name, is_repulsion_field, values in _SWEEP_SPECS:
        log.info(f"-- {param_name} (default={getattr(_DEFAULT_REPULSION, param_name) if is_repulsion_field else _DEFAULT_GOOD_ANGLE_THRESHOLD}) --")
        for value in values:
            if is_repulsion_field:
                repulsion_params = _dc_replace(_DEFAULT_REPULSION, **{param_name: value})
                good_angle_cos_threshold = _DEFAULT_GOOD_ANGLE_THRESHOLD
            else:
                repulsion_params = None
                good_angle_cos_threshold = value
            result = _run_eval(
                f"{param_name}={value}",
                _make_trainee_factory(
                    good_angle_cos_threshold=good_angle_cos_threshold, repulsion_params=repulsion_params,
                ),
                n_episodes, seed_base, 1,
            )
            log.info(
                f"{param_name:<28}{value:>10.3g}{result['win_rate_pct']:>7.1f}%"
                f"{result['win_rate_pct'] - baseline['win_rate_pct']:>+9.1f}pp"
                f"{result['win_rate_pct'] - current_defaults['win_rate_pct']:>+11.1f}pp"
                f"{result['mean_reward']:>11.3f}"
            )


# Round 5: the redesigned repulsion mechanic (steering.py's smooth
# tangential rotation + closing-velocity magnitude gating, replacing the old
# fixed-magnitude orthogonal nudge). "current defaults" here reads
# orders.json live -- i.e. whatever's currently playtested/tuned there
# (_DEFAULT_REPULSION = RepulsionParams.from_config(), read at import time),
# not the values from earlier rounds -- so this run's "current defaults" row
# genuinely is "try the new default" against stock Phase1RulesAI. 3 values
# each including current default, one-at-a-time. max_deflection_deg swapped
# out for max_tangent_deg (the new mechanism's own defining parameter, more
# relevant to re-test now than the already-well-characterized deflection cap).
_ROUND5_SPECS: list[tuple[str, bool, list[float]]] = [
    ("radius_m", True, [5.0, 7.0, 9.0]),
    ("strength_base", True, [0.2, 0.4, 0.8]),
    ("alignment_dot_threshold", True, [-0.75, -0.5, -0.25]),
    ("max_tangent_deg", True, [0.0, 40.0, 80.0]),
    ("good_angle_cos_threshold", False, [-0.5, -0.25, 0.0]),
]


def run_round5_sweep(n_episodes: int, seed_base: int) -> None:
    """5 vars x 3 values (including each one's CURRENT LIVE default from
    orders.json), one-at-a-time, on the redesigned repulsion mechanic.
    max_tangent_deg=0.0 isolates the rotation mechanism itself (still gets
    the new closing-velocity magnitude gating either way, since that's
    unconditional) from radius_m/strength_base/alignment_dot_threshold's
    own effects."""
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)
    current_defaults = _run_eval(
        "current defaults (live orders.json repulsion params, angle_threshold=0.0)",
        _make_trainee_factory(good_angle_cos_threshold=_DEFAULT_GOOD_ANGLE_THRESHOLD, repulsion_params=None),
        n_episodes, seed_base, 1,
    )
    log.info(
        f"\n{'param':<28}{'value':>10}{'win%':>8}{'vs base':>10}{'vs default':>12}{'mean_rew':>11}"
    )
    log.info(
        f"{'baseline (no repulsion)':<28}{'':>10}{baseline['win_rate_pct']:>7.1f}%{'':>10}{'':>12}"
        f"{baseline['mean_reward']:>11.3f}"
    )
    log.info(
        f"{'current defaults (live)':<28}{'':>10}{current_defaults['win_rate_pct']:>7.1f}%"
        f"{current_defaults['win_rate_pct'] - baseline['win_rate_pct']:>+9.1f}pp{'(ref)':>12}"
        f"{current_defaults['mean_reward']:>11.3f}"
    )

    for param_name, is_repulsion_field, values in _ROUND5_SPECS:
        default_val = getattr(_DEFAULT_REPULSION, param_name) if is_repulsion_field else _DEFAULT_GOOD_ANGLE_THRESHOLD
        log.info(f"-- {param_name} (live default={default_val}) --")
        for value in values:
            if is_repulsion_field:
                repulsion_params = _dc_replace(_DEFAULT_REPULSION, **{param_name: value})
                good_angle_cos_threshold = _DEFAULT_GOOD_ANGLE_THRESHOLD
            else:
                repulsion_params = None
                good_angle_cos_threshold = value
            result = _run_eval(
                f"{param_name}={value}",
                _make_trainee_factory(
                    good_angle_cos_threshold=good_angle_cos_threshold, repulsion_params=repulsion_params,
                ),
                n_episodes, seed_base, 1,
            )
            log.info(
                f"{param_name:<28}{value:>10.3g}{result['win_rate_pct']:>7.1f}%"
                f"{result['win_rate_pct'] - baseline['win_rate_pct']:>+9.1f}pp"
                f"{result['win_rate_pct'] - current_defaults['win_rate_pct']:>+11.1f}pp"
                f"{result['mean_reward']:>11.3f}"
            )


def run_round6_sweep(n_episodes: int, seed_base: int) -> None:
    """Small, focused sweep at a bigger N: repulsion_params always None (i.e.
    LIVE orders.json defaults throughout, never overridden), varying only
    good_angle_cos_threshold and intercept_ahead_s (both debug-script-only
    params, not RepulsionParams fields) -- one-at-a-time, each against the
    same no-repulsion baseline. Reward AND full outcome breakdown reported
    for every row (see _run_eval's own log line) so wins/losses/timeouts/
    invalids are all directly comparable, not just the win-rate scalar."""
    baseline = _run_eval("baseline (stock Phase1RulesAI, no repulsion)", Phase1RulesAI, n_episodes, seed_base, 1)
    current_defaults = _run_eval(
        "current defaults (live orders.json, angle_threshold=0.0, intercept_ahead_s=0.0)",
        _make_trainee_factory(good_angle_cos_threshold=_DEFAULT_GOOD_ANGLE_THRESHOLD, repulsion_params=None),
        n_episodes, seed_base, 1,
    )

    def _row(label: str, result) -> None:
        log.info(
            f"{label:<45}win={result['win_rate_pct']:>5.1f}%  "
            f"vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}"
        )

    log.info("\n-- good_angle_cos_threshold (live repulsion defaults, intercept_ahead_s=0.0) --")
    for a in (-0.25, 0.0, 0.25):
        result = _run_eval(
            f"good_angle_cos_threshold={a}",
            _make_trainee_factory(good_angle_cos_threshold=a, repulsion_params=None),
            n_episodes, seed_base, 1,
        )
        _row(f"good_angle_cos_threshold={a}", result)

    log.info("\n-- intercept_ahead_s (live repulsion defaults, good_angle_cos_threshold=0.0) --")
    for lead in (0.0, 0.15, 0.3):
        result = _run_eval(
            f"intercept_ahead_s={lead}",
            _make_trainee_factory(
                good_angle_cos_threshold=_DEFAULT_GOOD_ANGLE_THRESHOLD, repulsion_params=None,
                intercept_ahead_s=lead,
            ),
            n_episodes, seed_base, 1,
        )
        _row(f"intercept_ahead_s={lead}", result)


def _eval_repulsion_config(
    label: str, n_episodes: int, seed_base: int, *,
    good_angle_cos_threshold: float = _DEFAULT_GOOD_ANGLE_THRESHOLD,
    intercept_ahead_s: float = 0.0, **repulsion_overrides,
):
    repulsion_params = _dc_replace(_DEFAULT_REPULSION, **repulsion_overrides) if repulsion_overrides else None
    return _run_eval(
        label,
        _make_trainee_factory(
            good_angle_cos_threshold=good_angle_cos_threshold, repulsion_params=repulsion_params,
            intercept_ahead_s=intercept_ahead_s,
        ),
        n_episodes, seed_base, 1,
    )


def run_round2_sweep(n_episodes: int, seed_base: int) -> None:
    """Round 2, informed by round 1's results:
      - radius_m refined and capped at <=9 (round 1 showed smaller is
        better; explicitly avoiding the larger-radius direction per steer).
      - good_angle_cos_threshold and alignment_dot_threshold swept AGAIN,
        this time on top of radius_m=5.0 (round 1's best single change)
        instead of the old radius_m=9.0 default -- checks whether their
        own round-1 wins actually COMPOUND with the radius change, since
        round 1 only ever changed one param at a time off the old defaults.
    """
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)
    log.info(f"baseline (no repulsion): win={baseline['win_rate_pct']:.1f}%  mean_rew={baseline['mean_reward']:.3f}")

    log.info("\n-- Sweep A: radius_m alone, refined + capped <=9 --")
    for r in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
        result = _eval_repulsion_config(f"radius_m={r}", n_episodes, seed_base, radius_m=r)
        log.info(
            f"radius_m={r:<5.1f} win={result['win_rate_pct']:>5.1f}%  "
            f"vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}"
        )

    log.info("\n-- Sweep B: good_angle_cos_threshold on top of radius_m=5.0 --")
    for a in (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0):
        result = _eval_repulsion_config(
            f"radius5+angle={a}", n_episodes, seed_base, good_angle_cos_threshold=a, radius_m=5.0,
        )
        log.info(
            f"good_angle={a:<5.2f} (radius_m=5.0)  win={result['win_rate_pct']:>5.1f}%  "
            f"vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}"
        )

    log.info("\n-- Sweep C: alignment_dot_threshold on top of radius_m=5.0 --")
    for al in (-0.3, -0.25, -0.2, -0.15, -0.1):
        result = _eval_repulsion_config(
            f"radius5+align={al}", n_episodes, seed_base, radius_m=5.0, alignment_dot_threshold=al,
        )
        log.info(
            f"alignment={al:<6.2f} (radius_m=5.0)  win={result['win_rate_pct']:>5.1f}%  "
            f"vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}"
        )


def run_round3_sweep(n_episodes: int, seed_base: int) -> None:
    """Round 3, at a bigger N to confirm round 2's winner isn't a small-N
    fluke, plus following up on two loose threads from round 2:
      - Sweep A: good_angle_cos_threshold pushed FURTHER negative than
        round 2 tried (-0.5 was the most extreme value tested there and
        also the best result found all session -- extending the range
        checks whether the true peak is even further out, or -0.5 was it).
      - Sweep B: radius_m refined more finely around 5.0, this time on top
        of good_angle_cos_threshold=-0.5 (round 2's best) instead of the
        default 0.0, to confirm the radius/angle combination together.
      - Sweep C: intercept_ahead_s tried AGAIN, this time on top of the new
        best combo (radius_m=5.0, good_angle=-0.5) instead of the old bad
        defaults it was strictly monotonically awful against earlier this
        session -- worth a genuine second look in a completely different
        operating regime before writing the idea off entirely.
    """
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)
    log.info(f"baseline (no repulsion): win={baseline['win_rate_pct']:.1f}%  mean_rew={baseline['mean_reward']:.3f}")

    log.info("\n-- Sweep A: good_angle_cos_threshold pushed further negative, on top of radius_m=5.0 --")
    for a in (-0.8, -0.7, -0.6, -0.5, -0.4):
        result = _eval_repulsion_config(
            f"radius5+angle={a}", n_episodes, seed_base, good_angle_cos_threshold=a, radius_m=5.0,
        )
        log.info(
            f"good_angle={a:<5.2f} (radius_m=5.0)  win={result['win_rate_pct']:>5.1f}%  "
            f"vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}"
        )

    log.info("\n-- Sweep B: radius_m refined around 5.0, on top of good_angle_cos_threshold=-0.5 --")
    for r in (4.0, 4.5, 5.0, 5.5, 6.0):
        result = _eval_repulsion_config(
            f"radius={r}+angle-0.5", n_episodes, seed_base, good_angle_cos_threshold=-0.5, radius_m=r,
        )
        log.info(
            f"radius_m={r:<5.1f} (good_angle=-0.5)  win={result['win_rate_pct']:>5.1f}%  "
            f"vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}"
        )

    log.info("\n-- Sweep C: intercept_ahead_s on top of radius_m=5.0 + good_angle_cos_threshold=-0.5 --")
    for lead in (0.0, 0.1, 0.2, 0.3):
        result = _eval_repulsion_config(
            f"radius5+angle-0.5+lead={lead}", n_episodes, seed_base,
            good_angle_cos_threshold=-0.5, intercept_ahead_s=lead, radius_m=5.0,
        )
        log.info(
            f"intercept_ahead_s={lead:<4.2f} (radius_m=5.0, good_angle=-0.5)  win={result['win_rate_pct']:>5.1f}%  "
            f"vs_base={result['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  "
            f"mean_rew={result['mean_reward']:.3f}"
        )


# Round 4: the 5 variables with the strongest evidence of a REAL (not
# small-N noise) effect across rounds 1-3 -- velocity_lookahead_s (proven
# structurally inert in a 1v1: only one possible repulsion source ever
# exists, the carrier, so the ahead/behind gate has nothing else to
# exclude) and min_orthogonal_adjust_mps (zero variation across every
# tested value in round 1) are dropped. 3 values each including the
# CURRENT production default, one-at-a-time off that default (not off
# round 2/3's radius=5/angle=-0.5 guess, which round 3 showed doesn't
# robustly beat baseline anyway).
_ROUND4_SPECS: list[tuple[str, bool, list[float]]] = [
    ("radius_m", True, [5.0, 7.0, 9.0]),
    ("good_angle_cos_threshold", False, [-0.5, -0.25, 0.0]),
    ("strength_base", True, [0.25, 0.5, 1.0]),
    ("alignment_dot_threshold", True, [-0.75, -0.5, -0.25]),
    ("max_deflection_deg", True, [40.0, 60.0, 80.0]),
]


def run_round4_sweep(n_episodes: int, seed_base: int) -> None:
    """5 vars x 3 values (including each one's current default), one-at-a-
    time off the CURRENT PRODUCTION DEFAULTS (not round 2/3's radius=5/
    angle=-0.5 guess -- round 3 at N=150 showed that combo doesn't robustly
    beat baseline). At a properly-powered N this time (win-rate SEM at
    N=200 is ~3.2pp, vs ~9pp at N=25 -- round 1/2's "wins" were well within
    noise at that size)."""
    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, n_episodes, seed_base, 1)
    current_defaults = _run_eval(
        "current defaults (angle_threshold=0.0, stock repulsion params)",
        _make_trainee_factory(good_angle_cos_threshold=_DEFAULT_GOOD_ANGLE_THRESHOLD, repulsion_params=None),
        n_episodes, seed_base, 1,
    )
    log.info(
        f"\n{'param':<28}{'value':>10}{'win%':>8}{'vs base':>10}{'vs default':>12}{'mean_rew':>11}"
    )
    log.info(
        f"{'baseline (no repulsion)':<28}{'':>10}{baseline['win_rate_pct']:>7.1f}%{'':>10}{'':>12}"
        f"{baseline['mean_reward']:>11.3f}"
    )
    log.info(
        f"{'current defaults':<28}{'':>10}{current_defaults['win_rate_pct']:>7.1f}%"
        f"{current_defaults['win_rate_pct'] - baseline['win_rate_pct']:>+9.1f}pp{'(ref)':>12}"
        f"{current_defaults['mean_reward']:>11.3f}"
    )

    for param_name, is_repulsion_field, values in _ROUND4_SPECS:
        default_val = getattr(_DEFAULT_REPULSION, param_name) if is_repulsion_field else _DEFAULT_GOOD_ANGLE_THRESHOLD
        log.info(f"-- {param_name} (default={default_val}) --")
        for value in values:
            if is_repulsion_field:
                repulsion_params = _dc_replace(_DEFAULT_REPULSION, **{param_name: value})
                good_angle_cos_threshold = _DEFAULT_GOOD_ANGLE_THRESHOLD
            else:
                repulsion_params = None
                good_angle_cos_threshold = value
            result = _run_eval(
                f"{param_name}={value}",
                _make_trainee_factory(
                    good_angle_cos_threshold=good_angle_cos_threshold, repulsion_params=repulsion_params,
                ),
                n_episodes, seed_base, 1,
            )
            log.info(
                f"{param_name:<28}{value:>10.3g}{result['win_rate_pct']:>7.1f}%"
                f"{result['win_rate_pct'] - baseline['win_rate_pct']:>+9.1f}pp"
                f"{result['win_rate_pct'] - current_defaults['win_rate_pct']:>+11.1f}pp"
                f"{result['mean_reward']:>11.3f}"
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--seed-base", type=int, default=3_000_000)
    parser.add_argument("--repeats-per-seed", type=int, default=1)
    parser.add_argument(
        "--tackle-angle-stats", action="store_true",
        help="Baseline vs good_angle_cos_threshold=-0.25 (live orders.json "
             "repulsion defaults, stock sprint) -- same comparison as the "
             "N=2000 confirmation run, but also tallies the trainee's own "
             "tackle attempts/successes by approach half (front = frontal-"
             "to-side, back = side-to-behind, matching engine/tackling.py's "
             "tackle_angle_modifier() geometry exactly). See "
             "run_tackle_angle_stats(). Ignores every other experimental "
             "flag below.",
    )
    parser.add_argument(
        "--jockey-grid-final", action="store_true",
        help="margin in (3.0,5.0) x blend in (0.8,0.95), N=2000-scale, same "
             "win/reward/outcomes + front/back tackle breakdown + trigger "
             "rate as --jockey-tackle-stats-sweep. See "
             "run_jockey_grid_final(). Ignores every other experimental "
             "flag below.",
    )
    parser.add_argument(
        "--jockey-grid-final-harsh", action="store_true",
        help="Same as --jockey-grid-final, but with the SAME harsh "
             "TacklingParams override used by --tackle-angle-stats-harsh "
             "(tackle_attempt_tackler_speed_mult=0.6, "
             "tackle_attempt_tacklee_speed_mult=0.9, "
             "loser_speed_penalty_max=0.9, angle_modifier_behind=-0.6), "
             "in-simulation only (physics.json untouched), applied to "
             "baseline and every jockey variant equally. Ignores every "
             "other experimental flag below.",
    )
    parser.add_argument(
        "--combined-final", action="store_true",
        help="Both proven features together (tackle-angle curving + "
             "jockeying) vs stock baseline, under the SAME harsh "
             "TacklingParams override as --jockey-grid-final-harsh -- the "
             "final combined evaluation. Defaults: "
             "good_angle_cos_threshold=-0.25, jockey_give_up_margin=4.0, "
             "jockey_blend_factor=0.8 (override via --angle-threshold/"
             "--jockey-margins first value/--jockey-blends first value). "
             "See run_combined_final(). Ignores every other experimental "
             "flag below.",
    )
    parser.add_argument(
        "--jockey-tackle-stats-sweep", action="store_true",
        help="N=2000-scale sweep of jockey_give_up_margin in "
             "(1.5,1.8,2.0,2.5,3.0) at blend=0.8, with front/back tackle "
             "breakdown and trigger rate per point -- tests whether making "
             "the trigger stricter converges toward baseline or stays "
             "net-negative even when rarely firing. See "
             "run_jockey_tackle_stats_sweep(). Ignores every other "
             "experimental flag below.",
    )
    parser.add_argument(
        "--jockey-sweep3", action="store_true",
        help="Round 3: genuine 2D grid over margin in (1.3,1.4,1.5) x blend "
             "in (0.6,0.7,0.8), 9 combos -- rounds 1-2 only tested one axis "
             "at a time, this checks for a joint optimum they'd have "
             "missed. Logs the single best combo found. See "
             "run_jockey_sweep3(). Ignores every other experimental flag.",
    )
    parser.add_argument(
        "--jockey-sweep2", action="store_true",
        help="Round 2 jockey sweep: blend_factor in (0.7,0.8,0.9,0.95) at "
             "margin=1.5, margin in (1.4,1.6,1.8) at blend=0.8 -- pushing "
             "further in the direction round 1 trended toward. See "
             "run_jockey_sweep2(). Ignores every other experimental flag.",
    )
    parser.add_argument(
        "--jockey-sweep", action="store_true",
        help="One-at-a-time sweep over jockey_give_up_margin (1.1-2.0) and "
             "jockey_blend_factor (0.2-0.8), reporting win rate/reward AND "
             "trigger rate for every point. See run_jockey_sweep(). Ignores "
             "every other experimental flag below.",
    )
    parser.add_argument(
        "--jockey-test", action="store_true",
        help="Baseline vs jockey-enabled TackleAngleAwareRulesAI (tackle-"
             "angle curving left at its default, so this isolates the "
             "jockey feature specifically) -- when a loose ball's race is "
             "clearly lost (see _opponent_clearly_wins_loose_ball_race), "
             "positions halfway between the ball and the opponent's own "
             "box target instead of chasing hopelessly. "
             "--jockey-give-up-margin/--jockey-blend-factor control it. "
             "Ignores every other experimental flag below.",
    )
    parser.add_argument("--jockey-give-up-margin", type=float, default=1.5)
    parser.add_argument("--jockey-blend-factor", type=float, default=0.5)
    parser.add_argument(
        "--jockey-margins", type=str, default=None,
        help="Comma-separated jockey_give_up_margin values for "
             "--jockey-grid-final/--jockey-grid-final-harsh (default "
             "'3.0,5.0' if not set).",
    )
    parser.add_argument(
        "--jockey-blends", type=str, default=None,
        help="Comma-separated jockey_blend_factor values for "
             "--jockey-grid-final/--jockey-grid-final-harsh (default "
             "'0.8,0.95' if not set).",
    )
    parser.add_argument(
        "--tackle-angle-stats-harsh", action="store_true",
        help="Same as --tackle-angle-stats (baseline vs good_angle_cos_"
             "threshold=-0.25, with tackle attempt/success-by-approach-half "
             "tallies), but with a harsher TacklingParams override applied "
             "to BOTH runs equally, in-simulation only (physics.json "
             "untouched): angle_modifier_behind -0.5 -> -0.65 (tackling from "
             "behind is harder), loser_speed_penalty_max 0.8 -> 0.95 (a "
             "failed tackle attempt costs more speed). Tests whether making "
             "approach angle matter more finally surfaces a win-rate benefit "
             "from the curving AI, given tackle-angle-stats showed a real "
             "success-rate improvement that didn't move win rate under "
             "current (softer) tackling rules. Ignores every other "
             "experimental flag below.",
    )
    parser.add_argument(
        "--round6-sweep", action="store_true",
        help="Small sweep at a bigger N: repulsion_params always None (live "
             "orders.json defaults, never overridden), varying only "
             "good_angle_cos_threshold and intercept_ahead_s one-at-a-time "
             "against the no-repulsion baseline, with full outcome "
             "breakdowns. See run_round6_sweep(). Ignores every other "
             "experimental flag below.",
    )
    parser.add_argument(
        "--round5-sweep", action="store_true",
        help="5 vars (radius_m, strength_base, alignment_dot_threshold, "
             "max_tangent_deg, good_angle_cos_threshold) x 3 values each "
             "(including current default), one-at-a-time, on the REDESIGNED "
             "repulsion mechanic (smooth tangential rotation + closing-"
             "velocity gating). 'current defaults' reads orders.json LIVE, "
             "so this tests your current playtested tuning directly. See "
             "run_round5_sweep(). Ignores every other experimental flag.",
    )
    parser.add_argument(
        "--round4-sweep", action="store_true",
        help="5 vars (radius_m, good_angle_cos_threshold, strength_base, "
             "alignment_dot_threshold, max_deflection_deg) x 3 values each "
             "(including current default), one-at-a-time off current "
             "production defaults, at whatever --n-episodes you pass. "
             "See run_round4_sweep(). Ignores every other experimental flag.",
    )
    parser.add_argument(
        "--round3-sweep", action="store_true",
        help="Round 3, bigger N: pushes good_angle_cos_threshold further "
             "negative than round 2 tried (looking for the true peak past "
             "-0.5), refines radius_m around 5.0 on top of that, and tries "
             "intercept_ahead_s again on top of the new best combo instead "
             "of the old bad defaults. See run_round3_sweep(). Ignores "
             "every other experimental flag below.",
    )
    parser.add_argument(
        "--round2-sweep", action="store_true",
        help="Round 2, informed by --full-param-sweep's results: refined "
             "radius_m (capped <=9) alone, plus good_angle_cos_threshold and "
             "alignment_dot_threshold swept again on top of radius_m=5.0 to "
             "check compounding. See run_round2_sweep(). Ignores every other "
             "experimental flag below.",
    )
    parser.add_argument(
        "--full-param-sweep", action="store_true",
        help="Run a one-at-a-time sweep over radius_m/strength_base/"
             "alignment_dot_threshold/max_tangent_deg/"
             "max_deflection_deg/velocity_lookahead_s/good_angle_cos_threshold "
             "(see _SWEEP_SPECS), holding every OTHER param at its current "
             "default, all runs sharing the same seed_base/n_episodes for a "
             "direct comparison. Ignores every other experimental flag below.",
    )
    parser.add_argument(
        "--angle-threshold", type=float, default=0.0,
        help="cos(angle) threshold for 'safe to stop curving and close in' -- "
             "0.0 = frontal-to-side (matches tackle_angle_modifier's own side/"
             "frontal split at 90deg), negative allows a bit more of the "
             "behind range, positive demands a tighter frontal cone.",
    )
    parser.add_argument(
        "--sprint-min-dt-s", type=float, default=0.8,
        help="Absolute-time-buffer replacement for _should_sprint_to_ball's "
             "relative margin -- sprint whenever an opponent would beat us "
             "to the ball by at least this many seconds, instead of by a "
             "percentage of our own ETA (see _should_sprint_to_ball_min_dt's "
             "docstring for why the percentage version is weak at short ETAs).",
    )
    parser.add_argument(
        "--intercept-ahead-s", type=float, default=0.0,
        help="Extra lead distance (in seconds of the carrier's CURRENT "
             "velocity) added past the earliest-meeting-point intercept, so "
             "the chaser aims at open ground ahead of the carrier instead of "
             "their exact backside -- gives repulsion actual room to curve "
             "the approach instead of colliding into them immediately. "
             "0.0 (default) = no lead, identical to the earliest-point-only "
             "behaviour tested before.",
    )
    parser.add_argument(
        "--stock-sprint-eta", action="store_true",
        help="Use stock _should_sprint_to_ball (relative margin) instead of "
             "the --sprint-min-dt-s variant -- for isolating the angle change "
             "from the sprint-timing change.",
    )
    parser.add_argument(
        "--watch-seed", type=int, default=None,
        help="Skip eval entirely -- launch the match UI (pitch view only, "
             "paused) for this one seed, trainee playing the experimental "
             "AI (per --angle-threshold/--sprint-min-dt-s/--stock-sprint-eta/"
             "--intercept-ahead-s), opponent playing stock Phase1RulesAI. "
             "Same seed pairing eval uses, for eyeballing what a specific "
             "episode's numbers actually correspond to on the pitch.",
    )
    parser.add_argument(
        "--intercept-ahead-sweep", type=str, default=None,
        help="Comma-separated list of --intercept-ahead-s values to sweep, "
             "e.g. '0.05,0.1,0.15,0.2,0.3'. When set, runs baseline ONCE then "
             "one experimental eval per value (sprint pinned to stock, "
             "--sprint-min-dt-s/--stock-sprint-eta ignored) and prints a "
             "summary table -- overrides --intercept-ahead-s.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="NOTE: TackleAngleChaseOrder/TackleAngleAwareRulesAI are defined in "
             "this script, not importable by a spawned subprocess worker -- "
             "leave at 1 (sequential) unless you move those classes into an "
             "importable module first.",
    )
    args = parser.parse_args()
    if args.n_workers > 1:
        raise SystemExit(
            "--n-workers > 1 not supported: TackleAngleChaseOrder/TackleAngleAwareRulesAI "
            "live in this script and can't be pickled into a subprocess. Run sequentially."
        )
    sprint_min_dt_s = None if args.stock_sprint_eta else args.sprint_min_dt_s

    if args.jockey_grid_final or args.jockey_grid_final_harsh:
        tackling_params = None
        if args.jockey_grid_final_harsh:
            from footballcoach.engine.tackling import TacklingParams
            tackling_params = _dc_replace(
                TacklingParams.from_config(),
                tackle_attempt_tackler_speed_mult=0.6,
                tackle_attempt_tacklee_speed_mult=0.9,
                loser_speed_penalty_max=0.9,
                angle_modifier_behind=-0.6,
            )
            log.info(f"Harsh tackling_params override (in-simulation only): {tackling_params}")
        margins = tuple(float(x) for x in args.jockey_margins.split(",")) if args.jockey_margins else (3.0, 5.0)
        blends = tuple(float(x) for x in args.jockey_blends.split(",")) if args.jockey_blends else (0.8, 0.95)
        run_jockey_grid_final(
            args.n_episodes, args.seed_base, tackling_params=tackling_params, margins=margins, blends=blends,
        )
        return

    if args.combined_final:
        from footballcoach.engine.tackling import TacklingParams
        tackling_params = _dc_replace(
            TacklingParams.from_config(),
            tackle_attempt_tackler_speed_mult=0.6,
            tackle_attempt_tacklee_speed_mult=0.9,
            loser_speed_penalty_max=0.9,
            angle_modifier_behind=-0.6,
        )
        log.info(f"Harsh tackling_params override (in-simulation only): {tackling_params}")
        margin = float(args.jockey_margins.split(",")[0]) if args.jockey_margins else 4.0
        blend = float(args.jockey_blends.split(",")[0]) if args.jockey_blends else 0.8
        run_combined_final(
            args.n_episodes, args.seed_base, tackling_params=tackling_params,
            good_angle_cos_threshold=args.angle_threshold, jockey_give_up_margin=margin, jockey_blend_factor=blend,
        )
        return

    if args.jockey_tackle_stats_sweep:
        run_jockey_tackle_stats_sweep(args.n_episodes, args.seed_base)
        return

    if args.jockey_sweep3:
        run_jockey_sweep3(args.n_episodes, args.seed_base)
        return

    if args.jockey_sweep2:
        run_jockey_sweep2(args.n_episodes, args.seed_base)
        return

    if args.jockey_sweep:
        run_jockey_sweep(args.n_episodes, args.seed_base)
        return

    if args.jockey_test:
        baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, args.n_episodes, args.seed_base, 1)
        jockey = _run_eval(
            f"jockey (give_up_margin={args.jockey_give_up_margin}, blend_factor={args.jockey_blend_factor})",
            lambda: TackleAngleAwareRulesAI(
                jockey_give_up_margin=args.jockey_give_up_margin, jockey_blend_factor=args.jockey_blend_factor,
            ),
            args.n_episodes, args.seed_base, 1,
        )
        log.info(
            f"\nWin rate: baseline={baseline['win_rate_pct']:.1f}%  jockey={jockey['win_rate_pct']:.1f}%  "
            f"delta={jockey['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp  |  "
            f"mean_rew: baseline={baseline['mean_reward']:.3f}  jockey={jockey['mean_reward']:.3f}  "
            f"delta={jockey['mean_reward'] - baseline['mean_reward']:+.3f}"
        )
        return

    if args.tackle_angle_stats or args.tackle_angle_stats_harsh:
        tackling_params = None
        if args.tackle_angle_stats_harsh:
            from footballcoach.engine.tackling import TacklingParams
            tackling_params = _dc_replace(
                TacklingParams.from_config(),
                tackle_attempt_tackler_speed_mult=0.6,
                tackle_attempt_tacklee_speed_mult=0.9,
                loser_speed_penalty_max=0.9,
                angle_modifier_behind=-0.6,
            )
            log.info(f"Harsh tackling_params override (in-simulation only): {tackling_params}")
        run_tackle_angle_stats(
            "baseline (stock Phase1RulesAI)", Phase1RulesAI, args.n_episodes, args.seed_base,
            tackling_params=tackling_params,
        )
        run_tackle_angle_stats(
            f"good_angle_cos_threshold={args.angle_threshold}",
            lambda: TackleAngleAwareRulesAI(
                good_angle_cos_threshold=args.angle_threshold, sprint_min_dt_s=None, repulsion_params=None,
            ),
            args.n_episodes, args.seed_base, tackling_params=tackling_params,
        )
        return

    if args.round6_sweep:
        run_round6_sweep(args.n_episodes, args.seed_base)
        return

    if args.round5_sweep:
        run_round5_sweep(args.n_episodes, args.seed_base)
        return

    if args.round4_sweep:
        run_round4_sweep(args.n_episodes, args.seed_base)
        return

    if args.round3_sweep:
        run_round3_sweep(args.n_episodes, args.seed_base)
        return

    if args.round2_sweep:
        run_round2_sweep(args.n_episodes, args.seed_base)
        return

    if args.full_param_sweep:
        run_full_param_sweep(args.n_episodes, args.seed_base)
        return

    if args.watch_seed is not None:
        watch_seed(
            args.watch_seed,
            lambda: TackleAngleAwareRulesAI(
                good_angle_cos_threshold=args.angle_threshold, sprint_min_dt_s=sprint_min_dt_s,
                intercept_ahead_s=args.intercept_ahead_s,
            ),
        )
        return

    if args.intercept_ahead_sweep is not None:
        values = [float(v) for v in args.intercept_ahead_sweep.split(",")]
        log.info(f"Sweeping intercept_ahead_s={values} over {args.n_episodes} episodes each "
                 f"(sprint pinned to stock, angle_threshold={args.angle_threshold})")
        baseline = _run_eval(
            "baseline (stock Phase1RulesAI)", Phase1RulesAI, args.n_episodes, args.seed_base, args.repeats_per_seed,
        )
        rows = []
        for v in values:
            result = _run_eval(
                f"intercept_ahead_s={v}",
                lambda v=v: TackleAngleAwareRulesAI(
                    good_angle_cos_threshold=args.angle_threshold, sprint_min_dt_s=None, intercept_ahead_s=v,
                ),
                args.n_episodes, args.seed_base, args.repeats_per_seed,
            )
            rows.append((v, result))
        log.info("\nintercept_ahead_s  win%    delta      mean_rew   delta")
        log.info(f"baseline           {baseline['win_rate_pct']:5.1f}%              {baseline['mean_reward']:+.3f}")
        for v, result in rows:
            log.info(
                f"{v:<18.2f} {result['win_rate_pct']:5.1f}%  {result['win_rate_pct'] - baseline['win_rate_pct']:+6.1f}pp   "
                f"{result['mean_reward']:+.3f}    {result['mean_reward'] - baseline['mean_reward']:+.3f}"
            )
        return

    log.info(f"Running {args.n_episodes} episodes x{args.repeats_per_seed} repeats, "
             f"angle_threshold={args.angle_threshold}  sprint_min_dt_s={sprint_min_dt_s}  "
             f"intercept_ahead_s={args.intercept_ahead_s}")

    baseline = _run_eval("baseline (stock Phase1RulesAI)", Phase1RulesAI, args.n_episodes, args.seed_base, args.repeats_per_seed)

    _reset_trigger_stats()
    experimental = _run_eval(
        "experimental (tackle-angle-aware + sprint-min-dt)",
        lambda: TackleAngleAwareRulesAI(
            good_angle_cos_threshold=args.angle_threshold, sprint_min_dt_s=sprint_min_dt_s,
            intercept_ahead_s=args.intercept_ahead_s,
        ),
        args.n_episodes, args.seed_base, args.repeats_per_seed,
    )
    if sprint_min_dt_s is not None and _trigger_stats["calls"] > 0:
        ts = _trigger_stats
        log.info(
            f"[sprint trigger rates, min_dt_s={sprint_min_dt_s}] "
            f"calls={ts['calls']}  stock_fired={ts['stock']} ({ts['stock']/ts['calls']:.1%})  "
            f"margin_fired={ts['margin']} ({ts['margin']/ts['calls']:.1%})  "
            f"margin_only={ts['margin_only']} ({ts['margin_only']/ts['calls']:.1%}) "
            f"<- these are the ONLY calls where this change actually flips the sprint decision"
        )

    log.info(
        f"\nWin rate: baseline={baseline['win_rate_pct']:.1f}%  "
        f"experimental={experimental['win_rate_pct']:.1f}%  "
        f"delta={experimental['win_rate_pct'] - baseline['win_rate_pct']:+.1f}pp"
    )


if __name__ == "__main__":
    main()
