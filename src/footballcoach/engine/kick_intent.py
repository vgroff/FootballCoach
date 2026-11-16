"""Kick intents: what an order wants to kick this tick, applied by the engine.

Orders used to carry out their kick inside ``execute()`` (launching the ball,
ending a first touch, firing ``on_kick``), so the only way for a BC label to
learn what the rules AI would kick was to run the order for real and undo the
damage afterwards. Now a kicking order records a ``KickIntent`` on
``player.pending_kick`` and ``Match._process_orders`` applies it immediately
after that player's ``execute()`` returns -- the same point in the tick, and the
same per-player order, as the old in-``execute()`` kick, so play is unchanged.

A BC label's counterfactual ``execute()`` (ai/ppo/bc.py) reads the intent
instead: ``plan_kick_intent`` runs the SAME kick functions with
``release=False``, which return the planned launch (direction, speed, error
size) without drawing the random error, touching the ball or doing the kick
bookkeeping, and ``kick_label_fields`` turns that into the kick heads' targets.

Kinds, one per kicking order, each applied exactly as that order used to kick:

- ``"direction"`` (MoveOrder push-kick): ``Player.kick_with_direction``.
- ``"aim"`` (KickOrder): ``Player.kick_direct``.
- ``"shot"`` (ShootOrder): ``engine.kicking.kick_ball`` at ``power_fraction``
  (already run-compensated when the order asked for it).
- ``"pass"`` (PassOrder): ``engine.kicking.pass_ball``.

Not routed through here: the neural player's in-possession kick
(ai/action/apply_nn_action.py, which relies on the kick having happened when
it returns), the engine's own armed-kick on pickup, and the UI's manual kick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from footballcoach.mathutils import Vector3
from footballcoach.mathutils.tolerances import GEOM_ZERO_EPS

if TYPE_CHECKING:
    from footballcoach.engine.kicking import KickPlan
    from footballcoach.engine.match import Match
    from footballcoach.entities.player import Player


@dataclass(frozen=True)
class KickIntent:
    kind: str                                  # "direction" | "aim" | "shot" | "pass"
    power_fraction: Optional[float]            # as passed to the physics (None = pass auto-pace)
    direction_3d: Optional[Vector3] = None     # "direction"
    aim_point: Optional[Vector3] = None        # "aim", "shot"; the (led) target for "pass"
    spin: Vector3 = Vector3.zero()
    compensate_for_run: bool = False           # "aim" (kick_direct's own flag)
    power_multiplier: float = 1.0              # "pass"
    log_msg: str = ""                          # logged when applied, as the order used to


def _carry_out(player: "Player", match: "Match", intent: KickIntent, release: bool) -> "Optional[KickPlan]":
    """The kick, through the same function its order always used. With
    ``release=False`` that function returns its KickPlan instead of launching
    (None when the player cannot kick, exactly when a real kick would not
    happen)."""
    if intent.kind == "direction":
        return player.kick_with_direction(match, intent.direction_3d, intent.power_fraction, intent.spin,
                                          release=release)
    if intent.kind == "aim":
        return player.kick_direct(match, intent.aim_point, intent.power_fraction, intent.spin,
                                  intent.compensate_for_run, release=release)
    from footballcoach.engine.movement import effective_top_speed

    top_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina,
        has_ball=True, ball_control_attr=player.attributes.ball_control,
    )
    if intent.kind == "shot":
        from footballcoach.engine.kicking import kick_ball

        return kick_ball(
            match.ball,
            player.position,
            intent.aim_point,
            intent.power_fraction,
            player.attributes.kick_precision,
            player.attributes.kick_power,
            Vector3.zero(),
            match.rng_reduction,
            match.rng,
            match.kicking_params,
            kicker_velocity=player.velocity,
            kicker_top_speed_mps=top_speed,
            release=release,
        )
    if intent.kind == "pass":
        from footballcoach.engine.kicking import pass_ball

        return pass_ball(
            match.ball,
            player.position,
            intent.aim_point,
            player.attributes.kick_precision,
            match.rng_reduction,
            match.rng,
            match.passing_params,
            gravity_mps2=match.ball_physics_params.gravity_mps2,
            rolling_friction_coefficient=match.ball_physics_params.rolling_friction_coefficient,
            power_fraction=intent.power_fraction,
            running_power_coefficient=match.kicking_params.running_power_coefficient,
            kicker_velocity=player.velocity,
            kicker_top_speed_mps=top_speed,
            kick_power_attr=player.attributes.kick_power,
            kicking_params=match.kicking_params,
            power_multiplier=intent.power_multiplier,
            release=release,
        )
    raise ValueError(f"unknown kick intent kind {intent.kind!r}")


def apply_kick_intent(player: "Player", match: "Match", intent: KickIntent) -> None:
    """Carry out ``intent`` as its order's ``execute()`` used to, with the
    same kick bookkeeping for every kind (``Player._finish_kick``: end a first
    touch, set kicked_this_tick, count it, record last_kick_*, fire on_kick).
    Until 2026-11-16 shots and passes skipped that bookkeeping (they called the
    engine's kick_ball / pass_ball directly): a rules-AI shot never counted as a
    kick, and a first-touch shooter stayed in CONTROLLING_BALL (moving as if
    still carrying the ball) until the control timer ran out."""
    if intent.kind in ("direction", "aim"):
        # The player's own kick methods do the bookkeeping (Player._finish_kick).
        _carry_out(player, match, intent, release=True)
        if intent.kind == "direction" and intent.log_msg:
            match._log_debug(intent.log_msg)
        return
    from footballcoach.entities.player import PlayerState

    is_first_touch = player.state == PlayerState.CONTROLLING_BALL
    # The recorded power, in the same units as every other kick's
    # last_kick_power_fraction: the shot's own (run-compensated) power, or a
    # pass's equivalent power from its plan (computed before the ball moves).
    if intent.kind == "shot":
        recorded_power = float(intent.power_fraction)
    else:
        fields = kick_label_fields(player, match, intent)
        recorded_power = fields["power_fraction"] if fields is not None else 0.0
    _carry_out(player, match, intent, release=True)
    if intent.log_msg:
        if intent.kind == "shot":
            match._log_info(intent.log_msg)
        else:
            match._log_debug(intent.log_msg)
    player._finish_kick(match, is_first_touch, intent.kind, recorded_power, recorded_power, Vector3.zero())


def plan_kick_intent(player: "Player", match: "Match", intent: KickIntent) -> "Optional[KickPlan]":
    """The launch ``intent`` would produce now, before its random error, with
    nothing launched or recorded; None if the kick would not happen."""
    return _carry_out(player, match, intent, release=False)


def kick_label_fields(player: "Player", match: "Match", intent: KickIntent) -> Optional[dict]:
    """The kick heads' BC targets for ``intent``: {direction (unit 3-D launch
    direction, before the random error), power_fraction, spin}, or None if the
    kick would not happen.

    Power is in the neural kick head's own units: the fraction a
    ``kick_with_direction(direction, power, compensate_for_run=False)`` call
    (what the network's kick does) needs to launch at the planned speed, i.e.
    speed / (max kick speed x running multiplier along that direction). For a
    push-kick that is its own power fraction exactly, which is used as is."""
    plan = plan_kick_intent(player, match, intent)
    if plan is None:
        return None
    direction = plan.direction
    if intent.kind == "direction":
        power = float(intent.power_fraction)
    else:
        from footballcoach.engine.kicking import max_kick_speed_mps, running_power_multiplier
        from footballcoach.engine.movement import effective_top_speed

        top_speed = effective_top_speed(
            match.movement_params, player.attributes.top_speed, player.stamina,
            has_ball=True, ball_control_attr=player.attributes.ball_control,
        )
        run_mult = running_power_multiplier(
            match.kicking_params.running_power_coefficient, player.velocity, direction, top_speed,
        )
        power = plan.speed / max(max_kick_speed_mps(match.kicking_params, player.attributes.kick_power) * run_mult,
                                 GEOM_ZERO_EPS)
    spin = intent.spin if intent.kind in ("direction", "aim") else Vector3.zero()
    return {"direction": direction, "power_fraction": power, "spin": spin}
