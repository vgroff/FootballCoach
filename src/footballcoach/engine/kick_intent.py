"""Kick intents: what an order wants to kick this tick, applied by the engine.

Orders used to carry out their kick inside ``execute()`` (launching the ball,
ending a first touch, firing ``on_kick``), so the only way for a BC label to
learn what the rules AI would kick was to run the order for real and undo the
damage afterwards. Now a kicking order records a ``KickIntent`` on
``player.pending_kick`` and ``Match._process_orders`` applies it immediately
after that player's ``execute()`` returns -- the same point in the tick, and the
same per-player order, as the old in-``execute()`` kick, so play is unchanged.
A counterfactual ``execute()`` (ai/ppo/bc.py) can then read the intent without
the ball being touched.

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

if TYPE_CHECKING:
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


def apply_kick_intent(player: "Player", match: "Match", intent: KickIntent) -> None:
    """Carry out ``intent`` exactly as its order's ``execute()`` used to."""
    if intent.kind == "direction":
        player.kick_with_direction(match, intent.direction_3d, intent.power_fraction, intent.spin)
        if intent.log_msg:
            match._log_debug(intent.log_msg)
        return
    if intent.kind == "aim":
        player.kick_direct(match, intent.aim_point, intent.power_fraction, intent.spin, intent.compensate_for_run)
        return
    from footballcoach.engine.movement import effective_top_speed

    top_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina,
        has_ball=True, ball_control_attr=player.attributes.ball_control,
    )
    if intent.kind == "shot":
        from footballcoach.engine.kicking import kick_ball

        kick_ball(
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
        )
    elif intent.kind == "pass":
        from footballcoach.engine.kicking import pass_ball

        pass_ball(
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
        )
    else:
        raise ValueError(f"unknown kick intent kind {intent.kind!r}")
    if intent.log_msg:
        if intent.kind == "shot":
            match._log_info(intent.log_msg)
        else:
            match._log_debug(intent.log_msg)
    if player.on_kick is not None:
        player.on_kick(player)
