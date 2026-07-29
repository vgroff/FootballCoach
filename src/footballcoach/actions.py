"""High-level, one-shot player actions: thin shims over Player action methods.

DEPRECATED as a call site: prefer calling player methods directly
(e.g. ``player.get_possession()``, ``player.kick(...)``).  These module-level
functions remain to avoid updating every existing call site in the rules-based
AI, UI, and tests simultaneously.  New code should not add more callers here.

Each function is exactly one line: it delegates to the matching Player method.
"""
from __future__ import annotations

from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3

DEFAULT_SHOOT_HEIGHT_M = 1.1
DEFAULT_SHOOT_POWER_FRACTION = 0.85


def move_to(player: Player, target_position: Vector3, sprint: bool = True) -> None:
    """Moves `player` in a straight line to `target_position`."""
    player.move_to(target_position, sprint=sprint)


def opponent_goal_centre(pitch: Pitch, team: Team) -> Vector3:
    """The centre of the goal `team` is attacking (i.e. trying to score in),
    per the attacking-direction convention used throughout the engine
    (Team.LEFT attacks +x / the right goal; Team.RIGHT attacks -x / the
    left goal - see engine/offside.py)."""
    return pitch.right_goal_centre if team == Team.LEFT else pitch.left_goal_centre


def shoot(
    player: Player,
    pitch: Pitch,
    power_fraction: float = DEFAULT_SHOOT_POWER_FRACTION,
    aim_height_m: float = DEFAULT_SHOOT_HEIGHT_M,
    aim_point: "Vector3 | None" = None,
) -> None:
    """Shoots at goal. By default aims at the dead centre of the opponent's
    goal at `aim_height_m`; pass an explicit `aim_point` to override.
    Only has an effect if `player` currently has the ball."""
    if aim_point is None:
        aim_point = opponent_goal_centre(pitch, player.team).with_z(aim_height_m)
    player.kick(aim_point=aim_point, power_fraction=power_fraction, spin=Vector3.zero())


def pass_to(player: Player, target: "Player | Vector3", power_fraction: float | None = None) -> None:
    """Passes along the ground to a teammate or position.

    If `target` is a ``Player``, the pass is led (engine estimates future
    position).  If `target` is a ``Vector3``, a plain non-led pass is issued.
    """
    if isinstance(target, Player):
        player.pass_ball(
            target_position=target.position,
            target_player_id=target.player_id,
            power_fraction=power_fraction,
        )
    else:
        player.pass_ball(target_position=target, power_fraction=power_fraction)


def tackle(player: Player, target: Player | None = None) -> None:
    """Runs `player` towards the ball and wins possession.

    ``target`` is accepted for backward-compatibility but is ignored.
    """
    player.get_possession()


def stop(player: Player) -> None:
    """Decelerates `player` to a standstill using their normal braking."""
    player.stop()


def save(goalkeeper: Player) -> None:
    """Goalkeeper-only: continuously moves to the incoming shot's intercept point."""
    goalkeeper.save_goal()


def mark(player: Player, target: Player) -> None:
    """Mark a specific opposition player. Does not auto-complete."""
    player.mark_player(target_player_id=target.player_id)
