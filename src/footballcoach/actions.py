"""High-level, one-shot player actions: simplified helpers over the
lower-level Move/Kick/Pass/Tackle/Save orders in `orders.py`.

These exist per the project's requirement for straightforward, named
functions a caller (UI, tests, or eventually an RL action space) can invoke
directly without hand-building order objects:

- `move_to(player, point)` - move in a straight line to a point.
- `shoot(player, pitch)` - shoot at goal, aimed at the dead centre of the
  *opponent's* goal (which goal that is depends on `player.team` - see
  `entities.player.Team` / `engine.offside`'s attacking-direction
  convention: LEFT attacks +x, RIGHT attacks -x).
- `pass_to(player, point)` - pass along the ground to a point.
- `tackle(player, target)` - run straight at an opposing player and tackle
  once in range.
- `save(goalkeeper)` - goalkeeper-only: continuously track and move to the
  incoming shot's predicted goal-line crossing point.

Each of these just assigns the corresponding order to `player.current_order`
- `Match.step()` (via `engine/match.py`) does the actual per-tick work.
Callers still need to drive the match loop themselves; these functions only
issue the instruction, matching the "order held until complete" model used
throughout the rest of the engine.
"""
from __future__ import annotations

from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3
from footballcoach.entities.player import Player as _Player  # forward ref alias
from footballcoach.orders import ChaseTackleOrder, GetPossessionOrder, KickOrder, MoveOrder, PassOrder, SaveOrder, StopOrder

DEFAULT_SHOOT_HEIGHT_M = 1.1
DEFAULT_SHOOT_POWER_FRACTION = 0.85


def move_to(player: Player, target_position: Vector3, sprint: bool = True) -> None:
    """Moves `player` in a straight line to `target_position`."""
    player.current_order = MoveOrder(target_position=target_position, sprint=sprint)


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
    goal at `aim_height_m`; pass an explicit `aim_point` to override (e.g.
    to aim at a specific corner of the goal frame).
    Only has an effect if `player` currently has the ball."""
    if aim_point is None:
        aim_point = opponent_goal_centre(pitch, player.team).with_z(aim_height_m)
    player.current_order = KickOrder(aim_point=aim_point, power_fraction=power_fraction, spin=Vector3.zero())


def pass_to(player: Player, target: "Player | Vector3", power_fraction: float | None = None) -> None:
    """Passes along the ground to a teammate or position. Pace is auto-computed
    from distance unless `power_fraction` is given. Only has an effect if
    `player` currently has the ball.

    If `target` is a ``Player``, the pass is led: the engine estimates where
    the teammate will be when the ball arrives (based on their current velocity)
    and aims at the predicted intercept position instead of their current one.
    If `target` is a ``Vector3``, a plain (non-led) pass to that spot is issued.
    """
    if isinstance(target, Player):
        player.current_order = PassOrder(
            target_position=target.position,
            power_fraction=power_fraction,
            target_player_id=target.player_id,
        )
    else:
        player.current_order = PassOrder(target_position=target, power_fraction=power_fraction)


def tackle(player: Player, target: Player | None = None) -> None:
    """Runs `player` towards the ball and wins possession.

    If the ball is loose the player sprints to it directly. If someone else
    has it, the player chases that carrier and attempts a tackle on contact.
    ``target`` is accepted for backward-compatibility but is ignored - the
    order always tracks whoever currently holds the ball each tick.
    """
    player.current_order = GetPossessionOrder()


def stop(player: Player) -> None:
    """Decelerates `player` to a standstill using their normal braking."""
    player.current_order = StopOrder()


def save(goalkeeper: Player) -> None:
    """Goalkeeper-only: continuously moves to the incoming shot's predicted
    goal-line crossing point (or a sensible default position if no shot is
    incoming). Does not auto-complete - reissue only if replacing it with a
    different order."""
    goalkeeper.current_order = SaveOrder()
