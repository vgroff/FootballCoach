"""Simplified offside rule.

Per the design spec, this implements the "spirit" of offside rather than
the full law: if the ball is played and comes near any attacking player who
is positioned beyond the last defender (and beyond the ball itself), that
player is immediately flagged offside - regardless of whether the ball was
actually intended for them. "Last defender" here is the deepest opponent
(including the goalkeeper), which is a deliberate simplification of the
real second-last-defender rule.
"""
from __future__ import annotations

from footballcoach.entities.player import Player, Team


def _attacking_direction_sign(attacking_team: Team) -> int:
    """Returns +1 if the attacking team scores at +x (right goal), else -1."""
    return 1 if attacking_team == Team.LEFT else -1


def last_defender_x(players: list[Player], defending_team: Team, attacking_team: Team) -> float:
    """Returns the x-coordinate of the deepest defender (closest to their own
    goal line), i.e. the offside line. Includes the goalkeeper."""
    defenders = [p for p in players if p.team == defending_team]
    if not defenders:
        raise ValueError("no defending players supplied - cannot compute offside line")

    sign = _attacking_direction_sign(attacking_team)
    # The "deepest" defender is the one furthest in the -sign direction i.e.
    # closest to their own goal. For an attacking team scoring at +x, the
    # defending team's own goal is at -x, so deepest defender = min(x).
    if sign > 0:
        return min(p.position.x for p in defenders)
    return max(p.position.x for p in defenders)


def is_offside_position(
    attacker: Player,
    ball_carrier: Player,
    players: list[Player],
    attacking_team: Team,
    defending_team: Team,
) -> bool:
    """Returns True if `attacker` is in an offside position: beyond the last
    defender AND beyond the ball carrier, in the attacking direction.

    Per the simplified spec, a player level with or behind the ball or the
    last defender is onside.
    """
    if attacker.team != attacking_team:
        return False

    sign = _attacking_direction_sign(attacking_team)
    defender_line_x = last_defender_x(players, defending_team, attacking_team)
    ball_carrier_x = ball_carrier.position.x

    if sign > 0:
        beyond_defenders = attacker.position.x > defender_line_x
        beyond_ball_carrier = attacker.position.x > ball_carrier_x
    else:
        beyond_defenders = attacker.position.x < defender_line_x
        beyond_ball_carrier = attacker.position.x < ball_carrier_x

    return beyond_defenders and beyond_ball_carrier


def check_offside_on_pass(
    receiver: Player,
    ball_carrier: Player,
    players: list[Player],
    attacking_team: Team,
    defending_team: Team,
) -> bool:
    """Called when the ball is played (passed/kicked) towards `receiver`.
    Returns True if this should be flagged offside.

    Per the simplified spec: any teammate in an offside position who the
    ball comes near is flagged, whether or not the pass was intended for
    them - so this function is also suitable for checking *any* attacking
    teammate near the ball's path, not just the intended receiver.
    """
    return is_offside_position(receiver, ball_carrier, players, attacking_team, defending_team)
