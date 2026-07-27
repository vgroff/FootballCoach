from __future__ import annotations

import pytest

from footballcoach.engine.offside import is_offside_position, last_defender_x
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from tests.conftest import make_player


def test_last_defender_x_left_team_defending():
    defenders = [
        make_player("d1", Team.LEFT, position=Vector3(-40, 0, 0)),
        make_player("d2", Team.LEFT, position=Vector3(-20, 0, 0)),
    ]
    # Attacking team is RIGHT, scoring at -x, so "deepest" LEFT defender (their
    # own goal at +x... wait: LEFT team's goal is at -x). Attacking RIGHT team
    # moves in -x direction; deepest LEFT defender is max(x).
    x = last_defender_x(defenders, defending_team=Team.LEFT, attacking_team=Team.RIGHT)
    assert x == -20


def test_attacker_behind_last_defender_is_onside():
    attacker = make_player("a1", Team.LEFT, position=Vector3(-25, 0, 0))
    ball_carrier = make_player("bc", Team.LEFT, position=Vector3(-30, 0, 0))
    defenders = [make_player("d1", Team.RIGHT, position=Vector3(-10, 0, 0))]
    players = [attacker, ball_carrier] + defenders
    assert not is_offside_position(attacker, ball_carrier, players, Team.LEFT, Team.RIGHT)


def test_attacker_beyond_last_defender_and_ball_is_offside():
    # LEFT attacks towards +x (right goal).
    attacker = make_player("a1", Team.LEFT, position=Vector3(5, 0, 0))
    ball_carrier = make_player("bc", Team.LEFT, position=Vector3(-30, 0, 0))
    defenders = [make_player("d1", Team.RIGHT, position=Vector3(0, 0, 0))]
    players = [attacker, ball_carrier] + defenders
    assert is_offside_position(attacker, ball_carrier, players, Team.LEFT, Team.RIGHT)


def test_attacker_level_with_defender_is_onside():
    attacker = make_player("a1", Team.LEFT, position=Vector3(0, 5, 0))
    ball_carrier = make_player("bc", Team.LEFT, position=Vector3(-30, 0, 0))
    defenders = [make_player("d1", Team.RIGHT, position=Vector3(0, 0, 0))]
    players = [attacker, ball_carrier] + defenders
    assert not is_offside_position(attacker, ball_carrier, players, Team.LEFT, Team.RIGHT)


def test_attacker_ahead_of_defenders_but_behind_ball_is_onside():
    attacker = make_player("a1", Team.LEFT, position=Vector3(2, 0, 0))
    ball_carrier = make_player("bc", Team.LEFT, position=Vector3(10, 0, 0))
    defenders = [make_player("d1", Team.RIGHT, position=Vector3(0, 0, 0))]
    players = [attacker, ball_carrier] + defenders
    assert not is_offside_position(attacker, ball_carrier, players, Team.LEFT, Team.RIGHT)
