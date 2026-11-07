"""Match's id -> Player lookup, which replaced a linear scan and must behave like it."""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3


def _player(player_id: str, x: float = 0.0, team: Team = Team.LEFT) -> Player:
    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    return Player.create(player_id, team, attrs, position=Vector3(x, 0, 0))


def _match(players: list[Player]) -> Match:
    return Match(pitch=Pitch.standard(), players=players, ball=Ball.at_rest(Vector3(30, 0, 0)),
                 rng_reduction=1.0, rng=random.Random(0))


@pytest.fixture
def match() -> Match:
    return _match([_player("a", 1.0), _player("b", 2.0, Team.RIGHT), _player("c", 3.0)])


def test_returns_the_same_object_held_in_players(match):
    for player in match.players:
        assert match.player_by_id(player.player_id) is player
        assert match.player_by_id_or_none(player.player_id) is player


def test_unknown_id_raises_with_the_id_in_the_message(match):
    with pytest.raises(KeyError, match="no player with id ghost"):
        match.player_by_id("ghost")


def test_unknown_id_or_none_returns_none(match):
    assert match.player_by_id_or_none("ghost") is None


def test_first_player_wins_on_a_duplicate_id():
    first, second = _player("dup", 1.0), _player("dup", 2.0)
    assert _match([first, second]).player_by_id("dup") is first


def test_ball_carrier_resolves_through_the_lookup(match):
    match.ball.possessed_by = "b"
    assert match.ball_carrier() is match.players[1]


def test_stale_carrier_id_still_raises(match):
    match.ball.possessed_by = "ghost"
    with pytest.raises(KeyError):
        match.ball_carrier()


def test_lookup_table_is_not_part_of_equality_or_repr(match):
    field = Match.__dataclass_fields__["_players_by_id"]
    assert field.compare is False and field.repr is False and field.init is False
    assert "_players_by_id" not in repr(match)
