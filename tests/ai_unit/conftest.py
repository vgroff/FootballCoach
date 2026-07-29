"""Shared fixtures for ai_unit tests.

Provides minimal Match objects for testing the observation encoder and
order-translation logic without touching torch or the training loop.
"""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3


def _attrs(val: float = 0.5) -> PlayerAttributes:
    return PlayerAttributes(
        top_speed=val, acceleration=val, stamina=val,
        kick_precision=val, kick_power=val, dribbling=val,
        ball_control=val, tackling=val,
    )


@pytest.fixture
def standard_pitch() -> Pitch:
    return Pitch.standard()


@pytest.fixture
def solo_match(standard_pitch):
    """Single player, no opponent, ball loose nearby."""
    p = Player.create("p1", Team.LEFT, _attrs(), position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(3, 0, 0))
    return Match(pitch=standard_pitch, players=[p], ball=ball,
                 rng_reduction=1.0, rng=random.Random(0))


@pytest.fixture
def duel_match(standard_pitch):
    """1v1: p1 (LEFT, has ball) vs p2 (RIGHT, no ball)."""
    p1 = Player.create("p1", Team.LEFT, _attrs(), position=Vector3(0, 0, 0))
    p2 = Player.create("p2", Team.RIGHT, _attrs(), position=Vector3(10, 0, 0))
    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.possessed_by = "p1"
    return Match(pitch=standard_pitch, players=[p1, p2], ball=ball,
                 rng_reduction=1.0, rng=random.Random(0))


@pytest.fixture
def gk_match(standard_pitch):
    """GK on LEFT team vs attacker on RIGHT team with the ball."""
    gk = Player.create("gk", Team.LEFT, _attrs(), is_goalkeeper=True,
                       position=Vector3(-standard_pitch.half_length + 1, 0, 0))
    attacker = Player.create("att", Team.RIGHT, _attrs(),
                             position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.possessed_by = "att"
    return Match(pitch=standard_pitch, players=[gk, attacker], ball=ball,
                 rng_reduction=1.0, rng=random.Random(0))
