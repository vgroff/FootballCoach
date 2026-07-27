"""Builds Match instances for the UI's two non-freeplay modes:

- Training mode: one player + ball on a full pitch, no opponent, goals in
  either net are counted and the ball/player reset on a goal.
- Balance scenarios: small, illustrative re-creations of the statistical
  scenarios from tests/balance/, played out live/visually one trial at a
  time (rather than the thousands-of-trials-headless style used by pytest).

These are UI conveniences for *watching* the mechanics that the balance
tests already validate statistically - they are not a replacement for the
pytest balance suite.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, PlayerAttributes, Team
from footballcoach.entities.player import Player
from footballcoach.generation import generate_attributes
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder, MoveOrder, TackleOrder


def make_training_match(rng_reduction: float = 0.3, tier: str = "premier_league") -> Match:
    """One player + ball, full pitch, both goals live, no opponent."""
    pitch = Pitch.standard()
    attrs = generate_attributes(tier=tier)
    player = Player.create("trainee", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(3, 0, 0))
    return Match(pitch=pitch, players=[player], ball=ball, rng_reduction=rng_reduction, rng=random.Random())


@dataclass
class ScenarioDefinition:
    key: str
    label: str
    description: str
    build: Callable[[float], Match]


def build_penalty_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_penalty_balance.py's penalty setup: one
    kicker on the penalty spot, aiming at a bottom corner, no goalkeeper."""
    pitch = Pitch.standard()
    penalty_spot = pitch.penalty_spot(left=False)
    attrs = generate_attributes(tier="premier_league")
    kicker = Player.create("kicker", Team.LEFT, attrs, position=penalty_spot)

    ball = Ball.at_rest(penalty_spot)
    ball.possessed_by = "kicker"

    match = Match(pitch=pitch, players=[kicker], ball=ball, rng_reduction=rng_reduction, rng=random.Random())

    corner_offset_y = pitch.goal_width_m / 2.0 - 0.475
    aim_point = pitch.right_goal_centre + Vector3(0, corner_offset_y, 0.475)
    kicker.current_order = KickOrder(aim_point=aim_point, power_fraction=0.65, spin=Vector3.zero())
    return match


def build_tackle_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_tackling_balance.py's headline matchup:
    tackling=0.8 defender vs dribbling=0.6 attacker in possession, touching."""
    pitch = Pitch.standard()

    defender_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.8,
    )
    attacker_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.6,
    )
    defender = Player.create("defender", Team.LEFT, defender_attrs, position=Vector3(0, 0, 0))
    attacker = Player.create("attacker", Team.RIGHT, attacker_attrs, position=Vector3(0.5, 0, 0))

    ball = Ball.at_rest(Vector3(0.5, 0, 0))
    ball.possessed_by = "attacker"

    match = Match(pitch=pitch, players=[defender, attacker], ball=ball, rng_reduction=rng_reduction, rng=random.Random())
    defender.current_order = TackleOrder(target_player_id="attacker")
    return match


def build_sprint_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_sprint_balance.py: a top-attribute and a
    bottom-attribute player racing 100m from the same start line."""
    pitch = Pitch.standard()
    start_x = -pitch.half_length + 1

    top_attrs = PlayerAttributes(
        top_speed=1.0, acceleration=1.0, stamina=1.0, kick_precision=0.5,
        kick_power=0.5, dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    bottom_attrs = PlayerAttributes(
        top_speed=0.0, acceleration=0.0, stamina=0.5, kick_precision=0.5,
        kick_power=0.5, dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    top_player = Player.create("fast", Team.LEFT, top_attrs, position=Vector3(start_x, -3, 0))
    bottom_player = Player.create("slow", Team.RIGHT, bottom_attrs, position=Vector3(start_x, 3, 0))

    ball = Ball.at_rest(Vector3(start_x, 20, 0))  # kept well away so it's not auto-picked-up
    match = Match(
        pitch=pitch, players=[top_player, bottom_player], ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(),
    )

    for player in (top_player, bottom_player):
        player.current_order = MoveOrder(target_position=Vector3(pitch.half_length - 1, player.position.y, 0), sprint=True)
    return match


SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        key="penalty",
        label="Penalty (corner aim, no keeper)",
        description="A Premier-League-tier kicker takes a penalty aiming at the bottom corner.",
        build=build_penalty_scenario,
    ),
    ScenarioDefinition(
        key="tackle",
        label="Tackle challenge (0.8 tackling vs 0.6 dribbling)",
        description="A tackling=0.8 defender challenges a dribbling=0.6 attacker in possession.",
        build=build_tackle_scenario,
    ),
    ScenarioDefinition(
        key="sprint",
        label="Sprint race (top vs bottom attributes)",
        description="A max-attribute player races a min-attribute player over 100m.",
        build=build_sprint_scenario,
    ),
]
