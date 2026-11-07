"""Ball defaults, body-shape config, and the Order type alias."""
from __future__ import annotations

import dataclasses
import inspect

from footballcoach import orders
from footballcoach.config import load_attributes_config, load_physics_config
from footballcoach.engine.possession import ControlTimeParams
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.ball import Ball
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3


def test_default_ball_is_at_the_origin_at_rest():
    ball = Ball()
    assert ball.position == Vector3.zero()
    assert ball.velocity == Vector3.zero()
    assert ball.spin == Vector3.zero()


def test_default_balls_do_not_share_vectors():
    first, second = Ball(), Ball()
    assert first.position is not second.position
    assert first.velocity is not second.velocity
    assert first.spin is not second.spin


def test_ball_fields_are_still_positional_in_the_same_order():
    assert [f.name for f in dataclasses.fields(Ball)][:5] == ["position", "velocity", "spin", "radius_m", "mass_kg"]


def test_body_shape_lives_in_attributes_config_with_its_tuned_values():
    assert load_attributes_config()["body"] == {
        "radius_m": 0.3, "height_m": 1.8, "knee_height_m": 0.49, "waist_height_m": 0.95,
    }
    assert "player" not in load_physics_config()


def test_player_create_uses_the_body_config():
    player = Player.create("p", Team.LEFT, PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    assert player.radius_m == 0.3 and player.height_m == 1.8


def test_control_time_params_use_the_body_config():
    params = ControlTimeParams.from_config()
    assert (params.knee_height_m, params.waist_height_m, params.player_height_m) == (0.49, 0.95, 1.8)


def test_order_alias_covers_every_order_class():
    declared = {
        cls for name, cls in inspect.getmembers(orders, inspect.isclass)
        if name.endswith("Order") and cls.__module__ == orders.__name__
    }
    assert declared == set(orders.Order.__args__)
