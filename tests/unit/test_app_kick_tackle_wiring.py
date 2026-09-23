"""`App._wire_player_icon_callbacks`: the on_kick/on_tackle closures that trigger the kick/tackle
swing animation (`ui/knowledge.md` "Kick/tackle swing animation").

Constructs a bare `App` instance via `object.__new__` rather than `App()` -- the real constructor
does a full pygame window/display init that these tests don't need; `_wire_player_icon_callbacks`
and its closures only ever touch `self.renderer` and the `match` they're given.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from footballcoach.entities import PlayerAttributes, Team
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3
from footballcoach.ui.app import App
from footballcoach.ui.camera import Camera
from footballcoach.ui.renderer import Renderer

pygame.init()
pygame.display.set_mode((1, 1))


def _attrs():
    return PlayerAttributes(top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
                             kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.6)


def _app_with_renderer():
    from footballcoach.entities import Pitch

    app = object.__new__(App)  # skip App.__init__'s full pygame/window setup
    cam = Camera.fit_to_pitch(Pitch.standard())
    app.renderer = Renderer(cam)
    return app


class _FakeBall:
    """A stand-in with just the `.velocity` attribute `_kick_cb` reads -- not a real `Ball`,
    since these tests only exercise the callback wiring, not physics."""
    def __init__(self, vx, vy):
        self.velocity = Vector3(vx, vy, 0.0)


class _FakeMatch:
    """A stand-in with just `.ball` and `.players` -- everything `_wire_player_icon_callbacks`
    and its closures touch."""
    def __init__(self, ball, players):
        self.ball = ball
        self.players = players


def _wired_player(vx=0.0, vy=0.0):
    """A real Player, wired via the real `_wire_player_icon_callbacks` (not a hand-built fake
    callback) against a fake match with the given ball velocity. Returns (app, player)."""
    app = _app_with_renderer()
    p = Player.create("p", Team.LEFT, _attrs(), position=Vector3(0.0, 0.0, 0.0))
    match = _FakeMatch(_FakeBall(vx, vy), [p])
    app._wire_player_icon_callbacks(match)
    return app, p


def test_kick_cb_uses_last_kick_direction_when_present():
    """The proper field (not a workaround): when `player.last_kick_direction` is set -- always
    true for a raw `KickOrder`, via `Player._finish_kick` -- the swing aims at it."""
    app, p = _wired_player()
    p.last_kick_direction = Vector3(1.0, 1.0, 0.0)
    p.on_kick(p)

    from footballcoach.ui import player_sprites as ps

    side, direction, _ = app.renderer._player_swing[p.player_id]
    expected = ps.clamp_swing_direction(ps.local_direction_from_world(1.0, 1.0, p.heading_rad))
    assert direction == pytest.approx(expected, abs=1e-6)
    assert p.action_icon == "⚽"


def test_kick_cb_falls_back_to_straight_ahead_when_last_kick_direction_is_missing():
    """`PassOrder`/`ShootOrder` currently fire `on_kick` without ever setting
    `player.last_kick_direction` -- a real engine bug (`agent_plans/kick_recording_bug.md`), not
    fixed here. Per the user: don't read something else (e.g. the ball's velocity) to paper over
    it -- keep reading the proper field, and when it's missing, fall back to "straight ahead"
    (the player's own heading) rather than skipping the animation entirely, since `on_kick` only
    ever fires after a real kick, so we know one happened even without knowing which way."""
    app, p = _wired_player()
    p.heading_rad = 0.7
    assert p.last_kick_direction is None  # the exact PassOrder/ShootOrder scenario
    p.on_kick(p)

    import math

    side, direction, _ = app.renderer._player_swing[p.player_id]
    expected_world = (math.cos(p.heading_rad), math.sin(p.heading_rad))
    from footballcoach.ui import player_sprites as ps

    expected = ps.clamp_swing_direction(ps.local_direction_from_world(*expected_world, p.heading_rad))
    assert direction == pytest.approx(expected, abs=1e-6)
    # straight ahead in local space is exactly (0, 1), before any clamping
    assert direction == pytest.approx((0.0, 1.0), abs=1e-6)
    assert p.action_icon == "⚽"


def test_tackle_cb_still_uses_last_tackle_direction():
    """Unlike the kick path, `Match._attempt_tackle_contact` sets `last_tackle_direction` directly
    (not via an Order.execute() bypass) -- confirmed by tracing a real `GetPossessionOrder` through
    a real `Match`. `_tackle_cb` correctly still reads it (no equivalent bug here to work around)."""
    app, p = _wired_player()
    p.last_tackle_direction = Vector3(0.0, 1.0, 0.0)
    p.on_tackle(p)

    assert p.player_id in app.renderer._player_swing
    assert p.action_icon == "🦵"


def test_tackle_cb_is_a_noop_without_a_tackle_direction():
    app, p = _wired_player()
    assert p.last_tackle_direction is None
    p.on_tackle(p)

    assert p.player_id not in app.renderer._player_swing
    assert p.action_icon == "🦵"  # the icon still shows even though the swing didn't trigger
