"""`App._handle_keydown`'s `Z` / `Shift+Z` zoom hotkeys.

User feedback: "the zoom key zooms differently to the zoom buttons at the
top (faster?)". Investigated (see `ui/knowledge.md` "Ball-follow zoom"): the
step SIZE was always identical to the on-screen [-]/[+] buttons -- the real
asymmetry was that `Z` could only zoom IN and wrapped from max zoom straight
back to 1.0x in one press, with no keyboard zoom-out at all. Fixed by
routing `Shift+Z` to zoom out, mirroring the sim-speed `[`/`]` pair.

Constructs a bare `App` via `object.__new__` (as in
`test_app_kick_tackle_wiring.py`) with just the attributes `_handle_keydown`
actually touches for a `K_z` press: `camera`/`_ZOOM_LEVELS` (for
`_cycle_zoom`) and the handful of screen/match/help-overlay guards it must
pass through first.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from footballcoach.entities.pitch import Pitch
from footballcoach.ui.app import App, Screen
from footballcoach.ui.camera import Camera

pygame.init()
pygame.display.set_mode((1, 1))


def _app_in_match():
    app = object.__new__(App)
    app.camera = Camera.fit_to_pitch(Pitch.standard())
    app._ZOOM_LEVELS = tuple(1.0 + i * 0.5 for i in range(9))  # 1.0..5.0
    app.screen = Screen.MATCH
    app.show_help = False
    app.match = object()  # truthy stand-in; _handle_keydown only checks "is not None" past the zoom branch
    app.input_controller = object()
    return app


@pytest.fixture(autouse=True)
def _reset_mods():
    pygame.key.set_mods(0)
    yield
    pygame.key.set_mods(0)


def test_z_without_shift_zooms_in_by_one_step():
    app = _app_in_match()
    pygame.key.set_mods(0)
    App._handle_keydown(app, pygame.K_z, "z")
    assert app.camera.zoom_factor == pytest.approx(1.5)


def test_shift_z_zooms_out_by_one_step():
    app = _app_in_match()
    app.camera.set_zoom_level(2.0)
    pygame.key.set_mods(pygame.KMOD_SHIFT)
    App._handle_keydown(app, pygame.K_z, "Z")
    assert app.camera.zoom_factor == pytest.approx(1.5)


def test_shift_z_step_size_matches_the_on_screen_minus_button():
    """The on-screen [-] button calls _cycle_zoom(-1) directly (see
    App._handle_events); Shift+Z must produce the exact same result."""
    app_key = _app_in_match()
    app_key.camera.set_zoom_level(3.0)
    pygame.key.set_mods(pygame.KMOD_SHIFT)
    App._handle_keydown(app_key, pygame.K_z, "Z")

    app_button = _app_in_match()
    app_button.camera.set_zoom_level(3.0)
    App._cycle_zoom(app_button, -1)

    assert app_key.camera.zoom_factor == pytest.approx(app_button.camera.zoom_factor)


def test_shift_z_steps_down_instead_of_wrapping_from_the_bottom():
    """At 1.0x (the bottom level), Shift+Z wraps to the top -- same
    wraparound _cycle_zoom already does for plain Z at the top, just in the
    other direction. Not a regression: this mirrors existing behaviour."""
    app = _app_in_match()
    app.camera.set_zoom_level(1.0)
    pygame.key.set_mods(pygame.KMOD_SHIFT)
    App._handle_keydown(app, pygame.K_z, "Z")
    assert app.camera.zoom_factor == pytest.approx(5.0)


def test_plain_z_still_wraps_from_max_back_to_one():
    """Unchanged pre-existing behaviour, pinned so a future change is
    deliberate: Z alone still wraps forward past the top."""
    app = _app_in_match()
    app.camera.set_zoom_level(5.0)
    pygame.key.set_mods(0)
    App._handle_keydown(app, pygame.K_z, "z")
    assert app.camera.zoom_factor == pytest.approx(1.0)
