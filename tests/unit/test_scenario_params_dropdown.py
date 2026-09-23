"""SCENARIO_PARAMS screen dropdown usability: mouse-wheel scrolling (incl. the
high-precision/trackpad case), the draggable scrollbar thumb, and type-ahead
search -- see `ui/knowledge.md` "Scenario-params dropdown: wheel/scrollbar/
type-ahead" for the full rationale (this was written in response to "the
dropdown for checkpoint selection is a pain, I can't move wheel on it").

Constructs a bare `App` via `object.__new__` (as in `test_app_kick_tackle_wiring.py`)
since these tests only touch `_scenario_params_ui` and the handful of methods
under test -- no real pygame window is needed.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from footballcoach.ui import scenarios
from footballcoach.ui.app import App, ScenarioParamsUIState, Screen
from footballcoach.ui.camera import Camera
from footballcoach.ui.renderer import DropdownScrollbar, Renderer

pygame.init()
pygame.display.set_mode((1, 1))
pygame.font.init()


def _grouped_param(n_groups: int = 20, n_leaves: int = 3, name: str = "trainee_checkpoint"):
    groups = tuple(
        (f"checkpoints/run{i}", tuple(f"checkpoints/run{i}/ckpt{j}.pt" for j in range(n_leaves)))
        for i in range(n_groups)
    )
    return scenarios.ScenarioGroupedChoiceParam(name, "Trainee checkpoint", groups, groups[0][1][0])


def _app_with_ui(param, open_choice_param: str | None = None, open_choice_folder: str | None = None):
    """A bare App with a SCENARIO_PARAMS screen open on a single fake param,
    so `_type_ahead_jump`/`_type_ahead_confirm`'s `ui.definition.params`
    lookup resolves to *param* -- not whatever real checkpoints happen to
    exist on disk (a mistake caught during manual verification: reusing the
    real scenario definition silently searched the wrong, real param list)."""
    app = object.__new__(App)
    app.screen = Screen.SCENARIO_PARAMS
    definition = scenarios.ScenarioDefinition(
        key="fake", label="Fake", description="", build=lambda **kw: None, params=[param],
    )
    ui = ScenarioParamsUIState()
    ui.reset_for(definition)
    ui.open_choice_param = open_choice_param
    ui.open_choice_folder = open_choice_folder
    app._scenario_params_ui = ui
    return app


# ---------------------------------------------------------------------------
# scenarios.dropdown_items_for / leaf_label -- the single source of truth
# both Renderer and App's type-ahead read from
# ---------------------------------------------------------------------------

def test_dropdown_items_for_flat_choice_param():
    param = scenarios.ScenarioChoiceParam("p", "P", ("a", "b", "c"), "a")
    assert scenarios.dropdown_items_for(param, None) == [("a", "a"), ("b", "b"), ("c", "c")]


def test_dropdown_items_for_grouped_param_top_level_is_the_group_list():
    param = _grouped_param(n_groups=3, n_leaves=2)
    items = scenarios.dropdown_items_for(param, None)
    assert items == [("checkpoints/run0", "checkpoints/run0"),
                      ("checkpoints/run1", "checkpoints/run1"),
                      ("checkpoints/run2", "checkpoints/run2")]


def test_dropdown_items_for_grouped_param_inside_a_folder_is_its_leaves():
    param = _grouped_param(n_groups=3, n_leaves=2)
    items = scenarios.dropdown_items_for(param, "checkpoints/run1")
    assert items == [("checkpoints/run1/ckpt0.pt", "run1/ckpt0.pt"),
                      ("checkpoints/run1/ckpt1.pt", "run1/ckpt1.pt")]


# ---------------------------------------------------------------------------
# Mouse wheel -- the reported "can't move wheel on it" bug
# ---------------------------------------------------------------------------

def _wheel_event(y: int, precise_y: float | None):
    kwargs = dict(y=y, x=0, flipped=False)
    if precise_y is not None:
        kwargs["precise_y"] = precise_y
        kwargs["precise_x"] = 0.0
    return pygame.event.Event(pygame.MOUSEWHEEL, **kwargs)


def test_wheel_is_a_noop_when_no_dropdown_is_open():
    app = _app_with_ui(_grouped_param(), open_choice_param=None)
    app._scroll_open_dropdown(_wheel_event(1, 1.0))
    assert app._scenario_params_ui.dropdown_scroll == 0


def test_classic_whole_notch_wheel_still_scrolls_by_one_per_notch():
    """A normal mouse reports precise_y == float(y); this must keep working
    exactly as the old plain-`event.y` code did."""
    app = _app_with_ui(_grouped_param(), open_choice_param="trainee_checkpoint")
    ui = app._scenario_params_ui
    ui.dropdown_scroll = 5
    app._scroll_open_dropdown(_wheel_event(1, 1.0))
    assert ui.dropdown_scroll == 4
    app._scroll_open_dropdown(_wheel_event(-1, -1.0))
    assert ui.dropdown_scroll == 5


def test_wheel_falls_back_to_plain_y_when_precise_y_is_absent():
    """Older pygame builds / synthetic events without `precise_y` at all
    must still scroll, using `event.y` as before."""
    app = _app_with_ui(_grouped_param(), open_choice_param="trainee_checkpoint")
    ui = app._scenario_params_ui
    ui.dropdown_scroll = 5
    app._scroll_open_dropdown(_wheel_event(2, None))
    assert ui.dropdown_scroll == 3


def test_high_precision_fractional_wheel_deltas_accumulate_instead_of_vanishing():
    """The actual reported bug: a hi-res mouse/trackpad reporting e.g. 0.3
    per physical notch. Truncating each event's delta alone (as plain
    integer `event.y` does after SDL rounds it) means many such notches
    report y == 0 and the wheel looks completely dead. Accumulating the
    precise deltas across events must eventually commit a whole step."""
    app = _app_with_ui(_grouped_param(), open_choice_param="trainee_checkpoint")
    ui = app._scenario_params_ui
    ui.dropdown_scroll = 10
    for _ in range(3):
        app._scroll_open_dropdown(_wheel_event(0, 0.3))
    # 3 * 0.3 = 0.9 -- not yet a whole step
    assert ui.dropdown_scroll == 10
    app._scroll_open_dropdown(_wheel_event(0, 0.3))
    # 4 * 0.3 = 1.2 -- one whole step has now accumulated
    assert ui.dropdown_scroll == 9


def test_wheel_scroll_never_goes_negative():
    # Positive y/precise_y is "scroll up" (toward earlier items), which
    # decrements dropdown_scroll -- must clamp at 0, not go negative.
    app = _app_with_ui(_grouped_param(), open_choice_param="trainee_checkpoint")
    ui = app._scenario_params_ui
    ui.dropdown_scroll = 0
    app._scroll_open_dropdown(_wheel_event(1, 1.0))
    assert ui.dropdown_scroll == 0


# ---------------------------------------------------------------------------
# Draggable scrollbar
# ---------------------------------------------------------------------------

def test_scrollbar_drag_is_a_noop_without_geometry():
    app = _app_with_ui(_grouped_param(), open_choice_param="trainee_checkpoint")
    app._scenario_params_ui.scrollbar_geom = None
    app._scrollbar_drag_to(500)
    assert app._scenario_params_ui.dropdown_scroll == 0


def test_scrollbar_drag_maps_top_and_bottom_of_track_to_scroll_extremes():
    app = _app_with_ui(_grouped_param(), open_choice_param="trainee_checkpoint")
    ui = app._scenario_params_ui
    track = pygame.Rect(900, 100, 22, 248)
    ui.scrollbar_geom = DropdownScrollbar(track_rect=track, thumb_h=60, total=30, visible=10)

    app._scrollbar_drag_to(track.top)
    assert ui.dropdown_scroll == 0
    app._scrollbar_drag_to(track.bottom)
    assert ui.dropdown_scroll == 30 - 10
    app._scrollbar_drag_to(track.top - 500)  # dragged above the track entirely
    assert ui.dropdown_scroll == 0
    app._scrollbar_drag_to(track.bottom + 500)  # dragged below the track entirely
    assert ui.dropdown_scroll == 30 - 10


# ---------------------------------------------------------------------------
# Type-ahead search
# ---------------------------------------------------------------------------

def test_type_ahead_jump_scrolls_a_prefix_match_into_view():
    param = _grouped_param(n_groups=30)
    app = _app_with_ui(param, open_choice_param=param.name)
    ui = app._scenario_params_ui
    ui.type_ahead_buffer = "checkpoints/run27"
    app._type_ahead_jump()
    assert ui.dropdown_scroll == 27  # group index 27 == "run27"


def test_type_ahead_jump_falls_back_to_a_contains_match():
    """Typing a run number that isn't at the start of the label (e.g. just
    digits) should still find it."""
    param = _grouped_param(n_groups=30)
    app = _app_with_ui(param, open_choice_param=param.name)
    ui = app._scenario_params_ui
    ui.type_ahead_buffer = "run27"  # not a prefix of "checkpoints/run27"
    app._type_ahead_jump()
    assert ui.dropdown_scroll == 27


def test_type_ahead_jump_is_a_noop_with_no_match():
    param = _grouped_param(n_groups=5)
    app = _app_with_ui(param, open_choice_param=param.name)
    ui = app._scenario_params_ui
    ui.dropdown_scroll = 2
    ui.type_ahead_buffer = "nonexistent"
    app._type_ahead_jump()
    assert ui.dropdown_scroll == 2  # unchanged


def test_type_ahead_jump_is_a_noop_when_buffer_is_empty_or_dropdown_closed():
    param = _grouped_param(n_groups=5)
    app = _app_with_ui(param, open_choice_param=None)
    ui = app._scenario_params_ui
    ui.type_ahead_buffer = "checkpoints/run3"
    app._type_ahead_jump()  # no dropdown open
    assert ui.dropdown_scroll == 0

    ui.open_choice_param = param.name
    ui.type_ahead_buffer = ""
    app._type_ahead_jump()  # nothing typed
    assert ui.dropdown_scroll == 0


def test_type_ahead_confirm_drills_into_a_matched_folder():
    param = _grouped_param(n_groups=30)
    app = _app_with_ui(param, open_choice_param=param.name, open_choice_folder=None)
    ui = app._scenario_params_ui
    ui.type_ahead_buffer = "checkpoints/run12"
    ui.dropdown_scroll = 27  # simulate having jumped there already

    app._type_ahead_confirm()

    assert ui.open_choice_folder == "checkpoints/run12"
    assert ui.open_choice_param == param.name  # still open, now one level deeper
    assert ui.type_ahead_buffer == ""  # cleared for the new (leaf) list
    assert ui.dropdown_scroll == 0  # close_dropdown_list() reset it


def test_type_ahead_confirm_selects_a_matched_leaf_and_closes_the_dropdown():
    param = _grouped_param(n_groups=3, n_leaves=5)
    app = _app_with_ui(param, open_choice_param=param.name, open_choice_folder="checkpoints/run1")
    ui = app._scenario_params_ui
    ui.type_ahead_buffer = "run1/ckpt3"

    app._type_ahead_confirm()

    assert ui.values[param.name] == "checkpoints/run1/ckpt3.pt"
    assert ui.open_choice_param is None
    assert ui.open_choice_folder is None


def test_type_ahead_confirm_is_a_noop_with_no_match():
    param = _grouped_param(n_groups=3)
    app = _app_with_ui(param, open_choice_param=param.name, open_choice_folder=None)
    ui = app._scenario_params_ui
    ui.type_ahead_buffer = "totally-unmatched"

    app._type_ahead_confirm()

    assert ui.open_choice_folder is None
    assert ui.open_choice_param == param.name  # dropdown left open, nothing selected


# ---------------------------------------------------------------------------
# Renderer: DropdownScrollbar is only returned when the list is actually
# scrollable, and never crashes with a type-ahead query set.
# ---------------------------------------------------------------------------

def _renderer():
    from footballcoach.entities.pitch import Pitch
    return Renderer(Camera.fit_to_pitch(Pitch.standard()))


def test_scrollbar_geom_is_none_for_a_short_list():
    renderer = _renderer()
    surface = pygame.Surface((1400, 900))
    param = _grouped_param(n_groups=4)  # well under MAX_VISIBLE_ITEMS
    _rects, _scroll, geom = renderer.draw_scenario_params(
        surface, [param], {param.name: param.default}, open_choice_param=param.name,
    )
    assert geom is None


def test_scrollbar_geom_is_present_and_matches_list_length_when_scrollable():
    renderer = _renderer()
    surface = pygame.Surface((1400, 900))
    param = _grouped_param(n_groups=25)
    _rects, _scroll, geom = renderer.draw_scenario_params(
        surface, [param], {param.name: param.default}, open_choice_param=param.name,
    )
    assert geom is not None
    assert geom.total == 25
    assert geom.visible == 10
    assert geom.thumb_h < geom.track_rect.height  # thumb is smaller than the full track


def test_draw_scenario_params_accepts_a_type_ahead_query_without_crashing_or_losing_options():
    renderer = _renderer()
    surface = pygame.Surface((1400, 900))
    param = _grouped_param(n_groups=25)
    button_rects, _scroll, _geom = renderer.draw_scenario_params(
        surface, [param], {param.name: param.default}, open_choice_param=param.name,
        type_ahead_query="checkpoints/run3",
    )
    assert any("__folder__" in k for k in button_rects)
