"""Player sprites (ui/player_sprites.py) and how the renderer / app use them: the gait
maths (`advance_gait_phase`, `pick_pose`), the generated sprite sets, drawing a rotated
sprite at the player's heading, and -- the reason the animation clock exists -- that stride
and ball spin follow SIMULATION time, whatever the sim-speed setting or frame rate.

Pixel tests run on plain offscreen `pygame.Surface`s; the App test uses SDL's dummy video
driver (no window)."""
from __future__ import annotations

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from footballcoach.entities import PlayerAttributes, Team
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3
from footballcoach.ui import player_sprites as ps
from footballcoach.ui import style
from footballcoach.ui.sim_clock import SimTimeDelta

TWO_PI = 2.0 * math.pi


@pytest.fixture(scope="module")
def params() -> ps.PlayerSpriteParams:
    return ps.PlayerSpriteParams.from_config()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_params_load_from_graphics_config(params):
    assert params.size_scale > 0 and params.max_stride_hz > 0 and params.top_speed_for_stride_mps > 0
    assert params.extension_speed_exponent > 0
    for colour in (params.skin_color, params.hair_color, params.shorts_color, params.shoe_color):
        assert len(colour) == 3 and all(0 <= c <= 255 for c in colour)
    assert 0.0 < params.hair_coverage_deg < 360.0


# ---------------------------------------------------------------------------
# advance_gait_phase: stride phase, in SIM seconds
# ---------------------------------------------------------------------------

def test_a_standing_player_does_not_advance(params):
    assert ps.advance_gait_phase(1.234, 0.0, 0.5, params) == pytest.approx(1.234)


def test_phase_advance_is_proportional_to_the_sim_time_elapsed(params):
    speed = params.top_speed_for_stride_mps
    one = ps.advance_gait_phase(0.0, speed, 0.02, params)
    two = ps.advance_gait_phase(0.0, speed, 0.04, params)
    assert one == pytest.approx(TWO_PI * params.max_stride_hz * 0.02)
    assert two == pytest.approx(2 * one)
    assert ps.advance_gait_phase(0.0, speed, 0.0, params) == 0.0
    assert ps.advance_gait_phase(0.7, speed, -1.0, params) == pytest.approx(0.7)   # never runs backwards


def test_stride_frequency_scales_linearly_with_speed_and_saturates_at_top_speed(params):
    top, dt = params.top_speed_for_stride_mps, 0.02
    full = ps.advance_gait_phase(0.0, top, dt, params)
    assert ps.advance_gait_phase(0.0, top / 2, dt, params) == pytest.approx(full / 2)
    assert ps.advance_gait_phase(0.0, top * 3, dt, params) == pytest.approx(full)   # capped at max_stride_hz


def test_phase_wraps_into_one_cycle(params):
    p = 0.0
    for _ in range(200):
        p = ps.advance_gait_phase(p, params.top_speed_for_stride_mps, 0.03, params)
        assert 0.0 <= p < TWO_PI


def test_a_huge_time_step_cannot_alias_the_stride(params):
    """At very high sim speeds one frame spans several strides; the phase step is capped below half
    a cycle (where the alternating legs would alias into stalling / running backwards)."""
    step = ps.advance_gait_phase(0.0, params.top_speed_for_stride_mps, 10.0, params)
    assert step == pytest.approx(TWO_PI * ps._MAX_CYCLES_PER_UPDATE)
    assert ps._MAX_CYCLES_PER_UPDATE < 0.5
    # an update at the cap still moves forward each time (no stall), unlike an unbounded one could
    assert ps.advance_gait_phase(step, params.top_speed_for_stride_mps, 10.0, params) != pytest.approx(step)


# ---------------------------------------------------------------------------
# pick_pose
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", [0.0, 0.5, math.pi / 2, 2.0, math.pi, 4.0, 3 * math.pi / 2, 6.0])
def test_a_stationary_player_always_shows_the_standing_pose(params, phase):
    side, level = ps.pick_pose(phase, 0.0, params)
    assert level is None and side in (1, -1)


def test_full_speed_at_mid_stride_shows_the_fullest_stride_on_the_matching_side(params):
    top = params.top_speed_for_stride_mps
    assert ps.pick_pose(math.pi / 2, top, params) == (1, 1.0)
    assert ps.pick_pose(3 * math.pi / 2, top, params) == (-1, 1.0)


def test_the_leading_side_flips_every_half_cycle(params):
    top = params.top_speed_for_stride_mps
    assert ps.pick_pose(0.3, top, params)[0] == 1
    assert ps.pick_pose(math.pi - 0.3, top, params)[0] == 1
    assert ps.pick_pose(math.pi + 0.3, top, params)[0] == -1
    assert ps.pick_pose(TWO_PI - 0.3, top, params)[0] == -1
    assert ps.pick_pose(TWO_PI + 0.3, top, params) == ps.pick_pose(0.3, top, params)   # phase is periodic


def test_stride_extension_grows_with_speed_and_only_uses_the_defined_levels(params):
    top = params.top_speed_for_stride_mps
    levels = []
    for i in range(0, 41):
        _, level = ps.pick_pose(math.pi / 2, top * i / 40, params)
        assert level is None or level in ps.STRIDE_LEVELS
        levels.append(0.0 if level is None else level)
    assert levels == sorted(levels)
    assert levels[0] == 0.0 and levels[-1] == 1.0
    assert ps.pick_pose(math.pi / 2, top * 10, params) == (1, 1.0)     # faster than top speed saturates


def test_a_jog_shows_a_partial_stride(params):
    _, level = ps.pick_pose(math.pi / 2, params.top_speed_for_stride_mps * 0.5, params)
    assert level is not None and 0.25 <= level <= 0.75


def test_stride_is_at_its_smallest_at_the_hand_over_between_sides(params):
    top = params.top_speed_for_stride_mps
    assert ps.pick_pose(0.0, top, params)[1] is None
    assert ps.pick_pose(math.pi, top, params)[1] is None


# ---------------------------------------------------------------------------
# The sprite images
# ---------------------------------------------------------------------------

SHIRT = (13, 77, 201)


def _opaque(surface, threshold=127):
    w, h = surface.get_size()
    return [[surface.get_at((x, y))[3] > threshold for x in range(w)] for y in range(h)]


def _bbox(mask):
    ys = [y for y, row in enumerate(mask) if any(row)]
    xs = [x for x in range(len(mask[0])) if any(row[x] for row in mask)]
    return xs[0], ys[0], xs[-1], ys[-1]


def _mismatch(a, b):
    total = sum(len(r) for r in a)
    return sum(pa != pb for ra, rb in zip(a, b) for pa, pb in zip(ra, rb)) / total


def test_a_sprite_set_has_nine_transparent_base_size_poses(params):
    s = ps.PlayerSpriteSet(params, SHIRT)
    poses = [s.standing] + [s.get(side, lvl) for side in (1, -1) for lvl in ps.STRIDE_LEVELS]
    assert len(poses) == 9
    for pose in poses:
        assert pose.get_size() == (ps._BASE_SIZE, ps._BASE_SIZE)
        assert pose.get_flags() & pygame.SRCALPHA
        assert pose.get_at((0, 0))[3] == 0 and pose.get_at((ps._BASE_SIZE - 1, ps._BASE_SIZE - 1))[3] == 0
        assert 0.05 < sum(map(sum, _opaque(pose))) / ps._BASE_SIZE ** 2 < 0.6
    assert s.get(1, None) is s.standing and s.get(-1, None) is s.standing


def test_sprite_sets_are_cached_per_shirt_colour(params):
    a = ps.get_sprite_set(params, (11, 22, 33))
    assert ps.get_sprite_set(params, (11, 22, 33)) is a
    assert ps.get_sprite_set(params, (33, 22, 11)) is not a


def test_the_sprite_wears_the_shirt_colour_and_only_the_shirt_changes(params):
    a = ps.PlayerSpriteSet(params, SHIRT).standing
    b = ps.PlayerSpriteSet(params, (200, 30, 30)).standing
    assert tuple(a.get_at((22, 40)))[:3] == SHIRT                        # torso
    assert tuple(b.get_at((22, 40)))[:3] == (200, 30, 30)
    assert _mismatch(_opaque(a), _opaque(b)) == 0.0                      # same silhouette
    assert tuple(a.get_at((40, 48)))[:3] == tuple(b.get_at((40, 48)))[:3] == tuple(params.skin_color)   # face untouched


def test_the_head_has_hair_at_the_back_and_a_face_at_the_front(params):
    """The sprite faces down (+y): the hair wedge is on the far (-y) side of the head."""
    s = ps.PlayerSpriteSet(params, SHIRT).standing
    assert tuple(s.get_at((40, 26)))[:3] == tuple(params.hair_color)
    assert tuple(s.get_at((40, 48)))[:3] == tuple(params.skin_color)


def test_hair_and_skin_colours_come_from_the_params(params):
    import dataclasses

    custom = dataclasses.replace(params, skin_color=(10, 200, 10), hair_color=(200, 10, 200))
    s = ps._render_pose(custom, SHIRT, 0.0, 0.0, 1)
    assert tuple(s.get_at((40, 26)))[:3] == (200, 10, 200)
    assert tuple(s.get_at((40, 48)))[:3] == (10, 200, 10)


def test_hair_coverage_widens_the_hair_wedge(params):
    import dataclasses

    def hair_pixels(deg):
        s = ps._render_pose(dataclasses.replace(params, hair_coverage_deg=deg), SHIRT, 0.0, 0.0, 1)
        return sum(
            tuple(s.get_at((x, y)))[:3] == tuple(params.hair_color)
            for x in range(ps._BASE_SIZE) for y in range(ps._BASE_SIZE)
        )

    assert hair_pixels(100) < hair_pixels(220) < hair_pixels(300)


def test_the_standing_pose_is_left_right_symmetric(params):
    s = ps.PlayerSpriteSet(params, SHIRT).standing
    mask = _opaque(s)
    assert _mismatch(mask, [row[::-1] for row in mask]) < 0.01


@pytest.mark.parametrize("level", ps.STRIDE_LEVELS)
def test_the_two_leading_sides_are_mirror_images(params, level):
    s = ps.PlayerSpriteSet(params, SHIRT)
    a, b = _opaque(s.get(1, level)), _opaque(s.get(-1, level))
    assert _mismatch(a, [row[::-1] for row in b]) < 0.02
    assert _mismatch(a, b) > 0.0     # ...and they are not the same pose


def test_a_longer_stride_reaches_further_along_the_facing_axis(params):
    s = ps.PlayerSpriteSet(params, SHIRT)
    extent = {lvl: (lambda bb: bb[3] - bb[1])(_bbox(_opaque(s.get(1, lvl)))) for lvl in ps.STRIDE_LEVELS}
    standing = (lambda bb: bb[3] - bb[1])(_bbox(_opaque(s.standing)))
    lengths = [standing] + [extent[lvl] for lvl in ps.STRIDE_LEVELS]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)


# ---------------------------------------------------------------------------
# Drawing: rotation to the heading, colours, translucency, animation update
# ---------------------------------------------------------------------------

def _attrs():
    return PlayerAttributes(top_speed=0.78, acceleration=0.78, stamina=0.78, kick_precision=0.78,
                            kick_power=0.78, dribbling=0.78, ball_control=0.78, tackling=0.78)


def _player(pid="p", team=Team.LEFT, keeper=False, heading=0.0, speed=0.0):
    p = Player.create(pid, team, _attrs(), position=Vector3(0.0, 0.0, 0.0), is_goalkeeper=keeper)
    p.heading_rad = heading
    p.velocity = Vector3(speed, 0.0, 0.0)
    return p


def _scene(zoom=3.0):
    from footballcoach.entities import Pitch
    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    cam.set_zoom_level(zoom)
    cam.follow(0.0, 0.0)
    surface.fill(style.PITCH_GREEN)
    # Ask the renderer itself, rather than reimplementing its radius formula here -- the two
    # drifted out of sync once that formula stopped being a flat `* zoom_scale` (see
    # `_player_radius_px`'s size-floor-decay docstring).
    radius = renderer._player_radius_px(_player())
    return cam, surface, renderer, cam.world_to_screen(0.0, 0.0), radius


def _near(px, colour, tol=30):
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(px[:3], colour))


def _centroid_of(surface, centre, radius, colour):
    xs = ys = n = 0
    for x in range(centre[0] - 2 * radius, centre[0] + 2 * radius + 1):
        for y in range(centre[1] - 2 * radius, centre[1] + 2 * radius + 1):
            if _near(surface.get_at((x, y)), colour):
                xs, ys, n = xs + x, ys + y, n + 1
    assert n > 10, "no pixels of that colour drawn"
    return xs / n - centre[0], ys / n - centre[1]


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 0.7])
def test_the_sprite_is_rotated_so_the_hair_is_behind_the_players_heading(params, heading):
    cam, surface, renderer, pos, radius = _scene()
    renderer._draw_player_sprite(surface, _player(heading=heading), pos, radius, style.TEAM_LEFT_COLOUR, False)
    hx, hy = _centroid_of(surface, pos, radius, params.hair_color)
    facing = (math.cos(-heading), math.sin(-heading))          # the screen-space facing vector draw_player uses
    assert hx * facing[0] + hy * facing[1] < -0.1 * radius     # the hair sits on the back side
    assert abs(hx * facing[1] - hy * facing[0]) < 0.3 * radius  # ...and roughly on the heading axis


def test_the_sprite_is_scaled_to_size_scale_times_the_player_radius(params):
    """The whole 80px sprite square is scaled to `radius * size_scale` px; at heading 0 (facing +x) its
    length (local y) runs along screen x and its width (local x) along screen y."""
    cam, surface, renderer, pos, radius = _scene()
    renderer._draw_player_sprite(surface, _player(heading=0.0), pos, radius, style.TEAM_LEFT_COLOUR, False)
    grass = tuple(style.PITCH_GREEN)
    pts = [(x, y) for x in range(pos[0] - 3 * radius, pos[0] + 3 * radius + 1)
           for y in range(pos[1] - 3 * radius, pos[1] + 3 * radius + 1) if tuple(surface.get_at((x, y)))[:3] != grass]
    drawn_len = max(x for x, _ in pts) - min(x for x, _ in pts) + 1
    drawn_wid = max(y for _, y in pts) - min(y for _, y in pts) + 1

    x0, y0, x1, y1 = _bbox(_opaque(ps.get_sprite_set(params, style.TEAM_LEFT_COLOUR).standing))
    scale = radius * params.size_scale / ps._BASE_SIZE
    assert drawn_len == pytest.approx((y1 - y0 + 1) * scale, rel=0.12)
    assert drawn_wid == pytest.approx((x1 - x0 + 1) * scale, rel=0.12)
    # and it scales with the radius (zoom): a bigger player draws a proportionally bigger sprite
    cam2, surface2, renderer2, pos2, radius2 = _scene(zoom=5.0)
    assert radius2 > radius


def test_team_and_goalkeeper_colours_are_drawn():
    cam, surface, renderer, pos, radius = _scene()
    for colour in (style.TEAM_LEFT_COLOUR, style.TEAM_RIGHT_COLOUR, style.GOALKEEPER_COLOUR):
        surface.fill(style.PITCH_GREEN)
        renderer._draw_player_sprite(surface, _player(), pos, radius, colour, False)
        _centroid_of(surface, pos, radius, colour)          # asserts the shirt colour is present

    surface.fill(style.PITCH_GREEN)
    renderer.draw_player(surface, _player(keeper=True))          # public API: keeper in orange
    _centroid_of(surface, pos, radius, style.GOALKEEPER_COLOUR)
    surface.fill(style.PITCH_GREEN)
    renderer.draw_player(surface, _player(team=Team.RIGHT))
    _centroid_of(surface, pos, radius, style.TEAM_RIGHT_COLOUR)


def test_inactive_players_are_drawn_translucent():
    cam, surface, renderer, pos, radius = _scene()
    renderer._draw_player_sprite(surface, _player(), pos, radius, style.TEAM_LEFT_COLOUR, False)
    solid = [tuple(surface.get_at((pos[0] + dx, pos[1] + dy)))[:3] for dx in range(-radius, radius, 3) for dy in range(-radius, radius, 3)]
    surface.fill(style.PITCH_GREEN)
    renderer._draw_player_sprite(surface, _player(), pos, radius, style.TEAM_LEFT_COLOUR, True)
    faded = [tuple(surface.get_at((pos[0] + dx, pos[1] + dy)))[:3] for dx in range(-radius, radius, 3) for dy in range(-radius, radius, 3)]
    grass = tuple(style.PITCH_GREEN)
    differing = [(s, f) for s, f in zip(solid, faded) if s != grass and s != f]
    assert len(differing) > 10
    for s, f in differing:                      # each faded pixel lies between the grass and the solid one
        assert all(min(g, sc) - 1 <= fc <= max(g, sc) + 1 for g, sc, fc in zip(grass, s, f))


def test_a_striding_player_draws_a_longer_sprite_than_a_standing_one():
    """Drawn through `_draw_player_sprite` alone, so speed lines and rings don't add pixels."""
    cam, surface, renderer, pos, radius = _scene()

    def covered(speed, phase):
        surface.fill(style.PITCH_GREEN)
        p = _player(speed=speed, heading=0.0)
        renderer._player_gait_phase[p.player_id] = phase
        renderer._draw_player_sprite(surface, p, pos, radius, style.TEAM_LEFT_COLOUR, False)
        grass = tuple(style.PITCH_GREEN)
        xs = [x for x in range(pos[0] - 3 * radius, pos[0] + 3 * radius + 1)
              for y in range(pos[1] - 3 * radius, pos[1] + 3 * radius + 1) if tuple(surface.get_at((x, y)))[:3] != grass]
        return max(xs) - min(xs)

    top = renderer._sprite_params.top_speed_for_stride_mps
    assert covered(top, math.pi / 2) > covered(0.0, math.pi / 2) + radius       # heading 0 = along screen x


def test_update_player_animations_advances_each_players_phase_by_sim_time():
    cam, surface, renderer, pos, radius = _scene()
    params = renderer._sprite_params
    runner, stander = _player("runner", speed=params.top_speed_for_stride_mps), _player("stander")
    renderer.update_player_animations([runner, stander], 0.05)
    assert renderer._player_gait_phase["runner"] == pytest.approx(TWO_PI * params.max_stride_hz * 0.05)
    assert renderer._player_gait_phase["stander"] == 0.0
    renderer.update_player_animations([runner, stander], 0.05)
    assert renderer._player_gait_phase["runner"] == pytest.approx(TWO_PI * params.max_stride_hz * 0.10)


def test_animations_are_a_no_op_with_sprites_disabled():
    cam, surface, renderer, pos, radius = _scene()
    renderer._sprites_enabled = False
    renderer.update_player_animations([_player(speed=8.0)], 0.05)
    assert renderer._player_gait_phase == {}


# ---------------------------------------------------------------------------
# The animation clock (ui/sim_clock.py): sim time between frames
# ---------------------------------------------------------------------------

def test_sim_time_delta_reports_the_match_time_between_calls():
    clock, match = SimTimeDelta(), object()
    assert clock.delta(match, 5.0) == 0.0                # first call just syncs
    assert clock.delta(match, 5.0) == 0.0                # no sim time passed (e.g. paused)
    assert clock.delta(match, 5.25) == pytest.approx(0.25)
    assert clock.delta(match, 5.75) == pytest.approx(0.5)


def test_sim_time_delta_resyncs_for_a_new_match_or_a_clock_that_ran_backwards():
    clock, first, second = SimTimeDelta(), object(), object()
    clock.delta(first, 100.0)
    assert clock.delta(second, 0.5) == 0.0               # a new match/trial starts at t=0: no bogus jump
    assert clock.delta(second, 0.75) == pytest.approx(0.25)
    assert clock.delta(second, 0.1) == 0.0               # time going backwards
    assert clock.delta(second, 0.2) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# App wiring: strides and spin follow sim time, not the frame rate or the sim-speed setting
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    from footballcoach.ui import scenarios
    from footballcoach.ui.app import App

    a = App()
    a._start_match(scenarios.make_training_match(), "Training mode", is_training_mode=True)
    a.match.paused = False
    return a


def _rotation_angle(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


@pytest.mark.parametrize("sim_speed,fps", [(1.0, 60), (8.0, 60), (0.5, 30), (2.0, 144)])
def test_strides_and_spin_advance_by_the_sim_time_that_passed(app, sim_speed, fps):
    app._sim_speed, app._target_fps = sim_speed, fps
    r = app.renderer
    params = r._sprite_params
    runner = app.match.players[0]
    runner.velocity = Vector3(params.top_speed_for_stride_mps, 0.0, 0.0)
    app.match.ball.spin = Vector3(0.0, 0.0, 10.0)                       # 10 rad per sim second about z
    app.match.ball.velocity = Vector3(0.0, 0.0, 0.0)

    app._draw()                                                        # first frame syncs the clock
    r._player_gait_phase[runner.player_id] = 0.0
    r._ball_orientation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    app.match.time_s += 0.04                                           # 0.04 sim seconds pass...
    app._draw()                                                        # ...over one rendered frame

    assert r._player_gait_phase[runner.player_id] == pytest.approx(TWO_PI * params.max_stride_hz * 0.04)
    assert _rotation_angle(r._ball_orientation) == pytest.approx(10.0 * 0.04, abs=1e-6)


def test_no_sim_time_passing_means_no_animation(app):
    r = app.renderer
    runner = app.match.players[0]
    runner.velocity = Vector3(r._sprite_params.top_speed_for_stride_mps, 0.0, 0.0)
    app._draw()
    before = r._player_gait_phase[runner.player_id]
    app._draw()                                                        # match clock unchanged (e.g. sim not stepped)
    assert r._player_gait_phase[runner.player_id] == before


def test_a_paused_match_does_not_animate_and_does_not_bank_time(app):
    r = app.renderer
    runner = app.match.players[0]
    runner.velocity = Vector3(r._sprite_params.top_speed_for_stride_mps, 0.0, 0.0)
    app._draw()
    before = r._player_gait_phase[runner.player_id]
    app.match.paused = True
    app.match.time_s += 1.0                                            # (clock nudged while paused)
    app._draw()
    assert r._player_gait_phase[runner.player_id] == before
    app.match.paused = False
    app._draw()                                                        # first frame after unpausing: one paused
    # second of clock is still counted once (it is real sim time) but capped, never a runaway spin:
    assert (r._player_gait_phase[runner.player_id] - before) % TWO_PI <= TWO_PI * ps._MAX_CYCLES_PER_UPDATE + 1e-9


# ---------------------------------------------------------------------------
# Lighting: per-part shading of the sprites, and ground shadows (one scene light)
# ---------------------------------------------------------------------------
# The tests above measure the flat sprite art (colours, silhouettes), so the renderer's
# player shading / shadows are switched off for them; the tests below switch them on.

@pytest.fixture(autouse=True)
def _flat_players(monkeypatch):
    import copy

    from footballcoach.ui import renderer as renderer_mod

    cfg = copy.deepcopy(renderer_mod.load_graphics_config())
    cfg.setdefault("player_shading", {})["enabled"] = False
    cfg.setdefault("player_shadow", {})["enabled"] = False
    monkeypatch.setattr(renderer_mod, "load_graphics_config", lambda: cfg)


def _np():
    import numpy as np

    return np


def _luma_of(px):
    return 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]


def test_sprite_normals_are_unit_vectors_that_lean_outward_at_a_parts_edges(params):
    np = _np()
    sprite = ps.PlayerSpriteSet(params, SHIRT).standing
    alpha = ps._alpha_of(sprite)
    n = ps.sprite_normals(alpha)
    assert n.shape == (80, 80, 3)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-3)
    assert (n[..., 2] > 0).all()                               # always facing up out of the pitch
    xs = np.where(alpha[40] > 127)[0]
    assert n[40, xs.min() + 1, 0] < -0.15 and n[40, xs.max() - 1, 0] > 0.15    # left edge leans left, right leans right
    ys = np.where(alpha[:, 40] > 127)[0]
    assert n[ys.min() + 1, 40, 1] < -0.1 and n[ys.max() - 1, 40, 1] > 0.1      # top edge leans up, bottom down
    assert n[40, 40, 2] > 0.9                                                   # the middle of the head/torso is flat


def test_shade_factors_are_lit_on_the_light_side_and_flat_at_zero_strength(params):
    np = _np()
    s = ps.PlayerSpriteSet(params, SHIRT)
    alpha = ps._alpha_of(s.standing)
    n = s.normals(1, None)
    body = alpha > 127
    xs = np.arange(80)[None, :].repeat(80, axis=0)
    left, right = body & (xs < 33), body & (xs > 47)

    assert (ps.shade_factors(n, alpha, ps.light_from_angles(0, 30), 0.0) == 1.0).all()
    f = ps.shade_factors(n, alpha, ps.light_from_angles(0, 30), 0.9)          # light from the sprite's right
    assert 0.0 < f.min() and f.max() <= 1.0 + 1e-6 and f[body].max() == pytest.approx(1.0, abs=0.02)
    assert f[right].mean() > f[left].mean() + 0.1
    g = ps.shade_factors(n, alpha, ps.light_from_angles(180, 30), 0.9)        # ...and from its left
    assert g[left].mean() > g[right].mean() + 0.1
    o = ps.shade_factors(n, alpha, ps.light_from_angles(0, 90), 0.9)          # straight overhead: even, edges darker
    assert abs(o[left].mean() - o[right].mean()) < 0.03
    assert o[40, 40] > o[40, np.where(body[40])[0].min() + 1] + 0.05
    az, el = math.radians(37), math.radians(52)
    assert ps.light_from_angles(37, 52) == pytest.approx((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))


def test_a_lit_sprite_keeps_its_silhouette_never_brightens_and_is_cached(params):
    np = _np()
    s = ps.PlayerSpriteSet(params, (17, 29, 41))
    base = s.get(1, 0.5)
    lit = s.shaded(1, 0.5, 45, 40, 0.9)
    assert s.shaded(1, 0.5, 45, 40, 0.9) is lit
    assert s.shaded(1, 0.5, 225, 40, 0.9) is not lit
    assert lit.get_size() == base.get_size() and lit.get_flags() & pygame.SRCALPHA
    assert np.array_equal(ps._alpha_of(base), ps._alpha_of(lit))
    rgb0 = pygame.surfarray.array3d(base).astype(int)
    rgb1 = pygame.surfarray.array3d(lit).astype(int)
    assert (rgb1 <= rgb0 + 1).all() and (rgb1 < rgb0 - 20).any()
    flat = s.shaded(1, 0.5, 45, 40, 0.0)
    assert np.abs(pygame.surfarray.array3d(flat).astype(int) - rgb0).max() <= 1


def _lit_scene(world_x, world_y, zoom=3.0, **attrs):
    cam, surface, renderer, _, radius = _scene(zoom)
    for k, v in attrs.items():
        setattr(renderer, k, v)
    cam.follow(world_x, world_y)
    return cam, surface, renderer, radius


def _half_ratio(shaded, flat, pos, radius, toward, sign):
    num = den = 0.0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i * i + j * j <= (radius - 1) ** 2 and sign * (i * toward[0] + j * toward[1]) > 0:
                px = flat.get_at((pos[0] + i, pos[1] + j))
                if tuple(px)[:3] == tuple(style.PITCH_GREEN):
                    continue                                   # bare grass: not part of the figure
                num += _luma_of(shaded.get_at((pos[0] + i, pos[1] + j)))
                den += _luma_of(px)
    return num / den


@pytest.mark.parametrize("world", [(-25.0, 0.0), (30.0, -18.0), (-10.0, 22.0)])
@pytest.mark.parametrize("heading", [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 0.7])
def test_a_player_is_lit_from_the_pitch_centre_whichever_way_they_face(world, heading):
    """The light direction is rotated into the sprite's own frame, so the side of the drawn figure
    facing the pitch's middle is the bright one at every heading. (A low light, 8 m up, gives the
    shading enough contrast to measure by half-figure brightness.)"""
    frames = []
    for strength in (0.9, 0.0):
        cam, surface, renderer, radius = _lit_scene(*world, _player_shading_strength=strength, _light_height_m=8.0)
        p = _player(heading=heading)
        p.position = Vector3(world[0], world[1], 0.0)
        pos = cam.world_to_screen(*world)
        renderer._draw_player_sprite(surface, p, pos, radius, style.TEAM_LEFT_COLOUR, False)
        frames.append((surface, pos, cam))
    (shaded, pos, cam), (flat, _, _) = frames
    cx, cy = cam.world_to_screen_f(0.0, 0.0)
    bx, by = cam.world_to_screen_f(*world)
    n = math.hypot(cx - bx, cy - by)
    toward = ((cx - bx) / n, (cy - by) / n)
    near = _half_ratio(shaded, flat, pos, 2 * radius, toward, +1)
    far = _half_ratio(shaded, flat, pos, 2 * radius, toward, -1)
    assert near - far > 0.05


def _sprite_set(renderer, colour):
    return ps.get_sprite_set(renderer._sprite_params, colour)


def test_player_shading_can_be_turned_off_and_uses_the_light_cache():
    cam, surface, renderer, radius = _lit_scene(-25.0, 0.0, _player_shading_strength=0.0)
    p = _player()
    p.position = Vector3(-25.0, 0.0, 0.0)
    colour = (23, 45, 67)
    pos = cam.world_to_screen(-25.0, 0.0)
    renderer._draw_player_sprite(surface, p, pos, radius, colour, False)
    assert not _sprite_set(renderer, colour)._shaded                  # strength 0: the unlit sprite is used
    renderer._player_shading_strength = 0.9
    renderer._draw_player_sprite(surface, p, pos, radius, colour, False)
    assert len(_sprite_set(renderer, colour)._shaded) == 1
    renderer._draw_player_sprite(surface, p, pos, radius, colour, False)
    assert len(_sprite_set(renderer, colour)._shaded) == 1            # same light + pose: cache hit


def _darkened(surface, green=None, min_drop=10):
    green = tuple(green or style.PITCH_GREEN)
    w, h = surface.get_size()
    return [(x, y) for x in range(w) for y in range(h) if sum(green) - sum(surface.get_at((x, y))[:3]) >= min_drop]


@pytest.mark.parametrize("world", [(-25.0, 12.0), (35.0, -20.0), (-8.0, -30.0)])
def test_a_player_shadow_points_away_from_the_centre_with_the_point_light_length(world):
    cam, surface, renderer, radius = _lit_scene(*world, zoom=2.0, _player_shadow_enabled=True)
    p = _player()
    p.position = Vector3(world[0], world[1], 0.0)
    renderer.draw_player_shadows(surface, [p])
    pos = cam.world_to_screen(*world)
    pts = _darkened(surface)
    assert pts
    mx = sum(x for x, _ in pts) / len(pts) - pos[0]
    my = sum(y for _, y in pts) / len(pts) - pos[1]

    cx, cy = cam.world_to_screen_f(0.0, 0.0)
    bx, by = cam.world_to_screen_f(*world)
    ppm = cam.pixels_per_metre
    dist_m = math.hypot(bx - cx, by - cy) / ppm
    away = ((bx - cx) / (dist_m * ppm), (by - cy) / (dist_m * ppm))
    length_px = dist_m * p.height_m / (renderer._light_height_m - p.height_m) * ppm
    # a capsule's centroid is the middle of its axis: half the length out along the radial direction
    assert math.hypot(mx - away[0] * length_px / 2, my - away[1] * length_px / 2) < 0.15 * length_px + 2.5
    assert mx * away[0] + my * away[1] > 0


def test_a_shadow_grows_with_distance_from_the_centre_and_shrinks_when_the_light_is_higher():
    def extent(world, height=None, contact=None):
        cam, surface, renderer, radius = _lit_scene(*world, zoom=2.0, _player_shadow_enabled=True)
        if height is not None:
            renderer._light_height_m = height
        if contact is not None:
            renderer._player_shadow_contact_m = contact
        p = _player()
        p.position = Vector3(world[0], world[1], 0.0)
        renderer.draw_player_shadows(surface, [p])
        pos = cam.world_to_screen(*world)
        return max(math.hypot(x - pos[0], y - pos[1]) for x, y in _darkened(surface))

    assert extent((45.0, 0.0)) > extent((20.0, 0.0)) > extent((1.0, 0.0))     # ...but never vanishes at the middle
    assert extent((45.0, 0.0), height=20.0) > extent((45.0, 0.0), height=40.0) > extent((45.0, 0.0), height=80.0)
    # a player standing at the middle (where the point light casts no shadow at all) still gets the
    # thin contact shadow: longer than the bare round footprint
    assert extent((0.0, 0.0)) > extent((0.0, 0.0), contact=0.0) + 1


def test_player_shadows_can_be_turned_off():
    cam, surface, renderer, radius = _lit_scene(-25.0, 12.0, _player_shadow_enabled=False)
    p = _player()
    p.position = Vector3(-25.0, 12.0, 0.0)
    renderer.draw_player_shadows(surface, [p])
    assert not _darkened(surface, min_drop=1)


def test_the_shadow_sprite_is_cached_soft_edged_and_bounded():
    cam, surface, renderer, radius = _lit_scene(0.0, 0.0, _player_shadow_enabled=True)
    a = renderer._player_shadow_sprite(20, 45, 6)
    assert renderer._player_shadow_sprite(20, 45, 6) is a
    assert a.get_at((a.get_width() // 2, a.get_height() // 2))[3] == pytest.approx(renderer._player_shadow_alpha, abs=2)
    assert a.get_at((0, 0))[3] == 0
    assert any(0 < a.get_at((x, a.get_height() // 2))[3] < renderer._player_shadow_alpha for x in range(a.get_width()))
    for i in range(600):
        renderer._player_shadow_sprite(5 + i % 40, 5 * (i % 72), 3)
    assert len(renderer._shadow_cache) <= 512


def test_all_shadows_are_drawn_under_every_player_sprite_and_the_ball():
    """One shadow pass before anything else: a player drawn EARLIER must not be darkened by the
    shadow of a player drawn later (per-player shadows would land on top of the earlier sprite),
    and the ball stays clean inside a shadow. Players are drawn in the app's order: after
    draw_pitch_and_ball."""
    from footballcoach.entities import Pitch
    from footballcoach.entities.ball import Ball

    def scene(players, ball_xy):
        pitch = Pitch.standard()
        cam, surface, renderer, _, radius = _scene(3.0)
        renderer._player_shadow_enabled = True
        renderer._ball_shadow_enabled = False
        cam.follow(-21.0, 0.0)
        ball = Ball()
        ball.position = Vector3(ball_xy[0], ball_xy[1], 0.11)
        ball.velocity = Vector3(0.0, 0.0, 0.0)
        renderer.draw_pitch_and_ball(surface, pitch, ball, players=players)
        for pl in players:
            renderer.draw_player(surface, pl)
        return cam, surface

    near_centre = _player("a")                       # closer to the middle: its shadow reaches back over `b`
    near_centre.position = Vector3(-20.0, 0.0, 0.0)
    further = _player("b", team=Team.RIGHT)
    further.position = Vector3(-21.0, 0.0, 0.0)
    ball_xy = (-20.9, 0.9)                           # inside a's shadow too

    cam, both = scene([further, near_centre], ball_xy)          # b drawn FIRST, then a
    _, alone = scene([further], ball_xy)
    b_px = cam.world_to_screen(-21.0, 0.0)
    ball_px = cam.world_to_screen(*ball_xy)
    for dx in range(-3, 4):                                       # b's body centre: same with or without a's shadow
        assert both.get_at((b_px[0] + dx, b_px[1])) == alone.get_at((b_px[0] + dx, b_px[1]))
    assert both.get_at(ball_px) == alone.get_at(ball_px)
    # ...and the shadows really are drawn, on the bare grass
    pitch_only = scene([], ball_xy)[1]
    assert sum(
        1 for x in range(0, both.get_width(), 2) for y in range(0, both.get_height(), 2)
        if sum(pitch_only.get_at((x, y))[:3]) - sum(both.get_at((x, y))[:3]) >= 8
    ) > 200


def test_the_scene_light_is_one_setting_shared_by_ball_and_players():
    from footballcoach.entities.ball import Ball

    cam, surface, renderer, radius = _lit_scene(0.0, 0.0)
    assert renderer._light_height_m == 40.0
    az, el, dist = renderer._point_light_at(40.0, 0.0, 0.9)
    assert dist == pytest.approx(40.0, abs=0.05) and el == pytest.approx(math.degrees(math.atan2(40.0 - 0.9, 40.0)), abs=0.1)
    ball = Ball()
    ball.position = Vector3(40.0, 0.0, 0.11)
    high = renderer._ball_light_angles(ball)[1]
    renderer._light_height_m = 15.0
    assert renderer._ball_light_angles(ball)[1] < high                # one attribute moves ball and player lighting alike
    assert renderer._point_light_at(40.0, 0.0, 0.9)[1] < el


# ---------------------------------------------------------------------------
# Rings: translucent, in one layer UNDER the players and the ball
# ---------------------------------------------------------------------------

def test_there_is_no_possession_ring_and_draw_player_draws_no_rings_itself():
    """The white ring on the ball carrier was removed. All the other rings (selected, first-touch, tackled,
    low stamina, and the ball's state ring) are drawn by the separate translucent ring layer, not by
    `draw_player`."""
    import inspect

    from footballcoach.ui.renderer import Renderer

    params = inspect.signature(Renderer.draw_player).parameters
    assert "has_ball" not in params and "selected" not in params
    assert not hasattr(style, "POSSESSION_OUTLINE")
    assert hasattr(style, "INACTIVE_OUTLINE") and hasattr(style, "CONTROL_DELAY_OUTLINE") and hasattr(style, "SELECTED_OUTLINE")
    _, _, renderer, _, _ = _scene()
    assert not hasattr(renderer, "_possession_outline_thickness")
    assert 0 < renderer._ring_alpha < 255                       # translucent by default


def _ring_pixels(surface, pos, radius, colour, tol):
    """Angles (degrees, skipping the sector below the player where its label and stat bars are) at which the
    circle of `radius` around `pos` is within `tol` of `colour` on every channel."""
    hits = []
    for deg in range(0, 360, 3):
        if 45 <= deg <= 135:                                   # screen angle: +y is down
            continue
        x = pos[0] + radius * math.cos(math.radians(deg))
        y = pos[1] + radius * math.sin(math.radians(deg))
        px = surface.get_at((round(x), round(y)))[:3]
        if all(abs(px[i] - colour[i]) <= tol for i in range(3)):
            hits.append(deg)
    return hits


@pytest.mark.parametrize("state_name", ["ACTIVE", "INACTIVE_TACKLED"])
def test_drawing_a_player_never_adds_a_white_possession_ring(state_name):
    from footballcoach.entities.player import PlayerState

    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player(heading=0.0)
    p.state = getattr(PlayerState, state_name)
    renderer.draw_player(surface, p)
    renderer.draw_player_rings(surface, [p], selected_id=None)
    assert not _ring_pixels(surface, pos, radius + 2, (255, 255, 255), 2)


def _blend(bg, colour, alpha):
    return tuple(round(bg[i] * (1 - alpha / 255.0) + colour[i] * alpha / 255.0) for i in range(3))


def _ring_scene(state=None, selected=False, zoom=5.0, **attrs):
    from footballcoach.entities.player import PlayerState

    cam, surface, renderer, pos, radius = _scene(zoom)
    for k, v in attrs.items():
        setattr(renderer, k, v)
    p = _player(heading=0.0)
    if state is not None:
        p.state = getattr(PlayerState, state)
    renderer.draw_player_rings(surface, [p], selected_id=p.player_id if selected else None)
    return cam, surface, renderer, pos, radius, p


@pytest.mark.parametrize("state, colour_name, offset", [
    ("CONTROLLING_BALL", "CONTROL_DELAY_OUTLINE", 4),
    ("INACTIVE_TACKLED", "INACTIVE_OUTLINE", 4),
])
def test_the_state_rings_are_translucent_and_the_tackled_red_ring_is_kept(state, colour_name, offset):
    cam, surface, renderer, pos, radius, p = _ring_scene(state=state)
    colour = getattr(style, colour_name)
    expected = _blend(style.PITCH_GREEN, colour, renderer._ring_alpha)
    # the ring is 2px wide with outer radius radius+offset: sample its middle
    hits = _ring_pixels(surface, pos, radius + offset - 1, expected, 14)
    assert len(hits) > 60                                                  # (a ring all the way round, sector below skipped)
    assert not _ring_pixels(surface, pos, radius + offset - 1, colour, 14)  # ...and NOT the opaque colour


def test_the_selected_ring_is_translucent_and_a_normal_player_has_none():
    cam, surface, renderer, pos, radius, p = _ring_scene(selected=True)
    expected = _blend(style.PITCH_GREEN, style.SELECTED_OUTLINE, renderer._ring_alpha)
    assert len(_ring_pixels(surface, pos, radius + 6, expected, 14)) > 60
    cam, surface, renderer, pos, radius, p = _ring_scene(selected=False)      # ACTIVE, not selected
    assert not _ring_pixels(surface, pos, radius + 6, expected, 14)
    assert all(tuple(surface.get_at((pos[0] + i, pos[1] + j)))[:3] == tuple(style.PITCH_GREEN) for i in range(-3, 4) for j in range(-3, 4))


@pytest.mark.parametrize("alpha", [0, 60, 110, 200, 255])
def test_the_ring_alpha_is_configurable(alpha):
    cam, surface, renderer, pos, radius, p = _ring_scene(state="INACTIVE_TACKLED", _ring_alpha=alpha)
    expected = _blend(style.PITCH_GREEN, style.INACTIVE_OUTLINE, alpha)
    hits = _ring_pixels(surface, pos, radius + 3, expected, 14)
    if alpha == 0:
        assert not _ring_pixels(surface, pos, radius + 3, style.INACTIVE_OUTLINE, 60)      # nothing drawn at all
        assert all(tuple(surface.get_at((pos[0] + radius + 3, pos[1])))[:3] == tuple(style.PITCH_GREEN) for _ in (0,))
    else:
        assert len(hits) > 60


def test_the_ring_alpha_default_comes_from_the_config():
    from footballcoach.config import load_graphics_config

    _, _, renderer, _, _ = _scene()
    assert renderer._ring_alpha == int(load_graphics_config()["rings"]["alpha"])


def test_rings_are_drawn_under_the_player_sprite():
    """A ring lies under the sprite: wherever the sprite is fully opaque (its pixel is the same on two different
    backgrounds), the picture with the ring layer equals the picture without it; wherever the ring is clear of the
    sprite, it shows."""
    from footballcoach.entities.player import PlayerState

    def frame(with_rings, background):
        cam, surface, renderer, pos, radius = _scene(zoom=5.0)
        surface.fill(background)
        renderer._ring_alpha = 255                                     # strong, so any overdraw would be obvious
        renderer._inactive_alpha = 255                                 # solid sprite
        p = _player(heading=0.0)
        p.state = PlayerState.INACTIVE_TACKLED
        if with_rings:
            renderer.draw_player_rings(surface, [p], selected_id=p.player_id)
        renderer.draw_player(surface, p)
        return surface, pos, radius

    green, magenta = tuple(style.PITCH_GREEN), (255, 0, 255)
    plain_g, pos, radius = frame(False, green)
    plain_m, _, _ = frame(False, magenta)
    ringed_g, _, _ = frame(True, green)
    covered = shown = 0
    for ring_r in (radius + 4, radius + 7):                          # the tackled and the selected rings
        for deg in range(0, 360, 2):
            if 45 <= deg <= 135:
                continue
            x, y = round(pos[0] + ring_r * math.cos(math.radians(deg))), round(pos[1] + ring_r * math.sin(math.radians(deg)))
            if plain_g.get_at((x, y)) == plain_m.get_at((x, y)) and tuple(plain_g.get_at((x, y)))[:3] != green:   # opaque sprite
                covered += 1
                assert plain_g.get_at((x, y)) == ringed_g.get_at((x, y)), (ring_r, deg)
            elif tuple(plain_g.get_at((x, y)))[:3] == green and tuple(ringed_g.get_at((x, y)))[:3] != green:      # ring visible
                shown += 1
    # covered's bound dropped from >10 to >=10 after the shoulder shrink (0.95x on top of the
    # existing 0.9x, corner rounded 0.35->0.4): the shoulders now poke out a hair less far, so one
    # fewer of these fixed sample angles happens to land on opaque sprite -- measured exactly 10.
    # The actual invariant this test exists for (the ring never overdraws opaque sprite pixels) is
    # still asserted, unweakened, for every one of those 10 points.
    assert covered >= 10 and shown > 60


def test_the_ball_state_ring_is_translucent_and_drawn_by_the_ring_layer():
    """`draw_pitch_and_ball` draws the ball's state ring (in the ring layer, under the ball); standalone
    `draw_ball` draws it too by default and `ring=False` leaves it out. The ring is 1px wide, so it is measured
    as the pixels that differ from the same frame with the ring switched off."""
    from footballcoach.entities import Pitch
    from footballcoach.entities.ball import Ball

    pitch = Pitch.standard()
    ball = Ball()
    ball.position = Vector3(0.0, 0.0, 0.11)
    ball.velocity = Vector3(0.0, 0.0, 0.0)

    def render(show_ring, alpha=200, standalone=None):
        cam, surface, renderer, pos, radius = _scene(3.0)
        renderer._ring_alpha = alpha
        renderer._ring_show_rolling = show_ring
        if standalone is None:
            renderer.draw_pitch_and_ball(surface, pitch, ball)
        else:
            renderer.draw_ball(surface, ball, ring=standalone)
        return surface, pos, renderer

    def diff(a, b):
        return [(x, y) for x in range(a.get_width()) for y in range(a.get_height()) if a.get_at((x, y)) != b.get_at((x, y))]

    on, pos, renderer = render(True)
    off, _, _ = render(False)
    ring_r = renderer._ball_draw_radius_px(ball) + renderer._ring_offset_px
    changed = diff(on, off)
    assert len(changed) > 40                                                    # the ring is drawn by draw_pitch_and_ball...
    assert all(abs(math.hypot(x - pos[0], y - pos[1]) - (ring_r - 0.5)) <= 2.0 for x, y in changed)   # ...on its radius
    weaker = diff(render(True, alpha=60)[0], off)
    assert weaker and max(sum(abs(a - b) for a, b in zip(render(True, alpha=60)[0].get_at(p)[:3], off.get_at(p)[:3])) for p in weaker) < \
        max(sum(abs(a - b) for a, b in zip(on.get_at(p)[:3], off.get_at(p)[:3])) for p in changed)          # lower alpha = fainter

    alone_on, _, _ = render(True, standalone=True)
    alone_off, _, _ = render(True, standalone=False)
    blank, _, _ = render(False, standalone=True)
    assert len(diff(alone_on, blank)) > 40 and not diff(alone_off, blank)         # default draws it; ring=False does not


def test_the_stamina_flash_ring_is_in_the_ring_layer_too():
    """Low-stamina players get the pulsing outermost ring from the same translucent layer."""
    import unittest.mock as mock

    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player()
    p.stamina = 0.0
    with mock.patch("pygame.time.get_ticks", return_value=0):                   # 'on' half of the pulse
        renderer.draw_player_rings(surface, [p])
    expected = _blend(style.PITCH_GREEN, style.STAMINA_FLASH_OUTLINE, renderer._ring_alpha)
    # Threshold scaled down a little from the pre-size-floor-decay value (>60 of ~89 sampled
    # angles): the player is now closer to true-to-scale at this zoom, so the ring itself is a bit
    # smaller/thinner in absolute pixels, with proportionally more anti-aliased gaps -- measured 60.
    assert len(_ring_pixels(surface, pos, radius + 10, expected, 14)) >= 55


def _pipeline_frame(players_factory, ring_alpha, background=None, ball_xy=(-30.0, 10.0), ring_offset=None, zoom=5.0):
    """The app's order: draw_pitch_and_ball (shadows, rings, ball), then every player's sprite."""
    from footballcoach.entities import Pitch
    from footballcoach.entities.ball import Ball

    pitch = Pitch.standard()
    cam, surface, renderer, _, radius = _scene(zoom)
    cam.follow(0.0, 0.0)
    renderer._ring_alpha = ring_alpha
    renderer._inactive_alpha = 255
    if ring_offset is not None:
        renderer._ring_offset_px = ring_offset
    ball = Ball()
    ball.position = Vector3(ball_xy[0], ball_xy[1], 0.11)
    ball.velocity = Vector3(0.0, 0.0, 0.0)
    players = players_factory()
    renderer.draw_pitch_and_ball(surface, pitch, ball, players=players, selected_id=players[0].player_id if players else None)
    for p in players:
        renderer.draw_player(surface, p)
    return cam, surface, renderer, ball, players


def test_the_pipeline_draws_player_rings_under_every_sprite():
    """Through draw_pitch_and_ball then draw_player (the app's order), a ring never lands on top of a sprite:
    wherever the sprite is opaque the picture is the same with the rings on or off."""
    from footballcoach.entities.player import PlayerState

    def one():
        p = _player("a", heading=0.0)
        p.state = PlayerState.INACTIVE_TACKLED
        return [p]

    cam, on, renderer, _, players = _pipeline_frame(one, 255)
    _, off, _, _, _ = _pipeline_frame(one, 0)
    pos = cam.world_to_screen(0.0, 0.0)
    radius = renderer._player_radius_px(players[0])
    # which pixels are opaque sprite: identical when the player is drawn on two different backgrounds
    def sprite_only(bg):
        cam2, surface, r2, _, radius2 = _scene(5.0)
        surface.fill(bg)
        r2._inactive_alpha = 255
        p = one()[0]
        r2.draw_player(surface, p)
        return surface
    on_green, on_magenta = sprite_only(style.PITCH_GREEN), sprite_only((255, 0, 255))
    covered = 0
    changed_elsewhere = 0
    for ring_r in (radius + 4, radius + 7):
        for deg in range(0, 360, 2):
            if 45 <= deg <= 135:
                continue
            x, y = round(pos[0] + ring_r * math.cos(math.radians(deg))), round(pos[1] + ring_r * math.sin(math.radians(deg)))
            if on_green.get_at((x, y)) == on_magenta.get_at((x, y)) and tuple(on_green.get_at((x, y)))[:3] != tuple(style.PITCH_GREEN):
                covered += 1
                assert on.get_at((x, y)) == off.get_at((x, y)), (ring_r, deg)
            elif on.get_at((x, y)) != off.get_at((x, y)):
                changed_elsewhere += 1
    # see the matching comment in test_rings_are_drawn_under_the_player_sprite: the shoulder shrink
    # dropped this sample's covered count from >10 to exactly 10.
    assert covered >= 10 and changed_elsewhere > 60


def test_the_pipeline_draws_the_ball_ring_under_the_ball():
    """The ball's state ring is under the ball: with the ring pulled inside the ball's body (a negative
    offset) the ball's own pixels are the same with the ring on or off."""
    def none():
        return []

    ball_xy = (5.0, 3.0)
    cam, on, renderer, ball, _ = _pipeline_frame(none, 255, ball_xy=ball_xy, ring_offset=-3, zoom=3.0)
    _, off, _, _, _ = _pipeline_frame(none, 0, ball_xy=ball_xy, ring_offset=-3, zoom=3.0)
    bx, by = cam.world_to_screen(*ball_xy)
    rad = renderer._ball_draw_radius_px(ball)
    inside = [(x, y) for x in range(bx - rad + 2, bx + rad - 1) for y in range(by - rad + 2, by + rad - 1)
              if math.hypot(x - bx, y - by) <= rad - 2]
    assert inside and all(on.get_at(p) == off.get_at(p) for p in inside)


# ---------------------------------------------------------------------------
# Legs / upper-body split (ball drawn between them: legs under, torso over)
# ---------------------------------------------------------------------------

def test_legs_and_upper_layers_recomposite_close_to_the_combined_pose(params):
    """`_render_pose_layers` is an independent implementation (see its docstring for why), so its
    two layers, blitted legs-then-upper, should closely but not necessarily EXACTLY reproduce
    `_render_pose`'s single-surface output -- the only place they can legitimately differ is the
    ~1px seam where the shorts tuck under the shirt hem (two independent supersample/downscales
    instead of one)."""
    np = _np()
    for lvl, side in ((0.0, 1), (1.0, 1), (0.5, -1)):
        combined = ps._render_pose(params, SHIRT, lvl, lvl, side)
        legs, upper = ps._render_pose_layers(params, SHIRT, lvl, lvl, side)
        recomposited = pygame.Surface((ps._BASE_SIZE, ps._BASE_SIZE), pygame.SRCALPHA)
        recomposited.blit(legs, (0, 0))
        recomposited.blit(upper, (0, 0))
        a = pygame.surfarray.array3d(combined).astype(int)
        b = pygame.surfarray.array3d(recomposited).astype(int)
        diff = np.abs(a - b).max(axis=-1)
        mismatched = int((diff > 3).sum())
        assert mismatched < 80, (lvl, side, mismatched)          # a thin seam band, not a gross mismatch
        assert diff.max() < 90


def test_the_legs_layer_has_no_shirt_or_hair_colour_and_the_upper_layer_has_no_shorts_or_shoe_colour(params):
    s = ps.PlayerSpriteSet(params, SHIRT)
    for side in (1, -1):
        for level in (None,) + ps.STRIDE_LEVELS:
            legs = s.get_legs(side, level)
            upper = s.get_upper(side, level)
            legs_colours = {tuple(legs.get_at((x, y)))[:3] for x in range(0, ps._BASE_SIZE, 2) for y in range(0, ps._BASE_SIZE, 2)
                            if legs.get_at((x, y))[3] > 127}
            upper_colours = {tuple(upper.get_at((x, y)))[:3] for x in range(0, ps._BASE_SIZE, 2) for y in range(0, ps._BASE_SIZE, 2)
                             if upper.get_at((x, y))[3] > 127}
            assert SHIRT not in legs_colours and tuple(params.hair_color) not in legs_colours
            assert tuple(params.shoe_color) not in upper_colours


def test_legs_plus_upper_bounding_box_matches_the_combined_pose_bounding_box(params):
    """No part of the figure is lost or duplicated: the union of the two layers' opaque pixels has
    the same extent as the combined pose's."""
    s = ps.PlayerSpriteSet(params, SHIRT)
    for side in (1, -1):
        for level in ps.STRIDE_LEVELS:
            combined_bbox = _bbox(_opaque(s.get(side, level)))
            legs_mask, upper_mask = _opaque(s.get_legs(side, level)), _opaque(s.get_upper(side, level))
            union = [[a or b for a, b in zip(r1, r2)] for r1, r2 in zip(legs_mask, upper_mask)]
            assert _bbox(union) == combined_bbox


def test_draw_player_legs_draws_only_the_legs_layer():
    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player(heading=0.0)
    renderer.draw_player_legs(surface, [p])
    grass = tuple(style.PITCH_GREEN)
    drawn = {tuple(surface.get_at((x, y)))[:3] for x in range(pos[0] - 3 * radius, pos[0] + 3 * radius)
             for y in range(pos[1] - 3 * radius, pos[1] + 3 * radius) if tuple(surface.get_at((x, y)))[:3] != grass}
    assert style.TEAM_LEFT_COLOUR not in drawn                    # no shirt: the torso wasn't drawn
    assert drawn, "expected some leg/shorts/shoe pixels"


def test_draw_player_legs_is_a_noop_with_sprites_disabled():
    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    renderer._sprites_enabled = False
    before = surface.copy()
    renderer.draw_player_legs(surface, [_player()])
    assert all(surface.get_at((x, y)) == before.get_at((x, y)) for x in range(0, surface.get_width(), 23) for y in range(0, surface.get_height(), 17))


def test_draw_player_legs_false_draws_only_the_upper_layer():
    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player(heading=0.0)
    renderer.draw_player(surface, p, legs=False)
    grass = tuple(style.PITCH_GREEN)
    drawn = {tuple(surface.get_at((x, y)))[:3] for x in range(pos[0] - 3 * radius, pos[0] + 3 * radius)
             for y in range(pos[1] - 3 * radius, pos[1] + 3 * radius) if tuple(surface.get_at((x, y)))[:3] != grass}
    assert style.TEAM_LEFT_COLOUR in drawn                        # the torso/shirt IS there


def test_draw_player_default_legs_true_is_unchanged_from_before_the_split():
    """`draw_player`'s default matches the OLD, un-split behaviour exactly: it's still
    `_draw_player_sprite`, which still uses the untouched `_render_pose`/`sprite_set.get`."""
    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player(heading=0.7)
    renderer.draw_player(surface, p)
    a = surface.copy()
    surface.fill(style.PITCH_GREEN)
    renderer.draw_player(surface, p, legs=True)
    assert all(surface.get_at((x, y)) == a.get_at((x, y)) for x in range(surface.get_width()) for y in range(0, surface.get_height(), 3))


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, math.pi, -math.pi / 2])
def test_the_ball_is_drawn_over_an_outstretched_foot_but_under_the_torso(heading):
    """The actual point of the split: with the ball placed where a full-stride front foot lands,
    going through the real pipeline (`draw_pitch_and_ball` then `draw_player(..., legs=False)`) the
    ball ends up ON TOP of that foot (foot pixels there are replaced by white/ball pixels), whereas
    drawing the ball first and then the OLD combined sprite on top -- what every caller did before --
    left the foot fully covering the ball there instead."""
    from footballcoach.entities import Pitch
    from footballcoach.entities.ball import Ball

    pitch = Pitch.standard()
    top_speed = ps.PlayerSpriteParams.from_config().top_speed_for_stride_mps
    fx, fy = math.cos(-heading), math.sin(-heading)

    # Measure the actual outstretched front foot's reach for this exact setup, rather than a
    # hardcoded metres offset (which drifted once the size-floor decay changed how big the sprite
    # -- and so the foot's reach -- is at this zoom): a player straight ahead of the pitch centre,
    # to keep the light/shading side-effects out of it.
    cam0, surface0, renderer0, pos0, radius0 = _scene(zoom=6.0)
    probe = _player(heading=heading, speed=top_speed)
    renderer0._player_gait_phase[probe.player_id] = math.pi / 2
    renderer0.draw_player_legs(surface0, [probe])
    reach_px = max(
        math.hypot(x - pos0[0], y - pos0[1])
        for x in range(surface0.get_width()) for y in range(surface0.get_height())
        if tuple(surface0.get_at((x, y)))[:3] != tuple(style.PITCH_GREEN)
        and math.hypot(x - pos0[0], y - pos0[1]) < 3 * radius0
    )
    forward_m = 0.75 * reach_px / cam0.pixels_per_metre

    def render(split):
        cam, surface, renderer, pos, radius = _scene(zoom=6.0)
        p = _player(heading=heading, speed=top_speed)
        renderer._player_gait_phase[p.player_id] = math.pi / 2     # mid-cycle: full front-leg extension
        ball = Ball()
        ball.position = Vector3(fx * forward_m, fy * forward_m, 0.11)
        ball.velocity = Vector3(0.0, 0.0, 0.0)
        if split:
            renderer.draw_pitch_and_ball(surface, pitch, ball, players=[p])
            renderer.draw_player(surface, p, legs=False)
        else:
            renderer.draw_pitch_and_ball(surface, pitch, ball, players=[])
            renderer.draw_player(surface, p)
        return cam, surface

    cam, old_surface = render(False)
    _, new_surface = render(True)
    fpx, fpy = cam.world_to_screen(fx * forward_m, fy * forward_m)
    ball_white = [(x, y) for x in range(fpx - 14, fpx + 15) for y in range(fpy - 14, fpy + 15)
                  if sum(new_surface.get_at((x, y))[:3]) > 620]     # near-white: the ball's own body colour
    assert ball_white, "expected to find some ball-white pixels at the foot's landing spot"
    old_white = sum(1 for x, y in ball_white if sum(old_surface.get_at((x, y))[:3]) > 620)
    assert old_white < 0.8 * len(ball_white)                        # some of those spots were NOT white before (a foot covered them)


def test_a_ball_at_the_players_own_centre_still_sits_under_the_upper_body():
    """A ball placed right at the player (nothing to peek through -- there's no leg sticking out from
    directly under the body there, and the head/hair happen to be centred on the player's own position
    too) is still fully covered by the upper-body layer, exactly like before -- never left showing the
    ball's own near-white colour through."""
    from footballcoach.entities import Pitch
    from footballcoach.entities.ball import Ball

    pitch = Pitch.standard()
    cam, surface, renderer, pos, radius = _scene(zoom=6.0)
    p = _player(heading=0.3)
    ball = Ball()
    ball.position = Vector3(0.0, 0.0, 0.11)
    ball.velocity = Vector3(0.0, 0.0, 0.0)
    renderer.draw_pitch_and_ball(surface, pitch, ball, players=[p])
    renderer.draw_player(surface, p, legs=False)
    px = tuple(surface.get_at(pos))[:3]
    assert px != tuple(style.PITCH_GREEN) and sum(px) < 620       # some opaque body colour, not the ball's near-white


def test_app_draws_each_player_with_legs_false_since_draw_pitch_and_ball_already_drew_them():
    """Regression guard for the app.py wiring itself (not exercised by any Renderer-level test above):
    the per-player draw call in App._draw_match must pass legs=False, or every player would be drawn
    with their legs twice (once by draw_pitch_and_ball's shared pass, once again here) -- harmless
    pixel-wise (the second full sprite draws over the first) but defeats the ball-at-the-feet effect
    entirely, since the last thing drawn per player would once again be the WHOLE sprite over the ball."""
    import inspect

    from footballcoach.ui import app as app_module

    src = inspect.getsource(app_module.App._draw_match)
    assert "self.renderer.draw_player(" in src
    call_start = src.index("self.renderer.draw_player(")
    call_text = src[call_start:src.index(")", src.index(")", call_start) + 1) + 1]
    assert "legs=False" in call_text


# ---------------------------------------------------------------------------
# Size-floor zoom decay: ball/player minimum-visibility floors converge
# toward true-to-scale size as the camera zooms in, instead of staying a
# constant multiple too big forever.
# ---------------------------------------------------------------------------

def _oversize_ratio(renderer, cam, zoom, entity_radius_m, radius_fn):
    cam.set_zoom_level(zoom)
    drawn = radius_fn()
    true = cam.scale_length(entity_radius_m)
    return drawn / true


def test_the_size_floor_decay_default_comes_from_the_config():
    from footballcoach.config import load_graphics_config

    _, _, renderer, _, _ = _scene()
    assert renderer._size_floor_zoom_decay == pytest.approx(float(load_graphics_config()["size_floor"]["zoom_decay"]))
    assert 0.0 < renderer._size_floor_zoom_decay < 1.0


def test_ball_and_player_sizing_is_unchanged_at_the_default_zoom():
    """zoom_scale == 1 at 1x, and 1 ** anything == 1, so the decay must not change anything there."""
    from footballcoach.entities.ball import Ball

    cam, surface, renderer, pos, radius = _scene(zoom=1.0)
    ball = Ball()
    ball.position = Vector3(0.0, 0.0, 0.11)
    for decay in (1.0, 0.7, 0.45, 0.2, 0.0):
        renderer._size_floor_zoom_decay = decay
        assert renderer._player_radius_px(_player()) == renderer.min_player_radius_px
        assert renderer._ball_base_radius_px(ball) == pytest.approx(renderer.min_ball_radius_px)


def test_a_lower_decay_converges_faster_toward_true_scale_while_1_never_converges():
    from footballcoach.entities.ball import Ball

    cam, surface, renderer, pos, radius = _scene(zoom=1.0)
    ball = Ball()
    ball.position = Vector3(0.0, 0.0, 0.11)

    def ball_ratio(zoom, decay):
        # The TRUE size here is the raw (unquantised) formula, not `cam.scale_length` -- that
        # truncates to an int, which adds its own small quantisation noise (e.g. 0.99px -> 1,
        # 4.95px -> 4) unrelated to the floor behaviour this test is isolating.
        renderer._size_floor_zoom_decay = decay
        cam.set_zoom_level(zoom)
        return renderer._ball_base_radius_px(ball) / (ball.radius_m * cam.pixels_per_metre)

    r1_full, r5_full = ball_ratio(1.0, 1.0), ball_ratio(5.0, 1.0)
    assert r1_full == pytest.approx(r5_full, abs=0.01)              # the old behaviour: never converges

    r1_decay, r5_decay = ball_ratio(1.0, 0.45), ball_ratio(5.0, 0.45)
    assert r1_decay == pytest.approx(r1_full, abs=0.01)              # unchanged at 1x
    assert r5_decay < r5_full - 1.0                                  # ...but clearly closer to true scale by zoom 5

    r5_lower = ball_ratio(5.0, 0.2)
    assert r5_lower < r5_decay                                       # a smaller decay converges even faster


def test_ball_and_player_floors_use_the_same_decay_so_they_converge_together():
    """Both `_ball_base_radius_px` and `_player_radius_px` read the SAME `_size_floor_zoom_decay` --
    changing it moves both, rather than one entity converging to true scale long before the other."""
    from footballcoach.entities.ball import Ball

    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    ball = Ball()
    ball.position = Vector3(0.0, 0.0, 0.11)
    before_ball = renderer._ball_base_radius_px(ball)
    before_player = renderer._player_radius_px(_player())
    renderer._size_floor_zoom_decay = 0.1
    assert renderer._ball_base_radius_px(ball) < before_ball
    assert renderer._player_radius_px(_player()) < before_player


def test_click_hit_testing_radius_is_independent_of_the_drawn_size_floor():
    """input.py's SELECT_TOLERANCE_PX-based hit test uses the player's true radius_m directly (see
    ui/knowledge.md) -- confirm it takes no renderer/zoom_decay input at all, i.e. cannot be affected
    by this change."""
    import inspect

    from footballcoach.ui import input as input_module

    assert "player.radius_m" in inspect.getsource(input_module)


# ---------------------------------------------------------------------------
# Kick/tackle swing animation
# ---------------------------------------------------------------------------

def test_local_direction_from_world_is_its_own_inverse():
    """`local_direction_from_world`'s docstring claims the change-of-basis matrix is an involution
    (its own inverse) -- so applying it twice should return to the original vector. This is the
    actual property the function relies on (there's no separate "world_direction_from_local" to
    round-trip against), checked at several headings and directions rather than assumed."""
    import random

    rng = random.Random(0)
    for _ in range(20):
        heading = rng.uniform(-math.pi, math.pi)
        wx, wy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if math.hypot(wx, wy) < 1e-6:
            continue
        lx, ly = ps.local_direction_from_world(wx, wy, heading)
        rx, ry = ps.local_direction_from_world(lx, ly, heading)
        assert rx == pytest.approx(wx, abs=1e-9)
        assert ry == pytest.approx(wy, abs=1e-9)


def test_local_forward_and_sideways_map_to_the_expected_world_directions():
    """The two basis directions, checked against the values this module's docstring claims were
    measured empirically (rotating a real marker through the renderer's own rotozoom formula):
    local (0,1) ("forward") -> world (cos(heading), sin(heading)); local (1,0) ("local right") ->
    world 90 degrees CCW from that."""
    for heading_deg in (0, 45, 90, 135, 180, -45, -90):
        heading = math.radians(heading_deg)
        # local_direction_from_world is the INVERSE map; feed it the claimed world direction and
        # confirm it recovers the local basis vector, rather than re-deriving the forward map here.
        world_forward = (math.cos(heading), math.sin(heading))
        assert ps.local_direction_from_world(*world_forward, heading) == pytest.approx((0.0, 1.0), abs=1e-9)
        world_right = (math.cos(heading + math.pi / 2), math.sin(heading + math.pi / 2))
        assert ps.local_direction_from_world(*world_right, heading) == pytest.approx((1.0, 0.0), abs=1e-9)


def test_local_direction_matches_the_renderers_own_rotation_end_to_end():
    """Places a synthetic marker at a claimed local direction, rotates it through the ACTUAL
    renderer rotation formula (`_draw_player_pose_layer`'s `rotate_deg`, copied here since it's not
    itself a standalone function), and confirms it lands on the correct side on screen -- ties the
    direction math to the real rendering code, not just to its own docstring's claim."""
    from footballcoach.entities import Pitch
    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    for heading_deg in (0, 90, -90, 180):
        heading = math.radians(heading_deg)
        pitch = Pitch.standard()
        cam = Camera.fit_to_pitch(pitch)
        renderer = Renderer(cam)
        marker = pygame.Surface((ps._SIZE, ps._SIZE), pygame.SRCALPHA)
        marker.fill((0, 0, 0, 0))
        local_x, local_y = ps._SIZE / 2, ps._SIZE / 2 + ps._SIZE * 0.3  # local "forward" marker
        pygame.draw.circle(marker, (255, 255, 255, 255), (local_x, local_y), 6)

        hx, hy = math.cos(-heading), math.sin(-heading)
        rotate_deg = 90.0 - math.degrees(math.atan2(hy, hx))
        rotated = pygame.transform.rotozoom(marker, rotate_deg, 1.0)
        rw, rh = rotated.get_size()
        xs = ys = wsum = 0.0
        for y in range(rh):
            for x in range(rw):
                a = rotated.get_at((x, y))[3]
                if a > 30:
                    xs += x * a
                    ys += y * a
                    wsum += a
        sx, sy = xs / wsum - (rw - 1) / 2.0, ys / wsum - (rh - 1) / 2.0
        world_dx, world_dy = sx, -sy  # screen -> world (world_to_screen_f flips y, see camera.py)
        n = math.hypot(world_dx, world_dy)
        world_dx, world_dy = world_dx / n, world_dy / n
        assert (world_dx, world_dy) == pytest.approx((math.cos(heading), math.sin(heading)), abs=0.05)


def test_kick_swing_state_starts_and_ends_at_rest():
    frac0, arm0 = ps.kick_swing_state_at(0.0)
    assert frac0 == pytest.approx(0.0)
    assert arm0 == pytest.approx(0.0)
    frac_end, arm_end = ps.kick_swing_state_at(ps.KICK_SWING_DURATION_S)
    assert frac_end == pytest.approx(0.0, abs=1e-9)
    assert arm_end == pytest.approx(0.0, abs=1e-9)
    # well past the end: still rest, not an index error or extrapolation
    frac_late, arm_late = ps.kick_swing_state_at(ps.KICK_SWING_DURATION_S + 5.0)
    assert (frac_late, arm_late) == (0.0, 0.0)


def test_kick_swing_state_is_continuous_across_phase_boundaries():
    """No pop/discontinuity where the piecewise curve hands off between backswing -> strike ->
    recover -- each phase's own formula should agree with its neighbour's at the shared instant."""
    eps = 1e-7
    for boundary in (ps.KICK_BACKSWING_S, ps.KICK_BACKSWING_S + ps.KICK_STRIKE_S):
        before = ps.kick_swing_state_at(boundary - eps)
        after = ps.kick_swing_state_at(boundary + eps)
        assert before[0] == pytest.approx(after[0], abs=1e-3)
        assert before[1] == pytest.approx(after[1], abs=1e-3)


def test_kick_swing_goes_behind_then_reaches_past_full_extension():
    """The whole point of the axis design: negative (behind the hip) during backswing, ramping
    through zero (contact) up to `KICK_CONTACT_FRAC` (a full, reaching extension) during the
    strike -- and it actually reaches those configured extremes, not just trends toward them."""
    backswing_frac, _ = ps.kick_swing_state_at(ps.KICK_BACKSWING_S)
    assert backswing_frac == pytest.approx(ps.KICK_BACK_FRAC, abs=1e-6)
    strike_end_frac, _ = ps.kick_swing_state_at(ps.KICK_BACKSWING_S + ps.KICK_STRIKE_S)
    assert strike_end_frac == pytest.approx(ps.KICK_CONTACT_FRAC, abs=1e-6)

    # somewhere in the strike phase, the leg is actually behind (negative) at the start and ahead
    # (positive) by the end -- i.e. it genuinely swings THROUGH the hip, not just up to it
    fracs = [ps.kick_swing_state_at(ps.KICK_BACKSWING_S + f * ps.KICK_STRIKE_S)[0] for f in (0.0, 0.3, 0.6, 1.0)]
    assert fracs[0] < 0.0 < fracs[-1]
    assert fracs == sorted(fracs)  # monotonically increasing through the strike -- no back-and-forth wobble


def test_render_kick_pose_returns_the_expected_size_and_is_the_sum_of_its_layers():
    """The returned surface is bigger than the normal `_BASE_SIZE` gait sprites (padded so the
    reaching leg can't be clipped -- see `_kick_pad_px`), so the expected size is derived from that
    same formula rather than hardcoded -- a literal `_BASE_SIZE` would just be wrong here, not a
    meaningful regression check, once the padding exists at all."""
    params = ps.PlayerSpriteParams.from_config()
    legs = ps.render_kick_pose_legs(params, 1, (0.0, 1.0), 0.5)
    upper = ps.render_kick_pose_upper(params, (200, 30, 30), 1, 0.5)
    combined = ps.render_kick_pose(params, (200, 30, 30), 1, (0.0, 1.0), 0.5, 0.5)
    expected_base_size = (ps._SIZE + 2 * ps._kick_pad_px()) // ps._SUPERSAMPLE
    assert expected_base_size > ps._BASE_SIZE, "this test is only meaningful once padding is non-zero"
    assert legs.get_size() == (expected_base_size, expected_base_size)
    assert upper.get_size() == (expected_base_size, expected_base_size)
    assert combined.get_size() == (expected_base_size, expected_base_size)
    expected = legs.copy()
    expected.blit(upper, (0, 0))
    for y in range(expected_base_size):
        for x in range(expected_base_size):
            assert combined.get_at((x, y)) == expected.get_at((x, y))


def test_kick_pose_reference_size_matches_the_normal_gait_sprites_base_size():
    """`Renderer._draw_player_pose_layer` scales a swing sprite by `target_diameter /
    KICK_POSE_REFERENCE_SIZE`, not by its actual (padded) width -- confirm that reference really is
    the un-padded torso size, i.e. exactly what the normal gait sprites use, so a kicking player's
    on-screen torso size doesn't change just because the padding does."""
    assert ps.KICK_POSE_REFERENCE_SIZE == ps._BASE_SIZE


def test_shorts_length_is_clamped_to_the_legs_own_length_not_always_shorts_len_max():
    """Regression: `_draw_shorts_hem` originally drew an UNCONDITIONAL `shorts_len_max` hem
    regardless of how long the leg under it actually was -- for the planted leg (whose real length
    is just `base_min_leg_len`, much shorter than `shorts_len_max`), that put a shorts block AS
    LONG AS the entire leg, with no exposed shaft/skin at all -- reading as one flat white block
    rather than a leg wearing shorts (the user: "some real bizarre stuff going on with the
    shorts... I remember them being done well in the original sprites"). The gait poses always
    clamped shorts length to `min(total_len, shorts_len_max)`; the swing legs now match."""
    geo = ps._pose_geometry()
    short_leg_len = geo["base_min_leg_len"]
    assert short_leg_len < geo["shorts_len_max"], "this test needs a leg shorter than the shorts hem to be meaningful"

    params = ps.PlayerSpriteParams.from_config()
    surf = pygame.Surface((ps._SIZE, ps._SIZE), pygame.SRCALPHA)
    x, hip_y = ps._SIZE / 2, ps._SIZE / 2
    ps._draw_shorts_hem(surf, params, geo, x, hip_y, short_leg_len)
    # the rendered shorts hem's own bottom edge should sit at hip_y + short_leg_len, not
    # hip_y + shorts_len_max -- measured by the lowest shorts-coloured pixel at this column
    ys = [y for y in range(surf.get_height()) if _near(surf.get_at((int(x), y)), params.shorts_color, tol=15)]
    assert ys, "no shorts pixels rendered at all"
    measured_bottom = max(ys)
    assert measured_bottom == pytest.approx(hip_y + short_leg_len, abs=3)
    assert measured_bottom < hip_y + geo["shorts_len_max"] - 5, "shorts must NOT reach all the way to shorts_len_max for a short leg"


def test_the_swing_poses_shorts_are_not_fully_hidden_under_the_torso():
    """Regression: the swing legs' hips were originally placed at dead centre (`cy`), same as the
    torso's own centre -- since `_draw_shorts_hem` draws a short, FIXED-length hem straight down
    from the hip, that put the ENTIRE shorts rectangle inside the torso's own vertical span, so it
    was completely painted over once the upper body (drawn after) was composited on top -- the user
    caught this by eye ("doesn't this sprite have shorts?"). The gait poses avoid this by offsetting
    each leg's hip below centre by (torso_h/2 - hip_inset) before drawing its shorts; the swing legs
    now use that same offset. Checked directly: the shorts hem's own y-range must extend below the
    torso's bottom edge, not merely that shorts pixels exist somewhere (they existed before the fix
    too -- in the legs layer alone -- they just never survived compositing)."""
    geo = ps._pose_geometry()
    hip_y = ps._SIZE / 2 + (geo["torso_h"] / 2 - geo["hip_inset"])
    torso_bottom = ps._SIZE / 2 + geo["torso_h"] / 2
    shorts_bottom = hip_y + geo["shorts_len_max"]
    assert shorts_bottom > torso_bottom + 2, "the shorts hem must clear the torso's own bottom edge"

    # and confirm it survives actual compositing: render the full pose and find shorts-coloured
    # pixels (near-white) below where the torso's shirt colour ends
    params = ps.PlayerSpriteParams.from_config()
    combined = ps.render_kick_pose(params, (13, 77, 201), 1, (0.3, 0.9), 1.0, 0.3)
    w, h = combined.get_size()
    shorts_like = [
        (x, y) for x in range(w) for y in range(h)
        if _near(combined.get_at((x, y)), params.shorts_color, tol=15)
    ]
    assert shorts_like, "no shorts-coloured pixels survived compositing at all"


def test_a_kicking_players_torso_is_the_same_on_screen_size_as_a_normal_one():
    """End-to-end regression for the scaling bug `KICK_POSE_REFERENCE_SIZE` fixes: if
    `_draw_player_pose_layer` ever scaled a swing sprite by its actual (padded, hence bigger) width
    instead of that fixed reference, the whole sprite -- torso included -- would render visibly
    SMALLER than a non-swinging player at the same radius_px, since the same target_diameter would
    then be divided by a bigger denominator. Measured via the rendered TORSO's own pixel footprint
    (a contiguous blob of shirt colour near the player's centre), not just overall sprite bbox,
    since the padding intentionally makes the full bbox bigger -- only the torso itself must match."""
    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player(heading=0.3)
    shirt = renderer._player_draw_colour(p)

    def torso_area(swinging):
        surf = pygame.Surface(surface.get_size())
        surf.fill(style.PITCH_GREEN)
        if swinging:
            renderer.trigger_kick_swing(p, (1.0, 0.2))
            renderer.update_player_animations([p], ps.KICK_BACKSWING_S)
        renderer.draw_player_legs(surf, [p])
        renderer.draw_player(surf, p, legs=False)
        count = 0
        for dx in range(-30, 30):
            for dy in range(-30, 30):
                x, y = pos[0] + dx, pos[1] + dy
                if 0 <= x < surf.get_width() and 0 <= y < surf.get_height() and _near(surf.get_at((x, y)), shirt, tol=25):
                    count += 1
        return count

    plain_area = torso_area(False)
    renderer._player_swing.clear()
    swing_area = torso_area(True)
    assert plain_area > 20 and swing_area > 20
    assert swing_area == pytest.approx(plain_area, rel=0.35)


def test_kick_pad_is_analytically_big_enough_for_the_worst_case_direction():
    """A direct, pre-render arithmetic check on `_kick_pad_px()` against the exact worst-case
    (triangle-inequality) requirement its own docstring claims -- independent of any rendering, so
    it can't be fooled by a shape that clips so completely it leaves nothing at the edge at all
    (see `test_the_swing_leg_is_never_fully_clipped_off_canvas` for why that's a real risk, not a
    hypothetical one, for a shape entirely off-canvas rather than merely touching the border)."""
    geo = ps._pose_geometry()
    max_frac = max(abs(ps.KICK_BACK_FRAC), ps.KICK_CONTACT_FRAC)
    required_half = geo["hip_x_offset"] + max_frac * geo["leg_reach"] + geo["shoe_ry"] * 1.5
    pad = ps._kick_pad_px()
    actual_half = ps._SIZE / 2 + pad
    assert actual_half >= required_half


def test_the_swing_leg_is_never_fully_clipped_off_canvas():
    """The real failure mode an insufficient pad causes: not a visibly cut-off edge (a shape that's
    ENTIRELY outside the canvas draws nothing at all, leaving the edge transparent, not opaque --
    so a same only checking the border misses this), but the foot silently vanishing. Checked at
    the worst-case direction (aligned with the hip's own offset axis, where the hip offset and the
    swing distance add most directly -- see `_kick_pad_px`'s docstring) by confirming a skin/shoe-
    coloured blob of a sane minimum size actually exists somewhere in the rendered image, not just
    that pixels exist at all (an empty canvas with one stray pixel shouldn't pass either)."""
    params = ps.PlayerSpriteParams.from_config()
    for direction in ((1.0, 0.0), (-1.0, 0.0)):
        for frac in (ps.KICK_BACK_FRAC, ps.KICK_CONTACT_FRAC):
            surf = ps.render_kick_pose_legs(params, 1, direction, frac)
            opaque = sum(
                1 for x in range(surf.get_width()) for y in range(surf.get_height())
                if surf.get_at((x, y))[3] > 100
            )
            assert opaque > 40, (direction, frac, opaque)


def test_the_swing_leg_never_gets_clipped_by_the_padded_canvas_at_any_direction():
    """`_kick_pad_px`'s whole job: at the configured reach extremes, the foot must stay inside the
    canvas for EVERY direction, not just the ones already exercised by other tests -- checked here
    by sweeping many directions at both `KICK_BACK_FRAC` and `KICK_CONTACT_FRAC` and confirming the
    rendered opaque pixels never touch the canvas's own edge (a PARTIALLY clipped shape would have
    opaque pixels running right up to row/column 0 or the last one; a FULLY clipped one is instead
    caught by `test_the_swing_leg_is_never_fully_clipped_off_canvas`, since this check alone can't
    tell a shape that's fully off-canvas from one that comfortably fits)."""
    params = ps.PlayerSpriteParams.from_config()
    for angle_deg in range(0, 360, 15):
        rad = math.radians(angle_deg)
        direction = (math.sin(rad), math.cos(rad))
        for frac in (ps.KICK_BACK_FRAC, ps.KICK_CONTACT_FRAC):
            surf = ps.render_kick_pose_legs(params, 1, direction, frac)
            w, h = surf.get_size()
            edge_opaque = any(surf.get_at((x, 0))[3] > 0 or surf.get_at((x, h - 1))[3] > 0 for x in range(w))
            edge_opaque = edge_opaque or any(
                surf.get_at((0, y))[3] > 0 or surf.get_at((w - 1, y))[3] > 0 for y in range(h)
            )
            assert not edge_opaque, (angle_deg, frac)


def test_the_swinging_foot_moves_along_one_fixed_line_not_an_arc():
    """The actual regression this design fixes: the foot's position, sampled across the whole
    swing, should lie on ONE straight line through the hip (the fixed `direction`) -- not trace an
    arc, which is what an earlier (rejected) angle-sweeping version did. Checked by measuring the
    rendered foot's pixel centroid at several points in the swing and confirming they're all
    collinear with the hip, allowing for anti-aliasing/downscale noise."""
    params = ps.PlayerSpriteParams.from_config()

    def foot_centroid(signed_len_frac):
        surf = ps.render_kick_pose_legs(params, 1, (0.3, 0.95), signed_len_frac)
        # crop to the right half (swing_side=1 -> the swinging leg's hip is at cx + offset > cx)
        w, h = surf.get_size()
        xs = ys = wsum = 0.0
        for y in range(h):
            for x in range(w // 2, w):
                a = surf.get_at((x, y))[3]
                if a > 100:
                    xs += x * a
                    ys += y * a
                    wsum += a
        return xs / wsum, ys / wsum

    # backswing (behind) and two forward reach points -- all should sit on the same line through
    # the hip since `direction` never changes, only `signed_len_frac` (the distance along it) does
    points = [foot_centroid(f) for f in (-0.5, 0.3, 0.8, 1.15)]
    # fit: check every point's perpendicular distance from the line through the first two points is
    # small relative to the swing's own reach (a real arc would bow out by a large fraction of it)
    (x0, y0), (x1, y1) = points[0], points[1]
    dx, dy = x1 - x0, y1 - y0
    line_len = math.hypot(dx, dy)
    assert line_len > 1.0, "backswing and forward-reach points should be well separated"
    for x, y in points[2:]:
        perp_dist = abs(dx * (y0 - y) - (x0 - x) * dy) / line_len
        assert perp_dist < line_len * 0.15, (points, perp_dist)


def test_renderer_trigger_kick_swing_starts_the_animation():
    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    p = _player()
    assert p.player_id not in renderer._player_swing
    renderer.trigger_kick_swing(p, (1.0, 0.0))
    assert p.player_id in renderer._player_swing
    side, direction, elapsed_s = renderer._player_swing[p.player_id]
    assert elapsed_s == 0.0
    assert side in (1, -1)
    assert math.hypot(*direction) == pytest.approx(1.0, abs=1e-6)


def test_renderer_trigger_swing_is_a_noop_for_a_degenerate_direction():
    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    p = _player()
    renderer.trigger_kick_swing(p, (0.0, 0.0))
    assert p.player_id not in renderer._player_swing


def test_renderer_trigger_swing_is_a_noop_with_sprites_disabled():
    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    renderer._sprites_enabled = False
    p = _player()
    renderer.trigger_kick_swing(p, (1.0, 0.0))
    assert p.player_id not in renderer._player_swing


def test_update_player_animations_advances_and_then_clears_the_swing():
    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    p = _player()
    renderer.trigger_kick_swing(p, (1.0, 0.0))
    renderer.update_player_animations([p], ps.KICK_SWING_DURATION_S * 0.5)
    assert p.player_id in renderer._player_swing
    _, _, elapsed_s = renderer._player_swing[p.player_id]
    assert elapsed_s == pytest.approx(ps.KICK_SWING_DURATION_S * 0.5)

    renderer.update_player_animations([p], ps.KICK_SWING_DURATION_S)  # well past the remainder
    assert p.player_id not in renderer._player_swing


def _dir_at(angle_deg):
    """A unit direction `angle_deg` degrees from straight-forward (local (0,1)), matching
    `swinging_side_for_direction`'s own `atan2(dx, dy)` convention."""
    rad = math.radians(angle_deg)
    return math.sin(rad), math.cos(rad)


def test_swinging_side_for_direction_crosses_the_body_within_the_cross_angle():
    """The actual anatomy: real kicks plant the near-side foot and swing the FAR-side leg across
    the body to strike -- not the other way around -- for a target up to `MAX_CROSS_ANGLE_DEG` off
    straight ahead. `render_kick_pose_legs`'s swing leg hip sits at local `+swing_side *
    hip_x_offset`, so for a target with a positive local x-component (to the swing_side=+1 leg's
    own side), the CORRECT (crossing) choice is swing_side=-1 (the opposite leg), and vice versa.
    An earlier version of this picked the SAME-side leg (reasoned as "avoids an awkward cross-body
    reach") -- backwards, and the user caught it by eye ("nobody kicks outwards like that... you
    cross your right foot across your body")."""
    assert ps.swinging_side_for_direction(_dir_at(20), fallback_side=1) == -1
    assert ps.swinging_side_for_direction(_dir_at(-20), fallback_side=1) == 1
    assert ps.swinging_side_for_direction(_dir_at(ps.MAX_CROSS_ANGLE_DEG), fallback_side=1) == -1  # inclusive


def test_swinging_side_for_direction_uses_the_near_leg_past_the_cross_angle():
    """Past `MAX_CROSS_ANGLE_DEG`, even a real crossing kick stops being plausible -- switches back
    to the NEAR-side leg for a wide-angle target, per the user: "if the player is kicking past that
    point, use the nearer leg". This is the opposite side choice from the crossing case at the same
    lateral sign, not just a clamp on the same leg."""
    assert ps.swinging_side_for_direction(_dir_at(60), fallback_side=1) == 1
    assert ps.swinging_side_for_direction(_dir_at(-60), fallback_side=1) == -1
    assert ps.swinging_side_for_direction(_dir_at(ps.MAX_CROSS_ANGLE_DEG + 0.5), fallback_side=1) == 1


def test_swinging_side_for_direction_falls_back_for_a_near_straight_target():
    """A target close enough to straight ahead/behind that its lateral component is inside the
    deadzone -- either foot is equally natural, so the caller's fallback (the gait's current
    leading side) is used instead of forcing a side from a near-zero, noisy sign."""
    assert ps.swinging_side_for_direction((0.01, 1.0), fallback_side=1) == 1
    assert ps.swinging_side_for_direction((0.01, 1.0), fallback_side=-1) == -1
    assert ps.swinging_side_for_direction((0.01, -1.0), fallback_side=-1) == -1


def test_renderer_trigger_picks_the_crossing_side_not_the_gait_leading_side():
    """End-to-end: `_trigger_swing` must actually use `swinging_side_for_direction`, not just have
    it defined and unused -- confirmed by triggering with a lateral-but-within-cap target and
    checking the stored side is the CROSSING one regardless of what the gait phase alone would have
    picked. World (0.940, 0.342) at heading=0 -> local (0.342, 0.940), a 20 degree target (within
    `MAX_CROSS_ANGLE_DEG`) toward the swing_side=+1 leg's own side -- crossing needs side=-1."""
    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    p = _player(heading=0.0)
    renderer.trigger_kick_swing(p, (0.940, 0.342))
    side, direction, _ = renderer._player_swing[p.player_id]
    assert direction == pytest.approx((0.342, 0.940), abs=1e-3)
    assert side == -1


def test_clamp_swing_direction_actually_clamps_past_the_outer_cap():
    """`test_renderer_trigger_uses_the_near_leg_past_the_cross_angle` uses a 90-degree target, which
    sits right AT the default `MAX_SWING_ANGLE_DEG` boundary -- clamping is a no-op there regardless
    of what the cap's value actually is, so it can't catch a broken/disabled cap on its own. This
    test uses an angle clearly PAST the cap (150 degrees) and confirms the result is pulled back to
    exactly the cap, not left at 150."""
    far = ps.clamp_swing_direction(_dir_at(150), max_angle_deg=90.0)
    assert far == pytest.approx(_dir_at(90), abs=1e-6)
    far_negative = ps.clamp_swing_direction(_dir_at(-150), max_angle_deg=90.0)
    assert far_negative == pytest.approx(_dir_at(-90), abs=1e-6)
    # and using the module's actual current default, not a passed-in override
    default_far = ps.clamp_swing_direction(_dir_at(150))
    assert default_far == pytest.approx(_dir_at(ps.MAX_SWING_ANGLE_DEG), abs=1e-6)


def test_renderer_trigger_uses_the_near_leg_past_the_cross_angle():
    """Same as the crossing test above, but with a target BEYOND `MAX_CROSS_ANGLE_DEG` -- confirms
    the renderer's wiring picks up the near-leg switch too, not just the pure function in
    isolation. World (0, 1) at heading=0 -> local (1, 0), a 90 degree target -- past the cross
    angle, so the NEAR leg (side=+1) should be used, clamped to `MAX_SWING_ANGLE_DEG`."""
    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    p = _player(heading=0.0)
    renderer.trigger_kick_swing(p, (0.0, 1.0))
    side, direction, _ = renderer._player_swing[p.player_id]
    assert side == 1
    expected = ps.clamp_swing_direction((1.0, 0.0))
    assert direction == pytest.approx(expected, abs=1e-6)


def test_trigger_tackle_swing_uses_the_direction_toward_the_target():
    """`trigger_tackle_swing` takes the same (player, world_direction) shape as
    `trigger_kick_swing` -- confirm it stores that direction (converted to local, THEN clamped --
    see `test_renderer_trigger_uses_the_near_leg_past_the_cross_angle` for a dedicated clamp check;
    world (0,1) at heading=0 is a 90 degree target, past `MAX_SWING_ANGLE_DEG`) rather than
    silently ignoring it or reusing some other state."""
    cam, surface, renderer, pos, radius = _scene(zoom=3.0)
    p = _player(heading=0.0)
    renderer.trigger_tackle_swing(p, (0.0, 1.0))  # world "sideways" relative to heading=0 forward
    _, direction, _ = renderer._player_swing[p.player_id]
    raw = ps.local_direction_from_world(0.0, 1.0, 0.0)
    expected = ps.clamp_swing_direction(raw)
    assert direction == pytest.approx(expected, abs=1e-9)


def test_a_mid_swing_player_draws_a_different_sprite_than_the_normal_gait_pose():
    """Sanity/regression check that `_draw_player_pose_layer` actually substitutes the swing pose
    (rather than the trigger being recorded but silently ignored by the draw path): the rendered
    pixels differ between a player mid-swing and the same player with no swing active."""
    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player(heading=0.4)

    plain = pygame.Surface(surface.get_size())
    plain.fill(style.PITCH_GREEN)
    renderer.draw_player(plain, p)

    renderer.trigger_kick_swing(p, (1.0, 0.3))
    renderer.update_player_animations([p], ps.KICK_BACKSWING_S + ps.KICK_STRIKE_S * 0.5)  # mid-strike
    swinging = pygame.Surface(surface.get_size())
    swinging.fill(style.PITCH_GREEN)
    renderer.draw_player(swinging, p)

    diffs = sum(
        1 for x in range(plain.get_width()) for y in range(plain.get_height())
        if plain.get_at((x, y)) != swinging.get_at((x, y))
    )
    assert diffs > 20


def test_a_mid_swing_player_still_draws_correctly_through_the_split_legs_upper_path():
    """The app *always* draws players split (`draw_player_legs` then `draw_player(legs=False)`) --
    confirm a mid-swing player renders without error through that exact path too, not just the
    combined `legs=True` default (see `render_kick_pose_legs`'s docstring for why this matters)."""
    cam, surface, renderer, pos, radius = _scene(zoom=5.0)
    p = _player(heading=-1.2)
    renderer.trigger_kick_swing(p, (0.5, -0.5))
    renderer.update_player_animations([p], ps.KICK_BACKSWING_S)

    surface.fill(style.PITCH_GREEN)
    renderer.draw_player_legs(surface, [p])
    renderer.draw_player(surface, p, legs=False)  # should not raise, and should draw something
    non_green = sum(
        1 for x in range(surface.get_width()) for y in range(surface.get_height())
        if tuple(surface.get_at((x, y))[:3]) != style.PITCH_GREEN
    )
    assert non_green > 20
