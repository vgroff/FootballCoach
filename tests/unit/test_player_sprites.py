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
    radius = int(max(renderer.min_player_radius_px * cam.zoom_scale, cam.scale_length(0.3)))
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
    assert covered > 10 and shown > 60


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
    assert len(_ring_pixels(surface, pos, radius + 10, expected, 14)) > 60


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
    assert covered > 10 and changed_elsewhere > 60


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
