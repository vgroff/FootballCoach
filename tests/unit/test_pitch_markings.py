"""Geometry of the cosmetic pitch markings drawn by ui/renderer.py's draw_pitch:
the penalty arcs (the part of the 9.15m circle round each penalty spot that is
OUTSIDE the penalty box) and the corner arcs (quarter circles at each corner).

Only the pure point generators are tested here; the drawing itself is
pygame primitives and was checked by rendering to an offscreen surface."""
from __future__ import annotations

import math

import pytest

from footballcoach.entities import Pitch
from footballcoach.ui import style
from footballcoach.ui.renderer import corner_arc_world_points, penalty_arc_world_points

EPS = 1e-9


@pytest.fixture(autouse=True)
def _plain_cosmetics(monkeypatch):
    """The geometry tests below measure lines, nets and frames against exactly flat grass and a
    plain white ball, so the purely cosmetic layers (turf texture, ball shadow, ball shading) are
    off for every Renderer built here; the tests for those features switch them on explicitly."""
    import copy

    from footballcoach.ui import renderer as renderer_mod

    cfg = copy.deepcopy(renderer_mod.load_graphics_config())
    cfg.setdefault("turf", {})["enabled"] = False
    cfg.setdefault("ball_shadow", {})["enabled"] = False
    cfg.setdefault("ball_shading", {})["enabled"] = False
    monkeypatch.setattr(renderer_mod, "load_graphics_config", lambda: cfg)


@pytest.mark.parametrize("left", [True, False])
def test_penalty_arc_lies_on_the_circle_around_the_spot(left):
    pitch = Pitch.standard()
    spot = pitch.penalty_spot(left=left)
    pts = penalty_arc_world_points(pitch, left=left, radius_m=9.15)
    assert len(pts) >= 2
    for x, y in pts:
        assert math.hypot(x - spot.x, y - spot.y) == pytest.approx(9.15, abs=1e-9)


@pytest.mark.parametrize("left", [True, False])
def test_penalty_arc_is_entirely_outside_the_box_and_meets_its_edge(left):
    pitch = Pitch.standard()
    pts = penalty_arc_world_points(pitch, left=left, radius_m=9.15)
    if left:
        box_edge_x = -pitch.half_length + pitch.box_length_m
        assert all(x >= box_edge_x - EPS for x, _ in pts)
    else:
        box_edge_x = pitch.half_length - pitch.box_length_m
        assert all(x <= box_edge_x + EPS for x, _ in pts)
    # Both ends of the arc sit exactly on the box's front edge...
    dx = pitch.box_length_m - pitch.penalty_spot_distance_m
    expected_half_chord = math.sqrt(9.15 ** 2 - dx ** 2)
    for end in (pts[0], pts[-1]):
        assert end[0] == pytest.approx(box_edge_x, abs=1e-9)
        assert abs(end[1]) == pytest.approx(expected_half_chord, abs=1e-9)
    # ...and are on opposite sides of the centreline; the apex is on it.
    assert pts[0][1] == pytest.approx(-pts[-1][1], abs=1e-9)
    apex = max(pts, key=lambda p: abs(p[0] - box_edge_x))
    assert apex[1] == pytest.approx(0.0, abs=0.2)  # 3-degree steps: near, not necessarily on, y=0


def test_penalty_arcs_at_the_two_ends_are_mirror_images():
    pitch = Pitch.standard()
    left = penalty_arc_world_points(pitch, left=True)
    right = penalty_arc_world_points(pitch, left=False)
    assert len(left) == len(right)
    for (lx, ly), (rx, ry) in zip(left, right):
        assert rx == pytest.approx(-lx, abs=1e-9)
        assert ry == pytest.approx(ly, abs=1e-9)


def test_penalty_arc_is_empty_when_the_circle_does_not_clear_the_box():
    pitch = Pitch.standard()
    dx = pitch.box_length_m - pitch.penalty_spot_distance_m  # 5.5m
    assert penalty_arc_world_points(pitch, left=True, radius_m=dx) == []
    assert penalty_arc_world_points(pitch, left=True, radius_m=dx - 1.0) == []


@pytest.mark.parametrize("sx", [-1, 1])
@pytest.mark.parametrize("sy", [-1, 1])
def test_corner_arc_is_a_quarter_circle_inside_the_pitch(sx, sy):
    pitch = Pitch.standard()
    cx, cy = sx * pitch.half_length, sy * pitch.half_width
    pts = corner_arc_world_points(pitch, sx, sy, radius_m=1.0)
    for x, y in pts:
        assert math.hypot(x - cx, y - cy) == pytest.approx(1.0, abs=1e-9)
        assert abs(x) <= pitch.half_length + EPS
        assert abs(y) <= pitch.half_width + EPS
    # Ends land on the two lines that meet at the corner (1m along each).
    first, last = pts[0], pts[-1]
    assert first == pytest.approx((cx - sx * 1.0, cy), abs=1e-9)
    assert last == pytest.approx((cx, cy - sy * 1.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Goal frame: drawn depth matches the engine, crossbar leans over the net
# ---------------------------------------------------------------------------

def _drawn_pitch():
    """Renders the pitch to a plain offscreen Surface (no display needed)."""
    import pygame

    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    renderer.draw_pitch(surface, pitch)
    return pitch, cam, surface, renderer


# The net hatch is blended OVER the goal-footprint outline, so a line pixel under
# it reads ~232 rather than pure 255; hatch-only pixels are at most ~(180,202,184).
_LINE_MIN_CHANNEL = 225


# The goal frame is shaded to look round, so its pixels run from bright highlight to a mid grey
# instead of a uniform line white. They are still NEUTRAL grey (r == g == b), which nothing else
# near the frame is: the net, mouth tint and back-wall tint are all blended over green.
_FRAME_MIN_GREY = 120


def _is_frame_pixel(rgb):
    r, g, b = rgb[:3]
    return r == g == b >= _FRAME_MIN_GREY


def _has_line_white_near(surface, cam, world_x, world_y, *, dx_px=1, dy_px=0):
    px, py = cam.world_to_screen(world_x, world_y)
    return any(
        _is_frame_pixel(surface.get_at((px + i, py + j)))
        for i in range(-dx_px, dx_px + 1)
        for j in range(-dy_px, dy_px + 1)
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_drawn_net_depth_matches_the_engines_back_wall(sign):
    """The engine reflects the ball off a back wall at half_length +
    pitch.goal_depth_m (ball_physics.resolve_goal_boundary); the drawn back of
    the net must be there too, not at a hardcoded different depth."""
    pitch, cam, surface, _ = _drawn_pitch()
    engine_back_x = sign * (pitch.half_length + pitch.goal_depth_m)
    assert _has_line_white_near(surface, cam, engine_back_x, 0.0)
    # And no line at the old hardcoded 2.0m depth (0.45m short of it): only hatch there.
    old_back_x = sign * (pitch.half_length + 2.0)
    assert not _has_line_white_near(surface, cam, old_back_x, 0.0, dx_px=0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_crossbar_is_drawn_displaced_from_the_goal_line_over_the_net(sign):
    pitch, cam, surface, renderer = _drawn_pitch()
    lean = min(renderer._crossbar_lean_m, pitch.goal_depth_m)
    assert lean > 0.3  # default config: a visible offset, not hidden on the goal line
    bar_x = sign * (pitch.half_length + lean)
    # Sample a row a little way in from the post, so the goal line itself (which
    # is also white) can't account for it; the bar must be there at that y.
    y = pitch.goal_width_m / 2.0 - 1.5
    assert _has_line_white_near(surface, cam, bar_x, y, dx_px=1, dy_px=0)


# ---------------------------------------------------------------------------
# Goal frame: symmetric posts, open mouth, ball/crossbar layering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("left", [True, False])
def test_posts_are_symmetric_about_the_centre_row_and_sit_on_the_mouth_edges(left):
    """Regression: the top post used to sit 1px INSIDE the goal-mouth tint and
    the bottom post 1px OUTSIDE it (independent truncation of the two post
    rows, plus a downward-biased thick line vs an inward-drawn outline)."""
    pitch, cam, surface, renderer = _drawn_pitch()
    g = renderer._goal_px(pitch, left)
    centre_row = cam.world_to_screen(0.0, 0.0)[1]
    mid_x = (g.line_x + g.bar_x) // 2  # between the two post feet: only the posts are bright here

    bright = [y for y in range(centre_row - 80, centre_row + 80)
              if _is_frame_pixel(surface.get_at((mid_x, y)))]
    post_w = renderer._goal_post_px()
    assert len(bright) == 2 * post_w
    # Mirror images about the centre row (exactly, for odd widths; within the
    # unavoidable half pixel of an even-width line on an integer grid otherwise).
    assert bright[0] + bright[-1] == 2 * centre_row + (post_w % 2) - 1

    pitch_green = tuple(style.PITCH_GREEN)
    # Outside each post is plain pitch...
    assert tuple(surface.get_at((mid_x, bright[0] - 1)))[:3] == pitch_green
    assert tuple(surface.get_at((mid_x, bright[-1] + 1)))[:3] == pitch_green
    # ...and just inside each is the (identical) mouth tint.
    inner_top = tuple(surface.get_at((mid_x, bright[0] + post_w)))[:3]
    inner_bot = tuple(surface.get_at((mid_x, bright[-1] - post_w)))[:3]
    assert inner_top == inner_bot != pitch_green


@pytest.mark.parametrize("left", [True, False])
def test_no_net_in_the_goal_mouth_only_beyond_the_crossbar(left):
    pitch, cam, surface, renderer = _drawn_pitch()
    g = renderer._goal_px(pitch, left)
    post_w = renderer._goal_post_px()
    x_a, x_b = sorted((g.line_x, g.bar_x))
    # Interior of the mouth (clear of posts, goal line and crossbar): one flat tint.
    mouth = {
        tuple(surface.get_at((x, y)))[:3]
        for x in range(x_a + post_w, x_b - post_w + 1)
        for y in range(g.y_top + post_w + 1, g.y_bot - post_w)
    }
    assert len(mouth) == 1, f"net hatch is showing in the goal mouth: {mouth}"
    # Behind the crossbar the roof net IS drawn (more than one colour: hatch + grass).
    n_a, n_b = sorted((g.bar_x, g.back_x))
    roof = {
        tuple(surface.get_at((x, y)))[:3]
        for x in range(n_a + post_w, n_b - post_w + 1)
        for y in range(g.y_top + post_w + 1, g.y_bot - post_w)
    }
    assert len(roof) > 1


@pytest.mark.parametrize(
    "x, y, z, expected",
    [
        (-55.0, 0.0, 1.0, True),     # in the goal, under the bar: going in
        (-55.0, 0.0, 3.0, False),    # above the bar: going over
        (-55.0, 0.0, 2.43, True),    # just under the bar
        (-55.0, 0.0, 2.44, False),   # at bar height
        (-52.0, 0.0, 1.0, False),    # not past the goal line yet
        (-55.0, 5.0, 1.0, False),    # wide of the posts
        (-60.0, 0.0, 1.0, False),    # beyond the back of the net
        (55.0, 0.0, 1.0, True),      # mirror: right goal
        (55.0, 0.0, 3.0, False),
    ],
)
def test_ball_under_goal_frame(x, y, z, expected):
    from footballcoach.ui.renderer import ball_under_goal_frame

    assert ball_under_goal_frame(Pitch.standard(), x, y, z) is expected


def test_crossbar_is_over_a_ball_going_in_and_under_a_ball_going_over():
    import pygame

    from footballcoach.entities.ball import Ball
    from footballcoach.mathutils import Vector3
    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()

    def render(ball):
        cam = Camera.fit_to_pitch(pitch)
        surface = pygame.Surface((cam.screen_width, cam.screen_height))
        renderer = Renderer(cam)
        cam.set_zoom_level(4.0)
        cam.follow(-pitch.half_length - 0.6, 0.0)
        if ball is None:
            renderer.draw_pitch(surface, pitch)
        else:
            renderer.draw_pitch_and_ball(surface, pitch, ball)
        return renderer, surface

    renderer, frame_only = render(None)
    g = renderer._goal_px(pitch, True)
    half = renderer._goal_post_px() // 2
    bar = pygame.Rect(g.bar_x - half, g.y_top - half, renderer._goal_post_px(), g.y_bot - g.y_top + renderer._goal_post_px())
    bar_world_x = -pitch.half_length - renderer._crossbar_lean_m

    def changed_in_bar(z):
        _, surface = render(Ball.at_rest(Vector3(bar_world_x - 0.12, 1.5, z)))
        return sum(
            1 for x in range(bar.left, bar.right) for y in range(bar.top, bar.bottom)
            if surface.get_at((x, y)) != frame_only.get_at((x, y))
        )

    assert changed_in_bar(1.0) == 0      # under the bar: the bar is drawn over the ball
    assert changed_in_bar(3.2) > 0       # over the bar: the ball is drawn over the bar


# ---------------------------------------------------------------------------
# Goal post thickness; ball spin dots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zoom", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_posts_and_crossbar_are_clearly_thicker_than_the_net_lines(zoom):
    import pygame

    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import _GOAL_NET_LINE_PX, Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    cam.set_zoom_level(zoom)
    assert renderer._goal_post_px() >= _GOAL_NET_LINE_PX + 2


@pytest.mark.parametrize("zoom", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_goal_net_border_is_the_same_thickness_as_the_net_mesh(zoom):
    """The net's border (back line) used to be drawn at the zoom-scaled pitch
    line width -- 4px at 4x zoom -- while the mesh stayed 1px. Measured here as
    the contiguous run of solid line pixels across the back line, which the
    (translucent, ~180) mesh hatch alone never reaches."""
    import pygame

    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import _GOAL_NET_LINE_PX, Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    cam.set_zoom_level(zoom)
    cam.follow(-pitch.half_length + 1.0, 0.0)
    renderer.draw_pitch(surface, pitch)

    g = renderer._goal_px(pitch, True)
    row = cam.world_to_screen(0.0, 0.0)[1]

    def solid(x):
        return min(surface.get_at((x, row))[:3]) >= _LINE_MIN_CHANNEL

    assert solid(g.back_x)
    left = right = g.back_x
    while solid(left - 1):
        left -= 1
    while solid(right + 1):
        right += 1
    assert right - left + 1 == _GOAL_NET_LINE_PX == 1


_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _solid_runs(values):
    """Number of contiguous runs of True in a sequence."""
    runs, prev = 0, False
    for v in values:
        if v and not prev:
            runs += 1
        prev = v
    return runs


def _dot_has_holes(layer) -> bool:
    """A single dot is convex, so on every row and column its dark ('solid')
    pixels must form ONE contiguous run; a light pixel between two dark ones
    is a hole/notch. Judged as the dot looks composited over a white ball."""
    import pygame

    white = pygame.Surface(layer.get_size())
    white.fill((255, 255, 255))
    white.blit(layer, (0, 0))
    w, h = white.get_size()
    solid = [[white.get_at((x, y))[0] < 150 for x in range(w)] for y in range(h)]
    return any(_solid_runs(row) > 1 for row in solid) or any(
        _solid_runs([solid[y][x] for y in range(h)]) > 1 for x in range(w)
    )


@pytest.mark.parametrize("radius_px", [8, 14, 24, 40, 70])
@pytest.mark.parametrize("dot", [
    (0.0, 0.0, 1.0),                 # facing the camera: a full circle
    (0.6, 0.0, 0.8),
    (0.0, -0.7, 0.714),
    (0.8, 0.3, 0.52),
    (0.93, 0.1, 0.35),               # near the silhouette: strongly foreshortened
])
def test_ball_spin_dot_is_solid_without_light_holes(radius_px, dot):
    """Regression: the dots were drawn with gfxdraw.filled_polygon + aapolygon
    on a transparent layer, which left near-white pixels (measured 210-248 vs a
    ~85 interior) at the polygon's extreme vertices, inside the dark dot."""
    _, _, _, renderer = _drawn_pitch()
    n = math.sqrt(sum(c * c for c in dot))
    unit = tuple(c / n for c in dot)
    layer, _pad = renderer._ball_dots_layer(radius_px, orientation=_IDENTITY, dot_positions=[unit])
    assert not _dot_has_holes(layer)


def test_ball_spin_dot_interior_is_the_configured_dark_colour():
    """The dot colour is (30,30,30) at alpha 220, i.e. ~62 over a white ball;
    the old double alpha-blend gave a lighter ~85."""
    import pygame

    _, _, _, renderer = _drawn_pitch()
    layer, pad = renderer._ball_dots_layer(40, orientation=_IDENTITY, dot_positions=[(0.0, 0.0, 1.0)])
    white = pygame.Surface(layer.get_size())
    white.fill((255, 255, 255))
    white.blit(layer, (0, 0))
    centre = white.get_at((pad, pad))[0]
    assert centre == pytest.approx(30 * 220 / 255 + 255 * (1 - 220 / 255), abs=3)


# ---------------------------------------------------------------------------
# Goal net: a diamond lattice at every zoom, never a checkerboard
# ---------------------------------------------------------------------------

_GRASS = tuple(style.PITCH_GREEN)  # the pitch green


def _net_pixels(zoom: float, *, min_spacing_px: int | None = None):
    """The roof-net interior pixels at *zoom* (clear of posts, bar and border),
    and the gap used."""
    import pygame

    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    if min_spacing_px is not None:
        renderer._goal_net_min_spacing_px = min_spacing_px
    cam.set_zoom_level(zoom)
    cam.follow(-pitch.half_length + 1.0, 0.0)
    renderer.draw_pitch(surface, pitch)

    g = renderer._goal_px(pitch, True)
    post_w = renderer._goal_post_px()
    a, b = sorted((g.bar_x, g.back_x))
    xs = range(a + post_w // 2 + 2, b - 1)
    ys = range(g.y_top + post_w // 2 + 2, g.y_bot - post_w // 2 - 1)
    return [tuple(surface.get_at((x, y))[:3]) for x in xs for y in ys], renderer._goal_net_gap_px()


def _net_ink(zoom: float, *, min_spacing_px: int | None = None) -> tuple[float, int]:
    """Mean lightening of the net region over plain grass, 0..1 -- the fraction
    of "full-strength hatch" pixels. Anti-aliasing spreads a line over more
    pixels at lower intensity but preserves this total, so it stays a fair
    density measure (a binary lit/unlit pixel count does not). Two 1px diagonal
    families give ~0.2 at an 8px gap; a 3px-gap checkerboard is ~0.45."""
    pixels, gap = _net_pixels(zoom, min_spacing_px=min_spacing_px)
    ink = sum(min(p) - _GRASS[0] for p in pixels) / len(pixels) / (255 - _GRASS[0])
    return ink, gap


@pytest.mark.parametrize("zoom", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
def test_goal_net_is_a_sparse_diamond_lattice_at_every_zoom(zoom):
    ink, gap = _net_ink(zoom)
    # 7px is the floor chosen by eye from 6-16px sweeps at 1x (~5px is still blobby); the real
    # guard against a checkerboard is the ink measure below (~0.17-0.20 at 7px vs ~0.45 for a 3px checker).
    assert gap >= 7
    assert ink < 0.30, f"net looks like a checkerboard at zoom {zoom} (gap {gap}px, ink {ink:.2f})"


def test_net_checkerboard_metric_actually_detects_the_old_behaviour():
    """Sanity check on the metric above: with the gap floor removed, the default
    window's 3px gap is a checkerboard and the ink check would fail."""
    ink, gap = _net_ink(1.0, min_spacing_px=2)
    assert gap <= 4
    assert ink > 0.35


# ---------------------------------------------------------------------------
# Anti-aliasing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zoom", [1.0, 3.0])
def test_goal_net_mesh_is_anti_aliased_without_a_dark_halo(zoom):
    """The mesh was plain aliased `draw.line` (2-3 distinct colours in the net
    region); now supersampled + smoothscaled (a dozen). And the supersampled
    layer needs an opaque-colour/transparent background: with the default
    transparent BLACK, smoothscale averaged black into every edge pixel, so
    ~26% of net pixels came out darker than the grass under them."""
    pixels, _ = _net_pixels(zoom)
    # An aliased mesh has 2 colours at 1x and 3 at higher zooms (grass + one hatch level); the
    # mirrored, supersampled mesh is very regular at 1x (grass + 3 anti-aliased levels) and richer above.
    assert len({p for p in pixels}) >= 4
    assert not [p for p in pixels if any(p[i] < _GRASS[i] for i in range(3))]


def test_thick_pitch_arcs_are_anti_aliased():
    """At zoom the arcs are >1px thick, which used to take the aliased
    `draw.lines` path (only grass + line white in the region)."""
    import pygame

    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    cam.set_zoom_level(4.0)
    apex_x = -pitch.half_length + pitch.penalty_spot_distance_m + renderer._penalty_arc_radius_m
    cam.follow(apex_x, 0.0)
    renderer.draw_pitch(surface, pitch)
    px, py = cam.world_to_screen(apex_x, 0.0)
    region = {tuple(surface.get_at((x, y))[:3]) for x in range(px - 25, px + 25) for y in range(py - 25, py + 25)}
    assert len(region) >= 6


def test_aa_disc_and_rings_have_no_dark_fringe():
    """Same black-background bug as the net: smoothscale averaged transparent
    black into the edge RGB, dimming the disc's/ring's anti-aliased edge below
    both its own colour and whatever it sits on."""
    import pygame

    _, _, _, renderer = _drawn_pitch()

    # A translucent light disc over grass.
    bg = pygame.Surface((30, 30))
    bg.fill(_GRASS)
    bg.blit(renderer._aa_disc((255, 255, 255), 8, 200), (4, 4))
    pixels = [tuple(bg.get_at((x, y))[:3]) for x in range(30) for y in range(30)]
    assert len(set(pixels)) >= 4  # actually anti-aliased
    assert not [p for p in pixels if any(p[i] < _GRASS[i] for i in range(3))]

    # A coloured ring over grass: no channel may fall below both ring and grass.
    ring = (180, 230, 255)
    bg = pygame.Surface((60, 60))
    bg.fill(_GRASS)
    renderer._draw_ring(bg, ring, (30, 30), 14, 2)
    floor = tuple(min(ring[i], _GRASS[i]) for i in range(3))
    dark = [p for x in range(60) for y in range(60)
            if any((p := tuple(bg.get_at((x, y))[:3]))[i] < floor[i] - 1 for i in range(3))]
    assert not dark


# ---------------------------------------------------------------------------
# Back of the net (same parallax as the crossbar) and corner flags
# ---------------------------------------------------------------------------

def _drawn_pitch_at_zoom(
    zoom: float, follow=(-52.5 + 1.0, 0.0), *, no_roof=False, no_wall=False, renderer_attrs=None,
):
    """Render the pitch at *zoom*; ``no_roof`` / ``no_wall`` suppress the roof net /
    back-wall net so each layer can be measured on its own; ``renderer_attrs`` are set on the
    Renderer before drawing."""
    import pygame

    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    for name, value in (renderer_attrs or {}).items():
        setattr(renderer, name, value)
    cam.set_zoom_level(zoom)
    cam.follow(*follow)
    if no_roof:
        renderer._draw_goal_net = lambda *a, **k: None
    if no_wall:
        renderer._draw_goal_back_net = lambda *a, **k: None
    renderer.draw_pitch(surface, pitch)
    return pitch, cam, surface, renderer


def _bright(surface, x, y):
    return min(surface.get_at((x, y))[:3]) >= _LINE_MIN_CHANNEL


@pytest.mark.parametrize("left", [True, False])
def test_back_of_net_ground_line_and_top_edge_are_separate_lines(left):
    """The back of the net is drawn with the crossbar's parallax: a line where
    it meets the ground, and a separate thin line for its top edge displaced
    outward by the same lean, with nothing solid between them."""
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(3.0, (-51.5 if left else 51.5, 0.0))
    g = renderer._goal_px(pitch, left)
    row = cam.world_to_screen(0.0, 0.0)[1]
    lean_px = abs(g.back_top_x - g.back_x)
    assert lean_px >= 4
    # Displaced outward (away from the pitch), like the crossbar.
    assert (g.back_top_x < g.back_x) if left else (g.back_top_x > g.back_x)
    assert _bright(surface, g.back_x, row) and _bright(surface, g.back_top_x, row)
    lo, hi = sorted((g.back_x, g.back_top_x))
    assert not any(_bright(surface, x, row) for x in range(lo + 1, hi))
    # The top edge is the same displacement as the crossbar's -- to within the 1px that
    # world->screen integer truncation of the two endpoints can add or remove.
    assert abs(abs(g.back_top_x - g.back_x) - abs(g.bar_x - g.line_x)) <= 1


def test_back_wall_strip_is_tinted_more_than_the_roof_net_in_front_of_it():
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0)
    g = renderer._goal_px(pitch, True)
    post_w = renderer._goal_post_px()
    rows = range(g.y_top + post_w // 2 + 2, g.y_bot - post_w // 2 - 1)

    def ink(x_lo, x_hi):
        px = [surface.get_at((x, y))[:3] for x in range(x_lo, x_hi) for y in rows]
        return sum(min(p) - _GRASS[0] for p in px) / len(px) / (255 - _GRASS[0])

    back = ink(g.back_top_x + 2, g.back_x - 1)              # back wall strip (roof mesh + wall tint)
    roof = ink(g.back_x + 2, g.bar_x - post_w // 2 - 1)     # roof mesh only
    assert back > roof + 0.06


def test_defending_marker_sits_beyond_the_top_back_edge_of_the_net():
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0)
    g = renderer._goal_px(pitch, True)
    row = cam.world_to_screen(0.0, 0.0)[1]
    x = g.back_top_x - 1
    while tuple(surface.get_at((x, row))[:3]) == tuple(style.PITCH_GREEN):
        x -= 1
    # First non-grass pixel outward of the net is the marker, its 0.3m gap clear of the top edge.
    assert g.back_top_x - x >= round(0.3 * cam.pixels_per_metre) - 2


@pytest.mark.parametrize("sx", [-1, 1])
@pytest.mark.parametrize("sy", [-1, 1])
def test_corner_flag_pole_leans_away_from_the_centre_and_the_cloth_is_a_real_triangle(sx, sy):
    from footballcoach.ui.renderer import corner_flag_world_points

    pitch = Pitch.standard()
    lean, width, drop = 0.6, 0.55, 0.55
    geo = corner_flag_world_points(pitch, sx, sy, pole_lean_m=lean, flag_width_m=width, flag_drop_frac=drop)
    cx, cy = sx * pitch.half_length, sy * pitch.half_width
    assert geo.base == (cx, cy)

    # Top: exactly `lean` metres farther from the pitch centre, along the centre->corner line.
    norm = math.hypot(cx, cy)
    ux, uy = cx / norm, cy / norm
    assert geo.top == pytest.approx((cx + ux * lean, cy + uy * lean))
    assert math.hypot(*geo.top) > math.hypot(cx, cy)
    # Cloth attached along the top `drop` of the pole, leaving the rest of the pole bare.
    assert math.dist(geo.attach, geo.top) == pytest.approx(lean * drop)
    assert math.dist(geo.attach, geo.base) == pytest.approx(lean * (1 - drop))

    # Free end: `width` off the pole, PERPENDICULAR to it (not along the touchline,
    # which made a needle), on the side toward the pitch's middle along x.
    mid = ((geo.top[0] + geo.attach[0]) / 2, (geo.top[1] + geo.attach[1]) / 2)
    off = (geo.tip[0] - mid[0], geo.tip[1] - mid[1])
    assert math.hypot(*off) == pytest.approx(width)
    assert off[0] * ux + off[1] * uy == pytest.approx(0.0, abs=1e-9)
    assert -sx * off[0] > 0


def test_corner_flag_pole_and_cloth_are_drawn():
    from footballcoach.ui.renderer import corner_flag_world_points

    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(5.0, (-52.5 + 3.0, 34.0 - 3.0))
    geo = renderer._corner_flag_geometry(pitch, -1, 1)
    # A point on the bare lower pole is pole-coloured.
    lower = ((geo.base[0] + geo.attach[0]) / 2, (geo.base[1] + geo.attach[1]) / 2)
    got = surface.get_at(cam.world_to_screen(*lower))[:3]
    assert all(abs(got[i] - style.CORNER_FLAG_POLE_COLOUR[i]) <= 25 for i in range(3)), got
    # The cloth's centroid is flag-coloured.
    cen = tuple(sum(pt[i] for pt in (geo.top, geo.attach, geo.tip)) / 3 for i in (0, 1))
    got = surface.get_at(cam.world_to_screen(*cen))[:3]
    assert all(abs(got[i] - style.CORNER_FLAG_COLOUR[i]) <= 25 for i in range(3)), got


@pytest.mark.parametrize("rise_px, length_px, gap_px", [(22.0, 6.0, 8.0), (88.0, 26.0, 24.0), (110.0, 32.0, 30.0)])
def test_back_wall_net_segments_are_steep_mirrored_diagonals(rise_px, length_px, gap_px):
    """The back wall is vertical: the parallax maps its height to the short strip's
    x and its width to screen y, so a 45-degree mesh on it becomes two families of
    STEEP lines (rising `rise_px` over `length_px` >> a 45-degree lattice), mirror
    images of each other, parallel lines `gap_px` apart in y."""
    from footballcoach.ui.renderer import back_wall_net_segments

    height = 200
    segs = back_wall_net_segments(length_px, rise_px, height, gap_px)
    assert segs
    slope = rise_px / length_px
    assert slope > 2.0  # much steeper than the roof's 45 degrees
    down = [s for s in segs if s[1][1] > s[0][1]]   # y increases with x
    up = [s for s in segs if s[1][1] < s[0][1]]
    assert len(down) == len(up)
    for (x0, y0), (x1, y1) in segs:
        assert (x0, x1) == (0.0, length_px)
        assert abs((y1 - y0) / (x1 - x0)) == pytest.approx(slope)
    # Parallel lines are gap_px apart (measured along y at the ground edge).
    ys = sorted(s[0][1] for s in down)
    assert all(b - a == pytest.approx(gap_px) for a, b in zip(ys, ys[1:]))
    # The families are mirror images about the strip's centre line.
    assert sorted((s[0][1], s[1][1]) for s in down) == sorted((s[1][1], s[0][1]) for s in up)
    # And together they cover the whole strip height.
    assert min(min(s[0][1], s[1][1]) for s in segs) <= 0
    assert max(max(s[0][1], s[1][1]) for s in segs) >= height


@pytest.mark.parametrize("left", [True, False])
def test_back_wall_net_is_steep_and_the_roof_net_is_drawn_over_it(left):
    """The back-wall strip carries its own steep mesh (measured on its own), AND the roof
    net -- at crossbar height, so it extends to the top-back edge -- is drawn over it,
    so the combined strip shows both layers."""
    follow = (-51.5 if left else 51.5, 0.0)
    pitch, cam, full, renderer = _drawn_pitch_at_zoom(4.0, follow)
    _, _, wall_only, _ = _drawn_pitch_at_zoom(4.0, follow, no_roof=True)
    _, _, roof_only, _ = _drawn_pitch_at_zoom(4.0, follow, no_wall=True)
    g = renderer._goal_px(pitch, left)
    lo, hi = sorted((g.back_x, g.back_top_x))
    rows = range(g.y_top + 6, g.y_bot - 5)

    def x_over_y(surface, x_lo, x_hi):
        """Brightness variation along x vs along y: ~1 for a 45-degree lattice, >>1 for steep lines."""
        ex = ey = 0
        for x in range(x_lo, x_hi):
            for y in rows:
                v = min(surface.get_at((x, y))[:3])
                ex += abs(v - min(surface.get_at((x + 1, y))[:3]))
                ey += abs(v - min(surface.get_at((x, y + 1))[:3]))
        return ex / ey

    def ink(surface):
        px = [surface.get_at((x, y))[:3] for x in range(lo + 2, hi - 1) for y in rows]
        return sum(min(p) - _GRASS[0] for p in px) / len(px) / (255 - _GRASS[0])

    roof_region = (min(g.bar_x, g.back_x) + 8, max(g.bar_x, g.back_x) - 8)
    roof_ratio = x_over_y(full, *roof_region)
    wall_ratio = x_over_y(wall_only, lo + 2, hi - 2)
    full_ratio = x_over_y(full, lo + 2, hi - 2)

    assert len({tuple(full.get_at((x, y))[:3]) for x in range(lo + 2, hi - 1) for y in rows}) >= 8
    assert roof_ratio == pytest.approx(1.0, abs=0.35)      # roof: 45-degree diamonds
    assert wall_ratio > 2.0 * roof_ratio                    # the wall's own mesh: steep lines
    # Roof drawn over the wall: the combined strip is a mix of the two orientations,
    # and carries more ink than the wall mesh alone.
    assert roof_ratio + 0.25 < full_ratio < wall_ratio - 0.25
    assert ink(full) > ink(wall_only) + 0.03
    # And the roof mesh really does reach the strip (roof-only render has ink there too).
    assert ink(roof_only) > ink(_drawn_pitch_at_zoom(4.0, follow, no_roof=True, no_wall=True)[2]) + 0.03


# ---------------------------------------------------------------------------
# The two diagonal families of the net mesh are exact mirror images
# ---------------------------------------------------------------------------

def _alpha_rows(layer):
    w, h = layer.get_size()
    return [[layer.get_at((x, y))[3] for x in range(w)] for y in range(h)]


@pytest.mark.parametrize("zoom", [1.0, 2.0, 4.0])
def test_net_mesh_diagonal_families_are_exact_mirror_images(zoom):
    """Regression: `pygame.draw.line` rasterises through pixel centres, offset (+0.5, +0.5) from its
    coordinates -- along the line for one diagonal but perpendicular to it for the other -- so with
    both families drawn directly the up-right diagonals came out a different pixel profile (a
    single mirrored pair measured: a narrow 3-pixel core vs a 2-pixel split half a pixel off, i.e.
    wider and dimmer). Only one family is rasterised now; the other is its exact mirror, so the
    flat roof layer is left-right symmetric, and the back-wall layer and the SAGGING roof layer (the
    sag displaces points along x, so only a vertical mirror keeps the families alike) are top-bottom
    symmetric, to the pixel."""
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(zoom)
    roof = [layer for key, layer in renderer._net_layer_cache.items() if key[0] == "roof-sag"]
    back = [layer for key, layer in renderer._net_layer_cache.items() if key[0] == "back"]
    assert roof and back
    for layer in roof + back:
        rows = _alpha_rows(layer)
        assert rows == rows[::-1]

    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(zoom, renderer_attrs={"_goal_net_sag_m": 0.0})
    flat = [layer for key, layer in renderer._net_layer_cache.items() if key[0] == "roof"]
    assert flat and not [k for k in renderer._net_layer_cache if k[0] == "roof-sag"]
    for layer in flat:
        rows = _alpha_rows(layer)
        assert rows == [row[::-1] for row in rows]


def test_net_mesh_layers_are_cached_between_frames():
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(1.0)
    before = dict(renderer._net_layer_cache)
    assert before
    renderer.draw_pitch(surface, pitch)
    assert renderer._net_layer_cache.keys() == before.keys()
    assert all(renderer._net_layer_cache[k] is before[k] for k in before)


# ---------------------------------------------------------------------------
# Round, shaded goal frame
# ---------------------------------------------------------------------------

def test_light_comes_from_the_top_left_so_bars_are_lit_on_their_top_and_left():
    from footballcoach.ui.renderer import cylinder_shades

    for n in (3, 4, 8, 12):
        post = cylinder_shades(n, "y", 0.7)   # a post: width runs down the screen
        bar = cylinder_shades(n, "x", 0.7)    # the crossbar: width runs across the screen
        # the darkest strip is on the far (bottom / right) side, and it is clearly darker
        assert post[-1] == min(post) and post[-1] < 0.8 * max(post)
        assert bar[-1] == min(bar) and bar[-1] < 0.8 * max(bar)
        # brightest strip is on the lit half
        assert post.index(max(post)) <= n // 2 and bar.index(max(bar)) <= n // 2


def test_cylinder_shading_is_smooth_and_spans_a_real_range():
    from footballcoach.ui.renderer import cylinder_shades

    shades = cylinder_shades(12, "y", 0.7)
    peak = shades.index(max(shades))
    # falls monotonically from the highlight to the shadow edge
    assert all(a >= b for a, b in zip(shades[peak:], shades[peak + 1:]))
    assert max(shades) > 0.95 and min(shades) < 0.7


def test_zero_shade_strength_is_flat_and_never_exceeds_full_brightness():
    from footballcoach.ui.renderer import cylinder_shades

    assert cylinder_shades(6, "y", 0.0) == [1.0] * 6
    for strength in (0.3, 0.7, 1.0):
        assert all(0.0 < f <= 1.0 for f in cylinder_shades(9, "x", strength))


@pytest.mark.parametrize("left", [True, False])
def test_posts_and_crossbar_are_drawn_shaded_on_their_outside(left):
    """The frame is lit as if from a point above the middle of the goal/pitch: rows across a post
    (columns across the crossbar) are a gradient, the shadow is on the OUTSIDE of both posts
    (top post shaded on top, bottom post on the bottom) and of the crossbar (the side away from
    the pitch), and the two posts are exact mirror images of each other."""
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0, (-51.5 if left else 51.5, 0.0))
    g = renderer._goal_px(pitch, left)
    post_w = renderer._goal_post_px()
    half = post_w // 2
    mid_x = (g.line_x + g.bar_x) // 2

    def col(y0):
        return [surface.get_at((mid_x, y0 - half + i))[0] for i in range(post_w)]

    top, bot = col(g.y_top), col(g.y_bot)
    assert top == bot[::-1]                # mirror images of each other
    assert len(set(top)) >= 4              # a gradient, not a strip
    assert top[-1] > top[0] + 40           # top post: lit on its inner (bottom) edge, shaded on top
    assert bot[0] > bot[-1] + 40           # bottom post: lit on its inner (top) edge, shaded below

    mid_y = (g.y_top + g.y_bot) // 2
    bar = [surface.get_at((g.bar_x - half + i, mid_y))[0] for i in range(post_w)]
    assert len(set(bar)) >= 4
    outer, inner = (bar[0], bar[-1]) if left else (bar[-1], bar[0])   # outside = away from the pitch
    assert inner > outer + 40


def test_goal_frame_shading_can_be_turned_off():
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0)
    renderer._goal_shade_strength = 0.0
    renderer.draw_pitch(surface, pitch)
    g = renderer._goal_px(pitch, True)
    post_w = renderer._goal_post_px()
    mid_x = (g.line_x + g.bar_x) // 2
    rows = {tuple(surface.get_at((mid_x, g.y_top - post_w // 2 + i))[:3]) for i in range(post_w)}
    assert rows == {tuple(style.PITCH_LINE_WHITE)}


def test_shaded_foot_is_a_round_antialiased_dome_lit_from_the_top_left():
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0)
    r = renderer._goal_post_px() // 2 + 1
    foot = renderer._shaded_foot(r, tuple(style.PITCH_LINE_WHITE), 0.7)
    size = 2 * r + 3
    assert foot.get_size() == (size, size)
    alphas = [[foot.get_at((x, y))[3] for x in range(size)] for y in range(size)]
    assert alphas[r + 1][r + 1] == 255 and alphas[0][0] == 0           # solid core, clear corner
    assert any(0 < a < 255 for row in alphas for a in row)               # anti-aliased rim
    # brighter toward the light (up-left) than away from it (down-right), at equal radius
    up_left = foot.get_at((r + 1 - r // 2, r + 1 - r // 2))[0]
    down_right = foot.get_at((r + 1 + r // 2, r + 1 + r // 2))[0]
    assert up_left > down_right


def test_post_and_crossbar_shade_identically_so_their_corners_can_mitre():
    from footballcoach.ui.renderer import cylinder_shades

    for n in (3, 6, 11):
        assert cylinder_shades(n, "x", 0.7) == pytest.approx(cylinder_shades(n, "y", 0.7))


@pytest.mark.parametrize("left", [True, False])
def test_crossbar_corners_are_mitred_so_the_shadow_wraps_from_post_to_bar(left):
    """At each end of the crossbar the shadow continues from the post: the corner square is
    shaded by distance from the NEARER outer edge, so the outer corner is the darkest pixel, the
    outer edge is dark along both the bar and the post, and the inner corner is lit."""
    from footballcoach.ui.renderer import cylinder_shades

    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0, (-51.5 if left else 51.5, 0.0))
    g = renderer._goal_px(pitch, left)
    w = renderer._goal_post_px()
    half = w // 2

    def px(i, j, top_corner):        # i: column across the bar, j: row across the post
        y0 = (g.y_top if top_corner else g.y_bot) - half
        return surface.get_at((g.bar_x - half + i, y0 + j))[0]

    for top_corner in (True, False):
        def d(a, b):                 # pixel at distance (a, b) from the outer x / outer y edge
            i = a if left else w - 1 - a
            j = b if top_corner else w - 1 - b
            return px(i, j, top_corner)

        assert d(0, 0) == min(d(a, b) for a in range(w) for b in range(w))          # outer corner darkest
        assert d(w - 1, w - 1) > d(0, 0) + 40                                       # inner corner lit
        # every corner pixel is the bar/post profile at its distance from the NEARER outer edge
        prof = cylinder_shades(w, "x", renderer._goal_shade_strength, flip=True)
        for a in range(w):
            for b in range(w):
                assert d(a, b) == renderer._shaded(tuple(style.PITCH_LINE_WHITE), prof[min(a, b)])[0]


def test_flipping_the_light_mirrors_a_bar_and_the_foot():
    from footballcoach.ui.renderer import cylinder_shades

    for across in ("x", "y"):
        assert cylinder_shades(7, across, 0.7, flip=True) == pytest.approx(cylinder_shades(7, across, 0.7)[::-1])
    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0)
    r = renderer._goal_post_px() // 2 + 1
    white = tuple(style.PITCH_LINE_WHITE)
    base = renderer._shaded_foot(r, white, 0.7)
    flipped = renderer._shaded_foot(r, white, 0.7, flip_x=True, flip_y=True)
    size = 2 * r + 3
    # flipping both axes of the light == rotating the dome by 180 degrees
    for x, y in ((r, r), (r + 2, r + 1), (1, size - 2)):
        assert flipped.get_at((x, y)) == base.get_at((size - 1 - x, size - 1 - y))


# ---------------------------------------------------------------------------
# Goal foot, roof-net sag, ball shadow, ball shading, turf
# ---------------------------------------------------------------------------

def test_goal_foot_is_small_at_1x_and_unchanged_when_zoomed():
    _, _, _, r1 = _drawn_pitch()
    assert r1._goal_post_px() == 3
    assert r1._goal_foot_radius() < 2.0                       # was radius 2 (a 5px blob on a 3px post)
    assert r1._goal_foot_radius() >= r1._goal_post_px() / 2   # but never narrower than the post
    _, _, _, r4 = _drawn_pitch_at_zoom(4.0)
    w = r4._goal_post_px()
    assert r4._goal_foot_radius() == pytest.approx(w / 2 + 1.0)   # proud by a full pixel once zoomed


@pytest.mark.parametrize("left", [True, False])
def test_goal_foot_is_drawn_under_the_post(left):
    """The post covers the foot: where they overlap (the foot's centre pixel, on the post's end),
    the pixel is the post's own shade, not the dome's."""
    from footballcoach.ui.renderer import cylinder_shades

    pitch, cam, surface, renderer = _drawn_pitch_at_zoom(4.0, (-51.5 if left else 51.5, 0.0))
    g = renderer._goal_px(pitch, left)
    w = renderer._goal_post_px()
    half = w // 2
    white = tuple(style.PITCH_LINE_WHITE)
    for y, top_post in ((g.y_top, True), (g.y_bot, False)):
        rows = cylinder_shades(w, "y", renderer._goal_shade_strength, flip=top_post)
        for i in range(w):
            assert tuple(surface.get_at((g.line_x, y - half + i)))[:3] == renderer._shaded(white, rows[i])
    # ...while the dome still shows beyond the post's end (on the pitch side of the goal line)
    beyond = tuple(surface.get_at((g.line_x + half * (1 if left else -1), g.y_top)))[:3]
    assert beyond != tuple(style.PITCH_GREEN)


def test_roof_net_sags_toward_the_pitch_and_is_pinned_at_the_frame():
    import pygame

    def roof(sag):
        pitch, cam, surface, renderer = _drawn_pitch_at_zoom(
            5.0, no_wall=True, renderer_attrs={"_goal_net_sag_m": sag},
        )
        g = renderer._goal_px(pitch, True)
        x_a, x_b = sorted((g.bar_x, g.back_top_x))
        return surface, g, x_a, x_b, renderer

    flat, g, x_a, x_b, _ = roof(0.0)
    sag, _, _, _, renderer = roof(0.3)
    post_w = renderer._goal_post_px()
    mid_y = (g.y_top + g.y_bot) // 2

    def diff(x_lo, x_hi, y_lo, y_hi):
        return sum(
            abs(a - b) for x in range(x_lo, x_hi) for y in range(y_lo, y_hi)
            for a, b in zip(flat.get_at((x, y))[:3], sag.get_at((x, y))[:3])
        )

    # the mesh differs in the middle of the roof...
    assert diff(x_a + 6, x_b - 6, mid_y - 60, mid_y + 60) > 5000
    # ...but not along the top-back edge or the frame rows, where the displacement is zero
    assert diff(x_b - 1, x_b + 1, g.y_top + post_w, g.y_bot - post_w) == 0
    # the mesh is still a translucent lattice: it never darkens the grass
    grass = tuple(style.PITCH_GREEN)
    assert all(
        min(sag.get_at((x, y))[i] - grass[i] for i in range(3)) >= 0
        for x in range(x_a + 6, x_b - 6) for y in range(mid_y - 60, mid_y + 60, 3)
    )


def _luma(px):
    return 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]


def _ball_frame(z, x=0.0, y=0.0, zoom=5.0, **attrs):
    import pygame

    from footballcoach.entities.ball import Ball
    from footballcoach.mathutils import Vector3
    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    for k, v in attrs.items():
        setattr(renderer, k, v)
    cam.set_zoom_level(zoom)
    cam.follow(x, y)
    surface.fill(style.PITCH_GREEN)
    ball = Ball()
    ball.position = Vector3(x, y, z)
    ball.velocity = Vector3(0.0, 0.0, 0.0)
    renderer.draw_ball(surface, ball)
    return cam, surface, renderer, ball


def test_ball_is_shaded_like_a_sphere_lit_from_the_pitch_centre_but_keeps_its_dots():
    """The ball's light is the shadow's point light above the pitch centre, so the side of the ball
    facing the centre is the lit one -- whichever side of the pitch the ball is on."""
    def ball_pixels(x, y, **attrs):
        cam, surface, renderer, ball = _ball_frame(0.11, x=x, y=y, **attrs)
        px, py = cam.world_to_screen(ball.position.x, ball.position.y)
        cx, cy = cam.world_to_screen_f(0.0, 0.0)
        return surface, px, py, renderer, ((cx - px) / math.hypot(cx - px, cy - py), (cy - py) / math.hypot(cx - px, cy - py))

    for x, y in ((-25.0, 12.0), (30.0, -15.0)):
        flat, px, py, r, (ux, uy) = ball_pixels(x, y)
        shaded, _, _, rs, _ = ball_pixels(x, y, _ball_shading_strength=0.85)
        radius = int(r.min_ball_radius_px * 5.0)          # the ball's radius at zoom 5 (12px at the default config)
        # the side facing the pitch centre is clearly brighter than the far side (compared as shaded/flat
        # over each half-disc, which cancels the dots)
        def half_ratio(sign):
            num = den = 0.0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i * i + j * j <= (radius - 1) ** 2 and sign * (i * ux + j * uy) > 0:
                        num += _luma(shaded.get_at((px + i, py + j)))
                        den += _luma(flat.get_at((px + i, py + j)))
            return num / den

        assert half_ratio(+1) - half_ratio(-1) > 0.15
        assert half_ratio(+1) > 0.8
        # the whole ball is never brighter than the flat one, and the dots are still clearly dark
        box = range(-radius, radius + 1)
        assert all(_luma(shaded.get_at((px + i, py + j))) <= _luma(flat.get_at((px + i, py + j))) + 1 for i in box for j in box)
        assert min(_luma(shaded.get_at((px + i, py + j))) for i in box for j in box) < 90


def test_ball_shading_sprite_is_anti_aliased_cached_and_disc_shaped():
    _, _, renderer, _ = _ball_frame(0.11, _ball_shading_strength=0.85)
    a = renderer._ball_shade_sprite(9, 45, 40)          # light from the lower-right (screen y is down), 40 deg up
    assert a is renderer._ball_shade_sprite(9, 45, 40)   # cached
    assert a is not renderer._ball_shade_sprite(9, 225, 40)
    size = a.get_width()
    assert size == 2 * 9 + 3
    assert a.get_at((0, 0))[3] == 0                                              # clear corner
    assert a.get_at((size // 2 + 5, size // 2 + 5))[3] < 40                      # lit side
    assert a.get_at((size // 2 - 6, size // 2 - 6))[3] > 100                     # dark side
    assert any(0 < a.get_at((x, size // 2))[3] < 255 for x in range(size))       # soft rim
    # a light straight overhead (the pitch centre) shades evenly all round: bright middle, dark rim
    o = renderer._ball_shade_sprite(9, 0, 90)
    assert o.get_at((size // 2, size // 2))[3] < 20 < 45 < o.get_at((size // 2 + 8, size // 2))[3]
    assert abs(o.get_at((size // 2 + 6, size // 2))[3] - o.get_at((size // 2 - 6, size // 2))[3]) <= 3


def test_ball_light_is_the_shadows_point_light_above_the_pitch_centre():
    import math as m

    for x, y, z in ((0.0, 0.0, 0.11), (-25.0, 12.0, 0.11), (30.0, -15.0, 6.0), (10.0, 5.0, 0.11)):
        cam, _, renderer, ball = _ball_frame(z, x=x, y=y, zoom=1.0)
        az, el = renderer._ball_light_angles(ball)
        H = renderer._light_height_m
        dist = m.hypot(x, y)
        assert abs(el - m.degrees(m.atan2(H - z, dist))) <= 2.6            # quantised to 5 degrees
        if dist > 1.0:
            bx, by = cam.world_to_screen_f(x, y)
            cx, cy = cam.world_to_screen_f(0.0, 0.0)
            want = m.degrees(m.atan2(cy - by, cx - bx)) % 360               # toward the centre, on screen
            assert min(abs(az - want), 360 - abs(az - want)) <= 5.1
    # overhead at the centre, lower toward the edge; a raised ball is nearer the light's height, so it
    # sees the light at a LOWER angle than a ball on the ground at the same spot
    def elevation(z, x, y):
        _, _, renderer, ball = _ball_frame(z, x=x, y=y, zoom=1.0)
        return renderer._ball_light_angles(ball)[1]

    assert elevation(0.11, 0.0, 0.0) == 90
    assert elevation(0.11, 45.0, 0.0) < elevation(0.11, 10.0, 0.0) < 90        # lower the further out
    assert elevation(8.0, 45.0, 0.0) < elevation(0.11, 45.0, 0.0)


def _shadow_pixels(surface, exclude_centre, exclude_r):
    """Grass pixels that have been darkened, excluding a disc (the ball itself)."""
    grass = tuple(style.PITCH_GREEN)
    out = []
    for y in range(surface.get_height()):
        for x in range(surface.get_width()):
            if math.hypot(x - exclude_centre[0], y - exclude_centre[1]) <= exclude_r:
                continue
            p = surface.get_at((x, y))[:3]
            if sum(grass) - sum(p) > 30:
                out.append((x, y, sum(grass) - sum(p)))
    return out


def test_grounded_ball_has_a_thin_contact_shadow_and_it_can_be_turned_off():
    cam, surface, renderer, ball = _ball_frame(0.11, x=-20.0, y=10.0, zoom=5.0, _ball_shadow_enabled=True)
    bx, by = cam.world_to_screen(ball.position.x, ball.position.y)
    r = max(renderer.min_ball_radius_px * cam.zoom_scale, cam.scale_length(ball.radius_m))
    ring = _shadow_pixels(surface, (bx, by), r + 5)   # the ball's own state ring is within ~5px of its edge
    assert ring, "a grounded ball should still show a contact shadow"
    # ...pushed away from the pitch centre (this ball is up-left of it), i.e. the shadow is on the far side
    mx = sum(p[0] for p in ring) / len(ring) - bx
    my = sum(p[1] for p in ring) / len(ring) - by
    cx, cy = cam.world_to_screen_f(0.0, 0.0)
    assert mx * (bx - cx) + my * (by - cy) > 0

    cam, surface, renderer, ball = _ball_frame(0.11, x=-20.0, y=10.0, zoom=5.0)
    assert not _shadow_pixels(surface, (bx, by), r + 5)


def test_raised_ball_shadow_lands_on_the_ground_away_from_the_pitch_centre():
    """A point light H above the pitch centre puts the shadow of a ball at height z, distance d from
    the centre, at d * H / (H - z) from it: displaced radially outward by d * z / (H - z)."""
    x, y, z = 28.0, 16.0, 5.0
    cam, surface, renderer, ball = _ball_frame(z, x=x, y=y, zoom=2.0, _ball_shadow_enabled=True)
    H = renderer._light_height_m
    under = z - ball.radius_m
    bx, by = cam.world_to_screen_f(x, y)
    cx, cy = cam.world_to_screen_f(0.0, 0.0)
    k = under / (H - under)
    expect = (bx + (bx - cx) * k, by + (by - cy) * k)
    r_ball = renderer.min_ball_radius_px * cam.zoom_scale * (1 + z * renderer._ball_height_boost_per_m) + 8
    shadow = [p for p in _shadow_pixels(surface, (bx, by), r_ball) if math.hypot(p[0] - expect[0], p[1] - expect[1]) < 40]
    assert shadow
    total = sum(p[2] for p in shadow)
    sx = sum(p[0] * p[2] for p in shadow) / total
    sy = sum(p[1] * p[2] for p in shadow) / total
    assert math.hypot(sx - expect[0], sy - expect[1]) < 4.0
    assert math.hypot(sx - bx, sy - by) > 4.0 * renderer._light_height_m / 15.0   # visibly detached


def test_ball_shadow_is_fainter_and_softer_as_the_ball_rises():
    _, _, renderer, _ = _ball_frame(0.11)
    low = renderer._ball_shadow_sprite(5.0, 120, 0.2)
    high = renderer._ball_shadow_sprite(5.0, 66, 0.5)
    peak = lambda s: max(s.get_at((x, s.get_height() // 2))[3] for x in range(s.get_width()))
    assert peak(low) > peak(high)
    assert renderer._ball_shadow_sprite(5.0, 120, 0.2) is low        # cached


def _turf_renderer(**attrs):
    import pygame

    from footballcoach.ui.camera import Camera
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    surface = pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    for k, v in {"_turf_enabled": True, **attrs}.items():
        setattr(renderer, k, v)
    return pitch, cam, surface, renderer


def test_turf_is_textured_but_averages_to_the_pitch_green_and_darkens_toward_the_corners():
    import pygame

    pitch, cam, surface, renderer = _turf_renderer()
    renderer._draw_turf(surface, pitch)
    w, h = surface.get_size()
    cx, cy = w // 2, h // 2

    def mean(x, y, n=40):
        px = [surface.get_at((x + i, y + j))[:3] for i in range(-n // 2, n // 2) for j in range(-n // 2, n // 2)]
        return [sum(p[c] for p in px) / len(px) for c in range(3)]

    centre = mean(cx, cy)
    for c in range(3):
        assert abs(centre[c] - style.PITCH_GREEN[c]) < 0.04 * style.PITCH_GREEN[c] + 2   # stays "the" pitch green
    assert len({tuple(surface.get_at((cx + i, cy))[:3]) for i in range(60)}) >= 8          # not a flat fill
    corner = mean(30, 30)
    assert _luma(corner) < 0.95 * _luma(centre)                                            # vignette
    # ...but it is a texture, not a gradient: nowhere near a strong contrast
    lumas = [_luma(surface.get_at((x, y))) for x in range(0, w, 7) for y in range(0, h, 7)]
    assert max(lumas) / min(lumas) < 1.4


def test_turf_can_be_disabled_and_is_deterministic():
    pitch, cam, a, ra = _turf_renderer(_turf_enabled=False)
    ra.draw_pitch(a, pitch)
    assert tuple(a.get_at((300, 5))[:3]) == tuple(style.PITCH_GREEN) == tuple(a.get_at((700, 686))[:3])

    pitch, cam, b1, r1 = _turf_renderer()
    pitch, cam, b2, r2 = _turf_renderer()
    r1._draw_turf(b1, pitch)
    r2._draw_turf(b2, pitch)
    assert all(b1.get_at((x, y)) == b2.get_at((x, y)) for x in range(0, 1000, 37) for y in range(0, 690, 29))


def test_turf_background_is_cached_until_the_camera_moves():
    pitch, cam, surface, renderer = _turf_renderer()
    renderer._draw_turf(surface, pitch)
    first = renderer._turf_bg
    renderer._draw_turf(surface, pitch)
    assert renderer._turf_bg is first
    cam.set_zoom_level(3.0)
    cam.follow(10.0, 5.0)
    renderer._draw_turf(surface, pitch)
    assert renderer._turf_bg is not first


def test_turf_patches_are_anchored_to_the_world_not_the_screen():
    """Pan the camera 6m and the grass at a given WORLD point keeps its colour (patches only: grain
    and vignette, which are screen-space, are switched off here; the patch amplitude is exaggerated so a
    screen-anchored texture could not pass by accident)."""
    samples = []
    for follow in ((0.0, 0.0), (6.0, 3.0)):
        pitch, cam, surface, renderer = _turf_renderer(_turf_noise_amp=0.0, _turf_vignette=0.0, _turf_patch_amp=0.5)
        cam.set_zoom_level(3.0)
        cam.follow(*follow)
        renderer._draw_turf(surface, pitch)
        samples.append([tuple(surface.get_at(tuple(round(v) for v in cam.world_to_screen_f(wx, wy)))[:3])
                        for wx, wy in ((2.0, 1.0), (4.0, -1.0), (7.0, 2.0), (-1.0, 0.0))])
    for a, b in zip(*samples):
        assert all(abs(a[c] - b[c]) <= 2 for c in range(3))
    assert len({p for p in samples[0]}) > 1        # ...and the patches genuinely vary from place to place


def test_roof_sag_displaces_toward_the_pitch_by_the_parallax_of_the_sag_depth():
    """Lower points lean less, so a roof hanging `sag_m` below the crossbar shifts by
    sag_m / goal_height * lean, TOWARD the pitch (+x on screen for the left goal, -x for the right)."""
    calls = {}
    for left in (True, False):
        pitch, cam, surface, renderer = _drawn_pitch_at_zoom(3.0, (-51.5 if left else 51.5, 0.0), no_wall=True)
        seen = []
        renderer._draw_goal_net = lambda *a, **k: seen.append(k["sag_dx_px"])
        renderer._draw_goal_top(surface, pitch, left=left)
        calls[left] = (seen[0], renderer, pitch, cam)
    expected = lambda r, p, c: r._goal_net_sag_m / p.goal_height_m * r._goal_lean_m(p) * c.pixels_per_metre
    dx, r, p, c = calls[True]
    assert dx == pytest.approx(expected(r, p, c)) and dx > 0.5
    dx, r, p, c = calls[False]
    assert dx == pytest.approx(-expected(r, p, c)) and dx < -0.5


# ---------------------------------------------------------------------------
# Corner arcs stay inside the pitch; corner flag size and shadow
# ---------------------------------------------------------------------------

def _whiter_than_grass(px, margin=25):
    return min(px[i] - style.PITCH_GREEN[i] for i in range(3)) > margin


@pytest.mark.parametrize("zoom", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
@pytest.mark.parametrize("corner", [(-1, -1), (-1, 1), (1, -1), (1, 1)])
def test_corner_arcs_never_poke_out_past_the_boundary_lines(zoom, corner):
    """Regression: each arc is a stroke centred on world points that lie on the boundary lines'
    OUTER edge, so its thickness and anti-aliased edge landed 1-6 pixels outside the pitch at
    zoom 1-3. Nothing light may be drawn outside the rectangle the boundary lines are drawn in
    (flags are yellow/orange and shadows dark, so they don't count as 'light')."""
    sx, sy = corner
    pitch = Pitch.standard()
    follow = (0.0, 0.0) if zoom == 1.0 else (sx * (pitch.half_length - 3.0), sy * (pitch.half_width - 3.0))
    _, cam, surface, _ = _drawn_pitch_at_zoom(zoom, follow)
    b0 = cam.world_to_screen(-pitch.half_length, -pitch.half_width)
    b1 = cam.world_to_screen(pitch.half_length, pitch.half_width)
    left, top = min(b0[0], b1[0]), min(b0[1], b1[1])
    right, bottom = left + abs(b1[0] - b0[0]), top + abs(b1[1] - b0[1])       # exclusive, like pygame.draw.rect
    cx, cy = cam.world_to_screen(sx * pitch.half_length, sy * pitch.half_width)
    outside = [
        (x, y)
        for x in range(max(0, cx - 45), min(surface.get_width(), cx + 46))
        for y in range(max(0, cy - 45), min(surface.get_height(), cy + 46))
        if (x < left or x >= right or y < top or y >= bottom) and _whiter_than_grass(surface.get_at((x, y)))
    ]
    assert not outside, outside[:5]


@pytest.mark.parametrize("corner", [(-1, -1), (-1, 1), (1, -1), (1, 1)])
def test_corner_arcs_are_still_drawn_inside_the_pitch(corner):
    """...and clipping did not remove them: every sample along the quarter circle has light pixels
    right on it."""
    sx, sy = corner
    pitch = Pitch.standard()
    _, cam, surface, renderer = _drawn_pitch_at_zoom(3.0, (sx * (pitch.half_length - 3.0), sy * (pitch.half_width - 3.0)))
    pts = corner_arc_world_points(pitch, sx, sy, radius_m=renderer._corner_arc_radius_m)
    hits = 0
    samples = pts[len(pts) // 5: -(len(pts) // 5)]              # the middle of the arc, away from the lines it meets
    for wx, wy in samples:
        px, py = cam.world_to_screen(wx, wy)
        hits += any(_whiter_than_grass(surface.get_at((px + i, py + j))) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2))
    assert samples and hits == len(samples)


def test_the_pennant_keeps_its_size_however_tall_the_pole_is():
    """The cloth's length along the pole is `flag_length_m`, an absolute (apparent) size: a taller pole
    (a bigger parallax lean) must not make the pennant taller too."""
    pitch = Pitch.standard()
    lengths = []
    for pole_h in (2.4, 4.0, 6.0):
        _, _, _, renderer = _drawn_pitch()
        renderer._corner_pole_height_m = pole_h
        geo = renderer._corner_flag_geometry(pitch, -1, 1)
        lengths.append(math.hypot(geo.top[0] - geo.attach[0], geo.top[1] - geo.attach[1]))
    assert lengths == pytest.approx([renderer._corner_flag_length_m] * 3, rel=1e-6)
    # ...while the pole itself does get longer
    poles = []
    for pole_h in (2.4, 4.0):
        _, _, _, renderer = _drawn_pitch()
        renderer._corner_pole_height_m = pole_h
        geo = renderer._corner_flag_geometry(pitch, 1, -1)
        poles.append(math.hypot(geo.top[0] - geo.base[0], geo.top[1] - geo.base[1]))
    assert poles[1] > 1.6 * poles[0]


def test_default_flag_is_taller_but_not_bigger_than_the_old_one():
    _, _, _, renderer = _drawn_pitch()
    assert renderer._corner_pole_height_m >= 4.0
    old_cloth = 0.55 * (0.72 / 2.44 * 2.4)                 # old design: 55% of a 2.4m pole's apparent lean
    assert renderer._corner_flag_length_m < old_cloth       # "a little shorter"
    assert renderer._corner_flag_length_m > 0.6 * old_cloth


def _flag_scene(zoom=3.0, corner=(-1, 1), **attrs):
    sx, sy = corner
    pitch = Pitch.standard()
    _, cam, surface, renderer = _drawn_pitch_at_zoom(
        zoom, (sx * (pitch.half_length - 4.0), sy * (pitch.half_width - 3.0)), renderer_attrs=attrs,
    )
    return pitch, cam, surface, renderer


def _darkness(surface, cam, wx, wy):
    px = surface.get_at(cam.world_to_screen(wx, wy))[:3]
    return sum(style.PITCH_GREEN) - sum(px)


@pytest.mark.parametrize("corner", [(-1, 1), (1, -1), (1, 1), (-1, -1)])
def test_a_corner_flag_casts_a_shadow_straight_away_from_the_pitch_centre(corner):
    """The scene light is a point above the middle of the pitch, so the pole's shadow (height
    `shadow_height_m`) is d * h / (H - h) long from its foot, radially away from the centre -- well
    beyond the drawn pole's tip -- and nothing beside it is shaded."""
    sx, sy = corner
    pitch, cam, surface, renderer = _flag_scene(corner=corner)
    bx, by = sx * pitch.half_length, sy * pitch.half_width
    dist = math.hypot(bx, by)
    ux, uy = bx / dist, by / dist
    length = dist * renderer._corner_shadow_height_m / (renderer._light_height_m - renderer._corner_shadow_height_m)
    assert length > 1.4                                              # long enough to be a real streak
    pole_reach = renderer._goal_lean_m(pitch) / pitch.goal_height_m * renderer._corner_pole_height_m
    beyond_pole = pole_reach + 0.6 * (length - pole_reach)           # on the shadow's axis, past the drawn pole's tip
    assert _darkness(surface, cam, bx + ux * beyond_pole, by + uy * beyond_pole) > 8
    # the same distance out but well off to the side, and just past the shadow's far end: bare grass
    px, py = -uy, ux
    assert _darkness(surface, cam, bx + ux * beyond_pole + px * 1.6, by + uy * beyond_pole + py * 1.6) <= 2
    assert _darkness(surface, cam, bx + ux * (length + 1.5), by + uy * (length + 1.5)) <= 2


def test_flag_shadow_can_be_turned_off_and_shrinks_as_the_light_rises():
    corner = (-1, 1)
    pitch = Pitch.standard()
    bx, by = corner[0] * pitch.half_length, corner[1] * pitch.half_width
    ux, uy = bx / math.hypot(bx, by), by / math.hypot(bx, by)

    def dark_at(metres, **attrs):
        _, cam, surface, _ = _flag_scene(corner=corner, **attrs)
        return _darkness(surface, cam, bx + ux * metres, by + uy * metres)

    assert dark_at(1.6) > 8
    assert dark_at(1.6, _corner_shadow_alpha=0) <= 2
    assert dark_at(1.6, _light_height_m=400.0) <= 2                  # a very high light: the shadow hugs the pole


def test_the_flag_shadow_is_soft_and_never_covers_the_pole_or_cloth():
    pitch, cam, surface, renderer = _flag_scene(zoom=5.0, corner=(-1, 1))
    geo = renderer._corner_flag_geometry(pitch, -1, 1)
    lower = ((geo.base[0] + geo.attach[0]) / 2, (geo.base[1] + geo.attach[1]) / 2)
    got = surface.get_at(cam.world_to_screen(*lower))[:3]
    assert all(abs(got[i] - style.CORNER_FLAG_POLE_COLOUR[i]) <= 25 for i in range(3)), got    # pole colour, not shadowed
    cen = tuple(sum(pt[i] for pt in (geo.top, geo.attach, geo.tip)) / 3 for i in (0, 1))
    got = surface.get_at(cam.world_to_screen(*cen))[:3]
    assert all(abs(got[i] - style.CORNER_FLAG_COLOUR[i]) <= 25 for i in range(3)), got
    # the shadow's edge is anti-aliased: a range of grass tints, not just grass and one dark level
    bx, by = geo.base
    ux, uy = bx / math.hypot(bx, by), by / math.hypot(bx, by)
    tints = {
        tuple(surface.get_at(cam.world_to_screen(bx + ux * t + (-uy) * s, by + uy * t + ux * s))[:3])
        for t in (1.5, 1.7) for s in [i * 0.01 for i in range(-14, 15)]
    }
    assert len(tints) >= 3


@pytest.mark.parametrize("zoom", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("corner", [(-1, -1), (1, 1)])
def test_clipping_the_corner_arcs_removes_only_what_is_outside_the_pitch(zoom, corner):
    """Every light pixel of the UNCLIPPED arc that lies inside the boundary rectangle is still there
    in the real render (a clip that is too tight would leave the arc detached from the lines)."""
    import pygame

    sx, sy = corner
    pitch = Pitch.standard()
    follow = (0.0, 0.0) if zoom == 1.0 else (sx * (pitch.half_length - 3.0), sy * (pitch.half_width - 3.0))
    _, cam, surface, renderer = _drawn_pitch_at_zoom(zoom, follow)
    line_w = max(1, int(0.12 * cam.pixels_per_metre))
    bare = pygame.Surface(surface.get_size())
    bare.fill(style.PITCH_GREEN)
    renderer._draw_world_polyline(bare, corner_arc_world_points(pitch, sx, sy, radius_m=renderer._corner_arc_radius_m), line_w)

    b0 = cam.world_to_screen(-pitch.half_length, -pitch.half_width)
    b1 = cam.world_to_screen(pitch.half_length, pitch.half_width)
    left, top = min(b0[0], b1[0]), min(b0[1], b1[1])
    right, bottom = left + abs(b1[0] - b0[0]), top + abs(b1[1] - b0[1])
    cx, cy = cam.world_to_screen(sx * pitch.half_length, sy * pitch.half_width)
    missing, seen = [], 0
    for x in range(max(left, cx - 45), min(right, cx + 46)):
        for y in range(max(top, cy - 45), min(bottom, cy + 46)):
            if _whiter_than_grass(bare.get_at((x, y)), margin=40):
                seen += 1
                if not _whiter_than_grass(surface.get_at((x, y)), margin=25):
                    missing.append((x, y))
    assert seen > 10 and not missing, missing[:5]


@pytest.mark.parametrize("corner", [(-1, 1), (1, -1)])
def test_the_flag_shadow_is_in_proportion_to_the_drawn_pole(corner):
    """Regression: a physical 2.4m pole under the 40m scene light threw a ~4m shadow at a corner
    (the light is only ~33 degrees up there), 3.4x the ~1.2m stub that is drawn. The shadow may be a
    streak beyond the pole, but not more than about twice its drawn length."""
    sx, sy = corner
    pitch = Pitch.standard()
    _, _, _, renderer = _drawn_pitch()
    geo = renderer._corner_flag_geometry(pitch, sx, sy)
    pole = math.hypot(geo.top[0] - geo.base[0], geo.top[1] - geo.base[1])
    dist = math.hypot(sx * pitch.half_length, sy * pitch.half_width)
    h, light = renderer._corner_shadow_height_m, renderer._light_height_m
    shadow = dist * h / (light - h)
    assert pole < shadow <= 2.0 * pole


@pytest.mark.parametrize("corner", [(-1, 1), (1, 1), (-1, -1), (1, -1)])
def test_the_flag_shadow_starts_at_the_poles_foot_and_not_behind_it(corner):
    """Measured on the pixels: at zoom 5 the shadow is visible either side of the pole all the way
    from its foot outward, is absent just behind the foot (toward the pitch centre), and its far end is
    where d*h/(H-h) says."""
    sx, sy = corner
    pitch, cam, surface, renderer = _flag_scene(zoom=5.0, corner=corner)
    bx, by = sx * pitch.half_length, sy * pitch.half_width
    dist = math.hypot(bx, by)
    ux, uy = bx / dist, by / dist
    px, py = -uy, ux
    length = dist * renderer._corner_shadow_height_m / (renderer._light_height_m - renderer._corner_shadow_height_m)

    def dark_beside(t):        # the shadow line is 0.16m wide and the pole ~0.06m, so 0.055m off-axis is shadow, not pole
        return max(_darkness(surface, cam, bx + ux * t + px * s, by + uy * t + py * s) for s in (-0.055, 0.055))

    assert dark_beside(0.15) > 8 and dark_beside(0.5) > 8                       # right from the foot
    assert dark_beside(-0.5) <= 2 and dark_beside(-1.0) <= 2                     # nothing behind it
    assert dark_beside(length * 0.9) > 8
    assert dark_beside(length + 0.6) <= 2                                        # ends where the formula says
