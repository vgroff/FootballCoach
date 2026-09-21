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


def _has_line_white_near(surface, cam, world_x, world_y, *, dx_px=1, dy_px=0):
    px, py = cam.world_to_screen(world_x, world_y)
    return any(
        min(surface.get_at((px + i, py + j))[:3]) >= _LINE_MIN_CHANNEL
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
              if min(surface.get_at((mid_x, y))[:3]) >= _LINE_MIN_CHANNEL]
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
    from footballcoach.ui.renderer import Renderer

    pitch = Pitch.standard()
    cam = Camera.fit_to_pitch(pitch)
    pygame.Surface((cam.screen_width, cam.screen_height))
    renderer = Renderer(cam)
    cam.set_zoom_level(zoom)
    line_w = max(1, int(0.12 * cam.pixels_per_metre))
    post_w = renderer._goal_post_px()
    assert post_w >= line_w + 2
    assert post_w % 2 == 1  # odd: exactly centred on its row/column


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
