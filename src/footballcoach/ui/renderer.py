"""Draws the pitch, players, ball, and HUD to a pygame surface. Pure
rendering - no game logic or input handling lives here (see input.py / app.py).
"""
from __future__ import annotations

import collections
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pygame
import pygame.gfxdraw
from footballcoach.config import load_graphics_config
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.ui import player_sprites, style
from footballcoach.ui.camera import Camera

from footballcoach.mathutils import Vector3

if TYPE_CHECKING:
    from footballcoach.ui.gamelog import GameLog, LogLevel
    from footballcoach.ui.input import KickUIState
    from footballcoach.ui.scenarios import AnyScenarioParam, ScenarioBoolParam, ScenarioChoiceParam, ScenarioParam

def _format_match_clock(time_s: float) -> str:
    """MM:SS.mmm elapsed match time, e.g. "07:23.451" -- millisecond
    precision so events logged close together (draw_game_log's own display,
    and e.g. a suspected double-tackle) are actually distinguishable rather
    than all showing the same whole-second timestamp."""
    minutes, seconds = divmod(max(0.0, time_s), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def _describe_player_ai(player: Player) -> str:
    """Short label for the inspector panel: what's actually driving this
    player right now. Switches on rules_ai.py's AI hierarchy -- ``ai is
    None`` means the player is order-driven only (no per-tick AI callback
    at all, e.g. a human-controlled trainee); ``HybridPlayerAI`` is a
    neural net with an optional human/rules order-override channel (see
    its own docstring); any other ``PlayerAI`` subclass is a rules-based
    AI, labelled by its concrete class name."""
    from footballcoach.rules_ai import HybridPlayerAI, NeuralPlayerAI

    ai = player.ai
    if ai is None:
        return "No AI (order-driven only)"
    if isinstance(ai, HybridPlayerAI):
        return "Neural net (hybrid" + (", override active" if ai.order_override_active else "") + ")"
    if isinstance(ai, NeuralPlayerAI):
        return "Neural net"
    return f"Rules ({type(ai).__name__})"


def _format_order_value(value: object) -> str:
    from enum import Enum
    if isinstance(value, Vector3):
        return f"({value.x:.1f}, {value.y:.1f}, {value.z:.1f})"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, Enum):
        return value.name
    if callable(value):
        return "<fn>"
    return repr(value)


# PlayerAttributes field name -> short display label, in display order.
# See entities/attributes.py -- every field is a [0.0, 1.0] rating, so no
# per-attribute scale/unit conversion is needed for the gradient bars below.
_ATTRIBUTE_LABELS = [
    ("top_speed", "Top Speed"),
    ("acceleration", "Accel"),
    ("stamina", "Stamina"),
    ("kick_precision", "Kick Prec"),
    ("kick_power", "Kick Power"),
    ("dribbling", "Dribbling"),
    ("ball_control", "Ball Ctrl"),
    ("tackling", "Tackling"),
]


def _attribute_bar_colour(value: float) -> tuple[int, int, int]:
    """Red (0.0) -> green (1.0) via HSV hue interpolation (0 deg = red,
    120 deg = green) rather than a raw RGB lerp -- lerping (255,0,0) to
    (0,255,0) directly passes through a muddy brown/olive at the midpoint
    instead of a clean yellow, since RGB isn't a perceptually-ordered space
    for this."""
    value = max(0.0, min(1.0, value))
    colour = pygame.Color(0)
    colour.hsva = (value * 120.0, 75.0, 85.0, 100.0)
    return (colour.r, colour.g, colour.b)


def _format_order_lines(order: object) -> list[str]:
    """(order type name, then one "field: value" line per dataclass field)
    -- generic over every Order in orders.py via ``dataclasses.fields()``,
    so a newly added Order type or field shows up here automatically with
    no per-Order-type code. Skips ``on_complete`` (an internal callback,
    not a player-facing argument) and any leading-underscore private
    bookkeeping field (e.g. MoveOrder's ``_overshoot_timer_s``)."""
    import dataclasses
    lines = [type(order).__name__]
    for f in dataclasses.fields(order):
        if f.name == "on_complete" or f.name.startswith("_"):
            continue
        lines.append(f"  {f.name}: {_format_order_value(getattr(order, f.name))}")
    return lines


# Font family names tried in order when searching for a font that can render
# Unicode emoji/symbols.  The monochrome Noto Emoji font is best on Linux;
# Symbola is a good fallback; if none match we fall back to the pygame default
# (icons will render as replacement boxes on unsupported fonts, which is benign).
#
# NOTE: pygame.font.match_font() matches against the font's registered family
# name (spaces/case stripped), NOT its filename -- "segoeuiemoji" is required
# here, not "seguiemj" (the .ttf filename on disk). The old "seguiemj" entry
# never matched anything, so match_font() returned None for every candidate on
# Windows and silently fell back to the plain pygame default font, which has
# no emoji glyphs at all -- that's why action icons showed as blank/broken
# boxes on Windows despite Segoe UI Emoji being installed and rendering fine
# once actually loaded (confirmed: real colour glyph data, not blank).
_EMOJI_FONT_CANDIDATES = [
    "noto emoji",
    "notoemoji",
    "noto color emoji",
    "symbola",
    "unifont",
    "segoeuiemoji",
]


# Stroke width (px) of the goal net's mesh lines AND of its border (side/back
# outline) -- one constant so they can't drift apart: the border is drawn at the
# mesh's own thickness, not the (zoom-scaled) pitch line width.
_GOAL_NET_LINE_PX = 1

_ARC_STEP_RAD = math.radians(3.0)


def penalty_arc_world_points(pitch: Pitch, *, left: bool, radius_m: float = 9.15) -> list[tuple[float, float]]:
    """World-space vertices of the penalty arc ("D") for one end: the part of
    the circle of *radius_m* around that end's penalty spot that lies
    *outside* the penalty box (Law 1). It meets the box's front edge at
    y = +/- sqrt(r^2 - dx^2), where dx is the spot-to-box-edge distance.
    Empty if the circle doesn't reach past the box edge."""
    spot = pitch.penalty_spot(left=left)
    dx = pitch.box_length_m - pitch.penalty_spot_distance_m  # spot -> box edge, toward midfield
    if dx >= radius_m:
        return []
    half_sweep = math.acos(max(-1.0, dx / radius_m))
    n = max(2, math.ceil(2 * half_sweep / _ARC_STEP_RAD))
    toward_midfield = 1.0 if left else -1.0
    pts = []
    for i in range(n + 1):
        psi = -half_sweep + 2 * half_sweep * i / n
        pts.append((spot.x + toward_midfield * radius_m * math.cos(psi), spot.y + radius_m * math.sin(psi)))
    return pts


def corner_arc_world_points(pitch: Pitch, sx: int, sy: int, radius_m: float = 1.0) -> list[tuple[float, float]]:
    """World-space vertices of the corner arc: a quarter circle of *radius_m*
    centred on the corner at (sx * half_length, sy * half_width), curving
    inside the pitch (sx, sy in {-1, +1})."""
    cx, cy = sx * pitch.half_length, sy * pitch.half_width
    n = math.ceil((math.pi / 2) / _ARC_STEP_RAD)
    return [
        (cx - sx * radius_m * math.cos(phi), cy - sy * radius_m * math.sin(phi))
        for phi in (math.pi / 2 * i / n for i in range(n + 1))
    ]


class _GoalPx(NamedTuple):
    """One goal's integer screen coordinates (see ``Renderer._goal_px``)."""
    line_x: int   # x of the goal line
    bar_x: int    # x the crossbar is drawn at (goal line + apparent lean)
    back_x: int   # x of the back of the net, where it meets the ground
    back_top_x: int  # x the TOP of the back of the net is drawn at (back_x + the same apparent lean)
    y_top: int    # row of the top post's centre line
    y_bot: int    # row of the bottom post's centre line


_SHADE_AMBIENT = 0.4  # fraction of full brightness an unlit surface still keeps
_SHADE_GAIN = 1.1     # >1 so a round bar's best-lit strip reaches full brightness (its normal can't face the light head-on)


def _light_vector(flip_x: bool = False, flip_y: bool = False) -> tuple[float, float, float]:
    """Unit vector TOWARD the light in screen space (x right, y down, z up out of the
    pitch), from `style.LIGHT_DIR_XY` / `style.LIGHT_ELEVATION_DEG`. ``flip_x`` / ``flip_y``
    mirror the light across that axis: the goal frame is lit as if from a point above the
    pitch's middle, so each part flips the light to point toward the centre (see
    ``Renderer._draw_goal_top``)."""
    lx, ly = style.LIGHT_DIR_XY
    lx, ly = (-lx if flip_x else lx), (-ly if flip_y else ly)
    norm = math.hypot(lx, ly) or 1.0
    elev = math.radians(style.LIGHT_ELEVATION_DEG)
    horiz = math.cos(elev)
    return (lx / norm * horiz, ly / norm * horiz, math.sin(elev))


def surface_shade(
    nx: float, ny: float, nz: float, strength: float, flip_x: bool = False, flip_y: bool = False,
) -> float:
    """Brightness factor (<= 1) of a surface with unit normal (nx, ny, nz): Lambert against
    the scene light plus an ambient floor, blended toward flat by ``strength`` (0 = 1.0
    everywhere, 1 = the full lit-to-dark range)."""
    lx, ly, lz = _light_vector(flip_x, flip_y)
    lit = min(1.0, _SHADE_GAIN * max(0.0, nx * lx + ny * ly + nz * lz))
    full = _SHADE_AMBIENT + (1.0 - _SHADE_AMBIENT) * lit
    return 1.0 - strength * (1.0 - full)


def cylinder_shades(n_px: int, across: str, strength: float, samples: int = 8, flip: bool = False) -> list[float]:
    """Brightness factors for each pixel across a round bar ``n_px`` wide, lit by the scene
    light. ``across`` is the screen axis the width runs along ("y" for a bar running left-right
    such as a post, "x" for one running up-down such as the crossbar); index 0 is the low-coordinate
    side (top / left). Each pixel averages ``samples`` sub-positions of the cylinder's cross-section,
    so a 3px bar gets three sensible tones, not aliased extremes. ``flip`` mirrors the light
    across the width axis (highlight and shadow swap sides)."""
    out = []
    for i in range(n_px):
        acc = 0.0
        for s in range(samples):
            t = -1.0 + 2.0 * (i + (s + 0.5) / samples) / n_px
            nz = math.sqrt(max(0.0, 1.0 - t * t))
            if across == "x":
                acc += surface_shade(t, 0.0, nz, strength, flip_x=flip)
            else:
                acc += surface_shade(0.0, t, nz, strength, flip_y=flip)
        out.append(acc / samples)
    return out


def ball_under_goal_frame(pitch: Pitch, x: float, y: float, z: float, radius_m: float = 0.11) -> bool:
    """True if a ball at world (x, y, z) is *inside* a goal, i.e. beneath its
    crossbar and roof net: past the goal line, no further than the back wall,
    between the posts, and below crossbar height. This is the case where the
    frame should be drawn OVER the ball; a ball above the bar (going over), or
    outside the goal, is drawn over the frame instead. Mirrors the engine's
    own "inside the goal" test in ``ball_physics.resolve_goal_boundary``."""
    ax = abs(x)
    if ax <= pitch.half_length or ax > pitch.half_length + pitch.goal_depth_m + radius_m:
        return False
    if abs(y) > pitch.goal_width_m / 2.0 + radius_m:
        return False
    return z < pitch.goal_height_m


def back_wall_net_segments(
    length_px: float, rise_px: float, height_px: float, gap_px: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Line segments of the back-of-net mesh in the strip's local pixel space:
    x runs from 0 (where the wall meets the ground) to *length_px* (its top
    edge), y from 0 to *height_px* (along the goal's width).

    The back wall is vertical, and the top-down parallax maps a wall point's
    HEIGHT to screen x (the short strip, *length_px* for the full crossbar
    height) and its position ALONG THE GOAL to screen y. A 45-degree diamond
    mesh on that wall (horizontal run == height) therefore appears as two
    families of *steep* lines: over the strip's full length each rises
    *rise_px* = (goal height in metres) x (pixels per metre) in y -- far more
    than *length_px* -- rather than the roof's 45-degree lattice. *gap_px* is the
    spacing of parallel lines along y (at the ground edge). The two families
    are mirror images about the strip's centre line."""
    segments = []
    y = -math.ceil(rise_px)
    while y < height_px:
        segments.append(((0.0, float(y)), (length_px, y + rise_px)))
        segments.append(((0.0, y + rise_px), (length_px, float(y))))
        y += gap_px
    return segments


class CornerFlagGeometry(NamedTuple):
    """World-space (x, y) points describing one corner flag as seen from above
    with parallax; see ``corner_flag_world_points``."""
    base: tuple[float, float]    # where the pole meets the ground (the pitch corner)
    top: tuple[float, float]     # the top of the pole, displaced away from the pitch centre
    attach: tuple[float, float]  # lower end of the cloth's edge along the pole
    tip: tuple[float, float]     # free end of the cloth, streaming sideways off the pole


def corner_flag_world_points(
    pitch: Pitch, sx: int, sy: int, *, pole_lean_m: float, flag_width_m: float, flag_drop_frac: float,
) -> CornerFlagGeometry:
    """Where the parts of the corner flag at (sx * half_length, sy * half_width)
    appear. The pole is vertical, so in the top-down view (with the same
    parallax as the crossbar: tall things lean away from the point under the
    camera) its top is displaced *outward from the pitch centre* by
    *pole_lean_m*, along the centre->corner direction. The cloth is attached
    along the upper *flag_drop_frac* of the pole and its free end streams
    *flag_width_m* off the pole, perpendicular to it, on the side that runs
    along the touchline toward the pitch's middle.

    Perpendicular on purpose: aiming the tip along the touchline (the obvious
    "inward") makes the pennant a needle, because the pole leans along the
    corner diagonal and the cloth's base edge lies along the pole -- the tip
    ends up only ~30 degrees off that edge, so the triangle has almost no
    width. Perpendicular gives it its full *flag_width_m* of width."""
    cx, cy = sx * pitch.half_length, sy * pitch.half_width
    norm = math.hypot(cx, cy) or 1.0
    ux, uy = cx / norm, cy / norm
    top = (cx + ux * pole_lean_m, cy + uy * pole_lean_m)
    keep = 1.0 - flag_drop_frac  # fraction of the pole's lean at the cloth's lower end
    attach = (cx + ux * pole_lean_m * keep, cy + uy * pole_lean_m * keep)
    mid = ((top[0] + attach[0]) / 2.0, (top[1] + attach[1]) / 2.0)
    px, py = -uy, ux  # one of the two perpendiculars to the pole direction...
    if -sx * px < 0:  # ...take the one whose x component points toward the pitch's middle
        px, py = -px, -py
    tip = (mid[0] + px * flag_width_m, mid[1] + py * flag_width_m)
    return CornerFlagGeometry(base=(cx, cy), top=top, attach=attach, tip=tip)


class Renderer:
    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        pygame.font.init()
        self.hud_font = pygame.font.Font(style.FONT_NAME, style.HUD_FONT_SIZE)
        self.title_font = pygame.font.Font(style.FONT_NAME, style.TITLE_FONT_SIZE)

        # Load graphics config for tunable display constants.
        gcfg = load_graphics_config()

        icon_font_size = gcfg["action_icons"].get("font_size_px", style.ICON_FONT_SIZE)
        self._icon_target_px: int = icon_font_size

        # Find an emoji-capable font for action icons.
        # NotoColorEmoji is a fixed-size bitmap font (128px/glyph) that ignores
        # the size argument — we render at native size then scale down.
        # Other candidates (symbola, unifont) would scale normally if present.
        icon_font_path = None
        self._icon_font_is_bitmap = False
        for name in _EMOJI_FONT_CANDIDATES:
            path = pygame.font.match_font(name)
            if path:
                icon_font_path = path
                # Detect the NotoColorEmoji bitmap font: render a test char and
                # check if the rendered height is much larger than requested.
                probe = pygame.font.Font(path, icon_font_size)
                _, probe_h = probe.size("A")
                if probe_h > icon_font_size * 4:
                    # Bitmap font — load at native size, scale surface later.
                    self._icon_font_is_bitmap = True
                    self.icon_font = pygame.font.Font(path, 109)  # NotoColorEmoji native size
                else:
                    self.icon_font = probe
                break
        else:
            # No emoji font found — fall back to pygame default (boxes, but harmless).
            self.icon_font = pygame.font.Font(style.FONT_NAME, icon_font_size)

        # Pre-render + scale-down cache so we don't do the slow transform every frame.
        self._icon_cache: dict[str, pygame.Surface] = {}
        self.min_player_radius_px: int = gcfg["player"]["min_radius_px"]
        self._inactive_alpha: int = int(gcfg["player"].get("inactive_alpha", style.INACTIVE_ALPHA))
        # Opacity of every ring (player state / selection / stamina rings and the ball's state ring); they
        # are drawn under the players and the ball (see `draw_pitch_and_ball`).
        self._ring_alpha: int = max(0, min(255, int(gcfg.get("rings", {}).get("alpha", 110))))
        self.min_ball_radius_px: int = gcfg["ball"]["min_radius_px"]
        # How much the minimum-visibility SIZE FLOORS (min_radius_px, both entities) shrink, relative
        # to true-to-scale size, as the camera zooms in -- see `_ball_base_radius_px` / `_player_radius_px`.
        self._size_floor_zoom_decay: float = max(0.0, min(1.0, float(gcfg.get("size_floor", {}).get("zoom_decay", 0.45))))
        self._ball_outline: bool = gcfg["ball"].get("outline", False)
        # outline_width_px may be a float: values >= 1 → integer pixel width at full
        # opacity; values in (0, 1) → 1px outline drawn at that fraction of full opacity
        # (e.g. 0.5 → alpha 128), giving a visually softer/thinner border.
        self._ball_outline_width: float = max(0.0, float(gcfg["ball"].get("outline_width_px", 1)))
        self._ball_height_boost_per_m: float = float(gcfg["ball"].get("height_boost_per_metre", 0.35))
        sl = gcfg["speed_lines"]
        self._speed_line_threshold: float = sl["threshold_mps"]
        self._speed_line_count: int = sl["count"]
        self._speed_line_length_px: int = sl["length_px"]
        self._speed_line_gap_px: int = sl["gap_px"]
        sf = gcfg["stamina_flash"]
        self._stamina_flash_threshold: float = sf["threshold"]
        self._stamina_flash_hz: float = sf["flash_hz"]
        hi = gcfg.get("heading_indicator", {})
        self._heading_length_px: int = int(hi.get("length_px", 8))
        self._heading_alpha: int = int(hi.get("alpha", 255))
        self._heading_base_half_width_px: float = float(hi.get("base_half_width_px", 6.0))
        self._heading_base_inset_px: float = float(hi.get("base_inset_px", 3.0))
        _pn = gcfg.get("pause_notification", {})
        self.pause_notification_font = pygame.font.Font(style.FONT_NAME, _pn.get("font_size_px", 26))

        # Ball spin dots — 3D model projected to top-down view
        _sd = gcfg.get("ball_spin_dots", {})
        self._spin_dot_count: int = _sd.get("count", 12)
        self._spin_orbit_frac: float = _sd.get("projection_scale_fraction", _sd.get("orbit_radius_fraction", 0.93))
        self._spin_dot_radius_frac: float = _sd.get("dot_radius_fraction", 0.25)
        _col = _sd.get("color", [30, 30, 30])
        self._spin_dot_color: tuple = (int(_col[0]), int(_col[1]), int(_col[2]))
        # 3x3 rotation matrix tracking ball orientation (identity = initial pose)
        self._ball_orientation: list = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
        # Fixed dot positions on unit sphere (Fibonacci lattice)
        self._ball_dot_positions: list = self._make_ball_dot_positions(self._spin_dot_count)
        # Last ball position for estimating rolling velocity each frame
        self._last_ball_pos: tuple[float, float] = (0.0, 0.0)

        # Ball state rings
        _bsr = gcfg.get("ball_state_rings", {})
        self._flying_min_height_m: float = float(_bsr.get("flying_min_height_above_ground_m", 0.01))
        self._ring_show_flying: bool = bool(_bsr.get("show_flying", True))
        self._ring_show_rolling: bool = bool(_bsr.get("show_rolling", True))
        self._ring_show_bounced: bool = bool(_bsr.get("show_bounced", True))
        self._ring_offset_px: int = int(_bsr.get("offset_px", 3))
        self._ring_width_px: int = max(1, int(_bsr.get("width_px", 2)))
        def _rgb(key: str, default: tuple) -> tuple:
            v = _bsr.get(key, list(default))
            return (int(v[0]), int(v[1]), int(v[2]))
        self._ring_color_flying: tuple = _rgb("color_flying", style.BALL_STATE_FLYING_OUTLINE)
        self._ring_color_rolling: tuple = _rgb("color_rolling", style.BALL_STATE_ROLLING_OUTLINE)
        self._ring_color_bounced: tuple = _rgb("color_bounced", style.BALL_STATE_BOUNCED_OUTLINE)

        # Goal net mesh
        _pm = gcfg.get("pitch_markings", {})
        self._spot_radius_m: float = float(_pm.get("spot_radius_m", 0.18))
        self._penalty_arc_radius_m: float = float(_pm.get("penalty_arc_radius_m", 9.15))
        self._corner_arc_radius_m: float = float(_pm.get("corner_arc_radius_m", 1.0))
        _gf = gcfg.get("goal_frame", {})
        self._crossbar_lean_m: float = float(_gf.get("crossbar_lean_m", 0.9))
        self._goal_post_width_m: float = float(_gf.get("post_width_m", 0.16))
        # ONE alpha for both the goal-mouth tint and the back-wall tint (they used to differ --
        # 40 vs 30 -- which read as an unexplained colour mismatch between the two; now they're
        # either both this or both off). 0 = neither drawn, the goal footprint is plain pitch green.
        self._goal_footprint_tint_alpha: int = int(_gf.get("footprint_tint_alpha", 0))
        self._goal_shade_strength: float = max(0.0, min(1.0, float(_gf.get("shade_strength", 0.7))))
        self._foot_cache: dict[tuple, pygame.Surface] = {}
        # Ball: a soft ground shadow (light above the pitch's middle) and sphere shading.
        _bs = gcfg.get("ball_shadow", {})
        self._ball_shadow_enabled: bool = bool(_bs.get("enabled", True))
        # ONE scene light for every shadow and shading effect: a point this high above the middle
        # of the pitch (ball shadow + shading, player shadows + shading).
        self._light_height_m: float = max(1.0, float(gcfg.get("scene_light", {}).get("height_m", 40.0)))
        # Minimum contact shadow, as a fraction of the drawn ball's radius (so it scales with zoom
        # exactly like the ball does), with a small pixel floor for the default zoom.
        self._ball_shadow_contact_frac: float = max(0.0, float(_bs.get("contact_radius_frac", 0.33)))
        self._ball_shadow_contact_min_px: float = max(0.0, float(_bs.get("contact_min_px", 2.0)))
        self._ball_shadow_alpha: int = int(_bs.get("alpha", 100))
        # Along-axis fade (see `_shadow_ellipse_sprite`): like the players' capsules, the ball's
        # shadow is sharp/dark on the end nearest the ball and soft/light on the far tip.
        self._ball_shadow_tip_fade: float = max(0.0, min(0.95, float(_bs.get("tip_fade", 0.3))))
        # ...and it fades/softens further, the higher the ball is (an airborne shadow has no sharp
        # contact edge anywhere): ramps in linearly over `height_fade_ramp_m` of height, up to
        # `height_extra_soft` / `height_extra_fade` on top of the base fade above.
        self._ball_shadow_height_fade_ramp_m: float = max(0.1, float(_bs.get("height_fade_ramp_m", 3.3)))
        self._ball_shadow_height_extra_soft: float = max(0.0, float(_bs.get("height_extra_soft", 0.66)))
        self._ball_shadow_height_extra_fade: float = max(0.0, float(_bs.get("height_extra_fade", 0.168)))
        self._shadow_cache: dict[tuple, pygame.Surface] = {}     # soft capsules, shared by ball and players
        _bsh = gcfg.get("ball_shading", {})
        self._ball_shading_strength: float = max(0.0, min(1.0, float(_bsh.get("strength", 0.85)))) if _bsh.get("enabled", True) else 0.0
        self._ball_shade_cache: dict[tuple, pygame.Surface] = {}
        # Players: rounded per-part shading of the sprite art, and a ground shadow.
        _pl = gcfg.get("player_shading", {})
        self._player_shading_strength: float = max(0.0, min(1.0, float(_pl.get("strength", 0.9)))) if _pl.get("enabled", True) else 0.0
        _psh = gcfg.get("player_shadow", {})
        self._player_shadow_enabled: bool = bool(_psh.get("enabled", True))
        self._player_shadow_alpha: int = int(_psh.get("alpha", 90))
        self._player_shadow_contact_m: float = max(0.0, float(_psh.get("contact_m", 0.18)))
        # How much a player's shadow lightens from the feet (full strength) to the tip (1 - tip_fade).
        self._player_shadow_tip_fade: float = max(0.0, min(0.9, float(_psh.get("tip_fade", 0.3))))
        # Turf: soft world-anchored light/dark patches, fine grain and a faint vignette
        # instead of one flat green fill (see `_draw_turf`).
        _tf = gcfg.get("turf", {})
        self._turf_enabled: bool = bool(_tf.get("enabled", True))
        self._turf_patch_amp: float = float(_tf.get("patch_amp", 0.02))
        _cells = _tf.get("patch_cell_m", [6.0, 2.2])
        self._turf_patch_cells_m: tuple[float, float] = (float(_cells[0]), float(_cells[1]))
        self._turf_noise_amp: float = float(_tf.get("noise_amp", 0.02))
        self._turf_vignette: float = float(_tf.get("vignette", 0.14))
        self._turf_seed: int = int(_tf.get("seed", 7))
        self._turf_patch: pygame.Surface | None = None
        self._turf_patch_key: tuple | None = None
        self._turf_overlay: pygame.Surface | None = None
        self._turf_overlay_key: tuple | None = None
        self._turf_bg: pygame.Surface | None = None
        self._turf_bg_key: tuple | None = None
        _cf = gcfg.get("corner_flag", {})
        self._corner_pole_height_m: float = float(_cf.get("pole_height_m", 4.0))
        self._corner_pole_width_m: float = float(_cf.get("pole_width_m", 0.06))
        self._corner_flag_width_m: float = float(_cf.get("flag_width_m", 0.55))
        # The cloth's length along the pole is an absolute (apparent) size, so a taller pole
        # doesn't make the pennant taller too.
        self._corner_flag_length_m: float = float(_cf.get("flag_length_m", 0.32))
        self._corner_shadow_alpha: int = int(_cf.get("shadow_alpha", 75))
        self._corner_shadow_height_m: float = float(_cf.get("shadow_height_m", 1.2))
        _gn = gcfg.get("goal_net", {})
        self._goal_net_spacing_m: float = float(_gn.get("spacing_m", 0.35))
        self._goal_net_min_spacing_px: int = max(2, int(_gn.get("min_spacing_px", 8)))
        self._goal_back_net_gap_scale: float = float(_gn.get("back_spacing_scale", 2.0))
        self._aa_disc_cache: dict[tuple, pygame.Surface] = {}
        self._net_layer_cache: dict[tuple, pygame.Surface] = {}
        self._goal_net_alpha: int = int(_gn.get("alpha", style.GOAL_NET_ALPHA))
        self._goal_net_sag_m: float = max(0.0, float(_gn.get("sag_m", 0.3)))
        _gn_col = _gn.get("color", list(style.GOAL_NET_COLOUR))
        self._goal_net_colour: tuple = (int(_gn_col[0]), int(_gn_col[1]), int(_gn_col[2]))

        # Ball trail
        _bt = gcfg.get("ball_trail", {})
        self._trail_length: int = _bt.get("length", 10)
        self._trail_min_speed: float = _bt.get("min_speed_mps", 2.0)
        self._trail_max_alpha: int = _bt.get("max_alpha", 150)
        self._trail_radius_frac: float = _bt.get("radius_fraction", 0.75)
        # interp_steps: how many sub-samples to insert between each stored position
        # when drawing (1 = no interpolation, 3 = two extra points per gap).
        self._trail_radius_taper: float = max(0.0, min(1.0, float(_bt.get("radius_taper", 0.35))))
        self._trail_interp_steps: int = max(1, int(_bt.get("interp_steps", 1)))
        self._ball_trail: collections.deque = collections.deque(maxlen=self._trail_length)

        # Player sprites (see player_sprites.py) -- replaces the plain
        # circle when enabled. Per-player gait phase is renderer-side state
        # (purely cosmetic, so it doesn't belong on the engine's Player),
        # advanced once per rendered frame via update_player_animations.
        self._sprites_enabled: bool = bool(gcfg.get("player_sprites", {}).get("enabled", True))
        self._sprite_params = player_sprites.PlayerSpriteParams.from_config() if self._sprites_enabled else None
        self._player_gait_phase: dict[str, float] = {}

    @staticmethod
    def _make_icosahedron_vertices() -> list:
        """The 12 vertices of a regular icosahedron, projected onto the unit sphere -- the classic
        football's own pattern: a real ball is a TRUNCATED icosahedron (12 pentagons + 20 hexagons),
        and each pentagon's centre sits exactly where the untruncated icosahedron's 12 vertices are.
        It's also the known-optimal (Tammes problem) even distribution of 12 points on a sphere: every
        point's nearest neighbours are equally far away (measured: a uniform 63.4 degrees, vs. as
        close as 31 degrees between the two nearest dots in the old 15-point Fibonacci lattice --
        "two dots too close together" was a real, measurable unevenness, not just an impression)."""
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        raw = []
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                raw.append((0.0, s1, s2 * phi))
                raw.append((s1, s2 * phi, 0.0))
                raw.append((s2 * phi, 0.0, s1))
        points = []
        for x, y, z in raw:
            n = math.sqrt(x * x + y * y + z * z)
            points.append((x / n, y / n, z / n))
        return points

    @staticmethod
    def _make_fibonacci_sphere(n: int) -> list:
        """Evenly distribute n points on the unit sphere using the Fibonacci lattice. Used as a
        general fallback for any dot count OTHER than the classic 12 (see `_make_icosahedron_vertices`,
        `_make_ball_dot_positions`) -- reasonably even for most n, but not the provably-optimal
        spacing an exact regular solid gives for the counts that have one."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        for i in range(n):
            y = (1.0 - (i / (n - 1)) * 2.0) if n > 1 else 0.0
            r = math.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            points.append((math.cos(theta) * r, y, math.sin(theta) * r))
        return points

    @classmethod
    def _make_ball_dot_positions(cls, n: int) -> list:
        """Dot positions for the ball's spin markers: the icosahedron's 12 vertices for the classic
        count, a Fibonacci lattice otherwise."""
        return cls._make_icosahedron_vertices() if n == 12 else cls._make_fibonacci_sphere(n)

    @staticmethod
    def _mat_mul3(A: list, B: list) -> list:
        """3x3 matrix multiply."""
        return [
            [A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j] for j in range(3)]
            for i in range(3)
        ]

    def record_trail(self, ball: Ball) -> None:
        """Append or trim the ball ghost trail. Call once per physics tick."""
        speed = ball.velocity.length_xy()
        if speed >= self._trail_min_speed and ball.possessed_by is None:
            self._ball_trail.append((ball.position.x, ball.position.y, ball.position.z))
        elif speed < self._trail_min_speed * 0.5 or ball.possessed_by is not None:
            if self._ball_trail:
                self._ball_trail.popleft()

    def update_ball_effects(self, ball: Ball, dt_s: float) -> None:
        """Integrate 3D ball orientation from spin + rolling.
        Call once per rendered frame (only when not paused), with ``dt_s`` the SIMULATION
        seconds since the previous call (``ui/sim_clock.py``) -- ``ball.spin`` is in rad per
        sim second, so a wall-clock or 1/fps step under-rotates it whenever the sim is sped up.
        A zero ``dt_s`` (no sim time passed) leaves the orientation untouched."""
        # Estimate XY velocity from position delta (works for both free and possessed).
        cur_x, cur_y = ball.position.x, ball.position.y
        if dt_s <= 1e-9:
            self._last_ball_pos = (cur_x, cur_y)
            return
        est_vx = (cur_x - self._last_ball_pos[0]) / dt_s
        est_vy = (cur_y - self._last_ball_pos[1]) / dt_s
        self._last_ball_pos = (cur_x, cur_y)

        # Rolling contribution: a ball moving in direction v rolls around the axis
        # perpendicular to v in the horizontal plane — ẑ × v̂ — at ω = |v| / radius.
        # Combined with actual spin (topspin, sidespin from kick) for full orientation.
        r = max(ball.radius_m, 0.01)
        roll_wx = -est_vy / r
        roll_wy =  est_vx / r
        total_wx = ball.spin.x + roll_wx
        total_wy = ball.spin.y + roll_wy
        total_wz = ball.spin.z

        # Integrate orientation: Rodrigues rotation by ω*dt about ω axis
        total_mag = math.sqrt(total_wx*total_wx + total_wy*total_wy + total_wz*total_wz)
        if total_mag > 1e-9:
            angle = total_mag * dt_s
            ax = total_wx / total_mag
            ay = total_wy / total_mag
            az = total_wz / total_mag
            c, s = math.cos(angle), math.sin(angle)
            t = 1.0 - c
            dR = [
                [t*ax*ax + c,     t*ax*ay - s*az, t*ax*az + s*ay],
                [t*ax*ay + s*az,  t*ay*ay + c,    t*ay*az - s*ax],
                [t*ax*az - s*ay,  t*ay*az + s*ax, t*az*az + c   ],
            ]
            self._ball_orientation = self._mat_mul3(dR, self._ball_orientation)

    def update_player_animations(self, players: list[Player], dt_s: float) -> None:
        """Advances each player's stride-gait phase (see player_sprites.py).
        Call once per rendered frame (only when not paused), before drawing
        any players -- a no-op if sprites are disabled. ``dt_s`` is the SIMULATION
        time elapsed since the previous call (``ui/sim_clock.py``), so strides keep
        pace with the players' sim speed at any sim-speed setting."""
        if not self._sprites_enabled:
            return
        for player in players:
            phase = self._player_gait_phase.get(player.player_id, 0.0)
            self._player_gait_phase[player.player_id] = player_sprites.advance_gait_phase(
                phase, player.speed_mps, dt_s, self._sprite_params
            )

    def draw_pitch(self, surface: pygame.Surface, pitch: Pitch, *, goal_tops: bool = True) -> None:
        """Draws the pitch, goals and dressing. ``goal_tops=False`` leaves out
        the goal's "top" layer (roof net, posts, crossbar -- see
        ``draw_goal_tops``) so the caller can draw it AFTER the ball instead;
        ``draw_pitch_and_ball`` does exactly that when the ball is inside a
        goal. Everything else is always drawn here."""
        cam = self.camera
        if self._turf_enabled:
            self._draw_turf(surface, pitch)
        else:
            surface.fill(style.PITCH_GREEN)
        line_w = max(1, int(0.12 * cam.pixels_per_metre))

        def rect_world(x0: float, y0: float, x1: float, y1: float) -> None:
            p0 = cam.world_to_screen(x0, y0)
            p1 = cam.world_to_screen(x1, y1)
            left, top = min(p0[0], p1[0]), min(p0[1], p1[1])
            width, height = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
            pygame.draw.rect(surface, style.PITCH_LINE_WHITE, (left, top, width, height), line_w)

        # Outer boundary.
        rect_world(-pitch.half_length, -pitch.half_width, pitch.half_length, pitch.half_width)

        # Halfway line.
        p0 = cam.world_to_screen(0, -pitch.half_width)
        p1 = cam.world_to_screen(0, pitch.half_width)
        pygame.draw.line(surface, style.PITCH_LINE_WHITE, p0, p1, line_w)

        # Centre circle.
        centre = cam.world_to_screen(0, 0)
        radius_px = cam.scale_length(pitch.centre_circle_radius_m)
        pygame.draw.circle(surface, style.PITCH_LINE_WHITE, centre, radius_px, line_w)
        pygame.gfxdraw.aacircle(surface, centre[0], centre[1], radius_px, style.PITCH_LINE_WHITE)
        if line_w > 1:
            pygame.gfxdraw.aacircle(surface, centre[0], centre[1], max(0, radius_px - line_w + 1), style.PITCH_LINE_WHITE)
        self._draw_pitch_spot(surface, 0.0, 0.0)

        # Penalty boxes and six-yard boxes, both ends.
        half_box_w = pitch.box_width_m / 2.0
        half_six_w = pitch.six_yard_width_m / 2.0
        rect_world(-pitch.half_length, -half_box_w, -pitch.half_length + pitch.box_length_m, half_box_w)
        rect_world(pitch.half_length - pitch.box_length_m, -half_box_w, pitch.half_length, half_box_w)
        rect_world(-pitch.half_length, -half_six_w, -pitch.half_length + pitch.six_yard_length_m, half_six_w)
        rect_world(pitch.half_length - pitch.six_yard_length_m, -half_six_w, pitch.half_length, half_six_w)

        # Penalty spots, and the penalty arcs ("D") -- the part of the circle
        # around each spot that lies outside the penalty box.
        for left in (True, False):
            spot = pitch.penalty_spot(left=left)
            self._draw_pitch_spot(surface, spot.x, spot.y)
            self._draw_world_polyline(
                surface, penalty_arc_world_points(pitch, left=left, radius_m=self._penalty_arc_radius_m), line_w,
            )

        # Corner arcs: a quarter circle inside each corner. Each arc is drawn as a stroke centred
        # on the arc's world points, whose ends sit on the boundary lines' outer edge -- so the
        # stroke's thickness and anti-aliased edge would poke past the boundary lines (measured:
        # 1-6 pixels outside at zoom 1-3). The boundary is drawn INSIDE the rectangle
        # (`rect_world`), so clip the arcs to exactly that rectangle.
        b0 = cam.world_to_screen(-pitch.half_length, -pitch.half_width)
        b1 = cam.world_to_screen(pitch.half_length, pitch.half_width)
        boundary = pygame.Rect(min(b0[0], b1[0]), min(b0[1], b1[1]), abs(b1[0] - b0[0]), abs(b1[1] - b0[1]))
        previous_clip = surface.get_clip()
        surface.set_clip(boundary.clip(previous_clip))
        for sx in (-1, 1):
            for sy in (-1, 1):
                self._draw_world_polyline(
                    surface, corner_arc_world_points(pitch, sx, sy, radius_m=self._corner_arc_radius_m), line_w,
                )
        surface.set_clip(previous_clip)

        # Goals: ground footprint (side/back lines + faint mouth tint) and a
        # coloured bar behind each showing which team defends that end. The
        # goal's "top" layer (roof net, posts, crossbar) comes last, and only
        # here if the caller isn't going to draw it over the ball itself.
        half_goal_w = pitch.goal_width_m / 2.0
        goal_depth_m = pitch.goal_depth_m  # same depth the engine's back wall uses (ball_physics.resolve_goal_boundary)
        for left in (True, False):
            self._draw_goal_footprint(surface, pitch, left=left)
        # The marker sits behind the whole net, i.e. beyond the leaned top-back edge.
        marker_back_m = goal_depth_m + self._goal_lean_m(pitch)
        self._draw_defending_marker(surface, -pitch.half_length - marker_back_m, half_goal_w, style.TEAM_LEFT_COLOUR, faces_positive_x=False)
        self._draw_defending_marker(surface, pitch.half_length + marker_back_m, half_goal_w, style.TEAM_RIGHT_COLOUR, faces_positive_x=True)
        if goal_tops:
            self.draw_goal_tops(surface, pitch)

        self._draw_corner_flags(surface, pitch)
        self._draw_sideline_benches(surface, pitch)

    def _draw_pitch_spot(self, surface: pygame.Surface, x: float, y: float) -> None:
        """A filled pitch-marking spot (centre spot / penalty spot) at world
        (x, y). True-to-scale (`graphics.json["pitch_markings"]["spot_radius_m"]`)
        with a 2px floor so it stays visible when the whole pitch is on screen."""
        cx, cy = self.camera.world_to_screen(x, y)
        r = max(2, self.camera.scale_length(self._spot_radius_m))
        pygame.gfxdraw.filled_circle(surface, cx, cy, r, style.PITCH_LINE_WHITE)
        pygame.gfxdraw.aacircle(surface, cx, cy, r, style.PITCH_LINE_WHITE)

    def _draw_world_polyline(
        self, surface: pygame.Surface, points_world: list[tuple[float, float]], line_w: int,
        colour: tuple[int, int, int] = style.PITCH_LINE_WHITE,
    ) -> None:
        """Draws an open curve given as dense world-space vertices, at the
        same stroke width as the rest of the pitch lines, anti-aliased at every
        width. Segment count is the caller's (see `penalty_arc_world_points`).
        This avoids `pygame.draw.arc`, whose thick strokes tear into gaps.
        1px lines use `aalines` (matching the centre circle); thicker ones are
        built as a polygon strip (the polyline offset +/- half the width along
        its normals) and drawn with `gfxdraw.filled_polygon` + `aapolygon`, which
        blend correctly onto the opaque pitch -- a plain `draw.lines` at that
        width is aliased."""
        pts = [self.camera.world_to_screen_f(x, y) for x, y in points_world]
        if len(pts) < 2:
            return
        if line_w <= 1:
            pygame.draw.aalines(surface, colour, False, pts)
            return
        half = line_w / 2.0
        left, right = [], []
        last = len(pts) - 1
        for i, (x, y) in enumerate(pts):
            x0, y0 = pts[max(i - 1, 0)]
            x1, y1 = pts[min(i + 1, last)]
            tx, ty = x1 - x0, y1 - y0
            length = math.hypot(tx, ty) or 1.0
            nx, ny = -ty / length, tx / length
            left.append((round(x + nx * half), round(y + ny * half)))
            right.append((round(x - nx * half), round(y - ny * half)))
        strip = left + right[::-1]
        pygame.gfxdraw.filled_polygon(surface, strip, colour)
        pygame.gfxdraw.aapolygon(surface, strip, colour)

    def _goal_lean_m(self, pitch: Pitch) -> float:
        """The crossbar's (and back-of-net's) apparent displacement, in metres,
        clamped to the net depth. Also sets the parallax rate (metres per metre
        of height) the corner-flag poles use."""
        return min(self._crossbar_lean_m, pitch.goal_depth_m)

    def _goal_px(self, pitch: Pitch, left: bool) -> "_GoalPx":
        """Integer screen coordinates for one goal, shared by everything that
        draws it (footprint lines, mouth tint, net, posts, crossbar) so they
        line up to the pixel. The two post rows are placed *symmetrically*
        about the pitch's centre row (``y_top = c - k``, ``y_bot = c + k``)
        rather than each being independently truncated from world coords: the
        old independent rounding put the top post 1px inside the goal mouth
        and the bottom post 1px outside it."""
        cam = self.camera
        line_x_w = -pitch.half_length if left else pitch.half_length
        outward = -1.0 if left else 1.0
        lean_m = self._goal_lean_m(pitch)
        k = round(pitch.goal_width_m / 2.0 * cam.pixels_per_metre)
        c = cam.world_to_screen(0.0, 0.0)[1]
        return _GoalPx(
            line_x=cam.world_to_screen(line_x_w, 0.0)[0],
            bar_x=cam.world_to_screen(line_x_w + outward * lean_m, 0.0)[0],
            back_x=cam.world_to_screen(line_x_w + outward * pitch.goal_depth_m, 0.0)[0],
            back_top_x=cam.world_to_screen(line_x_w + outward * (pitch.goal_depth_m + lean_m), 0.0)[0],
            y_top=c - k,
            y_bot=c + k,
        )

    def _draw_goal_footprint(self, surface: pygame.Surface, pitch: Pitch, *, left: bool) -> None:
        """The goal's ground-level parts: the net's border (the two side lines
        and the back line, drawn at the mesh's own thickness,
        ``_GOAL_NET_LINE_PX`` -- NOT the zoom-scaled pitch line width, which
        made the border several times thicker than the mesh when zoomed in;
        the goal line itself is already the pitch boundary), plus faint tints
        over the goal mouth (the opening between the goal line and the
        crossbar) and over the back wall of the net (the strip between where the
        back of the net meets the ground and its top edge -- the same parallax
        as the crossbar). Drawn under the ball.

        All the goal's horizontal edges (ground and top) lie on the same two
        screen rows, since the parallax is purely along x, so the side lines
        run continuously from the goal line to the top-back edge."""
        g = self._goal_px(pitch, left)
        white = style.PITCH_LINE_WHITE
        line_w = _GOAL_NET_LINE_PX
        off = line_w // 2
        x_a, x_b = sorted((g.line_x, g.back_top_x))
        for y in (g.y_top, g.y_bot):
            surface.fill(white, pygame.Rect(x_a, y - off, x_b - x_a + 1, line_w))
        # Where the back of the net touches the ground.
        surface.fill(white, pygame.Rect(g.back_x - off, g.y_top - off, line_w, g.y_bot - g.y_top + line_w))

        if self._goal_footprint_tint_alpha > 0:
            b_a, b_b = sorted((g.back_x, g.back_top_x))
            wall = pygame.Surface((b_b - b_a + 1, g.y_bot - g.y_top + 1), pygame.SRCALPHA)
            wall.fill((*white, self._goal_footprint_tint_alpha))
            surface.blit(wall, (b_a, g.y_top))

            post_w = self._goal_post_px()
            m_a, m_b = sorted((g.line_x, g.bar_x))
            mouth = pygame.Surface((m_b - m_a + 1, g.y_bot - g.y_top + post_w), pygame.SRCALPHA)
            mouth.fill((*white, self._goal_footprint_tint_alpha))
            surface.blit(mouth, (m_a, g.y_top - post_w // 2))

    def _goal_post_px(self) -> int:
        """Post/crossbar stroke width in px: ``goal_frame.post_width_m`` at the
        current zoom, but never less than 2px thicker than the net's own lines
        (``_GOAL_NET_LINE_PX``) so the frame always reads heavier than the net
        behind it. Any pixel count works: both posts are placed symmetrically
        about the pitch centre row (see ``_goal_px``), an even width just
        leaves the pair half a pixel off the centre row, which is invisible."""
        return max(_GOAL_NET_LINE_PX + 2, round(self._goal_post_width_m * self.camera.pixels_per_metre))

    def draw_goal_tops(self, surface: pygame.Surface, pitch: Pitch) -> None:
        """The goal's "top" layer for both ends: the roof net, the two posts
        and the crossbar. Split out from ``draw_pitch`` because it must be
        drawn OVER a ball that is inside the goal (the ball is under the roof
        and the bar) but UNDER a ball that is above it -- see
        ``draw_pitch_and_ball``."""
        for left in (True, False):
            self._draw_goal_top(surface, pitch, left=left)

    def _draw_goal_top(self, surface: pygame.Surface, pitch: Pitch, *, left: bool) -> None:
        """Gives the goal a slight 3D look in the top-down view. The crossbar
        (2.44m up) is drawn with a little parallax -- displaced
        `graphics.json["goal_frame"]["crossbar_lean_m"]` from the goal line,
        *away from the pitch* (over the net) -- so it no longer hides on top
        of the goal line, and the goal mouth (the opening between the goal line
        on the ground and the crossbar) is visible, with the posts as the
        lines joining their feet to the crossbar ends. The net is drawn only
        BEYOND the crossbar (it is the roof net); the mouth itself is open.

        The displacement is purely apparent (clamped to the net depth); the
        ball's ground truth is still the goal line at x = +/-half_length
        (Pitch.is_goal)."""
        g = self._goal_px(pitch, left)
        white = style.PITCH_LINE_WHITE
        post_w = self._goal_post_px()
        half = post_w // 2

        # Back wall first (the strip between the ground line and the top-back edge),
        # with its own mesh at the right (steep) angle; then the roof net over
        # EVERYTHING from the crossbar to the top-back edge -- the roof is at
        # crossbar height, nearer the camera than the wall, so its (see-through)
        # mesh overlaps the wall strip and is drawn over it.
        self._draw_goal_back_net(surface, pitch, left=left)
        n_a, n_b = sorted((g.bar_x, g.back_top_x))
        # The roof net hangs a little below crossbar height in the middle; lower points lean
        # less, so the mesh bows toward the pitch there (0 at the frame, deepest in the middle).
        sag_dx = (1 if left else -1) * (
            self._goal_net_sag_m / pitch.goal_height_m * self._goal_lean_m(pitch) * self.camera.pixels_per_metre
        )
        self._draw_goal_net(surface, n_a, n_b, g.y_top, g.y_bot, sag_dx_px=sag_dx)
        # The top edge of the back of the net (thin, like the net's border).
        off = _GOAL_NET_LINE_PX // 2
        surface.fill(white, pygame.Rect(
            g.back_top_x - off, g.y_top - off, _GOAL_NET_LINE_PX, g.y_bot - g.y_top + _GOAL_NET_LINE_PX,
        ))

        # Posts (feet on the goal line -> tops at the crossbar) and crossbar.
        p_a, p_b = sorted((g.line_x, g.bar_x))
        strength = self._goal_shade_strength
        # The bars are round: each pixel row across a post (column across the crossbar) gets the
        # cylinder's shade for that position under the scene light, so the frame reads as tubes
        # rather than flat strips. Posts run along x (width across y), the crossbar along y.
        # The light is treated as a point above the middle of the goal / pitch: each part is lit
        # on the side facing the centre and shaded on its OUTSIDE (top post: shaded on top;
        # bottom post: on the bottom; crossbar: on the side away from the pitch).
        foot_r = self._goal_foot_radius()
        for y in (g.y_top, g.y_bot):
            top_post = y == g.y_top
            # The foot (a dome at the post's base) goes UNDER the post: only the part beyond the
            # post's end and any bulge wider than it shows.
            foot = self._shaded_foot(foot_r, white, strength, flip_x=left, flip_y=top_post)
            fh = foot.get_width() // 2
            surface.blit(foot, (g.line_x - fh, y - fh))
            for i, f in enumerate(cylinder_shades(post_w, "y", strength, flip=top_post)):
                surface.fill(self._shaded(white, f), pygame.Rect(p_a, y - half + i, p_b - p_a + 1, 1))
        for i, f in enumerate(cylinder_shades(post_w, "x", strength, flip=left)):
            surface.fill(self._shaded(white, f), pygame.Rect(g.bar_x - half + i, g.y_top - half, 1, g.y_bot - g.y_top + post_w))
        # The two top corners are mitred so the shadow runs on round the corner from post to bar
        # instead of the bar's column shading cutting straight across the post's end: each corner
        # pixel takes the shade at its distance from the NEARER outer edge (the seam is the
        # diagonal from the outer to the inner corner). Post and bar shade identically because
        # the light is at exactly 45 degrees, so the profile is continuous across the seam.
        outer_first = cylinder_shades(post_w, "x", strength, flip=True)  # index = distance from the outer edge
        for top_corner in (True, False):
            y0 = (g.y_top if top_corner else g.y_bot) - half
            for j in range(post_w):
                b = j if top_corner else post_w - 1 - j
                for i in range(post_w):
                    a = i if left else post_w - 1 - i
                    surface.set_at((g.bar_x - half + i, y0 + j), self._shaded(white, outer_first[min(a, b)]))

    @staticmethod
    def _shaded(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return (round(colour[0] * factor), round(colour[1] * factor), round(colour[2] * factor))

    def _goal_foot_radius(self) -> float:
        """Radius in px of the dome at the base of a post: a bit proud of the post's own
        half-width, and proud by less on thin posts so it stays small at 1x (0.25px beyond
        the post at 3px wide, growing 0.25px per extra post pixel up to 1px)."""
        w = self._goal_post_px()
        return w / 2.0 + min(1.0, 0.25 + max(0, w - 3) * 0.25)

    def _shaded_foot(
        self, radius: float, colour: tuple[int, int, int], strength: float,
        flip_x: bool = False, flip_y: bool = False,
    ) -> pygame.Surface:
        """The foot of a post: a shaded, anti-aliased dome (a sphere under the scene light),
        a square of side ``2 * (ceil(radius) + 1) + 1`` px with its centre pixel in the middle.
        Coverage and shade are computed per pixel from a 8x8 sub-sample grid (so the edge is
        anti-aliased without smoothscale darkening it) and the sprite is cached."""
        key = (round(radius, 3), colour, round(strength, 4), flip_x, flip_y)
        sprite = self._foot_cache.get(key)
        if sprite is None:
            margin = math.ceil(radius) + 1
            size, sub = 2 * margin + 1, 8
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            c = margin + 0.5  # pixel-centre coordinates: the centre pixel spans [margin, margin+1)
            for py in range(size):
                for px in range(size):
                    cover, acc = 0, 0.0
                    for sy in range(sub):
                        for sx in range(sub):
                            dx = (px + (sx + 0.5) / sub - c) / radius
                            dy = (py + (sy + 0.5) / sub - c) / radius
                            r2 = dx * dx + dy * dy
                            if r2 <= 1.0:
                                cover += 1
                                acc += surface_shade(dx, dy, math.sqrt(1.0 - r2), strength, flip_x, flip_y)
                    if cover:
                        shade = self._shaded(colour, acc / cover)
                        sprite.set_at((px, py), (*shade, round(255 * cover / (sub * sub))))
            self._foot_cache[key] = sprite
        return sprite

    def _draw_goal_back_net(self, surface: pygame.Surface, pitch: Pitch, *, left: bool) -> None:
        """The net on the back wall: the strip between where the net meets the
        ground and its top-back edge. Same anti-aliased mesh as the roof (see
        ``_net_layer``), but with the diagonals drawn as the vertical wall's mesh
        actually projects (steep, see ``back_wall_net_segments``)."""
        g = self._goal_px(pitch, left)
        x_a, x_b = sorted((g.back_x, g.back_top_x))
        length, h = x_b - x_a, g.y_bot - g.y_top + 1
        if length < 1 or h < 2:
            return
        w = length + 1
        scale = self._NET_SUPERSAMPLE
        line_w = max(1, round(_GOAL_NET_LINE_PX * scale * self._GOAL_BACK_NET_LINE_WEIGHT))
        rise = pitch.goal_height_m * self.camera.pixels_per_metre
        # Rounded: a scale like 9/6 gives 8.999999999999998 px, which pygame would truncate.
        gap = round(self._goal_net_gap_px() * self._goal_back_net_gap_scale, 4)
        # Only ONE family is rasterised (the lines whose y grows with x); `_net_layer`
        # mirrors it top-to-bottom for the other, so the two are exact mirror images.
        family = [seg for seg in back_wall_net_segments(length, rise, h, gap) if seg[1][1] > seg[0][1]]

        def draw_family(big: pygame.Surface, colour: tuple[int, int, int, int]) -> None:
            for (x0, y0), (x1, y1) in family:
                pygame.draw.line(big, colour, (x0 * scale, y0 * scale), (x1 * scale, y1 * scale), line_w)

        layer = self._net_layer(
            ("back", w, h, gap, rise, line_w, left), w, h, draw_family, mirror_y=True,
            flip_x=left,  # the segments put the ground edge at local x=0; for the left goal it is the strip's right side
        )
        surface.blit(layer, (x_a, g.y_top))

    def _net_layer(
        self, key: tuple, w: int, h: int, draw_family, *, mirror_x: bool = False, mirror_y: bool = False,
        flip_x: bool = False,
    ) -> pygame.Surface:
        """A cached, anti-aliased, translucent net-mesh layer (w x h px).

        ``draw_family(big, colour)`` draws ONE family of parallel diagonal lines
        onto the supersampled surface; the other family is that surface mirrored
        (left-right for ``mirror_x``, top-bottom for ``mirror_y``) and merged with
        ``BLEND_RGBA_MAX`` (so crossings aren't double-brightened). Mirroring rather
        than drawing both families matters: ``pygame.draw.line`` rasterises
        through pixel centres, i.e. offset (+0.5, +0.5) from its coordinates -- along
        the line for one diagonal but *perpendicular* to it for the other -- so two
        mirrored lines came out as different pixel profiles (measured on a single
        line: one diagonal a narrow core over 3 pixels, the other split over 2 pixels half a
        pixel off, looking wider and dimmer). A mirrored family is identical by
        construction. ``flip_x`` mirrors the finished layer (for placement).

        Drawn at ``_NET_SUPERSAMPLE``x and ``smoothscale``d down; the background is the
        line colour at alpha 0, NOT transparent black (smoothscale averages RGB and
        alpha independently, so black would darken every edge pixel -- measured ~26%
        of net pixels darker than the grass); the net's translucency is applied
        once to the whole layer with ``set_alpha``. Built once per ``key`` and cached
        (the mesh only depends on its size, spacing and colour), so the finer
        supersample costs nothing per frame."""
        rgb = self._goal_net_colour
        full_key = key + (tuple(rgb), self._goal_net_alpha, mirror_x, mirror_y, flip_x)
        layer = self._net_layer_cache.get(full_key)
        if layer is None:
            scale = self._NET_SUPERSAMPLE
            big = pygame.Surface((w * scale, h * scale), pygame.SRCALPHA)
            big.fill((*rgb, 0))
            draw_family(big, (*rgb, 255))
            mirrored = pygame.transform.flip(big, mirror_x, mirror_y)
            big.blit(mirrored, (0, 0), special_flags=pygame.BLEND_RGBA_MAX)
            layer = pygame.transform.smoothscale(big, (w, h))
            if flip_x:
                layer = pygame.transform.flip(layer, True, False)
            layer.set_alpha(self._goal_net_alpha)
            if len(self._net_layer_cache) >= 64:  # zoom changes make new sizes; keep this bounded
                self._net_layer_cache.clear()
            self._net_layer_cache[full_key] = layer
        return layer

    def _goal_net_gap_px(self) -> int:
        """Horizontal gap, in px, between the net's parallel diagonal mesh
        lines: the physical mesh size (`goal_net.spacing_m`) at the current
        zoom, but never below `goal_net.min_spacing_px`. The floor is what
        keeps it a diamond lattice at every zoom: with 1px lines, a gap of ~3-4px
        (what 0.35m works out to at the default window) aliases into a dense
        checkerboard. It defaults to 7px, chosen by eye from 6-16px sweeps at 1x
        (~5px is still blobby; 6 reads a bit checker-like now that both diagonals are
        drawn identically; 8px is cleaner but leaves only ~2 diamonds across the
        15px-wide roof)."""
        return max(self._goal_net_min_spacing_px, int(self._goal_net_spacing_m * self.camera.pixels_per_metre))

    def _draw_goal_net(
        self, surface: pygame.Surface, x_a: int, x_b: int, y_top: int, y_bot: int, sag_dx_px: float = 0.0,
    ) -> None:
        """Fills the screen-space rectangle (inclusive pixel bounds) with the
        translucent, anti-aliased 45-degree diamond net mesh (a cached layer, see
        ``_net_layer``: one diagonal family rasterised, the other its exact
        mirror).

        ``sag_dx_px``: how far, in px along x, the deepest point of a sagging roof is
        displaced (toward the pitch is the sign the caller passes). The mesh points are
        displaced by ``sag_dx_px * (1-u^2) * (1-v^2)`` (u, v in [-1, 1] across and along the
        roof), so the frame edges stay put and the middle bows. Below ~0.05px it is drawn
        as the plain flat lattice."""
        w, h = x_b - x_a + 1, y_bot - y_top + 1
        if w < 2 or h < 2:
            return
        scale = self._NET_SUPERSAMPLE
        spacing = self._goal_net_gap_px()
        line_w = max(1, round(_GOAL_NET_LINE_PX * scale * self._GOAL_NET_LINE_SUPERSAMPLE_BOOST))

        if abs(sag_dx_px) >= 0.05:
            def draw_sag_family(big: pygame.Surface, colour: tuple[int, int, int, int]) -> None:
                for i in range(-h, w, spacing):
                    pts = []
                    for k in range(0, 2 * h + 1):
                        y = k / 2.0
                        x = i + y
                        if 0.0 <= x <= w:
                            u, v = (y - h / 2.0) / (h / 2.0), (x - w / 2.0) / (w / 2.0)
                            pts.append(((x + sag_dx_px * (1.0 - u * u) * (1.0 - v * v)) * scale, y * scale))
                    if len(pts) > 1:
                        pygame.draw.lines(big, colour, False, pts, line_w)

            # Mirrored top-to-bottom, NOT left-to-right: the sag displacement is symmetric in y
            # but not in x, so only a vertical mirror keeps the two diagonal families identical.
            layer = self._net_layer(
                ("roof-sag", w, h, spacing, line_w, round(sag_dx_px, 2)), w, h, draw_sag_family, mirror_y=True,
            )
            surface.blit(layer, (x_a, y_top))
            return

        def draw_family(big: pygame.Surface, colour: tuple[int, int, int, int]) -> None:
            for i in range(-h, w, spacing):
                pygame.draw.line(big, colour, (i * scale, 0), ((i + h) * scale, h * scale), line_w)

        layer = self._net_layer(("roof", w, h, spacing, line_w), w, h, draw_family, mirror_x=True)
        surface.blit(layer, (x_a, y_top))

    def draw_pitch_and_ball(
        self, surface: pygame.Surface, pitch: Pitch, ball: Ball, players: Sequence[Player] = (),
        selected_id: str | None = None,
    ) -> None:
        """Draws the pitch and the ball with the goal's top layer (roof net,
        posts, crossbar) on the correct side of the ball, for a sense of depth:
        a ball that is inside a goal (past the goal line, between the posts,
        below the crossbar -- i.e. going in) is drawn UNDER the crossbar and
        net; any other ball -- including one above the crossbar, going over --
        is drawn OVER them. Callers draw players after this call.

        Every ground shadow -- each of ``players``' and the ball's -- is drawn in ONE pass
        right after the pitch's ground layer, i.e. under the goal frame, the ball, and every
        sprite (a shadow never falls over another player, the ball, or the net). The translucent
        RINGS -- every player's state / selection / stamina ring (``selected_id`` is the selected
        player's id) and the ball's state ring -- come next, still under the goal frame, the ball
        and every sprite. Then every player's LEGS layer (see ``draw_player_legs``), so a player
        standing over/near the ball shows it resting at their feet; callers draw the rest of each
        player (``draw_player(..., legs=False)``) after this call, so the upper body ends up in
        front of the ball."""
        under = ball_under_goal_frame(
            pitch, ball.position.x, ball.position.y, ball.position.z, ball.radius_m,
        )
        self.draw_pitch(surface, pitch, goal_tops=False)
        self.draw_player_shadows(surface, players)
        if self._ball_shadow_enabled:
            self._draw_ball_shadow(surface, ball, self._ball_base_radius_px(ball))
        self.draw_player_rings(surface, players, selected_id)
        self._draw_ball_state_ring(surface, ball)
        if not under:
            self.draw_goal_tops(surface, pitch)
        self.draw_player_legs(surface, players)
        self.draw_ball(surface, ball, shadow=False, ring=False)
        if under:
            self.draw_goal_tops(surface, pitch)

    def _draw_defending_marker(
        self, surface: pygame.Surface, back_x: float, half_goal_w: float,
        colour: tuple[int, int, int], faces_positive_x: bool,
    ) -> None:
        """A translucent bar set back a little behind the goal net, in the
        defending team's colour, spanning a bit wider than the goal mouth
        so it reads at a glance from the whole penalty area."""
        cam = self.camera
        gap_m = 0.3
        bar_depth_m = 1.0
        overhang_m = 1.5
        x0 = back_x + gap_m if faces_positive_x else back_x - gap_m
        x1 = x0 + bar_depth_m if faces_positive_x else x0 - bar_depth_m
        p0 = cam.world_to_screen(x0, -(half_goal_w + overhang_m))
        p1 = cam.world_to_screen(x1, half_goal_w + overhang_m)
        left, top = min(p0[0], p1[0]), min(p0[1], p1[1])
        w, h = max(1, abs(p1[0] - p0[0])), max(1, abs(p1[1] - p0[1]))
        bar_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        bar_surf.fill((*colour, style.DEFENDING_SIDE_MARKER_ALPHA))
        surface.blit(bar_surf, (left, top))

    def _draw_corner_flags(self, surface: pygame.Surface, pitch: Pitch) -> None:
        """A corner flag at each of the 4 pitch corners, with the same
        parallax as the goal frame: the pole is vertical, so its top is
        displaced *away from the pitch centre* (see ``corner_flag_world_points``)
        -- a short pole visible from the corner out toward the top -- with the
        cloth attached along the upper part of the pole and streaming sideways off
        it. The pole's apparent length uses the crossbar's parallax rate
        (``crossbar_lean_m / goal_height_m`` metres per metre of height), so it
        scales with that one setting; ``corner_flag`` in graphics.json sets the
        pole height, cloth size and shadow. Each flag casts a soft shadow from the scene light
        (`_draw_corner_flag_shadow`), drawn first so the pole and cloth sit on top of it."""
        cam = self.camera
        # 1px minimum (a 2px pole looked chunky and rectangular at the default zoom),
        # and the ground-contact dot appears only once the pole is 3px+ wide, sized to
        # the pole -- a fixed 2px-radius dot is a 5px blob on the ~6px pole at 1x zoom
        # and hides it entirely.
        pole_px = max(1, round(self._corner_pole_width_m * cam.pixels_per_metre))
        foot_r = pole_px // 2 + 1 if pole_px >= 3 else 0
        for sx in (-1, 1):
            for sy in (-1, 1):
                geo = self._corner_flag_geometry(pitch, sx, sy)
                if self._corner_shadow_alpha > 0:
                    self._draw_corner_flag_shadow(surface, geo, self._corner_flag_drop(pitch))
                self._draw_world_polyline(surface, [geo.base, geo.top], pole_px, style.CORNER_FLAG_POLE_COLOUR)
                if foot_r:
                    foot = cam.world_to_screen(*geo.base)
                    pygame.gfxdraw.filled_circle(surface, foot[0], foot[1], foot_r, style.CORNER_FLAG_POLE_COLOUR)
                    pygame.gfxdraw.aacircle(surface, foot[0], foot[1], foot_r, style.CORNER_FLAG_POLE_COLOUR)
                tri = [cam.world_to_screen(*pt) for pt in (geo.top, geo.attach, geo.tip)]
                pygame.gfxdraw.filled_polygon(surface, tri, style.CORNER_FLAG_COLOUR)
                pygame.gfxdraw.aapolygon(surface, tri, style.CORNER_FLAG_COLOUR)

    def _corner_flag_drop(self, pitch: Pitch) -> float:
        """Fraction of the pole (from the top) the cloth is attached along: ``flag_length_m`` as a
        share of the pole's apparent length, so the pennant keeps the same size however tall the
        pole is."""
        lean_m = self._goal_lean_m(pitch) / pitch.goal_height_m * self._corner_pole_height_m
        return min(0.95, self._corner_flag_length_m / lean_m) if lean_m > 1e-9 else 0.95

    def _corner_flag_geometry(self, pitch: Pitch, sx: int, sy: int) -> CornerFlagGeometry:
        rate = self._goal_lean_m(pitch) / pitch.goal_height_m
        return corner_flag_world_points(
            pitch, sx, sy, pole_lean_m=rate * self._corner_pole_height_m,
            flag_width_m=self._corner_flag_width_m, flag_drop_frac=self._corner_flag_drop(pitch),
        )

    def _draw_corner_flag_shadow(self, surface: pygame.Surface, geo: CornerFlagGeometry, drop: float) -> None:
        """The flag's ground shadow from the scene light (a point ``scene_light.height_m`` above the
        pitch's middle): a point at height z above ground point P throws its shadow at
        P * H / (H - z) from the centre, i.e. displaced straight away from it. The pole (height
        ``shadow_height_m`` -- deliberately short: the drawn pole is a stub, so a physically tall pole's
        ~4m shadow at a corner looked absurd next to it) casts a line from its
        foot and the cloth a triangle. Drawn 2x supersampled on a small transparent layer and
        averaged down (soft, anti-aliased edges; the layer is black so there is no edge halo), then
        blitted at ``shadow_alpha``."""
        cam = self.camera
        light_h, pole_h = self._light_height_m, self._corner_shadow_height_m

        def shadow_of(x: float, y: float, z: float) -> tuple[float, float]:
            k = light_h / max(light_h - z, 1.0)
            return x * k, y * k

        bx, by = geo.base
        mid = ((geo.top[0] + geo.attach[0]) / 2.0, (geo.top[1] + geo.attach[1]) / 2.0)
        tip_xy = (bx + geo.tip[0] - mid[0], by + geo.tip[1] - mid[1])       # the cloth's true (unleaned) end
        world = [
            geo.base,
            shadow_of(bx, by, pole_h),
            shadow_of(bx, by, pole_h * (1.0 - drop)),
            shadow_of(tip_xy[0], tip_xy[1], pole_h * (1.0 - drop / 2.0)),
        ]
        pts = [cam.world_to_screen_f(*p) for p in world]
        pad, ss = 4, 2
        x0, y0 = int(math.floor(min(p[0] for p in pts))) - pad, int(math.floor(min(p[1] for p in pts))) - pad
        w = int(math.ceil(max(p[0] for p in pts))) + pad - x0 + 1
        h = int(math.ceil(max(p[1] for p in pts))) + pad - y0 + 1
        if not pygame.Rect(x0, y0, w, h).colliderect(surface.get_clip()):
            return
        layer = pygame.Surface((w * ss, h * ss), pygame.SRCALPHA)
        layer.fill((0, 0, 0, 0))
        local = [((p[0] - x0) * ss, (p[1] - y0) * ss) for p in pts]
        pole_w = max(ss, round(0.16 * cam.pixels_per_metre * ss))
        pygame.draw.line(layer, (0, 0, 0, 255), local[0], local[1], pole_w)
        tri = [(int(round(x)), int(round(y))) for x, y in local[1:]]
        pygame.gfxdraw.filled_polygon(layer, tri, (0, 0, 0, 255))
        pygame.gfxdraw.aapolygon(layer, tri, (0, 0, 0, 255))
        soft = pygame.transform.smoothscale(layer, (w, h))
        soft.set_alpha(self._corner_shadow_alpha)
        surface.blit(soft, (x0, y0))

    # x-offsets (metres, from the halfway line) of each bench along a
    # touchline -- spread across the middle third of the pitch, well clear
    # of the penalty boxes and corners regardless of pitch size.
    _BENCH_X_OFFSETS_M = (-16.0, -8.5, -1.0, 6.5, 14.0)

    def _draw_sideline_benches(self, surface: pygame.Surface, pitch: Pitch) -> None:
        """A row of simple technical-area benches just outside each
        touchline -- pure dressing, no gameplay meaning."""
        cam = self.camera
        bench_len_m, bench_depth_m, gap_m = 5.0, 1.1, 0.6
        for side in (-1, 1):  # -1 = below the pitch (screen), 1 = above
            y0 = side * (pitch.half_width + gap_m)
            y1 = side * (pitch.half_width + gap_m + bench_depth_m)
            for x_centre in self._BENCH_X_OFFSETS_M:
                p0 = cam.world_to_screen(x_centre - bench_len_m / 2.0, y0)
                p1 = cam.world_to_screen(x_centre + bench_len_m / 2.0, y1)
                left, top = min(p0[0], p1[0]), min(p0[1], p1[1])
                w, h = max(1, abs(p1[0] - p0[0])), max(1, abs(p1[1] - p0[1]))
                rect = (left, top, w, h)
                pygame.draw.rect(surface, style.BENCH_SEAT_COLOUR, rect, border_radius=2)
                pygame.draw.rect(surface, style.BENCH_OUTLINE_COLOUR, rect, 1, border_radius=2)

    _TURF_PATCH_PX_PER_M = 4      # resolution of the world-anchored patch texture
    _TURF_PATCH_MARGIN_M = 15.0   # how far beyond the pitch the texture extends
    _TURF_GAIN = 0.95             # mean of the grain/vignette overlay; the base colour is pre-divided by it

    def _turf_patch_surface(self, pitch: Pitch) -> pygame.Surface:
        """World-anchored, low-frequency light/dark patches of the grass: two octaves of smooth
        value noise, slightly warmer where lighter, as a colour texture (pitch green times a
        factor of about 1 +/- ``patch_amp``) over the pitch plus a margin. Built once."""
        key = (pitch.half_length, pitch.half_width, self._turf_patch_amp, self._turf_patch_cells_m, self._turf_seed)
        if self._turf_patch is not None and self._turf_patch_key == key:
            return self._turf_patch
        res = self._TURF_PATCH_PX_PER_M
        tw = int(math.ceil(2 * (pitch.half_length + self._TURF_PATCH_MARGIN_M) * res))
        th = int(math.ceil(2 * (pitch.half_width + self._TURF_PATCH_MARGIN_M) * res))
        rng = np.random.default_rng(self._turf_seed)

        def octave(cell_m: float) -> np.ndarray:
            cell = max(1.0, cell_m * res)
            gw, gh = int(tw / cell) + 3, int(th / cell) + 3
            grid = rng.random((gw, gh))
            fx, fy = np.arange(tw) / cell, np.arange(th) / cell
            x0, y0 = fx.astype(int), fy.astype(int)
            tx, ty = fx - x0, fy - y0
            tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)   # smoothstep fade: soft blobs, no creases
            top = grid[x0][:, y0] * (1 - ty) + grid[x0][:, y0 + 1] * ty
            bot = grid[x0 + 1][:, y0] * (1 - ty) + grid[x0 + 1][:, y0 + 1] * ty
            return top * (1 - tx)[:, None] + bot * tx[:, None]

        n = 0.65 * octave(self._turf_patch_cells_m[0]) + 0.35 * octave(self._turf_patch_cells_m[1]) - 0.5
        n = n / max(float(np.abs(n).max()), 1e-6)
        amp = self._turf_patch_amp
        factor = np.stack([1 + 0.3 * amp * n, 1 + amp * n, 1 - 0.25 * amp * n], axis=-1)
        rgb = np.clip(np.array(style.PITCH_GREEN, dtype=float) / self._TURF_GAIN * factor, 0, 255).astype(np.uint8)
        self._turf_patch = pygame.surfarray.make_surface(rgb)
        self._turf_patch_key = key
        return self._turf_patch

    def _turf_overlay_surface(self, w: int, h: int) -> pygame.Surface:
        """Screen-sized multiply layer: a little per-pixel grain and a soft vignette (darker
        toward the corners), with mean ``_TURF_GAIN`` so it averages back to the pitch green."""
        key = (w, h, self._turf_noise_amp, self._turf_vignette, self._turf_seed)
        if self._turf_overlay is not None and self._turf_overlay_key == key:
            return self._turf_overlay
        rng = np.random.default_rng(self._turf_seed + 1)
        xs = (np.arange(w) - (w - 1) / 2) / (w / 2)
        ys = (np.arange(h) - (h - 1) / 2) / (h / 2)
        d = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2) / math.sqrt(2)   # 0 centre .. 1 corner
        vig = 1.0 - self._turf_vignette * d ** 2.2
        grain = 1.0 + self._turf_noise_amp * rng.standard_normal((w, h))
        g = np.clip(self._TURF_GAIN * vig * grain, 0.0, 1.0)
        v = (g * 255.0 + 0.5).astype(np.uint8)
        self._turf_overlay = pygame.surfarray.make_surface(np.stack([v, v, v], axis=-1))
        self._turf_overlay_key = key
        return self._turf_overlay

    def _draw_turf(self, surface: pygame.Surface, pitch: Pitch) -> None:
        """The grass: the world-anchored patch texture scaled to the camera, times a
        screen-space grain + vignette layer. The composed background is cached and only
        rebuilt when the camera actually moves (a static camera costs one blit)."""
        cam = self.camera
        w, h = surface.get_size()
        margin = self._TURF_PATCH_MARGIN_M
        x0, y0 = cam.world_to_screen_f(-pitch.half_length - margin, -pitch.half_width - margin)
        x1, y1 = cam.world_to_screen_f(pitch.half_length + margin, pitch.half_width + margin)
        left, right = sorted((x0, x1))
        top, bot = sorted((y0, y1))
        key = (w, h, round(left, 1), round(top, 1), round(right, 1), round(bot, 1),
               self._turf_patch_amp, self._turf_patch_cells_m, self._turf_noise_amp, self._turf_vignette, self._turf_seed)
        if self._turf_bg is None or self._turf_bg_key != key:
            patch = self._turf_patch_surface(pitch)
            tw, th = patch.get_size()
            bg = pygame.Surface((w, h))
            bg.fill(tuple(min(255, round(c / self._TURF_GAIN)) for c in style.PITCH_GREEN))
            sx, sy = (right - left) / tw, (bot - top) / th        # screen px per texture px
            if sx > 0 and sy > 0:
                # The part of the texture that is on screen, in whole texture pixels.
                tx0, ty0 = max(0, math.floor((0 - left) / sx)), max(0, math.floor((0 - top) / sy))
                tx1, ty1 = min(tw, math.ceil((w - left) / sx)), min(th, math.ceil((h - top) / sy))
                if tx1 > tx0 and ty1 > ty0:
                    dest = pygame.Rect(
                        round(left + tx0 * sx), round(top + ty0 * sy),
                        max(1, round((tx1 - tx0) * sx)), max(1, round((ty1 - ty0) * sy)),
                    )
                    src = patch.subsurface(pygame.Rect(tx0, ty0, tx1 - tx0, ty1 - ty0))
                    bg.blit(pygame.transform.smoothscale(src, dest.size), dest.topleft)
            bg.blit(self._turf_overlay_surface(w, h), (0, 0), special_flags=pygame.BLEND_RGB_MULT)
            self._turf_bg, self._turf_bg_key = bg, key
        surface.blit(self._turf_bg, (0, 0))

    _RING_SUPERSAMPLE = 4
    # The net meshes are built once and cached (see `_net_layer`), so they can afford a
    # finer supersample than the per-frame rings/dots.
    _NET_SUPERSAMPLE = 8
    # Weight of the net mesh at supersample size, tuned so its total ink after the
    # anti-aliased downscale matches the old aliased 1px mesh (~0.17-0.20 of the
    # available lightening; measured by tests/unit/test_pitch_markings.py's
    # ink metric). Raise it for a heavier net.
    _GOAL_NET_LINE_SUPERSAMPLE_BOOST = 1.0
    # pygame's thick lines are thicker (perpendicular to the line) the steeper they
    # are, so the steep back-wall mesh is drawn at a lighter weight to keep about the
    # same ink as the roof mesh.
    _GOAL_BACK_NET_LINE_WEIGHT = 0.75
    _DOT_POLY_SEGMENTS = 14  # vertices approximating each ball spin-dot's outline

    def _aa_disc(self, rgb: tuple[int, int, int], radius: int, alpha: int = 255) -> pygame.Surface:
        """A filled, anti-aliased disc of *radius* px on a transparent
        (radius*2+2)-square layer, centred at (radius+1, radius+1), with
        overall opacity *alpha*. For small translucent discs (ball trail,
        inactive-player fallback) where `draw.circle` is aliased and
        `gfxdraw.aacircle` doesn't blend correctly onto a transparent layer.
        Supersampled like `_draw_ring`; cached per (colour, radius) -- the
        returned surface is shared, so blit it immediately."""
        key = (tuple(rgb), radius)
        disc = self._aa_disc_cache.get(key)
        if disc is None:
            scale = self._RING_SUPERSAMPLE
            size = radius * 2 + 2
            big = pygame.Surface((size * scale, size * scale), pygame.SRCALPHA)
            big.fill((*rgb, 0))  # see _draw_goal_net: keep the edge RGB from averaging toward black
            c = (radius + 1) * scale
            pygame.draw.circle(big, (*rgb, 255), (c, c), radius * scale)
            disc = pygame.transform.smoothscale(big, (size, size))
            self._aa_disc_cache[key] = disc
        disc.set_alpha(alpha)
        return disc

    @classmethod
    def _draw_ring(
        cls, surface: pygame.Surface, colour: tuple[int, int, int],
        pos: tuple[int, int], outer_radius: int, width: int, alpha: int = 255,
    ) -> None:
        """Draws a smooth circular outline (a "ring"/annulus) around a player
        or the ball (selection/control-delay/inactive/stamina-flash/
        ball-state indicators), at overall opacity ``alpha`` (0-255).

        This deliberately avoids drawing a single ``pygame.draw.circle(...,
        width=N)`` at game resolution: at the small radii these rings use
        (roughly 8-20px), pygame-ce's circle rasteriser has a real, visible
        raggedness -- sampling actual rendered frames pixel-by-pixel found
        genuine isolated single-pixel gaps of pure background punched
        through an otherwise-solid stroke (a ring pixel, then one pure
        background pixel, then a ring pixel again, at the same angle),
        worse on the SE side of the circle and, to a lesser extent, the NW
        side. This is what looked like players "having a chunk eaten out of
        them" -- the ring sits close enough to the player's own fill circle
        that the hole reads as part of the player at normal zoom. Trying to
        patch this by building the ring from two separately-rasterised
        filled circles (an opaque outer disc plus a transparent inner
        "punch") does not help either -- each circle has its own staircase
        edge and the two don't line up in phase, which turned the single
        stray hole into a visibly dashed/toothed ring instead.

        Fix: render the ring at `_RING_SUPERSAMPLE`x resolution (where a
        multi-pixel gap in the oversized rasterisation is a sub-pixel sliver
        once scaled back down) and downsample with ``smoothscale``, which
        area-averages rather than picking one sample point -- any leftover
        staircase noise gets blended into a smooth anti-aliased edge instead
        of surviving as a hole. This is the standard way to get a clean
        result out of primitives that don't rasterise well at small sizes.
        """
        scale = cls._RING_SUPERSAMPLE
        pad = 2
        size = (outer_radius + pad) * 2
        big = pygame.Surface((size * scale, size * scale), pygame.SRCALPHA)
        # Background = the ring's own colour at alpha 0 (not transparent black),
        # or smoothscale averages black into the ring's edge pixels and dims them.
        big.fill((*colour[:3], 0))
        c = (size * scale) // 2
        pygame.draw.circle(big, colour, (c, c), outer_radius * scale, width * scale)
        small = pygame.transform.smoothscale(big, (size, size))
        if alpha < 255:
            small.set_alpha(alpha)
        surface.blit(small, (pos[0] - size // 2, pos[1] - size // 2))

    def _spin_dot_radius_px(self, radius_px: int) -> float:
        """A spin dot's drawn radius in pixels, as a float. `radius_px` (the ball's own drawn
        radius) is itself already an int -- the ball's outer circle has to be, for
        `smoothscale`/`pygame.draw` -- so truncating this to an int AGAIN on top of that made the
        dot/ball size ratio swing between ~0.17 and ~0.25 across zoom levels for a configured 0.25
        (measured before fixing it): two roundings compounding at these small pixel counts, not a
        real design choice. Kept as a float here and only rounded at the final per-vertex pixel
        snap (like every other coordinate in `_ball_dots_layer` already does), the ratio stays at
        the configured fraction regardless of zoom."""
        return max(1.0, radius_px * self._spin_dot_radius_frac)

    def _ball_dots_layer(
        self, radius_px: int, orientation=None, dot_positions=None,
    ) -> tuple[pygame.Surface, int]:
        """The ball's spin dots as an SRCALPHA layer (and its half-size ``pad``,
        for centring it on the ball), clipped to the ball's disc.

        Drawn at ``_RING_SUPERSAMPLE``x resolution with plain
        ``pygame.draw.polygon`` and downscaled with ``smoothscale`` -- the same
        technique as ``_draw_ring``. The previous ``gfxdraw.filled_polygon`` +
        ``gfxdraw.aapolygon`` pair, both alpha-blending onto a transparent
        layer, left visible defects on these small dots (~8px across at
        normal zoom): the fill stops short of the polygon's four extreme
        vertices while the AA pass only covers 10-30% of the pixels there, so
        near-white pixels (measured 210-248 against a dark ~85 interior) sat
        in the middle of the dark dot as cross-shaped notches, and the
        double-blended alpha made the interior lighter (~85) than the
        intended colour (~62 on a white ball). Supersample-then-downscale
        area-averages instead, so edges get a proper gradient and there are
        no holes; the clip disc is supersampled the same way.

        ``orientation`` / ``dot_positions`` default to the live ball's; they
        are parameters so the layer can be unit-tested for a single dot."""
        scale = self._RING_SUPERSAMPLE
        orbit_r = radius_px * self._spin_orbit_frac
        dot_r = self._spin_dot_radius_px(radius_px)
        pad = int(math.ceil(orbit_r + dot_r)) + 2
        big_size = pad * 2 * scale
        centre = pad * scale
        big = pygame.Surface((big_size, big_size), pygame.SRCALPHA)
        R = self._ball_orientation if orientation is None else orientation
        positions = self._ball_dot_positions if dot_positions is None else dot_positions
        dot_epsilon = dot_r / orbit_r if orbit_r > 0 else 0.0
        dot_colour = (*self._spin_dot_color, 220)
        orbit_big = orbit_r * scale
        for (lx, ly, lz) in positions:
            wx = R[0][0]*lx + R[0][1]*ly + R[0][2]*lz
            wy = R[1][0]*lx + R[1][1]*ly + R[1][2]*lz
            wz = R[2][0]*lx + R[2][1]*ly + R[2][2]*lz
            if wz < 0:
                continue  # back hemisphere -- hidden from top-down camera
            # Tangent basis (u, v) perpendicular to (wx, wy, wz): pick an
            # "up" reference not nearly parallel to it, to keep the cross
            # product well-conditioned near the poles.
            up = (0.0, 1.0, 0.0) if abs(wz) > 0.9 else (0.0, 0.0, 1.0)
            ux, uy, uz = wy*up[2] - wz*up[1], wz*up[0] - wx*up[2], wx*up[1] - wy*up[0]
            ulen = math.sqrt(ux*ux + uy*uy + uz*uz) or 1.0
            ux, uy, uz = ux/ulen, uy/ulen, uz/ulen
            vx, vy, vz = wy*uz - wz*uy, wz*ux - wx*uz, wx*uy - wy*ux
            poly = []
            for k in range(self._DOT_POLY_SEGMENTS):
                theta = 2 * math.pi * k / self._DOT_POLY_SEGMENTS
                ct, st = math.cos(theta), math.sin(theta)
                ex = wx + dot_epsilon * (ct*ux + st*vx)
                ey = wy + dot_epsilon * (ct*uy + st*vy)
                ez = wz + dot_epsilon * (ct*uz + st*vz)
                elen = math.sqrt(ex*ex + ey*ey + ez*ez) or 1.0
                poly.append((round(centre + (ex/elen)*orbit_big), round(centre - (ey/elen)*orbit_big)))
            pygame.draw.polygon(big, dot_colour, poly)
        # Clip dots inside the outline so they don't bleed over the alpha border.
        clip = pygame.Surface((big_size, big_size), pygame.SRCALPHA)
        clip_r = max(1, radius_px - int(max(1, self._ball_outline_width)))
        pygame.draw.circle(clip, (255, 255, 255, 255), (centre, centre), clip_r * scale)
        big.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        return pygame.transform.smoothscale(big, (pad * 2, pad * 2)), pad

    def _ball_base_radius_px(self, ball: Ball) -> float:
        """The ball's on-screen radius before the height boost (see `draw_ball`). The minimum-
        visibility floor (`min_ball_radius_px`) grows with zoom, but SLOWER than true size does
        (`** self._size_floor_zoom_decay`, < 1) -- see that attribute's comment and
        `_player_radius_px`'s docstring for why: at 1x (zoom_scale=1) this is identical to before
        (any exponent of 1 is 1), but as you zoom in, true size (growing linearly) increasingly
        overtakes the floor (growing sub-linearly), so the ball converges toward looking properly
        life-sized rather than staying a constant ~4.6x too big at every zoom forever."""
        cam = self.camera
        floor_px = self.min_ball_radius_px * cam.zoom_scale ** self._size_floor_zoom_decay
        return max(floor_px, cam.scale_length(ball.radius_m))

    def _ball_draw_radius_px(self, ball: Ball) -> int:
        """The ball's drawn radius: the enlarged ground radius times the height boost."""
        height_boost = 1.0 + min(ball.height_m, 5.0) * self._ball_height_boost_per_m
        return max(2, int(self._ball_base_radius_px(ball) * height_boost))

    def _draw_ball_state_ring(self, surface: pygame.Surface, ball: Ball) -> None:
        """The ball's state ring (just bounced amber > flying blue > rolling green), translucent
        (``rings.alpha``). Drawn by `draw_pitch_and_ball` UNDER the ball; `draw_ball(ring=True)` (the
        default, for standalone use) draws it on top."""
        pos = self.camera.world_to_screen(ball.position.x, ball.position.y)
        ring_r = self._ball_draw_radius_px(ball) + self._ring_offset_px
        if self._ring_show_bounced and ball.just_bounced_timer_s > 0.0:
            self._draw_ring(surface, self._ring_color_bounced, pos, ring_r, self._ring_width_px, self._ring_alpha)
        elif self._ring_show_flying and ball.position.z > ball.radius_m + self._flying_min_height_m and ball.possessed_by is None:
            self._draw_ring(surface, self._ring_color_flying, pos, ring_r, self._ring_width_px, self._ring_alpha)
        elif self._ring_show_rolling:
            self._draw_ring(surface, self._ring_color_rolling, pos, ring_r, self._ring_width_px, self._ring_alpha)

    def draw_ball(self, surface: pygame.Surface, ball: Ball, *, shadow: bool = True, ring: bool = True) -> None:
        """``shadow=False`` skips the ground shadow and ``ring=False`` the state ring:
        `draw_pitch_and_ball` draws both in its layers UNDER every sprite instead."""
        cam = self.camera
        pos = cam.world_to_screen(ball.position.x, ball.position.y)

        # A true-to-scale ball (radius 0.11m) is only ~1px at typical zoom
        # levels, so we enforce a minimum on-screen radius for visibility -
        # positions stay physically accurate, only the drawn dot size is
        # boosted. The height effect is then exaggerated on top of that
        # minimum, and a small height label is shown, per the design spec.
        # The floor itself scales with the camera's zoom (see
        # `Camera.zoom_scale`'s docstring) -- otherwise the ball stops
        # growing under the "[Z]" ball-follow zoom as soon as it hits this
        # floor, while players (radius_m 0.3 vs the ball's 0.11) keep
        # growing well past it, leaving the ball looking disproportionately
        # tiny next to them once zoomed in.
        base_radius_px = self._ball_base_radius_px(ball)
        height_boost = 1.0 + min(ball.height_m, 5.0) * self._ball_height_boost_per_m
        radius_px = max(2, int(base_radius_px * height_boost))

        if shadow and self._ball_shadow_enabled:
            self._draw_ball_shadow(surface, ball, base_radius_px)

        # --- Ghost trail: drawn before the ball so it's underneath ---
        samples = list(self._ball_trail)
        n = len(samples)
        if n >= 1:
            # Build interpolated point list (x, y, z).
            # The newest sample is always coincident with the live ball and would be
            # hidden underneath it, so we exclude it from drawing — but keep it in
            # samples for interpolation so the last visible ghost blends smoothly
            # toward the ball position.
            if n >= 2 and self._trail_interp_steps > 1:
                pts: list[tuple[float, float, float]] = []
                for i in range(n - 1):
                    pts.append(samples[i])
                    x0, y0, z0 = samples[i]
                    x1, y1, z1 = samples[i + 1]
                    for s in range(1, self._trail_interp_steps):
                        t = s / self._trail_interp_steps
                        pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
                # omit the final sample (coincident with live ball)
            elif n >= 2:
                pts = samples[:-1]
            else:
                pts = []  # only one sample — it's the live ball position, nothing to draw

            total = len(pts)
            for i, (wx, wy, wz) in enumerate(pts):
                # Oldest = index 0 = most faded; newest = index total-1 = brightest
                age_frac = (total - 1 - i) / max(total - 1, 1)  # 0=newest, 1=oldest
                alpha = int(self._trail_max_alpha * (1.0 - age_frac))
                if alpha < 6:
                    continue
                ghost_boost = 1.0 + min(wz, 5.0) * self._ball_height_boost_per_m
                ghost_base_r = max(1, int(base_radius_px * self._trail_radius_frac * (1.0 - age_frac * self._trail_radius_taper)))
                gr = max(1, int(ghost_base_r * ghost_boost))
                tp = self.camera.world_to_screen(wx, wy)
                surface.blit(self._aa_disc(style.BALL_COLOUR, gr, alpha), (tp[0] - gr - 1, tp[1] - gr - 1))

        pygame.draw.circle(surface, style.BALL_COLOUR, pos, radius_px)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px, style.BALL_COLOUR)
        if self._ball_outline and self._ball_outline_width > 0.0:
            if self._ball_outline_width >= 1.0:
                self._draw_ring(surface, style.BALL_OUTLINE, pos, radius_px, int(self._ball_outline_width))
            else:
                # Sub-pixel: draw 1px outline at reduced alpha for a softer border
                _oa = int(self._ball_outline_width * 255)
                _ots = pygame.Surface((radius_px * 2 + 2, radius_px * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(_ots, (*style.BALL_OUTLINE, _oa), (radius_px + 1, radius_px + 1), radius_px, 1)
                surface.blit(_ots, (pos[0] - radius_px - 1, pos[1] - radius_px - 1))

        # Ball state indicator ring (see `_draw_ball_state_ring`) -- only when drawing the ball standalone.
        if ring:
            self._draw_ball_state_ring(surface, ball)

        # --- Dots: fixed points on the 3D ball surface, projected top-down ---
        # Always shown; rotate as the ball spins. Front hemisphere only.
        # Clipped to the ball circle so dots don't bleed outside the edge.
        #
        # Each dot is a small circular patch on the sphere's surface, not a
        # flat disc floating in front of it -- so under orthographic
        # projection it must foreshorten into an ellipse as it nears the
        # silhouette edge (where the local surface normal is close to
        # perpendicular to the view axis), exactly like the panels on a real
        # football do. Rather than compute that foreshortening as an ad hoc
        # squish-and-rotate, this projects `_DOT_POLY_SEGMENTS` points
        # arranged in a small circle in the dot's own tangent plane (spanned
        # by `u`/`v`, both perpendicular to the dot's centre direction)
        # through the same rotation + orthographic projection as the centre
        # point. The correct ellipse (or near-silhouette sliver) shape falls
        # out automatically, with no separate foreshortening-factor math.
        layer, pad = self._ball_dots_layer(radius_px)
        surface.blit(layer, (pos[0] - pad, pos[1] - pad))

        if self._ball_shading_strength > 0.0:
            shade = self._ball_shade_sprite(radius_px, *self._ball_light_angles(ball))
            surface.blit(shade, (pos[0] - shade.get_width() // 2, pos[1] - shade.get_height() // 2))

        if ball.height_m > 0.15:
            label = self.hud_font.render(f"{ball.height_m:.1f}m", True, style.HUD_TEXT)
            surface.blit(label, (pos[0] + radius_px + 2, pos[1] - label.get_height() // 2))

    def _ball_light_angles(self, ball: Ball) -> tuple[int, int]:
        """Direction of the light on the ball as (azimuth, elevation) in whole degrees, quantised
        (10 / 5 degrees) so sprites can be cached. The light is the SAME point light as the ball's
        shadow (``scene_light.height_m`` above the pitch's middle, `_point_light_at`): horizontally it
        points from the ball toward the centre (azimuth = screen angle of that direction, y down), and
        its elevation is atan((H - ball height) / distance from the centre) -- overhead at the centre,
        lower and more sideways toward the edges."""
        azimuth, elevation, _ = self._point_light_at(ball.position.x, ball.position.y, ball.position.z)
        return int(round(azimuth / 10.0) * 10) % 360, int(min(90, round(elevation / 5.0) * 5))

    def _point_light_at(self, wx: float, wy: float, z_m: float) -> tuple[float, float, float]:
        """The scene's point light (``scene_light.height_m`` above the middle of the pitch) as seen
        from the point (wx, wy) at height z_m: (azimuth in degrees -- the screen-plane angle from
        the point toward the pitch centre, y down --, elevation in degrees above the ground plane,
        distance from the centre in metres). Overhead (90) at the centre, lower toward the edges."""
        cam = self.camera
        bx, by = cam.world_to_screen_f(wx, wy)
        cx, cy = cam.world_to_screen_f(0.0, 0.0)
        dist_m = math.hypot(cx - bx, cy - by) / cam.pixels_per_metre
        rise = max(self._light_height_m - z_m, 0.5)
        elevation = math.degrees(math.atan2(rise, dist_m))
        azimuth = math.degrees(math.atan2(cy - by, cx - bx)) if dist_m > 1e-6 else 0.0
        return azimuth, elevation, dist_m

    def _ball_shade_sprite(self, radius: int, azimuth_deg: int, elevation_deg: int) -> pygame.Surface:
        """Black overlay, per-pixel alpha, that shades the ball like a sphere lit from the given
        direction (see `_ball_light_angles`): dark on the side facing away, clear where it faces the
        light. Anti-aliased (6x sub-sampled coverage), (2r+3)px square centred on the ball, cached
        per (radius, direction). A white ball can't get whiter, so the highlight is simply the
        absence of shading; the overlay goes over the dots too, so they wrap round the sphere."""
        key = (radius, azimuth_deg, elevation_deg)
        sprite = self._ball_shade_cache.get(key)
        if sprite is None:
            ss, size = 6, 2 * radius + 3
            n = size * ss
            yy, xx = np.mgrid[0:n, 0:n].astype(float)
            dx, dy = (xx + 0.5 - n / 2) / (radius * ss), (yy + 0.5 - n / 2) / (radius * ss)
            r2 = dx * dx + dy * dy
            nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
            az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
            lx, ly, lz = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
            lit = np.clip(1.15 * (dx * lx + dy * ly + nz * lz), 0.0, 1.0)
            shade = _SHADE_AMBIENT + (1.0 - _SHADE_AMBIENT) * lit
            # Blinn-Phong highlight (view straight down) cancels the shading where it peaks.
            hx, hy, hz = lx, ly, lz + 1.0
            spec = np.clip((dx * hx + dy * hy + nz * hz) / math.sqrt(hx * hx + hy * hy + hz * hz), 0.0, 1.0) ** 40
            dark = np.clip(self._ball_shading_strength * (1.0 - shade) * (1.0 - spec), 0.0, 1.0) * (r2 <= 1.0)
            alpha = dark.reshape(size, ss, size, ss).mean(axis=(1, 3))
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            sprite.fill((0, 0, 0, 0))
            pygame.surfarray.pixels_alpha(sprite)[:] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8).T
            if len(self._ball_shade_cache) >= 512:
                self._ball_shade_cache.clear()
            self._ball_shade_cache[key] = sprite
        return sprite

    def _draw_ball_shadow(self, surface: pygame.Surface, ball: Ball, base_radius_px: float) -> None:
        """The ball's ground shadow, cast by the scene light (a point ``scene_light.height_m`` above
        the pitch's middle). A sphere's shadow is an ELLIPSE: semi-minor axis its radius ``r``,
        semi-major ``r / sin(e)`` for light elevation ``e`` (``atan((H - zc) / d)`` for a ball centre
        at height ``zc``, distance ``d`` from the middle), elongated along the light's direction
        (radially away from the centre) and centred where the light through the ball's centre lands:
        ``d * zc / (H - zc)`` beyond the ball. The radius used is the DRAWN (enlarged) ball's,
        ground-sized (not boosted with height), so the shadow keeps that ball's proportions. So a
        ball on the ground casts a circle straight under it at the centre of the pitch (overhead
        light -- hidden) that stretches and slides out with distance from the middle; a raised
        ball's whole shadow also lies further out.

        Where that would be too little to see (near the middle, near the ground) a MINIMUM contact
        shadow slides the ellipse outward until its far tip clears the ball's edge by
        ``contact_radius_frac`` of the ball's radius (at least ``contact_min_px``) -- a fraction of
        the DRAWN radius so it scales with zoom exactly like the ball, still pointing radially away
        from the centre (with a fixed lower-right bias within ~3m of it, where the overhead light
        has no true direction). It fades out as the ball rises (gone by 1.5m). Fainter and softer
        the higher the ball is, and (see `_shadow_ellipse_sprite`) darker/sharper on the end nearest
        the ball than on its far tip -- the same directional fade the players' shadows use -- with
        that fade widening further the higher the ball is."""
        cam = self.camera
        ppm = cam.pixels_per_metre
        light_h = self._light_height_m
        z_under = max(0.0, ball.position.z - ball.radius_m)
        r_px = base_radius_px * (1.0 + 0.04 * z_under)                    # ground radius of the shadow
        z_centre = z_under + base_radius_px / ppm                         # centre height of the DRAWN sphere (m)
        z_eff = min(z_centre, 0.8 * light_h)

        bx, by = cam.world_to_screen_f(ball.position.x, ball.position.y)
        cx, cy = cam.world_to_screen_f(0.0, 0.0)
        vx, vy = bx - cx, by - cy
        dist_px = math.hypot(vx, vy)
        elevation = math.atan2(light_h - z_eff, dist_px / ppm)
        semi_major = r_px / max(math.sin(elevation), 0.2)                # cap the stretch at a 5:1 ellipse
        k = z_eff / (light_h - z_eff)
        offset = dist_px * k                                              # where the sphere's centre casts, from the ball

        # Direction: radially away from the centre, with a lower-right bias that dominates within ~3m
        # of it so the direction is continuous as the ball crosses the middle.
        bias = 3.0 * ppm
        dx, dy = vx + bias * 0.7071, vy + bias * 0.7071
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        angle = int(round(math.degrees(math.atan2(uy, ux)) / 5.0) * 5) % 360

        wanted_tip = r_px + max(self._ball_shadow_contact_frac * r_px, self._ball_shadow_contact_min_px)
        grounded = max(0.0, 1.0 - z_under / 1.5)
        extra = max(0.0, wanted_tip - (offset + semi_major)) * grounded
        centre_x, centre_y = bx + vx * k + ux * extra, by + vy * k + uy * extra
        height_frac = min(z_under / self._ball_shadow_height_fade_ramp_m, 1.0)
        sprite = self._shadow_ellipse_sprite(
            semi_major, r_px, angle, self._ball_shadow_alpha * (1.0 - 0.45 * min(z_under / 6.0, 1.0)),
            0.15 + 0.06 * min(z_under, 6.0), tip_fade=self._ball_shadow_tip_fade, height_frac=height_frac,
            height_extra_soft=self._ball_shadow_height_extra_soft, height_extra_fade=self._ball_shadow_height_extra_fade,
        )
        surface.blit(sprite, (round(centre_x) - sprite.get_width() // 2, round(centre_y) - sprite.get_height() // 2))

    def _ball_shadow_sprite(self, radius: float, alpha: float, softness: float) -> pygame.Surface:
        """A soft-edged black disc (a circular ellipse; see `_shadow_ellipse_sprite`)."""
        return self._shadow_ellipse_sprite(radius, radius, 0, alpha, softness)

    # Near/far softness of the ball's directional fade, as multiples of the base `softness` (see
    # `_shadow_ellipse_sprite`) -- the same 0.2/0.8 split the players' capsules use, so both read as
    # the same kind of shadow. Only applied when `tip_fade` (or the height extras) is non-zero.
    _SHADOW_FADE_SOFT_NEAR_FRAC = 0.2
    _SHADOW_FADE_SOFT_FAR_FRAC = 0.8

    def _shadow_ellipse_sprite(
        self, semi_major_px: float, semi_minor_px: float, angle_deg: int, alpha: float, softness: float,
        *, tip_fade: float = 0.0, height_frac: float = 0.0, height_extra_soft: float = 0.0, height_extra_fade: float = 0.0,
    ) -> pygame.Surface:
        """A soft-edged black ellipse centred in a square sprite, its long axis pointing ``angle_deg``
        (screen angle, y down): solid to ``1 - softness`` of its radius in every direction (the elliptical
        distance) and fading to nothing at ``1 + softness``. ``alpha`` is 0-255 at full strength.

        With ``tip_fade`` > 0 (used for the ball's shadow; 0 -- the default -- is the plain uniform
        ellipse above, e.g. for `_ball_shadow_sprite`'s disc) the ellipse fades along its OWN major
        axis instead, from the NEAR end (``t=0``, the ``angle_deg + 180`` side -- nearer the object
        that cast it) to the FAR end (``t=1``, the ``angle_deg`` side): darker/sharper near, lighter/
        softer far, like a real shadow's sharp contact point diffusing into its penumbra -- the same
        idea already used for the players' capsule shadows (`_player_shadow_sprite`). The softness at
        a point is interpolated between ``softness * _SHADOW_FADE_SOFT_NEAR_FRAC`` and ``softness *
        _SHADOW_FADE_SOFT_FAR_FRAC`` by ``t``; the alpha is scaled by ``1 - tip_fade * t``.
        ``height_frac`` (0-1, how "airborne" the caster is -- 0 for a grounded ball) additionally
        widens the softness band by up to ``height_extra_soft`` and deepens the fade by up to
        ``height_extra_fade``, since an airborne shadow has no sharp contact edge anywhere.

        Parameters are quantised (axes 0.5px, alpha 5, softness 0.05, the fade knobs to 3dp) and the
        sprites cached, bounded at 512 (shared with the players' capsules)."""
        a = max(0.5, round(semi_major_px * 2) / 2.0)
        b = max(0.5, round(semi_minor_px * 2) / 2.0)
        alpha, softness = int(round(alpha / 5.0) * 5), round(softness * 20) / 20.0
        tip_fade = round(min(0.95, max(0.0, tip_fade)), 3)
        height_frac = round(min(1.0, max(0.0, height_frac)), 3)
        height_extra_soft = round(max(0.0, height_extra_soft), 3)
        height_extra_fade = round(max(0.0, height_extra_fade), 3)
        key = (a, b, angle_deg, alpha, softness, tip_fade, height_frac, height_extra_soft, height_extra_fade)
        sprite = self._shadow_cache.get(key)
        if sprite is None:
            soft_far_bound = softness * self._SHADOW_FADE_SOFT_FAR_FRAC + height_frac * height_extra_soft if tip_fade > 0.0 else softness
            pad = int(math.ceil(max(a, b) * (1.0 + soft_far_bound))) + 2
            yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1].astype(float)
            cos_a, sin_a = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
            along, across = xx * cos_a + yy * sin_a, -xx * sin_a + yy * cos_a
            rho = np.hypot(along / a, across / b)                       # 1.0 on the ellipse's edge
            if tip_fade > 0.0:
                axis_t = np.clip((along / a + 1.0) / 2.0, 0.0, 1.0)     # 0 at the near end, 1 at the far tip
                soft = (
                    softness * self._SHADOW_FADE_SOFT_NEAR_FRAC
                    + (softness * self._SHADOW_FADE_SOFT_FAR_FRAC - softness * self._SHADOW_FADE_SOFT_NEAR_FRAC) * axis_t
                    + height_frac * height_extra_soft
                )
                fade = min(0.95, tip_fade + height_frac * height_extra_fade)
                strength = 1.0 - fade * axis_t
            else:
                soft = softness
                strength = 1.0
            t = np.clip((1.0 + soft - rho) / np.maximum(2.0 * soft, 1e-6), 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t) * (alpha / 255.0) * strength
            sprite = pygame.Surface((2 * pad + 1, 2 * pad + 1), pygame.SRCALPHA)
            sprite.fill((0, 0, 0, 0))
            pygame.surfarray.pixels_alpha(sprite)[:] = np.clip(t * 255.0 + 0.5, 0, 255).astype(np.uint8).T
            if len(self._shadow_cache) >= 512:
                self._shadow_cache.clear()
            self._shadow_cache[key] = sprite
        return sprite

    _PLAYER_LIGHT_Z_M = 0.9   # height at which a player's body is lit (roughly the torso)

    def _draw_player_pose_layer(
        self, surface: pygame.Surface, player: Player, pos: tuple[int, int], radius_px: int,
        colour: tuple[int, int, int], is_inactive: bool, layer: str,
    ) -> None:
        """Draws the rotated/scaled top-down sprite in place of the plain circle (see
        player_sprites.py). `colour` is the same team/goalkeeper colour `draw_player` already
        resolved -- reused as the sprite's shirt colour, so a keeper's sprite set is cached under
        its own distinct colour exactly like the old circle was. `layer` selects which of
        `PlayerSpriteSet`'s three parallel sprite families to draw: "combined" (the whole figure,
        one sprite -- what `draw_player`'s default, non-split path uses), "legs" or "upper" (see
        `_render_pose_layers` / `draw_player_legs` for why the figure is ever split in two)."""
        side, level = player_sprites.pick_pose(
            self._player_gait_phase.get(player.player_id, 0.0), player.speed_mps, self._sprite_params
        )
        sprite_set = player_sprites.get_sprite_set(self._sprite_params, colour)
        get_fn, shaded_fn = {
            "combined": (sprite_set.get, sprite_set.shaded),
            "legs": (sprite_set.get_legs, sprite_set.legs_shaded),
            "upper": (sprite_set.get_upper, sprite_set.upper_shaded),
        }[layer]
        base_sprite = get_fn(side, level)

        # Rotate to match heading: the sprite's own local art faces "down"
        # (+y); `hx, hy` is the same screen-space facing vector the heading
        # arrow below uses. rotozoom's angle is counter-clockwise-as-viewed
        # (a plain image rotation, unrelated to the y-down pixel coordinate
        # system), so the rotation needed is (angle of the sprite's local
        # "down" reference, i.e. 90 degrees) minus (angle of the target
        # facing vector), both measured the same way `atan2` measures them
        # here (which -- since screen y grows downward -- reads as
        # clockwise-as-viewed, matching how the two cancel out below).
        hx, hy = math.cos(-player.heading_rad), math.sin(-player.heading_rad)
        rotate_deg = 90.0 - math.degrees(math.atan2(hy, hx))

        if self._player_shading_strength > 0.0:
            # Light the sprite BEFORE rotating it: the scene light's direction (screen frame) is
            # rotated into the sprite's own frame by the inverse of the rotation applied below.
            az, el, _ = self._point_light_at(player.position.x, player.position.y, self._PLAYER_LIGHT_Z_M)
            phi, elr, azr = math.radians(rotate_deg), math.radians(el), math.radians(az)
            lx, ly = math.cos(elr) * math.cos(azr), math.cos(elr) * math.sin(azr)
            local_az = math.degrees(math.atan2(math.sin(phi) * lx + math.cos(phi) * ly,
                                               math.cos(phi) * lx - math.sin(phi) * ly))
            base_sprite = shaded_fn(
                side, level, int(round(local_az / 15.0) * 15) % 360, int(min(90, round(el / 5.0) * 5)),
                self._player_shading_strength,
            )

        target_diameter = max(1.0, radius_px * self._sprite_params.size_scale)
        scale = target_diameter / base_sprite.get_width()
        rotated = pygame.transform.rotozoom(base_sprite, rotate_deg, scale)

        if is_inactive:
            rotated = rotated.copy()
            rotated.set_alpha(self._inactive_alpha)
        rect = rotated.get_rect(center=pos)
        surface.blit(rotated, rect)

    def _draw_player_sprite(
        self, surface: pygame.Surface, player: Player, pos: tuple[int, int], radius_px: int,
        colour: tuple[int, int, int], is_inactive: bool,
    ) -> None:
        """The whole player, one sprite (legs and upper body together) -- `draw_player`'s default
        (`legs=True`) path. See `_draw_player_pose_layer`."""
        self._draw_player_pose_layer(surface, player, pos, radius_px, colour, is_inactive, "combined")

    def _draw_player_legs_sprite(
        self, surface: pygame.Surface, player: Player, pos: tuple[int, int], radius_px: int,
        colour: tuple[int, int, int], is_inactive: bool,
    ) -> None:
        """Just the legs (shafts, feet, shoes, shorts) -- see `draw_player_legs`."""
        self._draw_player_pose_layer(surface, player, pos, radius_px, colour, is_inactive, "legs")

    def _draw_player_upper_sprite(
        self, surface: pygame.Surface, player: Player, pos: tuple[int, int], radius_px: int,
        colour: tuple[int, int, int], is_inactive: bool,
    ) -> None:
        """Just the upper body (arms, shoulders, torso, head) -- `draw_player(..., legs=False)`,
        used once the legs have already been drawn earlier by `draw_player_legs`."""
        self._draw_player_pose_layer(surface, player, pos, radius_px, colour, is_inactive, "upper")

    def _player_draw_colour(self, player: Player) -> tuple[int, int, int]:
        if player.is_goalkeeper:
            return style.GOALKEEPER_COLOUR
        return style.TEAM_LEFT_COLOUR if player.team == Team.LEFT else style.TEAM_RIGHT_COLOUR

    def draw_player_legs(self, surface: pygame.Surface, players: Sequence[Player]) -> None:
        """Every player's LEGS layer only (shafts, feet, shoes, shorts -- see
        `player_sprites._render_pose_layers`), meant to be called in the shared "under everything"
        pass, before the ball (see `draw_pitch_and_ball`): a player standing over/near the ball then
        shows it resting between their feet, with the torso/arms/head (drawn afterwards, by
        `draw_player(..., legs=False)`) appearing in front of it, the way an overhead photo of
        someone dribbling reads -- rather than the whole figure sitting flatly on one side of the
        ball. `draw_player`'s own default (`legs=True`) draws the combined sprite instead, for
        standalone use (tests, or a caller that never calls this).

        A no-op with sprites disabled -- the flat-circle fallback has no separate legs to draw.

        Known trade-off: legs are drawn for ALL players here, before ANY player's upper body, so if
        two players' sprites happen to overlap (standing right next to each other, e.g. a tackle)
        the later one's upper body can cover the earlier one's legs regardless of which of the two
        is meant to be "on top" (e.g. the ball carrier, normally drawn last) -- a rare case, and
        harmless since it only affects the few pixels where two players' sprites directly overlap."""
        if not self._sprites_enabled:
            return
        cam = self.camera
        for player in players:
            pos = cam.world_to_screen(player.position.x, player.position.y)
            radius_px = self._player_radius_px(player)
            colour = self._player_draw_colour(player)
            is_inactive = player.state == PlayerState.INACTIVE_TACKLED
            self._draw_player_legs_sprite(surface, player, pos, radius_px, colour, is_inactive)

    def draw_player_rings(
        self, surface: pygame.Surface, players: Sequence[Player], selected_id: str | None = None,
    ) -> None:
        """Every player's rings, translucent (``rings.alpha``) and in ONE layer meant to be drawn UNDER
        the sprites and the ball (see `draw_pitch_and_ball`): the pulsing low-stamina flash (outermost,
        +11px), the selected player's ring (+7px), and the state ring at +4px -- cyan for
        CONTROLLING_BALL (mid first-touch control delay), red for INACTIVE_TACKLED. There is no ring
        for possession."""
        alpha = self._ring_alpha
        if alpha <= 0:
            return
        cam = self.camera
        for player in players:
            pos = cam.world_to_screen(player.position.x, player.position.y)
            radius_px = self._player_radius_px(player)
            is_inactive = player.state == PlayerState.INACTIVE_TACKLED
            # --- Low-stamina flash: outermost ring, pulsing at configured hz ---
            if player.stamina < self._stamina_flash_threshold and not is_inactive:
                period_ms = 1000.0 / max(self._stamina_flash_hz, 0.1)
                if (pygame.time.get_ticks() % int(period_ms * 2)) < int(period_ms):
                    self._draw_ring(surface, style.STAMINA_FLASH_OUTLINE, pos, radius_px + 11, 2, alpha)
            # --- State ring: CONTROLLING_BALL (cyan) / INACTIVE_TACKLED (red) ---
            if player.state == PlayerState.CONTROLLING_BALL:
                self._draw_ring(surface, style.CONTROL_DELAY_OUTLINE, pos, radius_px + 4, 2, alpha)
            elif is_inactive:
                self._draw_ring(surface, style.INACTIVE_OUTLINE, pos, radius_px + 4, 2, alpha)
            if player.player_id == selected_id:
                self._draw_ring(surface, style.SELECTED_OUTLINE, pos, radius_px + 7, 2, alpha)

    def _player_radius_px(self, player: Player) -> int:
        """The player's on-screen radius. Same sub-linear floor-growth idea as `_ball_base_radius_px`
        (`Camera.zoom_scale ** self._size_floor_zoom_decay`, not a flat `zoom_scale`): a player is
        physically bigger than the ball (0.3m vs 0.11m) so overtakes its OWN floor sooner regardless,
        but without the shared decay exponent the two would converge to true scale at noticeably
        different zooms, briefly looking wrong-sized relative to EACH OTHER during the transition
        (this is what the docstring on `Camera.zoom_scale` warns a naive fix could cause -- a truly
        FIXED, non-zoom-scaled floor would make that worse still, not better, since the player would
        then race away toward true scale almost immediately while the ball stayed floor-locked)."""
        cam = self.camera
        floor_px = self.min_player_radius_px * cam.zoom_scale ** self._size_floor_zoom_decay
        return int(max(floor_px, cam.scale_length(player.radius_m)))

    def draw_player_shadows(self, surface: pygame.Surface, players: Sequence[Player]) -> None:
        """Every player's ground shadow, in one pass (call it BEFORE drawing any sprite -- see
        `draw_pitch_and_ball`, which does -- so a shadow never lands on top of another player).
        The light is the scene's point light above the middle of the pitch: the shadow of a
        figure of height h at distance d from the centre points straight away from it and is
        d * h / (H - h) long (a thin contact shadow, `player_shadow.contact_m`, so a player at the
        centre still has one), drawn as a capsule from the feet to the tip that softens and fades
        toward the tip (`_player_shadow_sprite`). An ellipse was tried for players and rejected --
        the bar read better on a figure -- while the BALL, a sphere, keeps its exact elliptical
        shadow (`_draw_ball_shadow`)."""
        if not self._player_shadow_enabled or self._player_shadow_alpha <= 0:
            return
        cam = self.camera
        ppm = cam.pixels_per_metre
        cx, cy = cam.world_to_screen_f(0.0, 0.0)
        for player in players:
            bx, by = cam.world_to_screen_f(player.position.x, player.position.y)
            vx, vy = bx - cx, by - cy
            dist_m = math.hypot(vx, vy) / ppm
            h = max(player.height_m, 0.1)
            length_px = max(dist_m * h / max(self._light_height_m - h, 1.0), self._player_shadow_contact_m) * ppm
            # Direction: radially away from the centre, with a lower-right bias that dominates within
            # ~1.5 m of it so the direction is continuous as a player crosses the middle.
            dx, dy = vx + 1.5 * ppm * 0.7071, vy + 1.5 * ppm * 0.7071
            angle = int(round(math.degrees(math.atan2(dy, dx)) / 5.0) * 5) % 360
            width = int(round(self._player_radius_px(player) * 0.85))
            sprite = self._player_shadow_sprite(int(round(length_px)), angle, width)
            pos = cam.world_to_screen(player.position.x, player.position.y)
            surface.blit(sprite, (pos[0] - sprite.get_width() // 2, pos[1] - sprite.get_height() // 2))

    # A player's shadow is sharpest at the feet and blurs (the penumbra widens) with distance from
    # them: half-softness of the edge at the feet / at the tip, as a fraction of the shadow's half-width.
    _PLAYER_SHADOW_SOFT_NEAR = 0.2
    _PLAYER_SHADOW_SOFT_FAR = 0.8

    def _player_shadow_sprite(self, length_px: int, angle_deg: int, radius_px: int) -> pygame.Surface:
        """A player's shadow: a capsule of half-width ``radius_px`` (its 50%-alpha edge is a constant-width
        bar with a round end) running from the CENTRE of the returned square sprite -- the player's feet --
        ``length_px`` px in direction ``angle_deg`` (screen angle, y down). Like a real shadow it is sharpest
        and darkest at the feet and gets softer-edged (`_PLAYER_SHADOW_SOFT_NEAR` -> `_FAR`) and lighter
        (``player_shadow.alpha`` at the feet to ``alpha * (1 - tip_fade)`` at the tip) with distance from
        them. Cached per (length, angle, radius, alpha, fade) in `_shadow_cache`, shared with the ball's
        ellipses (bounded at 512)."""
        key = ("capsule", length_px, angle_deg, radius_px, self._player_shadow_alpha, round(self._player_shadow_tip_fade, 3))
        sprite = self._shadow_cache.get(key)
        if sprite is None:
            r = max(1, radius_px)
            pad = int(math.ceil(length_px + r * (1 + self._PLAYER_SHADOW_SOFT_FAR))) + 2
            yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1].astype(float)
            sx, sy = math.cos(math.radians(angle_deg)) * length_px, math.sin(math.radians(angle_deg)) * length_px
            t = np.clip((xx * sx + yy * sy) / max(sx * sx + sy * sy, 1e-9), 0.0, 1.0)     # 0 at the feet, 1 at the tip
            d = np.hypot(xx - t * sx, yy - t * sy)
            soft = self._PLAYER_SHADOW_SOFT_NEAR + (self._PLAYER_SHADOW_SOFT_FAR - self._PLAYER_SHADOW_SOFT_NEAR) * t
            a = np.clip((r * (1 + soft) - d) / (2 * r * soft), 0.0, 1.0)
            a = a * a * (3.0 - 2.0 * a) * (self._player_shadow_alpha / 255.0) * (1.0 - self._player_shadow_tip_fade * t)
            sprite = pygame.Surface((2 * pad + 1, 2 * pad + 1), pygame.SRCALPHA)
            sprite.fill((0, 0, 0, 0))
            pygame.surfarray.pixels_alpha(sprite)[:] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8).T
            if len(self._shadow_cache) >= 512:
                self._shadow_cache.clear()
            self._shadow_cache[key] = sprite
        return sprite

    def draw_player(
        self, surface: pygame.Surface, player: Player, action_icon: str | None = None, *, legs: bool = True,
    ) -> None:
        """Draws one player. Per the design spec:
        - goalkeepers are drawn in a distinct orange colour rather than
          their team colour.
        - the rings (selected / control-delay / tackled / low stamina) are NOT drawn
          here: `draw_player_rings` draws them in a translucent layer under every sprite
          and the ball (there is no ring for possession any more). Callers still draw the
          ball-carrier *last* among players -- see app.py -- so they render on top of
          everyone else.
        - inactive players (`PlayerState.INACTIVE_TACKLED`, including a
          tackler briefly off-balance after a failed tackle - see
          engine/knowledge.md) are drawn translucent rather than solid,
          instead of a flat grey tint, so their team/goalkeeper colour is
          still faintly visible.
        - `action_icon`: if set, a small emoji/text label is drawn above the
          player for the configured linger duration (tracked by app.py).
        - `legs`: True (default) draws the WHOLE sprite, legs and upper body together, as one
          image -- the usual choice for a standalone call (e.g. every test in this file). False
          draws ONLY the upper body (arms/shoulders/torso/head), because the legs were already
          drawn earlier by `draw_player_legs`, in the shared pass under the ball -- this is what
          `App._draw_match` passes, via `draw_pitch_and_ball`, so the ball reads as resting at a
          nearby player's feet rather than flatly on top of or under their whole figure. Ignored
          with sprites disabled (the flat-circle fallback has no legs/upper split).
        """
        cam = self.camera
        pos = cam.world_to_screen(player.position.x, player.position.y)
        radius_px = self._player_radius_px(player)
        colour = self._player_draw_colour(player)

        is_inactive = player.state == PlayerState.INACTIVE_TACKLED

        # --- Speed lines: drawn first so they appear behind the player circle ---
        if player.speed_mps > self._speed_line_threshold:
            # Trail direction: opposite to heading (behind the player).
            trail_dx = -math.cos(-player.heading_rad)
            trail_dy = -math.sin(-player.heading_rad)
            # Perpendicular (90° to heading): lines are spread side-by-side
            # across the player's width, not stacked along the trail.
            perp_dx = -trail_dy
            perp_dy = trail_dx
            start_dist = radius_px + 2
            n = self._speed_line_count
            for i in range(n):
                perp_offset = (i - (n - 1) / 2.0) * self._speed_line_gap_px
                sx = pos[0] + trail_dx * start_dist + perp_dx * perp_offset
                sy = pos[1] + trail_dy * start_dist + perp_dy * perp_offset
                ex = sx + trail_dx * self._speed_line_length_px
                ey = sy + trail_dy * self._speed_line_length_px
                pygame.draw.aaline(surface, style.SPEED_LINE_COLOUR, (sx, sy), (ex, ey))

        if self._sprites_enabled:
            if legs:
                self._draw_player_sprite(surface, player, pos, radius_px, colour, is_inactive)
            else:
                self._draw_player_upper_sprite(surface, player, pos, radius_px, colour, is_inactive)
        elif is_inactive:
            # Draw on a small per-pixel-alpha surface so the player reads as
            # translucent rather than a flat grey substitute colour.
            surface.blit(
                self._aa_disc(colour, radius_px, self._inactive_alpha),
                (pos[0] - radius_px - 1, pos[1] - radius_px - 1),
            )
        else:
            pygame.draw.circle(surface, colour, pos, radius_px)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px, colour)

        # Heading indicator - a broad, thin "V": two lines touching the rim
        # at points spread wide around the front of the player, meeting at
        # a point just ahead in the facing direction. Not filled, no
        # outline -- just the two strokes (this replaced a filled-triangle
        # badge that read as too heavy/blocky). Skipped when sprites are
        # enabled -- the rotated sprite itself already shows facing (head/
        # hair asymmetry, limb pose), so the arrow would just be clutter on
        # top of it.
        if self._heading_alpha > 0 and not self._sprites_enabled:
            hx, hy = math.cos(-player.heading_rad), math.sin(-player.heading_rad)
            perp_x, perp_y = -hy, hx
            apex_r = radius_px + self._heading_length_px
            base_r = radius_px - self._heading_base_inset_px
            apex = (pos[0] + hx * apex_r, pos[1] + hy * apex_r)
            base_cx, base_cy = pos[0] + hx * base_r, pos[1] + hy * base_r
            bw = self._heading_base_half_width_px
            base_l = (base_cx + perp_x * bw, base_cy + perp_y * bw)
            base_r_pt = (base_cx - perp_x * bw, base_cy - perp_y * bw)

            pts = (apex, base_l, base_r_pt)
            pad = 2
            minx = int(min(p[0] for p in pts)) - pad
            miny = int(min(p[1] for p in pts)) - pad
            maxx = int(max(p[0] for p in pts)) + pad
            maxy = int(max(p[1] for p in pts)) + pad
            h_surf = pygame.Surface((maxx - minx, maxy - miny), pygame.SRCALPHA)
            loc = lambda p: (p[0] - minx, p[1] - miny)
            colour = (*style.HEADING_INDICATOR_COLOUR, self._heading_alpha)
            pygame.draw.aaline(h_surf, colour, loc(apex), loc(base_l))
            pygame.draw.aaline(h_surf, colour, loc(apex), loc(base_r_pt))
            surface.blit(h_surf, (minx, miny))

        label = self.hud_font.render(player.player_id, True, style.HUD_TEXT)
        surface.blit(label, (pos[0] - label.get_width() // 2, pos[1] + radius_px + 2))

        # Stat bars: stamina (top) and speed (bottom), always visible.
        bar_w = style.STAT_BAR_WIDTH_PX
        bar_h = style.STAT_BAR_HEIGHT_PX
        bar_x = pos[0] - bar_w // 2
        label_h = label.get_height()
        bar_y_stamina = pos[1] + radius_px + 2 + label_h + 2
        bar_y_speed = bar_y_stamina + bar_h + style.STAT_BAR_GAP_PX

        # Stamina bar
        pygame.draw.rect(surface, style.STAT_BAR_BG, (bar_x, bar_y_stamina, bar_w, bar_h))
        stamina_fill = max(0.0, min(1.0, player.stamina))
        if stamina_fill > 0.6:
            stamina_colour = style.STAMINA_BAR_HIGH
        elif stamina_fill > 0.3:
            stamina_colour = style.STAMINA_BAR_MID
        else:
            stamina_colour = style.STAMINA_BAR_LOW
        pygame.draw.rect(surface, stamina_colour, (bar_x, bar_y_stamina, int(bar_w * stamina_fill), bar_h))

        # Speed bar
        pygame.draw.rect(surface, style.STAT_BAR_BG, (bar_x, bar_y_speed, bar_w, bar_h))
        speed_fill = max(0.0, min(1.0, player.speed_mps / style.SPEED_BAR_MAX_MPS))
        pygame.draw.rect(surface, style.SPEED_BAR_COLOUR, (bar_x, bar_y_speed, int(bar_w * speed_fill), bar_h))

        # --- Action icon: emoji/text label floating above the player circle ---
        if action_icon is not None:
            if action_icon not in self._icon_cache:
                raw = self.icon_font.render(action_icon, True, style.HUD_TEXT)
                if self._icon_font_is_bitmap and raw.get_height() > self._icon_target_px * 2:
                    # Scale the native-size bitmap down to the configured target.
                    scale = self._icon_target_px / raw.get_height()
                    new_w = max(1, int(raw.get_width() * scale))
                    self._icon_cache[action_icon] = pygame.transform.smoothscale(
                        raw, (new_w, self._icon_target_px)
                    )
                else:
                    self._icon_cache[action_icon] = raw
            icon_surf = self._icon_cache[action_icon]
            icon_x = pos[0] - icon_surf.get_width() // 2
            icon_y = pos[1] - radius_px - icon_surf.get_height() - 4
            surface.blit(icon_surf, (icon_x, icon_y))

    def draw_drag_indicator(
        self,
        surface: pygame.Surface,
        start_world: tuple[float, float],
        end_screen: tuple[int, int],
        colour: tuple[int, int, int],
    ) -> None:
        start_screen = self.camera.world_to_screen(*start_world)
        pygame.draw.line(surface, colour, start_screen, end_screen, 2)
        pygame.draw.circle(surface, colour, end_screen, 4)

    def draw_kick_ui(
        self,
        surface: pygame.Surface,
        kick_state: "KickUIState",
        player: Player,
        goal_height_m: float,
        bottom_reserve_px: int = 0,
    ) -> None:
        """Draw the multi-phase kick UI: trajectory (coloured by height/direction)
        and a translucent 1-sigma XY error cone."""
        from footballcoach.ui.kick_trajectory import (
            compute_error_sigma,
            compute_launch_velocity,
            compute_speed_mps,
            build_cone_boundaries,
            height_to_colour,
            simulate_trajectory,
        )
        from footballcoach.ui.input import KickPhase

        cfg = load_graphics_config()["kick_ui"]
        duration_s: float = cfg["trajectory_duration_s"]

        ku = kick_state
        from footballcoach.engine.movement import effective_top_speed, MovementParams
        _mvparams = MovementParams.from_config()
        top_speed = effective_top_speed(
            _mvparams, player.attributes.top_speed, player.stamina,
            has_ball=True, ball_control_attr=player.attributes.ball_control,
        )
        speed_mps = compute_speed_mps(
            player.attributes.kick_power, ku.power_fraction,
            player_velocity=player.velocity,
            player_top_speed_mps=top_speed,
            aim_dir_x=ku.aim_dir_x, aim_dir_y=ku.aim_dir_y,
        )
        if speed_mps < 0.1:
            return

        launch_pos = player.position.with_z(max(player.position.z, 0.11))
        launch_vel = compute_launch_velocity(
            ku.aim_dir_x, ku.aim_dir_y, ku.elevation_angle_rad, speed_mps
        )
        points = simulate_trajectory(launch_pos, launch_vel, ku.spin, duration_s)

        if len(points) < 2:
            return

        # --- Error cone (1-sigma, XY only) ----------------------------------
        sigma_rad = compute_error_sigma(
            player.attributes.kick_precision, ku.power_fraction,
            player_velocity=player.velocity,
            aim_dir_x=ku.aim_dir_x, aim_dir_y=ku.aim_dir_y,
        )
        left_pts, right_pts = build_cone_boundaries(
            player.position,
            ku.aim_dir_x, ku.aim_dir_y,
            ku.elevation_angle_rad,
            speed_mps,
            sigma_rad,
            ku.spin,
            duration_s,
        )

        n_cone = min(len(left_pts), len(right_pts), len(points))
        if n_cone >= 2:
            left_screen = [self.camera.world_to_screen(p.x, p.y) for p in left_pts[:n_cone]]
            right_screen = [self.camera.world_to_screen(p.x, p.y) for p in right_pts[:n_cone]]
            # Polygon: left forward + right reversed.
            poly = left_screen + list(reversed(right_screen))
            if len(poly) >= 3:
                cone_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                pygame.draw.polygon(
                    cone_surf,
                    (*style.TRAJ_CONE_RGB, style.TRAJ_CONE_ALPHA),
                    poly,
                )
                # AA edge on top of the filled polygon.
                pygame.gfxdraw.aapolygon(
                    cone_surf,
                    [(int(x), int(y)) for x, y in poly],
                    (*style.TRAJ_CONE_RGB, min(255, style.TRAJ_CONE_ALPHA + 60)),
                )
                surface.blit(cone_surf, (0, 0))

        # --- Coloured trajectory segments ------------------------------------
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            s0 = self.camera.world_to_screen(p0.x, p0.y)
            s1 = self.camera.world_to_screen(p1.x, p1.y)

            colour = height_to_colour(p0.z, goal_height_m)

            # Two offset aalines approximate a smooth 2px-wide antialiased line.
            dx, dy = s1[0] - s0[0], s1[1] - s0[1]
            slen = math.hypot(dx, dy)
            if slen > 1e-6:
                nx, ny = -dy / slen, dx / slen  # perpendicular unit vector
                pygame.draw.aaline(surface, colour,
                    (s0[0] - nx * 0.5, s0[1] - ny * 0.5),
                    (s1[0] - nx * 0.5, s1[1] - ny * 0.5))
                pygame.draw.aaline(surface, colour,
                    (s0[0] + nx * 0.5, s0[1] + ny * 0.5),
                    (s1[0] + nx * 0.5, s1[1] + ny * 0.5))
            else:
                pygame.draw.aaline(surface, colour, s0, s1)

        # --- Apex ticks: short orthogonal line at each local z-maximum ------
        for i in range(1, len(points) - 1):
            if points[i].z > points[i - 1].z and points[i].z > points[i + 1].z:
                sx, sy = self.camera.world_to_screen(points[i].x, points[i].y)
                ax, ay = self.camera.world_to_screen(points[i - 1].x, points[i - 1].y)
                bx, by = self.camera.world_to_screen(points[i + 1].x, points[i + 1].y)
                tdx, tdy = bx - ax, by - ay
                tlen = math.hypot(tdx, tdy)
                if tlen > 1e-6:
                    ox, oy = -tdy / tlen, tdx / tlen
                    half = 6
                    pygame.draw.aaline(
                        surface, height_to_colour(points[i].z, goal_height_m),
                        (sx - ox * half, sy - oy * half),
                        (sx + ox * half, sy + oy * half),
                    )

        # --- Endpoint dot ---------------------------------------------------
        last = points[-1]
        end_screen = self.camera.world_to_screen(last.x, last.y)
        _end_col = height_to_colour(last.z, goal_height_m)
        pygame.gfxdraw.aacircle(surface, end_screen[0], end_screen[1], 4, _end_col)
        pygame.gfxdraw.filled_circle(surface, end_screen[0], end_screen[1], 4, _end_col)

        # --- Phase info + hint panel (bottom of screen, above hotkey bar) ----
        lines: list[tuple[str, tuple[int, int, int]]] = []
        max_height_m = max((p.z for p in points), default=0.0)
        height_str = f"{max_height_m:.1f}m (goal: {goal_height_m:.1f}m)"

        if ku.phase == KickPhase.AIM_XY:
            pct = int(ku.power_fraction * 100)
            lines.append((f"KICK  ·  Power: {pct}%    Peak: {height_str}", style.HUD_ACCENT))
            lines.append(("Move mouse to aim  —  farther = more power", style.HUD_TEXT))
            lines.append(("Left-click to confirm  ·  Right-click or Esc to cancel", style.HUD_TEXT))

        elif ku.phase == KickPhase.AIM_Z:
            pct = int(ku.power_fraction * 100)
            elev_deg = math.degrees(ku.elevation_angle_rad)
            lines.append((f"KICK  ·  Power: {pct}%    Elevation: {elev_deg:.1f}°    Peak: {height_str}", style.HUD_ACCENT))
            lines.append(("Close to player = loft  ·  Far away = flat", style.HUD_TEXT))
            lines.append(("Left-click to confirm  ·  Right-click to go back  ·  Esc cancel", style.HUD_TEXT))

        elif ku.phase == KickPhase.SPIN:
            pct = int(ku.power_fraction * 100)
            elev_deg = math.degrees(ku.elevation_angle_rad)
            spin_mag = ku.spin.length()
            spin_str = f"{spin_mag:.1f} rad/s" if spin_mag > 0.5 else "none"
            lines.append((f"KICK  ·  Power: {pct}%    Elev: {elev_deg:.1f}°    Spin: {spin_str}    Peak: {height_str}", style.HUD_ACCENT))
            lines.append(("Ahead = topspin  ·  Behind = backspin  ·  Left/right = sidespin", style.HUD_TEXT))
            lines.append(("Left-click to fire  ·  Right-click to go back  ·  Esc cancel", style.HUD_TEXT))

        if lines:
            bar_h = 34
            line_h = self.hud_font.get_height() + 3
            total_h = len(lines) * line_h + 10
            max_w = max(self.hud_font.size(t)[0] for t, _ in lines)
            bx = (surface.get_width() - max_w) // 2 - 12
            by = surface.get_height() - bar_h - total_h - 8 - bottom_reserve_px
            bg = pygame.Surface((max_w + 24, total_h), pygame.SRCALPHA)
            bg.fill((10, 10, 18, 200))
            surface.blit(bg, (bx, by))
            for i, (text, colour) in enumerate(lines):
                surf = self.hud_font.render(text, True, colour)
                surface.blit(surf, (bx + 12, by + 5 + i * line_h))

    def draw_hud_text(self, surface: pygame.Surface, lines: list[str], top_left: tuple[int, int] = (8, 8)) -> None:
        x, y = top_left
        for line in lines:
            rendered = self.hud_font.render(line, True, style.HUD_TEXT)
            surface.blit(rendered, (x, y))
            y += rendered.get_height() + 2

    def draw_hotkey_bar(
        self,
        surface: pygame.Surface,
        hotkeys: list[tuple[str, str, bool, bool]],
    ) -> None:
        """Draws a permanent hotkey reference strip at the bottom of the screen.

        Each entry is ``(key_label, action_label, enabled, active)``.
        - *active*: the key is the current mode (e.g. PASS/SHOOT) - rendered
          in accent colour.
        - *enabled*: the action is currently valid (player selected, has
          ball, etc.) - rendered bright.
        - *disabled*: action is not currently valid - rendered dim but
          readable so the player can see the key exists.
        """
        bar_h = 34
        bar_y = surface.get_height() - bar_h
        pygame.draw.rect(surface, style.HOTKEY_BAR_BG, (0, bar_y, surface.get_width(), bar_h))
        x = 10
        for key_text, label, enabled, active in hotkeys:
            if active:
                colour = style.HOTKEY_ACTIVE
            elif enabled:
                colour = style.HOTKEY_ENABLED
            else:
                colour = style.HOTKEY_DISABLED
            rendered = self.hud_font.render(f"{key_text} {label}", True, colour)
            surface.blit(rendered, (x, bar_y + (bar_h - rendered.get_height()) // 2))
            x += rendered.get_width() + 20

    def draw_game_log(
        self,
        surface: pygame.Surface,
        game_log: "GameLog",
        min_level: "LogLevel",
        max_lines: int = 8,
        mouse_pos: tuple[int, int] | None = None,
    ) -> None:
        """Draws the most recent log entries as a scrolling text box in the
        bottom-right corner of the screen.  Newest entries at the bottom.
        No interactive scrollbar — intentionally simple for a playtesting tool.
        Runs of consecutive identical messages are merged into one row
        (``GameLog.collapsed_entries``): the clock shows the first-last time
        range and the message gets a trailing "(Nx)", so spam can't crowd
        every other event out of the few visible lines.
        Each row is prefixed with its ``entry.time_s`` as elapsed match
        clock, MM:SS.mmm (``_format_match_clock``) -- millisecond precision
        so events logged close together are distinguishable, not all
        showing the same whole second.

        Entries carrying ``LogEntry.detail`` (see ui/gamelog.py -- e.g. a
        tackle's skill-check roll/modifier breakdown) get a dim
        ``[Explain]`` suffix; hovering ANYWHERE on that entry's row (not
        just the suffix text, which is a small target) shows its detail as
        a tooltip, positioned to the left of the log box. ``mouse_pos``
        (typically ``pygame.mouse.get_pos()``) is None in headless/test use,
        which simply skips the hover check -- no behaviour change from
        before this param existed.
        """
        from footballcoach.ui.gamelog import LogLevel
        entries = game_log.collapsed_entries(min_level)[-max_lines:]
        linger_frac = getattr(game_log, "linger_frac", 0.0)  # 0.0 = not lingering
        show_linger = linger_frac > 0.0
        if not entries and not show_linger:
            return

        line_h = self.hud_font.get_height() + 2
        box_w = 480
        # linger gets its own label row plus a separate bar row so the text
        # never overlaps the fill rect underneath it.
        linger_h = line_h * 2 + 2 if show_linger else 0
        box_h = len(entries) * line_h + 6 + linger_h
        bar_h = 34  # hotkey bar height — sit just above it
        box_x = surface.get_width() - box_w - 6
        box_y = surface.get_height() - bar_h - box_h - 4

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((10, 10, 18, 180))
        surface.blit(bg, (box_x, box_y))

        hovered_detail: str | None = None
        hovered_row_y = box_y
        for i, entry in enumerate(entries):
            colour = style.HUD_TEXT if entry.level == LogLevel.INFO else style.HOTKEY_DISABLED
            row_y = box_y + 3 + i * line_h

            # A merged run of identical messages shows its first-last clock
            # range (just the one clock if they all landed on the same tick)
            # and a trailing "(Nx)" -- see GameLog.collapsed_entries.
            clock_text = _format_match_clock(entry.time_s)
            if entry.end_time_s is not None and entry.end_time_s != entry.time_s:
                clock_text += "-" + _format_match_clock(entry.end_time_s)
            clock = self.hud_font.render(clock_text, True, style.HOTKEY_DISABLED)
            surface.blit(clock, (box_x + 4, row_y))
            msg_x = box_x + 4 + clock.get_width() + 8

            # box_w is a fixed width, not sized to content, so trim the
            # message to whatever pixel width is left after the clock (wider
            # on a merged row), the "(Nx)" count and any "[Explain]" suffix.
            # The message itself is what gets cut, so the count and [Explain]
            # are never the parts that fall off the end of the box.
            count_suffix = f" ({entry.count}x)" if entry.count > 1 else ""
            reserved_w = self.hud_font.size(count_suffix)[0] if count_suffix else 0
            if entry.detail is not None:
                reserved_w += self.hud_font.size(" [Explain]")[0]
            max_msg_w = box_w - (msg_x - box_x) - reserved_w - 6
            msg_text = entry.message
            while len(msg_text) > 1 and self.hud_font.size(msg_text)[0] > max_msg_w:
                msg_text = msg_text[:-1]
            line_text = msg_text + count_suffix
            text = self.hud_font.render(line_text, True, colour)
            surface.blit(text, (msg_x, row_y))
            if entry.detail is not None:
                suffix = self.hud_font.render(" [Explain]", True, style.HUD_ACCENT)
                surface.blit(suffix, (msg_x + text.get_width(), row_y))
                row_rect = pygame.Rect(box_x, row_y, box_w, line_h)
                if mouse_pos is not None and row_rect.collidepoint(mouse_pos):
                    hovered_detail = entry.detail
                    hovered_row_y = row_y

        if hovered_detail is not None:
            self._draw_log_tooltip(surface, hovered_detail, box_x, hovered_row_y)

        if show_linger:
            outcome_str = getattr(game_log, "linger_outcome", None) or "resetting"
            # Terminal-only reward approximation (see ScenarioLoop.
            # _compute_terminal_rewards / phase1_terminal_reward_only) --
            # appended to the existing outcome line rather than given its
            # own row, so this costs zero extra vertical space in an
            # already-crowded panel.
            rewards = getattr(game_log, "linger_rewards", None)
            reward_str = ""
            if rewards:
                reward_str = "  " + "  ".join(f"{pid} {val:+.2f}" for pid, val in rewards.items())
            label = self.hud_font.render(f"⏳ {outcome_str}{reward_str}", True, style.HUD_TEXT)
            label_y = box_y + box_h - linger_h
            surface.blit(label, (box_x + 8, label_y))

            bar_y = label_y + line_h
            filled_w = max(2, int((box_w - 8) * linger_frac))
            pygame.draw.rect(surface, (40, 40, 60), (box_x + 4, bar_y, box_w - 8, line_h - 4), border_radius=3)
            pygame.draw.rect(surface, style.HUD_ACCENT, (box_x + 4, bar_y, filled_w, line_h - 4), border_radius=3)

    def _draw_log_tooltip(self, surface: pygame.Surface, detail: str, log_box_x: int, row_y: int) -> None:
        """Tooltip for a hovered game-log entry's ``detail`` -- same panel
        style as draw_player_inspector/draw_pause_notification (translucent
        dark background + accent border), positioned to the LEFT of the
        log box (which sits in the bottom-right corner, so there's room),
        vertically anchored to the hovered row and clamped on-screen."""
        padding = 8
        line_h = self.hud_font.get_height() + 2
        lines = detail.split("\n")
        rendered = [self.hud_font.render(line[:80], True, style.HUD_TEXT) for line in lines]

        box_w = max((r.get_width() for r in rendered), default=0) + padding * 2
        box_h = len(rendered) * line_h + padding * 2
        box_x = max(4, log_box_x - box_w - 8)
        box_y = min(max(4, row_y), surface.get_height() - box_h - 4)

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((10, 10, 18, 235))
        surface.blit(bg, (box_x, box_y))
        pygame.draw.rect(surface, style.HUD_ACCENT, (box_x, box_y, box_w, box_h), 1, border_radius=6)

        for i, text_surf in enumerate(rendered):
            surface.blit(text_surf, (box_x + padding, box_y + padding + i * line_h))

    def draw_pause_notification(self, surface: pygame.Surface, message: str) -> None:
        """Draws a prominent centred banner when the game is auto-paused after
        a human-issued order completes.  Rendered above the hotkey bar."""
        sw, sh = surface.get_size()
        bar_h = 34  # hotkey bar height

        padding_x, padding_y = 28, 14
        text_surf = self.pause_notification_font.render(message, True, style.HUD_ACCENT)
        box_w = text_surf.get_width() + padding_x * 2
        box_h = text_surf.get_height() + padding_y * 2
        box_x = (sw - box_w) // 2
        box_y = sh - bar_h - box_h - 12

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((10, 10, 18, 220))
        surface.blit(bg, (box_x, box_y))
        pygame.draw.rect(surface, style.HUD_ACCENT, (box_x, box_y, box_w, box_h), 2, border_radius=6)
        surface.blit(text_surf, (box_x + padding_x, box_y + padding_y))

    def draw_player_inspector(self, surface: pygame.Surface, player: Player) -> None:
        """Side info panel (top-right, below the help button/speed control
        row) for the currently-selected player (either team -- selecting
        an opponent is inspection-only, see input.py's module docstring):
        which AI (if any) drives them, their active order's type and every
        field's current value, and their attribute ratings as red->green
        gradient bars. Read-only, no game-logic dependency beyond
        inspecting plain attribute/dataclass state already on
        Player/PlayerAI/Order/PlayerAttributes."""
        team_label = "LEFT" if player.team == Team.LEFT else "RIGHT"
        text_lines = [f"{player.player_id}  ({team_label})", _describe_player_ai(player)]
        order = player.current_order
        if order is not None:
            text_lines.append("")
            text_lines.extend(_format_order_lines(order))
        else:
            text_lines.append("(no active order)")
        text_lines.append("")
        text_lines.append("Attributes")

        line_h = self.hud_font.get_height() + 2
        padding = 8
        text_rendered = [
            self.hud_font.render(line[:60], True, style.HUD_ACCENT if i < 2 else style.HUD_TEXT)
            for i, line in enumerate(text_lines)
        ]

        bar_label_w, bar_w, bar_value_w, bar_h = 78, 100, 34, 10
        attrs_row_w = bar_label_w + bar_w + bar_value_w
        attrs_h = len(_ATTRIBUTE_LABELS) * line_h

        box_w = max(max((r.get_width() for r in text_rendered), default=0), attrs_row_w) + padding * 2
        box_h = len(text_lines) * line_h + attrs_h + padding * 2
        box_x = surface.get_width() - box_w - 6
        box_y = 48  # below the help button / speed control row

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((10, 10, 18, 200))
        surface.blit(bg, (box_x, box_y))
        pygame.draw.rect(surface, style.HUD_ACCENT, (box_x, box_y, box_w, box_h), 1, border_radius=6)

        y = box_y + padding
        for text_surf in text_rendered:
            surface.blit(text_surf, (box_x + padding, y))
            y += line_h

        attrs = player.attributes
        for field_name, label in _ATTRIBUTE_LABELS:
            value = getattr(attrs, field_name)
            label_surf = self.hud_font.render(label, True, style.HUD_TEXT)
            surface.blit(label_surf, (box_x + padding, y))

            bar_x = box_x + padding + bar_label_w
            bar_y = y + (line_h - bar_h) // 2
            pygame.draw.rect(surface, (50, 50, 62), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            fill_w = max(1, int(bar_w * max(0.0, min(1.0, value))))
            pygame.draw.rect(surface, _attribute_bar_colour(value), (bar_x, bar_y, fill_w, bar_h), border_radius=3)

            value_surf = self.hud_font.render(f"{value:.2f}", True, style.HUD_TEXT)
            surface.blit(value_surf, (bar_x + bar_w + 6, y))
            y += line_h

    def draw_speed_control(
        self,
        surface: pygame.Surface,
        sim_speed: float,
        right_x: int,
        top_y: int = 8,
    ) -> tuple[pygame.Rect, pygame.Rect]:
        """Draws a compact ``[-]  <speed>x  [+]`` control, right-aligned to
        `right_x` (same [-]/value/[+] look as `draw_scenario_params`'s numeric
        rows). Returns `(minus_rect, plus_rect)` for click handling."""
        btn_w, btn_h = 28, 32
        val_w = 56
        gap = 4
        mouse_pos = pygame.mouse.get_pos()

        plus_rect = pygame.Rect(right_x - btn_w, top_y, btn_w, btn_h)
        val_rect = pygame.Rect(plus_rect.x - gap - val_w, top_y, val_w, btn_h)
        minus_rect = pygame.Rect(val_rect.x - gap - btn_w, top_y, btn_w, btn_h)

        pygame.draw.rect(surface, (50, 50, 65), val_rect, border_radius=6)
        val_surf = self.hud_font.render(f"{sim_speed:g}x", True, style.HUD_ACCENT)
        surface.blit(val_surf, (
            val_rect.centerx - val_surf.get_width() // 2,
            val_rect.centery - val_surf.get_height() // 2,
        ))

        for rect, symbol in ((minus_rect, "-"), (plus_rect, "+")):
            hovered = rect.collidepoint(mouse_pos)
            bg_colour = (70, 70, 90) if hovered else (50, 50, 65)
            pygame.draw.rect(surface, bg_colour, rect, border_radius=6)
            sym_surf = self.hud_font.render(symbol, True, style.HUD_ACCENT)
            surface.blit(sym_surf, (
                rect.centerx - sym_surf.get_width() // 2,
                rect.centery - sym_surf.get_height() // 2,
            ))

        return minus_rect, plus_rect

    def draw_zoom_control(
        self,
        surface: pygame.Surface,
        zoom_level: float,
        right_x: int,
        top_y: int = 8,
    ) -> tuple[pygame.Rect, pygame.Rect]:
        """Draws a compact ``[-]  <zoom>x  [+]`` control, right-aligned to
        `right_x` (same look/layout as `draw_speed_control`; callers place
        this immediately to that control's left so it doesn't collide with
        the player-inspector panel that can appear just below this row).
        The value box is highlighted in the accent colour whenever
        `zoom_level` is actually engaged (> 1.0), so it's visible at a
        glance whether the ball-follow zoom is currently active. Returns
        `(minus_rect, plus_rect)` for click handling."""
        btn_w, btn_h = 28, 32
        val_w = 56
        gap = 4
        mouse_pos = pygame.mouse.get_pos()

        plus_rect = pygame.Rect(right_x - btn_w, top_y, btn_w, btn_h)
        val_rect = pygame.Rect(plus_rect.x - gap - val_w, top_y, val_w, btn_h)
        minus_rect = pygame.Rect(val_rect.x - gap - btn_w, top_y, btn_w, btn_h)

        active = zoom_level > 1.0
        val_bg = style.HUD_ACCENT if active else (50, 50, 65)
        val_text_colour = (10, 10, 18) if active else style.HUD_ACCENT
        pygame.draw.rect(surface, val_bg, val_rect, border_radius=6)
        val_surf = self.hud_font.render(f"{zoom_level:g}x", True, val_text_colour)
        surface.blit(val_surf, (
            val_rect.centerx - val_surf.get_width() // 2,
            val_rect.centery - val_surf.get_height() // 2,
        ))

        for rect, symbol in ((minus_rect, "-"), (plus_rect, "+")):
            hovered = rect.collidepoint(mouse_pos)
            bg_colour = (70, 70, 90) if hovered else (50, 50, 65)
            pygame.draw.rect(surface, bg_colour, rect, border_radius=6)
            sym_surf = self.hud_font.render(symbol, True, style.HUD_ACCENT)
            surface.blit(sym_surf, (
                rect.centerx - sym_surf.get_width() // 2,
                rect.centery - sym_surf.get_height() // 2,
            ))

        return minus_rect, plus_rect

    def draw_scenario_params(
        self,
        surface: pygame.Surface,
        params: list["AnyScenarioParam"],
        values: dict[str, object],
        title: str = "Scenario Parameters",
        open_choice_param: "str | None" = None,
        open_choice_folder: "str | None" = None,
        dropdown_scroll: int = 0,
    ) -> tuple[dict[str, tuple[pygame.Rect, pygame.Rect]], int]:
        """Draws the scenario-parameter adjustment screen. Returns
        ``(button_rects, clamped_dropdown_scroll)`` — ``button_rects`` maps
        param name → (left_rect, right_rect) for click detection.

        - ScenarioParam:              [-]  value  [+]
        - ScenarioChoiceParam:        click value area to open a scrollable
                                       dropdown, or [>] to cycle
        - ScenarioGroupedChoiceParam: click value area to open a two-level
                                       dropdown — a scrollable list of groups
                                       first, then (after picking one) a
                                       scrollable list of values within that
                                       group; [>] cycles the flat option space
        - ScenarioBoolParam:          [checkbox]  label

        When ``open_choice_param`` is set, a dropdown list is rendered below
        that param row. Plain-choice option rects are keyed as
        ``"{name}__option__{value}"``. Grouped-choice group rects are keyed
        as ``"{name}__folder__{group}"``, a back button (shown once a group
        is expanded) as ``"{name}__grpback__"``, and leaf value rects reuse
        ``"{name}__option__{value}"``.
        """
        from footballcoach.ui.scenarios import ScenarioBoolParam, ScenarioChoiceParam, ScenarioGroupedChoiceParam
        surface.fill(style.HUD_BG)
        sw, sh = surface.get_size()

        title_surf = self.title_font.render(title, True, style.HUD_ACCENT)
        surface.blit(title_surf, ((sw - title_surf.get_width()) // 2, 30))

        n_rows = len(params)
        available_h = sh - 160  # reserve top (title+padding) + bottom (buttons)
        row_h = max(32, min(44, available_h // max(n_rows, 1)))
        btn_h = max(22, row_h - 10)
        start_y = 100
        btn_w = 32
        gap = 12

        max_label_w = max(
            (self.hud_font.size(p.label)[0] for p in params),
            default=0,
        )
        block_w = max_label_w + gap + btn_w + gap + 200 + gap + btn_w
        col_label_x = (sw - block_w) // 2
        col_minus_x = col_label_x + max_label_w + gap
        col_val_x   = col_minus_x + btn_w + gap
        col_plus_x  = col_val_x + 200 + gap

        button_rects: dict[str, tuple[pygame.Rect, pygame.Rect]] = {}
        mouse_pos = pygame.mouse.get_pos()

        for i, param in enumerate(params):
            y = start_y + i * row_h
            label_surf = self.hud_font.render(param.label, True, style.HUD_TEXT)
            surface.blit(label_surf, (col_label_x, y + (row_h - label_surf.get_height()) // 2))

            if isinstance(param, ScenarioBoolParam):
                # Checkbox: single clickable box, no left/right buttons
                checked = bool(values.get(param.name, param.default))
                box_size = 22
                box_x = col_minus_x
                box_y = y + (row_h - box_size) // 2
                box_rect = pygame.Rect(box_x, box_y, box_size, box_size)
                hovered = box_rect.collidepoint(mouse_pos)
                bg = (70, 100, 70) if checked else ((60, 60, 80) if hovered else (40, 40, 55))
                pygame.draw.rect(surface, bg, box_rect, border_radius=4)
                pygame.draw.rect(surface, style.HUD_ACCENT, box_rect, 1, border_radius=4)
                if checked:
                    tick = self.hud_font.render("✓", True, style.HUD_ACCENT)
                    surface.blit(tick, (box_x + (box_size - tick.get_width()) // 2,
                                        box_y + (box_size - tick.get_height()) // 2))
                # Both rects point to the same box (toggle on either click)
                button_rects[param.name] = (box_rect, box_rect)

            elif isinstance(param, ScenarioChoiceParam):
                # Dropdown: clickable value area opens a list; [>] cycles.
                current = values.get(param.name, param.default)
                is_open = (open_choice_param == param.name)
                val_w = 200 + btn_w + gap  # value area spans to where [+] was
                val_rect = pygame.Rect(col_minus_x, y + (row_h - btn_h) // 2, val_w, btn_h)
                arrow_rect = pygame.Rect(col_plus_x, y + (row_h - btn_h) // 2, btn_w, btn_h)
                # Value / toggle area
                hovered_val = val_rect.collidepoint(mouse_pos)
                val_bg = (70, 90, 110) if is_open else ((60, 70, 90) if hovered_val else (40, 40, 55))
                pygame.draw.rect(surface, val_bg, val_rect, border_radius=4)
                pygame.draw.rect(surface, style.HUD_ACCENT if is_open else (80, 80, 100), val_rect, 1, border_radius=4)
                choice_str = str(current)
                if len(choice_str) > 30:
                    choice_str = "…" + choice_str[-29:]
                arrow_sym = "▲" if is_open else "▼"
                val_surf = self.hud_font.render(f"{choice_str}  {arrow_sym}", True, style.HUD_ACCENT)
                surface.blit(val_surf, (val_rect.x + 6, val_rect.y + (btn_h - val_surf.get_height()) // 2))
                # [>] cycle button (still available)
                hov_arr = arrow_rect.collidepoint(mouse_pos)
                pygame.draw.rect(surface, (70, 70, 90) if hov_arr else (40, 40, 55), arrow_rect, border_radius=4)
                sym = self.hud_font.render(">", True, style.HUD_ACCENT)
                surface.blit(sym, (arrow_rect.x + (btn_w - sym.get_width()) // 2,
                                   arrow_rect.y + (btn_h - sym.get_height()) // 2))
                # val_rect = "minus" (toggle), arrow_rect = "plus" (cycle)
                button_rects[param.name] = (val_rect, arrow_rect)

            elif isinstance(param, ScenarioGroupedChoiceParam):
                # Same row look as ScenarioChoiceParam; opens a two-level
                # (group -> value) dropdown instead of a flat list.
                current = values.get(param.name, param.default)
                is_open = (open_choice_param == param.name)
                val_w = 200 + btn_w + gap
                val_rect = pygame.Rect(col_minus_x, y + (row_h - btn_h) // 2, val_w, btn_h)
                arrow_rect = pygame.Rect(col_plus_x, y + (row_h - btn_h) // 2, btn_w, btn_h)
                hovered_val = val_rect.collidepoint(mouse_pos)
                val_bg = (70, 90, 110) if is_open else ((60, 70, 90) if hovered_val else (40, 40, 55))
                pygame.draw.rect(surface, val_bg, val_rect, border_radius=4)
                pygame.draw.rect(surface, style.HUD_ACCENT if is_open else (80, 80, 100), val_rect, 1, border_radius=4)
                choice_str = str(current)
                if len(choice_str) > 30:
                    choice_str = "…" + choice_str[-29:]
                arrow_sym = "▲" if is_open else "▼"
                val_surf = self.hud_font.render(f"{choice_str}  {arrow_sym}", True, style.HUD_ACCENT)
                surface.blit(val_surf, (val_rect.x + 6, val_rect.y + (btn_h - val_surf.get_height()) // 2))
                hov_arr = arrow_rect.collidepoint(mouse_pos)
                pygame.draw.rect(surface, (70, 70, 90) if hov_arr else (40, 40, 55), arrow_rect, border_radius=4)
                sym = self.hud_font.render(">", True, style.HUD_ACCENT)
                surface.blit(sym, (arrow_rect.x + (btn_w - sym.get_width()) // 2,
                                   arrow_rect.y + (btn_h - sym.get_height()) // 2))
                button_rects[param.name] = (val_rect, arrow_rect)

            else:
                # Standard numeric slider: [-] value [+]
                val = values.get(param.name, param.default)
                val_surf = self.hud_font.render(f"{val:.3g}", True, style.HUD_ACCENT)
                surface.blit(val_surf, (col_val_x, y + (row_h - val_surf.get_height()) // 2))
                minus_rect = pygame.Rect(col_minus_x, y + (row_h - btn_h) // 2, btn_w, btn_h)
                plus_rect  = pygame.Rect(col_plus_x,  y + (row_h - btn_h) // 2, btn_w, btn_h)
                for rect, symbol in ((minus_rect, "-"), (plus_rect, "+")):
                    hovered = rect.collidepoint(mouse_pos)
                    bg_colour = (70, 70, 90) if hovered else (40, 40, 55)
                    pygame.draw.rect(surface, bg_colour, rect, border_radius=4)
                    sym_surf = self.hud_font.render(symbol, True, style.HUD_ACCENT)
                    surface.blit(sym_surf, (rect.x + (btn_w - sym_surf.get_width()) // 2,
                                            rect.y + (btn_h - sym_surf.get_height()) // 2))
                button_rects[param.name] = (minus_rect, plus_rect)

        # Render open dropdown list (drawn after all rows so it overlays them).
        # Returns the clamped scroll offset actually used, so the caller can
        # persist it (list length can shrink between frames, e.g. switching
        # from a folder list to a shorter value list).
        clamped_scroll = dropdown_scroll
        MAX_VISIBLE_ITEMS = 10
        ITEM_H = 28

        def _draw_dropdown_list(
            items: list[tuple[str, str]],  # (key, display_label)
            selected_key: object,
            anchor_x: int,
            anchor_y: int,
            list_w: int,
            key_prefix: str,
            scroll_name: str,
        ) -> int:
            """Draws a scrollable list of items and registers their click
            rects under ``f"{key_prefix}{key}"``. When the list is truncated,
            also draws a click-to-page up/down chevron column (registered as
            ``f"{scroll_name}__scrollup__"`` / ``"__scrolldown__"``) alongside
            the mouse-wheel/keyboard scroll paths, since wheel events aren't
            reliably delivered on every platform/window-manager combo.
            Returns the clamped scroll offset used for this list."""
            total = len(items)
            visible = min(total, MAX_VISIBLE_ITEMS)
            scroll = max(0, min(dropdown_scroll, max(0, total - visible)))
            scrollable = total > visible
            chevron_w = 22 if scrollable else 0
            item_w = list_w - chevron_w
            list_rect = pygame.Rect(anchor_x - 2, anchor_y - 2, list_w + 4, visible * ITEM_H + 4)
            pygame.draw.rect(surface, (25, 25, 38), list_rect, border_radius=4)
            pygame.draw.rect(surface, style.HUD_ACCENT, list_rect, 1, border_radius=4)
            for vi in range(visible):
                ii = scroll + vi
                key, label = items[ii]
                item_rect = pygame.Rect(anchor_x, anchor_y + vi * ITEM_H, item_w, ITEM_H)
                hov = item_rect.collidepoint(mouse_pos)
                sel = (key == selected_key)
                item_bg = (60, 90, 60) if sel else ((55, 55, 75) if hov else (30, 30, 45))
                pygame.draw.rect(surface, item_bg, item_rect)
                label_str = label if len(label) <= 32 else "…" + label[-31:]
                lsurf = self.hud_font.render(label_str, True, style.HUD_ACCENT if sel else style.HUD_TEXT)
                surface.blit(lsurf, (item_rect.x + 6, item_rect.y + (ITEM_H - lsurf.get_height()) // 2))
                button_rects[f"{key_prefix}{key}"] = (item_rect, item_rect)
            if scrollable:
                track_h = visible * ITEM_H
                chevron_x = anchor_x + item_w
                up_h = track_h // 2
                up_rect = pygame.Rect(chevron_x, anchor_y, chevron_w, up_h)
                down_rect = pygame.Rect(chevron_x, anchor_y + up_h, chevron_w, track_h - up_h)
                for rect, symbol, enabled in (
                    (up_rect, "▲", scroll > 0),
                    (down_rect, "▼", scroll < total - visible),
                ):
                    hov = enabled and rect.collidepoint(mouse_pos)
                    pygame.draw.rect(surface, (70, 70, 90) if hov else (40, 40, 55), rect, border_radius=4)
                    colour = style.HUD_ACCENT if enabled else (75, 75, 85)
                    sym_surf = self.hud_font.render(symbol, True, colour)
                    surface.blit(sym_surf, (rect.x + (chevron_w - sym_surf.get_width()) // 2,
                                            rect.y + (rect.height - sym_surf.get_height()) // 2))
                button_rects[f"{scroll_name}__scrollup__"] = (up_rect, up_rect)
                button_rects[f"{scroll_name}__scrolldown__"] = (down_rect, down_rect)
            return scroll

        if open_choice_param is not None:
            open_param = next((p for p in params if p.name == open_choice_param), None)
            open_idx = params.index(open_param) if open_param is not None else -1
            list_w = 200 + btn_w + gap
            list_x = col_minus_x
            open_y = start_y + open_idx * row_h + row_h  # just below the row

            if isinstance(open_param, ScenarioChoiceParam):
                current = values.get(open_param.name, open_param.default)
                items = [(c, str(c)) for c in open_param.choices]
                clamped_scroll = _draw_dropdown_list(
                    items, current, list_x, open_y, list_w, f"{open_param.name}__option__", open_param.name,
                )

            elif isinstance(open_param, ScenarioGroupedChoiceParam):
                current = values.get(open_param.name, open_param.default)
                if open_choice_folder is None:
                    # Level 1: pick a group. Highlight the group containing
                    # the current value.
                    current_folder = next(
                        (g for g, vals in open_param.groups if current in vals), None,
                    )
                    items = [(g, g) for g, _vals in open_param.groups]
                    clamped_scroll = _draw_dropdown_list(
                        items, current_folder, list_x, open_y, list_w, f"{open_param.name}__folder__", open_param.name,
                    )
                else:
                    # Level 2: pick a value within the expanded group, with a
                    # back button above the list to return to group picking.
                    group_values = next(
                        (vals for g, vals in open_param.groups if g == open_choice_folder), (),
                    )
                    back_rect = pygame.Rect(list_x, open_y, list_w, ITEM_H)
                    hov_back = back_rect.collidepoint(mouse_pos)
                    pygame.draw.rect(surface, (55, 55, 75) if hov_back else (35, 35, 50), back_rect, border_radius=4)
                    back_surf = self.hud_font.render(f"← {open_choice_folder}", True, style.HUD_TEXT)
                    surface.blit(back_surf, (back_rect.x + 6, back_rect.y + (ITEM_H - back_surf.get_height()) // 2))
                    button_rects[f"{open_param.name}__grpback__"] = (back_rect, back_rect)

                    def _leaf_label(v: str) -> str:
                        _folder, _sep, rest = v.partition("/")
                        return rest if _sep else v

                    items = [(v, _leaf_label(v)) for v in group_values]
                    clamped_scroll = _draw_dropdown_list(
                        items, current, list_x, open_y + ITEM_H + 4, list_w, f"{open_param.name}__option__", open_param.name,
                    )

        # Start / Back buttons near the bottom.
        bottom_y = start_y + len(params) * row_h + 30
        start_rect = pygame.Rect(sw // 2 - 110, bottom_y, 100, 38)
        back_rect = pygame.Rect(sw // 2 + 20, bottom_y, 100, 38)
        for rect, label, colour in (
            (start_rect, "Start", style.HUD_ACCENT),
            (back_rect, "Back", style.HUD_TEXT),
        ):
            hovered = rect.collidepoint(pygame.mouse.get_pos())
            bg = (60, 90, 60) if (label == "Start" and hovered) else (60, 60, 80) if hovered else (35, 35, 50)
            pygame.draw.rect(surface, bg, rect, border_radius=6)
            txt = self.hud_font.render(label, True, colour)
            surface.blit(txt, (rect.x + (rect.width - txt.get_width()) // 2,
                                rect.y + (rect.height - txt.get_height()) // 2))
        button_rects["__start__"] = (start_rect, start_rect)
        button_rects["__back__"] = (back_rect, back_rect)
        return button_rects, clamped_scroll
