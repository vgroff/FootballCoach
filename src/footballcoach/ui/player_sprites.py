"""Procedurally-drawn top-down player sprites (head/shoulders/torso/limbs)
used by ``renderer.draw_player`` in place of the old plain circle.

Design history / rationale lives in the conversation that produced this
(iterated entirely by eye against a debug sprite sheet); the short version:

- Everything is drawn with pygame primitives at ``_SUPERSAMPLE``x resolution
  on an SRCALPHA surface, then ``smoothscale``d down to ``_BASE_SIZE`` --
  the same supersample-then-downscale trick ``Renderer._draw_ring`` and the
  ball spin dots use elsewhere in this package, needed here for the same
  reason: small rounded shapes (a ~20px-wide shoulder square, a leg only a
  few px thick) rasterise visibly raggedly at native size.
- The sprite is drawn "facing down" (+y) in its own local space; the caller
  (``Renderer.draw_player``) rotates the finished surface with
  ``pygame.transform.rotozoom`` to match ``player.heading_rad`` and to size
  it to the current on-screen player radius, in one call.
- 9 poses per shirt colour: standing, plus 2 "leading sides" (an internal
  animation-cycle label, not tied to actual on-screen left/right once the
  sprite is rotated) x 4 interpolated stride levels (`STRIDE_LEVELS`).
  ``PlayerSpriteSet`` builds and caches all 9 for one shirt colour;
  ``get_sprite_set`` caches one ``PlayerSpriteSet`` per distinct colour
  actually requested (team colours + the goalkeeper colour, typically 2-3
  total), since building them is a one-off cost, not a per-frame one.
- ``sprite_normals`` / ``shade_factors`` / ``PlayerSpriteSet.shaded`` light the flat art per body
  part (silhouette blurred into a height map -> normals -> Lambert), applied to the base pose
  BEFORE the caller's rotation, from a light direction given in the sprite's own frame.
- ``advance_gait_phase`` / ``pick_pose`` turn a player's current speed into
  an animated pose: stride frequency and how far the stride extends both
  scale with speed (a standing player mid-cycle at speed 0 still shows the
  neutral pose, since extension is scaled to 0 there regardless of phase).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame
import pygame.gfxdraw

RGB = tuple[int, int, int]

# One "full stride cycle" (both legs' forward extremes) is represented as a
# phase in [0, 2*pi): [0, pi) is the "leading_side=+1" half (that side's leg
# swinging forward, ramping 0->1->0 via a sine envelope), [pi, 2*pi) mirrors
# it for leading_side=-1. See `pick_pose`.
STRIDE_LEVELS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

_SUPERSAMPLE = 4
_BASE_SIZE = 80  # final cached sprite is _BASE_SIZE x _BASE_SIZE px
_SIZE = _BASE_SIZE * _SUPERSAMPLE
_PX = _SUPERSAMPLE  # 1 final-image pixel, expressed in _SIZE-space units

_OUTLINE_DARKEN = 0.55  # outline colour = fill colour darkened by this factor

# One animation update never advances the stride by more than this many full cycles. A cycle
# alternates the leading leg every half, so at >= 0.5 cycles per update the pose sequence
# aliases (strides appear to stall or run backwards, like a wheel in film); staying under
# that limit means a very fast sim slows the drawn stride slightly rather than misleading.
_MAX_CYCLES_PER_UPDATE = 0.4


def _darken(color: RGB, factor: float) -> RGB:
    return (max(0, int(color[0] * factor)), max(0, int(color[1] * factor)), max(0, int(color[2] * factor)))


@dataclass(frozen=True)
class PlayerSpriteParams:
    """Config-driven look/animation knobs. Finer body-shape geometry (torso
    proportions, limb reach, shoulder size, etc.) is intentionally NOT here
    -- those were tuned pixel-by-pixel as an interdependent set and live as
    constants in ``_render_pose`` below; retune them there if needed."""
    skin_color: RGB
    hair_color: RGB
    shorts_color: RGB
    shoe_color: RGB
    lace_color: RGB
    stud_color: RGB
    hair_coverage_deg: float
    size_scale: float
    top_speed_for_stride_mps: float
    max_stride_hz: float
    extension_speed_exponent: float

    @staticmethod
    def from_config() -> "PlayerSpriteParams":
        from footballcoach.config import load_graphics_config, require_section
        d = require_section(load_graphics_config(), "player_sprites", "graphics.json")
        return PlayerSpriteParams(
            skin_color=tuple(d["skin_color"]),
            hair_color=tuple(d["hair_color"]),
            shorts_color=tuple(d["shorts_color"]),
            shoe_color=tuple(d["shoe_color"]),
            lace_color=tuple(d["lace_color"]),
            stud_color=tuple(d["stud_color"]),
            hair_coverage_deg=float(d["hair_coverage_deg"]),
            size_scale=float(d["size_scale"]),
            top_speed_for_stride_mps=float(d["top_speed_for_stride_mps"]),
            max_stride_hz=float(d["max_stride_hz"]),
            extension_speed_exponent=float(d["extension_speed_exponent"]),
        )


# ---------------------------------------------------------------------------
# Low-level shape helpers (pygame has no built-in "filled shape with a
# separate-colour outline" for lines/ellipses -- draw the outline colour
# slightly larger underneath, then the fill on top, same trick used for both).
# ---------------------------------------------------------------------------

def _flat_line(surf: pygame.Surface, p0: tuple[float, float], p1: tuple[float, float],
                width: float, fill: RGB, outline: RGB | None) -> None:
    """A straight, flat-ended (butt cap) thick line -- used for the bare-skin
    leg shaft, whose ends are covered by the shorts hem / foot ellipse."""
    if outline is not None:
        pygame.draw.line(surf, outline, p0, p1, int(width) + 2)
    pygame.draw.line(surf, fill, p0, p1, max(1, int(width)))


def _capsule(surf: pygame.Surface, p0: tuple[float, float], p1: tuple[float, float],
             width: float, fill: RGB, outline: RGB | None) -> None:
    """A rounded-end thick line (line + a filled circle at each end)."""
    _flat_line(surf, p0, p1, width, fill, outline)
    r = width / 2
    for p in (p0, p1):
        if outline is not None:
            pygame.draw.circle(surf, outline, p, int(r) + 1)
        pygame.draw.circle(surf, fill, p, max(1, int(r)))


def _ellipse(surf: pygame.Surface, box: tuple[float, float, float, float],
             fill: RGB, outline: RGB | None, outline_width: float = 2.0) -> None:
    """box = (x0, y0, x1, y1), PIL-style, converted to a pygame Rect."""
    x0, y0, x1, y1 = box
    if outline is not None:
        ow = outline_width
        pygame.draw.ellipse(surf, outline, pygame.Rect(x0 - ow, y0 - ow, (x1 - x0) + 2 * ow, (y1 - y0) + 2 * ow))
    pygame.draw.ellipse(surf, fill, pygame.Rect(x0, y0, x1 - x0, y1 - y0))


def _rounded_rect(surf: pygame.Surface, box: tuple[float, float, float, float], radius: float,
                   corners: tuple[bool, bool, bool, bool], fill: RGB, outline: RGB | None,
                   outline_width: float = 2.0) -> None:
    """box = (x0, y0, x1, y1). corners = (top_left, top_right, bottom_left, bottom_right)."""
    x0, y0, x1, y1 = box
    tl, tr, bl, br = corners

    def _draw(bx0: float, by0: float, bw: float, bh: float, rad: float, color: RGB) -> None:
        rad = max(0, int(rad))
        pygame.draw.rect(
            surf, color, pygame.Rect(bx0, by0, bw, bh),
            border_top_left_radius=rad if tl else 0,
            border_top_right_radius=rad if tr else 0,
            border_bottom_left_radius=rad if bl else 0,
            border_bottom_right_radius=rad if br else 0,
        )

    if outline is not None:
        ow = outline_width
        _draw(x0 - ow, y0 - ow, (x1 - x0) + 2 * ow, (y1 - y0) + 2 * ow, radius + ow, outline)
    _draw(x0, y0, x1 - x0, y1 - y0, radius, fill)


def _pie_wedge(surf: pygame.Surface, box: tuple[float, float, float, float],
               start_deg: float, end_deg: float, fill: RGB, outline: RGB | None,
               outline_width: float = 2.0, segments: int = 40) -> None:
    """Fills the "chord" region of an ellipse between two angles (degrees,
    0=3-o'clock, increasing clockwise on screen -- matching PIL's arc/chord/
    pieslice convention, which the geometry constants below were tuned
    against). pygame has no chord primitive, so this samples points along
    the arc into a polygon; ``gfxdraw.filled_polygon`` closes the loop
    (connecting the last point back to the first) automatically, which is
    exactly the chord's straight closing edge."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    pts = []
    for i in range(segments + 1):
        t = start_deg + (end_deg - start_deg) * (i / segments)
        rad = math.radians(t)
        pts.append((cx + rx * math.cos(rad), cy + ry * math.sin(rad)))
    if outline is not None:
        ow = outline_width
        big = [(cx + (rx + ow) * math.cos(math.radians(start_deg + (end_deg - start_deg) * (i / segments))),
                cy + (ry + ow) * math.sin(math.radians(start_deg + (end_deg - start_deg) * (i / segments))))
               for i in range(segments + 1)]
        pygame.gfxdraw.filled_polygon(surf, big, outline)
        pygame.gfxdraw.aapolygon(surf, big, outline)
    pygame.gfxdraw.filled_polygon(surf, pts, fill)
    pygame.gfxdraw.aapolygon(surf, pts, fill)


# ---------------------------------------------------------------------------
# Leg / foot / shoe drawing
# ---------------------------------------------------------------------------

def _draw_shoe_laces(surf: pygame.Surface, params: PlayerSpriteParams, x: float, foot_y: float,
                      sign: float, shoe_rx: float, shoe_ry: float) -> None:
    """2 short, 1px-wide ticks near the ankle (proximal/hip) end of the
    shoe, for the FRONT foot -- spaced apart so they read as two distinct
    laces rather than merging into a block."""
    ankle_dir = -sign
    for frac in (0.18, 0.62):
        ly = foot_y + ankle_dir * shoe_ry * frac
        half_len = shoe_rx * 0.8
        pygame.draw.line(surf, params.lace_color, (x - half_len, ly), (x + half_len, ly), int(_PX))


def _draw_shoe_studs(surf: pygame.Surface, params: PlayerSpriteParams, x: float, foot_y: float,
                      shoe_rx: float, shoe_ry: float) -> None:
    """A handful of small dots (cleats) scattered across the BACK foot's
    shoe -- only drawn on the most-extended stride poses."""
    r = max(1.0, shoe_rx * 0.16)
    for ox, oy in ((0.0, 0.0), (-0.45, -0.35), (0.45, -0.35), (-0.45, 0.35), (0.45, 0.35)):
        px, py = x + ox * shoe_rx, foot_y + oy * shoe_ry
        pygame.draw.circle(surf, params.stud_color, (px, py), r)


def _draw_leg(surf: pygame.Surface, params: PlayerSpriteParams, x: float, hip_y: float, sign: float,
              total_len: float, shaft_width: float, shorts_width: float, shorts_len_max: float,
              foot_rx: float, foot_ry: float, shoe_rx: float, shoe_ry: float,
              is_front: bool, show_studs: bool) -> None:
    """One leg: a flat-ended shaft (hip end covered by shorts, foot end by
    the foot/shoe ellipses) from hip to foot, an oval foot, an oval shoe on
    top of the foot, then shorts cut flat across the shaft near the hip
    (only the hip-side corners rounded, for one clean straight hem) drawn
    LAST so they cover whatever part of the shoe/foot/shaft falls under
    them, rather than the shoe potentially covering the shorts."""
    if total_len <= 1e-6:
        return

    skin, skin_outline = params.skin_color, _darken(params.skin_color, _OUTLINE_DARKEN)
    shoe, shoe_outline = params.shoe_color, _darken(params.shoe_color, _OUTLINE_DARKEN)
    shorts, shorts_outline = params.shorts_color, _darken(params.shorts_color, 0.8)

    foot_y = hip_y + sign * total_len

    _flat_line(surf, (x, hip_y), (x, foot_y), shaft_width, skin, skin_outline)
    _ellipse(surf, (x - foot_rx, foot_y - foot_ry, x + foot_rx, foot_y + foot_ry), skin, skin_outline)
    _ellipse(surf, (x - shoe_rx, foot_y - shoe_ry, x + shoe_rx, foot_y + shoe_ry), shoe, shoe_outline)

    if is_front:
        _draw_shoe_laces(surf, params, x, foot_y, sign, shoe_rx, shoe_ry)
    elif show_studs:
        _draw_shoe_studs(surf, params, x, foot_y, shoe_rx, shoe_ry)

    shorts_len = min(total_len, shorts_len_max)
    shorts_far_y = hip_y + sign * shorts_len
    y0, y1 = sorted([hip_y, shorts_far_y])
    half_w = shorts_width / 2
    hip_end_is_top = sign > 0
    corners = (hip_end_is_top, hip_end_is_top, not hip_end_is_top, not hip_end_is_top)
    _rounded_rect(surf, (x - half_w, y0, x + half_w, y1), half_w, corners, shorts, shorts_outline)


def _draw_shoe_laces_dir(surf: pygame.Surface, params: PlayerSpriteParams, foot_x: float, foot_y: float,
                          direction: tuple[float, float], shoe_rx: float, shoe_ry: float) -> None:
    """Like `_draw_shoe_laces`, generalised to an arbitrary `direction` unit vector instead of
    assuming the leg points straight along +-y: the two lace ticks sit toward the ankle (i.e.
    `-direction` from the foot) and are drawn PERPENDICULAR to `direction` rather than always
    horizontal, so they still read as laces across the top of the foot at any swing angle."""
    dx, dy = direction
    perp_x, perp_y = -dy, dx
    for frac in (0.18, 0.62):
        cx = foot_x - dx * shoe_ry * frac
        cy = foot_y - dy * shoe_ry * frac
        half_len = shoe_rx * 0.8
        pygame.draw.line(
            surf, params.lace_color,
            (cx - perp_x * half_len, cy - perp_y * half_len),
            (cx + perp_x * half_len, cy + perp_y * half_len), int(_PX),
        )


def _draw_leg_swing(surf: pygame.Surface, params: PlayerSpriteParams, geo: dict, x: float, hip_y: float,
                     direction: tuple[float, float], signed_len: float) -> None:
    """One leg reaching toward an arbitrary DIRECTION (a unit vector in the sprite's own local
    frame, i.e. before the caller's heading rotation) by a SIGNED distance from the hip -- used for
    the kick/tackle swing (`render_kick_pose`), where the leg needs to reach toward the kick's
    actual target direction rather than only straight forward/backward like `_draw_leg`.

    `signed_len` negative draws the foot BEHIND the hip (backswing) along the SAME line, the
    opposite way -- not to the side. An earlier prototype swept the *direction* itself over time
    (angle from 0 to 180ish) and it visibly arced the foot out to the side and back before settling,
    which read as the leg "swinging around" rather than kicking; this version fixes `direction` for
    the whole animation and only moves `signed_len` back and forth along that one line, so there is
    no sideways component at all.

    The foot/shoe ellipses stay axis-aligned (not rotated to match `direction`) -- a deliberate
    simplification, matching what was shown and approved; for the swing angles this is actually
    used at (close to the kick's own forward-ish target direction, not extreme sideways/behind
    angles) the difference is barely visible. Shorts are NOT drawn here -- see `_draw_shorts_hem`,
    called separately, since real fabric drapes from the hip regardless of how the leg under it
    swings (unlike `_draw_leg`, which is only ever used for a plain vertical stride)."""
    dx, dy = direction
    foot_x, foot_y = x + dx * signed_len, hip_y + dy * signed_len

    skin, skin_outline = params.skin_color, _darken(params.skin_color, _OUTLINE_DARKEN)
    shoe, shoe_outline = params.shoe_color, _darken(params.shoe_color, _OUTLINE_DARKEN)

    _flat_line(surf, (x, hip_y), (foot_x, foot_y), geo["shaft_width"], skin, skin_outline)
    fr, fy = geo["foot_rx"], geo["foot_ry"]
    _ellipse(surf, (foot_x - fr, foot_y - fy, foot_x + fr, foot_y + fy), skin, skin_outline)
    sr, sy = geo["shoe_rx"], geo["shoe_ry"]
    _ellipse(surf, (foot_x - sr, foot_y - sy, foot_x + sr, foot_y + sy), shoe, shoe_outline)
    if signed_len >= 0.0:
        _draw_shoe_laces_dir(surf, params, foot_x, foot_y, direction, sr, sy)
    else:
        _draw_shoe_studs(surf, params, foot_x, foot_y, sr, sy)


def _draw_shorts_hem(surf: pygame.Surface, params: PlayerSpriteParams, geo: dict, x: float, hip_y: float,
                      leg_len: float) -> None:
    """The plain vertical shorts hem at one hip, hanging straight down -- like `_draw_leg`'s own
    shorts, clamped to `min(leg_len, shorts_len_max)` (NOT an unconditional `shorts_len_max`, which
    an earlier version of this used): a short/near-zero `leg_len` (the planted leg, whose visible
    length is just `base_min_leg_len`) would otherwise get a shorts block AS LONG AS the entire leg
    -- no exposed shaft/skin at all, reading as one flat block rather than a leg wearing shorts. The
    DIRECTION/angle still doesn't follow the swing (see `_draw_leg_swing`'s note -- real fabric
    drapes from the hip regardless of how the leg under it swings); only the LENGTH now tracks it,
    same as the gait poses always did."""
    shorts, shorts_outline = params.shorts_color, _darken(params.shorts_color, 0.8)
    shorts_len = min(abs(leg_len), geo["shorts_len_max"])
    half_w = geo["shorts_width"] / 2
    _rounded_rect(
        surf, (x - half_w, hip_y, x + half_w, hip_y + shorts_len), half_w,
        (True, True, False, False), shorts, shorts_outline,
    )


def _kick_pad_px() -> int:
    """Extra padding (in `_SIZE`-space units, beyond the module's normal canvas) the kick/tackle
    swing's two layer renderers need so the leg can't get clipped, for the CURRENTLY configured
    `KICK_BACK_FRAC`/`KICK_CONTACT_FRAC` -- computed, not hand-tuned, so it stays correct if those
    get retuned again. `_render_pose`'s normal `_SIZE` canvas was sized for a running stride, which
    never reaches anywhere near this far from centre (see those constants' own comment for the
    measured, visually-confirmed reason the swing needs to reach much further); the worst case
    (over every possible `direction`) is bounded by the triangle inequality: the swinging hip's own
    distance from centre, plus the full reach magnitude, is always >= the foot's actual distance
    from centre, regardless of which way `direction` points. The hip itself is offset from centre
    in BOTH x (`hip_x_offset`) and y (dropped below centre so the shorts hem clears the torso, see
    `render_kick_pose_legs`) -- both have to go into the hip's own distance from centre, not just
    the x one (an earlier version of this only used `hip_x_offset`, which under-padded once the y
    offset was added, and would have clipped the leg for some directions)."""
    geo = _pose_geometry()
    hip_y_offset = geo["torso_h"] / 2 - geo["hip_inset"]
    hip_dist_from_centre = math.hypot(geo["hip_x_offset"], hip_y_offset)
    max_frac = max(abs(KICK_BACK_FRAC), KICK_CONTACT_FRAC)
    max_reach = hip_dist_from_centre + max_frac * geo["leg_reach"] + geo["shoe_ry"] * 1.5
    extra = max_reach - _SIZE / 2
    return max(0, int(math.ceil(extra / _SUPERSAMPLE)) * _SUPERSAMPLE)


# The un-padded torso's reference size: `Renderer._draw_player_pose_layer` scales a swing sprite by
# `target_diameter / KICK_POSE_REFERENCE_SIZE` rather than by its actual (padded, so bigger than
# this) width -- otherwise the extra canvas `_kick_pad_px` adds to fit the reaching leg would also
# shrink the on-screen TORSO (since a bigger denominator makes the whole sprite scale down more),
# making a kicking player visibly shrink for the duration of the swing.
KICK_POSE_REFERENCE_SIZE = _BASE_SIZE


def render_kick_pose_legs(
    params: PlayerSpriteParams, swing_side: int, direction: tuple[float, float], signed_len_frac: float,
) -> pygame.Surface:
    """Just the legs half of the kick/tackle swing (planted leg + the swinging one, both legs'
    shorts) -- for `Renderer.draw_player_legs`'s per-player legs-under-the-ball pass while a swing
    is active (the app *always* draws players split, legs then upper -- see `draw_player_legs` /
    `draw_player(..., legs=False)` -- so a swing that only had a "combined" renderer would never
    actually show up in a real running match, only in a standalone `legs=True` call).

    `signed_len_frac` is a fraction of `leg_reach` (see `kick_swing_state_at`, which is what
    callers should drive this from -- pixel conversion happens here, not at the call site, so
    geometry stays encapsulated in this module). The returned surface is `_BASE_SIZE +
    2*_kick_pad_px()//_SUPERSAMPLE` square -- BIGGER than the normal `_BASE_SIZE` gait sprites, to
    fit the leg's reach (see `_kick_pad_px`); callers that scale by `target_diameter /
    width` need `KICK_POSE_REFERENCE_SIZE` instead of the actual width, or the extra canvas shrinks
    the on-screen torso (see there). `render_kick_pose` (composes this with
    `render_kick_pose_upper`) is the single-surface convenience version used by tests/demos. Not
    cached -- see `render_kick_pose`."""
    geo = _pose_geometry()
    pad = _kick_pad_px()
    big_size = _SIZE + 2 * pad
    big = pygame.Surface((big_size, big_size), pygame.SRCALPHA)
    inner = big.subsurface((pad, pad, _SIZE, _SIZE))  # shares pixels with `big`, offset by `pad`
    cx = cy = _SIZE / 2
    # The gait poses offset each leg's hip below centre by (torso_h/2 - hip_inset) -- the "front"
    # leg's own sign in _draw_pose_legs/_render_pose -- specifically so the shorts hem (drawn from
    # that hip, a short FIXED distance further down) clears the torso's own bottom edge and is
    # actually visible. Using a dead-centre hip_y here (`cy`) put the whole shorts hem INSIDE the
    # torso's vertical span, entirely hidden under it once composited -- confirmed by comparing the
    # spans, not just a look. Both legs share this one offset (unlike the gait poses' alternating
    # front/back), since a kicking/tackling stance doesn't have the running stride's depth cue.
    hip_y = cy + (geo["torso_h"] / 2 - geo["hip_inset"])
    signed_len_px = signed_len_frac * geo["leg_reach"]

    planted_x = cx - swing_side * geo["hip_x_offset"]
    _draw_leg_swing(inner, params, geo, planted_x, hip_y, (0.0, 1.0), geo["base_min_leg_len"])
    _draw_shorts_hem(inner, params, geo, planted_x, hip_y, geo["base_min_leg_len"])

    swing_x = cx + swing_side * geo["hip_x_offset"]
    _draw_leg_swing(inner, params, geo, swing_x, hip_y, direction, signed_len_px)
    _draw_shorts_hem(inner, params, geo, swing_x, hip_y, signed_len_px)

    base_size = big_size // _SUPERSAMPLE
    return pygame.transform.smoothscale(big, (base_size, base_size))


def render_kick_pose_upper(params: PlayerSpriteParams, shirt_color: RGB, swing_side: int, arm_extension: float) -> pygame.Surface:
    """Just the upper-body half of the kick/tackle swing (arms/shoulders/torso/head) -- the
    `legs=False` counterpart to `render_kick_pose_legs`, see there (including why this ALSO needs
    the same padded canvas -- not because the upper body needs the extra room, but so both layers
    are the same size/centred the same way, and stay aligned when `_draw_player_pose_layer` rotates
    and blits each one independently at the same screen position)."""
    geo = _pose_geometry()
    pad = _kick_pad_px()
    big_size = _SIZE + 2 * pad
    big = pygame.Surface((big_size, big_size), pygame.SRCALPHA)
    inner = big.subsurface((pad, pad, _SIZE, _SIZE))
    _draw_pose_upper(inner, params, geo, shirt_color, arm_extension, swing_side)
    base_size = big_size // _SUPERSAMPLE
    return pygame.transform.smoothscale(big, (base_size, base_size))


def render_kick_pose(
    params: PlayerSpriteParams, shirt_color: RGB, swing_side: int,
    direction: tuple[float, float], signed_len_frac: float, arm_extension: float,
) -> pygame.Surface:
    """One full frame of the kick/tackle swing -- `render_kick_pose_legs` and
    `render_kick_pose_upper` composited onto one surface (legs first, upper drawn over them, same
    order `_render_pose`/`_render_pose_layers` use), for standalone use (tests, demos) where the
    legs-under-the-ball split doesn't matter. The real per-frame draw path
    (`Renderer._draw_player_pose_layer`) calls the two layer functions directly instead, so a
    player's swinging leg still sits under the ball and their torso still sits over it."""
    legs = render_kick_pose_legs(params, swing_side, direction, signed_len_frac)
    upper = render_kick_pose_upper(params, shirt_color, swing_side, arm_extension)
    out = legs.copy()
    out.blit(upper, (0, 0))
    return out


# ---------------------------------------------------------------------------
# Kick/tackle swing timing curve
# ---------------------------------------------------------------------------

# Backswing depth and max forward reach, as a fraction of `leg_reach`.
#
# The first values tried here (-0.5 / 1.15) looked fine in an isolated leg-only prototype (no
# torso/arms drawn), but once actually composited with the rest of the body at real size, the leg
# was almost entirely swallowed by the torso -- `leg_reach` is calibrated for a normal running
# stride, where the foot only needs to peek out a little, not a full kick's much bigger reach.
# Bumping both up to -2.0 / 2.4 fixed that, but was ALSO tried and tuned before a separate, more
# fundamental bug was caught and fixed: `swinging_side_for_direction` -- which leg swings -- had
# the anatomy backwards (see there), so what was being reach-tuned at the time was a same-side,
# non-crossing "reach out sideways" motion, not a real kick; -2.0/2.4 was a correspondingly
# overcorrected reach for that wrong motion, and read as "way too much" once the leg was actually
# crossing the body properly. Re-tuned again after that fix, by eye against the same kind of
# composited-render sweep: -1.0 / 1.6 are visible and readable without the earlier values' excess.
KICK_BACK_FRAC = -1.0
KICK_CONTACT_FRAC = 1.6

# Backswing / strike / recovery durations (seconds) -- a first-pass timing, not yet tuned against
# real gameplay pacing: slow-ish wind-up, a fast sweep through contact, a slower recovery back to
# rest. `KICK_SWING_DURATION_S` is the total, so callers know when the animation is over.
KICK_BACKSWING_S = 0.15
KICK_STRIKE_S = 0.12
KICK_RECOVER_S = 0.18
KICK_SWING_DURATION_S = KICK_BACKSWING_S + KICK_STRIKE_S + KICK_RECOVER_S


def _ease_out_cubic(p: float) -> float:
    return 1.0 - (1.0 - p) ** 3


def _ease_in_out_cubic(p: float) -> float:
    return 4.0 * p ** 3 if p < 0.5 else 1.0 - (-2.0 * p + 2.0) ** 3 / 2.0


def kick_swing_state_at(t_s: float) -> tuple[float, float]:
    """(signed_len_frac, arm_extension) at `t_s` seconds into a kick/tackle swing (see
    `render_kick_pose`): `signed_len_frac` is a fraction of `leg_reach`, negative during the
    backswing (foot drawn behind the hip), ramping through 0 and up to `KICK_CONTACT_FRAC` during
    the strike, then easing back to 0 during recovery. `t_s` past `KICK_SWING_DURATION_S` returns
    the rest state (0.0, 0.0) -- callers should stop calling this and drop back to the normal gait
    pose once `t_s >= KICK_SWING_DURATION_S` (see `Renderer.update_player_animations`)."""
    if t_s < KICK_BACKSWING_S:
        p = t_s / KICK_BACKSWING_S
        return KICK_BACK_FRAC * _ease_out_cubic(p), 0.5 * _ease_out_cubic(p)
    if t_s < KICK_BACKSWING_S + KICK_STRIKE_S:
        p = (t_s - KICK_BACKSWING_S) / KICK_STRIKE_S
        frac = KICK_BACK_FRAC + (KICK_CONTACT_FRAC - KICK_BACK_FRAC) * _ease_in_out_cubic(p)
        return frac, 0.5 + 0.4 * _ease_in_out_cubic(p)
    if t_s < KICK_SWING_DURATION_S:
        p = (t_s - KICK_BACKSWING_S - KICK_STRIKE_S) / KICK_RECOVER_S
        return KICK_CONTACT_FRAC * (1.0 - _ease_out_cubic(p)), 0.9 * (1.0 - _ease_out_cubic(p))
    return 0.0, 0.0


def local_direction_from_world(wx: float, wy: float, heading_rad: float) -> tuple[float, float]:
    """Converts a world-space XY direction (e.g. `player.last_kick_direction`) to the sprite's own
    local frame (the frame `render_kick_pose`'s `direction` and every pose in this module is drawn
    in, BEFORE the caller's heading rotation) -- i.e. the exact inverse of the rotation
    `Renderer.draw_player` applies (`rotate_deg` from `heading_rad`, see there).

    Derived and verified empirically (not from pygame's rotozoom docs): a marker at local (0, +1)
    ("forward", per this module's docstring) lands, after that rotation, at world direction
    (cos(heading), sin(heading)) at every heading tried (0/45/90/180/-90/-45 degrees); a marker at
    local (1, 0) ("local right") lands at the world direction 90 degrees CCW from that. That 2x2
    change-of-basis matrix is both orthogonal AND symmetric (an involution), so its own inverse is
    itself -- this function applies the exact same matrix to go from world back to local.
    (wx, wy) need not be normalised; the result has the same magnitude."""
    s, c = math.sin(heading_rad), math.cos(heading_rad)
    return -wx * s + wy * c, wx * c + wy * s


# Beyond this angle (degrees) from straight-forward (local (0,1)), a CROSSING (far-side) leg stops
# looking plausible -- reaching that far across the body isn't how real kicks work -- so
# `swinging_side_for_direction` switches to the NEAR-side leg instead past this point (still able to
# reach a wide target, just without crossing).
MAX_CROSS_ANGLE_DEG = 40.0

# Absolute cap on how far off-forward the animated leg is EVER allowed to aim, near leg or far --
# past this a real player would turn to face the target rather than reach for it at all. This is
# the direct descendant of an earlier cap on how far a (since-rejected) angle-SWEEPING design's
# follow-through was allowed to cross past contact ("cap it before 60, maybe 30") -- that concept
# didn't carry over automatically when the design changed to a single FIXED direction (there's no
# sweep left to cap), so this reintroduces an equivalent limit for the new design. First tried at
# 90 (unconstrained sideways reach); the user brought it down to 60. Clamping only affects the
# ANIMATION's direction, never the actual kick physics -- the ball still goes exactly where aimed,
# only the leg's drawn reach is capped.
MAX_SWING_ANGLE_DEG = 60.0


def clamp_swing_direction(direction: tuple[float, float], max_angle_deg: float = MAX_SWING_ANGLE_DEG) -> tuple[float, float]:
    """Clamps `direction` (a local-frame unit vector, e.g. from `local_direction_from_world`) to
    within `max_angle_deg` of straight-forward (local (0,1)) -- see `MAX_SWING_ANGLE_DEG`.
    `direction` need not be normalised on the way in; the result always is (or exactly (0.0, 1.0)
    for a degenerate all-zero input, rather than dividing by zero)."""
    dx, dy = direction
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 0.0, 1.0
    dx, dy = dx / n, dy / n
    angle_deg = math.degrees(math.atan2(dx, dy))  # 0 = straight forward, +/- = to either side
    clamped_deg = max(-max_angle_deg, min(max_angle_deg, angle_deg))
    rad = math.radians(clamped_deg)
    return math.sin(rad), math.cos(rad)


def swinging_side_for_direction(
    direction: tuple[float, float], fallback_side: int, deadzone: float = 0.15,
    max_cross_angle_deg: float = MAX_CROSS_ANGLE_DEG,
) -> int:
    """Which side (the `swing_side` convention `render_kick_pose_legs` uses: the swinging leg's hip
    sits at local `+swing_side * hip_x_offset`) should be the leg that swings to reach `direction`.

    Up to `max_cross_angle_deg` off straight-forward, it's the leg on the OPPOSITE lateral side from
    the target, crossing the body, with the OTHER leg (on the target's own side) planted as the
    contact point -- real kicks plant the near-side foot next to the ball as the pivot/balance point
    and swing the FAR-side leg across to strike it; reaching sideways with the near-side foot has no
    power and barely happens. (An earlier version of this always picked the same-side leg, reasoned
    as "avoids an awkward cross-body reach" -- backwards, caught by the user: "nobody kicks outwards
    like that... you cross your right foot across your body".) PAST `max_cross_angle_deg`, though,
    even a real crossing kick stops being plausible -- so this switches back to the NEAR-side leg
    for a wide-angle target, a simpler reach/poke rather than a full crossing strike (per the user:
    "if the player is kicking past that point, use the nearer leg").

    Falls back to `fallback_side` (typically the gait's current leading side) only when
    `direction`'s lateral (x) component is within `deadzone` of straight ahead/behind, where either
    foot is equally natural."""
    dx, dy = direction
    if abs(dx) < deadzone:
        return fallback_side
    angle_deg = abs(math.degrees(math.atan2(dx, dy)))
    crossing = angle_deg <= max_cross_angle_deg
    if crossing:
        return -1 if dx > 0 else 1
    return 1 if dx > 0 else -1


def _pose_geometry() -> dict:
    """Body-proportion constants shared by `_render_pose` and `_render_pose_layers` (pure numbers,
    no drawing -- extracting them changes nothing about either function's output). See the module
    docstring before retuning any of these in isolation; they were tuned as an interdependent set."""
    torso_w, torso_h = _SIZE * 0.62 * 0.85, _SIZE * 0.44 * 0.70 * 0.90
    shorts_width = _SIZE * 0.15
    shoe_ry = shorts_width * 0.68 * 0.85
    shorts_len_max = _SIZE * 0.09 + 1 * _PX
    # 0.95x size, corner radius 0.4x of that size: chosen from a 9-way grid (size in {1.0,0.9,0.8} x
    # corner in {0.35,0.40,0.45}) shown to the user -- less square/"strong"-looking, a bit smaller,
    # without going as round/small as the more extreme variants shown alongside it.
    shoulder_size = _SIZE * 0.22 * 0.90 * 0.95
    shoulder_x_offset = torso_w / 2 + shoulder_size / 2 - (shoulder_size * 0.30 + 1 * _PX)
    return dict(
        torso_w=torso_w, torso_h=torso_h, corner_r=_SIZE * 0.14,
        hip_inset=_SIZE * 0.03, hip_x_offset=_SIZE * 0.13 - 1.5 * _PX,
        shorts_width=shorts_width, shaft_width=shorts_width - 2 * _PX,
        leg_reach=_SIZE * 0.30 * 0.75 * 0.80 * 1.10 * 1.10 + 2 * _PX,
        foot_rx=shorts_width * 0.42 * 0.85, foot_ry=shorts_width * 0.58 * 0.85,
        shoe_rx=shorts_width * 0.50 * 0.85, shoe_ry=shoe_ry,
        shorts_len_max=shorts_len_max,
        base_min_leg_len=max(shorts_len_max - shoe_ry + 1 * _PX, shoe_ry * 0.3),
        shoulder_size=shoulder_size, shoulder_corner_r=shoulder_size * 0.4,
        shoulder_x_offset=shoulder_x_offset, shoulder_y=_SIZE / 2 - torso_h * 0.12 + 2 * _PX,
        arm_width=_SIZE * 0.115, arm_reach=_SIZE * 0.30 * 1.10 * 0.90 * 0.90,
        head_rx=_SIZE * 0.19 * 0.94, head_ry=_SIZE * 0.19 * 1.04, head_y=_SIZE / 2 - _SIZE * 0.03,
    )


def _draw_pose_legs(
    surf: pygame.Surface, params: PlayerSpriteParams, geo: dict, leg_extension: float, leading_side: int,
) -> None:
    """Both legs (shafts, feet, shoes, shorts) -- the exact draw calls `_render_pose` uses,
    factored out so `_render_pose` and `_render_pose_layers` share one implementation (each
    still draws onto its OWN surface, so their own supersample/downscale stays independent)."""
    cx = cy = _SIZE / 2
    is_standing = leg_extension <= 1e-6
    for side, base_sign in ((leading_side, 1.0), (-leading_side, -1.0)):
        sign = 1.0 if is_standing else base_sign
        is_front = sign > 0
        show_studs = (not is_front) and (leg_extension >= 0.75 - 1e-6)
        min_leg_len = geo["base_min_leg_len"] + (1 * _PX if is_standing else -2 * _PX)
        hip_y = cy + sign * (geo["torso_h"] / 2 - geo["hip_inset"])
        total_len = min_leg_len + leg_extension * geo["leg_reach"]
        x = cx + side * geo["hip_x_offset"]
        is_second_least_extended = abs(leg_extension - 0.5) < 1e-6
        shorts_len_max_this = geo["shorts_len_max"] - (1 * _PX if is_second_least_extended else 0)
        _draw_leg(surf, params, x, hip_y, sign, total_len, geo["shaft_width"], geo["shorts_width"], shorts_len_max_this,
                  geo["foot_rx"], geo["foot_ry"], geo["shoe_rx"], geo["shoe_ry"], is_front, show_studs)


def _draw_pose_upper(
    surf: pygame.Surface, params: PlayerSpriteParams, geo: dict, shirt_color: RGB,
    arm_extension: float, leading_side: int,
) -> None:
    """Arms, shoulders, torso and head -- the exact draw calls `_render_pose` uses (see
    `_draw_pose_legs`), in the original order: arms -> shoulders (under the torso) -> torso
    -> head, so the torso overlaps the shoulders' inner edge and the head sits topmost."""
    cx = cy = _SIZE / 2
    shirt_outline = _darken(shirt_color, _OUTLINE_DARKEN)
    skin, skin_outline = params.skin_color, _darken(params.skin_color, _OUTLINE_DARKEN)

    for side, sign in ((leading_side, -1.0), (-leading_side, 1.0)):
        total_len = arm_extension * geo["arm_reach"]
        x = cx + side * geo["shoulder_x_offset"]
        if total_len > 1e-6:
            y1 = geo["shoulder_y"] + sign * total_len
            _capsule(surf, (x, geo["shoulder_y"]), (x, y1), geo["arm_width"], skin, skin_outline)
            hand_r = geo["arm_width"] * 0.42
            _ellipse(surf, (x - hand_r, y1 - hand_r, x + hand_r, y1 + hand_r), skin, skin_outline)

    for side in (leading_side, -leading_side):
        x = cx + side * geo["shoulder_x_offset"]
        ss = geo["shoulder_size"]
        _rounded_rect(
            surf, (x - ss / 2, geo["shoulder_y"] - ss / 2, x + ss / 2, geo["shoulder_y"] + ss / 2),
            geo["shoulder_corner_r"], (True, True, True, True), shirt_color, shirt_outline,
            outline_width=int(_SIZE * 0.02) + 1,
        )

    tw, th = geo["torso_w"], geo["torso_h"]
    _rounded_rect(
        surf, (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2),
        geo["corner_r"], (True, True, True, True), shirt_color, shirt_outline,
        outline_width=int(_SIZE * 0.02) + 1,
    )

    head_rx, head_ry, head_y = geo["head_rx"], geo["head_ry"], geo["head_y"]
    head_box = (cx - head_rx, head_y - head_ry, cx + head_rx, head_y + head_ry)
    _ellipse(surf, head_box, skin, skin_outline, outline_width=int(_SIZE * 0.015) + 1)

    # Hair: a patch centred on the back of the head (-y, opposite the +y
    # "down" facing direction). See ``_pie_wedge``'s docstring for the angle
    # convention; `hair_coverage_deg` is the total wedge width centred on
    # 270 degrees (straight up).
    half_cov = params.hair_coverage_deg / 2.0
    hair, hair_outline = params.hair_color, _darken(params.hair_color, 0.63)
    _pie_wedge(surf, head_box, 270.0 - half_cov, 270.0 + half_cov, hair, hair_outline,
               outline_width=int(_SIZE * 0.015) + 1)


def _render_pose(params: PlayerSpriteParams, shirt_color: RGB, leg_extension: float,
                  arm_extension: float, leading_side: int) -> pygame.Surface:
    """Renders one top-down player sprite pose at ``_BASE_SIZE``.

    leg_extension / arm_extension: 0.0 = neutral (standing), 1.0 = full stride.
    leading_side: +1 / -1, an internal label for which half of the gait
    cycle this is (see module docstring) -- not tied to on-screen left/right.
    Facing direction is "down" (+y) in this local space; the caller rotates
    the finished surface to the player's actual heading.
    """
    big = pygame.Surface((_SIZE, _SIZE), pygame.SRCALPHA)
    cx = cy = _SIZE / 2
    shirt_outline = _darken(shirt_color, _OUTLINE_DARKEN)

    # ---- Body proportions (tuned as an interdependent set -- see module
    # docstring before changing any of these in isolation). -------------------
    torso_w, torso_h = _SIZE * 0.62 * 0.85, _SIZE * 0.44 * 0.70 * 0.90
    corner_r = _SIZE * 0.14

    hip_inset = _SIZE * 0.03
    hip_x_offset = _SIZE * 0.13 - 1.5 * _PX

    shorts_width = _SIZE * 0.15
    shaft_width = shorts_width - 2 * _PX
    leg_reach = _SIZE * 0.30 * 0.75 * 0.80 * 1.10 * 1.10 + 2 * _PX
    foot_rx, foot_ry = shorts_width * 0.42 * 0.85, shorts_width * 0.58 * 0.85
    shoe_rx, shoe_ry = shorts_width * 0.50 * 0.85, shorts_width * 0.68 * 0.85
    shorts_len_max = _SIZE * 0.09 + 1 * _PX
    peek_past_shorts = 1 * _PX
    base_min_leg_len = max(shorts_len_max - shoe_ry + peek_past_shorts, shoe_ry * 0.3)

    # 0.95x size, corner radius 0.4x of that size -- see the matching comment in `_pose_geometry`;
    # kept in sync there by hand since this function stays independent (see the module note above).
    shoulder_size = _SIZE * 0.22 * 0.90 * 0.95
    shoulder_corner_r = shoulder_size * 0.4
    shoulder_overlap = shoulder_size * 0.30 + 1 * _PX
    shoulder_x_offset = torso_w / 2 + shoulder_size / 2 - shoulder_overlap
    shoulder_y = cy - torso_h * 0.12 + 2 * _PX

    arm_width = _SIZE * 0.115
    arm_reach = _SIZE * 0.30 * 1.10 * 0.90 * 0.90

    skin, skin_outline = params.skin_color, _darken(params.skin_color, _OUTLINE_DARKEN)

    # ---- Legs: each has its own hip point, mirrored to sit just inside
    # whichever torso edge it swings toward -- except standing, which has
    # no "forward/back" so both legs hang straight down together. -----------
    is_standing = leg_extension <= 1e-6
    for side, base_sign in ((leading_side, 1.0), (-leading_side, -1.0)):
        sign = 1.0 if is_standing else base_sign
        is_front = sign > 0
        show_studs = (not is_front) and (leg_extension >= 0.75 - 1e-6)
        min_leg_len = base_min_leg_len + (1 * _PX if is_standing else -2 * _PX)
        hip_y = cy + sign * (torso_h / 2 - hip_inset)
        total_len = min_leg_len + leg_extension * leg_reach
        x = cx + side * hip_x_offset
        is_second_least_extended = abs(leg_extension - 0.5) < 1e-6
        shorts_len_max_this = shorts_len_max - (1 * _PX if is_second_least_extended else 0)
        _draw_leg(big, params, x, hip_y, sign, total_len, shaft_width, shorts_width, shorts_len_max_this,
                  foot_rx, foot_ry, shoe_rx, shoe_ry, is_front, show_studs)

    # ---- Arms: drawn BEFORE the shoulder square so the square covers the
    # base of the limb -- only the part beyond the square is visible. -------
    for side, sign in ((leading_side, -1.0), (-leading_side, 1.0)):
        total_len = arm_extension * arm_reach
        x = cx + side * shoulder_x_offset
        if total_len > 1e-6:
            y1 = shoulder_y + sign * total_len
            _capsule(big, (x, shoulder_y), (x, y1), arm_width, skin, skin_outline)
            hand_r = arm_width * 0.42
            _ellipse(big, (x - hand_r, y1 - hand_r, x + hand_r, y1 + hand_r), skin, skin_outline)

    # ---- Shoulder squares, drawn UNDER the torso: legs -> arms -> shoulders
    # -> torso -> head, so the torso overlaps their inner edge and reads as
    # sitting above/in front of the shoulders. --------------------------------
    for side in (leading_side, -leading_side):
        x = cx + side * shoulder_x_offset
        _rounded_rect(
            big, (x - shoulder_size / 2, shoulder_y - shoulder_size / 2,
                  x + shoulder_size / 2, shoulder_y + shoulder_size / 2),
            shoulder_corner_r, (True, True, True, True), shirt_color, shirt_outline,
            outline_width=int(_SIZE * 0.02) + 1,
        )

    # ---- Torso (shirt) -- drawn last so it overlaps the shoulder squares. ---
    _rounded_rect(
        big, (cx - torso_w / 2, cy - torso_h / 2, cx + torso_w / 2, cy + torso_h / 2),
        corner_r, (True, True, True, True), shirt_color, shirt_outline,
        outline_width=int(_SIZE * 0.02) + 1,
    )

    # ---- Head: a very slight egg/skull-shaped oval, topmost. ----------------
    head_rx = _SIZE * 0.19 * 0.94
    head_ry = _SIZE * 0.19 * 1.04
    head_y = cy - _SIZE * 0.03
    head_box = (cx - head_rx, head_y - head_ry, cx + head_rx, head_y + head_ry)
    _ellipse(big, head_box, skin, skin_outline, outline_width=int(_SIZE * 0.015) + 1)

    # Hair: a patch centred on the back of the head (-y, opposite the +y
    # "down" facing direction). See ``_pie_wedge``'s docstring for the angle
    # convention; `hair_coverage_deg` is the total wedge width centred on
    # 270 degrees (straight up).
    half_cov = params.hair_coverage_deg / 2.0
    hair, hair_outline = params.hair_color, _darken(params.hair_color, 0.63)
    _pie_wedge(big, head_box, 270.0 - half_cov, 270.0 + half_cov, hair, hair_outline,
               outline_width=int(_SIZE * 0.015) + 1)

    return pygame.transform.smoothscale(big, (_BASE_SIZE, _BASE_SIZE))


def _render_pose_layers(
    params: PlayerSpriteParams, shirt_color: RGB, leg_extension: float, arm_extension: float, leading_side: int,
) -> tuple[pygame.Surface, pygame.Surface]:
    """Renders one pose as TWO separate ``_BASE_SIZE`` layers instead of one combined image:
    (legs -- both shafts, feet, shoes and shorts --, upper body -- arms, shoulders, torso, head).
    Lets the caller sandwich the ball between them (legs under, upper over) so a player standing
    over the ball reads as the ball resting at their feet, rather than the whole figure sitting flatly
    under or over it -- see ``Renderer.draw_player_legs`` / ``draw_player(..., legs=False)``.

    Independent of ``_render_pose`` (which keeps drawing everything onto ONE surface before its
    single downscale) on purpose: compositing two SEPARATELY downscaled layers is not pixel-identical
    to one combined downscale (their shared seam -- where the shorts tuck under the shirt hem -- would
    anti-alias slightly differently), and ``_render_pose``'s existing output is depended on by tests
    and by every OTHER caller (`draw_player`'s default, sprites-enabled non-split path); duplicating
    the handful of draw calls here (via the shared `_draw_pose_legs` / `_draw_pose_upper` /
    `_pose_geometry` helpers) keeps that path completely unchanged."""
    geo = _pose_geometry()
    legs_big = pygame.Surface((_SIZE, _SIZE), pygame.SRCALPHA)
    upper_big = pygame.Surface((_SIZE, _SIZE), pygame.SRCALPHA)
    _draw_pose_legs(legs_big, params, geo, leg_extension, leading_side)
    _draw_pose_upper(upper_big, params, geo, shirt_color, arm_extension, leading_side)
    return (
        pygame.transform.smoothscale(legs_big, (_BASE_SIZE, _BASE_SIZE)),
        pygame.transform.smoothscale(upper_big, (_BASE_SIZE, _BASE_SIZE)),
    )


class PlayerSpriteSet:
    """All 9 pose surfaces for one shirt colour, built once and cached (plus, on demand, lit
    variants of them -- see `shaded`), PLUS the same 9 poses as separate legs/upper-body layers
    (`get_legs` / `get_upper`; see `_render_pose_layers`) for the ball-between-the-feet effect."""

    _SHADED_CACHE_LIMIT = 1024

    def __init__(self, params: PlayerSpriteParams, shirt_color: RGB) -> None:
        self.standing = _render_pose(params, shirt_color, 0.0, 0.0, 1)
        self.running: dict[tuple[int, float], pygame.Surface] = {}
        for side in (1, -1):
            for level in STRIDE_LEVELS:
                self.running[(side, level)] = _render_pose(params, shirt_color, level, level, side)
        self._normals: dict[tuple[int, float | None], np.ndarray] = {}
        self._shaded: dict[tuple, pygame.Surface] = {}

        self.legs_standing, self.upper_standing = _render_pose_layers(params, shirt_color, 0.0, 0.0, 1)
        self.legs_running: dict[tuple[int, float], pygame.Surface] = {}
        self.upper_running: dict[tuple[int, float], pygame.Surface] = {}
        for side in (1, -1):
            for level in STRIDE_LEVELS:
                legs, upper = _render_pose_layers(params, shirt_color, level, level, side)
                self.legs_running[(side, level)] = legs
                self.upper_running[(side, level)] = upper
        self._legs_normals: dict[tuple[int, float | None], np.ndarray] = {}
        self._upper_normals: dict[tuple[int, float | None], np.ndarray] = {}
        self._legs_shaded: dict[tuple, pygame.Surface] = {}
        self._upper_shaded: dict[tuple, pygame.Surface] = {}

    def get(self, side: int, level: float | None) -> pygame.Surface:
        if level is None:
            return self.standing
        return self.running[(side, level)]

    def get_legs(self, side: int, level: float | None) -> pygame.Surface:
        if level is None:
            return self.legs_standing
        return self.legs_running[(side, level)]

    def get_upper(self, side: int, level: float | None) -> pygame.Surface:
        if level is None:
            return self.upper_standing
        return self.upper_running[(side, level)]

    def _normals_for(self, cache: dict, get_fn, side: int, level: float | None) -> np.ndarray:
        """Shared by `normals` / `legs_normals` / `upper_normals`: per-pixel unit surface normals
        of a pose (see `sprite_normals`), built once per (pose, layer) and cached."""
        key = (1 if level is None else side, level)
        cached = cache.get(key)
        if cached is None:
            cached = sprite_normals(_alpha_of(get_fn(side, level)))
            cache[key] = cached
        return cached

    def normals(self, side: int, level: float | None) -> np.ndarray:
        return self._normals_for(self._normals, self.get, side, level)

    def legs_normals(self, side: int, level: float | None) -> np.ndarray:
        return self._normals_for(self._legs_normals, self.get_legs, side, level)

    def upper_normals(self, side: int, level: float | None) -> np.ndarray:
        return self._normals_for(self._upper_normals, self.get_upper, side, level)

    def _shaded_for(
        self, cache: dict, get_fn, normals_fn, side: int, level: float | None,
        azimuth_deg: int, elevation_deg: int, strength: float,
    ) -> pygame.Surface:
        """Shared by `shaded` / `legs_shaded` / `upper_shaded`: the pose lit from a direction given
        in the SPRITE'S OWN frame (azimuth measured like screen angles: 0 = toward the sprite's
        right, 90 = toward its "down"/facing side; elevation above the ground plane): each pixel's
        colour times `shade_factors`, alpha untouched. Cached per (pose, direction, strength) -- the
        caller quantises the angles -- and bounded."""
        key = (1 if level is None else side, level, azimuth_deg, elevation_deg, round(strength, 3))
        cached = cache.get(key)
        if cached is None:
            base = get_fn(side, level)
            alpha = _alpha_of(base)
            factors = shade_factors(normals_fn(side, level), alpha,
                                    light_from_angles(azimuth_deg, elevation_deg), strength)
            rgb = pygame.surfarray.array3d(base).transpose(1, 0, 2).astype(np.float32) * factors[..., None]
            out = np.dstack([np.clip(rgb, 0, 255), alpha[..., None]]).astype(np.uint8)
            h, w = alpha.shape
            cached = pygame.image.frombuffer(np.ascontiguousarray(out).tobytes(), (w, h), "RGBA")
            if len(cache) >= self._SHADED_CACHE_LIMIT:
                cache.clear()
            cache[key] = cached
        return cached

    def shaded(self, side: int, level: float | None, azimuth_deg: int, elevation_deg: int,
               strength: float) -> pygame.Surface:
        return self._shaded_for(self._shaded, self.get, self.normals, side, level, azimuth_deg, elevation_deg, strength)

    def legs_shaded(self, side: int, level: float | None, azimuth_deg: int, elevation_deg: int,
                     strength: float) -> pygame.Surface:
        return self._shaded_for(self._legs_shaded, self.get_legs, self.legs_normals, side, level, azimuth_deg, elevation_deg, strength)

    def upper_shaded(self, side: int, level: float | None, azimuth_deg: int, elevation_deg: int,
                      strength: float) -> pygame.Surface:
        return self._shaded_for(self._upper_shaded, self.get_upper, self.upper_normals, side, level, azimuth_deg, elevation_deg, strength)


_sprite_set_cache: dict[RGB, PlayerSpriteSet] = {}


def get_sprite_set(params: PlayerSpriteParams, shirt_color: RGB) -> PlayerSpriteSet:
    """Builds (once) and returns the cached `PlayerSpriteSet` for
    `shirt_color`. Called at most a handful of times per process (one per
    distinct colour actually used: the two team colours, plus the
    goalkeeper colour the first time a keeper is drawn)."""
    cached = _sprite_set_cache.get(shirt_color)
    if cached is None:
        cached = PlayerSpriteSet(params, shirt_color)
        _sprite_set_cache[shirt_color] = cached
    return cached


# ---------------------------------------------------------------------------
# Lighting: per-part rounded shading of the flat sprite art.
# ---------------------------------------------------------------------------

_SHADE_AMBIENT = 0.45   # brightness of a surface facing away from the light, before `strength`


def _alpha_of(sprite: pygame.Surface) -> np.ndarray:
    """The sprite's alpha channel as a float32 (h, w) array in 0..255."""
    return np.array(pygame.surfarray.array_alpha(sprite), dtype=np.float32).T


def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur with edge replication (cumulative sums: cheap at any radius)."""
    radius = max(1, int(radius))
    out = a
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
        n = out.shape[axis]
        hi = np.take(c, np.arange(n) + 2 * radius + 1, axis=axis)
        lo = np.take(c, np.arange(n), axis=axis)
        out = (hi - lo) / (2 * radius + 1)
    return out.astype(np.float32)


def sprite_normals(alpha: np.ndarray) -> np.ndarray:
    """Surface normals (h, w, 3, unit length, z up out of the pitch) for a flat sprite, from its
    silhouette alone. Treats the (blurred, at two scales) alpha as a height map, so every body part
    reads as its own rounded lump -- shoulders, torso, head, each leg and arm -- with the tilt
    concentrated toward each part's edge, and broad flat areas facing straight up."""
    a = alpha / 255.0
    size = max(a.shape)
    height = 0.55 * _box_blur(_box_blur(_box_blur(a, size * 0.035), size * 0.035), size * 0.035) \
        + 0.45 * _box_blur(_box_blur(_box_blur(a, size * 0.09), size * 0.09), size * 0.09)
    gy, gx = np.gradient(height)
    k = size * 0.11
    nx, ny, nz = -k * gx, -k * gy, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)


def light_from_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Unit vector TOWARD a light: azimuth is the screen-plane angle of its horizontal direction
    (0 = +x, 90 = +y, y down), elevation the angle above the ground plane."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)


def shade_factors(normals: np.ndarray, alpha: np.ndarray, light: tuple[float, float, float],
                  strength: float) -> np.ndarray:
    """Brightness factors (h, w), <= 1, to multiply the sprite's colours by: Lambert against
    `light` plus an ambient floor, scaled so the best-lit part of the figure is exactly 1 (only
    the far sides darken, never the whole sprite) and blended toward flat by `strength`
    (0 = all ones)."""
    lx, ly, lz = light
    lit = np.clip(1.1 * (normals[..., 0] * lx + normals[..., 1] * ly + normals[..., 2] * lz), 0.0, 1.0)
    f = _SHADE_AMBIENT + (1.0 - _SHADE_AMBIENT) * lit
    body = alpha > 127
    top = float(np.percentile(f[body], 92)) if body.any() else 1.0
    f = np.clip(f / max(top, 1e-6), 0.0, 1.0)
    return (1.0 - strength * (1.0 - f)).astype(np.float32)


# ---------------------------------------------------------------------------
# Gait animation: turns a player's current speed into a pose.
# ---------------------------------------------------------------------------

def advance_gait_phase(phase_rad: float, speed_mps: float, dt_s: float, params: PlayerSpriteParams) -> float:
    """Advances the per-player stride phase by ``dt_s`` seconds of SIMULATION time
    (the time between two rendered frames on the match's own clock -- see
    ``ui/sim_clock.py``; a fixed 1/fps step made strides play at 1/sim_speed of
    their true rate). Stride frequency scales linearly with speed (0 at rest,
    `max_stride_hz` at `top_speed_for_stride_mps`), so a stationary player's phase
    simply stops advancing (see `pick_pose` for why a frozen phase still shows the
    correct pose in that case). A single update never advances more than
    `_MAX_CYCLES_PER_UPDATE` cycles, so a very high sim speed can't alias the stride."""
    top_speed = max(params.top_speed_for_stride_mps, 1e-6)
    speed_frac = max(0.0, min(1.0, speed_mps / top_speed))
    freq_hz = params.max_stride_hz * speed_frac
    cycles = min(freq_hz * max(0.0, dt_s), _MAX_CYCLES_PER_UPDATE)
    return (phase_rad + 2.0 * math.pi * cycles) % (2.0 * math.pi)


def pick_pose(phase_rad: float, speed_mps: float, params: PlayerSpriteParams) -> tuple[int, float | None]:
    """Maps (stride phase, current speed) to a (leading_side, stride_level)
    pose key (`stride_level=None` means the standing pose).

    The phase alone would freeze mid-stride if a player stops abruptly
    (frequency drops to 0, so `advance_gait_phase` stops advancing it) --
    to avoid a stationary player looking frozen mid-run, the actual stride
    *extension* shown is the phase's sine envelope multiplied by a
    speed-dependent cap that itself goes to 0 at rest (jogging players get
    a partial stride for the same reason, per `extension_speed_exponent`)."""
    top_speed = max(params.top_speed_for_stride_mps, 1e-6)
    speed_frac = max(0.0, min(1.0, speed_mps / top_speed))
    max_extension = speed_frac ** max(params.extension_speed_exponent, 1e-6)

    phase_rad = phase_rad % (2.0 * math.pi)
    side = 1 if phase_rad < math.pi else -1
    within_half = phase_rad if side == 1 else phase_rad - math.pi
    swing = math.sin(within_half)  # 0 -> 1 -> 0 across the half-cycle
    extension = swing * max_extension

    if extension < STRIDE_LEVELS[0] / 2.0:
        return side, None
    nearest = min(STRIDE_LEVELS, key=lambda lvl: abs(lvl - extension))
    return side, nearest
