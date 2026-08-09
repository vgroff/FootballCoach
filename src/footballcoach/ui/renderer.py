"""Draws the pitch, players, ball, and HUD to a pygame surface. Pure
rendering - no game logic or input handling lives here (see input.py / app.py).
"""
from __future__ import annotations

import collections
import math
from typing import TYPE_CHECKING

import pygame
import pygame.gfxdraw
from footballcoach.config import load_graphics_config
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.ui import style
from footballcoach.ui.camera import Camera

from footballcoach.mathutils import Vector3

if TYPE_CHECKING:
    from footballcoach.ui.gamelog import GameLog, LogLevel
    from footballcoach.ui.input import KickUIState
    from footballcoach.ui.scenarios import AnyScenarioParam, ScenarioBoolParam, ScenarioChoiceParam, ScenarioParam

# Font family names tried in order when searching for a font that can render
# Unicode emoji/symbols.  The monochrome Noto Emoji font is best on Linux;
# Symbola is a good fallback; if none match we fall back to the pygame default
# (icons will render as replacement boxes on unsupported fonts, which is benign).
_EMOJI_FONT_CANDIDATES = [
    "noto emoji",
    "notoemoji",
    "noto color emoji",
    "symbola",
    "unifont",
    "seguiemj",
]


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
        self._possession_outline_thickness: int = gcfg["player"].get("possession_outline_thickness", 2)
        self._inactive_alpha: int = int(gcfg["player"].get("inactive_alpha", style.INACTIVE_ALPHA))
        self.min_ball_radius_px: int = gcfg["ball"]["min_radius_px"]
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
        _pn = gcfg.get("pause_notification", {})
        self.pause_notification_font = pygame.font.Font(style.FONT_NAME, _pn.get("font_size_px", 26))

        # Ball spin dots — 3D model projected to top-down view
        _sd = gcfg.get("ball_spin_dots", {})
        self._spin_dot_count: int = _sd.get("count", 9)
        self._spin_orbit_frac: float = _sd.get("projection_scale_fraction", _sd.get("orbit_radius_fraction", 0.93))
        self._spin_dot_radius_frac: float = _sd.get("dot_radius_fraction", 0.25)
        _col = _sd.get("color", [30, 30, 30])
        self._spin_dot_color: tuple = (int(_col[0]), int(_col[1]), int(_col[2]))
        # 3x3 rotation matrix tracking ball orientation (identity = initial pose)
        self._ball_orientation: list = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
        # Fixed dot positions on unit sphere (Fibonacci lattice)
        self._ball_dot_positions: list = self._make_fibonacci_sphere(self._spin_dot_count)
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

    @staticmethod
    def _make_fibonacci_sphere(n: int) -> list:
        """Evenly distribute n points on the unit sphere using the Fibonacci lattice."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        for i in range(n):
            y = (1.0 - (i / (n - 1)) * 2.0) if n > 1 else 0.0
            r = math.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            points.append((math.cos(theta) * r, y, math.sin(theta) * r))
        return points

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
        Call once per rendered frame (only when not paused)."""
        # Estimate XY velocity from position delta (works for both free and possessed).
        cur_x, cur_y = ball.position.x, ball.position.y
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

    def draw_pitch(self, surface: pygame.Surface, pitch: Pitch) -> None:
        surface.fill(style.PITCH_GREEN)
        cam = self.camera
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

        # Penalty boxes and six-yard boxes, both ends.
        half_box_w = pitch.box_width_m / 2.0
        half_six_w = pitch.six_yard_width_m / 2.0
        rect_world(-pitch.half_length, -half_box_w, -pitch.half_length + pitch.box_length_m, half_box_w)
        rect_world(pitch.half_length - pitch.box_length_m, -half_box_w, pitch.half_length, half_box_w)
        rect_world(-pitch.half_length, -half_six_w, -pitch.half_length + pitch.six_yard_length_m, half_six_w)
        rect_world(pitch.half_length - pitch.six_yard_length_m, -half_six_w, pitch.half_length, half_six_w)

        # Penalty spots.
        for left in (True, False):
            spot = pitch.penalty_spot(left=left)
            spot_px = cam.world_to_screen(spot.x, spot.y)
            pygame.draw.circle(surface, style.PITCH_LINE_WHITE, spot_px, max(2, line_w // 2))

        # Goal mouths, drawn as a small rectangle protruding outside the pitch.
        half_goal_w = pitch.goal_width_m / 2.0
        goal_depth_m = 2.0
        rect_world(-pitch.half_length - goal_depth_m, -half_goal_w, -pitch.half_length, half_goal_w)
        rect_world(pitch.half_length, -half_goal_w, pitch.half_length + goal_depth_m, half_goal_w)

    def draw_ball(self, surface: pygame.Surface, ball: Ball) -> None:
        cam = self.camera
        pos = cam.world_to_screen(ball.position.x, ball.position.y)

        # A true-to-scale ball (radius 0.11m) is only ~1px at typical zoom
        # levels, so we enforce a minimum on-screen radius for visibility -
        # positions stay physically accurate, only the drawn dot size is
        # boosted. The height effect is then exaggerated on top of that
        # minimum, and a small height label is shown, per the design spec.
        base_radius_px = max(self.min_ball_radius_px, cam.scale_length(ball.radius_m))
        height_boost = 1.0 + min(ball.height_m, 5.0) * self._ball_height_boost_per_m
        radius_px = max(2, int(base_radius_px * height_boost))

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
                ts = pygame.Surface((gr * 2 + 2, gr * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(ts, (*style.BALL_COLOUR, alpha), (gr + 1, gr + 1), gr)
                surface.blit(ts, (tp[0] - gr - 1, tp[1] - gr - 1))

        pygame.draw.circle(surface, style.BALL_COLOUR, pos, radius_px)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px, style.BALL_COLOUR)
        if self._ball_outline and self._ball_outline_width > 0.0:
            if self._ball_outline_width >= 1.0:
                pygame.draw.circle(surface, style.BALL_OUTLINE, pos, radius_px, int(self._ball_outline_width))
                pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px, style.BALL_OUTLINE)
            else:
                # Sub-pixel: draw 1px outline at reduced alpha for a softer border
                _oa = int(self._ball_outline_width * 255)
                _ots = pygame.Surface((radius_px * 2 + 2, radius_px * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(_ots, (*style.BALL_OUTLINE, _oa), (radius_px + 1, radius_px + 1), radius_px, 1)
                surface.blit(_ots, (pos[0] - radius_px - 1, pos[1] - radius_px - 1))

        # Ball state indicator rings (drawn on top of the ball circle).
        # Priority: just_bounced > flying > rolling (mutually exclusive for display).
        _ring_r = radius_px + self._ring_offset_px
        if self._ring_show_bounced and ball.just_bounced_timer_s > 0.0:
            pygame.draw.circle(surface, self._ring_color_bounced, pos, _ring_r, self._ring_width_px)
        elif self._ring_show_flying and ball.position.z > ball.radius_m + self._flying_min_height_m and ball.possessed_by is None:
            pygame.draw.circle(surface, self._ring_color_flying, pos, _ring_r, self._ring_width_px)
        elif self._ring_show_rolling:
            pygame.draw.circle(surface, self._ring_color_rolling, pos, _ring_r, self._ring_width_px)

        # --- Dots: fixed points on the 3D ball surface, projected top-down ---
        # Always shown; rotate as the ball spins. Front hemisphere only.
        # Clipped to the ball circle so dots don't bleed outside the edge.
        orbit_r = radius_px * self._spin_orbit_frac
        dot_r = max(1, int(radius_px * self._spin_dot_radius_frac))
        pad = int(orbit_r) + dot_r + 2
        ds = pygame.Surface((pad * 2, pad * 2), pygame.SRCALPHA)
        R = self._ball_orientation
        for (lx, ly, lz) in self._ball_dot_positions:
            wx = R[0][0]*lx + R[0][1]*ly + R[0][2]*lz
            wy = R[1][0]*lx + R[1][1]*ly + R[1][2]*lz
            wz = R[2][0]*lx + R[2][1]*ly + R[2][2]*lz
            sx = pad + int(wx * orbit_r)
            sy = pad - int(wy * orbit_r)  # y inverted: world +y = screen up
            if wz < 0:
                continue  # back hemisphere — hidden from top-down camera
            pygame.draw.circle(ds, (*self._spin_dot_color, 220), (sx, sy), dot_r)
        # Clip dots inside the outline so they don't bleed over the alpha border
        clip = pygame.Surface((pad * 2, pad * 2), pygame.SRCALPHA)
        clip_r = max(1, radius_px - int(max(1, self._ball_outline_width)))
        pygame.draw.circle(clip, (255, 255, 255, 255), (pad, pad), clip_r)
        ds.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surface.blit(ds, (pos[0] - pad, pos[1] - pad))

        if ball.height_m > 0.15:
            label = self.hud_font.render(f"{ball.height_m:.1f}m", True, style.HUD_TEXT)
            surface.blit(label, (pos[0] + radius_px + 2, pos[1] - label.get_height() // 2))

    def draw_player(
        self, surface: pygame.Surface, player: Player, selected: bool,
        has_ball: bool = False, action_icon: str | None = None,
    ) -> None:
        """Draws one player. Per the design spec:
        - goalkeepers are drawn in a distinct orange colour rather than
          their team colour.
        - the player currently in possession (`has_ball`) gets a white
          outline. Callers are responsible for drawing the ball-carrier
          *last* among players (see app.py's draw order) so they render on
          top of everyone else, since a raw z-order isn't otherwise tracked.
        - inactive players (`PlayerState.INACTIVE_TACKLED`, including a
          tackler briefly off-balance after a failed tackle - see
          engine/knowledge.md) are drawn translucent rather than solid,
          instead of a flat grey tint, so their team/goalkeeper colour is
          still faintly visible.
        - `action_icon`: if set, a small emoji/text label is drawn above the
          player for the configured linger duration (tracked by app.py).
        """
        cam = self.camera
        pos = cam.world_to_screen(player.position.x, player.position.y)
        radius_px = max(self.min_player_radius_px, cam.scale_length(player.radius_m))

        if player.is_goalkeeper:
            colour = style.GOALKEEPER_COLOUR
        else:
            colour = style.TEAM_LEFT_COLOUR if player.team == Team.LEFT else style.TEAM_RIGHT_COLOUR

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

        if is_inactive:
            # Draw on a small per-pixel-alpha surface so the player reads as
            # translucent rather than a flat grey substitute colour.
            diameter = radius_px * 2 + 4
            player_surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
            centre = (diameter // 2, diameter // 2)
            pygame.draw.circle(player_surf, (*colour, self._inactive_alpha), centre, radius_px)
            surface.blit(player_surf, (pos[0] - centre[0], pos[1] - centre[1]))
        else:
            pygame.draw.circle(surface, colour, pos, radius_px)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px, colour)

        # --- Low-stamina flash: outermost ring, pulsing at configured hz ---
        if player.stamina < self._stamina_flash_threshold and not is_inactive:
            period_ms = 1000.0 / max(self._stamina_flash_hz, 0.1)
            flash_on = (pygame.time.get_ticks() % int(period_ms * 2)) < int(period_ms)
            if flash_on:
                pygame.draw.circle(surface, style.STAMINA_FLASH_OUTLINE, pos, radius_px + 11, 2)
                pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px + 11, style.STAMINA_FLASH_OUTLINE)

        # State outline rings: CONTROLLING_BALL (cyan) and INACTIVE_TACKLED (red).
        if player.state == PlayerState.CONTROLLING_BALL:
            pygame.draw.circle(surface, style.CONTROL_DELAY_OUTLINE, pos, radius_px + 4, 2)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px + 4, style.CONTROL_DELAY_OUTLINE)
        elif is_inactive:
            pygame.draw.circle(surface, style.INACTIVE_OUTLINE, pos, radius_px + 4, 2)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px + 4, style.INACTIVE_OUTLINE)

        if has_ball:
            pygame.draw.circle(surface, style.POSSESSION_OUTLINE, pos, radius_px + 2, self._possession_outline_thickness)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px + 2, style.POSSESSION_OUTLINE)
        if selected:
            pygame.draw.circle(surface, style.SELECTED_OUTLINE, pos, radius_px + 7, 2)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius_px + 7, style.SELECTED_OUTLINE)

        # Heading indicator - a short line showing facing direction.
        if self._heading_alpha > 0:
            heading_len_px = radius_px + self._heading_length_px
            tip_dx = math.cos(-player.heading_rad) * heading_len_px
            tip_dy = math.sin(-player.heading_rad) * heading_len_px
            # Use a small SRCALPHA surface so the alpha is respected.
            pad = 2
            sx = int(min(pos[0], pos[0] + tip_dx)) - pad
            sy = int(min(pos[1], pos[1] + tip_dy)) - pad
            sw = int(abs(tip_dx)) + pad * 2 + 2
            sh = int(abs(tip_dy)) + pad * 2 + 2
            h_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            lp0 = (pos[0] - sx, pos[1] - sy)
            lp1 = (pos[0] + tip_dx - sx, pos[1] + tip_dy - sy)
            r, g, b = style.PITCH_LINE_WHITE
            pygame.draw.aaline(h_surf, (r, g, b, self._heading_alpha), lp0, lp1)
            surface.blit(h_surf, (sx, sy))

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

            above_goal = p0.z > goal_height_m
            ascending = p1.z > p0.z

            if above_goal:
                colour = style.TRAJ_ABOVE_GOAL
            elif ascending:
                colour = style.TRAJ_ASCENDING
            else:
                colour = style.TRAJ_DESCENDING

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
                        surface, style.TRAJ_ASCENDING,
                        (sx - ox * half, sy - oy * half),
                        (sx + ox * half, sy + oy * half),
                    )

        # --- Endpoint dot ---------------------------------------------------
        last = points[-1]
        end_screen = self.camera.world_to_screen(last.x, last.y)
        _end_col = style.TRAJ_ABOVE_GOAL if last.z > goal_height_m else style.TRAJ_DESCENDING
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
    ) -> None:
        """Draws the most recent log entries as a scrolling text box in the
        bottom-right corner of the screen.  Newest entries at the bottom.
        No interactive scrollbar — intentionally simple for a playtesting tool.
        """
        from footballcoach.ui.gamelog import LogLevel
        entries = game_log.entries_above(min_level)[-max_lines:]
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

        for i, entry in enumerate(entries):
            colour = style.HUD_TEXT if entry.level == LogLevel.INFO else style.HOTKEY_DISABLED
            text = self.hud_font.render(entry.message[:72], True, colour)
            surface.blit(text, (box_x + 4, box_y + 3 + i * line_h))

        if show_linger:
            outcome_str = getattr(game_log, "linger_outcome", None) or "resetting"
            label = self.hud_font.render(f"⏳ {outcome_str}", True, style.HUD_TEXT)
            label_y = box_y + box_h - linger_h
            surface.blit(label, (box_x + 8, label_y))

            bar_y = label_y + line_h
            filled_w = max(2, int((box_w - 8) * linger_frac))
            pygame.draw.rect(surface, (40, 40, 60), (box_x + 4, bar_y, box_w - 8, line_h - 4), border_radius=3)
            pygame.draw.rect(surface, style.HUD_ACCENT, (box_x + 4, bar_y, filled_w, line_h - 4), border_radius=3)

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

    def draw_scenario_params(
        self,
        surface: pygame.Surface,
        params: list["AnyScenarioParam"],
        values: dict[str, object],
        title: str = "Scenario Parameters",
        open_choice_param: "str | None" = None,
    ) -> dict[str, tuple[pygame.Rect, pygame.Rect]]:
        """Draws the scenario-parameter adjustment screen and returns a dict
        mapping param name → (left_rect, right_rect) for click detection.

        - ScenarioParam:       [-]  value  [+]
        - ScenarioChoiceParam: click value area to open dropdown, or [>] to cycle
        - ScenarioBoolParam:   [checkbox]  label

        When ``open_choice_param`` is set, a dropdown list is rendered below that
        param row.  Dropdown option rects are keyed as ``"{name}__option__{value}"``.
        """
        from footballcoach.ui.scenarios import ScenarioBoolParam, ScenarioChoiceParam
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
        if open_choice_param is not None:
            open_param = next((p for p in params if p.name == open_choice_param), None)
            if open_param is not None and isinstance(open_param, ScenarioChoiceParam):
                open_idx = params.index(open_param)
                open_y = start_y + open_idx * row_h + row_h  # just below the row
                item_h = 30
                list_w = 200 + btn_w + gap
                list_x = col_minus_x
                # Draw backdrop
                list_rect = pygame.Rect(list_x - 2, open_y - 2, list_w + 4, len(open_param.choices) * item_h + 4)
                pygame.draw.rect(surface, (25, 25, 38), list_rect, border_radius=4)
                pygame.draw.rect(surface, style.HUD_ACCENT, list_rect, 1, border_radius=4)
                current = values.get(open_param.name, open_param.default)
                for ci, choice in enumerate(open_param.choices):
                    item_rect = pygame.Rect(list_x, open_y + ci * item_h, list_w, item_h)
                    hov = item_rect.collidepoint(mouse_pos)
                    sel = (choice == current)
                    item_bg = (60, 90, 60) if sel else ((55, 55, 75) if hov else (30, 30, 45))
                    pygame.draw.rect(surface, item_bg, item_rect)
                    label_str = str(choice)
                    if len(label_str) > 32:
                        label_str = "…" + label_str[-31:]
                    lsurf = self.hud_font.render(label_str, True, style.HUD_ACCENT if sel else style.HUD_TEXT)
                    surface.blit(lsurf, (item_rect.x + 6, item_rect.y + (item_h - lsurf.get_height()) // 2))
                    button_rects[f"{open_param.name}__option__{choice}"] = (item_rect, item_rect)

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
        return button_rects
