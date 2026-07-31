"""Draws the pitch, players, ball, and HUD to a pygame surface. Pure
rendering - no game logic or input handling lives here (see input.py / app.py).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from footballcoach.config import load_graphics_config
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.ui import style
from footballcoach.ui.camera import Camera

if TYPE_CHECKING:
    from footballcoach.ui.gamelog import GameLog, LogLevel
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
        self.min_ball_radius_px: int = gcfg["ball"]["min_radius_px"]
        sl = gcfg["speed_lines"]
        self._speed_line_threshold: float = sl["threshold_mps"]
        self._speed_line_count: int = sl["count"]
        self._speed_line_length_px: int = sl["length_px"]
        self._speed_line_gap_px: int = sl["gap_px"]
        sf = gcfg["stamina_flash"]
        self._stamina_flash_threshold: float = sf["threshold"]
        self._stamina_flash_hz: float = sf["flash_hz"]

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
        height_boost = 1.0 + min(ball.height_m, 5.0) * 0.35  # exaggerated
        radius_px = max(2, int(base_radius_px * height_boost))

        pygame.draw.circle(surface, style.BALL_COLOUR, pos, radius_px)
        pygame.draw.circle(surface, style.BALL_OUTLINE, pos, radius_px, 1)

        # Ball state indicator rings (drawn on top of the ball circle).
        # Priority: just_bounced > flying > rolling (mutually exclusive for display).
        if ball.just_bounced_timer_s > 0.0:
            pygame.draw.circle(surface, style.BALL_STATE_BOUNCED_OUTLINE, pos, radius_px + 3, 2)
        elif ball.position.z > 0.05 and ball.possessed_by is None:
            pygame.draw.circle(surface, style.BALL_STATE_FLYING_OUTLINE, pos, radius_px + 3, 2)
        elif ball.velocity.length_xy() > 0.05 and ball.possessed_by is None:
            pygame.draw.circle(surface, style.BALL_STATE_ROLLING_OUTLINE, pos, radius_px + 3, 2)

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
            # Trailing lines go opposite to heading direction (behind the player).
            trail_dx = -math.cos(-player.heading_rad)
            trail_dy = -math.sin(-player.heading_rad)
            for i in range(self._speed_line_count):
                start_dist = radius_px + (i + 1) * self._speed_line_gap_px
                sx = pos[0] + trail_dx * start_dist
                sy = pos[1] + trail_dy * start_dist
                ex = sx + trail_dx * self._speed_line_length_px
                ey = sy + trail_dy * self._speed_line_length_px
                pygame.draw.line(surface, style.SPEED_LINE_COLOUR, (int(sx), int(sy)), (int(ex), int(ey)), 2)

        if is_inactive:
            # Draw on a small per-pixel-alpha surface so the player reads as
            # translucent rather than a flat grey substitute colour.
            diameter = radius_px * 2 + 4
            player_surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
            centre = (diameter // 2, diameter // 2)
            pygame.draw.circle(player_surf, (*colour, style.INACTIVE_ALPHA), centre, radius_px)
            surface.blit(player_surf, (pos[0] - centre[0], pos[1] - centre[1]))
        else:
            pygame.draw.circle(surface, colour, pos, radius_px)

        # --- Low-stamina flash: outermost ring, pulsing at configured hz ---
        if player.stamina < self._stamina_flash_threshold and not is_inactive:
            period_ms = 1000.0 / max(self._stamina_flash_hz, 0.1)
            flash_on = (pygame.time.get_ticks() % int(period_ms * 2)) < int(period_ms)
            if flash_on:
                pygame.draw.circle(surface, style.STAMINA_FLASH_OUTLINE, pos, radius_px + 11, 2)

        # State outline rings: CONTROLLING_BALL (cyan) and INACTIVE_TACKLED (red).
        # These are separate from the possession/selection outlines, and stack
        # outward so they're each visible simultaneously.
        if player.state == PlayerState.CONTROLLING_BALL:
            pygame.draw.circle(surface, style.CONTROL_DELAY_OUTLINE, pos, radius_px + 4, 2)
        elif is_inactive:
            pygame.draw.circle(surface, style.INACTIVE_OUTLINE, pos, radius_px + 4, 2)

        if has_ball:
            pygame.draw.circle(surface, style.POSSESSION_OUTLINE, pos, radius_px + 2, 2)
        if selected:
            pygame.draw.circle(surface, style.SELECTED_OUTLINE, pos, radius_px + 7, 2)

        # Heading indicator - a short line showing facing direction.
        heading_len_px = radius_px + 8
        tip = (
            pos[0] + math.cos(-player.heading_rad) * heading_len_px,
            pos[1] + math.sin(-player.heading_rad) * heading_len_px,
        )
        pygame.draw.line(surface, style.PITCH_LINE_WHITE, pos, tip, 2)

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
        linger_h = line_h + 2 if show_linger else 0
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
            # Hourglass / sand-timer progress bar above the bottom of the log box.
            bar_y = box_y + box_h - linger_h
            filled_w = max(2, int((box_w - 8) * linger_frac))
            pygame.draw.rect(surface, (40, 40, 60), (box_x + 4, bar_y + 3, box_w - 8, line_h - 4), border_radius=3)
            pygame.draw.rect(surface, style.HUD_ACCENT, (box_x + 4, bar_y + 3, filled_w, line_h - 4), border_radius=3)
            label = self.hud_font.render("⏳ resetting…", True, style.HUD_TEXT)
            surface.blit(label, (box_x + 8, bar_y + 3))

    def draw_pause_notification(self, surface: pygame.Surface, message: str) -> None:
        """Draws a prominent centred banner when the game is auto-paused after
        a human-issued order completes.  Rendered above the hotkey bar."""
        sw, sh = surface.get_size()
        bar_h = 34  # hotkey bar height

        padding_x, padding_y = 28, 14
        text_surf = self.title_font.render(message, True, style.HUD_ACCENT)
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
