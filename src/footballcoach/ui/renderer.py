"""Draws the pitch, players, ball, and HUD to a pygame surface. Pure
rendering - no game logic or input handling lives here (see input.py / app.py).
"""
from __future__ import annotations

import math

import pygame

from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.ui import style
from footballcoach.ui.camera import Camera


class Renderer:
    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        pygame.font.init()
        self.hud_font = pygame.font.Font(style.FONT_NAME, style.HUD_FONT_SIZE)
        self.title_font = pygame.font.Font(style.FONT_NAME, style.TITLE_FONT_SIZE)

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
        base_radius_px = max(style.MIN_BALL_RADIUS_PX, cam.scale_length(ball.radius_m))
        height_boost = 1.0 + min(ball.height_m, 5.0) * 0.35  # exaggerated
        radius_px = max(2, int(base_radius_px * height_boost))

        pygame.draw.circle(surface, style.BALL_COLOUR, pos, radius_px)
        pygame.draw.circle(surface, style.BALL_OUTLINE, pos, radius_px, 1)

        if ball.height_m > 0.15:
            label = self.hud_font.render(f"{ball.height_m:.1f}m", True, style.HUD_TEXT)
            surface.blit(label, (pos[0] + radius_px + 2, pos[1] - label.get_height() // 2))

    def draw_player(self, surface: pygame.Surface, player: Player, selected: bool) -> None:
        cam = self.camera
        pos = cam.world_to_screen(player.position.x, player.position.y)
        radius_px = max(style.MIN_PLAYER_RADIUS_PX, cam.scale_length(player.radius_m))

        colour = style.TEAM_LEFT_COLOUR if player.team == Team.LEFT else style.TEAM_RIGHT_COLOUR
        if player.state == PlayerState.INACTIVE_TACKLED:
            colour = style.INACTIVE_TINT

        pygame.draw.circle(surface, colour, pos, radius_px)
        if player.is_goalkeeper:
            pygame.draw.circle(surface, style.GOALKEEPER_STRIPE, pos, radius_px, 2)
        if selected:
            pygame.draw.circle(surface, style.SELECTED_OUTLINE, pos, radius_px + 3, 2)

        # Heading indicator - a short line showing facing direction.
        heading_len_px = radius_px + 8
        tip = (
            pos[0] + math.cos(-player.heading_rad) * heading_len_px,
            pos[1] + math.sin(-player.heading_rad) * heading_len_px,
        )
        pygame.draw.line(surface, style.PITCH_LINE_WHITE, pos, tip, 2)

        label = self.hud_font.render(player.player_id, True, style.HUD_TEXT)
        surface.blit(label, (pos[0] - label.get_width() // 2, pos[1] + radius_px + 2))

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
