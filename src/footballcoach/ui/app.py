"""Top-level pygame application: a simple menu to choose between Training
mode and the Balance Scenario picker, plus the shared match-viewing/control
screen used by both.
"""
from __future__ import annotations

from enum import Enum, auto

import pygame

from footballcoach.engine.match import Match
from footballcoach.ui import scenarios, style
from footballcoach.ui.camera import Camera
from footballcoach.ui.input import MatchInputController, OrderMode
from footballcoach.ui.renderer import Renderer

FPS = 60


class Screen(Enum):
    MENU = auto()
    MATCH = auto()


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Football Coach")

        self.camera = Camera.fit_to_pitch(self._temp_pitch())
        self.surface = pygame.display.set_mode((self.camera.screen_width, self.camera.screen_height))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.camera)

        self.screen = Screen.MENU
        self.match: Match | None = None
        self.input_controller: MatchInputController | None = None
        self.mode_label = ""
        self.running = True
        self.is_training_mode = False
        self._last_goal_tally = (0, 0)
        self.show_help = False
        self.help_button_rect = pygame.Rect(0, 0, 90, 32)  # positioned per-screen in _draw_match

    @staticmethod
    def _temp_pitch():
        from footballcoach.entities import Pitch

        return Pitch.standard()

    def run(self) -> None:
        while self.running:
            dt_ms = self.clock.tick(FPS)
            self._handle_events()
            if self.screen == Screen.MATCH and self.match is not None:
                self._step_match()
            self._draw()
            pygame.display.flip()
        pygame.quit()

    # -- event handling -----------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
            elif self.screen == Screen.MENU and event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_menu_click(event.pos)
            elif self.screen == Screen.MATCH and event.type == pygame.MOUSEBUTTONDOWN and self.help_button_rect.collidepoint(event.pos):
                self.show_help = not self.show_help
            elif self.screen == Screen.MATCH and not self.show_help:
                self._handle_match_mouse_event(event)

    def _handle_keydown(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
            elif self.input_controller is not None and self.input_controller.order_mode == OrderMode.PASS:
                self.input_controller.cancel_pass_mode()
            elif self.screen == Screen.MATCH:
                self.screen = Screen.MENU
                self.match = None
                self.input_controller = None
            else:
                self.running = False
            return
        if key == pygame.K_h:
            self.show_help = not self.show_help
            return
        if self.show_help or self.match is None or self.input_controller is None:
            return
        if key == pygame.K_SPACE:
            self.match.paused = not self.match.paused
        elif key == pygame.K_p:
            self.input_controller.enter_pass_mode()
        elif key == pygame.K_s:
            self.input_controller.issue_save_order()
        elif key == pygame.K_x:
            self.input_controller.issue_stop_order()

    def _handle_match_mouse_event(self, event: pygame.event.Event) -> None:
        if self.input_controller is None:
            return
        shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.input_controller.handle_mouse_down(event.pos)
        elif event.type == pygame.MOUSEMOTION:
            self.input_controller.handle_mouse_motion(event.pos, shift_held)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.input_controller.handle_mouse_up(event.pos)

    def _handle_menu_click(self, pos: tuple[int, int]) -> None:
        item = self._menu_item_at(pos)
        if item is None:
            return
        kind, key = item
        if kind == "training":
            self._start_match(scenarios.make_training_match(), "Training mode", is_training_mode=True)
        elif kind == "scenario":
            definition = next(s for s in scenarios.SCENARIOS if s.key == key)
            self._start_match(definition.build(0.3), f"Balance scenario: {definition.label}", is_training_mode=False)

    def _start_match(self, match: Match, label: str, is_training_mode: bool = False) -> None:
        self.match = match
        self.input_controller = MatchInputController(match=match, camera=self.camera)
        self.mode_label = label
        self.screen = Screen.MATCH
        self.is_training_mode = is_training_mode
        self._last_goal_tally = (match.scoreboard.left_goals, match.scoreboard.right_goals)

    def _step_match(self) -> None:
        assert self.match is not None
        self.match.step()

        if self.is_training_mode:
            current_tally = (self.match.scoreboard.left_goals, self.match.scoreboard.right_goals)
            if current_tally != self._last_goal_tally:
                self._reset_training_positions()
                self._last_goal_tally = current_tally

    def _reset_training_positions(self) -> None:
        """Training mode is single-player free play: per the design spec, a
        goal resets both the ball (already done by Match) and the player
        back to a neutral starting position, rather than leaving the player
        wherever they happened to be standing."""
        from footballcoach.mathutils import Vector3

        assert self.match is not None
        player = self.match.players[0]
        player.position = Vector3(0, 0, 0)
        player.velocity = Vector3.zero()
        player.current_order = None
        self.match.ball.position = Vector3(3, 0, 0)
        self.match.ball.velocity = Vector3.zero()
        self.match.ball.possessed_by = None

    # -- menu layout ---------------------------------------------------------

    def _menu_items(self) -> list[tuple[str, str, str, pygame.Rect]]:
        """Returns (kind, key, label, screen_rect) for every clickable menu item."""
        items: list[tuple[str, str, str, pygame.Rect]] = []
        x, y = 60, 140
        row_h = 50
        items.append(("training", "", "Training mode (1 player + ball, free play)", pygame.Rect(x, y, 500, row_h - 10)))
        y += row_h + 20
        for definition in scenarios.SCENARIOS:
            items.append(("scenario", definition.key, f"Balance scenario: {definition.label}", pygame.Rect(x, y, 500, row_h - 10)))
            y += row_h
        return items

    def _menu_item_at(self, pos: tuple[int, int]) -> tuple[str, str] | None:
        for kind, key, _label, rect in self._menu_items():
            if rect.collidepoint(pos):
                return kind, key
        return None

    # -- drawing --------------------------------------------------------------

    def _draw(self) -> None:
        if self.screen == Screen.MENU:
            self._draw_menu()
        else:
            self._draw_match()

    def _draw_menu(self) -> None:
        self.surface.fill(style.HUD_BG)
        title = self.renderer.title_font.render("Football Coach", True, style.HUD_ACCENT)
        self.surface.blit(title, (60, 50))

        mouse_pos = pygame.mouse.get_pos()
        for _kind, _key, label, rect in self._menu_items():
            hovered = rect.collidepoint(mouse_pos)
            colour = style.HUD_ACCENT if hovered else style.HUD_TEXT
            pygame.draw.rect(self.surface, (40, 40, 50), rect, border_radius=6)
            text = self.renderer.hud_font.render(label, True, colour)
            self.surface.blit(text, (rect.x + 12, rect.y + (rect.height - text.get_height()) // 2))

        hint = self.renderer.hud_font.render("Esc to quit", True, style.HUD_TEXT)
        self.surface.blit(hint, (60, self.camera.screen_height - 40))

    def _draw_match(self) -> None:
        assert self.match is not None and self.input_controller is not None
        self.renderer.draw_pitch(self.surface, self.match.pitch)

        selected_id = self.input_controller.selected_player_id
        carrier_id = self.match.ball.possessed_by
        # Draw the ball carrier last so they always render on top of every
        # other player, per the design spec.
        ordered_players = sorted(self.match.players, key=lambda p: p.player_id == carrier_id)
        for player in ordered_players:
            self.renderer.draw_player(
                self.surface, player, selected=player.player_id == selected_id,
                has_ball=player.player_id == carrier_id,
            )
        self.renderer.draw_ball(self.surface, self.match.ball)

        drag = self.input_controller.drag_indicator()
        if drag is not None:
            start_world, current_screen = drag
            self.renderer.draw_drag_indicator(self.surface, start_world, current_screen, style.DRAG_KICK_LINE)

        left, right = self.match.scoreboard.left_goals, self.match.scoreboard.right_goals
        hud_lines = [
            self.mode_label,
            f"Score: LEFT {left} - {right} RIGHT",
            f"{'PAUSED' if self.match.paused else 'Playing'} (Space to pause/resume, Esc for menu, H for help)",
        ]
        if selected_id is not None:
            if self.input_controller.order_mode == OrderMode.PASS:
                hud_lines.append(f"Selected: {selected_id} - PASS mode: click a target to pass (Esc cancels)")
            else:
                hud_lines.append(
                    f"Selected: {selected_id} (click ground=move, drag=kick, click opponent=tackle, "
                    "P=pass mode, S=save order if goalkeeper)"
                )
        self.renderer.draw_hud_text(self.surface, hud_lines)
        self._draw_help_button()

        if self.show_help:
            self._draw_help_overlay()

    def _draw_help_button(self) -> None:
        self.help_button_rect = pygame.Rect(self.camera.screen_width - 100, 8, 90, 32)
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.help_button_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self.surface, (50, 50, 65) if not hovered else (70, 70, 90), self.help_button_rect, border_radius=6)
        label = self.renderer.hud_font.render("Help (H)", True, style.HUD_ACCENT)
        self.surface.blit(
            label,
            (
                self.help_button_rect.centerx - label.get_width() // 2,
                self.help_button_rect.centery - label.get_height() // 2,
            ),
        )

    def _draw_help_overlay(self) -> None:
        overlay = pygame.Surface((self.camera.screen_width, self.camera.screen_height), pygame.SRCALPHA)
        overlay.fill((10, 10, 15, 220))
        self.surface.blit(overlay, (0, 0))

        title = self.renderer.title_font.render("Controls", True, style.HUD_ACCENT)
        self.surface.blit(title, (60, 40))

        lines = [
            "Click a player           - select them (click again to deselect)",
            "Click a team-mate        - switch selection (or pass to them in Pass mode)",
            "Click an opponent        - get possession: chase whoever has the ball and tackle them",
            "Click empty ground       - move there (sprinting)",
            "Click-drag from selected - kick: drag direction=aim, length=power (only if they have the ball)",
            "Hold Shift while dragging- loft/chip the kick instead of driving it low",
            "P                        - pass mode: next click (ground or player) passes there",
            "S                        - issue a Save order (goalkeeper only): tracks and blocks shots",
            "X                        - stop: decelerate selected player to a standstill",
            "Space                    - pause/resume the simulation",
            "H or Help button         - toggle this help overlay",
            "Esc                      - close this overlay, or return to the menu / quit",
            "",
            "Visual indicators:",
            "White outline            - player currently in possession of the ball",
            "Orange fill              - goalkeeper",
            "Translucent              - player is temporarily inactive (just tackled, or",
            "                           just missed a tackle attempt)",
        ]
        y = 100
        for line in lines:
            rendered = self.renderer.hud_font.render(line, True, style.HUD_TEXT)
            self.surface.blit(rendered, (60, y))
            y += rendered.get_height() + 6


def run_app() -> None:
    App().run()
