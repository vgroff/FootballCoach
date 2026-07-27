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
from footballcoach.ui.input import MatchInputController
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
            elif self.screen == Screen.MATCH:
                self._handle_match_mouse_event(event)

    def _handle_keydown(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            if self.screen == Screen.MATCH:
                self.screen = Screen.MENU
                self.match = None
                self.input_controller = None
            else:
                self.running = False
        elif key == pygame.K_SPACE and self.match is not None:
            self.match.paused = not self.match.paused

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
        for player in self.match.players:
            self.renderer.draw_player(self.surface, player, selected=player.player_id == selected_id)
        self.renderer.draw_ball(self.surface, self.match.ball)

        drag = self.input_controller.drag_indicator()
        if drag is not None:
            start_world, current_screen = drag
            self.renderer.draw_drag_indicator(self.surface, start_world, current_screen, style.DRAG_KICK_LINE)

        left, right = self.match.scoreboard.left_goals, self.match.scoreboard.right_goals
        hud_lines = [
            self.mode_label,
            f"Score: LEFT {left} - {right} RIGHT",
            f"{'PAUSED' if self.match.paused else 'Playing'} (Space to pause/resume, Esc for menu)",
        ]
        if selected_id is not None:
            hud_lines.append(f"Selected: {selected_id} (click ground=move, drag=kick, click opponent=tackle)")
        self.renderer.draw_hud_text(self.surface, hud_lines)


def run_app() -> None:
    App().run()
