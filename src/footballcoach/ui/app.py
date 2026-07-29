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
from footballcoach.ui.gamelog import GameLog, LogLevel
from footballcoach.ui.input import MatchInputController, OrderMode
from footballcoach.ui.renderer import Renderer
from footballcoach.ui.scenarios import ScenarioBoolParam, ScenarioChoiceParam, ScenarioParam

FPS = 60


class Screen(Enum):
    MENU = auto()
    MATCH = auto()
    SCENARIO_PARAMS = auto()


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
        self._scenario_loop: scenarios.ScenarioLoop | None = None
        self.show_help = False
        self.help_button_rect = pygame.Rect(0, 0, 90, 32)

        # Game log
        self.game_log = GameLog(max_entries=50)
        self.log_min_level = LogLevel.INFO

        # Auto-pause notification: set when a human-issued order completes.
        # Cleared the next time the player resumes (Space).
        self._pause_notification: str | None = None

        # Pending scenario params (Screen.SCENARIO_PARAMS state)
        self._pending_scenario_definition: scenarios.ScenarioDefinition | None = None
        self._pending_scenario_params: dict[str, float] = {}
        self._params_button_rects: dict[str, tuple[pygame.Rect, pygame.Rect]] = {}

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

    def _log(self, level: LogLevel, msg: str) -> None:
        """Add a message to the game log (from UI layer, not engine)."""
        self.game_log.add(level, msg)

    # -- event handling -----------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
            elif self.screen == Screen.MENU and event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_menu_click(event.pos)
            elif self.screen == Screen.SCENARIO_PARAMS and event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_params_click(event.pos)
            elif self.screen == Screen.MATCH and event.type == pygame.MOUSEBUTTONDOWN and self.help_button_rect.collidepoint(event.pos):
                self.show_help = not self.show_help
            elif self.screen == Screen.MATCH and not self.show_help:
                self._handle_match_mouse_event(event)

    def _handle_keydown(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
            elif self.screen == Screen.SCENARIO_PARAMS:
                self.screen = Screen.MENU
            elif self.input_controller is not None and self.input_controller.order_mode != OrderMode.MOVE:
                self.input_controller.cancel_order_mode()
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
        if key == pygame.K_l and self.screen == Screen.MATCH:
            # Cycle log min_level: INFO -> DEBUG -> INFO
            if self.log_min_level == LogLevel.INFO:
                self.log_min_level = LogLevel.DEBUG
            else:
                self.log_min_level = LogLevel.INFO
            return
        if self.show_help or self.match is None or self.input_controller is None:
            return
        if key == pygame.K_SPACE:
            self.match.paused = not self.match.paused
            if not self.match.paused:
                self._pause_notification = None  # clear on resume
        elif key == pygame.K_p:
            self.input_controller.enter_pass_mode()
        elif key == pygame.K_k:
            self.input_controller.enter_shoot_mode()
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
        if kind == "header":
            return
        if kind == "training":
            self._start_match(scenarios.make_training_match(), "Training mode", is_training_mode=True)
        elif kind == "scenario":
            definition = next(s for s in scenarios.SCENARIOS if s.key == key)
            if definition.params:
                self._enter_params_screen(definition)
            else:
                self._start_scenario(definition)

    def _enter_params_screen(self, definition: scenarios.ScenarioDefinition) -> None:
        self._pending_scenario_definition = definition
        self._pending_scenario_params = {p.name: p.default for p in definition.params}
        self.screen = Screen.SCENARIO_PARAMS

    def _handle_params_click(self, pos: tuple[int, int]) -> None:
        if self._pending_scenario_definition is None:
            return
        for name, (minus_rect, plus_rect) in self._params_button_rects.items():
            if name == "__start__":
                if minus_rect.collidepoint(pos):
                    self._start_scenario_with_params()
                continue
            if name == "__back__":
                if minus_rect.collidepoint(pos):
                    self.screen = Screen.MENU
                continue
            param = next((p for p in self._pending_scenario_definition.params if p.name == name), None)
            if param is None:
                continue

            if isinstance(param, ScenarioBoolParam):
                if minus_rect.collidepoint(pos):  # same rect for both sides
                    self._pending_scenario_params[name] = not bool(
                        self._pending_scenario_params.get(name, param.default)
                    )
            elif isinstance(param, ScenarioChoiceParam):
                current = self._pending_scenario_params.get(name, param.default)
                idx = list(param.choices).index(current) if current in param.choices else 0
                if minus_rect.collidepoint(pos):
                    self._pending_scenario_params[name] = param.choices[(idx - 1) % len(param.choices)]
                elif plus_rect.collidepoint(pos):
                    self._pending_scenario_params[name] = param.choices[(idx + 1) % len(param.choices)]
            else:
                current = self._pending_scenario_params.get(name, param.default)
                if minus_rect.collidepoint(pos):
                    self._pending_scenario_params[name] = max(param.min_value, current - param.step)
                elif plus_rect.collidepoint(pos):
                    self._pending_scenario_params[name] = min(param.max_value, current + param.step)

    def _start_scenario_with_params(self) -> None:
        if self._pending_scenario_definition is None:
            return
        self._start_scenario(
            self._pending_scenario_definition,
            kwargs=dict(self._pending_scenario_params),
        )
        self._pending_scenario_definition = None
        self._pending_scenario_params = {}

    def _on_human_order_complete(self, player_id: str, order_name: str) -> None:
        """Callback fired by input.py when a human-issued order finishes."""
        if self.match is not None:
            self.match.paused = True
        self._pause_notification = f"{player_id}: {order_name} complete — Space to resume"

    def _start_match(self, match: Match, label: str, is_training_mode: bool = False) -> None:
        self.match = match
        self._wire_match_log(match)
        self.input_controller = MatchInputController(match=match, camera=self.camera)
        self.input_controller.on_order_complete = self._on_human_order_complete
        self.mode_label = label
        self.screen = Screen.MATCH
        self.is_training_mode = is_training_mode
        self._scenario_loop = None
        self._last_goal_tally = (match.scoreboard.left_goals, match.scoreboard.right_goals)
        self._pause_notification = None
        if is_training_mode and match.players:
            self.input_controller.selected_player_id = match.players[0].player_id

    def _start_scenario(
        self, definition: scenarios.ScenarioDefinition, kwargs: dict | None = None
    ) -> None:
        """Builds a ScenarioLoop and begins the first trial."""
        loop = scenarios.ScenarioLoop(definition=definition, kwargs=kwargs or {})
        self._scenario_loop = loop
        self.match = loop.match
        self._wire_match_log(loop.match)
        self.input_controller = MatchInputController(match=loop.match, camera=self.camera)
        self.input_controller.on_order_complete = self._on_human_order_complete
        self.mode_label = f"Balance scenario: {definition.label}"
        self.screen = Screen.MATCH
        self.is_training_mode = False
        self._last_goal_tally = (0, 0)
        self._pause_notification = None

    def _wire_match_log(self, match: Match) -> None:
        """Attach the game log callback to a newly created Match."""
        game_log = self.game_log
        match.log_callback = lambda level, msg: game_log.add(level, msg, match.time_s)

    def _step_match(self) -> None:
        assert self.match is not None

        if self._scenario_loop is not None:
            loop = self._scenario_loop
            trial_ended = loop.step()
            self.match = loop.match
            if loop.complete:
                self.screen = Screen.MENU
                self.match = None
                self.input_controller = None
                self._scenario_loop = None
            elif trial_ended:
                # New trial started — wire log to the fresh match and sync input.
                self._wire_match_log(loop.match)
                self.input_controller = MatchInputController(match=loop.match, camera=self.camera)
                self.input_controller.on_order_complete = self._on_human_order_complete
            else:
                if self.input_controller is not None:
                    self.input_controller.match = loop.match
            return

        self.match.step()
        if self.is_training_mode:
            current_tally = (self.match.scoreboard.left_goals, self.match.scoreboard.right_goals)
            # Wait until the engine's goal linger is done (linger_remaining == 0)
            # before repositioning the player — otherwise we'd reset mid-linger.
            if current_tally != self._last_goal_tally and self.match._goal_linger_remaining_s <= 0.0:
                self._reset_training_positions()
                self._last_goal_tally = current_tally

    def _reset_training_positions(self) -> None:
        """Training mode: reposition the player to start after a goal."""
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
        """Returns (kind, key, label, screen_rect) for every clickable menu item.

        kind: 'training' | 'scenario' | 'header' (headers are non-clickable labels)
        """
        items: list[tuple[str, str, str, pygame.Rect]] = []
        x, y = 60, 110
        row_h = 42
        header_h = 28

        # -- AI scenarios section --
        items.append(("header", "", "── AI Scenarios ──", pygame.Rect(x, y, 500, header_h)))
        y += header_h + 4
        ai_keys = {"phase1_neural_ai", "1v1_phase1"}
        for definition in scenarios.SCENARIOS:
            if definition.key in ai_keys:
                items.append(("scenario", definition.key, definition.label, pygame.Rect(x + 16, y, 500, row_h - 6)))
                y += row_h

        # -- Training mode --
        y += 8
        items.append(("header", "", "── Freeplay ──", pygame.Rect(x, y, 500, header_h)))
        y += header_h + 4
        items.append(("training", "", "Training mode (1 player + ball, free play)", pygame.Rect(x + 16, y, 500, row_h - 6)))
        y += row_h

        # -- Balance scenarios --
        y += 8
        items.append(("header", "", "── Balance Scenarios ──", pygame.Rect(x, y, 500, header_h)))
        y += header_h + 4
        balance_keys = ai_keys  # skip these
        for definition in scenarios.SCENARIOS:
            if definition.key not in balance_keys:
                items.append(("scenario", definition.key, definition.label, pygame.Rect(x + 16, y, 500, row_h - 6)))
                y += row_h

        return items

    def _menu_item_at(self, pos: tuple[int, int]) -> tuple[str, str] | None:
        for kind, key, _label, rect in self._menu_items():
            if kind != "header" and rect.collidepoint(pos):
                return kind, key
        return None

    # -- drawing --------------------------------------------------------------

    def _draw(self) -> None:
        if self.screen == Screen.MENU:
            self._draw_menu()
        elif self.screen == Screen.SCENARIO_PARAMS:
            self._draw_params_screen()
        else:
            self._draw_match()

    def _draw_menu(self) -> None:
        self.surface.fill(style.HUD_BG)
        title = self.renderer.title_font.render("Football Coach", True, style.HUD_ACCENT)
        self.surface.blit(title, (60, 50))

        mouse_pos = pygame.mouse.get_pos()
        for kind, _key, label, rect in self._menu_items():
            if kind == "header":
                text = self.renderer.hud_font.render(label, True, (120, 130, 160))
                self.surface.blit(text, (rect.x, rect.y + (rect.height - text.get_height()) // 2))
            else:
                hovered = rect.collidepoint(mouse_pos)
                colour = style.HUD_ACCENT if hovered else style.HUD_TEXT
                pygame.draw.rect(self.surface, (50, 50, 65) if hovered else (38, 38, 52), rect, border_radius=6)
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

        hud_lines = [self.mode_label]
        paused_str = "PAUSED" if self.match.paused else "Playing"
        if self._scenario_loop is not None:
            loop = self._scenario_loop
            if loop.max_trials > 0:
                hud_lines.append(f"Trial {loop.trial_count + 1}/{loop.max_trials}  |  {paused_str}")
            else:
                hud_lines.append(f"Trial {loop.trial_count + 1}  |  {paused_str}")
            o = loop.outcomes
            total = sum(o.values())
            if total > 0:
                tally = f"Goals: {o['goal']}  Saved: {o['saved']}  Miss: {o['miss']}"
                if o.get("dispossessed", 0):
                    tally += f"  Disp: {o['dispossessed']}"
                if o["other"]:
                    tally += f"  Other: {o['other']}"
                hud_lines.append(tally)
        else:
            left, right = self.match.scoreboard.left_goals, self.match.scoreboard.right_goals
            hud_lines.append(f"Score: LEFT {left} - {right} RIGHT  |  {paused_str}")
        if selected_id is not None:
            mode = self.input_controller.order_mode
            if mode != OrderMode.MOVE:
                hud_lines.append(f"Selected: {selected_id}  [{mode.name} mode - Esc cancels]")
            else:
                hud_lines.append(f"Selected: {selected_id}")
        self.renderer.draw_hud_text(self.surface, hud_lines)
        self._draw_help_button()
        self.renderer.draw_hotkey_bar(self.surface, self._hotkey_entries())
        self.renderer.draw_game_log(self.surface, self.game_log, self.log_min_level)
        if self._pause_notification:
            self.renderer.draw_pause_notification(self.surface, self._pause_notification)

        if self.show_help:
            self._draw_help_overlay()

    def _draw_params_screen(self) -> None:
        if self._pending_scenario_definition is None:
            self.screen = Screen.MENU
            return
        defn = self._pending_scenario_definition
        self._params_button_rects = self.renderer.draw_scenario_params(
            self.surface,
            defn.params,
            self._pending_scenario_params,
            title=f"Configure: {defn.label}",
        )

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
            "K                        - shoot mode: next click sets the aim point for a full-power shot",
            "S                        - issue a Save order (goalkeeper only): tracks and blocks shots",
            "X                        - stop: decelerate selected player to a standstill",
            "Space                    - pause/resume the simulation",
            "H or Help button         - toggle this help overlay",
            "L                        - cycle game log level (INFO / DEBUG)",
            "Esc                      - close this overlay, or return to the menu / quit",
            "",
            "Visual indicators:",
            "White outline            - player currently in possession of the ball",
            "Cyan ring                - player mid first-touch control delay",
            "Red ring                 - player temporarily inactive (just tackled)",
            "Orange fill              - goalkeeper",
            "Translucent              - player is temporarily inactive (just tackled, or",
            "                           just missed a tackle attempt)",
            "Blue ball ring           - ball airborne",
            "Green ball ring          - ball rolling",
            "Amber ball ring          - ball just bounced",
        ]
        y = 100
        for line in lines:
            rendered = self.renderer.hud_font.render(line, True, style.HUD_TEXT)
            self.surface.blit(rendered, (60, y))
            y += rendered.get_height() + 6


    def _hotkey_entries(self) -> list[tuple[str, str, bool, bool]]:
        """Returns (key_label, action_label, enabled, active) for every hotkey.

        *enabled* controls brightness: True = player can use this right now,
        False = action not currently valid (no selection, no ball, etc.) and
        rendered dim.  *active* highlights the key in accent colour when it
        is the current transient mode (PASS / SHOOT).
        """
        ic = self.input_controller
        match = self.match
        if ic is None or match is None:
            return []
        selected = ic.selected_player()
        has_ball = selected is not None and match.ball.possessed_by == selected.player_id
        is_gk = selected is not None and selected.is_goalkeeper
        is_selected = selected is not None
        mode = ic.order_mode
        return [
            ("[Spc]", "Pause" if not match.paused else "Resume", True, False),
            ("[P]",   "Pass",   has_ball,    mode == OrderMode.PASS),
            ("[K]",   "Shoot",  has_ball,    mode == OrderMode.SHOOT),
            ("[S]",   "Save",   is_gk,       False),
            ("[X]",   "Stop",   is_selected, False),
            ("[H]",   "Help",   True,        self.show_help),
            ("[Esc]", "Menu",   True,        False),
        ]


def run_app() -> None:
    App().run()
