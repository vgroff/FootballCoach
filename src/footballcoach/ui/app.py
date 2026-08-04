"""Top-level pygame application: a simple menu to choose between Training
mode and the Balance Scenario picker, plus the shared match-viewing/control
screen used by both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import pygame

from footballcoach.config import load_graphics_config
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


@dataclass
class ActionIconState:
    """Tracks per-player action icons (kick/tackle/save/etc.) so they can
    linger on screen for a moment after the triggering event, independent of
    how often the engine actually fires the underlying callback.
    """
    linger_s: float
    _icons: dict[str, tuple[str, float]] = field(default_factory=dict)

    def clear(self) -> None:
        self._icons.clear()

    def record(self, player_id: str, icon: str, now_s: float) -> None:
        self._icons[player_id] = (icon, now_s + self.linger_s)

    def active_icon(self, player_id: str, now_s: float) -> str | None:
        """Returns the still-lingering icon for *player_id*, if any, expiring
        (and forgetting) it if its linger window has passed."""
        entry = self._icons.get(player_id)
        if entry is None:
            return None
        icon, expiry = entry
        if now_s < expiry:
            return icon
        del self._icons[player_id]
        return None


@dataclass
class ScenarioParamsUIState:
    """State for the Screen.SCENARIO_PARAMS configuration screen (picking
    scenario parameters before starting a balance-scenario trial run)."""
    definition: scenarios.ScenarioDefinition | None = None
    values: dict[str, float] = field(default_factory=dict)
    button_rects: dict[str, tuple[pygame.Rect, pygame.Rect]] = field(default_factory=dict)
    open_choice_param: str | None = None  # which ScenarioChoiceParam dropdown is open

    def reset_for(self, definition: scenarios.ScenarioDefinition) -> None:
        self.definition = definition
        all_params = list(definition.params) + scenarios.UNIVERSAL_PARAMS
        self.values = {p.name: p.default for p in all_params}
        self.open_choice_param = None

    def clear(self) -> None:
        self.definition = None
        self.values = {}


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Football Coach")

        _gcfg = load_graphics_config()
        _cam_cfg = _gcfg.get("camera", {})
        self.camera = Camera.fit_to_pitch(
            self._temp_pitch(),
            pixels_per_metre=_cam_cfg.get("pixels_per_metre", 9.0),
            margin_px=int(_cam_cfg.get("margin_px", 40)),
        )
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
        # True iff the current pause was triggered automatically (order complete /
        # kick issued). False = user pressed Space manually.
        self._auto_paused: bool = False

        # Action icon state: player.action_icon is polled each frame after
        # stepping and recorded here with a wall-clock expiry so icons
        # linger briefly on screen (see ActionIconState).
        gcfg = load_graphics_config()
        self._action_icons = ActionIconState(linger_s=gcfg["action_icons"]["linger_s"])

        # Pending scenario params (Screen.SCENARIO_PARAMS state)
        self._scenario_params_ui = ScenarioParamsUIState()

        # Simulation speed multiplier: physics steps per visual frame.
        # Cycle with ] (faster) and [ (slower).
        self._sim_speed: int = 1
        self._SIM_SPEED_OPTIONS: tuple[int, ...] = (1, 2, 4, 8)

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
            elif self.input_controller is not None and self.input_controller.kick_ui_state() is not None:
                self.input_controller.cancel_kick_ui()
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
                self._pause_notification = None
                self._auto_paused = False
        elif key == pygame.K_p:
            self.input_controller.enter_pass_mode()
        elif key == pygame.K_k:
            if not self.input_controller.try_enter_kick_ui():
                self.input_controller.enter_shoot_mode()
        elif key == pygame.K_s:
            self.input_controller.issue_save_order()
        elif key == pygame.K_x:
            self.input_controller.issue_stop_order()
        elif key == pygame.K_RIGHTBRACKET:
            idx = self._SIM_SPEED_OPTIONS.index(self._sim_speed)
            self._sim_speed = self._SIM_SPEED_OPTIONS[(idx + 1) % len(self._SIM_SPEED_OPTIONS)]
        elif key == pygame.K_LEFTBRACKET:
            idx = self._SIM_SPEED_OPTIONS.index(self._sim_speed)
            self._sim_speed = self._SIM_SPEED_OPTIONS[(idx - 1) % len(self._SIM_SPEED_OPTIONS)]

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
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if self.input_controller.kick_ui_state() is not None:
                self.input_controller.regress_kick_ui()

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
            self._enter_params_screen(definition)

    def _enter_params_screen(self, definition: scenarios.ScenarioDefinition) -> None:
        self._scenario_params_ui.reset_for(definition)
        self.screen = Screen.SCENARIO_PARAMS

    def _handle_params_click(self, pos: tuple[int, int]) -> None:
        ui = self._scenario_params_ui
        if ui.definition is None:
            return

        # Check dropdown option rects first (they overlay other content when open).
        for key, (r, _) in ui.button_rects.items():
            if "__option__" in key and r.collidepoint(pos):
                param_name, _, idx_str = key.partition("__option__")
                ui.values[param_name] = idx_str  # stored as string key
                ui.open_choice_param = None
                return

        for name, (minus_rect, plus_rect) in ui.button_rects.items():
            if "__option__" in name:
                continue
            if name == "__start__":
                if minus_rect.collidepoint(pos):
                    ui.open_choice_param = None
                    self._start_scenario_with_params()
                continue
            if name == "__back__":
                if minus_rect.collidepoint(pos):
                    ui.open_choice_param = None
                    self.screen = Screen.MENU
                continue
            all_params = list(ui.definition.params) + scenarios.UNIVERSAL_PARAMS
            param = next((p for p in all_params if p.name == name), None)
            if param is None:
                continue

            if isinstance(param, ScenarioBoolParam):
                if minus_rect.collidepoint(pos):
                    ui.values[name] = not bool(ui.values.get(name, param.default))
                    break
            elif isinstance(param, ScenarioChoiceParam):
                # Clicking the value area (minus_rect == value_rect here) toggles the dropdown.
                # Clicking [>] cycles. Only act if this row was actually hit.
                if minus_rect.collidepoint(pos):
                    ui.open_choice_param = name if ui.open_choice_param != name else None
                    break
                elif plus_rect.collidepoint(pos):
                    current = ui.values.get(name, param.default)
                    idx = list(param.choices).index(current) if current in param.choices else 0
                    ui.values[name] = param.choices[(idx + 1) % len(param.choices)]
                    ui.open_choice_param = None
                    break
            else:
                current = ui.values.get(name, param.default)
                if minus_rect.collidepoint(pos):
                    ui.values[name] = max(param.min_value, current - param.step)
                    break
                elif plus_rect.collidepoint(pos):
                    ui.values[name] = min(param.max_value, current + param.step)
                    break

    def _start_scenario_with_params(self) -> None:
        ui = self._scenario_params_ui
        if ui.definition is None:
            return
        kwargs = dict(ui.values)
        timeout_ticks = int(kwargs.pop("timeout_ticks", 800))
        sim_speed_str = kwargs.pop("sim_speed", "1x")
        sim_speed = int(sim_speed_str.rstrip("x"))
        self._start_scenario(
            ui.definition,
            kwargs=kwargs,
            timeout_ticks=timeout_ticks,
            sim_speed=sim_speed,
        )
        ui.clear()

    def _on_human_order_complete(self, player_id: str, order_name: str) -> None:
        """Callback fired by input.py when a human-issued order finishes."""
        if self.match is not None:
            self.match.paused = True
            self._auto_paused = True
        self._pause_notification = f"{player_id}: {order_name} complete — Space to resume"

    def _on_human_order_issued(self, player_id: str, order_name: str, is_debug: bool) -> None:
        """Callback fired by input.py when the human issues any order."""
        level = LogLevel.DEBUG if is_debug else LogLevel.INFO
        t = self.match.time_s if self.match is not None else 0.0
        self.game_log.add(level, f"[You] {player_id}: {order_name}", t)

    def _on_new_order(self) -> None:
        """Callback fired by input.py whenever any new order is issued.

        Always resumes play regardless of whether the pause was automatic or
        manual — issuing an order is an intent to act, so the game should run
        to execute it without needing a separate Space press.
        """
        self._pause_notification = None
        self._auto_paused = False
        if self.match is not None:
            self.match.paused = False

    def _on_kick_ui_entered(self) -> None:
        """Callback fired when kick UI phase 1 is entered.

        Pauses the game immediately so the player freezes while the user aims.
        """
        if self.match is not None:
            self.match.paused = True
            self._auto_paused = True
        self._pause_notification = "Aiming kick — click to advance phases  ·  Esc/RClick cancel"

    def _on_kick_issued(self) -> None:
        """Callback fired when the kick order is queued (after phase 3 click).

        Game is already paused; just update the notification message.
        """
        self._pause_notification = "Kick queued — Space to execute"

    def _start_match(self, match: Match, label: str, is_training_mode: bool = False) -> None:
        self.match = match
        self._wire_match_log(match)
        self._wire_player_icon_callbacks(match)
        self._action_icons.clear()
        self.input_controller = MatchInputController(match=match, camera=self.camera)
        self.input_controller.on_order_complete = self._on_human_order_complete
        self.input_controller.on_new_order = self._on_new_order
        self.input_controller.on_kick_ui_entered = self._on_kick_ui_entered
        self.input_controller.on_kick_issued = self._on_kick_issued
        self.input_controller.on_order_issued = self._on_human_order_issued
        self.mode_label = label
        self.screen = Screen.MATCH
        self.is_training_mode = is_training_mode
        self._scenario_loop = None
        self._last_goal_tally = (match.scoreboard.left_goals, match.scoreboard.right_goals)
        self._pause_notification = None
        self._auto_paused = False
        if is_training_mode and match.players:
            self.input_controller.selected_player_id = match.players[0].player_id

    def _start_scenario(
        self, definition: scenarios.ScenarioDefinition, kwargs: dict | None = None,
        timeout_ticks: int = 800, sim_speed: int = 1,
    ) -> None:
        """Builds a ScenarioLoop and begins the first trial."""
        self._sim_speed = sim_speed
        loop = scenarios.ScenarioLoop(definition=definition, kwargs=kwargs or {}, timeout_ticks=timeout_ticks)
        self._scenario_loop = loop
        self.match = loop.match
        self._wire_match_log(loop.match)
        self._wire_player_icon_callbacks(loop.match)
        self._action_icons.clear()
        self.input_controller = MatchInputController(match=loop.match, camera=self.camera)
        self.input_controller.on_order_complete = self._on_human_order_complete
        self.input_controller.on_new_order = self._on_new_order
        self.input_controller.on_kick_ui_entered = self._on_kick_ui_entered
        self.input_controller.on_kick_issued = self._on_kick_issued
        self.input_controller.on_order_issued = self._on_human_order_issued
        self.mode_label = f"Balance scenario: {definition.label}"
        self.screen = Screen.MATCH
        self.is_training_mode = False
        self._last_goal_tally = (0, 0)
        self._pause_notification = None
        self._auto_paused = False

    def _wire_match_log(self, match: Match) -> None:
        """Attach the game log callback to a newly created Match."""
        game_log = self.game_log
        match.log_callback = lambda level, msg: game_log.add(level, msg, match.time_s)

    def _wire_player_icon_callbacks(self, match: Match) -> None:
        """Wire on_kick / on_tackle / on_possession_gained callbacks on every
        player in *match* so the engine can signal the UI action-icon system.

        Icons are set on the player's `action_icon` field; `_poll_action_icons`
        picks them up each frame and records them with a wall-clock expiry.
        """
        def _kick_cb(player):  # noqa: ANN001
            player.action_icon = "⚽"

        def _tackle_cb(player):  # noqa: ANN001
            player.action_icon = "🧤" if player.is_goalkeeper else "🦵"

        def _possession_cb(player):  # noqa: ANN001
            # Only show icon for goalkeepers (catch/save); outfield first-touch
            # control is already signalled via _update_loose_ball_pickup → "✋".
            if player.is_goalkeeper:
                player.action_icon = "🧤"

        for player in match.players:
            player.on_kick = _kick_cb
            player.on_tackle = _tackle_cb
            player.on_possession_gained = _possession_cb

    def _poll_action_icons(self, match: Match) -> None:
        """After each physics step, harvest `player.action_icon` signals set by
        the engine and record them in `self._action_icons` with a wall-clock
        expiry.  The field is cleared immediately so each event is only counted
        once even if multiple ticks fire between rendered frames.
        """
        now_s = pygame.time.get_ticks() / 1000.0
        for player in match.players:
            if player.action_icon is not None:
                self._action_icons.record(player.player_id, player.action_icon, now_s)
                player.action_icon = None

    def _step_match(self) -> None:
        assert self.match is not None
        # Run 0 steps when paused so orders (e.g. a queued kick) don't execute
        # until the user presses Space.  The HUD renders from match state and
        # doesn't need a physics tick to stay up to date.
        steps = self._sim_speed if not self.match.paused else 0

        if self._scenario_loop is not None:
            loop = self._scenario_loop
            for _ in range(steps):
                trial_ended = loop.step()
                self.match = loop.match
                if loop.complete:
                    self.screen = Screen.MENU
                    self.match = None
                    self.input_controller = None
                    self._scenario_loop = None
                    return
                elif trial_ended:
                    # New trial started — wire log and icon callbacks to the fresh match.
                    self._wire_match_log(loop.match)
                    self._wire_player_icon_callbacks(loop.match)
                    self._action_icons.clear()
                    self.input_controller = MatchInputController(match=loop.match, camera=self.camera)
                    self.input_controller.on_order_complete = self._on_human_order_complete
                    self.input_controller.on_new_order = self._on_new_order
                    self.input_controller.on_kick_ui_entered = self._on_kick_ui_entered
                    self.input_controller.on_kick_issued = self._on_kick_issued
                    self.input_controller.on_order_issued = self._on_human_order_issued
                    break  # render one frame of the new trial before stepping further
                else:
                    if self.input_controller is not None:
                        self.input_controller.match = loop.match
            self._poll_action_icons(self.match)
            return

        for _ in range(steps):
            self.match.step()
        self._poll_action_icons(self.match)
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

        Layout: two columns.  Left column = AI scenarios + Freeplay.
        Right column = Balance scenarios.
        """
        items: list[tuple[str, str, str, pygame.Rect]] = []
        col_w = 480
        row_h = 42
        header_h = 28
        margin = 60
        gap = 32  # gap between columns
        col1_x = margin
        col2_x = margin + col_w + gap

        # ── Left column ──────────────────────────────────────────────────────
        y1 = 110
        items.append(("header", "", "── AI Scenarios ──", pygame.Rect(col1_x, y1, col_w, header_h)))
        y1 += header_h + 4
        ai_keys = {"phase1_neural_ai", "1v1_phase1"}
        for definition in scenarios.SCENARIOS:
            if definition.key in ai_keys:
                items.append(("scenario", definition.key, definition.label,
                               pygame.Rect(col1_x + 16, y1, col_w - 16, row_h - 6)))
                y1 += row_h

        y1 += 8
        items.append(("header", "", "── Freeplay ──", pygame.Rect(col1_x, y1, col_w, header_h)))
        y1 += header_h + 4
        items.append(("training", "", "Training mode (1 player + ball, free play)",
                       pygame.Rect(col1_x + 16, y1, col_w - 16, row_h - 6)))

        # ── Right column ─────────────────────────────────────────────────────
        y2 = 110
        items.append(("header", "", "── Balance Scenarios ──", pygame.Rect(col2_x, y2, col_w, header_h)))
        y2 += header_h + 4
        balance_keys = ai_keys  # skip AI entries already in left column
        for definition in scenarios.SCENARIOS:
            if definition.key not in balance_keys:
                items.append(("scenario", definition.key, definition.label,
                               pygame.Rect(col2_x + 16, y2, col_w - 16, row_h - 6)))
                y2 += row_h

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
        now_s = pygame.time.get_ticks() / 1000.0
        for player in ordered_players:
            pid = player.player_id
            action_icon = self._action_icons.active_icon(pid, now_s)
            self.renderer.draw_player(
                self.surface, player, selected=pid == selected_id,
                has_ball=pid == carrier_id,
                action_icon=action_icon,
            )
        if not self.match.paused:
            self.renderer.update_ball_effects(self.match.ball, 1.0 / FPS)
        self.renderer.draw_ball(self.surface, self.match.ball)

        kick_state = self.input_controller.kick_ui_state()
        if kick_state is not None:
            selected = self.input_controller.selected_player()
            if selected is not None:
                self.renderer.draw_kick_ui(
                    self.surface,
                    kick_state,
                    selected,
                    self.match.pitch.goal_height_m,
                    bottom_reserve_px=70 if self._pause_notification else 0,
                )

        hud_lines = [self.mode_label]
        speed_str = f"  ⚡{self._sim_speed}x" if self._sim_speed > 1 else ""
        paused_str = "PAUSED" if self.match.paused else f"Playing{speed_str}"
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
            kick_st = self.input_controller.kick_ui_state()
            mode = self.input_controller.order_mode
            if kick_st is not None:
                hud_lines.append(f"Selected: {selected_id}  [KICK phase {kick_st.phase.name} - Esc cancels]")
            elif mode != OrderMode.MOVE:
                hud_lines.append(f"Selected: {selected_id}  [{mode.name} mode - Esc cancels]")
            else:
                hud_lines.append(f"Selected: {selected_id}")
        self.renderer.draw_hud_text(self.surface, hud_lines)
        self._draw_help_button()
        self.renderer.draw_hotkey_bar(self.surface, self._hotkey_entries())
        # Expose linger progress to the game-log renderer (0.0 = not lingering).
        if self._scenario_loop is not None and self._scenario_loop._pending_outcome is not None:
            loop = self._scenario_loop
            total = loop.linger_s if loop._pending_outcome in ("goal", "saved", "dispossessed", "other") else loop.linger_s * 0.5
            self.game_log.linger_frac = loop._linger_remaining_s / max(total, 0.001)
        else:
            self.game_log.linger_frac = 0.0
        self.renderer.draw_game_log(self.surface, self.game_log, self.log_min_level)
        if self._pause_notification:
            self.renderer.draw_pause_notification(self.surface, self._pause_notification)

        if self.show_help:
            self._draw_help_overlay()

    def _draw_params_screen(self) -> None:
        ui = self._scenario_params_ui
        if ui.definition is None:
            self.screen = Screen.MENU
            return
        defn = ui.definition
        all_params = list(defn.params) + scenarios.UNIVERSAL_PARAMS
        ui.button_rects = self.renderer.draw_scenario_params(
            self.surface,
            all_params,
            ui.values,
            title=f"Configure: {defn.label}",
            open_choice_param=ui.open_choice_param,
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
            "] / [                    - increase / decrease simulation speed (1x / 2x / 4x / 8x)",
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
