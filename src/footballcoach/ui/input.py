"""Translates mouse/keyboard input into Move/Kick/Pass/Tackle/Save orders on
the currently selected player.

Interaction scheme (per project design):
- Click a player -> select them (click the same selected player again to
  deselect; click a same-team player to switch selection).
- Click an opposing-team player while a player is selected -> issue a
  ChaseTackleOrder from the selected player at the clicked player.
- Click empty ground while a player is selected -> issue a MoveOrder to that
  point (default `OrderMode.MOVE`).
- Click-and-drag starting ON the selected player (only meaningful if they
  currently have the ball) -> issue a KickOrder: drag direction sets the aim
  direction, drag length sets power (capped), release fires the kick. Hold
  Shift while dragging to loft the kick higher (chip/lob) instead of a low
  driven kick. Independent of order mode - a kick drag always fires a kick.
- Press `P` -> enter `OrderMode.PASS` for one order: the *next* click
  (ground or any player) issues a `PassOrder` targeted at that position
  instead of a `MoveOrder`, then the mode automatically reverts to
  `OrderMode.MOVE` for subsequent clicks.
- Press `S` -> issues a `SaveOrder` immediately if the selected player is a
  goalkeeper (no-op otherwise). Per `orders.SaveOrder`, this does not
  auto-complete - it stays in effect until another order replaces it.
- See `ui/app.py`'s help overlay (`H` key / help button) for the full,
  user-facing control list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from footballcoach.engine.match import Match
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import ChaseTackleOrder, GetPossessionOrder, KickOrder, MoveOrder, PassOrder, SaveOrder, ShootOrder, StopOrder
from footballcoach.ui.camera import Camera

CLICK_DRAG_THRESHOLD_PX = 6
MAX_KICK_DRAG_M = 15.0
GROUND_AIM_HEIGHT_M = 0.3
LOFTED_AIM_HEIGHT_M = 2.0
SELECT_TOLERANCE_PX = 6


class OrderMode(Enum):
    """Which order a ground/player click issues to the selected player.

    MOVE is the persistent default (click ground=move, click opponent=
    tackle). PASS and SHOOT are transient: each applies to exactly one
    click then automatically reverts to MOVE."""
    MOVE = auto()
    PASS = auto()
    SHOOT = auto()


@dataclass
class DragState:
    active: bool = False
    on_player_id: str | None = None
    start_screen: tuple[int, int] = (0, 0)
    current_screen: tuple[int, int] = (0, 0)
    lofted: bool = False


@dataclass
class MatchInputController:
    match: Match
    camera: Camera
    selected_player_id: str | None = None
    drag: DragState = field(default_factory=DragState)
    order_mode: OrderMode = OrderMode.MOVE
    # Called when a human-issued order completes: (player_id, order_name).
    # Set by App after construction to wire the auto-pause notification.
    on_order_complete: Callable[[str, str], None] | None = field(default=None, repr=False, compare=False)

    def selected_player(self) -> Player | None:
        if self.selected_player_id is None:
            return None
        for p in self.match.players:
            if p.player_id == self.selected_player_id:
                return p
        return None

    def _issue_order(self, player: Player, order, order_name: str) -> None:
        """Assign *order* to *player* and attach the on_complete callback so
        the app can auto-pause and display a notification when it finishes."""
        if self.on_order_complete is not None:
            pid = player.player_id
            cb = self.on_order_complete
            name = order_name
            order.on_complete = lambda: cb(pid, name)
        player.current_order = order

    def _player_at_screen_pos(self, screen_pos: tuple[int, int]) -> Player | None:
        best: Player | None = None
        best_dist = float("inf")
        for player in self.match.players:
            px, py = self.camera.world_to_screen(player.position.x, player.position.y)
            radius_px = self.camera.scale_length(player.radius_m) + SELECT_TOLERANCE_PX
            dist = math.hypot(screen_pos[0] - px, screen_pos[1] - py)
            if dist <= radius_px and dist < best_dist:
                best = player
                best_dist = dist
        return best

    def handle_mouse_down(self, screen_pos: tuple[int, int]) -> None:
        clicked_player = self._player_at_screen_pos(screen_pos)
        selected = self.selected_player()

        self.drag = DragState(active=True, start_screen=screen_pos, current_screen=screen_pos)
        if selected is not None and clicked_player is not None and clicked_player.player_id == selected.player_id:
            self.drag.on_player_id = selected.player_id

    def handle_mouse_motion(self, screen_pos: tuple[int, int], shift_held: bool) -> None:
        if self.drag.active:
            self.drag.current_screen = screen_pos
            self.drag.lofted = shift_held

    def handle_mouse_up(self, screen_pos: tuple[int, int]) -> None:
        if not self.drag.active:
            return

        drag_distance_px = math.hypot(
            screen_pos[0] - self.drag.start_screen[0], screen_pos[1] - self.drag.start_screen[1]
        )
        is_drag = drag_distance_px > CLICK_DRAG_THRESHOLD_PX

        if is_drag and self.drag.on_player_id is not None:
            self._finish_kick_drag(screen_pos)
        else:
            self._handle_click(screen_pos)

        self.drag = DragState()

    def _handle_click(self, screen_pos: tuple[int, int]) -> None:
        clicked_player = self._player_at_screen_pos(screen_pos)
        selected = self.selected_player()

        if clicked_player is not None:
            if selected is None:
                self.selected_player_id = clicked_player.player_id
                return
            if clicked_player.player_id == selected.player_id:
                self.selected_player_id = None  # deselect
                return
            if clicked_player.team == selected.team:
                if self.order_mode == OrderMode.PASS:
                    self._issue_order(selected, PassOrder(target_position=clicked_player.position), "Pass")
                    self.order_mode = OrderMode.MOVE  # PASS is a one-shot mode
                elif self.order_mode == OrderMode.SHOOT:
                    # Shoot at the clicked player's position (unusual but valid).
                    self._issue_order(selected, ShootOrder(aim_point=clicked_player.position, power_fraction=1.0), "Shoot")
                    self.order_mode = OrderMode.MOVE
                else:
                    self.selected_player_id = clicked_player.player_id  # switch selection
                return
            # Opposing player - chase and get possession.
            self._issue_order(selected, GetPossessionOrder(), "Get Possession")
            return

        # Empty ground - issue a move, pass, or shoot order to the selected player.
        if selected is not None:
            world_x, world_y = self.camera.screen_to_world(*screen_pos)
            if self.order_mode == OrderMode.PASS:
                target = Vector3(world_x, world_y, 0.0)
                self._issue_order(selected, PassOrder(target_position=target), "Pass")
                self.order_mode = OrderMode.MOVE  # PASS is a one-shot mode
            elif self.order_mode == OrderMode.SHOOT:
                # Aim at the clicked point at a mid-goal height (1.0m).
                target = Vector3(world_x, world_y, 1.0)
                self._issue_order(selected, ShootOrder(aim_point=target, power_fraction=1.0), "Shoot")
                self.order_mode = OrderMode.MOVE  # SHOOT is a one-shot mode
            else:
                target = Vector3(world_x, world_y, 0.0)
                self._issue_order(selected, MoveOrder(target_position=target, sprint=True), "Move")

    def enter_pass_mode(self) -> None:
        """The next click (ground or player) issues a PassOrder instead of a
        MoveOrder/selection-switch, then reverts to OrderMode.MOVE."""
        self.order_mode = OrderMode.PASS

    def enter_shoot_mode(self) -> None:
        """The next click on the pitch issues a ShootOrder aimed at the
        clicked point (at goal-frame height), then reverts to OrderMode.MOVE."""
        self.order_mode = OrderMode.SHOOT

    def cancel_order_mode(self) -> None:
        self.order_mode = OrderMode.MOVE

    def cancel_pass_mode(self) -> None:
        self.order_mode = OrderMode.MOVE

    def issue_save_order(self) -> None:
        """Issues a SaveOrder to the selected player if they're a
        goalkeeper; no-ops otherwise (mirrors SaveOrder's own silent no-op
        for non-goalkeepers in Match._process_orders)."""
        selected = self.selected_player()
        if selected is not None and selected.is_goalkeeper:
            # SaveOrder is persistent (never auto-completes), so no pause callback.
            selected.current_order = SaveOrder()

    def issue_stop_order(self) -> None:
        """Decelerates the selected player to a standstill."""
        selected = self.selected_player()
        if selected is not None:
            self._issue_order(selected, StopOrder(), "Stop")

    def _finish_kick_drag(self, release_screen: tuple[int, int]) -> None:
        player = self.selected_player()
        if player is None:
            return

        start_x, start_y = self.camera.world_to_screen(player.position.x, player.position.y)
        dx_px = release_screen[0] - start_x
        dy_px = release_screen[1] - start_y

        # Screen y grows downward; world y grows "upward" (toward +y), so
        # flip dy when converting to world-space direction.
        drag_length_px = math.hypot(dx_px, dy_px)
        drag_length_m = drag_length_px / self.camera.pixels_per_metre
        if drag_length_m < 1e-6:
            return

        direction_x = dx_px / drag_length_px
        direction_y = -dy_px / drag_length_px

        power_fraction = min(1.0, drag_length_m / MAX_KICK_DRAG_M)
        aim_distance_m = min(drag_length_m * 2.0, 60.0)  # project the aim point out along the drag direction
        aim_height = LOFTED_AIM_HEIGHT_M if self.drag.lofted else GROUND_AIM_HEIGHT_M

        aim_point = Vector3(
            player.position.x + direction_x * aim_distance_m,
            player.position.y + direction_y * aim_distance_m,
            aim_height,
        )

        self._issue_order(player, KickOrder(aim_point=aim_point, power_fraction=power_fraction, spin=Vector3.zero()), "Kick")

    def drag_indicator(self) -> tuple[tuple[float, float], tuple[int, int]] | None:
        """Returns (player_world_xy, current_mouse_screen_pos) if a kick drag
        is in progress, for the renderer to draw an aim line."""
        if not self.drag.active or self.drag.on_player_id is None:
            return None
        player = self.selected_player()
        if player is None:
            return None
        return (player.position.x, player.position.y), self.drag.current_screen
