"""Translates mouse/keyboard input into Move/Kick/Pass/Tackle/Save orders on
the currently selected player.

Interaction scheme (per project design):
- Click a player -> select them (click the same selected player again to
  deselect; click a same-team player to switch selection).
- Click an opposing-team player while a player is selected -> issue a
  TackleOrder from the selected player at the clicked player.
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

from footballcoach.engine.match import Match
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, KickOrder, MoveOrder, PassOrder, SaveOrder, StopOrder, TackleOrder
from footballcoach.ui.camera import Camera

CLICK_DRAG_THRESHOLD_PX = 6
MAX_KICK_DRAG_M = 15.0
GROUND_AIM_HEIGHT_M = 0.3
LOFTED_AIM_HEIGHT_M = 2.0
SELECT_TOLERANCE_PX = 6


class OrderMode(Enum):
    """Which order a ground/player click issues to the selected player.

    MOVE is the persistent default (click ground=move, click opponent=
    tackle). PASS is transient: it applies to exactly one click, then
    automatically reverts to MOVE, so the user doesn't have to remember to
    switch back after every pass."""
    MOVE = auto()
    PASS = auto()


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

    def selected_player(self) -> Player | None:
        if self.selected_player_id is None:
            return None
        for p in self.match.players:
            if p.player_id == self.selected_player_id:
                return p
        return None

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
                    selected.current_order = PassOrder(target_position=clicked_player.position)
                    self.order_mode = OrderMode.MOVE  # PASS is a one-shot mode
                else:
                    self.selected_player_id = clicked_player.player_id  # switch selection
                return
            # Opposing player - chase and get possession.
            selected.current_order = GetPossessionOrder()
            return

        # Empty ground - issue a move or pass order to the selected player.
        if selected is not None:
            world_x, world_y = self.camera.screen_to_world(*screen_pos)
            target = Vector3(world_x, world_y, 0.0)
            if self.order_mode == OrderMode.PASS:
                selected.current_order = PassOrder(target_position=target)
                self.order_mode = OrderMode.MOVE  # PASS is a one-shot mode
            else:
                selected.current_order = MoveOrder(target_position=target, sprint=True)

    def enter_pass_mode(self) -> None:
        """The next click (ground or player) issues a PassOrder instead of a
        MoveOrder/selection-switch, then reverts to OrderMode.MOVE."""
        self.order_mode = OrderMode.PASS

    def cancel_pass_mode(self) -> None:
        self.order_mode = OrderMode.MOVE

    def issue_save_order(self) -> None:
        """Issues a SaveOrder to the selected player if they're a
        goalkeeper; no-ops otherwise (mirrors SaveOrder's own silent no-op
        for non-goalkeepers in Match._process_orders)."""
        selected = self.selected_player()
        if selected is not None and selected.is_goalkeeper:
            selected.current_order = SaveOrder()

    def issue_stop_order(self) -> None:
        """Decelerates the selected player to a standstill."""
        selected = self.selected_player()
        if selected is not None:
            selected.current_order = StopOrder()

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

        player.current_order = KickOrder(aim_point=aim_point, power_fraction=power_fraction, spin=Vector3.zero())

    def drag_indicator(self) -> tuple[tuple[float, float], tuple[int, int]] | None:
        """Returns (player_world_xy, current_mouse_screen_pos) if a kick drag
        is in progress, for the renderer to draw an aim line."""
        if not self.drag.active or self.drag.on_player_id is None:
            return None
        player = self.selected_player()
        if player is None:
            return None
        return (player.position.x, player.position.y), self.drag.current_screen
