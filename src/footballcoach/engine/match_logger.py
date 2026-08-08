"""Simplified per-match event logger for post-hoc inspection.

Records key events during a match (possession changes, kicks, tackles,
goals, ball-out) plus periodic consistency snapshots. Designed to be
attached to a Match and saved to JSON for offline analysis.

Usage (engine level):
    logger = MatchLogger()
    match.match_logger = logger
    # ... run match ...
    logger.save(Path("match_logs/episode_001.json"))
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from footballcoach.engine.match import Match
    from footballcoach.entities.ball import Ball
    from footballcoach.entities.player import Player


@dataclass
class MatchEvent:
    time_s: float
    event: str  # "start" | "possession_change" | "kick" | "tackle_attempt" | "ball_out" | "goal" | "consistency" | "step_reward" | "episode_end"
    ball_pos: tuple[float, float, float]
    # Only for "start": all players' positions keyed by player_id (values include "team")
    player_positions: Optional[dict] = None
    # Only for "start": ball velocity
    ball_vel: Optional[tuple[float, float, float]] = None
    # Triggering player (for non-consistency, non-start events)
    player_id: Optional[str] = None
    player_pos: Optional[tuple[float, float, float]] = None
    # For "consistency" and "possession_change": who currently has the ball
    possessor_id: Optional[str] = None
    # For "step_reward" / "episode_end" events: this step's reward and breakdown
    reward_total: Optional[float] = None
    reward_components: Optional[dict[str, float]] = None
    # Running episode totals up to and including this step
    reward_cumulative: Optional[dict[str, float]] = None
    # For "episode_end": how the episode ended
    outcome: Optional[str] = None


def _vec(v) -> tuple[float, float, float]:
    return (round(float(v.x), 3), round(float(v.y), 3), round(float(v.z), 3))


def _player_pos(p: "Player") -> tuple[float, float, float]:
    return _vec(p.position)


class MatchLogger:
    """Lightweight event log attached to a Match instance.

    Call ``record_start()`` once when a match/episode begins, then let
    Match's internal methods drive recording via the ``notify_*`` methods.
    Call ``check_consistency()`` at the end of every engine tick (from
    ``Match.step()``) — it only emits an entry when the 5-second silence
    window has elapsed.
    """

    def __init__(self, consistency_interval_s: float = 2.5) -> None:
        self.consistency_interval_s = consistency_interval_s
        self.events: list[MatchEvent] = []
        self._last_event_time_s: float = -999.0
        self._cumulative: dict[str, float] = {}
        self._pending_total: float = 0.0
        self._pending_components: dict[str, float] = {}

    def reset(self) -> None:
        self.events = []
        self._last_event_time_s = -999.0
        self._cumulative = {}
        self._pending_total = 0.0
        self._pending_components = {}

    # ------------------------------------------------------------------
    # Public notify methods — called by Match internals
    # ------------------------------------------------------------------

    def record_start(self, match: "Match") -> None:
        positions = {
            p.player_id: {"pos": _player_pos(p), "team": p.team.name.lower()}
            for p in match.players
        }
        self.events.append(MatchEvent(
            time_s=round(match.time_s, 3),
            event="start",
            ball_pos=_vec(match.ball.position),
            ball_vel=_vec(match.ball.velocity),
            player_positions=positions,
        ))
        self._last_event_time_s = match.time_s

    def notify_possession_change(
        self, time_s: float, ball_pos, new_owner: Optional["Player"], old_owner_id: Optional[str]
    ) -> None:
        total, comps, cumul = self._drain()
        self.events.append(MatchEvent(
            time_s=round(time_s, 3),
            event="possession_change",
            ball_pos=_vec(ball_pos),
            player_id=new_owner.player_id if new_owner is not None else None,
            player_pos=_player_pos(new_owner) if new_owner is not None else None,
            possessor_id=new_owner.player_id if new_owner is not None else None,
            reward_total=total,
            reward_components=comps,
            reward_cumulative=cumul,
        ))
        self._last_event_time_s = time_s

    def notify_kick(self, time_s: float, ball_pos, player: "Player") -> None:
        total, comps, cumul = self._drain()
        self.events.append(MatchEvent(
            time_s=round(time_s, 3),
            event="kick",
            ball_pos=_vec(ball_pos),
            player_id=player.player_id,
            player_pos=_player_pos(player),
            reward_total=total,
            reward_components=comps,
            reward_cumulative=cumul,
        ))
        self._last_event_time_s = time_s

    def notify_tackle_attempt(
        self, time_s: float, ball_pos, tackler: "Player", target: "Player"
    ) -> None:
        total, comps, cumul = self._drain()
        self.events.append(MatchEvent(
            time_s=round(time_s, 3),
            event="tackle_attempt",
            ball_pos=_vec(ball_pos),
            player_id=tackler.player_id,
            player_pos=_player_pos(tackler),
            reward_total=total,
            reward_components=comps,
            reward_cumulative=cumul,
        ))
        self._last_event_time_s = time_s

    def notify_goal(self, time_s: float, ball_pos, scoring_side: str) -> None:
        total, comps, cumul = self._drain()
        self.events.append(MatchEvent(
            time_s=round(time_s, 3),
            event="goal",
            ball_pos=_vec(ball_pos),
            player_id=scoring_side,  # "left" or "right"
            reward_total=total,
            reward_components=comps,
            reward_cumulative=cumul,
        ))
        self._last_event_time_s = time_s

    def notify_ball_out(self, time_s: float, ball_pos) -> None:
        total, comps, cumul = self._drain()
        self.events.append(MatchEvent(
            time_s=round(time_s, 3),
            event="ball_out",
            ball_pos=_vec(ball_pos),
            reward_total=total,
            reward_components=comps,
            reward_cumulative=cumul,
        ))
        self._last_event_time_s = time_s

    def check_consistency(self, time_s: float, ball_pos, possessor_id: Optional[str]) -> None:
        if time_s - self._last_event_time_s >= self.consistency_interval_s:
            total, comps, cumul = self._drain()
            self.events.append(MatchEvent(
                time_s=round(time_s, 3),
                event="consistency",
                ball_pos=_vec(ball_pos),
                possessor_id=possessor_id,
                reward_total=total,
                reward_components=comps,
                reward_cumulative=cumul,
            ))
            self._last_event_time_s = time_s

    def accumulate_reward(self, total: float, components: dict[str, float]) -> None:
        """Called by ScenarioEnv after each non-terminal decision step."""
        self._pending_total += float(total)
        for k, v in components.items():
            self._pending_components[k] = self._pending_components.get(k, 0.0) + float(v)

    def _drain(self) -> tuple[Optional[float], Optional[dict], Optional[dict]]:
        """Drain pending reward into cumulative; return (total, components, cumulative) or Nones."""
        if not self._pending_components and self._pending_total == 0.0:
            return None, None, None
        for k, v in self._pending_components.items():
            self._cumulative[k] = self._cumulative.get(k, 0.0) + v
        total = round(self._pending_total, 5)
        comps = {k: round(v, 5) for k, v in self._pending_components.items() if v != 0.0}
        cumul = {k: round(v, 5) for k, v in self._cumulative.items() if v != 0.0}
        self._pending_total = 0.0
        self._pending_components = {}
        return total, comps, cumul

    def notify_episode_end(self, time_s: float, ball_pos, outcome: str, reward_total: float, components: dict[str, float]) -> None:
        self.accumulate_reward(reward_total, components)
        total, comps, cumul = self._drain()
        self.events.append(MatchEvent(
            time_s=round(time_s, 3),
            event="episode_end",
            ball_pos=_vec(ball_pos),
            outcome=outcome,
            reward_total=total,
            reward_components=comps,
            reward_cumulative=cumul,
        ))
        self._last_event_time_s = time_s

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for ev in self.events:
            d = asdict(ev)
            # Drop None fields to keep the JSON compact
            records.append({k: v for k, v in d.items() if v is not None})
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
