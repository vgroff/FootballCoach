"""In-game log: a small ring buffer of timestamped messages displayed in the
UI's bottom corner.  Designed to be zero-cost in headless/test use: Match
only calls the log_callback when it is non-None, and GameLog itself is only
created by the App, not by the engine.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum, auto


class LogLevel(Enum):
    """Message priority, analogous to Python's logging module levels."""
    INFO = auto()   # brief one-line summaries (goal, tackle outcome, possession)
    DEBUG = auto()  # verbose numeric breakdowns (rolls, modifiers, etc.)


@dataclass
class LogEntry:
    time_s: float
    level: LogLevel
    message: str
    # Optional multi-line ("\n"-separated) breakdown shown in a hover
    # tooltip next to this entry in the UI (renderer.py's draw_game_log) --
    # e.g. a tackle's skill-check roll/modifier numbers. None = no hover
    # detail, the common case. Generic over any future caller: nothing
    # here is tackle-specific, so any _log_info(msg, detail=...) call gets
    # the same hover-explain treatment for free.
    detail: str | None = None


class GameLog:
    """Ring-buffer log of match events for UI display.

    ``max_entries`` — maximum entries kept (oldest evicted automatically).
    ``add(level, msg, time_s, detail)`` — append a new entry.
    ``entries_above(min_level)`` — iterate entries at or above min_level in
    insertion order (oldest first, newest last).
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._entries: collections.deque[LogEntry] = collections.deque(maxlen=max_entries)
        self.max_entries = max_entries

    def add(self, level: LogLevel, message: str, time_s: float = 0.0, detail: str | None = None) -> None:
        self._entries.append(LogEntry(time_s=time_s, level=level, message=message, detail=detail))

    def entries_above(self, min_level: LogLevel) -> list[LogEntry]:
        """Return all entries at or above *min_level* in insertion order.

        INFO is always included; DEBUG entries are filtered out unless
        min_level is DEBUG.
        """
        if min_level == LogLevel.DEBUG:
            return list(self._entries)
        return [e for e in self._entries if e.level == LogLevel.INFO]

    @property
    def all_entries(self) -> list[LogEntry]:
        return list(self._entries)
