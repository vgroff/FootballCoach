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
    # Only ever set on the merged copies returned by
    # ``GameLog.collapsed_entries`` (never on the raw buffered entries):
    # how many consecutive identical messages this row stands for, and the
    # time of the last of them (``time_s`` stays the first's). A raw,
    # unmerged entry is ``count == 1`` with ``end_time_s == time_s``.
    count: int = 1
    end_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.end_time_s is None:
            self.end_time_s = self.time_s


class GameLog:
    """Ring-buffer log of match events for UI display.

    ``max_entries`` — maximum entries kept (oldest evicted automatically).
    ``add(level, msg, time_s, detail)`` — append a new entry.
    ``entries_above(min_level)`` — iterate entries at or above min_level in
    insertion order (oldest first, newest last).
    ``collapsed_entries(min_level)`` — same, but runs of consecutive
    identical messages are merged into one row (what the UI draws).
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

    def collapsed_entries(self, min_level: LogLevel) -> list[LogEntry]:
        """Like ``entries_above`` but with each run of consecutive entries
        sharing the same ``message`` merged into a single row, so spam (e.g.
        the same player tackling over and over) doesn't push everything else
        out of the UI's few visible lines.

        "Consecutive" is judged on the level-filtered list, so an INFO
        message interleaved with DEBUG lines still collapses in INFO view.
        Only exact message matches merge, and never across a different
        message in between (A, B, A stays three rows — order matters).
        The merged row keeps the first entry's ``time_s``, takes the last's
        ``end_time_s``, and carries the *latest* entry's ``detail`` (the
        hover breakdown of the most recent occurrence). Returns new
        ``LogEntry`` objects — the buffered raw entries are never mutated.

        Counts only cover entries still in the ring buffer: a run longer
        than ``max_entries`` reports at most ``max_entries``.
        """
        merged: list[LogEntry] = []
        for e in self.entries_above(min_level):
            last = merged[-1] if merged else None
            if last is not None and last.message == e.message and last.level == e.level:
                last.count += e.count
                last.end_time_s = e.end_time_s
                last.detail = e.detail
            else:
                merged.append(LogEntry(
                    time_s=e.time_s, level=e.level, message=e.message,
                    detail=e.detail, count=e.count, end_time_s=e.end_time_s,
                ))
        return merged

    @property
    def all_entries(self) -> list[LogEntry]:
        return list(self._entries)
