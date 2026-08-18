"""Lightweight, dependency-free progress reporting for long rollout-collection
loops (no tqdm dependency in this repo).

One reporter, two rendering modes, sharing the same elapsed/rate bookkeeping:

- ``live=True`` -- a single in-place ``\\r``-updating bar with an ETA.  Only
  safe when this process is the SOLE writer to the terminal, e.g. the main
  process's own single-process rollout collection loop.  Multiple processes
  each emitting ``\\r`` updates to the same terminal stomp on each other.
- ``live=False`` -- coarse, newline-terminated milestone lines (every
  ``milestone_pct`` crossed).  Safe when several processes (parallel rollout
  workers) share the same inherited stdout: each line is self-contained and
  never overwrites another, only their relative interleaving order is
  unspecified.

Used by both ``debug_value_network.py``'s on-policy rollout collection and
``ai/ppo/ppo_trainer.py``'s ``_collect_value_pretrain_rollout()`` /
``ai/ppo/rollout_worker.py``'s parallel workers -- one implementation shared
across the single-process and parallel-worker rollout-collection paths in
both places, rather than four bespoke progress-printing snippets.
"""
from __future__ import annotations

import sys
import time
from typing import TextIO


class ProgressReporter:
    """Tracks elapsed time / rate for a ``0..total`` counter and renders it
    as either a live in-place bar or coarse milestone lines (see module
    docstring). Call ``update(current)`` as often as convenient -- both
    modes internally throttle how often they actually print."""

    def __init__(
        self,
        total: int,
        prefix: str = "",
        live: bool = True,
        milestone_pct: int = 10,
        min_interval_s: float = 0.2,
        bar_width: int = 30,
        stream: TextIO | None = None,
    ):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.milestone_pct = max(1, int(milestone_pct))
        self.min_interval_s = min_interval_s
        self.bar_width = bar_width
        self.stream = stream if stream is not None else sys.stderr
        # A \r-updating live bar only actually overwrites in place when the
        # destination is a real terminal. Piped/redirected output (e.g.
        # `| tee run.log`) has no concept of "in place" -- every \r-prefixed
        # write lands as its own line, blowing up the file with hundreds of
        # near-duplicate lines. Auto-downgrade to milestone mode whenever the
        # stream isn't a tty, regardless of what the caller asked for, so
        # `live=True` callers don't need to know or care whether they're
        # about to be piped.
        try:
            is_tty = self.stream.isatty()
        except Exception:
            is_tty = False
        self.live = bool(live) and is_tty
        self._start = time.monotonic()
        self._last_update_s = -1.0
        self._last_milestone = -1
        self._done = False

    def update(self, current: int) -> None:
        """Report progress at ``current`` (out of ``total``). Throttled
        internally -- safe to call every loop iteration. Always renders once
        ``current >= total`` (final call), even if throttled, so the
        terminal is left in a clean state (trailing newline for the live
        bar; a final 100% line for milestone mode)."""
        if self._done:
            return
        now = time.monotonic()
        finished = current >= self.total
        if self.live:
            if not finished and (now - self._last_update_s) < self.min_interval_s:
                return
            self._last_update_s = now
            self._render_live(current, now)
        else:
            pct = int(current * 100 / self.total)
            milestone = (pct // self.milestone_pct) * self.milestone_pct
            if not finished and milestone <= self._last_milestone:
                return
            self._last_milestone = milestone
            self._render_milestone(current, now)
        if finished:
            self._done = True

    def _render_live(self, current: int, now: float) -> None:
        elapsed = max(now - self._start, 1e-9)
        rate = current / elapsed
        frac = min(current / self.total, 1.0)
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        eta_s = (self.total - current) / rate if rate > 0 else float("inf")
        eta_str = f"{eta_s:5.0f}s" if eta_s < 3600 else "  >1h"
        end = "\n" if current >= self.total else ""
        print(
            f"\r{self.prefix}[{bar}] {current}/{self.total} ({frac * 100:5.1f}%)  "
            f"{rate:6.1f} steps/s  eta {eta_str}",
            end=end, file=self.stream, flush=True,
        )

    def _render_milestone(self, current: int, now: float) -> None:
        elapsed = max(now - self._start, 1e-9)
        rate = current / elapsed
        frac = min(current / self.total, 1.0)
        print(
            f"{self.prefix}{current}/{self.total} ({frac * 100:5.1f}%)  {rate:6.1f} steps/s",
            file=self.stream, flush=True,
        )

    def finish(self, current: int, n_episodes: int | None = None) -> None:
        """Print a one-line summary -- total elapsed time, time/step, and
        (when ``n_episodes`` is given) time/episode -- once collection is
        actually done. Separate from the automatic 100%-reached rendering
        in ``update()`` (which only knows about steps): episode count is
        usually only known to the caller once its collection loop has fully
        exited, so this is called explicitly, once, after that loop ends."""
        elapsed = max(time.monotonic() - self._start, 1e-9)
        per_step_ms = 1000.0 * elapsed / max(current, 1)
        msg = f"{self.prefix}done: {elapsed:.1f}s total  ({per_step_ms:.2f} ms/step"
        if n_episodes:
            msg += f", {elapsed / n_episodes:.2f} s/episode over {n_episodes} episode(s)"
        msg += ")"
        print(msg, file=self.stream, flush=True)
