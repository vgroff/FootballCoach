"""How much *simulation* time has passed between two rendered frames.

Purely cosmetic animations (a player's stride, the ball's spin) must advance with
the match's own clock, not the wall clock and not ``1 / target_fps``: the sim runs
``sim_speed`` match-seconds per real second (and steps a whole number of physics
ticks per frame), so a fixed per-frame step made strides and spin play at
``1 / sim_speed`` of their true rate whenever the sim was sped up. Driving them from
``match.time_s`` is right whatever the speed, the frame rate, the step rounding, or a
pause.
"""
from __future__ import annotations


class SimTimeDelta:
    """Turns a match's ever-growing ``time_s`` into per-frame deltas.

    ``delta(owner, sim_time_s)`` returns the sim seconds elapsed since the previous call
    for the same ``owner`` (the match object). It returns 0.0 -- and resyncs -- on the
    first call, when ``owner`` is a different object (a new match or scenario trial
    starts its clock at 0), or if the clock ran backwards, so an animation never jumps
    by a bogus amount across such a boundary."""

    def __init__(self) -> None:
        self._owner: object | None = None
        self._time_s: float = 0.0

    def delta(self, owner: object, sim_time_s: float) -> float:
        elapsed = 0.0
        if owner is self._owner and sim_time_s >= self._time_s:
            elapsed = sim_time_s - self._time_s
        self._owner, self._time_s = owner, sim_time_s
        return elapsed
