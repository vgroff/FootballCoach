"""`App._ticks_this_frame` / `App._step_match`: how many physics ticks run per
rendered frame for a given `sim_speed`. See `ui/knowledge.md` "Sim speed vs.
actual tick rate" for the full writeup.

User-reported: "the simulation speed shown in game doesn't reflect that set
in the menu." Root cause (verified, not guessed): the old computation --
`steps = max(1, round(physics_tick_hz * sim_speed / target_fps))`, freshly
computed each frame with no memory of past frames -- only tracks `sim_speed`
correctly when `physics_tick_hz * sim_speed / target_fps` happens to be (near)
an integer. With `physics_tick_hz == target_fps` (both 60 by default), that
collapses to `round(sim_speed)`: every non-integer `sim_speed` silently
snapped to the nearest whole multiple of real-time, and the `max(1, ...)`
floor made anything below 1.0x (real-time) unreachable, even though the menu
slider and in-game HUD control both offer sub-1.0 steps meant to allow slow
motion.

Fixed with a fixed-timestep accumulator (`self._physics_acc_s`, previously
initialised in three places but never actually read by the stepping code --
dead state left over from an evidently-intended-but-never-wired accumulator
design): real elapsed time, scaled by `sim_speed`, accumulates and whole
ticks are drained off it, so the LONG-RUN average tick rate converges to
exactly `physics_tick_hz * sim_speed`, for any `sim_speed`, not just
grid-aligned ones.

Constructs a bare `App` via `object.__new__` (as in
`test_app_kick_tackle_wiring.py`) since `_ticks_this_frame` only touches
`_physics_tick_hz` / `_sim_speed` / `_physics_acc_s`.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from footballcoach.ui.app import App

pygame.init()


def _app(physics_tick_hz: float = 60.0, sim_speed: float = 1.0) -> App:
    app = object.__new__(App)
    app._physics_tick_hz = physics_tick_hz
    app._sim_speed = sim_speed
    app._physics_acc_s = 0.0
    return app


def _run_frames(app: App, dt_s: float, n_frames: int) -> list[int]:
    return [app._ticks_this_frame(dt_s) for _ in range(n_frames)]


@pytest.mark.parametrize("sim_speed", [0.1, 0.25, 0.5, 0.75, 1.0, 1.6, 2.0, 2.6, 3.0, 8.0, 16.0])
def test_long_run_tick_rate_matches_sim_speed_exactly(sim_speed):
    """The core regression test: over many frames, total ticks must converge
    to physics_tick_hz * sim_speed * elapsed_real_seconds -- for EVERY value
    here, not just the ones that happen to be integers. Before the fix, every
    one of these except exactly-integer values produced the same flat 2.0x
    (with physics_tick_hz == target_fps == 60, the pre-fix formula reduces to
    round(sim_speed), and 0.1 through 2.9 all round to a step count that
    nets out at 2.0x once clamped/rounded -- see the module docstring)."""
    app = _app(physics_tick_hz=60.0, sim_speed=sim_speed)
    dt_s = 1.0 / 60.0
    n_frames = 1200  # 20 real seconds -- long enough to average out rounding
    total_ticks = sum(_run_frames(app, dt_s, n_frames))
    elapsed_s = n_frames * dt_s
    expected = 60.0 * sim_speed * elapsed_s
    # Worst-case error is well under 1 whole tick regardless of run length.
    assert total_ticks == pytest.approx(expected, abs=1.0)


def test_sub_one_speed_actually_produces_zero_tick_frames():
    """The old `max(1, ...)` floor made real slow-motion (< 1.0x) impossible
    -- every frame ran at least one tick, i.e. at least real-time. The fixed
    version must actually skip ticks on most frames at sim_speed < 1.0."""
    app = _app(physics_tick_hz=60.0, sim_speed=0.25)
    ticks = _run_frames(app, 1.0 / 60.0, 240)
    assert any(t == 0 for t in ticks)
    assert sum(ticks) < len(ticks)  # strictly slower than real-time overall


def test_default_sim_speed_of_two_is_unaffected():
    """default_sim_speed=2.0 (gameplay.json) is an exact integer multiple at
    the default physics_tick_hz==target_fps==60, so it already worked before
    -- this is why the bug went unnoticed until someone picked a fractional
    value. Guard against a regression on the one value everyone already
    relied on."""
    app = _app(physics_tick_hz=60.0, sim_speed=2.0)
    ticks = _run_frames(app, 1.0 / 60.0, 120)
    assert all(t == 2 for t in ticks)  # steady state: exactly 2 ticks/frame


def test_accumulator_persists_leftover_fraction_across_frames():
    """At sim_speed=1.5 with dt_s exactly 1/60s (physics_tick_hz=60), each
    frame owes 0.025 ticks -- alternating 1,2,1,2,... ticks/frame is the
    correct way to average 1.5, not silently rounding every frame to the
    same number."""
    app = _app(physics_tick_hz=60.0, sim_speed=1.5)
    ticks = _run_frames(app, 1.0 / 60.0, 10)
    assert set(ticks) == {1, 2}  # never a flat, wrong average like all-1 or all-2
    assert sum(ticks) == pytest.approx(15, abs=1)


def test_stalled_frame_does_not_queue_a_huge_catch_up_burst():
    """A single pathologically long frame (window drag, OS hiccup) must not
    produce a huge burst of physics ticks on the next call -- dt_s is
    clamped before being scaled into the accumulator."""
    app = _app(physics_tick_hz=60.0, sim_speed=1.0)
    ticks = app._ticks_this_frame(5.0)  # simulate a 5-second stall
    assert ticks <= round(60.0 * 0.25) + 1  # clamped to ~0.25s worth, not 5s worth


def test_paused_match_never_advances_the_accumulator():
    """`_step_match` returns before calling `_ticks_this_frame` while
    `match.paused`, so no sim-time debt builds up during a pause (verified
    via the real early-return in `_step_match`, not just this unit)."""
    class _FakeMatch:
        paused = True

    app = _app(physics_tick_hz=60.0, sim_speed=4.0)
    app.match = _FakeMatch()
    App._step_match(app, 1.0)  # a whole second, paused
    assert app._physics_acc_s == 0.0
