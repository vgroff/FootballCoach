"""Round-trip equivalence test for the Orders <-> execution-network boundary.

**Documentation must stay in sync with code.** Any significant change, and
any change that conflicts with existing documentation, must be followed by
additions or edits to the relevant documentation (this file, knowledge.md
files, design docs, plans). Otherwise documentation goes stale and confusion
occurs.

What this test does
--------------------
Runs `Phase1RulesAI` vs. an immobile opponent (`build_1v1_scenario`'s
default) for a whole episode ("real" match), capturing at EVERY physics tick
(not just decision ticks) the raw execution-level fields the order machinery
actually produced on the trainee: `desired_direction`, `desired_speed_mode`,
`kicked_this_tick`/`last_kick_*`, and `tackle_armed`/order-type for
tackle_attempt. This mirrors exactly the field set `ai/knowledge.md`'s
"Orders vs execution-network labels boundary" section mandates as the only
legitimate source for execution-network labels.

Those captured per-tick records are then replayed tick-for-tick through the
REAL inference-time decode function, `apply_action_to_player()`
(`ai/action/apply_nn_action.py`) -- the exact function `NeuralPlayerAI` calls
at inference -- on a second ("shadow") `Match` built from the identical seed.
No Order is ever assigned to the shadow trainee, matching "the neural network
never issues Orders."

If encode (capture) -> decode (`apply_action_to_player`) is faithful, the two
matches must produce bit-identical physics every tick: same positions,
velocities, headings, ball state, and possession.

Known, accepted sources of intentional divergence (not bugs)
--------------------------------------------------------------
- Kick spin: `apply_action_to_player()` hardcodes kick spin to zero (spin is
  disabled for the neural network for now). `Phase1RulesAI` never imparts
  spin either, so this is a non-issue for this specific rules AI -- flagged
  here so it isn't silently assumed to generalise to other label sources.

Kick capture: pre-noise angle, not post-noise direction
---------------------------------------------------------
`Player.kick_direct()` (rules AI path) and `Player.kick_with_direction()`
(decode path) both add ONE independent Gaussian yaw/pitch perturbation
around a "pre-noise" base angle (`kicking._launch_ball` /
`kick_ball_from_direction`, both drawing exactly two `rng.gauss()` calls with
the same `kick_sigma_rad()` formula) -- so with both matches' RNGs seeded and
advanced identically, feeding the SAME pre-noise base angle into both sides
reproduces the SAME actual post-noise draw bit-for-bit.

Naively capturing `Player.last_kick_direction` (the ACTUAL post-noise
direction) instead would NOT round-trip: `kick_ball`'s `running_power_multiplier`
("running onto the ball" pace boost/penalty) is computed from the pre-noise
aim direction and used to pre-divide `power_fraction`
(`compensate_power_for_run_mult`) so the kick delivers the *intended* power
regardless of run direction -- this cancels out exactly on the real side
(same aim direction fed to the division and to `kick_ball`'s own internal
recomputation). But `kick_ball_from_direction` recomputes the SAME multiplier
from whatever direction it's handed; feeding it the noisy actual direction
(a few degrees off the pre-noise aim line) makes that recomputed multiplier
disagree with the one baked into the captured `power_fraction`, so the
compensation no longer cancels -- a small per-kick speed/direction error that
compounds across dozens of push-kicks during a sustained dribble. Capturing
the PRE-noise angle instead sidesteps this: both sides then compute the
identical `run_mult`, `sigma`, and (via the synced RNG) identical noise draw,
so the decoded kick's actual outgoing velocity matches the real kick's exactly
-- with the real game's kicking config completely untouched.

We capture this via a hook on `kicking._launch_ball` (the shared
aim-then-launch step used by both `kick_ball` and `pass_ball`), reading back
the pre-noise `yaw`/`pitch` it computes right before adding noise -- the same
two lines it uses internally, not re-derived from scratch.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

import footballcoach.engine.kicking as kicking_mod
from footballcoach.ai.action.apply_nn_action import apply_action_to_player
from footballcoach.ai.action.gating import GatingResult, SelectedAction
from footballcoach.engine.movement import SpeedMode
from footballcoach.entities.player import PlayerAI, Team
from footballcoach.orders import ChaseTackleOrder, GetPossessionOrder
from footballcoach.rules_ai import Phase1RulesAI
from footballcoach.ui.scenarios import build_1v1_scenario

TRAINEE_ID = "trainee"
OPPONENT_ID = "opponent"
EPISODE_TICKS = 400  # ~13s at 30Hz -- long enough to see several kicks/moves


def _install_movement_capture_hook(match, player_id: str) -> dict:
    """`Match._apply_movement()` consumes `desired_direction`/`desired_speed_mode`
    and resets the latter to `None` at the end of the SAME tick (see
    `match.py::_apply_movement` docstring: "desired_speed_mode is cleared to
    None after application so each tick is independent"). Reading those
    fields off the player AFTER `match.step()` returns therefore always sees
    `None` -- we must snapshot them at the point they're actually consumed,
    inside `_apply_movement`, before the reset. Returns a mutable dict that
    holds the latest snapshot after each `match.step()` call."""
    captured: dict = {}
    original = type(match)._apply_movement

    def _patched(self, dt):
        p = self.player_by_id(player_id)
        captured["direction"] = p.desired_direction
        captured["speed_mode"] = p.desired_speed_mode
        return original(self, dt)

    match._apply_movement = types.MethodType(_patched, match)
    return captured


def _install_kick_angle_capture_hook() -> dict:
    """Monkeypatch both real kick-physics entry points to record the
    PRE-noise `yaw`/`pitch` each computes right before perturbing them --
    see module docstring "Kick capture: pre-noise angle, not post-noise
    direction". Mirrors each function's own yaw/pitch lines exactly (reusing
    `solve_launch_pitch_rad` for the ballistic pitch solve, not re-deriving
    it) so this can't drift from the real formula. MUST be uninstalled via
    the returned restore function -- both are global module patches.

    Two entry points, because push-kicks (both rules-AI, via orders.py's
    _try_push_kick, and the NN) are flat direction-only kicks -- no
    ballistic aim-point solve at all -- and go through
    `kick_ball_from_direction`, NOT `kick_ball`/`_launch_ball`. Only
    KickOrder/ShootOrder/PassOrder-style explicit-aim-point kicks go through
    `_launch_ball`. Phase1RulesAI's box-run push-kicks are exactly the kicks
    this test exercises, so both paths must be captured or `kick_angle`
    silently stays empty on every push-kick tick.
    """
    captured: dict = {}
    original_launch = kicking_mod._launch_ball
    original_from_direction = kicking_mod.kick_ball_from_direction

    def _patched_launch(ball, kicker_position, aim_point, speed, sigma, spin, rng, gravity_mps2):
        launch_position = kicker_position.with_z(ball.position.z)
        delta = aim_point - launch_position
        horizontal_distance = delta.xy().length()
        yaw = delta.xy().angle_xy() if horizontal_distance > 1e-9 else 0.0
        pitch = kicking_mod.solve_launch_pitch_rad(horizontal_distance, delta.z, speed, gravity_mps2)
        captured["yaw"] = yaw
        captured["pitch"] = pitch
        # State BEFORE the two rng.gauss() noise draws below -- see
        # "rng_state_before_kick_noise" note in _capture_tick_record. Even
        # with the per-tick resync in the main test loop, a decision-layer
        # draw earlier in THIS SAME tick (e.g. Phase1RulesAI picking a new
        # box-run target) can still offset real's kick-noise draws further
        # down the stream than the shadow's would be at kick time -- this
        # captures the exact starting point so the shadow can be resynced
        # surgically right before its own two draws.
        captured["rng_state_before_noise"] = rng.getstate()
        return original_launch(ball, kicker_position, aim_point, speed, sigma, spin, rng, gravity_mps2)

    def _patched_from_direction(ball, kicker_position, direction_unit3, *args, **kwargs):
        import math as _math
        import random as _random
        horiz_len = direction_unit3.xy().length()
        captured["yaw"] = direction_unit3.xy().angle_xy() if horiz_len > 1e-9 else 0.0
        captured["pitch"] = _math.atan2(direction_unit3.z, max(horiz_len, 1e-9))
        # rng is the 6th positional arg (index 5 of *args) or the "rng" kwarg.
        rng = kwargs.get("rng", args[5] if len(args) > 5 else None) or _random
        captured["rng_state_before_noise"] = rng.getstate()
        return original_from_direction(ball, kicker_position, direction_unit3, *args, **kwargs)

    kicking_mod._launch_ball = _patched_launch
    kicking_mod.kick_ball_from_direction = _patched_from_direction

    def _restore():
        kicking_mod._launch_ball = original_launch
        kicking_mod.kick_ball_from_direction = original_from_direction

    captured["_restore"] = _restore
    return captured


def _force_fresh_movement_decision(player, match, decision_ai, *, is_gap_tick: bool):
    """Mirror `bc.py::phase1_labels()`'s snapshot/force-decision/restore
    pattern for ticks where the live order machinery has no current order in
    effect (a transient gap between one order completing and the AI's next
    decision -- e.g. right after a push-kick releases the ball, before
    Phase1RulesAI decides the next GetPossessionOrder). `Match._apply_movement`
    now correctly coasts the player under their existing velocity during this
    gap (see `match.py`'s inertial-coasting fix).

    `decision_ai` must be a PERSISTENT `Phase1RulesAI` instance (constructed
    once, outside the capture loop, with the SAME decision_interval_ticks as
    the real trainee's own AI -- see this test's own call site) rather than
    a fresh one per call: Phase1RulesAI's decision cadence is stateful
    (`_ticks_since_decision`), and this function is called once per REAL
    match tick -- EVERY tick, not just gap ticks, see the call site -- in
    lockstep with the real trainee AI's own `.act()` calls, so a
    `decision_ai` ticked the same way stays in the exact same decision-phase
    as the real one for the whole episode.

    `is_gap_tick` gates whether the decided order's `execute()` actually
    runs: `decide()` itself is pure/deterministic (no rng draw -- see
    rules_ai.py's own comments), but `execute()` can trigger a REAL push-
    kick, which draws real kick-angle noise from `match.rng`. Only the
    ACTUAL gap-tick call needs that result; calling `execute()` on every
    decision tick regardless (an earlier, wrong version of this function)
    drew EXTRA, spurious rng noise on every ordinary decision tick too --
    confirmed via direct instrumentation to shift ball_vel by a measurable
    amount and break the real/shadow equivalence this test checks, even
    though decide()'s own phase-locked cadence was already correct by that
    point. So: always tick decision_ai (keeps the counter phase-locked),
    only ever execute()/consume rng when the result is actually needed.
    """
    from footballcoach.orders import GetPossessionOrder, MoveOrder

    snap_pos = player.position
    snap_vel = player.velocity
    snap_heading = player.heading_rad
    snap_desired_dir = player.desired_direction
    snap_desired_speed = player.desired_speed_mode
    snap_current_order = player.current_order
    snap_kicked_this_tick = player.kicked_this_tick
    snap_last_kick_dir = player.last_kick_direction
    snap_last_kick_power = player.last_kick_power_fraction
    snap_last_kick_spin = player.last_kick_spin
    snap_on_kick = player.on_kick
    snap_on_tackle = player.on_tackle
    snap_ball_possessed_by = match.ball.possessed_by
    snap_ball_velocity = match.ball.velocity
    snap_ball_position = match.ball.position

    player.current_order = None
    player.on_kick = None
    player.on_tackle = None
    exec_move, sprint, move_direction = False, False, None
    try:
        decision_ai.act(player, match, trial_tick=0)
        order = player.current_order
        if is_gap_tick and isinstance(order, (MoveOrder, GetPossessionOrder)):
            order.execute(player, match, match.dt_s)
            if player.desired_speed_mode is not None:
                exec_move = player.desired_speed_mode is not SpeedMode.STANDSTILL
                sprint = player.desired_speed_mode is SpeedMode.SPRINT
                d = player.desired_direction
                if d.length_xy() > 1e-6:
                    move_direction = np.array([d.x, d.y], dtype=np.float64)
    finally:
        player.position = snap_pos
        player.velocity = snap_vel
        player.heading_rad = snap_heading
        player.desired_direction = snap_desired_dir
        player.desired_speed_mode = snap_desired_speed
        player.current_order = snap_current_order
        player.kicked_this_tick = snap_kicked_this_tick
        player.last_kick_direction = snap_last_kick_dir
        player.last_kick_power_fraction = snap_last_kick_power
        player.last_kick_spin = snap_last_kick_spin
        player.on_kick = snap_on_kick
        player.on_tackle = snap_on_tackle
        match.ball.possessed_by = snap_ball_possessed_by
        match.ball.velocity = snap_ball_velocity
        match.ball.position = snap_ball_position
    return exec_move, sprint, move_direction


def _capture_tick_record(
    player, match, movement_snapshot: dict, kick_angle: dict, decision_ai,
) -> dict:
    """Read back the raw execution-level fields the order machinery produced
    on `player` THIS tick -- the same field set (and the same rule: never
    hand-derive from Order fields) `ai/knowledge.md`'s BC-label corollary
    mandates. Movement fields come from `movement_snapshot` (see
    `_install_movement_capture_hook`); kick/tackle fields are read directly
    off `player` since those are NOT reset until the next tick's
    `_process_orders()`. `decision_ai`: see _force_fresh_movement_decision's
    own docstring -- must be the SAME persistent, phase-synchronized
    instance across the whole capture loop."""
    speed_mode = movement_snapshot["speed_mode"]
    # Tick decision_ai EVERY call, unconditionally -- not just on gap ticks
    # -- so its internal decision-cadence counter advances once per real
    # match tick, exactly like the real trainee AI's own .act() (called
    # every tick by Match._process_orders regardless of gap/no-gap). Only
    # gating this call on speed_mode is None would mean decision_ai simply
    # never gets ticked on ordinary (non-gap) ticks at all, letting its
    # phase drift arbitrarily far from the real one -- confirmed via direct
    # instrumentation: with that (wrong, earlier) version, decision_ai's
    # counter stayed frozen for ~10 real ticks in a row between gaps
    # instead of advancing 1 per tick like the real one does.
    forced_exec_move, forced_sprint, forced_move_direction = _force_fresh_movement_decision(
        player, match, decision_ai, is_gap_tick=speed_mode is None,
    )
    if speed_mode is None:
        # Gap tick -- no order in effect. Use the synthetic decision just
        # computed above (see _force_fresh_movement_decision's own
        # docstring for why a phase-synchronized decision_ai makes this
        # correctly match the real match's own coast-vs-decide cadence).
        exec_move, sprint, move_direction = forced_exec_move, forced_sprint, forced_move_direction
    else:
        exec_move = speed_mode != SpeedMode.STANDSTILL
        sprint = speed_mode == SpeedMode.SPRINT
        # Direction is captured whenever _apply_movement processed this player
        # at all this tick -- including SpeedMode.STANDSTILL. step_player_towards()
        # turns a STANDSTILL player to face a nonzero target_direction while
        # decelerating (turn-in-place), e.g. the rules AI settling into
        # first-touch ball control; apply_action_to_player() now applies
        # move_direction regardless of exec_move to reproduce that (see its
        # docstring) -- capturing move_direction=None here whenever exec_move
        # was False would silently drop that turn-in-place intent.
        d = movement_snapshot["direction"]
        # float64, NOT the float32 GatingResult carries in production. There
        # a real trained (PyTorch, float32) network is the source, and that
        # precision loss is real and expected. Here we hand-construct
        # GatingResult straight from captured ground truth with no network
        # involved -- float32 would only be an artificial, self-inflicted
        # rounding error unrelated to what this test checks, compounding
        # over many ticks of turning into a spurious heading/position drift.
        move_direction = np.array([d.x, d.y], dtype=np.float64)

    kick_this_tick = bool(player.kicked_this_tick)
    kick_direction = None
    kick_power_fraction = 0.0
    rng_state_before_kick_noise = None
    if kick_this_tick and player.last_kick_direction is not None:
        # PRE-noise ballistic angle -- see module docstring. NOT
        # player.last_kick_direction (that's the post-noise actual result).
        yaw, pitch = kick_angle["yaw"], kick_angle["pitch"]
        kick_direction = np.array(
            [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)],
            dtype=np.float64,
        )
        kick_power_fraction = float(player.last_kick_power_fraction or 0.0)
        rng_state_before_kick_noise = kick_angle["rng_state_before_noise"]
        # Rules AI never imparts spin -- see module docstring.
        assert player.last_kick_spin is None or (
            abs(player.last_kick_spin.x) < 1e-9
            and abs(player.last_kick_spin.y) < 1e-9
            and abs(player.last_kick_spin.z) < 1e-9
        ), "Phase1RulesAI produced nonzero kick spin -- test assumption violated"

    is_gp_tackling = isinstance(player.current_order, GetPossessionOrder) and player.tackle_armed
    tackle_attempt = isinstance(player.current_order, ChaseTackleOrder) or is_gp_tackling

    return dict(
        exec_move=exec_move,
        sprint=sprint,
        move_direction=move_direction,
        kick_this_tick=kick_this_tick,
        kick_direction=kick_direction,
        kick_power_fraction=kick_power_fraction,
        tackle_attempt=tackle_attempt,
        rng_state_before_kick_noise=rng_state_before_kick_noise,
    )


class _ReplayAI(PlayerAI):
    """Feeds pre-captured per-tick records through the REAL decode function
    (`apply_action_to_player`) -- never issues an Order, matching the
    neural-network inference contract exactly."""

    def __init__(self, records: list[dict]):
        self._records = records
        self._i = 0

    def act(self, player, match, trial_tick: int) -> None:
        rec = self._records[self._i]
        self._i += 1
        if rec["kick_this_tick"] and rec["rng_state_before_kick_noise"] is not None:
            # Surgical resync right before the kick-noise draws -- see
            # _install_kick_angle_capture_hook. The per-tick resync in the
            # main test loop handles cross-tick drift, but a decision-layer
            # draw earlier in THIS tick (that _ReplayAI never makes, since it
            # never runs Phase1RulesAI) can still leave real's kick-noise
            # draws further down the stream than a tick-start resync alone
            # accounts for.
            match.rng.setstate(rec["rng_state_before_kick_noise"])
        gating = GatingResult(
            selected=SelectedAction.NONE,
            exec_move=rec["exec_move"],
            move_direction=rec["move_direction"],
            sprint=rec["sprint"],
            kick_this_tick=rec["kick_this_tick"],
            kick_direction=rec["kick_direction"],
            kick_power_fraction=rec["kick_power_fraction"],
            kick_spin=np.zeros(3, dtype=np.float64),
            tackle_attempt=rec["tackle_attempt"],
        )
        apply_action_to_player(gating, player, match, slot_player_ids=[], decision_physical={})


def _snapshot(match) -> dict:
    ball = match.ball
    out = {
        "ball_pos": (ball.position.x, ball.position.y, ball.position.z),
        "ball_vel": (ball.velocity.x, ball.velocity.y, ball.velocity.z),
        "ball_possessed_by": ball.possessed_by,
    }
    for p in match.players:
        out[f"{p.player_id}_pos"] = (p.position.x, p.position.y, p.position.z)
        out[f"{p.player_id}_vel"] = (p.velocity.x, p.velocity.y, p.velocity.z)
        out[f"{p.player_id}_heading"] = p.heading_rad
        out[f"{p.player_id}_stamina"] = p.stamina
    return out


def _assert_snapshots_close(a: dict, b: dict, tick: int) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, tuple):
            for i, (xa, xb) in enumerate(zip(va, vb)):
                assert xa == pytest.approx(xb, abs=1e-6), (
                    f"tick={tick} key={key}[{i}] real={xa} shadow={xb}"
                )
        elif isinstance(va, float):
            assert va == pytest.approx(vb, abs=1e-6), f"tick={tick} key={key} real={va} shadow={vb}"
        else:
            assert va == vb, f"tick={tick} key={key} real={va} shadow={vb}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rules_ai_replay_through_apply_nn_action_is_identical(seed: int):
    real_match = build_1v1_scenario(seed=seed, trainee_team=Team.LEFT)
    real_trainee_ai = Phase1RulesAI()
    real_match.player_by_id(TRAINEE_ID).ai = real_trainee_ai
    # Opponent is immobile by default (opponent_immobile_prob=1.0) -- ai=None.
    # Both matches use the real, untouched default kicking_params/physics
    # config -- see module docstring for why no config needs zeroing.

    # Persistent, phase-synchronized shadow decision AI for gap ticks -- see
    # _force_fresh_movement_decision's own docstring. Same decision_interval_
    # ticks as the real trainee AI (read off it directly, not re-derived, so
    # this can never drift out of sync with whatever the real default is),
    # ticked exactly once per real match tick below, same as the real one.
    shadow_decision_ai = Phase1RulesAI(decision_interval_ticks=real_trainee_ai.decision_interval_ticks)

    movement_snapshot = _install_movement_capture_hook(real_match, TRAINEE_ID)
    kick_angle = _install_kick_angle_capture_hook()

    records: list[dict] = []
    snapshots: list[dict] = []
    rng_states: list[tuple] = [real_match.rng.getstate()]  # state BEFORE tick 0
    try:
        for _ in range(EPISODE_TICKS):
            real_match.step()
            records.append(
                _capture_tick_record(
                    real_match.player_by_id(TRAINEE_ID), real_match, movement_snapshot, kick_angle,
                    shadow_decision_ai,
                )
            )
            snapshots.append(_snapshot(real_match))
            rng_states.append(real_match.rng.getstate())
    finally:
        kick_angle["_restore"]()

    shadow_match = build_1v1_scenario(seed=seed, trainee_team=Team.LEFT)
    shadow_match.player_by_id(TRAINEE_ID).ai = _ReplayAI(records)

    # Sync shadow_match.rng to real_match's post-tick state each tick, rather
    # than relying on seed-matching to keep the two matches' RNG draw counts
    # aligned. Phase1RulesAI's OWN decision logic (e.g. rules_ai.py's
    # match.rng.uniform(...) box-run target pick, made once right after
    # gaining possession) consumes draws from match.rng as a side effect of
    # DECIDING what to do -- draws the shadow's _ReplayAI never makes, since
    # it replays captured motor output only and never runs any rules-AI
    # decision logic. That one-draw offset would otherwise permanently shift
    # the shadow's RNG stream relative to the real match's from that tick
    # onward, corrupting every later rng-dependent physics event (kick
    # accuracy noise, tackle rolls, ...) even though the inputs feeding them
    # (speed, sigma, direction, ...) are otherwise bit-identical. This isn't
    # an apply_action_to_player() bug -- it's inherent to replaying motor
    # output while skipping the decision logic that happens to also draw
    # randomness -- so we sync the shared resource directly instead.
    for tick in range(EPISODE_TICKS):
        shadow_match.rng.setstate(rng_states[tick])  # state BEFORE real_match's tick `tick`
        shadow_match.step()
        _assert_snapshots_close(snapshots[tick], _snapshot(shadow_match), tick)
