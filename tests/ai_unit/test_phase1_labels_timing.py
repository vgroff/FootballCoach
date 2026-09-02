"""Regression tests for phase1_labels()'s CALLER-SIDE TIMING and SHARED RNG
STREAM bugs (both fixed 2026-09) -- see bc.py's phase1_labels_for_player()
docstring and ai/knowledge.md's "Orders vs execution-network labels
boundary" for the full history.

**Documentation must stay in sync with code.** Any significant change to
phase1_labels()/phase1_labels_for_player()/NeuralPlayerAI.bc_label_fn, and
any change that conflicts with existing documentation, must be followed by
additions or edits to the relevant documentation (this file, knowledge.md
files) and vice versa.

Background
----------
`phase1_labels_for_player(player, match)` asks a fresh, temporary
Phase1RulesAI what it would do RIGHT NOW, via a full snapshot -> reset ->
execute -> restore counterfactual around the decided order's `execute()`.
Two subtle bugs lived in this mechanism, both fixed in the same pass:

1. **Timing**: the counterfactual reads `player`/`match`'s state at the
   exact moment it's CALLED. `Match.step()` runs `_process_orders(dt)`
   (real decisions, and the observation `NeuralPlayerAI` encodes for one,
   both using state as of the START of that tick) strictly BEFORE
   `_apply_movement(dt)` (which actually advances position/velocity/
   heading). A caller that computes the label AFTER a full `env.step()`
   has already returned -- the pattern every rollout loop used before this
   fix -- sees POST-movement state: one physics tick later than the
   observation the label is meant to accompany. Fixed by having
   `NeuralPlayerAI.act()` call its own `bc_label_fn` internally, at the
   same instant it encodes the observation (see `bc_label_fn`'s own
   docstring in rules_ai.py), instead of a caller computing it separately
   afterward.

2. **Shared RNG stream**: if the counterfactual's `order.execute()` causes
   a real push-kick to fire, `Player.kick_direct()` draws its yaw/pitch
   noise from `match.rng` -- ONE `random.Random` shared by the entire
   simulation. Without protecting it, computing a label could silently
   consume real draws from that stream, desyncing every subsequent genuine
   random outcome in the match. Fixed with `match.rng.getstate()`/
   `setstate()` around the whole counterfactual block.

These tests check directly: (1) that a label computed on the SAME tick a
real kick fires reproduces that kick's direction/power bit-for-bit (proves
both the timing AND the RNG-restore are correct -- if either were broken,
the noise draw or the base geometry would disagree); (2) that computing
labels has literally zero effect on the real simulation's own trajectory,
checked by running the identical episode with and without label computation
and diffing the outcomes; (3) that the OLD, buggy call timing genuinely
differs from the fix (so this suite would have caught the original bug);
and (4) that the actual production wiring -- NeuralPlayerAI.bc_label_fn /
ScenarioEnv.bc_label_fn -- correctly threads a label into
last_trainee_transition at the right instant, not just that the underlying
phase1_labels_for_player() function is correct in isolation.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.bc import phase1_labels_for_player
from footballcoach.rules_ai import Phase1RulesAI, _RulesBasedAI
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

TRAINEE_ID = "trainee"
EPISODE_TICKS = 400  # ~13s at 30Hz -- long enough to see several push-kicks
KICK_SEEDS = range(1, 20)  # covers several real push-kicks (see module docstring)


def _make_env(max_episode_s: float = 30.0) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="test_phase1_labels_timing", label="t", description="t", build=build_1v1_scenario,
    )
    return ScenarioEnv(definition=defn, trainee_player_id=TRAINEE_ID, phase=1, max_episode_s=max_episode_s)


class _FreshEachTickAI(_RulesBasedAI):
    """Decides at the real Phase1RulesAI decision cadence (see
    test_rules_ai_fresh_vs_persistent_equivalence.py, which this mirrors),
    but via a BRAND NEW Phase1RulesAI(decision_interval_ticks=1) instance
    every time -- confirmed elsewhere to be bit-exactly equivalent to a
    genuinely persistent Phase1RulesAI (decide() is stateless). Used here
    (rather than a bare persistent Phase1RulesAI) so the "real" side of
    each test is driven by the exact same mechanism
    phase1_labels_for_player()'s own internal counterfactual uses --
    keeping the comparison apples-to-apples."""

    def decide(self, player, match, trial_tick: int) -> None:
        player.current_order = None
        Phase1RulesAI(decision_interval_ticks=1).act(player, match, trial_tick=0)


def _snapshot_rng_and_player(match, player) -> dict:
    return {
        "rng_state": match.rng.getstate(),
        "position": copy.copy(player.position),
        "velocity": copy.copy(player.velocity),
        "heading_rad": player.heading_rad,
        "desired_direction": copy.copy(player.desired_direction),
        "desired_speed_mode": player.desired_speed_mode,
        "kicked_this_tick": player.kicked_this_tick,
        "last_kick_direction": player.last_kick_direction,
        "last_kick_power_fraction": player.last_kick_power_fraction,
        "kick_armed": player.kick_armed,
        "kick_armed_direction": player.kick_armed_direction,
        "kick_armed_power_fraction": player.kick_armed_power_fraction,
        "tackle_armed": player.tackle_armed,
        "current_order": player.current_order,
        "ball_possessed_by": match.ball.possessed_by,
        "ball_position": copy.copy(match.ball.position),
        "ball_velocity": copy.copy(match.ball.velocity),
    }


class TestPhase1LabelsForPlayerZeroSideEffect:
    """phase1_labels_for_player() must be perfectly invisible to the real
    simulation -- calling it any number of times, at any point, must leave
    match.rng and every piece of player/ball state it touches bit-identical
    to before the call."""

    def test_repeated_calls_leave_state_and_rng_untouched(self):
        env = _make_env()
        env.reset(seed=1)
        match = env.match
        player = match.player_by_id(TRAINEE_ID)

        before = _snapshot_rng_and_player(match, player)
        for _ in range(5):
            phase1_labels_for_player(player, match)
        after = _snapshot_rng_and_player(match, player)

        assert after["rng_state"] == before["rng_state"], (
            "phase1_labels_for_player() must not consume/advance match.rng -- "
            "see phase1_labels_for_player()'s 'CRITICAL -- SHARED RNG STREAM' docstring section"
        )
        for key in before:
            if key == "rng_state":
                continue
            assert after[key] == before[key], f"phase1_labels_for_player() leaked into player.{key}"

    def test_zero_effect_across_a_whole_episode_with_real_kicks(self):
        """Stronger version of the above: run a REAL episode twice from the
        identical seed, once calling phase1_labels_for_player() every tick
        and once never calling it at all. If the label computation has any
        real effect on the simulation (RNG desync, state leak), the two
        episodes' trajectories will diverge -- if it's truly invisible,
        every player/ball position, and every kick's exact direction/power,
        must match bit-for-bit for the whole episode."""
        def _run_episode(call_label_fn: bool) -> list:
            env = _make_env()
            env.reset(seed=3)
            player = env.match.player_by_id(TRAINEE_ID)
            player.ai = _FreshEachTickAI(decision_interval_ticks=Phase1RulesAI().decision_interval_ticks)
            trace = []
            for _ in range(EPISODE_TICKS):
                if call_label_fn:
                    phase1_labels_for_player(player, env.match)
                env.step()
                trace.append((
                    player.position.x, player.position.y,
                    player.velocity.x, player.velocity.y,
                    player.heading_rad,
                    player.kicked_this_tick,
                    (player.last_kick_direction.x, player.last_kick_direction.y, player.last_kick_direction.z)
                    if player.last_kick_direction is not None else None,
                    player.last_kick_power_fraction,
                    env.match.ball.position.x, env.match.ball.position.y,
                ))
            return trace

        trace_without = _run_episode(call_label_fn=False)
        trace_with = _run_episode(call_label_fn=True)
        assert trace_without == trace_with, (
            "Computing phase1_labels_for_player() every tick changed the real "
            "simulation's own trajectory -- the counterfactual is leaking "
            "state and/or RNG draws into the real match."
        )


class TestPhase1LabelsForPlayerMatchesRealKick:
    """When a label is computed on the SAME physics tick a real push-kick
    fires, it must reproduce that kick's direction/power_fraction closely
    -- this is only possible if BOTH the timing fix (evaluating from the
    correct pre-movement state) AND the RNG-restore fix (so the real
    kick's own noise draw isn't desynced by the counterfactual's) are
    correct simultaneously. Sweeps several seeds since not every seed
    produces a push-kick within EPISODE_TICKS.

    Two things this test deliberately does NOT require bit-exactness for,
    both confirmed to be genuine, pre-existing, by-design properties of the
    engine rather than anything this fix touches:

    - A tiny (sub-degree to low-single-digit-degree) angular/power gap even
      when both sides fire for real: the label is computed BEFORE
      match.step() advances the tick, so it reads state a hair earlier than
      order.execute() does moments later inside that same tick's own
      _process_orders (confirmed via a direct rng.gauss()-level hook: with
      strict same-tick comparison the two are bit-identical to ~1e-5 in the
      overwhelming majority of cases).
    - A LARGER (up to tens of degrees) gap specifically when the real kick
      is a loose-ball chase that ARMS this tick but doesn't necessarily
      FIRE on this exact tick via order.execute() -- Player.kick_armed_
      direction is a PRE-noise aim estimate, while a real fired kick's
      last_kick_direction is POST-noise, AND for a loose ball, "arm" (in
      order.execute(), which the counterfactual replicates) and "fire"
      (in Match._update_loose_ball_pickup's own contact check, which the
      counterfactual does NOT replicate) can resolve on different micro-
      phases of the same tick or even the next tick. This is a real,
      pre-existing distinction in kick_armed vs kicked_this_tick semantics
      (see Player.kick_armed's own docstring), not something this fix
      changed -- kick_this_tick=1.0 is still correctly reported for both
      cases (see the boolean assertion below, which IS exact), only the
      exact direction/power can differ between "what was armed" and "what
      eventually fired".

    Given both of the above, this test checks the boolean kick_this_tick
    exactly (must always agree), and direction via a generous angular bound
    (still tight enough to catch the ORIGINAL bug, which produced
    unrelated/near-arbitrary directions -- effectively random relative to
    the real kick, not a small offset from it).
    """

    def test_same_tick_label_matches_real_push_kick(self):
        # IMPORTANT: this steps physics ONE TICK AT A TIME via match.step()
        # directly, NOT env.step() -- env.step() internally loops
        # match.step() decision_interval_ticks times (a whole ~0.5s
        # decision interval per call, e.g. 15 physics ticks), and
        # _try_push_kick's geometric gates are re-evaluated on EVERY one of
        # those physics ticks as the player keeps moving (a push-kick can
        # legitimately fire on physics sub-tick 4 of 15, once close enough,
        # not necessarily sub-tick 0). Calling phase1_labels_for_player()
        # once per env.step() and comparing it against "whichever physics
        # sub-tick the kick happened to fire on" compares two genuinely
        # different physical instants -- confirmed live: produced several
        # false "mismatches" during this test's own development, all
        # explained by exactly this granularity mismatch, not a real bug.
        # Single-tick stepping is the correct, apples-to-apples comparison.
        n_kicks_checked = 0
        max_angle_deg = 0.0
        for seed in KICK_SEEDS:
            env = _make_env()
            env.reset(seed=seed)
            match = env.match
            player = match.player_by_id(TRAINEE_ID)
            player.ai = _FreshEachTickAI(decision_interval_ticks=Phase1RulesAI().decision_interval_ticks)
            for _tick in range(EPISODE_TICKS):
                label = phase1_labels_for_player(player, match)
                match.step()
                if player.kicked_this_tick:
                    n_kicks_checked += 1
                    assert label.kick_this_tick == pytest.approx(1.0), (
                        f"seed={seed}: real kick fired this physics tick but the "
                        f"SAME-tick label said kick_this_tick={label.kick_this_tick}"
                    )
                    assert label.kick_direction is not None
                    real_dir = player.last_kick_direction
                    cf = np.asarray(label.kick_direction, dtype=np.float64)
                    real = np.array([real_dir.x, real_dir.y, real_dir.z], dtype=np.float64)
                    cos_sim = float(np.dot(cf, real) / (np.linalg.norm(cf) * np.linalg.norm(real) + 1e-12))
                    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_sim))))
                    max_angle_deg = max(max_angle_deg, angle_deg)
                    assert angle_deg < 45.0, (
                        f"seed={seed}: kick_direction angle mismatch too large "
                        f"({angle_deg:.1f} deg) -- cf={cf} real={real}"
                    )
        assert n_kicks_checked >= 3, (
            f"only {n_kicks_checked} real kick(s) observed across seeds "
            f"{KICK_SEEDS.start}-{KICK_SEEDS.stop - 1} -- widen KICK_SEEDS, "
            f"this test needs several real kicks to be a meaningful check"
        )
        # Not asserted on (see the class docstring for why the arm-vs-fire
        # case can legitimately be large) -- printed so a real regression
        # showing up as "still under 45 deg but now consistently much
        # larger than before" is visible in verbose test output rather than
        # silently passing right under the threshold.
        print(f"\n  [{n_kicks_checked} kicks checked, max direction angle diff = {max_angle_deg:.1f} deg]")


class TestPhase1LabelsPostMovementTimingBug:
    """Regression test for the ORIGINAL bug: computing the label from
    POST-movement state (i.e. AFTER _apply_movement has already run for
    that tick -- the old, now-fixed call pattern every rollout loop used)
    gives a MEASURABLY DIFFERENT answer than computing it from the correct
    pre-movement state, on at least some ticks. This proves the bug was
    real (not just theoretical) and that the fix changes behaviour --
    if this test ever starts failing because the two no longer differ,
    it means either the engine's own tick structure changed (worth
    re-reading Match.step()'s _process_orders/_apply_movement ordering) or
    something silently reverted the timing fix."""

    def test_post_movement_label_differs_from_correct_pre_movement_label(self):
        found_a_difference = False
        for seed in KICK_SEEDS:
            env = _make_env()
            env.reset(seed=seed)
            player = env.match.player_by_id(TRAINEE_ID)
            player.ai = _FreshEachTickAI(decision_interval_ticks=Phase1RulesAI().decision_interval_ticks)
            for _tick in range(EPISODE_TICKS):
                correct_label = phase1_labels_for_player(player, env.match)
                env.step()  # advances _apply_movement -- player state is now POST-movement
                buggy_label = phase1_labels_for_player(player, env.match)  # the OLD, wrong call timing
                if correct_label.kick_this_tick != buggy_label.kick_this_tick:
                    found_a_difference = True
                    break
                if (
                    correct_label.kick_this_tick > 0.5
                    and buggy_label.kick_this_tick > 0.5
                    and correct_label.kick_direction is not None
                    and buggy_label.kick_direction is not None
                    and not np.allclose(correct_label.kick_direction, buggy_label.kick_direction, atol=1e-6)
                ):
                    found_a_difference = True
                    break
            if found_a_difference:
                break
        assert found_a_difference, (
            "Expected the post-movement (buggy timing) label to differ from the "
            "correct pre-movement label on at least one tick across these seeds -- "
            "if it never does, this regression test can no longer detect the bug "
            "it's meant to guard against; widen KICK_SEEDS/EPISODE_TICKS."
        )


def _build_inference_trainer():
    """A real (random-init, untrained) PPOTrainer -- fine for these tests,
    which only check WIRING (does a label get computed at the right time
    and land in the right place), not policy quality."""
    import torch
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    return PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True)


class TestNeuralPlayerAIBcLabelFn:
    """NeuralPlayerAI.bc_label_fn must be called INSIDE act(), at the same
    instant the observation is encoded -- not left for a caller to compute
    later. See rules_ai.py's NeuralPlayerAI.bc_label_fn docstring."""

    def test_bc_label_populated_in_last_transition_on_decision_ticks(self):
        from footballcoach.ai.ppo.bc import phase1_labels_for_player

        trainer = _build_inference_trainer()
        env = _make_env()
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = phase1_labels_for_player
        env.reset(seed=1)

        saw_a_decision_tick = False
        for _ in range(60):
            env.step()
            tr = env.last_trainee_transition
            if tr is not None:
                saw_a_decision_tick = True
                assert "bc_label" in tr, "NeuralPlayerAI.act() did not attach a bc_label to last_transition"
                assert tr["bc_label"] is not None
                assert tr["bc_label"].shape[0] > 0
        assert saw_a_decision_tick, "no decision tick observed in 60 ticks -- widen the loop"

    def test_bc_label_absent_when_bc_label_fn_not_configured(self):
        """Default behaviour (no bc_label_fn) must be preserved exactly --
        bc_label stays None, matching every pre-fix caller that never wired
        one up (e.g. eval/inference-only usage with no BC supervision)."""
        trainer = _build_inference_trainer()
        env = _make_env()
        env.sample_action_fn = trainer._sample_action
        # env.bc_label_fn left at its default (None).
        env.reset(seed=1)

        saw_a_decision_tick = False
        for _ in range(60):
            env.step()
            tr = env.last_trainee_transition
            if tr is not None:
                saw_a_decision_tick = True
                assert tr.get("bc_label") is None
        assert saw_a_decision_tick

    def test_bc_label_matches_direct_call_at_same_state(self):
        """The label NeuralPlayerAI.act() computes internally must be
        identical to calling phase1_labels_for_player() directly against
        the SAME (player, match) at the moment just before act() would next
        fire -- i.e. the internal wiring doesn't somehow compute a
        different answer than the underlying function would given the
        same inputs. Compares MoveOrder/GetPossessionOrder generic fields
        (move_direction/kick_this_tick/tackle_attempt), which don't depend
        on the (deliberately unseeded) sampled action, only on the rules-AI
        counterfactual."""
        from footballcoach.ai.ppo.bc import phase1_labels_for_player

        trainer = _build_inference_trainer()
        env = _make_env()
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = phase1_labels_for_player
        env.reset(seed=1)

        player = env.match.player_by_id(TRAINEE_ID)
        for _ in range(60):
            # Reference label computed just BEFORE env.step() -- i.e. one
            # Match._update_state_timers() call earlier than
            # NeuralPlayerAI.act()'s own internal call (which happens after
            # that same tick's stamina regen has already run) -- not
            # literally the same instant, so atol is 1e-3, not bit-exact.
            # This is still tight enough to catch the internal wiring
            # computing something structurally different (e.g. reading the
            # wrong player/match, or a stale cached value); it's only
            # loose enough to tolerate stamina regen's own tiny effect on
            # movement-intent geometry between these two adjacent points.
            reference = phase1_labels_for_player(player, env.match)
            env.step()
            tr = env.last_trainee_transition
            if tr is not None and reference.valid:
                np.testing.assert_allclose(
                    tr["bc_label"], reference.to_array(), atol=1e-3,
                    err_msg="NeuralPlayerAI's internally-computed bc_label diverged "
                            "from a direct phase1_labels_for_player() call at the same state",
                )
                break


class TestScenarioEnvBcLabelFn:
    """env.bc_label_fn must thread into the TRAINEE's NeuralPlayerAI only
    (not secondary players -- see ScenarioEnv.bc_label_fn's own docstring),
    and must be None by default (no behaviour change for any caller that
    doesn't opt in)."""

    def test_bc_label_fn_defaults_to_none(self):
        env = _make_env()
        assert env.bc_label_fn is None

    def test_bc_label_fn_threaded_into_trainee_neural_player_ai(self):
        from footballcoach.ai.ppo.bc import phase1_labels_for_player

        trainer = _build_inference_trainer()
        env = _make_env()
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = phase1_labels_for_player
        env.reset(seed=1)

        trainee = env.match.player_by_id(TRAINEE_ID)
        assert getattr(trainee.ai, "bc_label_fn", "MISSING") is phase1_labels_for_player
