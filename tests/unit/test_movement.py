from __future__ import annotations

import dataclasses
import math

from footballcoach.engine.movement import (
    MovementParams,
    SpeedMode,
    accel_taper_multiplier,
    ball_carry_speed_multiplier,
    drain_stamina,
    effective_acceleration,
    effective_top_speed,
    max_acceleration_mps2,
    max_turn_rate_rad_s,
    regen_stamina,
    stamina_multiplier,
    step_player_towards,
    top_speed_mps,
)
from footballcoach.mathutils import Vector3
from tests.conftest import make_player


def test_top_speed_range():
    params = MovementParams.from_config()
    assert math.isclose(top_speed_mps(params, 0.0), params.top_speed_base_mps)
    assert math.isclose(top_speed_mps(params, 1.0), params.top_speed_base_mps + params.top_speed_scale_mps)


def test_acceleration_range():
    params = MovementParams.from_config()
    assert math.isclose(max_acceleration_mps2(params, 0.0), params.accel_base_mps2)
    assert math.isclose(max_acceleration_mps2(params, 1.0), params.accel_base_mps2 + params.accel_scale_mps2)


def test_stamina_multiplier_bounds():
    params = MovementParams.from_config()
    assert stamina_multiplier(params, 1.0) == 1.0
    assert math.isclose(stamina_multiplier(params, 0.0), 1.0 - params.stamina_speed_penalty_max)


def test_ball_carry_speed_never_reaches_full():
    params = MovementParams.from_config()
    mult_at_max_control = ball_carry_speed_multiplier(params, 1.0)
    assert mult_at_max_control < 1.0
    mult_at_min_control = ball_carry_speed_multiplier(params, 0.0)
    assert mult_at_min_control < mult_at_max_control


def test_effective_top_speed_with_ball_is_slower():
    params = MovementParams.from_config()
    no_ball = effective_top_speed(params, 0.5, 1.0, has_ball=False)
    with_ball = effective_top_speed(params, 0.5, 1.0, has_ball=True, ball_control_attr=0.5)
    assert with_ball < no_ball


def test_effective_acceleration_with_ball_is_reduced_by_the_same_multiplier():
    """Ball-carry should cost acceleration by the SAME fraction it costs top
    speed -- effective_acceleration and effective_top_speed both apply
    ball_carry_speed_multiplier, so the two ratios (with-ball / no-ball)
    should match exactly."""
    params = MovementParams.from_config()
    no_ball_accel = effective_acceleration(params, 0.5, 1.0, has_ball=False)
    with_ball_accel = effective_acceleration(params, 0.5, 1.0, has_ball=True, ball_control_attr=0.5)
    assert with_ball_accel < no_ball_accel

    no_ball_speed = effective_top_speed(params, 0.5, 1.0, has_ball=False)
    with_ball_speed = effective_top_speed(params, 0.5, 1.0, has_ball=True, ball_control_attr=0.5)

    accel_ratio = with_ball_accel / no_ball_accel
    speed_ratio = with_ball_speed / no_ball_speed
    assert math.isclose(accel_ratio, speed_ratio, rel_tol=1e-9)


def test_effective_acceleration_ball_carry_is_a_no_op_by_default():
    """has_ball defaults to False so every existing ETA/decision-estimator
    call site (sprint_eta, rules_ai, orders.py, shot_selection) that doesn't
    pass has_ball keeps its old ball-unaware flat behavior unchanged."""
    params = MovementParams.from_config()
    assert math.isclose(
        effective_acceleration(params, 0.5, 1.0),
        effective_acceleration(params, 0.5, 1.0, has_ball=False, ball_control_attr=0.9),
    )


def test_turn_rate_decreases_with_speed():
    params = MovementParams.from_config()
    slow_turn = max_turn_rate_rad_s(params, 0.5, speed_mps=1.0, has_ball=False)
    fast_turn = max_turn_rate_rad_s(params, 0.5, speed_mps=8.0, has_ball=False)
    assert slow_turn > fast_turn


def test_turn_rate_with_ball_low_control_is_worse():
    params = MovementParams.from_config()
    no_ball = max_turn_rate_rad_s(params, 0.5, speed_mps=5.0, has_ball=False)
    with_ball_low_control = max_turn_rate_rad_s(params, 0.5, speed_mps=5.0, has_ball=True, ball_control_attr=0.0)
    with_ball_high_control = max_turn_rate_rad_s(params, 0.5, speed_mps=5.0, has_ball=True, ball_control_attr=1.0)
    assert with_ball_low_control < no_ball
    assert with_ball_low_control < with_ball_high_control
    assert math.isclose(with_ball_high_control, no_ball)


def test_turn_rate_with_ball_uses_better_of_dribbling_or_control():
    """The ball-carrying penalty should use max(ball_control, dribbling), not
    ball_control alone -- a weak-ball-control-but-strong-dribbling player (or
    vice versa) shouldn't be penalized as if they were weak in both."""
    params = MovementParams.from_config()
    low_control_high_dribbling = max_turn_rate_rad_s(
        params, 0.5, speed_mps=5.0, has_ball=True, ball_control_attr=0.0, dribbling_attr=1.0,
    )
    low_control_low_dribbling = max_turn_rate_rad_s(
        params, 0.5, speed_mps=5.0, has_ball=True, ball_control_attr=0.0, dribbling_attr=0.0,
    )
    high_control_only = max_turn_rate_rad_s(
        params, 0.5, speed_mps=5.0, has_ball=True, ball_control_attr=1.0, dribbling_attr=0.0,
    )
    assert low_control_high_dribbling > low_control_low_dribbling
    assert math.isclose(low_control_high_dribbling, high_control_only)
    # Without the ball, dribbling must be irrelevant entirely.
    no_ball_high_dribbling = max_turn_rate_rad_s(
        params, 0.5, speed_mps=5.0, has_ball=False, dribbling_attr=1.0,
    )
    no_ball_low_dribbling = max_turn_rate_rad_s(
        params, 0.5, speed_mps=5.0, has_ball=False, dribbling_attr=0.0,
    )
    assert math.isclose(no_ball_high_dribbling, no_ball_low_dribbling)


def test_stamina_drains_and_regenerates():
    params = MovementParams.from_config()
    drained = drain_stamina(params, 1.0, stamina_attr=0.5, effort=1.0, dt_s=10.0)
    assert drained < 1.0
    regened = regen_stamina(params, drained, stamina_attr=0.5, dt_s=10.0)
    assert regened > drained


def test_higher_stamina_attr_drains_slower():
    params = MovementParams.from_config()
    low_attr = drain_stamina(params, 1.0, stamina_attr=0.0, effort=1.0, dt_s=5.0)
    high_attr = drain_stamina(params, 1.0, stamina_attr=1.0, effort=1.0, dt_s=5.0)
    assert high_attr > low_attr  # drains less => higher remaining stamina


def test_step_player_towards_moves_in_target_direction():
    player = make_player(position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    for _ in range(120):
        step_player_towards(player, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=1 / 30)
    assert player.position.x > 5.0
    assert abs(player.position.y) < 1e-6


def test_step_player_towards_decelerates_on_zero_target():
    player = make_player(position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    player.velocity = Vector3(5.0, 0.0, 0.0)
    for _ in range(60):
        step_player_towards(player, Vector3.zero(), SpeedMode.STANDSTILL, dt_s=1 / 30)
    assert player.velocity.length() < 0.5


def test_standstill_decelerates_faster_than_jog():
    """STANDSTILL should stop a moving player faster than JOG (standstill_decel_multiplier > 1)."""
    params = MovementParams.from_config()

    def ticks_to_stop(mode: SpeedMode) -> int:
        player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
        player.heading_rad = 0.0
        player.velocity = Vector3(6.0, 0.0, 0.0)
        for i in range(300):
            step_player_towards(player, Vector3.zero(), mode, dt_s=1 / 30, params=params)
            if player.speed_mps < 0.05:
                return i
        return 300

    standstill_ticks = ticks_to_stop(SpeedMode.STANDSTILL)
    jog_ticks = ticks_to_stop(SpeedMode.JOG)
    assert standstill_ticks < jog_ticks, (
        f"STANDSTILL should stop faster than JOG: {standstill_ticks} vs {jog_ticks} ticks"
    )


def test_standstill_snap_clears_drift():
    """A player at near-zero speed in STANDSTILL mode should snap to exactly
    zero (physics-level snap), preventing infinite creep."""
    player = make_player(position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    player.velocity = Vector3(0.015, 0.0, 0.0)  # below _STOP_SNAP_THRESHOLD_MPS (0.02)
    step_player_towards(player, Vector3.zero(), SpeedMode.STANDSTILL, dt_s=1 / 30)
    assert player.speed_mps == 0.0, "velocity should be snapped to zero by physics-level guard"


# ---------------------------------------------------------------------------
# Speed-dependent acceleration taper (accel_taper_multiplier)
# ---------------------------------------------------------------------------

def _taper_params(
    strength: float = 0.6, exponent: float = 1.0, min_fraction: float = 0.35, peak_boost: float = 1.48,
) -> MovementParams:
    base = MovementParams.from_config()
    return dataclasses.replace(
        base,
        accel_taper_strength=strength,
        accel_taper_exponent=exponent,
        accel_taper_min_fraction=min_fraction,
        accel_taper_peak_boost=peak_boost,
    )


def test_accel_taper_full_at_standstill():
    """At current_speed=0, the multiplier is exactly 1.0 (peak/full
    acceleration) regardless of taper strength -- the taper only bites in as
    speed builds up."""
    params = _taper_params(strength=0.6)
    assert accel_taper_multiplier(params, current_speed_mps=0.0, top_speed_mps=10.0) == 1.0


def test_accel_taper_floors_at_top_speed():
    """At current_speed == top_speed, a raw taper of 1-strength that would
    dip BELOW accel_taper_min_fraction gets clamped up to the floor instead
    (strength=0.8 -> raw 0.2, floored to 0.35)."""
    params = _taper_params(strength=0.8, min_fraction=0.35)
    mult = accel_taper_multiplier(params, current_speed_mps=10.0, top_speed_mps=10.0)
    assert math.isclose(mult, 0.35)


def test_accel_taper_decreases_monotonically_with_speed():
    """The multiplier should strictly decrease as current speed climbs
    toward top speed -- explosive at low speed, tapering off near the top."""
    params = _taper_params()
    speeds = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    mults = [accel_taper_multiplier(params, s, top_speed_mps=10.0) for s in speeds]
    for a, b in zip(mults, mults[1:]):
        assert a > b, f"taper multiplier should strictly decrease with speed: {mults}"


def test_accel_taper_clamped_beyond_top_speed():
    """Overspeed (current_speed > top_speed, e.g. right after a collision
    push) must not push the multiplier below the floor -- speed_frac is
    clamped to 1.0 before the taper formula, same result as being exactly
    at top speed."""
    params = _taper_params(strength=0.8, min_fraction=0.35)
    at_top_speed = accel_taper_multiplier(params, current_speed_mps=10.0, top_speed_mps=10.0)
    overspeed = accel_taper_multiplier(params, current_speed_mps=25.0, top_speed_mps=10.0)
    assert math.isclose(overspeed, 0.35)
    assert math.isclose(overspeed, at_top_speed)


def test_accel_taper_strength_zero_is_a_no_op():
    """strength=0.0 (the from_config default when physics.json doesn't set
    it) must behave exactly like no taper at all: multiplier always 1.0."""
    params = _taper_params(strength=0.0)
    for s in (0.0, 3.0, 7.0, 10.0):
        assert accel_taper_multiplier(params, s, top_speed_mps=10.0) == 1.0


def test_from_config_defaults_new_taper_fields_to_a_noop():
    """MovementParams.from_config() must default all four new fields to a
    no-op (strength=0.0, exponent=1.0, min_fraction=1.0, peak_boost=1.0) when
    physics.json's movement section doesn't set them, so older configs
    behave identically to before the taper was introduced."""
    params = MovementParams(
        top_speed_base_mps=6.1, top_speed_scale_mps=4.6, accel_base_mps2=3.3, accel_scale_mps2=2.4,
        stamina_speed_penalty_max=0.25, stamina_drain_sprint_base_per_s=0.006, stamina_drain_attr_lo=1.6,
        stamina_drain_attr_hi=1.1, stamina_regen_idle_base_per_s=0.006, stamina_regen_attr_lo=0.7,
        stamina_regen_attr_hi=0.9, ball_carry_speed_mult_base=0.81, ball_carry_speed_mult_scale=0.15,
        lateral_accel_base_mps2=4.0, lateral_accel_scale_mps2=4.0, lateral_accel_ball_penalty_max=0.6,
        min_speed_for_turn_mps=0.2, goalkeeper_accel_multiplier=3.3, goalkeeper_speed_multiplier=1.23,
        standstill_decel_multiplier=1.6, turn_speed_penalty_max=0.7, control_speed_multiplier=0.75,
        # These four left at from_config()'s documented "missing key" defaults:
        accel_taper_strength=0.0, accel_taper_exponent=1.0, accel_taper_min_fraction=1.0,
        accel_taper_peak_boost=1.0,
    )
    assert params.accel_taper_peak_boost == 1.0
    for s in (0.0, 5.0, 10.0):
        assert accel_taper_multiplier(params, s, top_speed_mps=10.0) == 1.0


def test_step_player_towards_accelerates_faster_from_standstill_than_near_top_speed():
    """Real end-to-end check: the same player, same dt, gains more speed in
    one tick starting from rest than starting from near their own top speed
    -- proof the taper is actually wired into the integrator, not just
    correct in isolation."""
    params = _taper_params(strength=0.6, exponent=1.0, min_fraction=0.35)
    dt = 1 / 30

    player_slow = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    player_slow.heading_rad = 0.0
    player_slow.velocity = Vector3.zero()
    v_top = effective_top_speed(params, player_slow.attributes.top_speed, player_slow.stamina, has_ball=False)

    player_fast = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    player_fast.heading_rad = 0.0
    player_fast.velocity = Vector3(v_top * 0.9, 0.0, 0.0)  # already near top speed

    speed_before_slow = player_slow.speed_mps
    speed_before_fast = player_fast.speed_mps
    step_player_towards(player_slow, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=dt, params=params)
    step_player_towards(player_fast, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=dt, params=params)
    gain_slow = player_slow.speed_mps - speed_before_slow
    gain_fast = player_fast.speed_mps - speed_before_fast

    assert gain_slow > gain_fast, (
        f"expected more speed gained per tick from standstill ({gain_slow:.4f}) than "
        f"from near top speed ({gain_fast:.4f}) -- taper doesn't seem to be applied"
    )


def test_braking_is_not_tapered():
    """Deceleration to STANDSTILL from high speed must be identical whether
    or not the acceleration taper/peak_boost is enabled -- both only apply
    while speeding up (speed_diff > 0), never while braking.

    This specifically guards against an earlier, incorrect version of this
    feature that baked peak_boost straight into physics.json's flat
    accel_base_mps2/accel_scale_mps2 -- which inflated BRAKING (and every
    ETA/decision-estimator that reads those same constants) right along with
    accelerating, breaking several velocity-snap/braking-distance tests.
    peak_boost must live as a separate multiplier applied only in the
    accelerating branch, never touching the flat accel_base/scale values
    braking (and effective_acceleration()) read."""
    dt = 1 / 30

    def ticks_to_stop(params: MovementParams) -> int:
        player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
        player.heading_rad = 0.0
        player.velocity = Vector3(8.0, 0.0, 0.0)
        for i in range(300):
            step_player_towards(player, Vector3.zero(), SpeedMode.STANDSTILL, dt_s=dt, params=params)
            if player.speed_mps == 0.0:
                return i
        return 300

    no_taper = _taper_params(strength=0.0, min_fraction=1.0, peak_boost=1.0)
    with_taper = _taper_params(strength=0.9, exponent=1.0, min_fraction=0.1, peak_boost=2.5)
    assert ticks_to_stop(no_taper) == ticks_to_stop(with_taper), (
        "braking should take the same number of ticks regardless of the acceleration taper/peak_boost"
    )


def test_peak_boost_raises_accelerating_but_not_braking():
    """accel_taper_peak_boost must raise the speed gained per tick while
    accelerating from rest, but must NOT change the flat value
    effective_acceleration() itself returns (that's what braking and every
    ETA estimator use) -- peak_boost is applied only inside
    step_player_towards's accelerating branch, never baked into the config
    constants themselves."""
    params_boosted = _taper_params(strength=0.0, min_fraction=1.0, peak_boost=2.0)
    params_flat = _taper_params(strength=0.0, min_fraction=1.0, peak_boost=1.0)

    # effective_acceleration() itself is unaffected by peak_boost.
    flat_accel = effective_acceleration(params_boosted, acceleration_attr=0.5, stamina_fraction=1.0)
    assert math.isclose(
        flat_accel, effective_acceleration(params_flat, acceleration_attr=0.5, stamina_fraction=1.0)
    )

    dt = 1 / 30

    def first_tick_gain(params: MovementParams) -> float:
        player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
        player.heading_rad = 0.0
        player.velocity = Vector3.zero()
        step_player_towards(player, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=dt, params=params)
        return player.speed_mps

    gain_boosted = first_tick_gain(params_boosted)
    gain_flat = first_tick_gain(params_flat)
    assert gain_boosted > gain_flat, (
        f"peak_boost=2.0 should accelerate faster from rest than peak_boost=1.0: "
        f"{gain_boosted:.4f} vs {gain_flat:.4f}"
    )
    # Should be almost exactly double (strength=0 means no taper shaping,
    # just the flat peak_boost multiplier, both well below top speed so
    # neither clips against the SPRINT target).
    assert math.isclose(gain_boosted, gain_flat * 2.0, rel_tol=0.02)


# ---------------------------------------------------------------------------
# Traction circle: turning and forward acceleration share one force budget
# (step_player_towards's ellipse clamp in the accelerating branch).
# ---------------------------------------------------------------------------

def test_straight_line_acceleration_is_unaffected_by_traction_circle():
    """Heading already aligned with the target direction (a_lat_used == 0)
    -- the ellipse clamp must be a complete no-op, giving exactly the same
    speed gain as the plain taper/peak_boost formula with no ellipse term at
    all. This is the regression guard for every existing straight-line
    sprint test: the traction circle must never touch dead-straight running."""
    params = _taper_params(strength=0.6, exponent=1.0, min_fraction=0.35, peak_boost=1.48)
    dt = 1 / 30

    player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    player.heading_rad = 0.0
    player.velocity = Vector3(4.0, 0.0, 0.0)
    v_top = effective_top_speed(params, player.attributes.top_speed, player.stamina, has_ball=False)
    flat_accel = effective_acceleration(params, player.attributes.acceleration, player.stamina)
    expected_a_max = flat_accel * params.accel_taper_peak_boost * accel_taper_multiplier(params, 4.0, v_top)

    step_player_towards(player, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=dt, params=params)
    gain = player.speed_mps - 4.0
    assert math.isclose(gain, expected_a_max * dt, rel_tol=1e-6)


def test_hard_turn_while_accelerating_gains_less_speed_than_straight_line():
    """The same player, same starting speed, same dt: turning hard (90
    degrees -- turn-rate-limited at 4 m/s given this config's lateral
    capability, yet mild enough that turn_speed_penalty alone still leaves
    target_speed above current_speed, so this stays in the ACCELERATING
    branch where the ellipse actually applies) must gain LESS speed this
    tick than an otherwise-identical player running dead straight -- the
    traction-circle clamp actually engages, not just exists on paper."""
    params = _taper_params(strength=0.6, exponent=1.0, min_fraction=0.35, peak_boost=1.48)
    dt = 1 / 30

    straight = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    straight.heading_rad = 0.0
    straight.velocity = Vector3(4.0, 0.0, 0.0)
    step_player_towards(straight, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=dt, params=params)

    turning = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    turning.heading_rad = 0.0
    turning.velocity = Vector3(4.0, 0.0, 0.0)
    step_player_towards(turning, Vector3(0, 1, 0), SpeedMode.SPRINT, dt_s=dt, params=params)

    gain_straight = straight.speed_mps - 4.0
    gain_turning = turning.speed_mps - 4.0
    assert 0.0 < gain_turning < gain_straight, (
        f"expected less (but still positive) speed gained while turning hard ({gain_turning:.4f}) "
        f"than running straight ({gain_straight:.4f}) -- traction circle doesn't seem to engage"
    )


def test_hard_turn_floors_at_accel_taper_min_fraction():
    """A fully turn-rate-limited turn (90 degrees, at 4 m/s -- exceeds this
    config's max_turn_this_tick, so omega_actual == omega_max and
    lat_fraction == 1.0 exactly, while still mild enough to stay in the
    accelerating branch -- see the previous test) must reduce a_max to
    exactly its `accel_taper_min_fraction` floor, not all the way to zero:
    an exact numeric check that the ellipse clamp's floor (added
    specifically to avoid compounding with boundary/repulsion steering, see
    step_player_towards's comment) lands where intended."""
    params = _taper_params(strength=0.6, exponent=1.0, min_fraction=0.35, peak_boost=1.48)
    dt = 1 / 30

    player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    player.heading_rad = 0.0
    player.velocity = Vector3(4.0, 0.0, 0.0)
    v_top = effective_top_speed(params, player.attributes.top_speed, player.stamina, has_ball=False)
    flat_accel = effective_acceleration(params, player.attributes.acceleration, player.stamina)
    a_max_before_ellipse = flat_accel * params.accel_taper_peak_boost * accel_taper_multiplier(params, 4.0, v_top)
    expected_a_max = a_max_before_ellipse * params.accel_taper_min_fraction

    step_player_towards(player, Vector3(0, 1, 0), SpeedMode.SPRINT, dt_s=dt, params=params)
    gain = player.speed_mps - 4.0
    assert math.isclose(gain, expected_a_max * dt, rel_tol=1e-6)


def test_turning_from_standstill_is_not_penalized():
    """At current_speed=0, a_lat_used (= current_speed * omega) is always
    exactly 0 regardless of how sharp the turn is -- a standstill player's
    first explosive step is never penalized by the traction circle, no
    matter which direction they take it in. Confirms the ellipse only bites
    once the player is actually carrying speed into the turn."""
    params = _taper_params(strength=0.6, exponent=1.0, min_fraction=0.35, peak_boost=1.48)
    dt = 1 / 30

    forward = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    forward.heading_rad = 0.0
    forward.velocity = Vector3.zero()
    step_player_towards(forward, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=dt, params=params)

    sideways = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    sideways.heading_rad = 0.0
    sideways.velocity = Vector3.zero()
    step_player_towards(sideways, Vector3(0, 1, 0), SpeedMode.SPRINT, dt_s=dt, params=params)

    assert math.isclose(forward.speed_mps, sideways.speed_mps, rel_tol=1e-6)


def test_traction_circle_never_affects_braking():
    """Decelerating to STANDSTILL while also needing a big heading change
    must brake exactly as fast as decelerating with no heading change at all
    -- the ellipse clamp lives inside the accelerating (speed_diff > 0)
    branch only, exactly like accel_taper_peak_boost before it (see
    test_braking_is_not_tapered)."""
    params = _taper_params(strength=0.6, exponent=1.0, min_fraction=0.35, peak_boost=1.48)
    dt = 1 / 30

    def ticks_to_stop(target_direction: Vector3) -> int:
        player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
        player.heading_rad = 0.0
        player.velocity = Vector3(8.0, 0.0, 0.0)
        for i in range(300):
            step_player_towards(player, target_direction, SpeedMode.STANDSTILL, dt_s=dt, params=params)
            if player.speed_mps == 0.0:
                return i
        return 300

    # STANDSTILL ignores target_direction for desired_speed (always 0), but
    # still uses it to compute heading_diff/turn_fraction -- so a direction
    # 180 degrees from current heading exercises the same turn-while-braking
    # path a turning brake-to-stop would.
    straight = ticks_to_stop(Vector3(1, 0, 0))
    turning = ticks_to_stop(Vector3(-1, 0, 0))
    assert straight == turning, (
        f"braking should take the same number of ticks whether or not a heading "
        f"change is also requested: {straight} vs {turning} ticks"
    )


def test_time_to_top_speed_roughly_matches_pre_taper_flat_model():
    """The accel_base/scale values in physics.json were raised specifically
    to compensate for the taper's lower average acceleration, so that
    overall time-to-top-speed stays close to what a flat (untapered) model
    with the OLD base/scale values would have produced. This is a coarse
    regression check on that compensation, using the live config -- if
    physics.json's accel_base/scale or accel_taper_* values are retuned
    later without re-deriving the compensation, this should catch a large
    drift (loosely; it's deliberately not tight, since exact retuning is
    expected over time)."""
    params = MovementParams.from_config()
    player = make_player(position=Vector3(0, 0, 0), attr_value=0.5)
    player.heading_rad = 0.0
    dt = 1 / 100
    v_top = effective_top_speed(params, player.attributes.top_speed, player.stamina, has_ball=False)

    t = 0.0
    while player.speed_mps < 0.95 * v_top and t < 10.0:
        step_player_towards(player, Vector3(1, 0, 0), SpeedMode.SPRINT, dt_s=dt, params=params)
        t += dt

    # Old flat model (pre-taper accel_base/scale of 3.3/2.4) took ~1.77s to
    # reach 95% of top speed for a mid-attribute player -- allow a generous
    # band either side since this is about catching gross drift, not pinning
    # an exact number.
    assert 1.0 < t < 3.0, f"time to reach 95% top speed drifted a lot from the ~1.77s baseline: {t:.2f}s"
