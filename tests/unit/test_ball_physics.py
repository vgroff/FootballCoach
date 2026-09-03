from __future__ import annotations

import pytest

from footballcoach.engine.ball_physics import BallPhysicsParams, step_ball
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3


def test_ball_falls_under_gravity():
    ball = Ball.at_rest(Vector3(0, 0, 5.0))
    for _ in range(5):
        step_ball(ball, 1 / 30)
    assert ball.velocity.z < 0


def test_ball_bounces_and_loses_energy():
    ball = Ball.at_rest(Vector3(0, 0, 2.0))
    max_heights = []
    for _ in range(300):
        step_ball(ball, 1 / 30)
        max_heights.append(ball.position.z)
    # Ball should have bounced (velocity sign change happened) and settled
    # lower than initial drop height due to restitution < 1.
    assert max(max_heights[100:]) < 2.0


def test_ball_eventually_settles_on_ground():
    ball = Ball.at_rest(Vector3(0, 0, 1.0))
    ball.velocity = Vector3(3.0, 0.0, 0.0)
    for _ in range(600):
        step_ball(ball, 1 / 30)
    assert ball.position.z < 0.2


def test_rolling_ball_decelerates_due_to_friction():
    ball = Ball.at_rest(Vector3(0, 0, 0.11))
    ball.velocity = Vector3(5.0, 0.0, 0.0)
    speeds = []
    for _ in range(90):
        step_ball(ball, 1 / 30)
        speeds.append(ball.velocity.length_xy())
    assert speeds[-1] < speeds[0]


def test_rolling_ball_decelerates_at_the_analytically_correct_rate():
    """Regression test for a bug where the ground-collision code
    misidentified ordinary rolling contact as a full bounce every tick
    (since gravity's per-tick integration nudges a grounded ball's velocity
    slightly negative even at rest), applying the bounce friction/spin
    impulse EVERY tick instead of the intended gentle rolling friction. That
    bug decayed speed roughly 30x faster than intended. This test asserts
    the measured deceleration over a short window matches
    mu_roll * g analytically, not just "some" deceleration.

    Starts with spin already matching true rolling (v = r*omega) so
    _resolve_ground_friction's sliding-to-rolling spin-up cost (see
    test_zero_spin_ball_pays_spinup_cost_transitioning_to_rolling below) is
    a no-op here -- this test is specifically isolating the flat
    rolling_friction_coefficient decay, not the spin-up transition."""
    params = BallPhysicsParams.from_config()
    ball = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    initial_speed = 5.0
    ball.velocity = Vector3(initial_speed, 0.0, 0.0)
    ball.spin = Vector3(0.0, initial_speed / params.ball_radius_m, 0.0)

    dt = 1 / 30
    n_ticks = 15  # 0.5s - short enough that drag's contribution is minor
    for _ in range(n_ticks):
        step_ball(ball, dt, params)

    elapsed = n_ticks * dt
    expected_decel = params.rolling_friction_coefficient * params.gravity_mps2
    expected_speed = initial_speed - expected_decel * elapsed

    # Allow tolerance for the (smaller, but non-negligible at 5 m/s)
    # aerodynamic drag contribution that rolling friction alone doesn't
    # account for - the key assertion is "same order of magnitude as
    # analytic rolling friction", not an exact match, since drag adds a bit
    # more deceleration on top.
    assert abs(ball.velocity.length_xy() - expected_speed) < 0.2


def test_rolling_ball_travels_plausible_distance_before_stopping():
    """Sanity check from engine/knowledge.md: a ball ALREADY rolling
    without slipping at ~5 m/s should travel a plausible distance before
    stopping on grass, not stop within a couple of metres (the pre-fix
    behaviour). Starts with matching spin (see the analytically-correct-rate
    test above) to isolate rolling_friction_coefficient's own decay from
    _resolve_ground_friction's separate spin-up cost -- see
    test_zero_spin_ball_travels_less_far_than_a_ball_already_rolling below
    for that comparison."""
    params = BallPhysicsParams.from_config()
    ball = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    ball.velocity = Vector3(5.0, 0.0, 0.0)
    ball.spin = Vector3(0.0, 5.0 / params.ball_radius_m, 0.0)
    start_x = ball.position.x

    dt = 1 / 30
    for _ in range(30 * 15):  # up to 15s, plenty of time to stop
        step_ball(ball, dt, params)
        if ball.velocity.length_xy() < 0.05:
            break

    distance_travelled = ball.position.x - start_x
    assert 5.0 < distance_travelled < 30.0


def test_zero_spin_ball_pays_spinup_cost_transitioning_to_rolling():
    """A ball placed in ground contact with NO spin (e.g. a plain grounded
    kick) must pay a real, physically-derived speed cost spinning up to
    true rolling-without-slipping, via _resolve_ground_friction -- this is
    the classic sliding-to-rolling-transition result (the same mechanics
    behind the textbook "5/7 v0" answer for a solid sphere): a ball that
    fully grips retains exactly 1/(1+k) of its pre-slip speed, where k is
    ball_inertia_shell_factor. Added 2026-09-03 alongside
    _resolve_ground_friction itself -- see agent_plans/physics_update.md."""
    params = BallPhysicsParams.from_config()
    initial_speed = 10.0
    ball = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    ball.velocity = Vector3(initial_speed, 0.0, 0.0)
    assert ball.spin == Vector3.zero()

    dt = 1 / 30
    r = params.ball_radius_m
    for _ in range(300):
        step_ball(ball, dt, params)
        slip = ball.velocity.x - r * ball.spin.y
        if abs(slip) < 1e-3:
            break
    else:
        raise AssertionError("ball never reached true rolling (v = r*omega) within 300 ticks")

    expected_retention = 1.0 / (1.0 + params.ball_inertia_shell_factor)
    actual_retention = ball.velocity.x / initial_speed
    # Generous tolerance: unlike the isolated analytic (friction-only)
    # derivation, this ran through the real step_ball tick loop, which also
    # applies aerodynamic drag concurrently during the ~0.85s spin-up
    # transition (real, physical extra loss on top of the pure-friction
    # prediction -- not a bug; confirmed by hand-estimating drag's
    # contribution over that window, which accounts for essentially all of
    # the gap at initial_speed=10).
    assert abs(actual_retention - expected_retention) < 0.08, (
        f"expected ~{expected_retention:.3f} retention (1/(1+k)), got {actual_retention:.3f}"
    )


def test_zero_spin_ball_travels_less_far_than_a_ball_already_rolling():
    """A ball launched with zero spin must travel less far than an
    otherwise-identical ball that starts already rolling without slipping,
    since the zero-spin ball additionally pays _resolve_ground_friction's
    spin-up cost before settling into the same rolling_friction_coefficient
    decay the other ball has from tick 0."""
    params = BallPhysicsParams.from_config()
    dt = 1 / 30

    already_rolling = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    already_rolling.velocity = Vector3(5.0, 0.0, 0.0)
    already_rolling.spin = Vector3(0.0, 5.0 / params.ball_radius_m, 0.0)

    zero_spin = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    zero_spin.velocity = Vector3(5.0, 0.0, 0.0)

    for _ in range(30 * 15):
        if already_rolling.velocity.length_xy() >= 0.05:
            step_ball(already_rolling, dt, params)
        if zero_spin.velocity.length_xy() >= 0.05:
            step_ball(zero_spin, dt, params)

    assert zero_spin.position.x < already_rolling.position.x


def test_drag_reduces_horizontal_speed_in_flight():
    ball = Ball.at_rest(Vector3(0, 0, 1.0))
    ball.velocity = Vector3(20.0, 0.0, 5.0)
    for _ in range(10):
        step_ball(ball, 1 / 30)
    assert ball.velocity.x < 20.0


# ---------------------------------------------------------------------------
# just_bounced_timer_s tests (Phase G)
# ---------------------------------------------------------------------------

def test_just_bounced_timer_set_on_real_bounce():
    """A ball dropped from height and allowed to bounce must have
    just_bounced_timer_s > 0 immediately after the real bounce tick."""
    params = BallPhysicsParams.from_config()
    ball = Ball.at_rest(Vector3(0, 0, 2.0))
    assert ball.just_bounced_timer_s == 0.0

    dt = 1 / 30
    for _ in range(200):
        step_ball(ball, dt, params)
        if ball.just_bounced_timer_s > 0.0:
            break
    else:
        raise AssertionError("just_bounced_timer_s was never set after 200 ticks")

    assert ball.just_bounced_timer_s == pytest.approx(
        params.just_bounced_display_duration_s, abs=1e-9
    ), "timer must be reset to the full display duration on a real bounce"


def test_just_bounced_timer_not_set_on_settling_contact():
    """A ball placed at rest on the ground (no real bounce) must not trigger
    the just_bounced indicator — the settling-contact branch leaves it 0."""
    params = BallPhysicsParams.from_config()
    # Start at ground level with only horizontal velocity (no vertical drop).
    ball = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    ball.velocity = Vector3(3.0, 0.0, 0.0)

    dt = 1 / 30
    for _ in range(30):
        step_ball(ball, dt, params)
        assert ball.just_bounced_timer_s == 0.0, (
            "just_bounced_timer_s should not be set for a rolling ball with no bounce"
        )


def test_just_bounced_timer_decays_to_zero():
    """After a bounce sets just_bounced_timer_s, it must decay to 0 over
    approximately just_bounced_display_duration_s seconds."""
    params = BallPhysicsParams.from_config()
    ball = Ball.at_rest(Vector3(0, 0, 2.0))
    dt = 1 / 30

    # Run until a bounce occurs.
    for _ in range(200):
        step_ball(ball, dt, params)
        if ball.just_bounced_timer_s > 0.0:
            break
    else:
        raise AssertionError("No bounce detected in 200 ticks")

    # Now let it decay.
    display_s = params.just_bounced_display_duration_s
    expected_ticks = round(display_s / dt)
    for i in range(expected_ticks + 5):
        step_ball(ball, dt, params)
        # Once the timer reaches 0 it must stay at 0 (no negative values).
        assert ball.just_bounced_timer_s >= 0.0

    assert ball.just_bounced_timer_s == pytest.approx(0.0, abs=dt + 1e-9), (
        f"timer should have decayed to 0 after ~{expected_ticks} ticks; "
        f"got {ball.just_bounced_timer_s:.4f}"
    )


def test_just_bounced_timer_reset_on_second_bounce():
    """Each real bounce must reset the timer to the full display duration,
    not accumulate."""
    params = BallPhysicsParams.from_config()
    ball = Ball.at_rest(Vector3(0, 0, 2.0))
    dt = 1 / 30
    bounce_count = 0

    for _ in range(400):
        prev_timer = ball.just_bounced_timer_s
        step_ball(ball, dt, params)
        if ball.just_bounced_timer_s == pytest.approx(params.just_bounced_display_duration_s, abs=1e-9) \
                and prev_timer < params.just_bounced_display_duration_s:
            bounce_count += 1
            if bounce_count >= 2:
                break

    assert bounce_count >= 2, "Expected at least 2 bounces in 400 ticks"


def test_magnus_effect_curves_spinning_ball():
    params = BallPhysicsParams.from_config()
    ball_with_spin = Ball.at_rest(Vector3(0, 0, 1.0))
    ball_with_spin.velocity = Vector3(15.0, 0.0, 0.0)
    ball_with_spin.spin = Vector3(0.0, 0.0, 20.0)  # spin about vertical axis -> curves in y

    for _ in range(20):
        step_ball(ball_with_spin, 1 / 30, params)

    assert abs(ball_with_spin.position.y) > 0.01


def test_possessed_ball_does_not_move_under_free_physics():
    ball = Ball.at_rest(Vector3(0, 0, 0.11))
    ball.possessed_by = "p1"
    original_position = ball.position
    step_ball(ball, 1 / 30)
    assert ball.position == original_position
