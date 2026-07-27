from __future__ import annotations

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
    slightly negative even at rest), applying bounce_restitution_horizontal
    (0.8) EVERY tick instead of the intended gentle rolling friction. That
    bug decayed speed roughly 30x faster than intended. This test asserts
    the measured deceleration over a short window matches
    mu_roll * g analytically, not just "some" deceleration."""
    params = BallPhysicsParams.from_config()
    ball = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    initial_speed = 5.0
    ball.velocity = Vector3(initial_speed, 0.0, 0.0)

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
    """Sanity check from engine/knowledge.md: a ball rolled at ~5 m/s should
    travel roughly 20m before stopping on grass (mu_roll=0.06 was tuned for
    this), not stop within a couple of metres (the pre-fix behaviour)."""
    params = BallPhysicsParams.from_config()
    ball = Ball.at_rest(Vector3(0, 0, params.ball_radius_m))
    ball.velocity = Vector3(5.0, 0.0, 0.0)
    start_x = ball.position.x

    dt = 1 / 30
    for _ in range(30 * 15):  # up to 15s, plenty of time to stop
        step_ball(ball, dt, params)
        if ball.velocity.length_xy() < 0.05:
            break

    distance_travelled = ball.position.x - start_x
    assert 10.0 < distance_travelled < 30.0


def test_drag_reduces_horizontal_speed_in_flight():
    ball = Ball.at_rest(Vector3(0, 0, 1.0))
    ball.velocity = Vector3(20.0, 0.0, 5.0)
    for _ in range(10):
        step_ball(ball, 1 / 30)
    assert ball.velocity.x < 20.0


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
