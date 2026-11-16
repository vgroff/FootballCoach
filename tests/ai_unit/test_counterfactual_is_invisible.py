"""A BC label's counterfactual order execution must leave the real match
exactly as it was. Before 2026-11-16 a counterfactual push-kick ended the REAL
trainee's first touch (state CONTROLLING_BALL -> ACTIVE, state_timer_s -> 0)
and overwrote the ball's spin, because the restore was a hand-kept list of
fields; it now restores every attribute of the player and the ball."""
from __future__ import annotations

import math

from footballcoach.ai.curriculum.envs import build_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.ppo.bc import phase1_labels_for_player
from footballcoach.entities.player import PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import StopWhenIdleAI


def _state(obj):
    return {k: repr(v) for k, v in vars(obj).items() if not callable(v) and k != "ai"}


def _push_kick_state():
    """A trainee in its first touch of a spinning ball, running at the opponent
    box with nobody near: the rules AI's box-run fires a push-kick."""
    env = build_env(PHASES_BY_ID[1])
    for seed in range(40):
        env.reset(seed=seed)
        m = env._loop.match
        tr, op = m.player_by_id("trainee"), m.player_by_id("opponent")
        sign = -1.0 if tr.team == Team.RIGHT else 1.0          # +x when attacking the right box
        op.ai, op.current_order, op.velocity = StopWhenIdleAI(), None, Vector3.zero()
        op.position = Vector3(-sign * 45.0, 30.0, 0.0)
        tr.position = Vector3(-sign * 10.0, 0.0, 0.0)
        tr.heading_rad = 0.0 if sign > 0 else math.pi
        tr.velocity = Vector3(sign * 6.0, 0.0, 0.0)
        m.ball.position = Vector3(tr.position.x + sign * 0.3, 0.0, m.ball.radius_m)
        m._set_possession("trainee")
        m._sync_possessed_ball()
        tr.state, tr.state_timer_s = PlayerState.CONTROLLING_BALL, 0.2
        m.ball.spin = Vector3(3.0, -2.0, 1.0)
        if phase1_labels_for_player(tr, m).kick_this_tick == 1.0:
            # Restore the setup the probe call may have touched (it is what is
            # under test), then hand back a fresh copy of the same state.
            tr.state, tr.state_timer_s = PlayerState.CONTROLLING_BALL, 0.2
            m.ball.spin = Vector3(3.0, -2.0, 1.0)
            return m, tr
    raise AssertionError("no seed gave a push-kick counterfactual; the test setup needs adjusting")


def test_a_counterfactual_push_kick_changes_nothing():
    m, tr = _push_kick_state()
    before = {p.player_id: _state(p) for p in m.players}
    ball_before, rng_before = _state(m.ball), m.rng.getstate()
    label = phase1_labels_for_player(tr, m)
    assert label.kick_this_tick == 1.0                      # the counterfactual really kicked
    assert {p.player_id: _state(p) for p in m.players} == before
    assert _state(m.ball) == ball_before
    assert m.rng.getstate() == rng_before
    assert tr.state is PlayerState.CONTROLLING_BALL and tr.state_timer_s == 0.2
    assert m.ball.spin == Vector3(3.0, -2.0, 1.0)
