"""Equivalence test: a single, persistent Phase1RulesAI driving a whole
episode vs. a BRAND NEW Phase1RulesAI instance constructed and asked to
decide from scratch on every single tick.

**Documentation must stay in sync with code.** Any significant change, and
any change that conflicts with existing documentation, must be followed by
additions or edits to the relevant documentation (this file, knowledge.md
files, design docs, plans). Otherwise documentation goes stale and confusion
occurs.

Why this matters
-----------------
``bc.py``'s ``phase1_labels()`` -- the live counterfactual query used for (a)
on-policy BC-aux-loss during PPO training and (b) DAgger's rollout-state
labelling (``ai/ppo/dagger.py``'s ``collect_dagger_rollout()``) -- derives
its ``move_direction``/``sprint``/``exec_move``/decision-head fields by
clearing ``player.current_order``, then asking a FRESHLY CONSTRUCTED
``Phase1RulesAI(decision_interval_ticks=1)`` what it would do, every single
call. That's structurally different from how Phase1RulesAI is actually used
everywhere else in this codebase (demonstration recording, the UI, this
test's own sibling ``test_rules_ai_nn_replay_equivalence.py``): ONE instance
constructed once, its ``.act()`` called every tick for the whole episode,
letting whatever instance-level state it carries (see
``_RulesBasedAI.__init__``'s ``decision_interval_ticks``/
``_ticks_since_decision``) accumulate naturally.

``Phase1RulesAI.decide()`` itself holds no ``self.*`` state (a pure function
of ``player``/``match``, confirmed by inspection -- every branch reads only
``player.current_order``/``match.ball.possessed_by``/geometry), and the base
class's ``act()`` forces an immediate decision whenever
``player.current_order is None`` (exactly the state ``phase1_labels()``
always leaves it in beforehand), bypassing the decision-cadence throttle
entirely. So in theory, "fresh instance, current_order cleared first" and
"persistent instance, real gameplay" should be indistinguishable. This test
checks that theory directly, by running two full, live episodes side by side
-- not by asserting properties about the code in isolation -- since this
exact assumption was never covered by ``test_rules_ai_nn_replay_equivalence.py``
(that test checks encode/decode round-tripping through the neural-network
action format, not the fresh-vs-persistent Phase1RulesAI question) and was
never otherwise tested.

Unlike ``test_rules_ai_nn_replay_equivalence.py``, no RNG resyncing trickery
is needed here: ``Phase1RulesAI.decide()`` draws zero RNG (confirmed by its
own "deterministic: no rng draw here" comment in ``rules_ai.py``), and BOTH
sides of this test run the real, live ``Order.execute()`` path (unlike that
other test's shadow side, which replays captured motor output and skips
decision logic entirely, hence needing manual RNG-state syncing). If the two
sides' decisions and RNG-consuming physics (kick noise, tackle rolls, ...)
line up tick-for-tick, their RNG streams stay naturally in lockstep with no
help needed.
"""
from __future__ import annotations

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities.player import Player, Team
from footballcoach.rules_ai import Phase1RulesAI, _RulesBasedAI
from footballcoach.ui.scenarios import build_1v1_scenario

TRAINEE_ID = "trainee"
EPISODE_TICKS = 400  # ~13s at 30Hz -- long enough to see several box-run push-kicks


class _FreshEachTickAI(_RulesBasedAI):
    """Decides at the SAME cadence a real, persistent Phase1RulesAI would --
    inherited from ``_RulesBasedAI``'s own base ``act()``, which calls
    ``decide()`` once every ``decision_interval_ticks`` ticks AND forces an
    immediate out-of-schedule redecision whenever ``current_order`` goes
    ``None`` mid-interval (an order self-completing -- arrival or overshoot
    -- between scheduled decision points; see ``_RulesBasedAI.act()``'s own
    docstring). Subclassing plain ``PlayerAI`` instead of ``_RulesBasedAI``
    (an earlier version of this class) is wrong here: it has the interval
    throttle but NOT the immediate-redecide-on-None-order rule, so an order
    that finishes early would sit idle until the next SCHEDULED tick instead
    of being immediately replaced the way real Phase1RulesAI always is --
    confirmed via direct instrumentation to be exactly what caused this
    test's remaining divergence after the target-locking fix below: a
    several-tick gap on the fresh side (``current_order`` briefly ``None``)
    with no matching gap on the persistent side.

    Only ``decide()`` itself differs from real Phase1RulesAI, mirroring
    ``bc.py``'s ``phase1_labels()`` EXACTLY at each decision point: clears
    ``current_order``, then hands the decision to a BRAND NEW
    ``Phase1RulesAI(decision_interval_ticks=1)`` instance, discarded right
    after -- never a persistent instance whose own state could carry
    anything across decision points.

    Matching the real cadence (rather than deciding every physics tick, an
    earlier version of this class) matters: ``phase1_labels()`` is only ever
    invoked once per REAL decision in production (ppo_trainer.py's rollout
    loop calls it only on ticks a real decision was just made), not every
    tick -- testing a stricter "redecide every tick" cadence than production
    actually uses would conflate a decision-frequency mismatch with the
    actual question this test is asking (whether a fresh instance's
    decision, AT the real decision cadence, agrees with a persistent one's)."""

    def decide(self, player: Player, match: Match, trial_tick: int) -> None:
        player.current_order = None
        Phase1RulesAI(decision_interval_ticks=1).act(player, match, trial_tick=0)


def _snapshot(match: Match) -> dict:
    """Same field set as test_rules_ai_nn_replay_equivalence.py's own
    ``_snapshot()`` -- full physical state, not just the trainee, so a
    divergence caused by the trainee's behaviour affecting the ball/opponent
    (e.g. a push-kick landing differently) is caught too."""
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
        out[f"{p.player_id}_current_order_type"] = type(p.current_order).__name__
        out[f"{p.player_id}_kicked_this_tick"] = bool(p.kicked_this_tick)
    return out


def _assert_snapshots_close(a: dict, b: dict, tick: int) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, tuple):
            for i, (xa, xb) in enumerate(zip(va, vb)):
                assert xa == pytest.approx(xb, abs=1e-6), (
                    f"tick={tick} key={key}[{i}] persistent={xa} fresh_each_tick={xb}"
                )
        elif isinstance(va, float):
            assert va == pytest.approx(vb, abs=1e-6), (
                f"tick={tick} key={key} persistent={va} fresh_each_tick={vb}"
            )
        else:
            assert va == vb, f"tick={tick} key={key} persistent={va} fresh_each_tick={vb}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fresh_each_tick_matches_persistent_ai(seed: int) -> None:
    """A continuously-running Phase1RulesAI and a freshly-reconstructed one
    (current_order cleared first) per tick must drive the trainee
    identically for the whole episode -- exactly the assumption
    phase1_labels() relies on for its move_direction/sprint/exec_move/
    decision-head fields."""
    persistent_match = build_1v1_scenario(seed=seed, trainee_team=Team.LEFT)
    real_ai = Phase1RulesAI()
    persistent_match.player_by_id(TRAINEE_ID).ai = real_ai
    # Opponent is immobile by default (opponent_immobile_prob=1.0) -- ai=None.

    # Same decision_interval_ticks as real_ai -- read off it directly (not
    # re-derived) so this can never silently drift out of sync with whatever
    # the real default resolves to.
    fresh_match = build_1v1_scenario(seed=seed, trainee_team=Team.LEFT)
    fresh_match.player_by_id(TRAINEE_ID).ai = _FreshEachTickAI(
        decision_interval_ticks=real_ai.decision_interval_ticks
    )

    for tick in range(EPISODE_TICKS):
        persistent_match.step()
        fresh_match.step()
        _assert_snapshots_close(_snapshot(persistent_match), _snapshot(fresh_match), tick)
