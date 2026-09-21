"""> **Documentation must stay in sync with code.** (see agent_plans/masked_action_training_plan.md)

L0 / L1 / L2 coverage for the per-decision-interval action-opportunity flags
(entities/action_opportunity.py; hooks in Match, NeuralPlayerAI, ScenarioEnv):

* L0 -- scripted scenarios x {kick bit 0/1} x {tackle bit 0/1}: the flags land where the event semantics say.
* L1 -- invariants (fired => bit was 1 and a raw opportunity existed; flags are 0/1).
* L2 -- the independence property: from an identical seeded state the gate bit being trained must not change
  whether an opportunity was recorded (cut flags identical across all four bit combinations).

Scripting: the real network runs (so observations/other heads are real) and only the kick / tackle bits (and
the movement, to keep the trainee still) are overridden on the sampled execution dict.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from footballcoach.entities.action_opportunity import ActionOpportunity
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import StopWhenIdleAI

from tests.ai_scenario.test_batched_rollout import _make_envs, trainer as _trainer  # noqa: F401  (fixture re-export)

trainer = _trainer

KEYS = ("opp_kick", "opp_kick_raw", "fired_kick", "opp_tack", "opp_tack_raw", "fired_tack", "opp_partial")


def _flags(env) -> dict[str, float]:
    re = env.last_trainee_transition["raw_exec"]
    return {k: float(re[k][0]) for k in KEYS}


def _scripted_env(trainer, kick: int, tack: int, *, seed: int, move: bool = False, max_episode_s: float = 6.0):
    envs = _make_envs(trainer, 1, max_episode_s=max_episode_s)
    env = envs[0]
    env._opp_flags_enabled = True

    def scripted(obs_dict):
        res = list(trainer._sample_action(obs_dict))
        ex = dict(res[4])
        ex["kick_this_tick"] = bool(kick)
        ex["kick_direction"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ex["kick_power_fraction"] = 0.5
        ex["tackle_attempt"] = bool(tack)
        if not move:
            ex["exec_move"] = False
        res[4] = ex
        return tuple(res)

    env.sample_action_fn = scripted
    torch.manual_seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    return env


def _place(env, *, ball_xy, carrier: str | None, opp_xy=None, trainee_xy=(0.0, 0.0)):
    """Teleport trainee / ball / opponent; the opponent (if placed) is made stationary."""
    m = env._loop.match
    tr = m.player_by_id("trainee")
    op = m.player_by_id("opponent")
    tr.position = Vector3(trainee_xy[0], trainee_xy[1], 0.0)
    tr.velocity = Vector3.zero()
    op.ai = StopWhenIdleAI()
    op.current_order = None
    if opp_xy is not None:
        op.position = Vector3(opp_xy[0], opp_xy[1], 0.0)
    else:
        op.position = Vector3(30.0, 20.0, 0.0)
    op.velocity = Vector3.zero()
    m.ball.position = Vector3(ball_xy[0], ball_xy[1], m.ball.radius_m)
    m.ball.velocity = Vector3.zero()
    m._set_possession(carrier)
    if carrier is not None:
        m._sync_possessed_ball()


def _run(trainer, kick, tack, setup, *, seed=3):
    env = _scripted_env(trainer, kick, tack, seed=seed)
    setup(env)
    env.step()
    return env, _flags(env)


# ---------------------------------------------------------------------------
# L0 -- event semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kick", [0, 1])
def test_possession_is_a_kick_opportunity_and_kick_fires_iff_bit(trainer, kick):
    env, f = _run(trainer, kick, 0, lambda e: _place(e, ball_xy=(0.0, 0.0), carrier="trainee"))
    assert f["opp_kick"] == 1.0 and f["opp_kick_raw"] == 1.0
    assert f["fired_kick"] == float(kick)
    assert f["opp_tack"] == 0.0 and f["fired_tack"] == 0.0


@pytest.mark.parametrize("kick", [0, 1])
def test_far_loose_ball_and_far_opponent_is_no_opportunity_for_anything(trainer, kick):
    env, f = _run(trainer, kick, 1, lambda e: _place(e, ball_xy=(20.0, 15.0), carrier=None))
    assert f["opp_kick"] == 0.0 and f["opp_kick_raw"] == 0.0 and f["fired_kick"] == 0.0
    assert f["opp_tack"] == 0.0 and f["opp_tack_raw"] == 0.0 and f["fired_tack"] == 0.0


@pytest.mark.parametrize("kick", [0, 1])
def test_loose_ball_pickup_is_an_opportunity_whether_or_not_a_kick_was_armed(trainer, kick):
    env, f = _run(trainer, kick, 0, lambda e: _place(e, ball_xy=(0.0, 0.0), carrier=None))
    assert f["opp_kick"] == 1.0, "a pickup must be recorded as an opportunity even for kick=0 (I1)"
    assert f["fired_kick"] == float(kick), "settled ball at the feet: the armed kick fires at pickup iff kick=1"


@pytest.mark.parametrize("tack", [0, 1])
def test_opposing_carrier_in_contact_is_a_tackle_opportunity_and_fires_iff_bit(trainer, tack):
    env, f = _run(
        trainer, 0, tack,
        lambda e: _place(e, ball_xy=(0.2, 0.0), carrier="opponent", opp_xy=(0.2, 0.0)),
    )
    assert f["opp_tack"] == 1.0 and f["opp_tack_raw"] == 1.0
    assert f["fired_tack"] == float(tack)
    assert f["opp_kick"] == 0.0


@pytest.mark.parametrize("tack", [0, 1])
def test_opposing_carrier_far_is_not_a_tackle_opportunity_even_if_armed(trainer, tack):
    env, f = _run(
        trainer, 0, tack,
        lambda e: _place(e, ball_xy=(12.0, 0.0), carrier="opponent", opp_xy=(12.0, 0.0)),
    )
    assert f["opp_tack"] == 0.0 and f["fired_tack"] == 0.0


# ---------------------------------------------------------------------------
# L1 -- invariants over many random states
# ---------------------------------------------------------------------------

def _random_setup(rng: random.Random):
    """A random reachable-ish configuration; the same rng seed gives the same placement."""
    kind = rng.choice(["carry", "loose_near", "loose_far", "opp_carry_near", "opp_carry_far"])
    tx, ty = rng.uniform(-8, 8), rng.uniform(-5, 5)
    ang = rng.uniform(0, 6.283)
    r = {"carry": 0.0, "loose_near": rng.uniform(0.0, 1.2), "loose_far": rng.uniform(8, 20),
         "opp_carry_near": rng.uniform(0.15, 1.6), "opp_carry_far": rng.uniform(6, 15)}[kind]
    bx, by = tx + r * np.cos(ang), ty + r * np.sin(ang)
    carrier = {"carry": "trainee", "opp_carry_near": "opponent", "opp_carry_far": "opponent"}.get(kind)
    opp_xy = (bx, by) if carrier == "opponent" else None

    def setup(env):
        _place(env, ball_xy=(bx, by), carrier=carrier, opp_xy=opp_xy, trainee_xy=(tx, ty))
    return setup


def test_invariants_and_independence_over_random_states(trainer):
    """L1 + L2: for each seeded random state run all four (kick, tackle) bit combinations from the identical
    starting state. (a) fired => the bit was 1 and a RAW opportunity existed; (b) the CUT opportunity flags
    are identical across the four runs (the gate bit never leaks into whether an opportunity was recorded);
    (c) the RAW flags are allowed to differ only through the documented cross-head effect, and that is
    counted."""
    n_states = 60
    leak_kick = leak_tack = both_raw = fired_any = opp_kick_any = opp_tack_any = 0
    for s in range(n_states):
        outs = {}
        for kick in (0, 1):
            for tack in (0, 1):
                rng = random.Random(1000 + s)
                env = _scripted_env(trainer, kick, tack, seed=50 + s)
                _random_setup(rng)(env)
                env.step()
                outs[(kick, tack)] = _flags(env)
        ref = outs[(0, 0)]
        for (kick, tack), f in outs.items():
            for k in KEYS:
                assert f[k] in (0.0, 1.0)
            assert f["fired_kick"] <= float(kick) and f["fired_tack"] <= float(tack), (s, kick, tack, f)
            if f["fired_kick"]:
                assert f["opp_kick_raw"] == 1.0, (s, kick, tack, f)
            if f["fired_tack"]:
                assert f["opp_tack_raw"] == 1.0, (s, kick, tack, f)
            assert f["opp_kick"] == ref["opp_kick"], f"kick opportunity depends on the gate bits: state {s} {kick, tack} {f} vs {ref}"
            assert f["opp_tack"] == ref["opp_tack"], f"tackle opportunity depends on the gate bits: state {s} {kick, tack} {f} vs {ref}"
            leak_kick += f["opp_kick_raw"] != ref["opp_kick_raw"]
            leak_tack += f["opp_tack_raw"] != ref["opp_tack_raw"]
            both_raw += f["opp_kick_raw"] == 1.0 and f["opp_tack_raw"] == 1.0
        fired_any += any(f["fired_kick"] or f["fired_tack"] for f in outs.values())
        opp_kick_any += ref["opp_kick"]
        opp_tack_any += ref["opp_tack"]
    # The states must actually exercise the machinery, or the assertions above are vacuous.
    assert opp_kick_any >= 5 and opp_tack_any >= 3 and fired_any >= 5, (opp_kick_any, opp_tack_any, fired_any)
    print(f"[independence] states={n_states} raw-flag leaks kick={leak_kick} tack={leak_tack} both_raw_runs={both_raw}")


def test_cross_head_cutoff_first_opportunity_wins():
    a = ActionOpportunity()
    a.begin_interval(10, 0, 0)
    a.note_kick_opp(12)
    a.note_tack_opp(14)
    f = a.flags(0, 0, cross_head_cutoff=True)
    assert f.kick_opp and not f.tack_opp and f.kick_opp_raw and f.tack_opp_raw
    b = ActionOpportunity()
    b.begin_interval(10, 0, 0)
    b.note_tack_opp(11)
    b.note_kick_opp(15)
    g = b.flags(0, 0, cross_head_cutoff=True)
    assert g.tack_opp and not g.kick_opp
    h = b.flags(0, 0, cross_head_cutoff=False)
    assert h.tack_opp and h.kick_opp


def test_begin_interval_resets_markers_and_snapshots_counters():
    a = ActionOpportunity()
    a.begin_interval(0, 3, 1)
    a.note_kick_opp(2)
    a.note_tack_opp(3)
    a.begin_interval(6, 4, 2)
    f = a.flags(4, 2)
    assert not f.kick_opp and not f.tack_opp and not f.kick_fired and not f.tack_fired
    assert a.flags(5, 3).kick_fired and a.flags(5, 3).tack_fired


# ---------------------------------------------------------------------------
# validity is decided by ONE piece of engine code, shared with the opportunity accounting
# ---------------------------------------------------------------------------

from footballcoach.entities.player import PlayerState  # noqa: E402


@pytest.mark.parametrize("who", ["tackler", "carrier"])
def test_inactive_player_is_never_a_tackle_opportunity(trainer, who):
    """A knocked-down (INACTIVE_TACKLED) tackler cannot tackle, and an inactive carrier cannot be tackled -- both
    come from the SAME predicate the armed-tackle resolution uses (Match._tackle_contact_possible)."""
    def setup(e):
        _place(e, ball_xy=(0.2, 0.0), carrier="opponent", opp_xy=(0.2, 0.0))
        m = e._loop.match
        victim = m.player_by_id("trainee" if who == "tackler" else "opponent")
        victim.state = PlayerState.INACTIVE_TACKLED
        victim.state_timer_s = 10.0
    _, f = _run(trainer, 0, 1, setup)
    assert f["opp_tack"] == 0.0 and f["opp_tack_raw"] == 0.0 and f["fired_tack"] == 0.0
    # control: same geometry, nobody inactive
    _, g = _run(trainer, 0, 1, lambda e: _place(e, ball_xy=(0.2, 0.0), carrier="opponent", opp_xy=(0.2, 0.0)))
    assert g["opp_tack"] == 1.0 and g["fired_tack"] == 1.0


def test_tackle_accounting_matches_resolution_and_the_written_spec(trainer):
    """Random geometry x availability: the recorded tackle opportunity, the actual armed-tackle resolution and an
    independent restatement of the rule (opposing teams, overlap distance, both available) all agree."""
    env = _scripted_env(trainer, 0, 0, seed=1)
    m = env._loop.match
    tp = m.tackling_params
    rng = random.Random(7)
    n_true = n_false = 0
    for _ in range(400):
        tr, op = m.player_by_id("trainee"), m.player_by_id("opponent")
        tr.position = Vector3(0.0, 0.0, 0.0)
        r = rng.choice([0.2, 0.5, 0.7, 0.85, 0.95, 1.2, 3.0])
        a = rng.uniform(0, 6.28)
        op.position = Vector3(r * np.cos(a), r * np.sin(a), 0.0)
        tr.state = rng.choice([PlayerState.ACTIVE, PlayerState.INACTIVE_TACKLED])
        op.state = rng.choice([PlayerState.ACTIVE, PlayerState.ACTIVE, PlayerState.INACTIVE_TACKLED])
        m.ball.velocity = Vector3.zero()
        m._set_possession("opponent")
        tr.tackle_armed = True
        tr.opportunity.begin_interval(m.tick_index, tr.kick_count, tr.tackle_fire_count)
        fires_before = tr.tackle_fire_count
        spec = (
            tr.team != op.team
            and np.hypot(op.position.x - tr.position.x, op.position.y - tr.position.y)
            < tp.auto_tackle_overlap_factor * (tr.radius_m + op.radius_m)
            and tr.state != PlayerState.INACTIVE_TACKLED and op.state != PlayerState.INACTIVE_TACKLED
        )
        m._check_armed_tackles()
        recorded = tr.opportunity.tack_first_tick >= 0
        fired = tr.tackle_fire_count > fires_before
        assert recorded == spec, (r, tr.state, op.state)
        assert fired == spec, (r, tr.state, op.state)
        n_true += spec
        n_false += not spec
        # reset whatever the resolution changed so the next trial starts clean
        tr.state, op.state = PlayerState.ACTIVE, PlayerState.ACTIVE
        tr.tackle_armed = False
        m.tick_index += 1
    assert n_true >= 50 and n_false >= 50, (n_true, n_false)


def test_kick_validity_is_one_predicate_shared_by_both_executors_and_the_accounting(trainer):
    """Player.can_kick decides (1) whether either kick executor does anything, (2) the fire-vs-arm branch of
    apply_action_to_player, and (3) whether a kick opportunity is recorded."""
    from footballcoach.ai.action.apply_nn_action import apply_action_to_player
    from footballcoach.ai.action.gating import GatingResult

    env = _scripted_env(trainer, 0, 0, seed=2)
    m = env._loop.match
    tr = m.player_by_id("trainee")
    rng = random.Random(3)
    for _ in range(60):
        holder = rng.choice(["trainee", "opponent", None])
        m.ball.position = Vector3(tr.position.x, tr.position.y, m.ball.radius_m)
        m.ball.velocity = Vector3.zero()
        m._set_possession(holder)
        can = holder == "trainee"
        assert tr.can_kick(m) == can
        for execute in (
            lambda: tr.kick_with_direction(m, Vector3(1.0, 0.0, 0.0), 0.5, Vector3.zero()),
            lambda: tr.kick_direct(m, Vector3(20.0, 0.0, 0.0), 0.5, Vector3.zero()),
        ):
            m._set_possession(holder)
            before = tr.kick_count
            execute()
            assert (tr.kick_count > before) == can
        # fire-vs-arm branch of the NN action applier
        m._set_possession(holder)
        tr.kick_armed = False
        before = tr.kick_count
        g = GatingResult(selected=None, target_slot=None, exec_move=False, move_direction=None, sprint=False,
                         kick_this_tick=True, kick_direction=np.array([1.0, 0.0, 0.0]), kick_power_fraction=0.5,
                         kick_spin=None, tackle_attempt=False)
        apply_action_to_player(g, tr, m, [None] * 21, {})
        assert (tr.kick_count > before) == can and tr.kick_armed == (not can)
        # accounting
        tr.opportunity.begin_interval(m.tick_index, tr.kick_count, tr.tackle_fire_count)
        tr.note_kick_opportunity(m)
        m._set_possession(holder)
        tr.opportunity.begin_interval(m.tick_index, tr.kick_count, tr.tackle_fire_count)
        tr.note_kick_opportunity(m)
        assert (tr.opportunity.kick_first_tick >= 0) == can


@pytest.mark.parametrize("vz", [0.0, 3.0, 6.0])
@pytest.mark.parametrize("kick", [0, 1])
def test_first_touch_is_a_kick_opportunity_settled_or_bouncing(trainer, kick, vz):
    """A kick is valid on first touch both ways the engine allows it: an ARMED kick firing at pickup when the
    ball has settled (vz below armed_redirect_settle_vz_mps), or -- when the ball is still bouncing and the
    pickup instead grants ordinary possession -- the cached kick intent firing next tick from CONTROLLING_BALL.
    Either way the pickup is recorded as an opportunity whatever the kick bit, and the kick fires iff the bit is 1."""
    def setup(e):
        _place(e, ball_xy=(0.0, 0.0), carrier=None)
        m = e._loop.match
        m.ball.velocity = Vector3(0.0, 0.0, vz)
        m.ball.position = Vector3(0.0, 0.0, m.ball.radius_m + 0.05)
    _, f = _run(trainer, kick, 0, setup)
    assert f["opp_kick"] == 1.0 and f["opp_kick_raw"] == 1.0
    assert f["fired_kick"] == float(kick)
