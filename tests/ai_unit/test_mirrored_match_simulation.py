"""A real match and its physical mirror image, both driven by the neural networks, must evolve identically (mirrored).

Follows test_mirrored_match.py (state-level) with DYNAMICS: every player of a random engine ``Match`` is driven by a
``NeuralPlayerAI`` (real ``encode_observation`` -> wrapped networks with the frozen physics encoders enabled -> action gating ->
engine orders), the mirror image is driven the same way, and both are stepped tick by tick. At every tick the physical state
(positions, velocities, headings, stamina, ball position/velocity/spin, possession, ...) must be the mirror of the other match; at
every decision the network INPUTS (after canonicalisation), physics-encoder outputs, latent vector, value, every head and the chosen
(deterministic) action must agree.

Mirrors: x (the other team's view: team swap), y, and both. Randomness: ``rng_reduction=1.0`` switches off the engine's kick/control
noise, and decoding is deterministic. (Sampled actions are mirror-symmetric only in DISTRIBUTION -- the sample noise is drawn after
the outputs are mirrored back -- which the log-prob/KL tests cover, so they cannot be compared sample by sample.)
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.obs.canonical import canonicalize_obs
from footballcoach.ai.obs.y_canonical import flip_decision_heads_y, flip_exec_heads_y, mirror_y_obs, y_flip_mask
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.rules_ai import NeuralPlayerAI

from .test_mirrored_match import _MIRRORS, _mirror_spin, _random_match, mirror_match
from .test_physics_encoder_wiring import (
    BALL_OUTPUT_DIM, PLAYER_OUTPUT_DIM, _make_fake_ball_checkpoint, _make_fake_player_checkpoint,
)
from .test_y_canonical import _assert_heads_close

_TICKS = 75                                    # decision_interval_ticks=15 -> 5 decisions per player
_PLAYERS = ("A1", "A2", "B1", "B2")


def _trainer(tmp_path, y_canonical: bool) -> PPOTrainer:
    from footballcoach.ai.obs.schema import BALL_FEATURE_DIM as BD, PLAYER_FEATURE_DIM as PD

    torch.manual_seed(1)
    dn = DecisionNetwork(
        ball_physics_encoder_checkpoint=_make_fake_ball_checkpoint(tmp_path),
        player_physics_encoder_checkpoint=_make_fake_player_checkpoint(tmp_path),
    )
    en = ExecutionNetwork(self_dim=PD + PLAYER_OUTPUT_DIM, ball_dim=BD + BALL_OUTPUT_DIM)
    # A random-init network's small head outputs are bias-dominated (nobody ever moves, kicks or tackles). Amplify the action heads so the
    # decisions vary from state to state and the simulated match has real dynamics.
    with torch.no_grad():
        for net in (dn, en):
            for name, m in net.named_modules():
                if isinstance(m, torch.nn.Linear) and "value" not in name and "physics" not in name and any(
                        k in name for k in ("logit", "get_possession", "move_direction", "kick_direction", "move_arrival", "attack_defence")):
                    m.weight.mul_(25.0)
        # bias the heads toward acting: usually move, sometimes kick / tackle, and pursue the ball
        en.exec_move_logit.bias.fill_(2.0)
        en.kick_logit.bias.fill_(-1.0)
        en.tackle_attempt_logit.bias.fill_(-1.0)
        dn.move_logit.bias.fill_(1.0)
        dn.get_possession_raw.bias.fill_(1.0)
    cfg = load_ai_config()
    return PPOTrainer(decision_net=dn, execution_net=en, cfg=cfg, device=torch.device("cpu"), inference_only=True,
                      y_canonical=y_canonical)


class _Recorder:
    """sample_action_fn for NeuralPlayerAI: records the network inputs/heads of every decision, returns the deterministic action."""

    def __init__(self, trainer: PPOTrainer, y_canonical: bool):
        self.trainer, self.y_canonical, self.log = trainer, y_canonical, []

    def __call__(self, obs_dict):
        batch = {k: (v.unsqueeze(0) if v.dim() in (1, 2) and k != "exists_mask" or (k == "exists_mask" and v.dim() == 1) else v)
                 for k, v in obs_dict.items()}
        with torch.no_grad():
            d, e, value, _, _ = self.trainer._sample_action_networks(batch)
            result = self.trainer._sample_action(obs_dict, deterministic=True)
        sf, of, bf = batch["self_feat"], batch["other_feat"], batch["ball_feat"]
        if self.y_canonical:
            sf, of, bf = mirror_y_obs(sf, of, bf, y_flip_mask(sf))
        sf, of, bf, _ = canonicalize_obs(sf, of, bf)
        self.log.append(dict(inputs=dict(self_feat=sf, other_feat=of, ball_feat=bf, global_feat=batch["global_feat"],
                                         exists_mask=batch["exists_mask"], self_ai_type=batch["self_ai_type"],
                                         other_ai_type=batch["other_ai_type"]), d=d, e=e, value=value, result=result))
        return result


def _attach_neural_ai(match, recorder):
    for i, p in enumerate(match.players):
        p.ai = NeuralPlayerAI(recorder, decision_interval_ticks=15, max_episode_s=60.0, rng=random.Random(100 + i))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _state_deviation(a, b, mx: bool, my: bool) -> float:
    """Max abs deviation between match ``b`` and the (mx, my)-mirror of match ``a`` over every physical quantity."""
    sx, sy = (-1.0 if mx else 1.0), (-1.0 if my else 1.0)
    dev = 0.0
    for pa, pb in zip(a.players, b.players):
        assert pa.player_id == pb.player_id
        dev = max(dev, abs(pb.position.x - sx * pa.position.x), abs(pb.position.y - sy * pa.position.y), abs(pb.position.z - pa.position.z))
        dev = max(dev, abs(pb.velocity.x - sx * pa.velocity.x), abs(pb.velocity.y - sy * pa.velocity.y), abs(pb.velocity.z - pa.velocity.z))
        th = pa.heading_rad
        th = math.pi - th if mx else th
        th = -th if my else th
        dev = max(dev, abs(_wrap(pb.heading_rad - th)))
        dev = max(dev, abs(pb.stamina - pa.stamina), abs(pb.state_timer_s - pa.state_timer_s))
        if pb.state != pa.state:
            return math.inf
        da, db = pa.desired_direction, pb.desired_direction
        dev = max(dev, abs(db.x - sx * da.x), abs(db.y - sy * da.y))
    ba, bb = a.ball, b.ball
    dev = max(dev, abs(bb.position.x - sx * ba.position.x), abs(bb.position.y - sy * ba.position.y), abs(bb.position.z - ba.position.z))
    dev = max(dev, abs(bb.velocity.x - sx * ba.velocity.x), abs(bb.velocity.y - sy * ba.velocity.y), abs(bb.velocity.z - ba.velocity.z))
    ms = _mirror_spin(ba.spin, mx, my)
    dev = max(dev, abs(bb.spin.x - ms.x), abs(bb.spin.y - ms.y), abs(bb.spin.z - ms.z))
    if ba.possessed_by != bb.possessed_by or ba.last_touched_by_player_id != bb.last_touched_by_player_id:
        return math.inf
    return dev


def _shadow_decisions(match, recorder, tick: int):
    """Run the wrapped networks on every player's observation exactly like a decision would (without acting on it)."""
    from footballcoach.ai.obs.encoder import encode_observation

    for i, p in enumerate(match.players):
        obs = encode_observation(match, p.player_id, time_remaining_s=max(0.0, 60.0 - tick / 30.0), attack_defence_smoothed=0.0,
                                 rng=random.Random(1000 * tick + i))
        recorder(obs.to_torch_dict())


def run_mirrored_simulation(tmp_path, seed: int, possessed_by, mirror: str, y_canonical: bool = True, ticks: int = _TICKS,
                            perturb: float = 0.0, driver: str = "neural", contact_damping: bool = False) -> SimpleNamespace:
    """``driver="neural"``: the networks act. ``driver="rules"``: the rules AI acts (real dribbling / kicking / tackling) and the networks
    run in shadow mode on the same observations every 15 ticks."""
    from footballcoach.rules_ai import Phase1RulesAI

    mx, my = _MIRRORS[mirror]
    trainer = _trainer(tmp_path, y_canonical)
    base = _random_match(seed, possessed_by)
    other = mirror_match(base, mx, my)
    rec_a, rec_b = _Recorder(trainer, y_canonical), _Recorder(trainer, y_canonical)
    if not contact_damping:
        # Engine knife-edge (see test_contact_between_opposing_players_is_not_mirror_stable): after the first push-apart a colliding pair sits
        # within 1 ulp of the touching distance, and the second push / velocity damping are decided by that last bit, so ~1e-16 rounding
        # noise becomes an O(1) m/s velocity difference at the first contact. Switching the damping off (retention 1.0) isolates it.
        from footballcoach.engine.collision import CollisionParams
        for m in (base, other):
            m.collision_params = CollisionParams(collision_velocity_retention=1.0,
                                                 collision_damping_min_closing_speed_mps=m.collision_params.collision_damping_min_closing_speed_mps)
    if driver == "neural":
        _attach_neural_ai(base, rec_a)
        _attach_neural_ai(other, rec_b)
    else:
        for m in (base, other):
            for p in m.players:
                p.ai = Phase1RulesAI()
    if perturb:
        other.ball.velocity = type(other.ball.velocity)(other.ball.velocity.x + perturb, other.ball.velocity.y, other.ball.velocity.z)
    devs = [_state_deviation(base, other, mx, my)]
    trace = []
    poss = [base.ball.possessed_by]
    for t in range(ticks):
        if driver == "rules" and t % 15 == 0:
            _shadow_decisions(base, rec_a, t)
            _shadow_decisions(other, rec_b, t)
        base.step()
        other.step()
        poss.append(base.ball.possessed_by)
        devs.append(_state_deviation(base, other, mx, my))
        trace.append(((base.ball.position.x, base.ball.position.y), (other.ball.position.x, other.ball.position.y),
                      (base.players[0].position.x, base.players[0].position.y), (other.players[0].position.x, other.players[0].position.y)))
    return SimpleNamespace(mx=mx, my=my, state_dev=devs, trace=trace, poss=poss, log_a=rec_a.log, log_b=rec_b.log, base=base, other=other)


@pytest.mark.parametrize("driver", ["neural", "rules"])
@pytest.mark.parametrize("mirror", ["x", "y", "xy"])
@pytest.mark.parametrize("seed,possessed_by", [(0, None), (1, "A2"), (2, None)])
def test_mirrored_match_evolves_identically_state_and_network_io(tmp_path, mirror, seed, possessed_by, driver):
    r = run_mirrored_simulation(tmp_path, seed, possessed_by, mirror, driver=driver)
    # 1. physical state, every tick
    assert max(r.state_dev) < 1e-6, f"state deviation {max(r.state_dev):.3e} at tick {int(np.argmax(r.state_dev))}"
    # the run actually moved things (a frozen match would trivially be 'identical')
    assert max(abs(x - y) for (bp, _, _, _) in r.trace[-1:] for x, y in zip(bp, (0.0, 0.0))) >= 0.0
    p0 = r.trace[0][2]
    assert math.hypot(r.trace[-1][2][0] - p0[0], r.trace[-1][2][1] - p0[1]) > 0.5, "player 0 did not move -- vacuous test"
    # 2. network inputs / physics-encoder outputs / latent / heads / actions, every decision
    assert len(r.log_a) == len(r.log_b) >= 4 * (_TICKS // 15)
    for k, (la, lb) in enumerate(zip(r.log_a, r.log_b)):
        for key in la["inputs"]:
            assert torch.allclose(la["inputs"][key], lb["inputs"][key], atol=1e-5), f"decision {k}: network input {key} differs"
        da, ea = la["d"], la["e"]
        if r.my:                                       # y-signed outputs come back in each match's own world frame
            flip = torch.ones(1, dtype=torch.bool)
            da, ea = flip_decision_heads_y(da, flip), flip_exec_heads_y(ea, flip)
        _assert_heads_close(da, lb["d"], atol=1e-4)
        _assert_heads_close(ea, lb["e"], atol=1e-4)
        assert torch.allclose(la["d"].latent_vector, lb["d"].latent_vector, atol=1e-4)
        assert torch.allclose(la["d"].ball_physics_full, lb["d"].ball_physics_full, atol=1e-4)
        assert torch.allclose(la["d"].self_physics_full, lb["d"].self_physics_full, atol=1e-4)
        assert torch.allclose(la["d"].other_physics_full, lb["d"].other_physics_full, atol=1e-4)
        assert torch.allclose(la["value"], lb["value"], atol=1e-4)
        # the chosen (deterministic) action, in each match's world frame
        (_, _, va, pa, xa, dpa, _, _, _), (_, _, vb, pb, xb, dpb, _, _, _) = la["result"], lb["result"]
        sx, sy = (-1.0 if r.mx else 1.0), (-1.0 if r.my else 1.0)
        assert abs(va - vb) < 1e-4
        assert all(abs(pa[k2] - pb[k2]) < 1e-5 for k2 in pa)
        for flag in ("exec_move", "sprint", "kick_this_tick", "tackle_attempt"):
            assert xa[flag] == xb[flag]
        assert np.allclose(np.asarray(xb["move_direction"]), np.asarray(xa["move_direction"]) * [sx, sy], atol=1e-5)
        assert np.allclose(np.asarray(xb["kick_direction"])[:2], np.asarray(xa["kick_direction"])[:2] * [sx, sy], atol=1e-5)
        assert np.allclose(np.asarray(dpb["move_region_center_m"]), np.asarray(dpa["move_region_center_m"]) * [sx, sy], atol=1e-4)


def test_x_mirrored_simulation_also_holds_with_x_canonicalisation_alone(tmp_path):
    r = run_mirrored_simulation(tmp_path, 0, None, "x", y_canonical=False)
    assert max(r.state_dev) < 1e-6
    for la, lb in zip(r.log_a, r.log_b):
        for key in la["inputs"]:
            assert torch.allclose(la["inputs"][key], lb["inputs"][key], atol=1e-5), key
        _assert_heads_close(la["e"], lb["e"], atol=1e-4)


def test_the_comparison_can_fail_a_y_mirror_is_not_identical_without_y_canonicalisation(tmp_path):
    r = run_mirrored_simulation(tmp_path, 0, None, "y", y_canonical=False, ticks=30)
    bad = False
    for la, lb in zip(r.log_a, r.log_b):
        for key in la["inputs"]:
            bad |= not torch.allclose(la["inputs"][key], lb["inputs"][key], atol=1e-5)
    assert bad


def test_a_tiny_asymmetry_between_the_two_matches_is_detected(tmp_path):
    """Sensitivity control: a 1 mm/s difference in one ball velocity component must show up in the state comparison."""
    ok = run_mirrored_simulation(tmp_path, 0, None, "y", ticks=45)
    bad = run_mirrored_simulation(tmp_path, 0, None, "y", ticks=45, perturb=1e-3)
    assert max(ok.state_dev) < 1e-6
    assert max(bad.state_dev) > 1e-4


@pytest.mark.parametrize("mirror", ["x", "y", "xy"])
@pytest.mark.parametrize("seed,possessed_by", [(0, None), (1, "A2"), (2, None)])
def test_neural_driven_mirrored_match_stays_identical_for_ten_seconds(tmp_path, mirror, seed, possessed_by):
    r = run_mirrored_simulation(tmp_path, seed, possessed_by, mirror, ticks=300)
    assert max(r.state_dev) < 1e-6, f"state deviation {max(r.state_dev):.3e} first exceeding 1e-6 at tick {int(np.argmax(np.array(r.state_dev) > 1e-6))}"


@pytest.mark.parametrize("mirror", ["x", "y", "xy"])
@pytest.mark.parametrize("seed,possessed_by", [(0, None), (1, "A2"), (2, None)])
def test_rules_ai_driven_mirrored_match_stays_identical_for_almost_seven_seconds(tmp_path, mirror, seed, possessed_by):
    """The rules AI (dribbling, kicking, tackling, pursuit) plus the engine are mirror-symmetric to rounding. Past ~230 ticks the pursuit
    dynamics amplify the ~1e-15 rounding noise exponentially (measured: x1e6 in ~65 ticks, i.e. 4e-15 at tick 104 -> 1e-9 at tick 229 -> 1e-6
    at tick 295 in seed 0), so the horizon is bounded to 200 ticks; that growth is chaos, not a symmetry violation."""
    r = run_mirrored_simulation(tmp_path, seed, possessed_by, mirror, ticks=200, driver="rules")
    assert max(r.state_dev) < 1e-6, f"state deviation {max(r.state_dev):.3e} first exceeding 1e-6 at tick {int(np.argmax(np.array(r.state_dev) > 1e-6))}"


@pytest.mark.xfail(strict=True, reason="engine knife-edge: collision push-apart leaves a pair within 1 ulp of the touching distance and the second push / "
                                       "velocity damping are decided by rounding (~1e-16) -> O(1) m/s velocity differences between a match and its mirror "
                                       "after the first contact; flips to XPASS (and fails) when the engine is made rounding-robust")
def test_contact_between_opposing_players_is_not_mirror_stable(tmp_path):
    r = run_mirrored_simulation(tmp_path, 0, None, "x", ticks=300, contact_damping=True)
    assert max(r.state_dev) < 1e-6
