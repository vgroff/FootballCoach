"""Unit coverage for ai/obs/y_canonical.py on the REAL decision/execution networks (random init).

The property that matters: the wrapped policy is exactly flip_y-equivariant, and it changes nothing for
observers that are already at y >= 0.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.obs.canonical import CanonicalNetworkWrapper
from footballcoach.ai.obs.schema import BALL_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM
from footballcoach.ai.obs.y_canonical import (
    Y_POS_FIELD_IDX,
    YCanonicalNetworkWrapper,
    flip_decision_heads_y,
    flip_exec_heads_y,
    mirror_y_obs,
)


def _batch(n=32, seed=0, force_sign=None):
    g = torch.Generator().manual_seed(seed)
    sf = torch.randn(n, PLAYER_FEATURE_DIM, generator=g)
    if force_sign is not None:
        sf[:, Y_POS_FIELD_IDX] = force_sign * (sf[:, Y_POS_FIELD_IDX].abs() + 0.05)
    of = torch.randn(n, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM, generator=g)
    em = torch.ones(n, MAX_OTHER_PLAYERS)
    bf = torch.randn(n, BALL_FEATURE_DIM, generator=g)
    gf = torch.randn(n, GLOBAL_FEATURE_DIM, generator=g)
    return sf, of, em, bf, gf


@pytest.fixture(scope="module")
def nets():
    torch.manual_seed(0)
    dn, en = DecisionNetwork.from_config().eval(), ExecutionNetwork.from_config().eval()
    plain = (CanonicalNetworkWrapper(dn), CanonicalNetworkWrapper(en))
    wrapped = (YCanonicalNetworkWrapper(plain[0], "decision"), YCanonicalNetworkWrapper(plain[1], "execution"))
    return plain, wrapped


def _heads(pair, obs):
    sf, of, em, bf, gf = obs
    with torch.no_grad():
        d = pair[0](sf, of, em, bf, gf, None, None)
        return d, pair[1](sf, of, em, bf, gf, d, None, None)


def _assert_heads_close(a, b, atol=1e-5):
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if torch.is_tensor(x):
            assert torch.allclose(x, y, atol=atol), f"{f.name} differs (max {float((x - y).abs().max()):.2e})"


def test_wrapped_policy_is_exactly_flip_y_equivariant(nets):
    _, wrapped = nets
    sf, of, em, bf, gf = _batch(64)
    keep = sf[:, Y_POS_FIELD_IDX] != 0
    obs = tuple(t[keep] for t in (sf, of, em, bf, gf))
    allrows = torch.ones(len(obs[0]), dtype=torch.bool)
    fsf, fof, fbf = mirror_y_obs(obs[0], obs[1], obs[3], allrows)
    d, e = _heads(wrapped, obs)
    fd, fe = _heads(wrapped, (fsf, fof, obs[2], fbf, obs[4]))
    _assert_heads_close(fd, flip_decision_heads_y(d, allrows))
    _assert_heads_close(fe, flip_exec_heads_y(e, allrows))


def test_wrapper_changes_nothing_for_observers_at_nonnegative_y(nets):
    plain, wrapped = nets
    obs = _batch(32, force_sign=+1)
    dp, ep = _heads(plain, obs)
    dw, ew = _heads(wrapped, obs)
    _assert_heads_close(dp, dw, atol=0.0)
    _assert_heads_close(ep, ew, atol=0.0)


def test_negative_y_rows_get_the_mirror_of_the_positive_y_answer(nets):
    plain, wrapped = nets
    obs = _batch(32, force_sign=-1)
    everyone = torch.ones(32, dtype=torch.bool)
    msf, mof, mbf = mirror_y_obs(obs[0], obs[1], obs[3], everyone)
    dp, ep = _heads(plain, (msf, mof, obs[2], mbf, obs[4]))       # plain net on the y>=0 mirror image
    dw, ew = _heads(wrapped, obs)
    _assert_heads_close(dw, flip_decision_heads_y(dp, everyone))
    _assert_heads_close(ew, flip_exec_heads_y(ep, everyone))


def test_mixed_batch_rows_are_independent(nets):
    _, wrapped = nets
    obs = _batch(40)
    d_all, e_all = _heads(wrapped, obs)
    for i in (0, 7, 23):
        one = tuple(t[i:i + 1] for t in obs)
        d1, e1 = _heads(wrapped, one)
        assert torch.allclose(e_all.move_direction[i:i + 1], e1.move_direction, atol=1e-5)
        assert torch.allclose(e_all.exec_move_logit[i:i + 1], e1.exec_move_logit, atol=1e-4)
        assert torch.allclose(d_all.move_region_center[i:i + 1], d1.move_region_center, atol=1e-5)


def test_value_only_path_and_value_invariance(nets):
    _, wrapped = nets
    sf, of, em, bf, gf = _batch(24)
    with torch.no_grad():
        d = wrapped[0](sf, of, em, bf, gf, None, None)
        v_only = wrapped[1](sf, of, em, bf, gf, d, None, None, value_only=True)
        full = wrapped[1](sf, of, em, bf, gf, d, None, None)
    assert torch.is_tensor(v_only) and torch.allclose(v_only.reshape(-1), full.value.reshape(-1), atol=1e-6)
    everyone = torch.ones(24, dtype=torch.bool)
    fsf, fof, fbf = mirror_y_obs(sf, of, bf, everyone)
    with torch.no_grad():
        fd = wrapped[0](fsf, fof, em, fbf, gf, None, None)
        fv = wrapped[1](fsf, fof, em, fbf, gf, fd, None, None, value_only=True)
    assert torch.allclose(v_only, fv, atol=1e-5)


def test_state_dict_keys_and_attribute_delegation_match_the_plain_wrapper(nets):
    plain, wrapped = nets
    assert set(plain[0].state_dict()) == set(wrapped[0].state_dict())
    assert set(plain[1].state_dict()) == set(wrapped[1].state_dict())
    assert wrapped[1].move_dir_log_kappa is plain[1].move_dir_log_kappa
    assert wrapped[0].ball_physics_encoder is plain[0].ball_physics_encoder


def test_role_validation():
    with pytest.raises(ValueError):
        YCanonicalNetworkWrapper(torch.nn.Identity(), "value")


# --- with the frozen physics encoders enabled (tiny synthetic checkpoints, as in test_physics_encoder_wiring.py) ---
def _physics_nets(tmp_path):
    from .test_physics_encoder_wiring import (
        BALL_OUTPUT_DIM, PLAYER_OUTPUT_DIM, _make_fake_ball_checkpoint, _make_fake_player_checkpoint,
    )
    from footballcoach.ai.obs.schema import BALL_FEATURE_DIM as BD, PLAYER_FEATURE_DIM as PD

    torch.manual_seed(1)
    dn = DecisionNetwork(
        ball_physics_encoder_checkpoint=_make_fake_ball_checkpoint(tmp_path),
        player_physics_encoder_checkpoint=_make_fake_player_checkpoint(tmp_path),
    ).eval()
    en = ExecutionNetwork(self_dim=PD + PLAYER_OUTPUT_DIM, ball_dim=BD + BALL_OUTPUT_DIM).eval()
    plain = (CanonicalNetworkWrapper(dn), CanonicalNetworkWrapper(en))
    return plain, (YCanonicalNetworkWrapper(plain[0], "decision"), YCanonicalNetworkWrapper(plain[1], "execution"))


def test_equivariance_holds_with_physics_encoders_enabled(tmp_path):
    _, wrapped = _physics_nets(tmp_path)
    sf, of, em, bf, gf = _batch(48, seed=3)
    bf[:, -1] = 1.0
    everyone = torch.ones(48, dtype=torch.bool)
    fsf, fof, fbf = mirror_y_obs(sf, of, bf, everyone)
    d, e = _heads(wrapped, (sf, of, em, bf, gf))
    fd, fe = _heads(wrapped, (fsf, fof, em, fbf, gf))
    _assert_heads_close(fe, flip_exec_heads_y(e, everyone))
    _assert_heads_close(fd, flip_decision_heads_y(d, everyone))


def test_trainer_physics_cache_is_computed_on_the_mirrored_rows_the_wrapper_feeds_the_network(tmp_path):
    from types import SimpleNamespace

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    plain, wrapped = _physics_nets(tmp_path)
    sf, of, em, bf, gf = _batch(64, seed=4)
    bf[:, -1] = 1.0
    assert 0 < int((sf[:, Y_POS_FIELD_IDX] < 0).sum()) < 64
    batch = {"obs/self_feat": sf, "obs/other_feat": of, "obs/ball_feat": bf, "obs/global_feat": gf}
    fake = SimpleNamespace(decision_net=wrapped[0], device=torch.device("cpu"), y_canonical=True)
    cache = PPOTrainer._precompute_physics_full(fake, batch, 32)
    kw = {k.replace("obs/", ""): v for k, v in cache.items()}
    with torch.no_grad():
        with_cache = wrapped[0](sf, of, em, bf, gf, None, None, **kw)
        without = wrapped[0](sf, of, em, bf, gf, None, None)
    assert torch.allclose(with_cache.latent_vector, without.latent_vector, atol=1e-5)
    assert torch.allclose(with_cache.move_logit, without.move_logit, atol=1e-5)
    # sanity: a cache computed WITHOUT the mirror would not match, i.e. the test can fail
    stale = PPOTrainer._precompute_physics_full(SimpleNamespace(decision_net=wrapped[0], device=torch.device("cpu"), y_canonical=False), batch, 32)
    kw_stale = {k.replace("obs/", ""): v for k, v in stale.items()}
    with torch.no_grad():
        bad = wrapped[0](sf, of, em, bf, gf, None, None, **kw_stale)
    assert not torch.allclose(bad.latent_vector, without.latent_vector, atol=1e-5)
