"""Tests for the frozen ball/player physics-dynamics encoder integration
into DecisionNetwork/ExecutionNetwork (see ai/knowledge.md "Frozen
physics-dynamics encoders" and agent_plans/ball_physics_pretrain_plan.md
section 8).

Builds tiny SYNTHETIC checkpoints on the fly (matching the exact schema
load_frozen_ball_encoder()/load_frozen_player_encoder() expect) rather than
depending on any real trained artifact under checkpoints/physics_pretrain/
-- that directory is gitignored, so a real path would not be portable
across machines/CI.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.obs.canonical import CanonicalNetworkWrapper
from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
    PlayerFeatures,
)
from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsEncoder
from footballcoach.ai.physics_pretrain.ball_episode_gen import N_INPUT_FIELDS as BALL_N_INPUT_FIELDS
from footballcoach.ai.physics_pretrain.live_encoder_features import (
    BALL_AUX_HEAD_DIMS,
    PLAYER_AUX_HEAD_DIMS,
)
from footballcoach.ai.physics_pretrain.player_dynamics_net import PlayerDynamicsEncoder
from footballcoach.ai.physics_pretrain.player_episode_gen import N_INPUT_FIELDS as PLAYER_N_INPUT_FIELDS

BALL_OUTPUT_DIM = 6 + sum(BALL_AUX_HEAD_DIMS.values())
PLAYER_OUTPUT_DIM = 5 + sum(PLAYER_AUX_HEAD_DIMS.values())


def _make_fake_ball_checkpoint(tmp_path, latent_dim: int = 6, hidden_dim: int = 8) -> str:
    encoder = BallDynamicsEncoder(
        input_dim=BALL_N_INPUT_FIELDS, hidden_dim=hidden_dim, latent_dim=latent_dim, bottleneck_dim=8,
    )
    model_state_dict = {}
    for name, out_dim in BALL_AUX_HEAD_DIMS.items():
        head = nn.Linear(latent_dim, out_dim)
        model_state_dict[f"{name}.weight"] = head.weight.detach().clone()
        model_state_dict[f"{name}.bias"] = head.bias.detach().clone()
    ckpt = {
        "config_snapshot": {"hidden_dim": hidden_dim, "latent_dim": latent_dim, "encoder_bottleneck_dim": 8},
        "encoder_state_dict": encoder.state_dict(),
        "model_state_dict": model_state_dict,
    }
    path = tmp_path / "fake_ball_encoder.pt"
    torch.save(ckpt, path)
    return str(path)


def _make_fake_player_checkpoint(tmp_path, latent_dim: int = 5, hidden_dim: int = 8) -> str:
    encoder = PlayerDynamicsEncoder(
        input_dim=PLAYER_N_INPUT_FIELDS, hidden_dim=hidden_dim, latent_dim=latent_dim, bottleneck_dim=8,
    )
    model_state_dict = {}
    for name, out_dim in PLAYER_AUX_HEAD_DIMS.items():
        head = nn.Linear(latent_dim, out_dim)
        model_state_dict[f"{name}.weight"] = head.weight.detach().clone()
        model_state_dict[f"{name}.bias"] = head.bias.detach().clone()
    ckpt = {
        "config_snapshot": {"hidden_dim": hidden_dim, "latent_dim": latent_dim, "encoder_bottleneck_dim": 8},
        "encoder_state_dict": encoder.state_dict(),
        "model_state_dict": model_state_dict,
    }
    path = tmp_path / "fake_player_encoder.pt"
    torch.save(ckpt, path)
    return str(path)


def _make_batch(batch_size: int = 3):
    self_feat = torch.randn(batch_size, PLAYER_FEATURE_DIM)
    other_feat = torch.randn(batch_size, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)
    exists_mask = torch.ones(batch_size, MAX_OTHER_PLAYERS)
    ball_feat = torch.randn(batch_size, BALL_FEATURE_DIM)
    ball_feat[:, -1] = 1.0  # is_loose = 1 by default so masking doesn't zero everything
    global_feat = torch.randn(batch_size, GLOBAL_FEATURE_DIM)
    return self_feat, other_feat, exists_mask, ball_feat, global_feat


def test_output_dims_and_shapes(tmp_path):
    ball_ckpt = _make_fake_ball_checkpoint(tmp_path)
    player_ckpt = _make_fake_player_checkpoint(tmp_path)
    d = DecisionNetwork(
        ball_physics_encoder_checkpoint=ball_ckpt, player_physics_encoder_checkpoint=player_ckpt,
    )
    assert d.ball_physics_encoder.output_dim == BALL_OUTPUT_DIM
    assert d.player_physics_encoder.output_dim == PLAYER_OUTPUT_DIM

    e = ExecutionNetwork(
        self_dim=PLAYER_FEATURE_DIM + PLAYER_OUTPUT_DIM,
        ball_dim=BALL_FEATURE_DIM + BALL_OUTPUT_DIM,
    )
    self_feat, other_feat, exists_mask, ball_feat, global_feat = _make_batch()
    d_heads = d(self_feat, other_feat, exists_mask, ball_feat, global_feat)
    assert d_heads.ball_physics_full.shape == (3, BALL_OUTPUT_DIM)
    assert d_heads.self_physics_full.shape == (3, PLAYER_OUTPUT_DIM)
    assert d_heads.other_physics_full.shape == (3, MAX_OTHER_PLAYERS, PLAYER_OUTPUT_DIM)

    e_heads = e(self_feat, other_feat, exists_mask, ball_feat, global_feat, d_heads)
    assert e_heads.move_direction.shape == (3, 2)
    assert not torch.isnan(d_heads.value).any()
    assert not torch.isnan(e_heads.value).any()


def test_disabled_by_default_is_none():
    d = DecisionNetwork()
    self_feat, other_feat, exists_mask, ball_feat, global_feat = _make_batch()
    d_heads = d(self_feat, other_feat, exists_mask, ball_feat, global_feat)
    assert d.ball_physics_encoder is None
    assert d.player_physics_encoder is None
    assert d_heads.ball_physics_full is None
    assert d_heads.self_physics_full is None
    assert d_heads.other_physics_full is None


def test_is_loose_masks_ball_physics_full(tmp_path):
    ball_ckpt = _make_fake_ball_checkpoint(tmp_path)
    d = DecisionNetwork(ball_physics_encoder_checkpoint=ball_ckpt)
    self_feat, other_feat, exists_mask, ball_feat, global_feat = _make_batch(batch_size=2)
    ball_feat[0, -1] = 0.0  # is_loose = 0
    ball_feat[1, -1] = 1.0  # is_loose = 1
    d_heads = d(self_feat, other_feat, exists_mask, ball_feat, global_feat)
    assert torch.all(d_heads.ball_physics_full[0] == 0.0)
    assert torch.any(d_heads.ball_physics_full[1] != 0.0)


def test_frozen_encoders_never_receive_gradients(tmp_path):
    ball_ckpt = _make_fake_ball_checkpoint(tmp_path)
    player_ckpt = _make_fake_player_checkpoint(tmp_path)
    d = DecisionNetwork(
        ball_physics_encoder_checkpoint=ball_ckpt, player_physics_encoder_checkpoint=player_ckpt,
    )
    e = ExecutionNetwork(
        self_dim=PLAYER_FEATURE_DIM + PLAYER_OUTPUT_DIM,
        ball_dim=BALL_FEATURE_DIM + BALL_OUTPUT_DIM,
    )
    self_feat, other_feat, exists_mask, ball_feat, global_feat = _make_batch()
    d_heads = d(self_feat, other_feat, exists_mask, ball_feat, global_feat)
    e_heads = e(self_feat, other_feat, exists_mask, ball_feat, global_feat, d_heads)

    loss = d_heads.value.sum() + e_heads.value.sum() + e_heads.move_direction.sum()
    loss.backward()

    for p in d.ball_physics_encoder.parameters():
        assert p.grad is None
        assert p.requires_grad is False
    for p in d.player_physics_encoder.parameters():
        assert p.grad is None
        assert p.requires_grad is False

    # Trainable modules DID get a gradient -- confirms the physics-encoder
    # output is still connected to (and doesn't block) the real training path.
    assert d.ball_mlp[0].weight.grad is not None
    assert d.ball_mlp[0].weight.grad.abs().sum() > 0
    assert d.self_mlp[0].weight.grad is not None
    assert d.self_mlp[0].weight.grad.abs().sum() > 0


def test_execution_network_never_recomputes_physics_encoders(tmp_path):
    """The core 'compute once, shared via d_heads' property: each frozen
    encoder must be invoked by DecisionNetwork only -- ExecutionNetwork
    reuses the result off d_heads instead of calling it again."""
    ball_ckpt = _make_fake_ball_checkpoint(tmp_path)
    player_ckpt = _make_fake_player_checkpoint(tmp_path)
    d = DecisionNetwork(
        ball_physics_encoder_checkpoint=ball_ckpt, player_physics_encoder_checkpoint=player_ckpt,
    )
    e = ExecutionNetwork(
        self_dim=PLAYER_FEATURE_DIM + PLAYER_OUTPUT_DIM,
        ball_dim=BALL_FEATURE_DIM + BALL_OUTPUT_DIM,
    )
    ball_calls = 0
    player_calls = 0
    orig_ball_forward = d.ball_physics_encoder.forward
    orig_player_forward = d.player_physics_encoder.forward

    def spy_ball(*args, **kwargs):
        nonlocal ball_calls
        ball_calls += 1
        return orig_ball_forward(*args, **kwargs)

    def spy_player(*args, **kwargs):
        nonlocal player_calls
        player_calls += 1
        return orig_player_forward(*args, **kwargs)

    d.ball_physics_encoder.forward = spy_ball
    d.player_physics_encoder.forward = spy_player

    self_feat, other_feat, exists_mask, ball_feat, global_feat = _make_batch()
    d_heads = d(self_feat, other_feat, exists_mask, ball_feat, global_feat)
    e(self_feat, other_feat, exists_mask, ball_feat, global_feat, d_heads)

    assert ball_calls == 1, "ball encoder must run exactly once per observation (decision only)"
    # Player encoder legitimately runs twice within DecisionNetwork itself
    # (once for self_feat, once for other_feat) -- the property under test
    # is that ExecutionNetwork adds ZERO further calls, not that the total
    # is 1.
    assert player_calls == 2, "player encoder must run exactly twice (self+other), both inside decision_net only"


def test_canonical_mirror_changes_ball_physics_full(tmp_path):
    """Proves ai_config's 'no special-casing needed' claim empirically:
    the physics latent is computed from whatever ball_feat/self_feat
    CanonicalNetworkWrapper already mirrored for the observer's team, so it
    must differ between a Team.LEFT and Team.RIGHT observer of the same
    raw ball state."""
    ball_ckpt = _make_fake_ball_checkpoint(tmp_path)
    d = CanonicalNetworkWrapper(DecisionNetwork(ball_physics_encoder_checkpoint=ball_ckpt))
    d.eval()

    self_feat, other_feat, exists_mask, ball_feat, global_feat = _make_batch(batch_size=1)
    adx = [f.name for f in __import__("dataclasses").fields(PlayerFeatures)].index("attacking_direction")
    self_feat_left = self_feat.clone()
    self_feat_left[:, adx] = 1.0
    self_feat_right = self_feat.clone()
    self_feat_right[:, adx] = -1.0

    with torch.no_grad():
        d_left = d(self_feat_left, other_feat, exists_mask, ball_feat, global_feat)
        d_right = d(self_feat_right, other_feat, exists_mask, ball_feat, global_feat)

    assert not torch.allclose(d_left.ball_physics_full, d_right.ball_physics_full)


def test_state_dict_excludes_physics_encoders_but_load_is_tolerant(tmp_path):
    ball_ckpt = _make_fake_ball_checkpoint(tmp_path)
    d = DecisionNetwork(ball_physics_encoder_checkpoint=ball_ckpt)
    sd = d.state_dict()
    assert not any(k.startswith("ball_physics_encoder.") for k in sd.keys())

    # A fresh instance (own fresh load of the same checkpoint) must accept
    # this state_dict with strict=False without error, and keep its own
    # (already correctly loaded) physics-encoder weights untouched.
    d2 = DecisionNetwork(ball_physics_encoder_checkpoint=ball_ckpt)
    before = {k: v.clone() for k, v in d2.ball_physics_encoder.state_dict().items()}
    d2.load_state_dict(sd, strict=False)
    after = d2.ball_physics_encoder.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k])
