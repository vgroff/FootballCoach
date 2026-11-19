"""The keeper dive timers as network inputs (PlayerFeatures.gk_dive_remaining /
gk_dive_recovery, appended last) and the zero-column padding that lets older
checkpoints load unchanged (ai/ppo/checkpoint_io.pad_player_feature_columns)."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.obs.schema import PLAYER_FEATURE_DIM, PlayerFeatures
from footballcoach.ai.ppo.checkpoint_io import APPENDED_PLAYER_FEATURES, pad_player_feature_columns
from footballcoach.engine.movement import SpeedMode
from footballcoach.ui.scenarios import build_1v2_scenario

NAMES = [f.name for f in dataclasses.fields(PlayerFeatures)]
I_REM, I_REC = NAMES.index("gk_dive_remaining"), NAMES.index("gk_dive_recovery")


def test_the_timers_are_the_last_player_features():
    assert NAMES[-len(APPENDED_PLAYER_FEATURES):] == list(APPENDED_PLAYER_FEATURES)


def _self_feat(match, pid):
    obs = encode_observation(match, pid, time_remaining_s=10.0)
    return obs.self_feat if not torch.is_tensor(obs.self_feat) else obs.self_feat.numpy()


def test_a_diving_keeper_shows_his_timers_outfielders_show_zero():
    m = build_1v2_scenario(seed=4)
    m.movement_params = dataclasses.replace(m.movement_params, goalkeeper_dive_gating=True,
                                            goalkeeper_dive_duration_s=1.5, goalkeeper_dive_recovery_s=1.5)
    gk = m.player_by_id("keeper")
    for _ in range(15):                     # 0.5 s of a dive
        m._gate_goalkeeper_dive(gk, SpeedMode.SPRINT, 1 / 30)
    f = _self_feat(m, "keeper")
    assert f[I_REM] == pytest.approx(1.0 / 1.5, abs=0.03) and f[I_REC] == 0.0
    m._gate_goalkeeper_dive(gk, SpeedMode.JOG, 1 / 30)           # stops: recovery starts
    f = _self_feat(m, "keeper")
    assert f[I_REM] == 0.0 and f[I_REC] == pytest.approx(1.0)
    a = _self_feat(m, "attacker")
    assert a[I_REM] == 0.0 and a[I_REC] == 0.0


def test_old_width_weights_get_zero_columns_right_after_the_player_block():
    k = len(APPENDED_PLAYER_FEATURES)
    n_phys = 7                               # some physics-encoder outputs after the player block
    new = torch.nn.Linear(PLAYER_FEATURE_DIM + n_phys, 5)
    module = torch.nn.Module(); module.self_mlp = torch.nn.Sequential(new)
    old_w = torch.randn(5, PLAYER_FEATURE_DIM - k + n_phys)
    sd = {"self_mlp.0.weight": old_w.clone(), "self_mlp.0.bias": torch.zeros(5)}
    assert pad_player_feature_columns(sd, module, "test")
    w = sd["self_mlp.0.weight"]
    at = PLAYER_FEATURE_DIM - k
    assert w.shape == new.weight.shape
    assert torch.equal(w[:, :at], old_w[:, :at])                  # old player columns
    assert torch.equal(w[:, at:PLAYER_FEATURE_DIM], torch.zeros(5, k))   # the new timers
    assert torch.equal(w[:, PLAYER_FEATURE_DIM:], old_w[:, at:])  # physics outputs, shifted
    x_old = torch.randn(3, old_w.shape[1])
    x_new = torch.cat([x_old[:, :at], torch.randn(3, k), x_old[:, at:]], dim=1)
    assert torch.allclose(x_new @ w.T, x_old @ old_w.T, atol=1e-5)  # new inputs add nothing


def test_current_width_weights_are_left_alone():
    new = torch.nn.Linear(PLAYER_FEATURE_DIM, 4)
    module = torch.nn.Module(); module.self_mlp = torch.nn.Sequential(new)
    sd = {"self_mlp.0.weight": new.weight.detach().clone()}
    assert not pad_player_feature_columns(sd, module, "test")
