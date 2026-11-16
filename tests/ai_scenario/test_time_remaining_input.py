"""The true time-remaining input (2026-11-16), ai/ppo/time_input_remap.py, and
the UI's task phase (curriculum.envs.ui_task_phase)."""
from __future__ import annotations

import math
import random

import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_env, build_role_env, ui_task_phase
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.ppo.time_input_remap import TIME_REMAINING_COL, remap_time_remaining_input
from footballcoach.ai.ppo.trainer_common import _ai_types

OBS = load_ai_config()["observation"]
NORM = math.log1p(float(OBS["time_remaining_norm_max_s"]))


def _trainer(separate_value_net=True):
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(4)
    return PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=separate_value_net)


@pytest.fixture(scope="module")
def trainer():
    return _trainer()


def _env(trainer, phase_id):
    ph = PHASES_BY_ID[phase_id]
    env = build_env(ph) if not ph.is_role_based else build_role_env(
        ph, controller_weights={"attacker": {"neural": 1}, "defender": {"rules": 1}, "goalkeeper": {"rules": 1}})
    env.sample_action_fn = trainer._sample_action
    return env


def _rows_vs_env(trainer, phase_id, n_steps=12):
    """(row time feature, env observation time feature at the same instant,
    legacy formula) for the trainee's first decisions of a seeded episode."""
    env = _env(trainer, phase_id)
    obs = env.reset(seed=3)
    out = []
    for _ in range(n_steps):
        ticks = env._episode_ticks
        env_f = float(obs.to_torch_dict()["global_feat"].flatten()[TIME_REMAINING_COL])
        obs, _, done, _ = env.step()
        row_f = float(env.last_trainee_transition["obs"]["global_feat"].flatten()[TIME_REMAINING_COL])
        legacy = math.log1p(max(0.0, env.max_episode_s - (ticks + 1) / 30.0)) / NORM
        out.append((row_f, env_f, legacy))
        if done:
            break
    return out


@pytest.mark.parametrize("phase_id", [1, 2])
def test_rows_carry_the_true_clock_identical_to_the_env_observation(trainer, phase_id):
    rows = _rows_vs_env(trainer, phase_id)
    assert all(abs(r - e) < 1e-6 for r, e, _ in rows), rows   # rows and bootstrap agree
    assert any(abs(r - legacy) > 1e-4 for r, _, legacy in rows)  # and it is not the old clock


def _obs():
    env = build_env(PHASES_BY_ID[1])
    env.reset(seed=12)
    return {k: v.unsqueeze(0) for k, v in encode_observation(
        match=env.match, player_id="trainee", time_remaining_s=9.0, attack_defence_smoothed=0.5,
        rng=random.Random(0), phase=1).to_torch_dict().items()}


def _out(t, obs, f):
    d = {k: v.clone() for k, v in obs.items()}
    d["global_feat"][..., TIME_REMAINING_COL] = f
    sf, of, em, bf, gf = d["self_feat"], d["other_feat"], d["exists_mask"], d["ball_feat"], d["global_feat"]
    sat, oat = _ai_types(d)
    with torch.no_grad():
        dh = t.decision_net(sf, of, em, bf, gf, sat, oat)
        eh = t.execution_net(sf, of, em, bf, gf, dh, sat, oat)
        vals = [t._get_value(d)]
        if t.value_net is not None:
            vals.append(float(t.value_net(sf, of, em, bf, gf, dh, sat, oat, value_only=True)))
    return torch.cat([dh.move_logit.flatten(), dh.latent_vector.flatten(), eh.kick_logit.flatten(),
                      eh.sprint_logit.flatten(), torch.tensor(vals, dtype=torch.float32)])


@pytest.mark.parametrize("separate_value_net", [True, False])
def test_remap_reproduces_the_old_outputs_exactly_on_an_exactly_linear_change(separate_value_net):
    t = _trainer(separate_value_net)
    import copy
    remapped = copy.deepcopy(t)
    a, b = 0.07, 0.8
    report = remap_time_remaining_input(remapped, a=a, b=b)
    layers = {(r["network"], r["layer"]) for r in report}
    assert ("execution_net", "value_head.1") in layers and ("decision_net", "global_mlp.0") in layers
    obs = _obs()
    for f_new in (0.1, 0.25, 0.4):
        old = _out(t, obs, a + b * f_new)
        new = _out(remapped, obs, f_new)
        assert float((old - new).abs().max()) < 1e-5, (f_new, float((old - new).abs().max()))
    assert not torch.allclose(_out(t, obs, 0.25), _out(remapped, obs, 0.25))
    assert remapped.true_time_input_ready


def test_training_gate_and_checkpoint_marker_for_the_time_input(monkeypatch, tmp_path):
    t = _trainer()
    t._check_task_id_inputs()                                  # fresh weights: fine
    t._save_checkpoint_to(tmp_path / "true.pt")
    ck = torch.load(tmp_path / "true.pt", map_location="cpu", weights_only=False)
    assert ck["true_time_input_ready"] is True
    del ck["true_time_input_ready"]
    torch.save(ck, tmp_path / "old.pt")                        # a checkpoint from before the fix
    fresh = _trainer()
    fresh.load_checkpoint(tmp_path / "old.pt")
    assert fresh.true_time_input_ready is False
    with pytest.raises(RuntimeError, match="remap-time-input"):
        fresh._check_task_id_inputs()
    remap_time_remaining_input(fresh, a=0.0, b=1.0)
    fresh.task_id_inputs_ready = True                          # (the task-id gate is tested elsewhere)
    fresh._check_task_id_inputs()
    fresh.load_checkpoint(tmp_path / "true.pt")
    assert fresh.true_time_input_ready is True


def test_ui_task_phase_is_the_scenario_phase():
    assert ui_task_phase("1v1") == 1 and ui_task_phase("1v2") == 2
    assert ui_task_phase("pass") is None          # phase 3 is a stub: its task id never trained
    assert ui_task_phase(None) is None and ui_task_phase("no_such_scenario") is None


def test_ui_checkpoint_picker_encodes_the_phase1_task_id():
    from footballcoach.ui import scenarios
    assert scenarios._picker_task_phase() == 1
