"""ai/ppo/new_phase_inputs.py and the task ids in the training rows (always set since 2026-11-16).

White-box: the same match state encoded with and without a task id (or
player flag) is run through every network and the outputs compared."""
from __future__ import annotations

import dataclasses
import random

import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_env, build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.obs.schema import GlobalFeatures, PlayerFeatures
from footballcoach.ai.ppo.new_phase_inputs import init_new_phase_inputs
from footballcoach.ai.ppo.trainer_common import _ai_types

GK_COL = [f.name for f in dataclasses.fields(PlayerFeatures)].index("is_goalkeeper")
G_NAMES = [f.name for f in dataclasses.fields(GlobalFeatures)]
T0, T1 = G_NAMES.index("task_id_0"), G_NAMES.index("task_id_1")


@pytest.fixture
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(3)
    return PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)


def _obs(phase):
    env = build_env(PHASES_BY_ID[1])
    env.reset(seed=12)
    return {k: v.unsqueeze(0) for k, v in encode_observation(
        match=env.match, player_id="trainee", time_remaining_s=10.0, attack_defence_smoothed=0.5,
        rng=random.Random(0), phase=phase).to_torch_dict().items()}


def _outputs(trainer, obs):
    sf, of, em, bf, gf = obs["self_feat"], obs["other_feat"], obs["exists_mask"], obs["ball_feat"], obs["global_feat"]
    sat, oat = _ai_types(obs)
    with torch.no_grad():
        d = trainer.decision_net(sf, of, em, bf, gf, sat, oat)
        e = trainer.execution_net(sf, of, em, bf, gf, d, sat, oat)
        v = trainer.value_net(sf, of, em, bf, gf, d, sat, oat, value_only=True)
    return torch.cat([d.move_logit.flatten(), d.latent_vector.flatten(), e.move_direction.flatten(),
                      e.kick_logit.flatten(), v.flatten()])


def test_zeroed_task_ids_have_exactly_no_effect(trainer):
    none, p1, p2 = _obs(None), _obs(1), _obs(2)
    assert not torch.equal(_outputs(trainer, none), _outputs(trainer, p1))
    report = init_new_phase_inputs(trainer, task_phases=[1, 2])
    assert {(r["network"], r["layer"], r["column"]) for r in report} >= {
        (n, "global_mlp.0.weight", c) for n in ("decision_net", "execution_net", "value_net")
        for c in ("task_id_0", "task_id_1")}
    assert all(r["action"] == "zero" for r in report)
    # Bit-identical: 1.0 x 0.0 adds an exact zero to the sum.
    assert torch.equal(_outputs(trainer, none), _outputs(trainer, p1))
    assert torch.equal(_outputs(trainer, none), _outputs(trainer, p2))
    assert trainer.task_id_inputs_ready


def test_a_zeroed_task_id_column_still_learns(trainer):
    """One real optimizer step on a loss through the networks: the zeroed
    columns get a gradient (input 1 on these rows) and move off zero."""
    init_new_phase_inputs(trainer, task_phases=[1])
    obs = _obs(1)
    cols = [(n, w) for n, w in trainer.decision_net.named_parameters() if n.endswith("global_mlp.0.weight")]
    assert cols and float(cols[0][1].detach()[:, T0].abs().max()) == 0.0
    opt = torch.optim.SGD(trainer.decision_net.parameters(), lr=0.1)
    sf, of, em, bf, gf = obs["self_feat"], obs["other_feat"], obs["exists_mask"], obs["ball_feat"], obs["global_feat"]
    sat, oat = _ai_types(obs)
    d = trainer.decision_net(sf, of, em, bf, gf, sat, oat)
    loss = d.move_logit.sum() + d.latent_vector.pow(2).sum()
    opt.zero_grad()
    loss.backward()
    grad = cols[0][1].grad[:, T0]
    assert float(grad.abs().max()) > 0.0
    assert float(cols[0][1].grad[:, T1].abs().max()) == 0.0   # input 0 on these rows: no gradient
    opt.step()
    assert float(cols[0][1].detach()[:, T0].abs().max()) > 0.0


def test_zeroed_player_flag_has_no_effect(trainer):
    obs = _obs(1)
    flagged = {k: v.clone() for k, v in obs.items()}
    flagged["self_feat"][..., GK_COL] = 1.0
    flagged["other_feat"][..., GK_COL] = 1.0
    assert not torch.equal(_outputs(trainer, obs), _outputs(trainer, flagged))
    report = init_new_phase_inputs(trainer, zero_player_features=["is_goalkeeper"])
    assert {r["layer"] for r in report} == {"self_mlp.0.weight", "entity_encoder.per_entity_mlp.0.weight"}
    assert torch.equal(_outputs(trainer, obs), _outputs(trainer, flagged))


def test_value_head_hidden_layer_is_never_touched(trainer):
    before = {n: p.detach().clone() for n, p in trainer.execution_net.named_parameters() if n.startswith("value_head")}
    init_new_phase_inputs(trainer, task_phases=[1, 2], zero_player_features=["is_goalkeeper"])
    for n, p in trainer.execution_net.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), n


def test_bad_arguments_raise(trainer):
    with pytest.raises(ValueError, match="task-id slot"):
        init_new_phase_inputs(trainer, task_phases=[21])
    with pytest.raises(ValueError):
        init_new_phase_inputs(trainer, zero_player_features=["no_such_feature"])


# --- task ids in the training rows -----------------------------------------

def _row_and_env_task_ids(trainer, phase_id):
    ph = PHASES_BY_ID[phase_id]
    env = build_env(ph) if not ph.is_role_based else build_role_env(
        ph, controller_weights={"attacker": {"neural": 1}, "defender": {"neural": 1}, "goalkeeper": {"rules": 1}})
    env.sample_action_fn = trainer._sample_action
    env_obs = env.reset(seed=5).to_torch_dict()["global_feat"].flatten()
    env.step()
    rows = [env.last_trainee_transition["obs"]["global_feat"].flatten()]
    rows += [torch.as_tensor(s["obs"]["global_feat"]).flatten() for s in env.last_secondary_results]
    return [(float(r[T0]), float(r[T1])) for r in rows], (float(env_obs[T0]), float(env_obs[T1]))


@pytest.mark.parametrize("phase_id,expected", [(1, (1.0, 0.0)), (2, (0.0, 1.0))])
def test_rows_always_carry_the_phase_task_id(trainer, phase_id, expected):
    rows, env_ids = _row_and_env_task_ids(trainer, phase_id)
    assert env_ids == expected and all(r == expected for r in rows)
    if phase_id == 2:
        assert len(rows) == 2                               # the secondary's rows too


def test_training_refuses_unprepared_weights(trainer):
    trainer.task_id_inputs_ready = False
    with pytest.raises(RuntimeError, match="init-new-phase-inputs"):
        trainer._check_task_id_inputs()
    init_new_phase_inputs(trainer, task_phases=[1])
    trainer._check_task_id_inputs()


def test_checkpoint_round_trip_of_the_marker(trainer, tmp_path):
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    init_new_phase_inputs(trainer, task_phases=[1])
    trainer._save_checkpoint_to(tmp_path / "ready.pt")
    trainer.task_id_inputs_ready = False
    trainer._save_checkpoint_to(tmp_path / "unready.pt")
    ck = torch.load(tmp_path / "ready.pt", map_location="cpu", weights_only=False)
    del ck["task_id_inputs_ready"]
    torch.save(ck, tmp_path / "old.pt")                   # a checkpoint from before the marker existed
    fresh = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    for name, want in [("ready.pt", True), ("unready.pt", False), ("old.pt", False)]:
        fresh.load_checkpoint(tmp_path / name)
        assert fresh.task_id_inputs_ready is want, name
