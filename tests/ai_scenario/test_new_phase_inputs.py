"""ai/ppo/new_phase_inputs.py: after init_new_phase_inputs, a new phase's
observations are processed exactly like the source phase's, and a zeroed
player flag has no effect, on every network (white-box: same match state
encoded both ways, outputs compared)."""
from __future__ import annotations

import dataclasses
import random

import pytest
import torch

from footballcoach.ai.curriculum.envs import build_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.obs.schema import PlayerFeatures
from footballcoach.ai.ppo.new_phase_inputs import init_new_phase_inputs
from footballcoach.ai.ppo.trainer_common import _ai_types

GK_COL = [f.name for f in dataclasses.fields(PlayerFeatures)].index("is_goalkeeper")


@pytest.fixture
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(3)
    return PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)


def _obs(phase: int):
    env = build_env(PHASES_BY_ID[1])
    env.reset(seed=12)
    m = env.match
    return {k: v.unsqueeze(0) for k, v in encode_observation(
        match=m, player_id="trainee", time_remaining_s=10.0, attack_defence_smoothed=0.5,
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


def test_new_phase_observations_match_the_source_phase_after_init(trainer):
    p1, p2 = _obs(1), _obs(2)
    assert not torch.equal(p1["global_feat"], p2["global_feat"])
    gap_before = float((_outputs(trainer, p1) - _outputs(trainer, p2)).abs().max())
    assert gap_before > 1e-3
    report = init_new_phase_inputs(trainer, from_phase=1, to_phases=[2])
    assert {(r["network"], r["layer"]) for r in report} >= {
        ("decision_net", "global_mlp.0.weight"), ("execution_net", "global_mlp.0.weight"),
        ("value_net", "global_mlp.0.weight")}
    assert all(r["column"] == "task_id_1" and r["action"] == "copy task_id_0" for r in report)
    # Equal up to float rounding only: the 1.0 sits in a different column, so
    # the matmul adds the same weight at a different point of its sum.
    gap_after = float((_outputs(trainer, p1) - _outputs(trainer, p2)).abs().max())
    assert gap_after < 1e-6, (gap_before, gap_after)


def test_zeroed_player_flag_has_no_effect(trainer):
    obs = _obs(1)
    flagged = {k: v.clone() for k, v in obs.items()}
    flagged["self_feat"][..., GK_COL] = 1.0
    flagged["other_feat"][..., GK_COL] = 1.0
    assert not torch.equal(_outputs(trainer, obs), _outputs(trainer, flagged))
    report = init_new_phase_inputs(trainer, from_phase=1, to_phases=[], zero_player_features=["is_goalkeeper"])
    assert {r["layer"] for r in report} == {"self_mlp.0.weight", "entity_encoder.per_entity_mlp.0.weight"}
    assert torch.equal(_outputs(trainer, obs), _outputs(trainer, flagged))


def test_value_head_hidden_layer_is_never_touched(trainer):
    before = {n: p.detach().clone() for n, p in trainer.execution_net.named_parameters() if n.startswith("value_head")}
    init_new_phase_inputs(trainer, from_phase=1, to_phases=[2], zero_player_features=["is_goalkeeper"])
    for n, p in trainer.execution_net.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), n


def test_bad_arguments_raise(trainer):
    with pytest.raises(ValueError, match="task-id slot"):
        init_new_phase_inputs(trainer, from_phase=1, to_phases=[21])
    with pytest.raises(ValueError):
        init_new_phase_inputs(trainer, from_phase=1, to_phases=[2], zero_player_features=["no_such_feature"])
