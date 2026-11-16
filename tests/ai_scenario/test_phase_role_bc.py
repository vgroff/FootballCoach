"""Per-phase, per-role BC (ai/ppo/bc_sources.py): config and source ids, the
role env labelling exactly the BC-enabled roles' neural players (trainee or
secondary) with their source id, the batched worker storing it, and the PPO
update weighting each row by its own source's coefficients only."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.ppo import bc_sources
from footballcoach.ai.ppo.bc import _I_VALID, BC_LABEL_DIM, bc_loss_from_tensor

PHASE2 = PHASES_BY_ID[2]
KICK_ONLY = {"attacker": {"kick": {"coeff_start": 0.6, "coeff_end": 0.0, "anneal_fraction": 0.5}}}


@pytest.fixture
def bc_config(monkeypatch):
    cfg = load_ai_config()["curriculum"]["phase2"]

    def _set(value):
        monkeypatch.setitem(cfg, "bc", value)
    _set({})
    return _set


@pytest.fixture(scope="module")
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    t.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    return t


# ----------------------------------------------------------------- config

def test_no_bc_configured_means_no_sources(bc_config):
    assert bc_sources.sources() == []
    assert bc_sources.source_id(2, "attacker") == 0


def test_sources_and_their_schedules(bc_config):
    bc_config({**KICK_ONLY, "goalkeeper": {"other": {"coeff_start": 1.0, "coeff_end": 1.0, "anneal_fraction": 1.0}}})
    assert bc_sources.sources() == [(2, "attacker"), (2, "goalkeeper")]
    assert bc_sources.source_id(2, "attacker") == 1 and bc_sources.source_id(2, "defender") == 0
    assert bc_sources.describe(1) == "phase2_1v2/attacker"
    t0 = bc_sources.coeff_table(0.0, (0.1, 0.2, 0.3))                # columns: bc.GROUP_ROW_COLUMNS
    assert t0.tolist()[0] == pytest.approx([0.1, 0.2, 0.2, 0.2, 0.3])  # row 0: the global schedules
    assert t0.tolist()[1] == pytest.approx([0.0, 0.6, 0.6, 0.6, 0.0])  # attacker: kick (all three parts)
    assert t0.tolist()[2] == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])
    assert bc_sources.coeff_table(0.25, (0, 0, 0))[1, 1].item() == pytest.approx(0.3)   # halfway through its anneal
    assert bc_sources.coeff_table(0.9, (0, 0, 0))[1, 1].item() == pytest.approx(0.0)


def test_kick_parts_override_kick(bc_config):
    const = lambda c: {"coeff_start": c, "coeff_end": c, "anneal_fraction": 1.0}
    bc_config({"attacker": {"kick": const(0.3), "kick_power": const(1.0), "kick_gate": const(0.0)}})
    from footballcoach.ai.ppo.bc import GROUP_ROW_COLUMNS
    row = dict(zip(GROUP_ROW_COLUMNS, bc_sources.coeff_table(0.0, (0, 0, 0))[1].tolist()))
    assert row == pytest.approx({"other": 0.0, "kick_gate": 0.0, "kick_power": 1.0, "kick_aim": 0.3, "tackle": 0.0})
    bc_config({"attacker": {"kick_power": const(0.5)}})                  # a part alone, no "kick"
    row = dict(zip(GROUP_ROW_COLUMNS, bc_sources.coeff_table(0.0, (0, 0, 0))[1].tolist()))
    assert row["kick_power"] == pytest.approx(0.5) and row["kick_gate"] == 0.0 and row["kick_aim"] == 0.0


def test_a_held_schedule(bc_config):
    bc_config({"attacker": {"kick": {"coeff_start": 1.0, "coeff_end": 0.2, "hold_fraction": 0.5,
                                     "anneal_fraction": 1.0}}})
    kick = [bc_sources.coeff_table(p, (0, 0, 0))[1, 1].item() for p in (0.0, 0.49, 0.5, 0.75, 1.0, 1.2)]
    assert kick == pytest.approx([1.0, 1.0, 1.0, 0.6, 0.2, 0.2])


@pytest.mark.parametrize("bad,match", [
    ({"striker": {"kick": KICK_ONLY["attacker"]["kick"]}}, "unknown role"),
    ({"attacker": {"heading": KICK_ONLY["attacker"]["kick"]}}, "unknown BC group"),
    ({"attacker": {"kick": {"coeff_start": 1.0}}}, "schedule needs"),
    ({"attacker": {"kick": {**KICK_ONLY["attacker"]["kick"], "hold_fraction": 0.7}}}, "hold_fraction"),
    ({"attacker": {"kick": {**KICK_ONLY["attacker"]["kick"], "hold_frac": 0.2}}}, "unknown schedule key"),
])
def test_bad_config_is_refused(bc_config, bad, match):
    bc_config(bad)
    with pytest.raises(ValueError, match=match):
        bc_sources.sources()


# ---------------------------------------------------------------- the env

def _episode_rows(trainer, controllers, seed=4, steps=12):
    env = build_role_env(PHASE2, controller_weights=controllers)
    env.sample_action_fn = trainer._sample_action
    env.rng = random.Random(seed)
    env.reset(seed=seed)
    trainee, secondary = [], []
    for _ in range(steps):
        _, _, done, _ = env.step()
        if env.last_trainee_transition is not None:
            trainee.append(env.last_trainee_transition)
        secondary += env.last_secondary_results
        if done:
            break
    return trainee, secondary


NEURAL_ATK_DEF = {"attacker": {"neural": 1}, "defender": {"neural": 1}, "goalkeeper": {"rules": 1}}


def test_only_bc_roles_get_labels_and_their_source(bc_config, trainer):
    bc_config(KICK_ONLY)
    trainee, secondary = _episode_rows(trainer, NEURAL_ATK_DEF)
    assert trainee and secondary
    for tr in trainee:                                     # the attacker (first neural role)
        assert tr["bc_source"] == 1
        assert tr["bc_label"] is not None and tr["bc_label"][_I_VALID] == 1.0
    for sec in secondary:                                  # the defender: no BC configured
        assert sec["bc_source"] == 0 and sec["bc_label"] is None


def test_a_secondary_in_a_bc_role_is_labelled(bc_config, trainer):
    bc_config({**KICK_ONLY, "defender": {"tackle": {"coeff_start": 0.2, "coeff_end": 0.2, "anneal_fraction": 1.0}}})
    _, secondary = _episode_rows(trainer, NEURAL_ATK_DEF)
    assert secondary
    defender_id = bc_sources.source_id(2, "defender")
    for sec in secondary:
        assert sec["bc_source"] == defender_id
        assert sec["bc_label"] is not None and sec["bc_label"][_I_VALID] == 1.0


def test_no_bc_configured_leaves_phase2_rows_unlabelled(bc_config, trainer):
    trainee, secondary = _episode_rows(trainer, NEURAL_ATK_DEF)
    assert all(tr.get("bc_source", 0) == 0 and tr["bc_label"] is None for tr in trainee)
    assert all(s["bc_source"] == 0 and s["bc_label"] is None for s in secondary)


def test_the_batched_worker_stores_sources_and_secondary_labels(bc_config, trainer):
    from footballcoach.ai.ppo.batched_rollout_worker import BatchedEnvGroup
    bc_config({**KICK_ONLY, "defender": {"tackle": {"coeff_start": 0.2, "coeff_end": 0.2, "anneal_fraction": 1.0}}})
    envs = []
    for _ in range(2):
        e = build_role_env(PHASE2, controller_weights=NEURAL_ATK_DEF)
        e.sample_action_fn = trainer._sample_action
        e.require_neural_player = True
        envs.append(e)
    group = BatchedEnvGroup(envs, trainer, seeds=[11, 12])
    rows = 0
    for res in group.collect(300):
        b = res["buffer"]
        for tid, src, lab in zip(b.track_ids, b.bc_sources, b.bc_labels):
            want = 1 if tid == "trainee" else bc_sources.source_id(2, "defender")
            assert src == want and lab[_I_VALID] == 1.0
            rows += 1
        t = b.as_tensors([0.0] * len(b.rewards), [0.0] * len(b.rewards))
        assert t["bc_sources"].dtype == torch.long and t["bc_sources"].shape[0] == len(b.rewards)
    assert rows > 0


# --------------------------------------------------------------- the loss

def _labels_and_heads(trainer, n=24, seed=0):
    from footballcoach.ai.ppo.trainer_common import _ai_types
    env = build_role_env(PHASE2, controller_weights={"attacker": {"neural": 1}, "defender": {"rules": 1},
                                                    "goalkeeper": {"rules": 1}})
    env.sample_action_fn = trainer._sample_action
    env.reset(seed=seed)
    from footballcoach.ai.obs.encoder import encode_observation
    obs = encode_observation(match=env.match, player_id="attacker", time_remaining_s=10.0,
                             attack_defence_smoothed=0.5, rng=random.Random(0), phase=2).to_torch_dict()
    batch = {k: v.unsqueeze(0).repeat(n, *([1] * v.dim())) for k, v in obs.items()}
    rng = np.random.default_rng(seed)
    labels = torch.from_numpy(rng.uniform(0, 1, (n, BC_LABEL_DIM)).astype(np.float32))
    labels[:, _I_VALID] = 1.0
    labels[:, 12] = torch.from_numpy((rng.uniform(0, 1, n) > 0.5).astype(np.float32))   # kick_this_tick
    sf, of, em, bf, gf = batch["self_feat"], batch["other_feat"], batch["exists_mask"], batch["ball_feat"], batch["global_feat"]
    sat, oat = _ai_types(batch)
    d = trainer.decision_net(sf, of, em, bf, gf, sat, oat)
    e = trainer.execution_net(sf, of, em, bf, gf, d, sat, oat)
    return labels, trainer._bc_heads_for_loss(d), e


def _per_row_weighted(trainer, labels, d, e, coeffs):
    _, _, split = bc_loss_from_tensor(labels, d, e, **trainer._bc_loss_kwargs(), return_breakdown=True,
                                      split_kick=True, split_tackle=True, return_group_rows=True)
    valid = split["valid"]
    return ((coeffs * split["rows"]).sum(dim=1) * valid).sum() / valid.sum().clamp(min=1)


def test_per_row_coefficients_on_source_0_equal_the_old_weighted_loss(trainer):
    labels, d, e = _labels_and_heads(trainer)
    c = (0.3, 0.7, 0.2)
    _, _, split = bc_loss_from_tensor(labels, d, e, **trainer._bc_loss_kwargs(), return_breakdown=True,
                                      split_kick=True, split_tackle=True)
    old = c[0] * split["other_group_loss"] + c[1] * split["kick_group_loss"] + c[2] * split["tackle_group_loss"]
    per_column = [c[0], c[1], c[1], c[1], c[2]]                       # kick on all three kick parts
    new = _per_row_weighted(trainer, labels, d, e, torch.tensor([per_column]).repeat(labels.shape[0], 1))
    assert float(new) == pytest.approx(float(old), rel=1e-5)


def test_a_phase2_coefficient_puts_no_gradient_through_phase1_rows(trainer):
    labels, d, e = _labels_and_heads(trainer)
    n = labels.shape[0]
    phase2 = torch.zeros(n, dtype=torch.bool)
    phase2[: n // 2] = True
    table = torch.tensor([[0.0] * 5, [0.0, 0.8, 0.8, 0.8, 0.0]])     # global 0, attacker kick 0.8
    coeffs = table[phase2.long()]
    loss = _per_row_weighted(trainer, labels, d, e, coeffs)
    # Changing a phase-1 row's labels must not change the loss at all.
    labels2 = labels.clone()
    labels2[~phase2, 12] = 1.0 - labels2[~phase2, 12]
    labels2[~phase2, 18:21] = -labels2[~phase2, 18:21]
    assert float(_per_row_weighted(trainer, labels2, d, e, coeffs)) == pytest.approx(float(loss), rel=1e-6)
    # And the reverse: the phase-2 rows' kick labels do move it.
    labels3 = labels.clone()
    labels3[phase2, 12] = 1.0 - labels3[phase2, 12]
    assert float(_per_row_weighted(trainer, labels3, d, e, coeffs)) != pytest.approx(float(loss), rel=1e-6)


def test_the_kick_part_columns_sum_to_the_kick_group(trainer):
    labels, d, e = _labels_and_heads(trainer)
    _, _, split = bc_loss_from_tensor(labels, d, e, **trainer._bc_loss_kwargs(), return_breakdown=True,
                                      split_kick=True, split_tackle=True, return_group_rows=True)
    rows, valid = split["rows"], split["valid"]
    assert rows.shape[1] == 5
    assert float(rows[valid, 1:4].sum(dim=1).mean()) == pytest.approx(float(split["kick_group_loss"]), rel=1e-5)
    assert float(rows[valid, 0].mean()) == pytest.approx(float(split["other_group_loss"]), rel=1e-5)
    assert float(rows[valid, 4].mean()) == pytest.approx(float(split["tackle_group_loss"]), rel=1e-5)


def test_augmentation_carries_bc_sources():
    from footballcoach.ai.obs.augment import augment_batch
    from footballcoach.ai.obs.schema import BALL_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM
    n = 3
    batch = {"obs/self_feat": torch.zeros(n, PLAYER_FEATURE_DIM),
             "obs/other_feat": torch.zeros(n, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM),
             "obs/exists_mask": torch.zeros(n, MAX_OTHER_PLAYERS), "obs/ball_feat": torch.zeros(n, BALL_FEATURE_DIM),
             "obs/global_feat": torch.zeros(n, GLOBAL_FEATURE_DIM)}
    batch.update({k: torch.zeros(n) for k in ("log_probs", "values", "rewards", "advantages", "returns", "dones")})
    batch["sample_weights"] = torch.ones(n)
    batch["bc_sources"] = torch.tensor([0, 1, 2])
    out = augment_batch(batch, n_slot_shuffles=2, rng=random.Random(0))
    reps = out["obs/self_feat"].shape[0] // n
    assert out["bc_sources"].tolist() == [0, 1, 2] * reps
