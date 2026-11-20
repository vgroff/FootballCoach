"""Role BC labels are world frame; the heads are canonical (observer attacks +x).

Regression (2026-11-19): distill_pretrain.attach_role_bc and the PPO BC loss
(ppo_update) fed the world-frame labels straight into the loss, so for a
right-team player -- the phase-2 keeper -- every x-signed target (move
direction, region, kick direction / spin) was mirrored: "step back onto my
line" was taught as "step out". The end-to-end test fits keeper rows from a
real phase-2 env with the distill role-BC loss and checks the REAL sampler's
world-frame move direction points where the rules keeper's does."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.obs.canonical import x_sign_of
from footballcoach.ai.ppo import bc as B
from footballcoach.ai.ppo.bc_sources import source_id
from footballcoach.ai.ppo.distill_pretrain import _student_forward, attach_role_bc, role_bc_loss
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

PHASE2 = PHASES_BY_ID[2]


@pytest.fixture(scope="module")
def keeper_rows():
    """(obs dict, world-frame labels) of rules-labelled neural-keeper
    decisions whose labelled move has a clear x component."""
    torch.manual_seed(0)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    ctrl = {"attacker": "rules", "defender": "rules", "goalkeeper": "neural"}
    rows = []
    for seed in range(150):
        env = build_role_env(PHASE2, controller_weights={r: {c: 1} for r, c in ctrl.items()})
        env.controller_overrides = ctrl
        env.sample_action_fn = trainer._sample_action
        env.scenario_kwargs = {**env.scenario_kwargs, "no_keeper_prob": 0.0}
        env.rng = random.Random(seed)
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _ = env.step()
            t = env.last_trainee_transition
            if env.trainee_player_id != "keeper" or t is None or t.get("bc_label") is None:
                continue
            lab = np.asarray(t["bc_label"], dtype=np.float32)
            if lab[B._I_VALID] > 0.5 and lab[B._I_EXEC_MOVE] > 0.5 and abs(lab[B._I_DIR_X]) > 0.5:
                rows.append((t["obs"], lab))
        if len(rows) >= 64:
            break
    assert len(rows) >= 12, f"only {len(rows)} keeper rows with an x-ward labelled move"
    obs = {k: torch.as_tensor(np.stack([np.asarray(r[0][k]) for r in rows])) for k in rows[0][0]}
    return obs, torch.tensor(np.stack([r[1] for r in rows]))


def test_keeper_rows_are_right_team(keeper_rows):
    obs, _ = keeper_rows
    assert bool((x_sign_of(obs["self_feat"]) < 0).all())


def test_attach_role_bc_mirrors_x_for_right_team_rows():
    from footballcoach.ai.obs.canonical import X_SIGN_FIELD_IDX
    from footballcoach.ai.obs.schema import PLAYER_FEATURE_DIM
    sf = torch.zeros(2, PLAYER_FEATURE_DIM)
    sf[0, X_SIGN_FIELD_IDX], sf[1, X_SIGN_FIELD_IDX] = 1.0, -1.0
    labels = torch.zeros(2, B.BC_LABEL_DIM)
    labels[:, B._I_DIR_X], labels[:, B._I_DIR_Y] = 0.8, 0.6
    data = {"returns": torch.zeros(2), "kl_w": torch.ones(2), "obs": {"self_feat": sf}}
    sid = source_id(2, "goalkeeper")
    batch = {"bc_labels": labels, "bc_sources": torch.tensor([sid, sid])}
    attach_role_bc(data, batch, {"goalkeeper": 1.0})
    assert data["bc_labels"][0, B._I_DIR_X] == pytest.approx(0.8)       # left team: unchanged
    assert data["bc_labels"][1, B._I_DIR_X] == pytest.approx(-0.8)      # right team: x mirrored
    assert data["bc_labels"][1, B._I_DIR_Y] == pytest.approx(0.6)


def test_role_bc_teaches_the_keeper_the_rules_world_direction(keeper_rows):
    obs, world_labels = keeper_rows
    n = len(world_labels)
    torch.manual_seed(1)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    sid = source_id(2, "goalkeeper")
    data = {"returns": torch.zeros(n), "kl_w": torch.ones(n), "obs": obs}
    attach_role_bc(data, {"bc_labels": world_labels, "bc_sources": torch.full((n,), sid)}, {"goalkeeper": 1.0})
    params = [p for p in list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())
              if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    mb = {"bc_w": data["bc_w"], "bc_labels": data["bc_labels"]}
    for _ in range(150):
        d, e = _student_forward(trainer, obs)
        loss, _ = role_bc_loss(trainer, d, e, mb, 0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()

    cos = []
    for k in range(n):
        out = trainer._sample_action({key: v[k] for key, v in obs.items()}, deterministic=True)
        world = np.asarray(out[4]["move_direction"], dtype=np.float64)[:2]
        target = world_labels[k, B._I_DIR_X:B._I_DIR_Y + 1].numpy().astype(np.float64)
        cos.append(float(world @ target / (np.linalg.norm(world) * np.linalg.norm(target) + 1e-9)))
    assert np.median(cos) > 0.9, f"sampled world move vs rules world move: median cos {np.median(cos):.2f}"


def test_direction_moving_only_ignores_standing_rows():
    """bc_loss_from_tensor(direction_moving_only=True) fits the move direction
    only on rows whose label moves; a standing row's direction (the arrival
    point the rules keeper is already on) adds nothing."""
    torch.manual_seed(2)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    from footballcoach.ai.obs.schema import PLAYER_FEATURE_DIM  # noqa: F401
    from footballcoach.ai.ppo.bc import bc_loss_from_tensor
    obs = {k: v[:2] for k, v in _fake_obs(trainer).items()}
    d, e = _student_forward(trainer, obs)
    labels = torch.zeros(2, B.BC_LABEL_DIM)
    labels[:, B._I_VALID] = 1.0
    labels[:, B._I_DIR_X] = 1.0                      # both rows carry a direction
    labels[0, B._I_EXEC_MOVE] = 1.0                  # row 0 moves, row 1 stands
    kw = trainer._bc_loss_kwargs()
    _, all_rows = bc_loss_from_tensor(labels, d, e, **kw, return_breakdown=True)
    _, moving = bc_loss_from_tensor(labels, d, e, **kw, return_breakdown=True, direction_moving_only=True)
    standing_only = labels.clone(); standing_only[0, B._I_DIR_X] = 0.0
    _, none = bc_loss_from_tensor(standing_only, d, e, **kw, return_breakdown=True, direction_moving_only=True)
    assert none["direction"] == pytest.approx(0.0)
    assert moving["direction"] != pytest.approx(all_rows["direction"])


def _fake_obs(trainer):
    """A small batch of real phase-2 observations (any players), for head shapes."""
    ctrl = {"attacker": "rules", "defender": "rules", "goalkeeper": "neural"}
    env = build_role_env(PHASE2, controller_weights={r: {c: 1} for r, c in ctrl.items()})
    env.controller_overrides = ctrl
    env.sample_action_fn = trainer._sample_action
    env.scenario_kwargs = {**env.scenario_kwargs, "no_keeper_prob": 0.0}
    env.rng = random.Random(0)
    obs0 = env.reset(seed=3)
    rows = []
    for _ in range(4):
        env.step()
        t = env.last_trainee_transition
        if t is not None:
            rows.append(t["obs"])
    return {k: torch.as_tensor(np.stack([np.asarray(r[k]) for r in rows])) for k in rows[0]}
