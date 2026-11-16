"""Per-phase teacher distillation (ai/ppo/distill.py, PPOTrainer._ppo_distill_loss):
the config, the phase rows read from the task-id one-hot, the KL to a frozen
teacher (zero against itself, only on the distilled phase's rows), and
means_only keeping the gradient off the spread parameters."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.ppo import distill
from footballcoach.ai.ppo.ppo_update import _UpdateStats
from footballcoach.ai.ppo.trainer_common import _ai_types

N = 8
N_PHASE1 = 5


@pytest.fixture
def distill_config(monkeypatch):
    cur = load_ai_config()["curriculum"]

    def _set(value):
        monkeypatch.setitem(cur, "distill", value)
    _set({})
    return _set


@pytest.fixture
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    t.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    return t


def _batch(trainer):
    """N copies of a phase-2 attacker observation; the first N_PHASE1 relabelled
    phase 1 through the task-id one-hot (what phase_row_mask reads)."""
    env = build_role_env(PHASES_BY_ID[2], controller_weights={"attacker": {"neural": 1}, "defender": {"rules": 1},
                                                             "goalkeeper": {"rules": 1}})
    env.sample_action_fn = trainer._sample_action
    env.reset(seed=3)
    obs = encode_observation(match=env.match, player_id="attacker", time_remaining_s=10.0,
                             attack_defence_smoothed=0.5, rng=random.Random(0), phase=2,
                             episode_length_s=15.0).to_torch_dict()
    b = {k: v.unsqueeze(0).repeat(N, *([1] * v.dim())).clone() for k, v in obs.items()}
    c1, c2 = distill.task_id_column(1), distill.task_id_column(2)
    b["global_feat"][:N_PHASE1, c1], b["global_feat"][:N_PHASE1, c2] = 1.0, 0.0
    actions = {"exec_move": torch.ones(N, 1), "kick": torch.ones(N, 1),
               "pass_": torch.zeros(N, 1), "tackle": torch.zeros(N, 1), "mark": torch.zeros(N, 1)}
    return b, actions


def _heads(trainer, b):
    sat, oat = _ai_types(b)
    d = trainer.decision_net(b["self_feat"], b["other_feat"], b["exists_mask"], b["ball_feat"], b["global_feat"], sat, oat)
    e = trainer.execution_net(b["self_feat"], b["other_feat"], b["exists_mask"], b["ball_feat"], b["global_feat"], d, sat, oat)
    return d, e


def _teacher_of(trainer, tmp_path, name="teacher.pt"):
    path = tmp_path / name
    trainer._save_checkpoint_to(path)
    return distill.teacher_nets(str(path), trainer.separate_value_net, trainer.device)


def _loss(trainer, nets, b, actions, means_only=True, phase_id=1):
    ctx = SimpleNamespace(distill_terms=[{"phase_id": phase_id, "coeff": 1.0, "means_only": means_only, "nets": nets}])
    stats = _UpdateStats()
    d, e = _heads(trainer, b)
    loss = trainer._ppo_distill_loss(ctx, stats, b, d, e, actions, b["exists_mask"])
    return loss, stats


def _perturb_means(trainer, scale=0.05):
    """Noise on every execution-net weight except the global spreads."""
    spreads = {"move_dir_log_kappa", "kick_dir_log_kappa", "kick_dir_z_log_std", "kick_power_log_std", "kick_spin_log_std"}
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in trainer.execution_net.named_parameters():
            if name.split(".")[-1] not in spreads:
                p.add_(scale * torch.randn(p.shape, generator=g))


# ----------------------------------------------------------------- config

def test_no_config_means_off(distill_config):
    assert distill.distill_config() == {}
    assert distill.active_terms(0.0, False, "cpu") is None


@pytest.mark.parametrize("bad,match", [
    ({"9": {"teacher": "x", "coeff_start": 1, "coeff_end": 1, "anneal_fraction": 1}}, "unknown phase"),
    ({"1": {"teacher": "x", "coeff_start": 1}}, "needs"),
    ({"1": {"teacher": "x", "coeff_start": 1, "coeff_end": 1, "anneal_fraction": 1, "coef": 2}}, "unknown key"),
])
def test_bad_config_is_refused(distill_config, bad, match):
    distill_config(bad)
    with pytest.raises(ValueError, match=match):
        distill.distill_config()


def test_phase_rows_come_from_the_task_id(trainer):
    b, _ = _batch(trainer)
    assert distill.phase_row_mask(b["global_feat"], 1).tolist() == [True] * N_PHASE1 + [False] * (N - N_PHASE1)
    assert distill.phase_row_mask(b["global_feat"], 2).tolist() == [False] * N_PHASE1 + [True] * (N - N_PHASE1)


# --------------------------------------------------------------- the loss

def test_zero_against_itself_positive_once_the_student_moves(trainer, tmp_path):
    b, actions = _batch(trainer)
    nets = _teacher_of(trainer, tmp_path)
    loss, stats = _loss(trainer, nets, b, actions)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    _perturb_means(trainer)
    loss, stats = _loss(trainer, nets, b, actions)
    assert float(loss) > 1e-4
    (kl, by_head), = stats.distill_by_phase[1]
    assert kl == pytest.approx(float(loss)) and by_head["kick"] > 0.0


def test_agreement_is_exact_against_itself_and_not_after_a_move(trainer, tmp_path):
    from footballcoach.ai.ppo.ppo_update import _DISTILL_AGREE_KEYS
    b, actions = _batch(trainer)
    nets = _teacher_of(trainer, tmp_path)
    _, stats = _loss(trainer, nets, b, actions)
    same = dict(zip(_DISTILL_AGREE_KEYS, stats.distill_agree_by_phase[1][0]))
    assert same["move_dir_err_deg"] == pytest.approx(0.0, abs=0.05)
    assert same["p_kick_student"] == pytest.approx(same["p_kick_teacher"], abs=1e-6)
    assert same["power_diff"] == pytest.approx(0.0, abs=1e-6)
    assert same["exec_move_agree"] == 1.0 and same["sprint_agree"] == 1.0
    _perturb_means(trainer, scale=0.2)
    _, stats = _loss(trainer, nets, b, actions)
    moved = dict(zip(_DISTILL_AGREE_KEYS, stats.distill_agree_by_phase[1][0]))
    assert moved["move_dir_err_deg"] > 0.1 and moved["power_diff"] > 1e-4
    assert moved["p_kick_student"] != pytest.approx(moved["p_kick_teacher"], abs=1e-4)


def test_only_the_distilled_phase_rows_count(trainer, tmp_path):
    b, actions = _batch(trainer)
    nets = _teacher_of(trainer, tmp_path)
    _perturb_means(trainer)
    loss_p1, _ = _loss(trainer, nets, b, actions)
    # The same rows all relabelled phase 2: nothing to distil.
    b2 = {k: v.clone() for k, v in b.items()}
    b2["global_feat"][:, distill.task_id_column(1)] = 0.0
    b2["global_feat"][:, distill.task_id_column(2)] = 1.0
    loss_none, stats = _loss(trainer, nets, b2, actions)
    assert float(loss_p1) > 1e-4 and float(loss_none) == 0.0 and not stats.distill_by_phase


def test_the_mean_is_over_the_phase_rows_alone(trainer, tmp_path):
    """Phase-2 rows made different (so their KL differs): the mixed batch's
    loss equals the loss of its phase-1 rows on their own."""
    b, actions = _batch(trainer)
    g = torch.Generator().manual_seed(2)
    b["self_feat"][N_PHASE1:] += 0.5 * torch.randn(b["self_feat"][N_PHASE1:].shape, generator=g)
    nets = _teacher_of(trainer, tmp_path)
    _perturb_means(trainer)
    mixed, _ = _loss(trainer, nets, b, actions)
    sub = {k: v[:N_PHASE1] for k, v in b.items()}
    alone, _ = _loss(trainer, nets, sub, {k: v[:N_PHASE1] for k, v in actions.items()})
    assert float(mixed) == pytest.approx(float(alone), rel=1e-5)
    as_phase2, _ = _loss(trainer, nets, b, actions, phase_id=2)       # the other rows: a different number
    assert float(as_phase2) != pytest.approx(float(alone), rel=1e-3)


def test_means_only_keeps_the_gradient_off_the_spreads(trainer, tmp_path):
    b, actions = _batch(trainer)
    nets = _teacher_of(trainer, tmp_path)
    _perturb_means(trainer)
    with torch.no_grad():
        trainer.execution_net.move_dir_log_kappa.add_(-0.7)       # the student's own, relaxed spread
    en = trainer.execution_net
    for means_only, want_grad in ((True, False), (False, True)):
        en.zero_grad(set_to_none=True)
        loss, _ = _loss(trainer, nets, b, actions, means_only=means_only)
        loss.backward()
        g = en.move_dir_log_kappa.grad
        has = g is not None and float(g.abs().sum()) > 0.0
        assert has is want_grad, (means_only, g)
        trunk_g = sum(float(p.grad.abs().sum()) for n, p in en.named_parameters()
                      if p.grad is not None and n.split(".")[-1] not in ("move_dir_log_kappa",))
        assert trunk_g > 0.0


def test_active_terms_follow_the_schedule(distill_config, trainer, tmp_path):
    path = tmp_path / "t.pt"
    trainer._save_checkpoint_to(path)
    distill_config({"1": {"teacher": str(path), "coeff_start": 2.0, "coeff_end": 0.0, "anneal_fraction": 0.5}})
    (t0,) = distill.active_terms(0.0, trainer.separate_value_net, trainer.device)
    assert t0["phase_id"] == 1 and t0["coeff"] == pytest.approx(2.0) and t0["means_only"] is True
    assert distill.active_terms(0.25, trainer.separate_value_net, trainer.device)[0]["coeff"] == pytest.approx(1.0)
    assert distill.active_terms(0.9, trainer.separate_value_net, trainer.device) is None   # annealed to 0
