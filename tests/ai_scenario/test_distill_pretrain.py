"""Multi-teacher distillation pretraining (ai/ppo/distill_pretrain.py): the
task-id swap, the real / counterfactual copies, each row routed to its target
teacher, the teacher-probability sub-head weights, and the loss (zero against
itself, value only on real rows)."""
from __future__ import annotations

import random

import pytest
import torch

from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.ppo import distill
from footballcoach.ai.ppo import distill_pretrain as dp

N = 6
N_PHASE1 = 4


@pytest.fixture
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    t.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    return t


def _obs(trainer):
    env = build_role_env(PHASES_BY_ID[2], controller_weights={"attacker": {"neural": 1}, "defender": {"rules": 1},
                                                             "goalkeeper": {"rules": 1}})
    env.sample_action_fn = trainer._sample_action
    env.reset(seed=3)
    o = encode_observation(match=env.match, player_id="attacker", time_remaining_s=10.0,
                           attack_defence_smoothed=0.5, rng=random.Random(0), phase=2,
                           episode_length_s=15.0).to_torch_dict()
    b = {k: v.unsqueeze(0).repeat(N, *([1] * v.dim())).clone() for k, v in o.items()}
    g = torch.Generator().manual_seed(5)
    b["self_feat"] += 0.3 * torch.randn(b["self_feat"].shape, generator=g)   # rows differ
    c1, c2 = distill.task_id_column(1), distill.task_id_column(2)
    b["global_feat"][:N_PHASE1, c1], b["global_feat"][:N_PHASE1, c2] = 1.0, 0.0
    return b


def _teacher(trainer, tmp_path, name, perturb=0.0):
    import copy
    t = copy.deepcopy(trainer)
    if perturb:
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for n_, p in t.execution_net.named_parameters():
                if not n_.endswith("log_kappa") and not n_.endswith("log_std"):
                    p.add_(perturb * torch.randn(p.shape, generator=g))
    path = tmp_path / name
    t._save_checkpoint_to(path)
    return distill.teacher_nets(str(path), trainer.separate_value_net, trainer.device)


def test_swap_task_ids_exchanges_the_two_columns(trainer):
    b = _obs(trainer)
    out = dp.swap_task_ids(b["global_feat"], 1, 2)
    assert distill.phase_row_mask(out, 2).tolist() == [True] * N_PHASE1 + [False] * (N - N_PHASE1)
    assert distill.phase_row_mask(out, 1).tolist() == [False] * N_PHASE1 + [True] * (N - N_PHASE1)
    others = [i for i in range(out.shape[1]) if i not in (distill.task_id_column(1), distill.task_id_column(2))]
    assert torch.equal(out[:, others], b["global_feat"][:, others])


def test_copies_real_and_counterfactual(trainer):
    b = _obs(trainer)
    ret = torch.arange(N, dtype=torch.float32)
    data = dp.build_copies(b, ret, (1, 2), counterfactual=True, counterfactual_weight=0.5)
    assert len(data["returns"]) == 2 * N
    real, cf = ~data["is_cf"], data["is_cf"]
    assert data["target_phase"][real].tolist() == [1] * N_PHASE1 + [2] * (N - N_PHASE1)
    assert data["target_phase"][cf].tolist() == [2] * N_PHASE1 + [1] * (N - N_PHASE1)
    assert data["value_w"][real].tolist() == [1.0] * N and data["value_w"][cf].tolist() == [0.0] * N
    assert data["kl_w"][cf].tolist() == [0.5] * N
    # Every copy's task id is its target phase (what that teacher sees).
    for pid in (1, 2):
        rows = data["target_phase"] == pid
        assert bool(distill.phase_row_mask(data["obs"]["global_feat"], pid)[rows].all())
    off = dp.build_copies(b, ret, (1, 2), counterfactual=False, counterfactual_weight=1.0)
    assert len(off["returns"]) == N and not bool(off["is_cf"].any())


def test_rows_without_either_task_id_are_refused(trainer):
    b = _obs(trainer)
    b["global_feat"][0, distill.task_id_column(1)] = 0.0
    b["global_feat"][0, distill.task_id_column(2)] = 0.0
    with pytest.raises(ValueError, match="exactly one"):
        dp.build_copies(b, torch.zeros(N), (1, 2), True, 1.0)


def test_each_row_gets_its_target_teachers_output(trainer, tmp_path):
    b = _obs(trainer)
    teachers = {1: _teacher(trainer, tmp_path, "t1.pt"), 2: _teacher(trainer, tmp_path, "t2.pt", perturb=0.2)}
    data = dp.build_copies(b, torch.zeros(N), (1, 2), True, 1.0)
    anchor = dp.teacher_anchor(trainer, data["obs"], data["target_phase"], teachers, 4)
    for pid, nets in teachers.items():
        rows = torch.nonzero(data["target_phase"] == pid).squeeze(-1)
        direct = trainer._ppg_snapshot_anchor({k: v[rows] for k, v in data["obs"].items()}, 64, nets=nets)
        assert torch.allclose(anchor["kick_logit"][rows], direct["kick_logit"], atol=1e-6)
        assert torch.allclose(anchor["move_direction"][rows], direct["move_direction"], atol=1e-6)
    # The two teachers really differ, so routing is what made those match.
    r1 = torch.nonzero(data["target_phase"] == 1).squeeze(-1)
    other = trainer._ppg_snapshot_anchor({k: v[r1] for k, v in data["obs"].items()}, 64, nets=teachers[2])
    assert not torch.allclose(anchor["kick_logit"][r1], other["kick_logit"], atol=1e-4)
    acts = dp.soft_actions(anchor, len(data["returns"]))
    assert torch.allclose(acts["kick"].reshape(-1), torch.sigmoid(anchor["kick_logit"]).reshape(-1))


def test_concat_and_history(trainer, tmp_path):
    b = _obs(trainer)
    same = _teacher(trainer, tmp_path, "agg.pt")
    parts = []
    for i in range(2):
        d = dp.build_copies(b, torch.full((N,), float(i)), (1, 2), True, 0.3)
        d["anchor"] = dp.teacher_anchor(trainer, d["obs"], d["target_phase"], {1: same, 2: same}, 64)
        d["actions"] = dp.soft_actions(d["anchor"], len(d["returns"]))
        parts.append(d)
    old = dp.as_history(parts[0])
    assert float(old["value_w"].sum()) == 0.0 and float(parts[0]["value_w"].sum()) == N   # original untouched
    both = dp.concat_data([old, parts[1]])
    assert len(both["returns"]) == 4 * N
    assert both["returns"][:2 * N].tolist() == [0.0] * (2 * N) and both["returns"][2 * N:].tolist() == [1.0] * (2 * N)
    assert both["obs"]["self_feat"].shape[0] == 4 * N and both["anchor"]["kick_logit"].shape[0] == 4 * N
    assert both["actions"]["kick"].shape[0] == 4 * N
    assert float(both["value_w"][:2 * N].sum()) == 0.0 and float(both["value_w"][2 * N:].sum()) == N
    assert dp.concat_data([parts[1]]) is parts[1]


def test_kick_only_counterfactuals(trainer, tmp_path):
    """Counterfactual rows targeting phase 2: kick heads from teacher 2, the
    rest from the row's real (phase-1) teacher; everything else untouched."""
    b = _obs(trainer)
    t1 = _teacher(trainer, tmp_path, "k1.pt")
    t2 = _teacher(trainer, tmp_path, "k2.pt", perturb=0.3)
    teachers = {1: t1, 2: t2}
    data = dp.build_copies(b, torch.zeros(N), (1, 2), True, 1.0)
    anchor = dp.teacher_anchor(trainer, data["obs"], data["target_phase"], teachers, 64)
    out = dp.kick_only_counterfactuals(trainer, data, anchor, teachers, [2], 64)
    sel = data["is_cf"] & (data["target_phase"] == 2)          # 1v1 states as task id 2
    rows = torch.nonzero(sel).squeeze(-1)
    as_t1 = trainer._ppg_snapshot_anchor({k: v[rows] for k, v in data["obs"].items()}, 64, nets=t1)
    for k in dp.KICK_KEYS:
        if k in out:
            assert torch.equal(out[k][rows], anchor[k][rows])            # kicks: still teacher 2
    assert torch.allclose(out["move_direction"][rows], as_t1["move_direction"], atol=1e-6)   # movement: teacher 1
    assert torch.allclose(out["exec_move_logit"][rows], as_t1["exec_move_logit"], atol=1e-6)
    assert not torch.allclose(anchor["move_direction"][rows], as_t1["move_direction"], atol=1e-4)  # teachers differ
    for k in anchor:                                                # all other rows untouched
        assert torch.equal(out[k][~sel], anchor[k][~sel])
    assert dp.kick_only_counterfactuals(trainer, data, anchor, teachers, [], 64) is anchor


def test_warmup_lr():
    assert dp.warmup_lr(1e-4, 0, 0) == 1e-4
    assert dp.warmup_lr(1e-4, 0, 100) == pytest.approx(1e-6)
    assert dp.warmup_lr(1e-4, 49, 100) == pytest.approx(5e-5)
    assert dp.warmup_lr(1e-4, 99, 100) == pytest.approx(1e-4)
    assert dp.warmup_lr(1e-4, 5000, 100) == pytest.approx(1e-4)


def test_label_smoothing_caps_the_bernoulli_targets():
    import math
    anchor = {"kick_logit": torch.tensor([[-20.0], [0.0], [3.0], [25.0]]), "move_direction": torch.ones(4, 2)}
    out = dp.smooth_bernoulli_targets(anchor, 1e-5)
    cap = math.log((1 - 5e-6) / 5e-6)
    assert out["kick_logit"].abs().max().item() <= cap + 1e-4
    assert out["kick_logit"][0].item() == pytest.approx(-cap, abs=1e-3)       # saturated: capped
    assert out["kick_logit"][1].item() == pytest.approx(0.0, abs=1e-6)        # 0.5 stays 0.5
    assert out["kick_logit"][2].item() == pytest.approx(3.0, abs=1e-3)        # ordinary: ~unchanged
    assert torch.equal(out["move_direction"], anchor["move_direction"])        # continuous: untouched
    assert dp.smooth_bernoulli_targets(anchor, 0.0) is anchor


def _mb(trainer, data):
    data = dict(data)
    data["actions"] = dp.soft_actions(data["anchor"], len(data["returns"]))
    return dp._slice(data, torch.arange(len(data["returns"])), trainer.device)


def test_loss_is_zero_against_itself_and_value_counts_only_real_rows(trainer, tmp_path):
    b = _obs(trainer)
    same = _teacher(trainer, tmp_path, "same.pt")
    data = dp.build_copies(b, torch.zeros(N), (1, 2), True, 1.0)
    data["anchor"] = dp.teacher_anchor(trainer, data["obs"], data["target_phase"], {1: same, 2: same}, 64)
    mb = _mb(trainer, data)
    _, _, kl = dp.minibatch_loss(trainer, mb, 1.0, 5.0, 0.0)
    assert float(kl) == pytest.approx(0.0, abs=1e-5)
    loss_a, vloss_a, _ = dp.minibatch_loss(trainer, mb, 1.0, 0.0, 1.0)
    mb2 = dict(mb)
    mb2["returns"] = mb["returns"].clone()
    mb2["returns"][N:] += 100.0                                   # counterfactual rows' returns: ignored
    _, vloss_b, _ = dp.minibatch_loss(trainer, mb2, 1.0, 0.0, 1.0)
    assert float(vloss_a) == pytest.approx(float(vloss_b), rel=1e-6)
    mb3 = dict(mb)
    mb3["returns"] = mb["returns"].clone()
    mb3["returns"][:N] += 1.0                                     # real rows' returns: counted
    _, vloss_c, _ = dp.minibatch_loss(trainer, mb3, 1.0, 0.0, 1.0)
    assert float(vloss_c) != pytest.approx(float(vloss_a), rel=1e-3)


def test_the_loss_trains_the_student_towards_two_different_teachers(trainer, tmp_path):
    b = _obs(trainer)
    teachers = {1: _teacher(trainer, tmp_path, "a.pt", perturb=0.1), 2: _teacher(trainer, tmp_path, "b.pt", perturb=0.1)}
    data = dp.build_copies(b, torch.zeros(N), (1, 2), True, 1.0)
    data["anchor"] = dp.teacher_anchor(trainer, data["obs"], data["target_phase"], teachers, 64)
    mb = _mb(trainer, data)
    params = [p for p in list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    first = None
    for _ in range(40):
        loss, _, kl = dp.minibatch_loss(trainer, mb, 1.0, 1.0, 0.0)
        first = float(kl) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(kl) < 0.5 * first
    for name in dp._SPREAD_KEYS:                                   # means only: spreads untouched
        assert getattr(trainer.execution_net, name).grad is None or \
            float(getattr(trainer.execution_net, name).grad.abs().sum()) == 0.0
