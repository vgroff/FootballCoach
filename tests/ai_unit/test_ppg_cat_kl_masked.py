"""_ppg_kl_penalty's target-categorical KL (pass/tackle/mark targets) with
absent player slots: same value as torch's Categorical KL, and a finite
gradient (torch's own on -inf-masked logits gives NaN in the backward)."""
from __future__ import annotations

import pytest
import torch

from footballcoach.ai.action.distributions import MaskedCategorical


@pytest.fixture(scope="module")
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    return PPOTrainer.from_config(device=torch.device("cpu"))


def _kl_pass(trainer, new_logits, anchor_logits, exists_mask):
    """Only the pass head's (parent + target) KL, via _ppg_kl_penalty."""
    from types import SimpleNamespace
    n, k = new_logits.shape
    z = torch.zeros(n, 1)
    d = SimpleNamespace(shoot_logit=z, pass_logit=z, move_logit=z, tackle_logit=z, get_possession_raw=z,
                        mark_logit=z, hold_position_logit=z, pass_target_logits=new_logits,
                        tackle_target_logits=torch.zeros(n, k), mark_target_logits=torch.zeros(n, k))
    e = SimpleNamespace(exec_move_logit=z, sprint_logit=z, kick_logit=z, tackle_attempt_logit=z,
                        move_direction=torch.tensor([[1.0, 0.0]] * n), kick_direction=torch.tensor([[1.0, 0.0, 0.0]] * n),
                        kick_power=z, kick_spin=torch.zeros(n, 3))
    anchor = {**{f: z for f in ("shoot_logit", "pass_logit", "move_logit", "tackle_logit", "get_possession_raw",
                                "mark_logit", "hold_position_logit", "exec_move_logit", "sprint_logit", "kick_logit",
                                "tackle_attempt_logit", "kick_power")},
              "pass_target_logits": anchor_logits, "tackle_target_logits": torch.zeros(n, k),
              "mark_target_logits": torch.zeros(n, k), "move_direction": e.move_direction,
              "kick_direction": e.kick_direction, "kick_spin": e.kick_spin}
    en = trainer.execution_net
    globals_ = {g: getattr(en, g).detach() for g in ("move_dir_log_kappa", "kick_dir_log_kappa",
                                                    "kick_dir_z_log_std", "kick_power_log_std", "kick_spin_log_std")}
    acts = {"exec_move": z, "kick": z, "pass_": torch.ones(n, 1), "tackle": z, "mark": z}
    _, per_head, _, rows = trainer._ppg_kl_penalty(d, e, anchor, globals_, acts, exists_mask, return_per_row=True)
    return per_head["pass_"], rows["pass_"]


def test_value_matches_torch_and_gradient_is_finite(trainer):
    g = torch.Generator().manual_seed(0)
    k = 9
    new = torch.randn(4, k, generator=g, requires_grad=True)
    anc = torch.randn(4, k, generator=g)
    mask = torch.ones(4, k)
    mask[0, 3:] = 0.0          # absent slots
    mask[1, :] = 0.0           # no other players at all (the guarded row)
    mean, rows = _kl_pass(trainer, new, anc, mask)
    want = torch.distributions.kl_divergence(MaskedCategorical(new.detach(), mask).dist,
                                             MaskedCategorical(anc, mask).dist)
    assert rows.tolist() == pytest.approx(want.tolist(), rel=1e-5, abs=1e-6)
    mean.backward()
    assert torch.isfinite(new.grad).all()
    assert float(new.grad[0, 3:].abs().sum()) == 0.0     # absent slots get no gradient
