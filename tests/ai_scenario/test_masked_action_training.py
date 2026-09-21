"""> **Documentation must stay in sync with code.** (see agent_plans/masked_action_training_plan.md)

L4 / L5 coverage for ``ppo.mask_untriggered_actions`` (Phase 3 of the masked-action-training plan):

* the per-head log-prob columns sum to the total log-prob the PPO ratio uses (D7/D8),
* all-ones masks change nothing; the epoch-0 ratio stays 1 with real masks,
* masked heads receive EXACTLY zero gradient from masked-out rows, unmasked heads still train,
* entropy masking, the gradient rescale option is value-preserving,
* a full ``_ppo_update`` with masks on runs and stays finite,
* L5: a synthetic bandit where the action only matters when an event flag is set -- masked training converges to
  the analytic optimum, unmasked training drifts (entropy bonus in irrelevant states), and the rejected
  "fire-only positives, all negatives" design (Design A) is demonstrably biased (kept so nobody re-introduces it).
"""
from __future__ import annotations

import functools
import random

import numpy as np
import pytest
import torch

from footballcoach.ai.action.distributions import IndependentBernoulli
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.action_opportunity_stats import opportunity_masks, opportunity_stats
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy, _ai_types
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS, RolloutBuffer
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

KI, TI = HEAD_LP_KEYS.index("kick"), HEAD_LP_KEYS.index("tackle_attempt")
KDI, KPI = HEAD_LP_KEYS.index("kick_dir"), HEAD_LP_KEYS.index("kick_power")


# ---------------------------------------------------------------------------
# fixtures: a real rollout with flags, on a trainer whose kick / tackle gates fire often
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trainer():
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    fh = PHASES_BY_ID[1].frozen_heads
    if fh:
        t.set_frozen_heads(fh)
    t.mask_untriggered_actions = True
    t.log_action_opportunity_stats = True
    with torch.no_grad():
        t.execution_net.kick_logit.bias.fill_(0.5)
        t.execution_net.tackle_attempt_logit.bias.fill_(0.5)
    return t


def _make_env(trainer, seed: int) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="masked_action_test", label="masked action test", description="1v1, rules opponent",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0),
        on_tick=phase1_training_on_tick,
    )
    env = ScenarioEnv(defn, trainee_player_id="trainee", phase=1, max_episode_s=5.0, secondary_player_ids=["opponent"])
    env._opp_flags_enabled = True
    env.sample_action_fn = trainer._sample_action
    return env


def _collect(trainer, n_steps: int, seed: int = 5):
    """Rollout with the flags on. Episodes alternate between starting with the trainee holding the ball, a loose ball
    at its feet, and the opponent carrying the ball next to it, so kick / tackle opportunities are common."""
    torch.manual_seed(seed)
    random.seed(seed)
    env = _make_env(trainer, seed)
    buf = RolloutBuffer()
    episode = {"i": 0}
    orig_reset = env.reset

    def reset(*a, **k):
        out = orig_reset(*a, **k)
        m = env._loop.match
        tr, op = m.player_by_id("trainee"), m.player_by_id("opponent")
        kind = episode["i"] % 3
        episode["i"] += 1
        m.ball.velocity = Vector3.zero()
        if kind == 0:
            m.ball.position = Vector3(tr.position.x, tr.position.y, m.ball.radius_m)
            m._set_possession("trainee")
        elif kind == 1:
            m.ball.position = Vector3(tr.position.x, tr.position.y, m.ball.radius_m)
            m._set_possession(None)
        else:
            op.position = Vector3(tr.position.x + 0.3, tr.position.y, 0.0)
            op.velocity = Vector3.zero()
            m.ball.position = Vector3(op.position.x, op.position.y, m.ball.radius_m)
            m._set_possession("opponent")
        return out

    env.reset = reset
    env.reset()
    last_obs = None
    for _ in range(n_steps):
        next_obs, reward, done, _info = env.step()
        tr_ = env.last_trainee_transition
        if tr_ is not None:
            buf.add(
                obs=tr_["obs"], action=_action_to_numpy(tr_["action"], tr_["raw_exec"]), log_prob=tr_["log_prob"],
                value=tr_["value"], reward=reward, done=1.0 if done else 0.0, head_log_probs=tr_.get("head_log_probs"),
            )
            last_obs = next_obs
        if done:
            env.reset()
    with torch.no_grad():
        lv = trainer._get_value({k: v.unsqueeze(0).to(trainer.device) for k, v in last_obs.to_torch_dict().items()})
    adv, ret = buf.compute_gae(trainer.gamma, trainer.lam, lv)
    batch = buf.as_tensors(adv, ret)
    batch.update(trainer._precompute_physics_full(batch, 256))
    return batch


@pytest.fixture(scope="module")
def batch(trainer):
    b = _collect(trainer, 900)
    st = opportunity_stats(b)
    # The fixture must actually exercise every path the assertions below rely on.
    assert st["n_fired_kick"] >= 10 and st["n_train_kick"] >= 30 and st["n_train_tack"] >= 8, st
    assert st["violations"] == 0, st
    return b


def _heads(trainer, batch, idx):
    mb_obs = {k.replace("obs/", ""): batch[k][idx] for k in batch if k.startswith("obs/")}
    sat, oat = _ai_types(mb_obs)
    kw = dict(ball_physics_full=mb_obs.get("ball_physics_full"), self_physics_full=mb_obs.get("self_physics_full"),
              other_physics_full=mb_obs.get("other_physics_full"))
    d = trainer.decision_net(mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"], mb_obs["ball_feat"],
                             mb_obs["global_feat"], sat, oat, **kw)
    e = trainer.execution_net(mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"], mb_obs["ball_feat"],
                              mb_obs["global_feat"], d, sat, oat)
    acts = {k.replace("action/", ""): batch[k][idx] for k in batch if k.startswith("action/")}
    return d, e, acts, mb_obs["exists_mask"]


def _flag_mask(trainer, batch):
    return trainer._opp_head_mask_matrix(batch)


# ---------------------------------------------------------------------------
# L4 -- log-prob identity, masks, gradients
# ---------------------------------------------------------------------------

def test_per_head_columns_sum_to_the_total_log_prob(trainer, batch):
    """D7/D8: the masked ratio is only valid if the per-head columns are exactly the terms of the total."""
    n = len(batch["log_probs"])
    idx = torch.arange(n)
    with torch.no_grad():
        d, e, acts, em = _heads(trainer, batch, idx)
        cols = trainer._per_head_new_log_probs(d, e, acts, em)
        total = trainer._recompute_log_prob(d, e, acts, em)
    assert torch.allclose(cols.sum(-1), total, atol=1e-4), (cols.sum(-1) - total).abs().max()
    # the STORED old side: per-head columns recorded at sample time vs the stored total
    assert torch.allclose(batch["head_log_probs"].sum(-1), batch["log_probs"], atol=1e-3), (
        (batch["head_log_probs"].sum(-1) - batch["log_probs"]).abs().max()
    )


def test_all_ones_mask_changes_nothing(trainer, batch):
    idx = torch.arange(len(batch["log_probs"]))
    with torch.no_grad():
        d, e, acts, em = _heads(trainer, batch, idx)
        cols = trainer._per_head_new_log_probs(d, e, acts, em)
        total = trainer._recompute_log_prob(d, e, acts, em)
    out = PPOTrainer._mask_total_log_probs(total, cols, torch.ones_like(cols))
    assert torch.allclose(out, total, atol=1e-6)


def test_epoch_zero_ratio_is_one_with_real_masks(trainer, batch):
    """Old and new sides drop the same terms, so before any update the masked ratio is 1 on every row."""
    n = len(batch["log_probs"])
    idx = torch.arange(n)
    M = _flag_mask(trainer, batch)
    assert (M == 0).any() and (M == 1).any(), "masks must be non-trivial for this test to mean anything"
    with torch.no_grad():
        d, e, acts, em = _heads(trainer, batch, idx)
        new = PPOTrainer._mask_total_log_probs(
            trainer._recompute_log_prob(d, e, acts, em), trainer._per_head_new_log_probs(d, e, acts, em), M)
        old = PPOTrainer._mask_total_log_probs(batch["log_probs"], batch["head_log_probs"], M)
    ratio = torch.exp(new - old)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3), (ratio - 1).abs().max()


def _grads_of(loss, params):
    for p in params:
        p.grad = None
    loss.backward()
    return [None if p.grad is None else p.grad.clone() for p in params]


def _masked_head_params(trainer):
    ex = trainer.execution_net
    return {
        "kick_logit": list(ex.kick_logit.parameters()),
        "tackle_attempt_logit": list(ex.tackle_attempt_logit.parameters()),
        "kick_direction": list(ex.kick_direction.parameters()) + [ex.kick_dir_log_kappa, ex.kick_dir_z_log_std],
        "kick_power": list(ex.kick_power.parameters()) + [ex.kick_power_log_std],
    }


def _policy_loss(trainer, batch, M, idx=None):
    n = len(batch["log_probs"])
    idx = torch.arange(n) if idx is None else idx
    d, e, acts, em = _heads(trainer, batch, idx)
    new = PPOTrainer._mask_total_log_probs(
        trainer._recompute_log_prob(d, e, acts, em), trainer._per_head_new_log_probs(d, e, acts, em), M[idx])
    old = PPOTrainer._mask_total_log_probs(batch["log_probs"][idx], batch["head_log_probs"][idx], M[idx])
    adv = batch["advantages"][idx]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return -(torch.exp(new - old) * adv).mean()


def test_masked_out_heads_get_exactly_zero_gradient_and_unmasked_heads_still_train(trainer, batch):
    n = len(batch["log_probs"])
    M = torch.ones(n, len(HEAD_LP_KEYS))
    M[:, [KI, TI, KDI, KPI, HEAD_LP_KEYS.index("kick_spin")]] = 0.0
    groups = _masked_head_params(trainer)
    all_p = [p for ps in groups.values() for p in ps]
    other = list(trainer.execution_net.exec_move_logit.parameters()) + list(trainer.execution_net.move_direction.parameters())
    trainer.execution_net.zero_grad(); trainer.decision_net.zero_grad()
    grads = _grads_of(_policy_loss(trainer, batch, M), all_p + other)
    for p, g in zip(all_p, grads[: len(all_p)]):
        assert g is None or bool((g == 0).all()), "a fully masked head received gradient"
    assert any(g is not None and bool((g != 0).any()) for g in grads[len(all_p):]), "unmasked heads must still train"

    # With the real (partial) flag masks the gates DO receive gradient again.
    Mreal = _flag_mask(trainer, batch)
    grads_real = _grads_of(_policy_loss(trainer, batch, Mreal), all_p)
    by_head = {}
    i = 0
    for name, ps in groups.items():
        by_head[name] = grads_real[i:i + len(ps)]
        i += len(ps)
    assert any(g is not None and bool((g != 0).any()) for g in by_head["kick_logit"])
    assert any(g is not None and bool((g != 0).any()) for g in by_head["tackle_attempt_logit"])


def test_kick_gate_gradient_equals_the_gradient_over_opportunity_rows_only(trainer, batch):
    """Independent reference: with ratio ~ 1, d/dw of the masked loss on the kick gate equals d/dw of
    -(adv * mask * logp_kick).mean() computed directly from the per-head column."""
    n = len(batch["log_probs"])
    idx = torch.arange(n)
    Mreal = _flag_mask(trainer, batch)
    w = trainer.execution_net.kick_logit.weight
    g_masked = _grads_of(_policy_loss(trainer, batch, Mreal), [w])[0]
    d, e, acts, em = _heads(trainer, batch, idx)
    col = trainer._per_head_new_log_probs(d, e, acts, em)[:, KI]
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    g_ref = _grads_of(-(adv * Mreal[:, KI] * col).mean(), [w])[0]
    assert torch.allclose(g_masked, g_ref, rtol=2e-3, atol=1e-7), (g_masked - g_ref).abs().max()
    assert float(g_masked.abs().max()) > 0


def test_grad_scale_is_value_preserving_and_scales_only_masked_heads(trainer, batch):
    n = len(batch["log_probs"])
    Mreal = _flag_mask(trainer, batch)
    scale = torch.ones(len(HEAD_LP_KEYS))
    scale[KI] = 10.0
    w = trainer.execution_net.kick_logit.weight
    w_exec = trainer.execution_net.exec_move_logit.weight

    def loss(grad_scale):
        idx = torch.arange(n)
        d, e, acts, em = _heads(trainer, batch, idx)
        new = PPOTrainer._mask_total_log_probs(
            trainer._recompute_log_prob(d, e, acts, em), trainer._per_head_new_log_probs(d, e, acts, em), Mreal, grad_scale)
        return new, -(new * batch["advantages"]).mean()

    new0, l0 = loss(None)
    new1, l1 = loss(scale)
    assert torch.allclose(new0, new1, atol=1e-6), "value must not change"
    g0 = _grads_of(l0, [w, w_exec])
    g1 = _grads_of(l1, [w, w_exec])
    assert torch.allclose(g1[0], 10.0 * g0[0], rtol=1e-3, atol=1e-7)
    assert torch.allclose(g1[1], g0[1], rtol=1e-4, atol=1e-7), "unscaled heads' gradients unchanged"


def test_entropy_masking(trainer, batch):
    n = len(batch["log_probs"])
    idx = torch.arange(n)
    d, e, acts, em = _heads(trainer, batch, idx)
    zeros = {"kick": torch.zeros(n), "tackle_attempt": torch.zeros(n)}
    _, bk0 = trainer._compute_entropy(d, e, em, return_breakdown=True, opp_masks=zeros)[:2]
    assert bk0["kick"] == 0.0 and bk0["tackle_attempt"] == 0.0 and bk0["kick_dir"] == 0.0 and bk0["kick_power"] == 0.0
    ones = {"kick": torch.ones(n), "tackle_attempt": torch.ones(n)}
    _, bk1 = trainer._compute_entropy(d, e, em, return_breakdown=True, opp_masks=ones)[:2]
    ref_kick = float(IndependentBernoulli(e.kick_logit).entropy().mean().detach())
    assert abs(bk1["kick"] - ref_kick) < 1e-6
    ref_ta = float(IndependentBernoulli(e.tackle_attempt_logit).entropy().mean().detach())
    assert abs(bk1["tackle_attempt"] - ref_ta) < 1e-6
    # a real mask only keeps opportunity rows
    Mreal = _flag_mask(trainer, batch)
    real = {"kick": Mreal[:, KI], "tackle_attempt": Mreal[:, TI]}
    _, bkr = trainer._compute_entropy(d, e, em, return_breakdown=True, opp_masks=real)[:2]
    ref = float((Mreal[:, KI] * IndependentBernoulli(e.kick_logit).entropy().reshape(-1)).mean().detach())
    assert abs(bkr["kick"] - ref) < 1e-6
    # the unmasked path is exactly the old formula
    _, bku = trainer._compute_entropy(d, e, em, return_breakdown=True)[:2]
    assert abs(bku["kick"] - ref_kick) < 1e-6


def test_missing_flags_with_masking_on_fails_loudly(trainer, batch):
    stripped = {k: v for k, v in batch.items() if not k.startswith("action/opp_") and k != "action/fired_kick"}
    with pytest.raises(ValueError, match="no action/opp_"):
        trainer._opp_head_mask_matrix(stripped)
    trainer.mask_untriggered_actions = False
    try:
        assert trainer._opp_head_mask_matrix(stripped) is None
    finally:
        trainer.mask_untriggered_actions = True


def test_opportunity_masks_default_to_all_ones_without_flags(batch):
    stripped = {k: v for k, v in batch.items() if not k.startswith("action/opp_") and k != "action/fired_kick"}
    m = opportunity_masks(stripped)
    assert all(bool((v == 1).all()) for v in m.values())


def test_full_ppo_update_with_masks_runs_and_is_finite(trainer, batch, caplog):
    import copy
    b = copy.copy(batch)
    with caplog.at_level("INFO"):
        out = trainer._ppo_update(b, progress=0.0)
    assert all(np.isfinite(v) for v in out.values() if isinstance(v, (int, float)))
    assert any("[action opp]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# L5 -- synthetic bandit: the action only matters when an event flag is set
# ---------------------------------------------------------------------------

def _bandit(mode: str, steps: int = 400, n: int = 40000, seed: int = 0, ent: float = 0.0):
    """Gate logit = a on 'opportunity' rows (z = 1), b elsewhere (separate parameters: only the irrelevant-state
    drift is under test, not parameter sharing). Fire pays +1 only when z=1 (ΔQ=1); in z=0 rows the
    advantage has a positive offset (baseline off by 0.5 -- a normal value-net error) and noise, but is
    independent of the action. mode: 'masked' (Design B), 'unmasked', 'design_a' (drop fired-but-irrelevant rows)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor(-3.0, requires_grad=True)
    b = torch.tensor(-3.0, requires_grad=True)
    opt = torch.optim.SGD([a, b], lr=0.5)
    for _ in range(steps):
        z = (torch.rand(n, generator=g) < 0.3).float()
        logit = torch.where(z > 0.5, a, b)
        p = torch.sigmoid(logit)
        act = (torch.rand(n, generator=g) < p.detach()).float()
        noise = torch.randn(n, generator=g)
        # advantage of the sampled action relative to the state baseline
        adv = torch.where(z > 0.5, (act - p.detach()) * 1.0 + 0.2 * noise, 0.5 + 0.5 * noise)
        lp = act * torch.log(p + 1e-8) + (1 - act) * torch.log(1 - p + 1e-8)
        ent_t = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        if mode == "masked":
            w = z
        elif mode == "unmasked":
            w = torch.ones(n)
        elif mode == "design_a":
            w = torch.where(z > 0.5, torch.ones(n), 1.0 - act)     # positives only where relevant, all negatives
        else:
            raise ValueError(mode)
        loss = -(w * adv * lp).mean() - ent * (w if mode != "design_a" else torch.ones(n)).mul(ent_t).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(torch.sigmoid(a.detach())), float(torch.sigmoid(b.detach()))


def test_bandit_masked_training_converges_and_leaves_irrelevant_states_alone():
    p_opp, p_non = _bandit("masked", ent=0.01)
    assert p_opp > 0.95, p_opp                       # learns to fire when it matters
    assert abs(p_non - float(torch.sigmoid(torch.tensor(-3.0)))) < 0.02, p_non   # b untouched by irrelevant rows


def test_bandit_unmasked_entropy_drags_irrelevant_states_off_their_optimum():
    """The plan's D-symptom: an entropy bonus applied on every row inflates the gate where it cannot matter."""
    _, p_non = _bandit("unmasked", ent=0.2)
    assert p_non > 0.12, p_non                         # b was -3 (p=0.047); the unmasked entropy bonus inflates it


def test_bandit_design_a_is_biased():
    """'Fire-only positives, all negatives' pushes the irrelevant-state gate the wrong way from a value-baseline
    offset that is independent of the action; Design B does not. Kept as a negative test."""
    _, p_non_b = _bandit("masked", ent=0.0)
    _, p_non_a = _bandit("design_a", ent=0.0)
    ref = float(torch.sigmoid(torch.tensor(-3.0)))
    assert abs(p_non_b - ref) < 0.02
    assert p_non_a < ref - 0.02, (p_non_a, ref)
