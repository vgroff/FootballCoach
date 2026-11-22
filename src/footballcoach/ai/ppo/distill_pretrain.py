"""Multi-teacher distillation pretraining: one student network mimicking a
frozen teacher per phase, with no PPO (``train.py --distill-pretrain-only``).

Why: PPO with BC and a phase mix pulled the shared network several ways at
once (phase-2 shooting leaked into phase 1; BC's kick gate raced its power).
This fits the student, supervised, to the teachers' action distributions on
the states the STUDENT visits (DAgger-style rollouts over the phase mix), and
fits the value head to Monte Carlo returns on the side so PPO can resume later
with a warm critic.

Each rollout row is used twice when ``counterfactual`` is on:

- the real copy: its own task id, target = its own phase's teacher, plus the
  value loss (the return is only valid for the task actually played);
- the counterfactual copy: the task id switched to the other phase, target =
  THAT phase's teacher on this copy, no value loss.

So on every state the task id alone decides which teacher's behaviour the
student produces. Each teacher always sees its own task id (the copy it is the
target of), which matters for a teacher trained with task ids (e.g. a mixed
run). Two phases only, for now (the swap needs a partner phase).

The KL is the PPG refit's analytic per-head KL (``_ppg_kl_penalty``), means
only (the student's global spreads kappa / log_std are detached, so two
teachers with different spreads do not fight over them), with the kick and
move sub-heads weighted by the TEACHER's kick / move probabilities, so power
and aim are matched where the teacher would kick, not only where the student
already does.

Config ``distill_pretrain`` (ai_config.json)::

    {"teachers": {"1": <checkpoint>, "2": <checkpoint>}, "counterfactual": true,
     "counterfactual_weight": 1.0, "kl_coef": 5.0, "value_coef": 1.0,
     "epochs": 6, "lr": 1e-4, "batch_size": 8000, "rollout_steps": 540000,
     "num_rollouts": 4}
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("footballcoach.ai.ppo")

_SPREAD_KEYS = ("move_dir_log_kappa", "kick_dir_log_kappa", "kick_dir_z_log_std", "kick_power_log_std")


DEFAULTS = {"counterfactual": True, "counterfactual_weight": 1.0, "kl_coef": 5.0, "value_coef": 1.0,
            "label_smoothing": 0.0, "epochs": 6, "lr": 1e-4, "lr_warmup_steps": 0, "batch_size": 8000,
            "rollout_steps": 540000, "num_rollouts": 4, "aggregate_cycles": 1,
            "counterfactual_kick_only_phases": (), "holdout_episode_fraction": 0.0, "holdout_reference": None}


def holdout_episode_mask(track_ids, dones, fraction: float, generator: torch.Generator) -> torch.Tensor:
    """(N,) bool over the rollout's rows: ``fraction`` of its COMPLETE episodes
    (per track, trainer_common.complete_episode_rows -- the same segmentation
    as PPO's held-out validation episodes), every row of a chosen episode.
    Rows of no complete episode stay in the fit. 0 episodes held out when the
    fraction rounds to 0 or there are fewer than 2 complete episodes."""
    from footballcoach.ai.ppo.trainer_common import complete_episode_rows
    episodes = complete_episode_rows(track_ids, dones)
    mask = torch.zeros(len(dones), dtype=torch.bool)
    n_hold = round(fraction * len(episodes)) if len(episodes) >= 2 else 0
    if n_hold <= 0:
        return mask
    for k in torch.randperm(len(episodes), generator=generator)[:n_hold].tolist():
        mask[episodes[k]] = True
    return mask

# The anchor fields of the kick sub-heads (gate, power, direction, spin).
KICK_KEYS = ("kick_logit", "kick_power", "kick_direction", "kick_spin")


def kick_only_counterfactuals(trainer, data: dict, anchor: dict, teachers: dict, kick_only_phases, batch_size: int) -> dict:
    """For counterfactual rows whose target phase is in ``kick_only_phases``:
    keep the target teacher's KICK heads (``KICK_KEYS``) but take every other
    head from the row's REAL phase's teacher. E.g. 1v1 states labelled task id
    2: shoot like the phase-2 teacher, move like the phase-1 one -- the task id
    then only has to switch the kicking, and both copies of a state agree on
    the movement (measured overnight: the phase-2 counterfactual's KL was
    mostly move direction, 0.98 of 1.33)."""
    phases = set(int(p) for p in kick_only_phases)
    if not phases:
        return anchor
    rows = data["is_cf"] & torch.isin(data["target_phase"], torch.tensor(sorted(phases)))
    idx = torch.nonzero(rows).squeeze(-1)
    if len(idx) == 0:
        return anchor
    pa, pb = sorted(teachers)
    real_phase = torch.where(data["target_phase"][idx] == pa, torch.full_like(idx, pb), torch.full_like(idx, pa))
    out = {k: v.clone() for k, v in anchor.items()}
    teacher_obs = {k: v for k, v in data["obs"].items() if not k.endswith("physics_full")}
    for pid in sorted(set(real_phase.tolist())):
        sub = idx[real_phase == pid]
        alt = trainer._ppg_snapshot_anchor({k: v[sub] for k, v in teacher_obs.items()}, batch_size, nets=teachers[pid])
        for k, v in alt.items():
            if k not in KICK_KEYS and k in out:
                out[k][sub] = v.cpu()
    return out


_DATA_TENSORS = ("returns", "target_phase", "value_w", "kl_w", "is_cf", "bc_w", "bc_labels")
_DATA_DICTS = ("obs", "anchor", "actions")


def concat_data(parts: list) -> dict:
    """One fit dataset from several cycles' (build_copies + anchor + actions)
    dicts, row-wise."""
    if len(parts) == 1:
        return parts[0]
    out = {k: torch.cat([p[k] for p in parts]) for k in _DATA_TENSORS}
    for d in _DATA_DICTS:
        out[d] = {k: torch.cat([p[d][k] for p in parts]) for k in parts[0][d]}
    return out


def as_history(data: dict) -> dict:
    """A cycle's data kept for later cycles (DAgger aggregation): its teacher
    targets stay valid (the teachers are frozen) but its Monte Carlo returns
    came from an earlier student, so it no longer trains the value head."""
    out = dict(data)
    out["value_w"] = torch.zeros_like(data["value_w"])
    return out


def warmup_lr(base_lr: float, step: int, warmup_steps: int) -> float:
    """Linear warmup from ~0 to ``base_lr`` over the first ``warmup_steps``
    optimiser steps of the whole run (once, not per cycle: Adam's state
    carries over between cycles, only its start is cold)."""
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup_steps)


def load_config() -> dict:
    from footballcoach.ai.config import load_ai_config
    cfg = dict(load_ai_config().get("distill_pretrain") or {})
    teachers = {int(k): v for k, v in (cfg.get("teachers") or {}).items() if not str(k).startswith("_")}
    if len(teachers) != 2:
        raise ValueError(f"distill_pretrain.teachers needs exactly two phases (got {sorted(teachers)})")
    cfg["teachers"] = teachers
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if not k.startswith("_")}}


def swap_task_ids(global_feat: torch.Tensor, phase_a: int, phase_b: int) -> torch.Tensor:
    """A copy of ``global_feat`` with the two phases' task-id columns exchanged."""
    from footballcoach.ai.ppo.distill import task_id_column
    ca, cb = task_id_column(phase_a), task_id_column(phase_b)
    out = global_feat.clone()
    out[:, ca], out[:, cb] = global_feat[:, cb], global_feat[:, ca]
    return out


def _bc_label_dim() -> int:
    from footballcoach.ai.ppo.bc import BC_LABEL_DIM
    return BC_LABEL_DIM


def goalkeeper_rows(obs: dict) -> torch.Tensor:
    """(N,) bool: rows whose OWN player is a goalkeeper (PlayerFeatures.is_goalkeeper)."""
    import dataclasses
    from footballcoach.ai.obs.schema import PlayerFeatures
    col = [f.name for f in dataclasses.fields(PlayerFeatures)].index("is_goalkeeper")
    return obs["self_feat"][:, col] > 0.5


def build_copies(obs: dict, returns: torch.Tensor, phases: tuple[int, int], counterfactual: bool,
                 counterfactual_weight: float, counterfactual_rows: Optional[torch.Tensor] = None) -> dict:
    """The rows to fit: the real copy (own phase, value target) and, if on,
    the task-id-swapped copy. Returns {"obs", "returns", "target_phase",
    "value_w", "kl_w", "is_cf"}; every row must carry one of ``phases``' task ids.

    ``counterfactual_rows``: optional (N,) bool, which rows get a counterfactual
    copy (None = all). Real copies come first, then the counterfactual ones,
    so the first ``(~is_cf).sum()`` rows are always the real rollout rows in
    order. Counterfactual copies never carry a value target (value_w = 0)."""
    from footballcoach.ai.ppo.distill import phase_row_mask
    pa, pb = phases
    in_a, in_b = phase_row_mask(obs["global_feat"], pa), phase_row_mask(obs["global_feat"], pb)
    if not bool((in_a ^ in_b).all()):
        raise ValueError("distill_pretrain: every row must carry exactly one of the two teachers' task ids")
    real_phase = torch.where(in_a, torch.full_like(in_a, pa, dtype=torch.long), torch.full_like(in_a, pb, dtype=torch.long))
    n = len(returns)
    parts = [(obs, real_phase, torch.ones(n), torch.ones(n), torch.zeros(n, dtype=torch.bool))]
    cf_idx = None
    if counterfactual:
        cf_idx = (torch.arange(n) if counterfactual_rows is None
                  else torch.nonzero(counterfactual_rows.bool()).squeeze(-1))
        m = len(cf_idx)
        cf_obs = {k: v[cf_idx] for k, v in obs.items()}
        cf_obs["global_feat"] = swap_task_ids(cf_obs["global_feat"], pa, pb)
        own = real_phase[cf_idx]
        other = torch.where(own == pa, torch.full_like(own, pb), torch.full_like(own, pa))
        parts.append((cf_obs, other, torch.zeros(m), torch.full((m,), float(counterfactual_weight)),
                      torch.ones(m, dtype=torch.bool)))
    total = sum(len(p[1]) for p in parts)
    return {
        "obs": {k: torch.cat([p[0][k] for p in parts]) for k in obs},
        "returns": torch.cat([returns] + ([returns[cf_idx]] if cf_idx is not None else [])),
        "target_phase": torch.cat([p[1] for p in parts]),
        "value_w": torch.cat([p[2] for p in parts]),
        "kl_w": torch.cat([p[3] for p in parts]),
        "is_cf": torch.cat([p[4] for p in parts]),
        # Role BC (attach_role_bc): none by default.
        "bc_w": torch.zeros(total),
        "bc_labels": torch.zeros(total, _bc_label_dim()),
    }


def teacher_anchor(trainer, obs: dict, target_phase: torch.Tensor, teachers: dict, batch_size: int) -> dict:
    """Per-row anchor (the ``_ppg_snapshot_anchor`` layout) from each row's target teacher."""
    anchor: dict[str, torch.Tensor] = {}
    teacher_obs = {k: v for k, v in obs.items() if not k.endswith("physics_full")}
    for pid, nets in teachers.items():
        idx = torch.nonzero(target_phase == pid).squeeze(-1)
        if len(idx) == 0:
            continue
        sub = trainer._ppg_snapshot_anchor({k: v[idx] for k, v in teacher_obs.items()}, batch_size, nets=nets)
        for k, v in sub.items():
            if k not in anchor:
                anchor[k] = torch.zeros((len(target_phase),) + tuple(v.shape[1:]), dtype=v.dtype)
            anchor[k][idx] = v.cpu()
    return anchor


# The anchor's Bernoulli logits (decision + execution binary heads).
BERNOULLI_KEYS = ("shoot_logit", "pass_logit", "move_logit", "tackle_logit", "get_possession_raw", "mark_logit",
                  "hold_position_logit", "exec_move_logit", "sprint_logit", "kick_logit", "tackle_attempt_logit")


def smooth_bernoulli_targets(anchor: dict, eps: float) -> dict:
    """The teachers' Bernoulli targets mixed with a fair coin,
    p' = (1 - eps) p + eps / 2, back as logits: a cap of
    |logit| <= log((1 - eps/2) / (eps/2)) (~11.5 at eps = 1e-5) on what the
    student is pulled towards, so saturated teacher heads (logits of -20) do
    not drive the student's logits to where the sigmoid is flat and PPO can
    no longer move them. Continuous heads (means only) and the frozen target
    categoricals are left as they are."""
    if eps <= 0.0:
        return anchor
    out = dict(anchor)
    for k in BERNOULLI_KEYS:
        if k in out:
            p = torch.sigmoid(out[k].double())
            p = (1.0 - eps) * p + eps / 2.0
            out[k] = torch.log(p / (1.0 - p)).to(anchor[k].dtype)
    return out


def soft_actions(anchor: dict, n: int) -> dict:
    """The teacher's move / kick probabilities as the KL's sub-head weights."""
    z = torch.zeros(n, 1)
    return {"exec_move": torch.sigmoid(anchor["exec_move_logit"]).reshape(n, 1),
            "kick": torch.sigmoid(anchor["kick_logit"]).reshape(n, 1),
            "pass_": z, "tackle": z, "mark": z}


def _student_forward(trainer, mb_obs: dict, nets: Optional[tuple] = None):
    """The student's heads on ``mb_obs``; ``nets`` = (decision_net, execution_net)
    to run instead (e.g. a frozen reference copy, see holdout_reference)."""
    from footballcoach.ai.ppo.trainer_common import _ai_types
    d_net, e_net = nets if nets is not None else (trainer.decision_net, trainer.execution_net)
    sat, oat = _ai_types(mb_obs)
    d = d_net(
        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"], mb_obs["ball_feat"], mb_obs["global_feat"],
        sat, oat, ball_physics_full=mb_obs.get("ball_physics_full"),
        self_physics_full=mb_obs.get("self_physics_full"), other_physics_full=mb_obs.get("other_physics_full"),
    )
    e = e_net(
        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"], mb_obs["ball_feat"], mb_obs["global_feat"],
        d, sat, oat,
    )
    return d, e


def role_bc_loss(trainer, d, e, mb: dict, label_smoothing: float) -> tuple[torch.Tensor, torch.Tensor]:
    """BC to the role's rules AI (role_bc) on the rows with bc_w > 0: the
    per-row loss (sum over bc.GROUP_ROW_COLUMNS) weighted by bc_w, averaged
    over those rows; Bernoulli targets smoothed y -> y (1 - s) + s / 2.

    Returns (loss to train on, value to LOG). The logged value leaves out the
    decision heads frozen in this phase (trainer._ppo_lp_masked_heads): with
    bc.bc_trains_frozen_heads their terms still train the shared trunk (kept,
    so they train normally once unfrozen), but they cannot act, so they are
    not part of how well the role is matched."""
    from footballcoach.ai.ppo.bc import bc_loss_from_tensor
    w = mb["bc_w"]
    if not bool((w > 0).any()):
        z = torch.zeros((), device=w.device)
        return z, z
    kw = {**trainer._bc_loss_kwargs(), "dec_label_smoothing": label_smoothing, "exec_label_smoothing": label_smoothing,
          **(getattr(trainer, "_role_bc_loss_overrides", None) or {})}
    frozen = tuple(getattr(trainer, "_ppo_lp_masked_heads", ()) or ())
    _, _, split = bc_loss_from_tensor(mb["bc_labels"], trainer._bc_heads_for_loss(d), e, **kw,
                                      return_breakdown=True, split_kick=True, split_tackle=True, return_group_rows=True,
                                      log_exclude_decision_heads=frozen)
    m = (w > 0) & split["valid"].bool()
    denom = m.sum().clamp(min=1)
    train = (split["rows"].sum(dim=1) * w * m).sum() / denom
    logged = (split["rows_logged"].sum(dim=1) * w * m).sum() / denom
    return train, logged


def minibatch_loss(trainer, mb: dict, ret_std: float, kl_coef: float, value_coef: float,
                   bc_label_smoothing: float = 0.0) -> tuple:
    """(loss, value_mse_norm, kl, bc) on one minibatch dict of device tensors.
    ``kl`` covers the rows with kl_w > 0 (the frozen teachers), ``bc`` the rows
    with bc_w > 0 (a role taught by its rules AI, see role_bc)."""
    d, e = _student_forward(trainer, mb["obs"])
    globals_ = {k: getattr(trainer.execution_net, k).detach() for k in _SPREAD_KEYS}
    kl, _ = trainer._ppg_kl_penalty(
        d, e, mb["anchor"], globals_, mb["actions"], mb["obs"]["exists_mask"],
        row_weights=mb["kl_w"], detach_spread=True, soft_action_weights=True, forward_kl=True,
    )
    vw = mb["value_w"]
    sq = (e.value.squeeze(-1) - mb["returns"]) ** 2
    vloss = (sq * vw).sum() / vw.sum().clamp(min=1.0) / (ret_std ** 2)
    bc, bc_logged = role_bc_loss(trainer, d, e, mb, bc_label_smoothing)
    # The (||raw|| - 1)^2 direction-magnitude regularizer every other training
    # path applies (bc.direction_magnitude_reg, bc.direction_mag_reg_coef; 0 =
    # off): without it the raw move/kick direction drifts long (median ~28
    # measured 2026-11-20) and every angle loss's gradient shrinks ~1/||raw||.
    from footballcoach.ai.ppo.bc import direction_magnitude_reg
    mag = direction_magnitude_reg(e, float(getattr(trainer, "_bc_dir_mag_reg_coef", 0.0)))
    return kl_coef * kl + value_coef * vloss + bc + mag, vloss, kl, bc_logged


def _slice(data: dict, idx: torch.Tensor, device) -> dict:
    return {
        "obs": {k: v[idx].to(device) for k, v in data["obs"].items()},
        "anchor": {k: v[idx].to(device) for k, v in data["anchor"].items()},
        "actions": {k: v[idx].to(device) for k, v in data["actions"].items()},
        "returns": data["returns"][idx].to(device),
        "value_w": data["value_w"][idx].to(device),
        "kl_w": data["kl_w"][idx].to(device),
        "bc_w": data["bc_w"][idx].to(device),
        "bc_labels": data["bc_labels"][idx].to(device),
    }


def group_report(trainer, data: dict, batch_size: int, max_rows: int = 40000, nets: Optional[tuple] = None) -> dict:
    """{(phase of the row, "real"/"cf"): {"kl", "agree": {...}}} on up to
    ``max_rows`` rows per group, no grad. ``nets``: score that (decision,
    execution) pair instead of the live student, at the live student's spreads."""
    from footballcoach.ai.ppo.distill import phase_row_mask
    from footballcoach.ai.ppo.ppo_update import _DISTILL_AGREE_KEYS, _distill_agreement
    out = {}
    gf = data["obs"]["global_feat"]
    for pid in sorted(set(data["target_phase"].tolist())):
        for cf in (False, True):
            rows = torch.nonzero((data["target_phase"] == pid) & (data["is_cf"] == cf) & (data["kl_w"] > 0)).squeeze(-1)
            if len(rows) == 0:
                continue
            rows = rows[:max_rows]
            kls, agrees, n = [], [], 0
            heads: dict[str, float] = {}
            with torch.no_grad():
                for s in range(0, len(rows), batch_size):
                    idx = rows[s:s + batch_size]
                    mb = _slice(data, idx, trainer.device)
                    d, e = _student_forward(trainer, mb["obs"], nets)
                    globals_ = {k: getattr(trainer.execution_net, k).detach() for k in _SPREAD_KEYS}
                    kl, per_head = trainer._ppg_kl_penalty(d, e, mb["anchor"], globals_, mb["actions"], mb["obs"]["exists_mask"],
                                                           detach_spread=True, soft_action_weights=True, forward_kl=True)
                    kls.append(float(kl) * len(idx))
                    for h, v in per_head.items():
                        heads[h] = heads.get(h, 0.0) + float(v) * len(idx)
                    agrees.append(_distill_agreement(e, mb["anchor"], torch.ones(len(idx), dtype=torch.bool,
                                                                                device=trainer.device), mb["actions"]))
                    n += len(idx)
            # The task-id phase of these rows (what the student should behave as).
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # e.g. no teacher-kick rows -> NaN power diff
                agree = dict(zip(_DISTILL_AGREE_KEYS, np.nanmean(np.asarray(agrees), axis=0).tolist()))
            out[(pid, "cf" if cf else "real")] = {"kl": sum(kls) / max(n, 1), "agree": agree, "n": n,
                                                  "by_head": {h: v / max(n, 1) for h, v in heads.items()}}
    return out


def attach_role_bc(data: dict, batch: dict, role_bc: dict) -> None:
    """In place: give ``data`` (build_copies' real rows, counterfactual off)
    the rollout's BC labels and per-row BC weights. A row whose BC source is a
    role in ``role_bc`` ({role: coeff}; its rules AI labels it, see
    ai/bc/role_experts.RulesExpert and the phase's curriculum bc config) is
    taught by that label instead of a teacher network: bc_w = coeff, kl_w = 0.

    The labels are recorded in the world frame (ai/ppo/bc.py); the student's
    heads come out in the canonical frame (observer attacks +x), so they are
    canonicalized here (obs/canonical.canonicalize_bc_labels) -- without it a
    right-team player's (the keeper's) direction / region / kick targets have
    their x mirrored."""
    from footballcoach.ai.curriculum.phases import ALL_PHASES
    from footballcoach.ai.obs.canonical import canonicalize_bc_labels, x_sign_of
    from footballcoach.ai.ppo.bc import BC_LABEL_DIM
    from footballcoach.ai.ppo.bc_sources import source_id
    total = len(data["returns"])
    is_cf = data.get("is_cf")
    n = total if is_cf is None else int((~is_cf).sum())     # the real rows come first (build_copies)
    labels = batch.get("bc_labels")
    data["bc_labels"] = torch.zeros(total, BC_LABEL_DIM)
    if labels is not None and len(labels) == n:
        data["bc_labels"][:n] = canonicalize_bc_labels(labels.float(), x_sign_of(data["obs"]["self_feat"][:n]))
    data["bc_w"] = torch.zeros(total)
    if not role_bc:
        return
    if total != n:
        raise ValueError("distill_pretrain role_bc: needs counterfactual off")
    src = batch.get("bc_sources")
    if src is None or len(src) != n:
        raise ValueError("distill_pretrain role_bc: the rollout carries no per-row BC sources")
    for role, coeff in role_bc.items():
        sids = [source_id(ph.phase_id, role) for ph in ALL_PHASES if ph.is_role_based and role in ph.roles.values()]
        sids = [x for x in sids if x]
        if not sids:
            raise ValueError(f"distill_pretrain role_bc: role {role!r} has no BC configured in any phase "
                             "(curriculum.<phase>.bc must list it so its rules AI labels its rows)")
        m = torch.isin(src, torch.tensor(sids))
        data["bc_w"][m] = coeff
        data["kl_w"][m] = 0.0


def bc_report(trainer, data: dict, batch_size: int, smoothing: float, max_rows: int = 40000,
              unweighted: bool = False) -> float:
    """Mean LOGGED role-BC loss (frozen decision heads left out, see
    role_bc_loss) over (up to max_rows of) the BC-taught rows, no grad.
    Weighted by each row's role_bc coefficient (the value trained on), or
    with ``unweighted`` the plain per-row loss (every taught row weight 1)."""
    rows = torch.nonzero(data["bc_w"] > 0).squeeze(-1)[:max_rows]
    if len(rows) == 0:
        return float("nan")
    tot, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            idx = rows[start:start + batch_size]
            mb = _slice(data, idx, trainer.device)
            if unweighted:
                mb = {**mb, "bc_w": (mb["bc_w"] > 0).to(mb["bc_w"].dtype)}
            d, e = _student_forward(trainer, mb["obs"])
            tot += float(role_bc_loss(trainer, d, e, mb, smoothing)[1]) * len(idx)
            n += len(idx)
    return tot / max(n, 1)


def bc_report_line(trainer, data: dict, batch_size: int, smoothing: float) -> str:
    """'<weighted> (unweighted <plain>)' for the pretrain log."""
    return (f"{bc_report(trainer, data, batch_size, smoothing):.4f} "
            f"(unweighted {bc_report(trainer, data, batch_size, smoothing, unweighted=True):.3f})")


def log_report(tag: str, report: dict) -> None:
    for (pid, copy), r in sorted(report.items()):
        a = r["agree"]
        log.info(f"  [distill pretrain {tag}] task-id phase {pid} ({copy}, n={r['n']}): kl={r['kl']:.4f}  "
                 f"move_dir err={a['move_dir_err_deg']:.1f}deg  P(kick) s/t={a['p_kick_student']:.3f}/{a['p_kick_teacher']:.3f}  "
                 f"|power diff|={a['power_diff']:.3f}  exec_move agree={100 * a['exec_move_agree']:.1f}%  "
                 f"sprint agree={100 * a['sprint_agree']:.1f}%")
        top = sorted(r.get("by_head", {}).items(), key=lambda kv: -kv[1])[:5]
        if top:
            log.info(f"  [distill pretrain {tag}]   task-id phase {pid} ({copy}) KL by head: "
                     + "  ".join(f"{h}={v:.4f}" for h, v in top))


def holdout_line(report: dict) -> str:
    """One compact line of a group_report: KL per (phase, real/cf) with move / kick direction split."""
    def one(r):
        h = r.get("by_head", {})
        return (f"{r['kl']:.4f} (move_dir {h.get('move_dir', 0.0):.3f} kick_dir {h.get('kick_dir', 0.0):.3f}, "
                f"n={r['n']})")
    return "  ".join(f"p{pid} {copy} {one(r)}" for (pid, copy), r in sorted(report.items()))


def reference_delta_line(report: dict, ref: dict) -> str:
    """Current minus reference KL per (phase, real/cf), with move / kick direction split
    (negative = the current student is closer to the teachers than the reference)."""
    def d(r, q):
        h, hq = r.get("by_head", {}), q.get("by_head", {})
        return (f"{r['kl'] - q['kl']:+.4f} (move_dir {h.get('move_dir', 0.0) - hq.get('move_dir', 0.0):+.3f} "
                f"kick_dir {h.get('kick_dir', 0.0) - hq.get('kick_dir', 0.0):+.3f})")
    return "  ".join(f"p{pid} {copy} {d(r, ref[(pid, copy)])}" for (pid, copy), r in sorted(report.items()) if (pid, copy) in ref)


def composition(obs: dict, track_ids, phases: tuple[int, int], stats: dict) -> str:
    """One line: rows per phase split by track (trainee / secondary, i.e. the
    phase-1 opponent's or a phase-2 secondary neural player's rows) and the
    episodes per phase-1 opponent type, so the mix is visibly the training mix."""
    from collections import Counter

    from footballcoach.ai.ppo.distill import phase_row_mask
    parts = []
    for pid in phases:
        rows = phase_row_mask(obs["global_feat"], pid)
        tracks = Counter(("trainee" if str(t) == "trainee" else "secondary")
                         for t, r in zip(track_ids, rows.tolist()) if r) if track_ids is not None else {}
        parts.append(f"phase {pid}: {int(rows.sum())} rows ("
                     + ", ".join(f"{k} {v}" for k, v in sorted(tracks.items())) + ")")
    opp = {k: len(stats.get(f"outcomes_vs_{k}", []) or []) for k in ("rules", "immobile", "neural")}
    return "  ".join(parts) + "  |  phase-1 episodes vs " + "  ".join(f"{k} {v}" for k, v in opp.items())


def distill_pretrain(trainer, phase_id: int, cfg: Optional[dict] = None, checkpoint_path=None) -> dict:
    """Run ``num_rollouts`` collect -> fit cycles (see module docstring).
    Returns the last cycle's group report. Saves ``checkpoint_path`` after
    every cycle when given."""
    from footballcoach.ai.ppo.distill import teacher_nets
    cfg = {**DEFAULTS, **(cfg or load_config())}
    role_bc = {r: float(c) for r, c in (cfg.get("role_bc") or {}).items() if not str(r).startswith("_")}
    smoothing = float(cfg.get("bc_label_smoothing", 0.0))
    # Role-BC-only loss options (role_bc_loss): fit the move direction only on
    # rows whose label moves, and/or its own direction weight (default: bc's).
    overrides = {}
    if cfg.get("role_bc_direction_moving_only"):
        overrides["direction_moving_only"] = True
    if cfg.get("role_bc_direction_loss_weight") is not None:
        overrides["direction_loss_weight"] = float(cfg["role_bc_direction_loss_weight"])
    trainer._role_bc_loss_overrides = overrides
    if role_bc and cfg["counterfactual"]:
        raise ValueError("distill_pretrain: role_bc needs counterfactual off (a task-id-swapped copy of a "
                         "rules-taught row would have no teacher)")
    hold_frac = float(cfg.get("holdout_episode_fraction") or 0.0)
    if role_bc and hold_frac > 0:
        raise ValueError("distill_pretrain: holdout_episode_fraction is not supported with role_bc "
                         "(the BC labels are aligned to the full rollout)")
    hold_gen = torch.Generator().manual_seed(int(cfg.get("holdout_seed", 0)))
    # holdout_reference: a FROZEN model scored on the same held-out episodes
    # every cycle ("start" = the student as it was before cycle 1, or a
    # checkpoint path), so current-minus-reference on identical rows washes
    # out the rollout-to-rollout noise of the absolute numbers.
    ref_nets, ref_name = None, cfg.get("holdout_reference")
    if ref_name:
        if ref_name == "start":
            import copy
            ref_nets = copy.deepcopy((trainer.decision_net, trainer.execution_net))   # one call keeps shared modules shared
            for m in ref_nets:
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        else:
            ref_nets = teacher_nets(ref_name, trainer.separate_value_net, trainer.device)
    teachers = {pid: teacher_nets(path, trainer.separate_value_net, trainer.device) for pid, path in cfg["teachers"].items()}
    phases = tuple(sorted(teachers))
    bs = int(cfg["batch_size"])
    params = list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(cfg["lr"]), eps=1e-5)
    log.info(f"Distill pretrain: teachers {cfg['teachers']}, role_bc={role_bc} (bc_label_smoothing={smoothing}, "
             f"role-BC loss overrides {overrides}), "
             f"label_smoothing={cfg['label_smoothing']}, "
             f"counterfactual_kick_only_phases={list(cfg['counterfactual_kick_only_phases'])}, "
             f"counterfactual_skip_goalkeeper={bool(cfg.get('counterfactual_skip_goalkeeper'))}, "
             f"aggregate_cycles={cfg['aggregate_cycles']}, holdout_episode_fraction={hold_frac} "
             f"(reference {ref_name if ref_nets is not None else None}), "
             f"counterfactual={cfg['counterfactual']} "
             f"(weight {cfg['counterfactual_weight']}), kl_coef={cfg['kl_coef']}, value_coef={cfg['value_coef']}, "
             f"{cfg['num_rollouts']} cycle(s) x {cfg['rollout_steps']} steps x {cfg['epochs']} epoch(s), lr={cfg['lr']} "
             f"(linear warmup over {cfg['lr_warmup_steps']} steps)")
    pool = trainer._spawn_value_pretrain_workers(phase_id)
    report: dict = {}
    step = 0
    keep = max(1, int(cfg["aggregate_cycles"]))
    history: list = []          # previous cycles' data (as_history), newest last
    try:
        for cycle in range(int(cfg["num_rollouts"])):
            log.info(f"=== Distill pretrain: cycle {cycle + 1}/{cfg['num_rollouts']} ===")
            batch, _stats = trainer._collect_value_pretrain_rollout(
                None, int(cfg["rollout_steps"]), phase_id, use_gae=False, pool=pool)
            obs = {k[len("obs/"):]: v for k, v in batch.items() if k.startswith("obs/")}
            log.info("  [distill pretrain] rollout: " + composition(obs, batch.get("track_ids"), phases, _stats))
            returns = batch["returns"].float()
            ret_std = float(returns.std().clamp(min=1.0))

            def prepare(o: dict, r: torch.Tensor, b: dict) -> dict:
                """Rows to fit / report on: copies, teacher anchors, soft actions, physics features."""
                cf_rows = ~goalkeeper_rows(o) if cfg.get("counterfactual_skip_goalkeeper") else None
                d = build_copies(o, r, phases, bool(cfg["counterfactual"]), float(cfg["counterfactual_weight"]),
                                 counterfactual_rows=cf_rows)
                attach_role_bc(d, b, role_bc)
                d["anchor"] = smooth_bernoulli_targets(
                    kick_only_counterfactuals(trainer, d, teacher_anchor(trainer, d["obs"], d["target_phase"], teachers, bs),
                                              teachers, cfg["counterfactual_kick_only_phases"], bs),
                    float(cfg["label_smoothing"]))
                d["actions"] = soft_actions(d["anchor"], len(d["returns"]))
                phys = trainer._precompute_physics_full({f"obs/{k}": v for k, v in d["obs"].items()}, bs)
                d["obs"].update({k[len("obs/"):]: v for k, v in phys.items()})
                return d

            # Held-out episodes (holdout_episode_fraction > 0): whole episodes of
            # this rollout kept out of the fit (and of later cycles' aggregation),
            # reported after every epoch, so the fit's overfitting is visible
            # separately from the next rollout's distribution shift.
            hold = None
            hold_rows = (holdout_episode_mask(batch.get("track_ids") or None, batch["dones"], hold_frac, hold_gen)
                         if hold_frac > 0 else None)
            if hold_rows is not None and bool(hold_rows.any()):
                fit_rows = ~hold_rows
                hold = prepare({k: v[hold_rows] for k, v in obs.items()}, returns[hold_rows], {})
                data = prepare({k: v[fit_rows] for k, v in obs.items()}, returns[fit_rows], {})
                log.info(f"  [distill pretrain] held out {int(hold_rows.sum())} of {len(hold_rows)} rollout rows "
                         f"(whole episodes, fraction {hold_frac}): not fitted, reported per epoch")
            else:
                data = prepare(obs, returns, batch)
            del batch, obs
            n_new = len(data["returns"])
            log.info(f"  [distill pretrain] {n_new} rows ({int((~data['is_cf']).sum())} real, {int(data['is_cf'].sum())} counterfactual)"
                     f"  returns std={ret_std:.2f}")
            log_report(f"cycle {cycle + 1} before", group_report(trainer, data, bs))
            if ref_nets is not None:
                # The whole fresh rollout (every row, not the first 40k per group),
                # scored for the live student and the frozen reference on the same rows.
                cur_all = group_report(trainer, data, bs, max_rows=1 << 62)
                ref_all = group_report(trainer, data, bs, max_rows=1 << 62, nets=ref_nets)
                log.info(f"  [distill pretrain reference] cycle {cycle + 1} before fit, whole rollout, current: "
                         + holdout_line(cur_all))
                log.info(f"  [distill pretrain reference] cycle {cycle + 1} before fit, whole rollout, reference ({ref_name}): "
                         + holdout_line(ref_all))
                log.info(f"  [distill pretrain reference] cycle {cycle + 1} before fit, whole rollout, current - reference: "
                         + reference_delta_line(cur_all, ref_all))
            ref_report = hold_report = None
            if hold is not None:
                log.info(f"  [distill pretrain holdout] cycle {cycle + 1} before fit: "
                         + holdout_line(group_report(trainer, hold, bs)))
                if ref_nets is not None:
                    ref_report = group_report(trainer, hold, bs, nets=ref_nets)
                    log.info(f"  [distill pretrain holdout] cycle {cycle + 1} reference ({ref_name}): "
                             + holdout_line(ref_report))
            if role_bc:
                log.info(f"  [distill pretrain] role BC rows: {int((data['bc_w'] > 0).sum())}  "
                         f"bc loss before={bc_report_line(trainer, data, bs, smoothing)}")
            # DAgger aggregation (aggregate_cycles > 1): fit on this cycle's rows
            # plus the last (aggregate_cycles - 1) cycles' rows.
            fit = concat_data(history + [data])
            n = len(fit["returns"])
            if history:
                log.info(f"  [distill pretrain] fitting {n} rows: this cycle {n_new} + {len(history)} earlier cycle(s) "
                         f"{n - n_new} (KL only: their returns are another student's)")
            for epoch in range(int(cfg["epochs"])):
                perm = torch.randperm(n)
                vals, kls, bcs = [], [], []
                for s in range(0, n, bs):
                    mb = _slice(fit, perm[s:s + bs], trainer.device)
                    loss, vloss, kl, bcl = minibatch_loss(trainer, mb, ret_std, float(cfg["kl_coef"]),
                                                          float(cfg["value_coef"]), smoothing)
                    for g in opt.param_groups:
                        g["lr"] = warmup_lr(float(cfg["lr"]), step, int(cfg["lr_warmup_steps"]))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, trainer.max_grad_norm)
                    opt.step()
                    step += 1
                    vals.append(float(vloss.detach()))
                    kls.append(float(kl.detach()))
                    bcs.append(float(bcl.detach()))
                log.info(f"  [distill pretrain] epoch {epoch + 1}/{cfg['epochs']}: kl={np.mean(kls):.4f}  bc(weighted)={np.mean(bcs):.4f}  "
                         f"value_mse/var={np.mean(vals):.4f} (rmse {ret_std * math.sqrt(max(np.mean(vals), 0.0)):.2f})")
                if hold is not None:
                    hold_report = group_report(trainer, hold, bs)
                    log.info(f"  [distill pretrain holdout] cycle {cycle + 1} epoch {epoch + 1}/{cfg['epochs']}: "
                             + holdout_line(hold_report))
            if ref_report is not None and hold_report is not None:
                log.info(f"  [distill pretrain holdout] cycle {cycle + 1} vs reference ({ref_name}), same held-out rows: "
                         + reference_delta_line(hold_report, ref_report))
            report = group_report(trainer, data, bs)
            log_report(f"cycle {cycle + 1} after", report)
            if role_bc:
                log.info(f"  [distill pretrain] bc loss after={bc_report_line(trainer, data, bs, smoothing)}")
            del fit
            if keep > 1:
                history = (history + [as_history(data)])[-(keep - 1):]
            del data
            if checkpoint_path is not None:
                trainer._save_checkpoint_to(checkpoint_path)
                log.info(f"  [distill pretrain] checkpoint saved: {checkpoint_path}")
    finally:
        trainer._close_value_pretrain_workers(pool)
    return report
