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
            "rollout_steps": 540000, "num_rollouts": 4, "aggregate_cycles": 1}


_DATA_TENSORS = ("returns", "target_phase", "value_w", "kl_w", "is_cf")
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


def build_copies(obs: dict, returns: torch.Tensor, phases: tuple[int, int], counterfactual: bool,
                 counterfactual_weight: float) -> dict:
    """The rows to fit: the real copy (own phase, value target) and, if on,
    the task-id-swapped copy. Returns {"obs", "returns", "target_phase",
    "value_w", "kl_w", "is_cf"}; every row must carry one of ``phases``' task ids."""
    from footballcoach.ai.ppo.distill import phase_row_mask
    pa, pb = phases
    in_a, in_b = phase_row_mask(obs["global_feat"], pa), phase_row_mask(obs["global_feat"], pb)
    if not bool((in_a ^ in_b).all()):
        raise ValueError("distill_pretrain: every row must carry exactly one of the two teachers' task ids")
    real_phase = torch.where(in_a, torch.full_like(in_a, pa, dtype=torch.long), torch.full_like(in_a, pb, dtype=torch.long))
    n = len(returns)
    parts = [(obs, real_phase, torch.ones(n), torch.ones(n), torch.zeros(n, dtype=torch.bool))]
    if counterfactual:
        cf_obs = dict(obs)
        cf_obs["global_feat"] = swap_task_ids(obs["global_feat"], pa, pb)
        other = torch.where(real_phase == pa, torch.full_like(real_phase, pb), torch.full_like(real_phase, pa))
        parts.append((cf_obs, other, torch.zeros(n), torch.full((n,), float(counterfactual_weight)),
                      torch.ones(n, dtype=torch.bool)))
    return {
        "obs": {k: torch.cat([p[0][k] for p in parts]) for k in obs},
        "returns": torch.cat([returns] * len(parts)),
        "target_phase": torch.cat([p[1] for p in parts]),
        "value_w": torch.cat([p[2] for p in parts]),
        "kl_w": torch.cat([p[3] for p in parts]),
        "is_cf": torch.cat([p[4] for p in parts]),
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


def _student_forward(trainer, mb_obs: dict):
    from footballcoach.ai.ppo.trainer_common import _ai_types
    sat, oat = _ai_types(mb_obs)
    d = trainer.decision_net(
        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"], mb_obs["ball_feat"], mb_obs["global_feat"],
        sat, oat, ball_physics_full=mb_obs.get("ball_physics_full"),
        self_physics_full=mb_obs.get("self_physics_full"), other_physics_full=mb_obs.get("other_physics_full"),
    )
    e = trainer.execution_net(
        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"], mb_obs["ball_feat"], mb_obs["global_feat"],
        d, sat, oat,
    )
    return d, e


def minibatch_loss(trainer, mb: dict, ret_std: float, kl_coef: float, value_coef: float) -> tuple:
    """(loss, value_mse_norm, kl) on one minibatch dict of device tensors."""
    d, e = _student_forward(trainer, mb["obs"])
    globals_ = {k: getattr(trainer.execution_net, k).detach() for k in _SPREAD_KEYS}
    kl, _ = trainer._ppg_kl_penalty(
        d, e, mb["anchor"], globals_, mb["actions"], mb["obs"]["exists_mask"],
        row_weights=mb["kl_w"], detach_spread=True, soft_action_weights=True, forward_kl=True,
    )
    vw = mb["value_w"]
    sq = (e.value.squeeze(-1) - mb["returns"]) ** 2
    vloss = (sq * vw).sum() / vw.sum().clamp(min=1.0) / (ret_std ** 2)
    return kl_coef * kl + value_coef * vloss, vloss, kl


def _slice(data: dict, idx: torch.Tensor, device) -> dict:
    return {
        "obs": {k: v[idx].to(device) for k, v in data["obs"].items()},
        "anchor": {k: v[idx].to(device) for k, v in data["anchor"].items()},
        "actions": {k: v[idx].to(device) for k, v in data["actions"].items()},
        "returns": data["returns"][idx].to(device),
        "value_w": data["value_w"][idx].to(device),
        "kl_w": data["kl_w"][idx].to(device),
    }


def group_report(trainer, data: dict, batch_size: int, max_rows: int = 40000) -> dict:
    """{(phase of the row, "real"/"cf"): {"kl", "agree": {...}}} on up to
    ``max_rows`` rows per group, no grad."""
    from footballcoach.ai.ppo.distill import phase_row_mask
    from footballcoach.ai.ppo.ppo_update import _DISTILL_AGREE_KEYS, _distill_agreement
    out = {}
    gf = data["obs"]["global_feat"]
    for pid in sorted(set(data["target_phase"].tolist())):
        for cf in (False, True):
            rows = torch.nonzero((data["target_phase"] == pid) & (data["is_cf"] == cf)).squeeze(-1)
            if len(rows) == 0:
                continue
            rows = rows[:max_rows]
            kls, agrees, n = [], [], 0
            heads: dict[str, float] = {}
            with torch.no_grad():
                for s in range(0, len(rows), batch_size):
                    idx = rows[s:s + batch_size]
                    mb = _slice(data, idx, trainer.device)
                    d, e = _student_forward(trainer, mb["obs"])
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
    teachers = {pid: teacher_nets(path, trainer.separate_value_net, trainer.device) for pid, path in cfg["teachers"].items()}
    phases = tuple(sorted(teachers))
    bs = int(cfg["batch_size"])
    params = list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(cfg["lr"]), eps=1e-5)
    log.info(f"Distill pretrain: teachers {cfg['teachers']}, label_smoothing={cfg['label_smoothing']}, counterfactual={cfg['counterfactual']} "
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
            data = build_copies(obs, returns, phases, bool(cfg["counterfactual"]), float(cfg["counterfactual_weight"]))
            del batch, obs
            data["anchor"] = smooth_bernoulli_targets(
                teacher_anchor(trainer, data["obs"], data["target_phase"], teachers, bs), float(cfg["label_smoothing"]))
            data["actions"] = soft_actions(data["anchor"], len(data["returns"]))
            phys = trainer._precompute_physics_full({f"obs/{k}": v for k, v in data["obs"].items()}, bs)
            data["obs"].update({k[len("obs/"):]: v for k, v in phys.items()})
            n_new = len(data["returns"])
            log.info(f"  [distill pretrain] {n_new} rows ({int((~data['is_cf']).sum())} real, {int(data['is_cf'].sum())} counterfactual)"
                     f"  returns std={ret_std:.2f}")
            log_report(f"cycle {cycle + 1} before", group_report(trainer, data, bs))
            # DAgger aggregation (aggregate_cycles > 1): fit on this cycle's rows
            # plus the last (aggregate_cycles - 1) cycles' rows.
            fit = concat_data(history + [data])
            n = len(fit["returns"])
            if history:
                log.info(f"  [distill pretrain] fitting {n} rows: this cycle {n_new} + {len(history)} earlier cycle(s) "
                         f"{n - n_new} (KL only: their returns are another student's)")
            for epoch in range(int(cfg["epochs"])):
                perm = torch.randperm(n)
                vals, kls = [], []
                for s in range(0, n, bs):
                    mb = _slice(fit, perm[s:s + bs], trainer.device)
                    loss, vloss, kl = minibatch_loss(trainer, mb, ret_std, float(cfg["kl_coef"]), float(cfg["value_coef"]))
                    for g in opt.param_groups:
                        g["lr"] = warmup_lr(float(cfg["lr"]), step, int(cfg["lr_warmup_steps"]))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, trainer.max_grad_norm)
                    opt.step()
                    step += 1
                    vals.append(float(vloss.detach()))
                    kls.append(float(kl.detach()))
                log.info(f"  [distill pretrain] epoch {epoch + 1}/{cfg['epochs']}: kl={np.mean(kls):.4f}  "
                         f"value_mse/var={np.mean(vals):.4f} (rmse {ret_std * math.sqrt(max(np.mean(vals), 0.0)):.2f})")
            report = group_report(trainer, data, bs)
            log_report(f"cycle {cycle + 1} after", report)
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
