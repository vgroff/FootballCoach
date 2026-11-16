"""Per-phase teacher distillation: keep a phase's behaviour while training others.

When a run trains several phases with a shared network, learning one can
overwrite another (phase-2 shooting leaked into phase 1: phase-1 wins vs rules
65% -> 17% after phase-2-only training, and a slow slide even at an 80/20 mix
with phase-2 kick BC on). This anchors a phase's rows to a frozen TEACHER
network's action distributions with the analytic per-head KL the PPG refit
already uses (``PPOTrainer._ppg_kl_penalty``), on the states the student
visits (on-policy, like DAgger), so the student can only change that phase's
behaviour against a penalty.

Config, keyed by phase id (generic: any phase, role-based or not)::

    "curriculum": {"distill": {"1": {
        "teacher": "checkpoints/migrated/phase1_run321_checkpoint763_migrated.pt",
        "coeff_start": 1.0, "coeff_end": 1.0, "anneal_fraction": 1.0,   # + optional hold_fraction
        "means_only": true
    }}}

- Rows of a phase are the ones whose global task-id one-hot is that phase's
  (``phase_row_mask``), so nothing new travels through the rollout buffer.
- ``means_only`` (default true): the KL pulls the student's state-dependent
  means / logits towards the teacher's; the continuous heads' spreads (kappa,
  log_std, global parameters shared with every phase) keep the student's own
  values and get no gradient from this term.
- The coefficient anneals like the per-role BC schedules (bc_sources._anneal).
- The teacher is loaded once per process, frozen, inference only.
- Logged per phase as ``distill [<phase>] kl=<mean KL> x<coeff>`` plus the
  largest per-head terms.
- Off (no "distill" entry) = no teacher, no extra forward pass, no change.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

_REQUIRED = {"teacher", "coeff_start", "coeff_end", "anneal_fraction"}
_OPTIONAL = {"hold_fraction", "means_only"}


def distill_config() -> dict[int, dict]:
    """{phase_id: schedule} of the phases with distillation configured (validated)."""
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    raw = load_ai_config().get("curriculum", {}).get("distill") or {}
    out: dict[int, dict] = {}
    for key, cfg in raw.items():
        if str(key).startswith("_"):
            continue
        pid = int(key)
        if pid not in PHASES_BY_ID:
            raise ValueError(f"distill: unknown phase {key!r}")
        missing = _REQUIRED - set(cfg)
        if missing:
            raise ValueError(f"distill[{pid}]: needs {sorted(missing)}")
        unknown = set(k for k in cfg if not k.startswith("_")) - _REQUIRED - _OPTIONAL
        if unknown:
            raise ValueError(f"distill[{pid}]: unknown key(s) {sorted(unknown)}")
        if not 0.0 <= float(cfg.get("hold_fraction", 0.0)) <= float(cfg["anneal_fraction"]):
            raise ValueError(f"distill[{pid}]: hold_fraction must be in [0, anneal_fraction]")
        out[pid] = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return out


def coeff(schedule: dict, progress: float) -> float:
    from footballcoach.ai.ppo.bc_sources import _anneal
    return _anneal(schedule, progress)


def task_id_column(phase_id: int) -> int:
    """Column of global_feat holding phase ``phase_id``'s task-id flag."""
    import dataclasses

    from footballcoach.ai.obs.schema import GlobalFeatures
    return [f.name for f in dataclasses.fields(GlobalFeatures)].index(f"task_id_{phase_id - 1}")


def phase_row_mask(global_feat, phase_id: int):
    """(N,) bool: the rows encoded with phase ``phase_id``'s task id."""
    return global_feat[:, task_id_column(phase_id)] > 0.5


@lru_cache(maxsize=4)
def _teacher_nets_cached(path: str, separate_value_net: bool, device_str: str):
    import torch
    from pathlib import Path

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    t = PPOTrainer.from_config(device=torch.device(device_str), inference_only=True,
                               separate_value_net=separate_value_net)
    t.load_checkpoint(Path(path), reset_optimizer=True)
    nets = (t.decision_net.eval(), t.execution_net.eval())
    for m in nets:
        for p in m.parameters():
            p.requires_grad_(False)
    return nets


def teacher_nets(path: str, separate_value_net: bool, device) -> tuple:
    """(decision_net, execution_net) of the frozen teacher at ``path``, loaded once."""
    return _teacher_nets_cached(str(path), bool(separate_value_net), str(device))


def active_terms(progress: float, separate_value_net: bool, device) -> Optional[list[dict]]:
    """[{phase_id, coeff, means_only, nets}] for this update, None when off."""
    cfg = distill_config()
    if not cfg:
        return None
    terms = []
    for pid, sched in sorted(cfg.items()):
        c = coeff(sched, progress)
        if c > 0.0:
            terms.append({"phase_id": pid, "coeff": c, "means_only": bool(sched.get("means_only", True)),
                          "nets": teacher_nets(sched["teacher"], separate_value_net, device)})
    return terms or None
