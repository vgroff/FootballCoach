"""Per-phase, per-role BC: which rows get BC from which schedule.

A role-based phase switches BC on for a role in ai_config
``curriculum.<phase config_key>.bc``::

    "bc": {
      "attacker":   {"kick": {"coeff_start": 0.5, "coeff_end": 0.0, "anneal_fraction": 0.3}},
      "goalkeeper": {"other": {...}, "kick": {...}, "tackle": {...}}
    }

Groups are bc_loss_from_tensor's split: "kick" (kick_this_tick / direction /
power / spin), "tackle" (tackle_attempt) and "other" (everything else:
decision heads, movement, region). A group left out has coefficient 0 for
that role. Each schedule anneals linearly from coeff_start to coeff_end over
the first anneal_fraction of training, exactly like the global bc.aux_coeff_*
schedules. Default: no "bc" entry = no labels, no BC.

Every training row carries a BC SOURCE id: 0 for every row the global
schedules govern (phase 1, and any role without BC), i for the i-th
configured (phase, role). The PPO update gathers each row's three group
coefficients by its source (coeff_table), so a phase-2 attacker's kick BC
never touches a phase-1 row, nor the reverse. Labels for a configured role are
made by that role's own rules AI (ai/bc/role_experts.RulesExpert), for every
neural player in it, trainee or secondary.

The ids are a pure function of the config (sorted), so rollout workers and
the trainer agree on them without passing anything.
"""
from __future__ import annotations

from typing import Optional

GROUPS: tuple[str, ...] = ("other", "kick", "tackle")
_SCHEDULE_KEYS = {"coeff_start", "coeff_end", "anneal_fraction"}


def _phase_bc_config(phase) -> dict:
    from footballcoach.ai.config import load_ai_config
    if not phase.is_role_based:
        return {}
    cfg = load_ai_config().get("curriculum", {}).get(phase.config_key, {}).get("bc") or {}
    roles = set(phase.roles.values())
    for role, groups in cfg.items():
        if role.startswith("_"):
            continue
        if role not in roles:
            raise ValueError(f"{phase.name}: BC configured for unknown role {role!r} (roles: {sorted(roles)})")
        for group, sched in groups.items():
            if group.startswith("_"):
                continue
            if group not in GROUPS:
                raise ValueError(f"{phase.name}/{role}: unknown BC group {group!r} (groups: {GROUPS})")
            missing = _SCHEDULE_KEYS - set(sched)
            if missing:
                raise ValueError(f"{phase.name}/{role}/{group}: schedule needs {sorted(missing)}")
    return {r: {g: s for g, s in gs.items() if not g.startswith("_")}
            for r, gs in cfg.items() if not r.startswith("_") and gs}


def bc_roles(phase) -> dict[str, dict]:
    """role -> {group: schedule} for a role-based phase's BC-enabled roles."""
    return _phase_bc_config(phase)


def sources() -> list[tuple[int, str]]:
    """[(phase_id, role), ...] in id order (id = index + 1)."""
    from footballcoach.ai.curriculum.phases import ALL_PHASES
    out = []
    for phase in ALL_PHASES:
        if phase.is_role_based:
            out += [(phase.phase_id, role) for role in sorted(_phase_bc_config(phase))]
    return out


def source_id(phase_id: int, role: str) -> int:
    """The BC source id of (phase, role), 0 if it has no BC configured."""
    try:
        return sources().index((phase_id, role)) + 1
    except ValueError:
        return 0


def _anneal(sched: dict, progress: float) -> float:
    start, end, frac = float(sched["coeff_start"]), float(sched["coeff_end"]), float(sched["anneal_fraction"])
    if frac <= 0.0:
        return end
    t = min(progress / frac, 1.0)
    return start + (end - start) * t


def coeff_table(progress: float, global_coeffs: tuple[float, float, float]):
    """(n_sources + 1, 3) tensor of (other, kick, tackle) coefficients at this
    training progress: row 0 = the global schedules (as given), row i = the
    i-th configured (phase, role)."""
    import torch

    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    rows = [list(global_coeffs)]
    for phase_id, role in sources():
        groups = _phase_bc_config(PHASES_BY_ID[phase_id])[role]
        rows.append([_anneal(groups[g], progress) if g in groups else 0.0 for g in GROUPS])
    return torch.tensor(rows, dtype=torch.float32)


def describe(source: int) -> Optional[str]:
    """'phase2_1v2/attacker' for a source id (None for 0)."""
    if source == 0:
        return None
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    phase_id, role = sources()[source - 1]
    return f"{PHASES_BY_ID[phase_id].name}/{role}"
