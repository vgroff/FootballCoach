"""Pure helpers for ``PPOTrainer.consistency_refit`` (flip_y consistency distillation).

The idea: for every state, pick the orientation in which the observer's own y is >= 0 (the
"primary" orientation), ask a frozen teacher what it does there, and train the student so that the
primary state AND its y-mirror both reproduce that answer (mirrored for the mirror image). Both
orientations of a state therefore share one target, which makes the student flip_y-consistent, with
the observer's own side as the tie-break for states where the teacher itself answers a state and its
mirror image differently.

Everything here works on CPU tensors / plain dicts so it can be unit-tested without a trainer.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Optional

import torch

from footballcoach.ai.obs.augment import BALL_FLIP_Y_IDX, PLAYER_FLIP_Y_IDX
from footballcoach.ai.obs.canonical import _mirror_columns
from footballcoach.ai.obs.schema import PlayerFeatures

#: Index of PlayerFeatures.pos_y within a self_feat row.
POS_Y_IDX: int = [f.name for f in fields(PlayerFeatures)].index("pos_y")

#: Anchor/head-output keys that are Bernoulli logits (categorical *_target_logits are NOT included).
BERNOULLI_KEYS: tuple[str, ...] = (
    "shoot_logit", "pass_logit", "move_logit", "tackle_logit", "get_possession_raw", "mark_logit",
    "hold_position_logit", "exec_move_logit", "sprint_logit", "kick_logit", "tackle_attempt_logit",
)

_OBS_SELF, _OBS_OTHER, _OBS_BALL = "obs/self_feat", "obs/other_feat", "obs/ball_feat"


def flip_y_obs(obs: dict, mask: Optional[torch.Tensor] = None) -> dict:
    """Return a shallow copy of ``obs`` with the y-signed self/other/ball features mirrored.

    Rows where ``mask`` is True are mirrored (all rows when ``mask`` is None). The transform is
    its own inverse. Uses the same index lists as ``augment.py``'s flip_y.
    """
    n = obs[_OBS_SELF].shape[0]
    sign = (
        torch.full((n,), -1.0) if mask is None
        else torch.where(mask, torch.tensor(-1.0), torch.tensor(1.0))
    )
    out = dict(obs)
    out[_OBS_SELF] = _mirror_columns(obs[_OBS_SELF], PLAYER_FLIP_Y_IDX, sign)
    out[_OBS_OTHER] = _mirror_columns(obs[_OBS_OTHER], PLAYER_FLIP_Y_IDX, sign)
    out[_OBS_BALL] = _mirror_columns(obs[_OBS_BALL], BALL_FLIP_Y_IDX, sign)
    return out


def orientation_pair(obs: dict) -> tuple[dict, dict]:
    """(primary, mirror): primary has the observer at y >= 0 in every row; mirror is its y-flip."""
    primary = flip_y_obs(obs, obs[_OBS_SELF][:, POS_Y_IDX] < 0)
    return primary, flip_y_obs(primary, None)


def mirror_anchor_y(anchor: dict) -> dict:
    """y-mirror a head-output snapshot (``_ppg_snapshot_anchor`` layout).

    Direction vectors negate y; kick_spin is a pseudovector (-wx, wy, -wz); logits, kick_power and
    the slot-indexed categorical logits are flip-invariant.
    """
    out = dict(anchor)
    for key in ("move_direction", "kick_direction"):
        if key in out:
            v = out[key].clone()
            v[:, 1] = -v[:, 1]
            out[key] = v
    if "kick_spin" in out:
        v = out["kick_spin"].clone()
        v[:, [0, 2]] = -v[:, [0, 2]]
        out["kick_spin"] = v
    return out


def bound_bernoulli_anchor(anchor: dict, bound: float) -> dict:
    """Clamp every Bernoulli logit in ``anchor`` to ``[-bound, bound]`` (bound <= 0 disables)."""
    if bound <= 0.0:
        return dict(anchor)
    return {k: (v.clamp(-bound, bound) if k in BERNOULLI_KEYS else v) for k, v in anchor.items()}


def concat_anchors(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]], dim=0) for k in a}


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


def flip_consistency_metrics(head_p: dict, head_q: dict, moving: Optional[torch.Tensor] = None) -> dict[str, float]:
    """How consistent is a policy between a state (P) and its y-mirror (Q)? Inputs are head-output
    snapshots of the SAME network on the primary rows and on their mirror images.

    ``moving`` (bool mask of rows whose sampled exec_move was 1) restricts the move-direction
    statistics to rows where that head is actually used.
    """
    q = mirror_anchor_y(head_q)
    out: dict[str, float] = {}
    ang_p = torch.atan2(head_p["move_direction"][:, 1], head_p["move_direction"][:, 0])
    ang_q = torch.atan2(q["move_direction"][:, 1], q["move_direction"][:, 0])
    err = torch.rad2deg(_wrap_pi(ang_p - ang_q).abs())
    if moving is not None and bool(moving.any()):
        err = err[moving]
    out["move_dir_err_p50_deg"] = float(err.quantile(0.5)) if len(err) else float("nan")
    out["move_dir_err_p90_deg"] = float(err.quantile(0.9)) if len(err) else float("nan")
    out["move_dir_err_gt90_pct"] = float((err > 90).float().mean() * 100) if len(err) else float("nan")
    for key in ("exec_move_logit", "sprint_logit", "kick_logit", "tackle_attempt_logit", "move_logit", "get_possession_raw"):
        if key not in head_p:
            continue
        lp, lq = head_p[key].reshape(-1), q[key].reshape(-1)
        short = key.replace("_logit", "")
        out[f"{short}_sign_disagree_pct"] = float(((lp > 0) != (lq > 0)).float().mean() * 100)
        out[f"{short}_abs_logit_diff_p99"] = float((lp - lq).abs().quantile(0.99))
    return out


def bounded_share_pct(head: dict, bound: float) -> dict[str, float]:
    """Percent of rows whose Bernoulli logit magnitude is within ``bound`` (per head)."""
    return {
        k.replace("_logit", ""): float((head[k].abs() <= bound + 1e-6).float().mean() * 100)
        for k in ("exec_move_logit", "sprint_logit", "kick_logit", "tackle_attempt_logit", "move_logit", "get_possession_raw")
        if k in head
    }
