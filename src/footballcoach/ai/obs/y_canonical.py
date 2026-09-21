"""Y-canonical AI frame: an opt-in second mirror applied at the network forward boundary.

``canonical.py`` already mirrors x so the observer always attacks +x. This module does the same for y:
every row whose observer (``self_feat.pos_y``) is at y < 0 is mirrored about the x-axis before the network
sees it, and the network's y-signed outputs are mirrored back. The wrapped policy is therefore EXACTLY
flip_y-equivariant by construction (the observer's own side, y >= 0, is the tie-break), which makes flip_y
augmentation redundant: a flipped copy maps to the identical canonical state.

Cost: the policy is discontinuous across y = 0 wherever the underlying network is asymmetric.

Enabled by ``ppo.y_canonical`` (see ai_config.json). Uses the same index lists as ``augment.py``'s flip_y.

Frozen-physics-encoder passthrough tensors (``*_physics_full`` kwargs) are passed through untouched: the
trainer computes them from the y-canonical observation (``PPOTrainer._precompute_physics_full``). When they
are absent the wrapped network computes them from the mirrored observation itself.
"""
from __future__ import annotations

import dataclasses
from dataclasses import fields

import torch

from footballcoach.ai.obs.augment import BALL_FLIP_Y_IDX, PLAYER_FLIP_Y_IDX
from footballcoach.ai.obs.canonical import _mirror_columns
from footballcoach.ai.obs.schema import PlayerFeatures

#: Index of PlayerFeatures.pos_y within a self_feat row.
Y_POS_FIELD_IDX: int = [f.name for f in fields(PlayerFeatures)].index("pos_y")


def y_flip_mask(self_feat: torch.Tensor) -> torch.Tensor:
    """(N,) bool: rows to mirror (observer at y < 0). y == 0 keeps its orientation."""
    return self_feat[..., Y_POS_FIELD_IDX] < 0


def mirror_y_obs(self_feat, other_feat, ball_feat, mask):
    """Mirror the y-signed obs fields of the rows in ``mask`` (a self-inverse transform)."""
    sign = torch.where(mask, -1.0, 1.0).to(self_feat.dtype)
    return (
        _mirror_columns(self_feat, PLAYER_FLIP_Y_IDX, sign),
        _mirror_columns(other_feat, PLAYER_FLIP_Y_IDX, sign),
        _mirror_columns(ball_feat, BALL_FLIP_Y_IDX, sign),
    )


def _neg_cols(t: torch.Tensor, cols: list[int], mask: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    sign = torch.where(mask, -1.0, 1.0).to(t.dtype).unsqueeze(-1)
    out[..., cols] = out[..., cols] * sign
    return out


def flip_decision_heads_y(heads, mask):
    """Mirror the y-signed fields of a DecisionHeadsRaw for the rows in ``mask``."""
    return dataclasses.replace(
        heads,
        move_region_center=_neg_cols(heads.move_region_center, [1], mask),
        region_of_play_center=_neg_cols(heads.region_of_play_center, [1], mask),
    )


def flip_exec_heads_y(heads, mask):
    """Mirror the y-signed fields of an ExecutionHeadsRaw for the rows in ``mask``.

    kick_spin is a pseudovector: flip_y maps (wx, wy, wz) -> (-wx, wy, -wz), matching augment.py.
    """
    return dataclasses.replace(
        heads,
        move_direction=_neg_cols(heads.move_direction, [1], mask),
        move_direction_unnormalized=_neg_cols(heads.move_direction_unnormalized, [1], mask),
        kick_direction=_neg_cols(heads.kick_direction, [1], mask),
        kick_direction_unnormalized=_neg_cols(heads.kick_direction_unnormalized, [1], mask),
        kick_spin=_neg_cols(heads.kick_spin, [0, 2], mask),
    )


class YCanonicalNetworkWrapper(torch.nn.Module):
    """Wraps a (Canonical-wrapped) DecisionNetwork or ExecutionNetwork; see module docstring.

    ``role="decision"``: call convention ``(sf, of, em, bf, gf, sat, oat, **kw)``.
    ``role="execution"``: ``(sf, of, em, bf, gf, decision_heads, sat, oat, **kw)`` where ``decision_heads``
    arrive already mirrored back by the decision wrapper and are re-mirrored for the flipped rows here, so the
    wrapped execution net sees the heads its own decision net produced.
    """

    def __init__(self, wrapped: torch.nn.Module, role: str):
        super().__init__()
        if role not in ("decision", "execution"):
            raise ValueError(f"role must be 'decision' or 'execution', got {role!r}")
        self._wrapped = wrapped
        self._role = role

    def forward(self, self_feat, other_feat, *rest, **kwargs):
        mask = y_flip_mask(self_feat)
        if not bool(mask.any()):
            return self._wrapped(self_feat, other_feat, *rest, **kwargs)
        sf, of, bf = mirror_y_obs(self_feat, other_feat, rest[1], mask)
        new_rest = (rest[0], bf) + rest[2:]
        if self._role == "decision":
            return flip_decision_heads_y(self._wrapped(sf, of, *new_rest, **kwargs), mask)
        new_rest = new_rest[:3] + (flip_decision_heads_y(rest[3], mask),) + new_rest[4:]
        out = self._wrapped(sf, of, *new_rest, **kwargs)
        return out if torch.is_tensor(out) else flip_exec_heads_y(out, mask)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("_wrapped"), name)

    def state_dict(self, *args, **kwargs):
        return self._wrapped.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._wrapped.load_state_dict(*args, **kwargs)
