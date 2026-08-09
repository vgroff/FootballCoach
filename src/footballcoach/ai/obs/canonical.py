"""Canonical AI frame: thin wrapper applied at the network forward boundary.

``obs/encoder.py`` and recorded BC ``.npz`` data are ALWAYS in raw
world/engine-frame coordinates (``Team.LEFT`` attacks +x, ``Team.RIGHT``
attacks -x, per ``entities/player.py``/``engine/offside.py`` convention).
Nothing about that changes.

This module implements a separate, permanent mirror that both networks
consume: on the way IN, every x-signed observation field is negated for a
``Team.RIGHT`` observer so the network always sees "my own team attacks
+x"; on the way OUT, the network's x-signed action outputs (move direction,
kick direction, move-region target) are negated back to world frame before
they touch engine state (``Player.desired_direction``, ``kick_direct()``,
...). The transform is its own inverse (a reflection), so both directions
use the same ``mirror_x()`` helper.

Why a separate module instead of baking this into ``encoder.py``/``bc.py``:
  - Recorded ``.npz`` demonstrations stay in plain world-frame coordinates,
    matching match logs and UI replays, and never need re-recording if the
    convention changes.
  - There is exactly ONE implementation of the mirror, reused by live
    rollout collection, BC dataset consumption, and action application —
    instead of the mirror being hand-duplicated at each of those sites
    (which is exactly the kind of drift already documented as a repeated
    bug pattern in this codebase — see ai/knowledge.md).

The per-sample sign is derived from ``PlayerFeatures.attacking_direction``
(already present on every self-feature vector: +1.0 for ``Team.LEFT``,
-1.0 for ``Team.RIGHT`` — see ``obs/encoder.py``) rather than requiring the
caller to separately plumb through ``player.team``. This makes
canonicalization purely a function of the observation itself.

Which fields get mirrored on the way in is exactly the field set
``obs/augment.py``'s ``PLAYER_FLIP_X_IDX``/``BALL_FLIP_X_IDX`` already
enumerate — this module imports and reuses those same index lists rather
than re-deriving them, so the two stay in sync by construction.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch

from footballcoach.ai.obs.augment import (
    BALL_FLIP_X_IDX,
    PLAYER_FLIP_X_IDX,
    _BC_DIR_X_COL,
    _BC_KICK_DIR_X_COL,
    _BC_KICK_SPIN_Y_COL,
    _BC_KICK_SPIN_Z_COL,
    _BC_REGION_X_COL,
)
from footballcoach.ai.obs.schema import PlayerFeatures

#: Index of PlayerFeatures.attacking_direction within a self_feat/other_feat
#: row — the per-sample canonicalization sign lives here (+1.0 Team.LEFT,
#: -1.0 Team.RIGHT). Derived from the schema so it can never drift.
X_SIGN_FIELD_IDX: int = [f.name for f in fields(PlayerFeatures)].index("attacking_direction")


def team_x_sign(team) -> float:
    """+1.0 for ``Team.LEFT``, -1.0 for ``Team.RIGHT``.

    Convenience for call sites that only have a ``Player``/``Team`` handy
    (e.g. ``apply_nn_action.py``) and not yet an encoded observation.
    """
    from footballcoach.entities.player import Team
    return 1.0 if team == Team.LEFT else -1.0


def x_sign_of(self_feat) -> "torch.Tensor | np.ndarray | float":
    """Extract the per-sample canonicalization sign from ``self_feat``.

    Works on a single unbatched vector (shape ``(PLAYER_FEATURE_DIM,)``) or
    a batch (shape ``(N, PLAYER_FEATURE_DIM)``) — returns a scalar or a
    ``(N,)`` tensor/array respectively, matching the input type.
    """
    return self_feat[..., X_SIGN_FIELD_IDX]


def _mirror_columns(tensor: torch.Tensor, cols: list[int], x_sign) -> torch.Tensor:
    """Return a COPY of ``tensor`` with ``cols`` multiplied by ``x_sign``.

    ``tensor`` may be ``(D,)``, ``(N, D)``, or ``(N, S, D)`` (other-player
    slots). ``x_sign`` may be a python float (unbatched) or a ``(N,)``
    tensor (batched) — broadcast against the leading dims.
    """
    out = tensor.clone()
    if tensor.dim() == 1:
        out[cols] = out[cols] * x_sign
    elif tensor.dim() == 2:
        # x_sign: scalar or (N,) -> (N, 1) for broadcasting against (N, len(cols))
        xs = x_sign if np.ndim(x_sign) == 0 else torch.as_tensor(x_sign, dtype=tensor.dtype).unsqueeze(-1)
        out[:, cols] = out[:, cols] * xs
    elif tensor.dim() == 3:
        xs = x_sign if np.ndim(x_sign) == 0 else torch.as_tensor(x_sign, dtype=tensor.dtype).view(-1, 1, 1)
        out[:, :, cols] = out[:, :, cols] * xs
    else:
        raise ValueError(f"Unsupported tensor rank {tensor.dim()} for canonicalization")
    return out


def canonicalize_obs(
    self_feat: torch.Tensor,
    other_feat: torch.Tensor,
    ball_feat: torch.Tensor,
    x_sign=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | float"]:
    """Mirror x-signed observation fields into the canonical AI frame.

    Accepts unbatched (``self_feat: (D,)``, ``other_feat: (S, D)``,
    ``ball_feat: (B,)``) or batched (leading ``N`` dim) tensors — shape
    handling matches ``_mirror_columns()``.

    Args:
        self_feat/other_feat/ball_feat: raw world-frame observation tensors
            straight out of ``obs/encoder.py`` (or a stacked batch thereof).
        x_sign: Optional precomputed sign (scalar or ``(N,)`` tensor). If
            omitted (the common case), it is derived from ``self_feat``'s
            ``attacking_direction`` field via ``x_sign_of()`` — this is
            always correct since it's the same field written by the
            encoder for this exact observation, with no risk of staleness.

    Returns:
        (self_feat_canon, other_feat_canon, ball_feat_canon, x_sign) — the
        returned ``x_sign`` is what was used, so callers can reuse it to
        decanonicalize the resulting action without recomputing it.
    """
    if x_sign is None:
        x_sign = x_sign_of(self_feat)
    self_c = _mirror_columns(self_feat, PLAYER_FLIP_X_IDX, x_sign)
    other_c = _mirror_columns(other_feat, PLAYER_FLIP_X_IDX, x_sign)
    ball_c = _mirror_columns(ball_feat, BALL_FLIP_X_IDX, x_sign)
    return self_c, other_c, ball_c, x_sign


class CanonicalNetworkWrapper(torch.nn.Module):
    """Transparent wrapper: canonicalizes ``self_feat``/``other_feat``/
    ``ball_feat`` before delegating to the wrapped network's real
    ``forward()``. Every existing call site
    (``self.decision_net(sf, of, em, bf, gf, ...)`` etc.) keeps working
    completely unchanged — this is the ONLY place the canonical-AI-frame
    transform is applied on the input side.

    Network OUTPUTS (``DecisionHeadsRaw``/``ExecutionHeadsRaw``) are
    returned AS-IS, i.e. still in canonical frame — this is intentional:
    log_prob/loss computations that compare network output against other
    canonical-frame quantities (BC labels via ``canonicalize_bc_labels()``,
    stored raw action samples) must stay self-consistent in canonical
    frame. Only the final physical action that actually touches engine
    state gets decanonicalized, at the one boundary in
    ``PPOTrainer._sample_action()`` (via ``mirror_x()``).

    Transparently forwards attribute access (``.value_head``,
    ``.move_dir_log_std``, etc.) and ``state_dict()``/``parameters()`` to
    the wrapped module via ``__getattr__``, so existing code that reaches
    through ``self.execution_net.value_head`` or similar continues to work
    without modification.
    """

    def __init__(self, wrapped: torch.nn.Module):
        super().__init__()
        # Registered as a real submodule (not just an attribute) so
        # .parameters()/.state_dict()/.to(device)/.train()/.eval() on the
        # wrapper correctly recurse into the wrapped network.
        self._wrapped = wrapped

    def forward(self, self_feat, other_feat, *rest, **kwargs):
        # rest[0] may be exists_mask (decision/value net) — ball_feat is
        # always rest[1] for DecisionNetwork/ExecutionNetwork's shared
        # (self_feat, other_feat, exists_mask, ball_feat, global_feat, ...)
        # signature. Canonicalizing only self_feat/other_feat/ball_feat
        # (exists_mask/global_feat/decision_heads have no geometric content).
        ball_feat = rest[1]
        self_feat_c, other_feat_c, ball_feat_c, _ = canonicalize_obs(self_feat, other_feat, ball_feat)
        new_rest = (rest[0], ball_feat_c) + rest[2:]
        return self._wrapped(self_feat_c, other_feat_c, *new_rest, **kwargs)

    def __getattr__(self, name):
        # nn.Module.__getattr__ already checks self._modules (where
        # self._wrapped lives) before raising — this override only adds a
        # SECOND fallback onto the wrapped module's own non-module attrs
        # (e.g. execution_net.move_dir_log_std, a plain nn.Parameter, IS
        # found via the first fallback already; this covers plain python
        # attributes some call sites might reach for).
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = super().__getattr__("_wrapped")
            return getattr(wrapped, name)

    def state_dict(self, *args, **kwargs):
        # Delegate directly to the wrapped module so checkpoint keys are
        # IDENTICAL to pre-wrapper checkpoints (no "_wrapped." prefix) —
        # critical for load/save compatibility with every existing .pt file.
        return self._wrapped.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._wrapped.load_state_dict(*args, **kwargs)


def canonicalize_bc_labels(bc_labels: torch.Tensor, x_sign) -> torch.Tensor:
    """Mirror a batch of world-frame BC label rows (see ``ai/ppo/bc.py``'s
    ``BC_LABEL_DIM`` layout) into the canonical AI frame, matching
    ``canonicalize_obs()``'s treatment of the corresponding observation.

    Must be applied to labels whenever they're compared against network
    outputs produced from a canonicalized observation (BC loss computation)
    — labels themselves are generated/stored in world frame (see
    ``ai/ppo/bc.py`` module docstring), same as ``obs/encoder.py``.

    ``x_sign``: scalar or ``(N,)`` tensor, one entry per row of ``bc_labels``.
    """
    out = bc_labels.clone()
    xs = x_sign if torch.is_tensor(x_sign) and x_sign.dim() > 0 else x_sign
    out[:, _BC_DIR_X_COL] = out[:, _BC_DIR_X_COL] * xs
    out[:, _BC_REGION_X_COL] = out[:, _BC_REGION_X_COL] * xs
    out[:, _BC_KICK_DIR_X_COL] = out[:, _BC_KICK_DIR_X_COL] * xs
    out[:, _BC_KICK_SPIN_Y_COL] = out[:, _BC_KICK_SPIN_Y_COL] * xs
    out[:, _BC_KICK_SPIN_Z_COL] = out[:, _BC_KICK_SPIN_Z_COL] * xs
    return out


def mirror_x(vec, x_sign):
    """Negate the x-component (index 0) of a 2D/3D direction-or-position
    vector (or batch thereof) by ``x_sign``. Works for numpy or torch,
    single vectors ``(2,)``/``(3,)`` or batches ``(N, 2)``/``(N, 3)``.

    This is the SAME transform in both directions (world->canonical on
    observations, canonical->world on actions) since a reflection is its
    own inverse — one helper serves both call sites.
    """
    if vec is None:
        return None
    out = vec.copy() if isinstance(vec, np.ndarray) else vec.clone()
    if out.ndim == 1:
        out[0] = out[0] * x_sign
    else:
        xs = x_sign if np.ndim(x_sign) == 0 else (
            np.asarray(x_sign).reshape(-1) if isinstance(out, np.ndarray)
            else torch.as_tensor(x_sign, dtype=out.dtype).view(-1)
        )
        out[:, 0] = out[:, 0] * xs
    return out
