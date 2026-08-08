"""Geometric and slot-permutation augmentation for PPO training batches.

APPLIES TO ALL AI TRAINING IN THIS REPOSITORY (phases 1, 2, and all future
phases).  Any training loop that uses RolloutBuffer + PPOTrainer gets this
for free because augmentation is applied inside _ppo_update().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The football environment has two exact symmetries:

  flip_x  — reflect the pitch about the x-axis (negate all x-direction
             quantities: positions, velocities, headings, spin components,
             attacking directions, action direction vectors).
  flip_y  — reflect the pitch about the y-axis (negate all y-direction
             quantities).

Combined they give flip_xy (180° rotation), so four variants in total.

The observation encoding (relative positions, normalised velocities, etc.)
and reward function (possession bonus, ball-progress shaping, box-terminal)
are all invariant under these reflections.  Augmenting with flipped copies
teaches the network this invariance without changing environment dynamics.

Additionally, the other-player slots are randomly ordered (permutation
invariance via attention).  Producing n_slot_shuffles different orderings
per step gives the attention mechanism more diverse training signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH EXPANSION FACTOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  total multiplier = 4 (flips) × n_slot_shuffles

Default n_slot_shuffles = 3  →  12× augmentation per rollout step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRECTNESS OF REUSING old_log_probs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PPO requires π_old(a|s) to compute the importance-sampling ratio.  For
augmented samples we reuse the log_prob from the original (s, a) pair.

  • Slot permutations:  exact — permutation invariance means
    log π(perm(a)|perm(s)) = log π(a|s) for any permutation.

  • Geometric flips:  approximate early in training, exact once the
    network has learned equivariance.  The mismatch provides a gradient
    signal toward equivariance and the approximation error shrinks as
    training progresses.  Empirically this is stable and beneficial.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD INDICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Indices are computed at import time from the dataclass field order so they
automatically stay correct if new fields are added to the schema.

Angular velocity (spin) transforms as a pseudovector under reflections:
    flip_x (R=diag(-1,1,1), det=-1):  ω → det·R·ω  = (ω_x, -ω_y, -ω_z)
    flip_y (R=diag(1,-1,1), det=-1):  ω → det·R·ω  = (-ω_x, ω_y, -ω_z)

Heading angle θ under flip_x → π-θ:  sin unchanged, cos negated.
Heading angle θ under flip_y → -θ:   sin negated,   cos unchanged.
"""
from __future__ import annotations

import random
from dataclasses import fields

import torch

from footballcoach.ai.obs.schema import BallFeatures, PlayerFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field_index(dataclass_type, name: str) -> int:
    for i, f in enumerate(fields(dataclass_type)):
        if f.name == name:
            return i
    raise KeyError(f"{name!r} not found in {dataclass_type.__name__}")


# ---------------------------------------------------------------------------
# Precomputed flip indices (derived from schema — never hardcode raw numbers)
# ---------------------------------------------------------------------------

#: PlayerFeatures indices negated by flip_x
PLAYER_FLIP_X_IDX: list[int] = [
    _field_index(PlayerFeatures, "rel_dx"),
    _field_index(PlayerFeatures, "ball_rel_dx"),
    _field_index(PlayerFeatures, "ball_vel_rel_x"),
    _field_index(PlayerFeatures, "velocity_x"),
    _field_index(PlayerFeatures, "attacking_direction"),  # +x → -x
    _field_index(PlayerFeatures, "pos_x"),             # absolute x negated under flip_x
]
# ball_closing_speed is flip-invariant: both rel-velocity and direction components
# negate under a flip, so their dot product (= closing speed) is unchanged.

#: PlayerFeatures indices negated by flip_y
PLAYER_FLIP_Y_IDX: list[int] = [
    _field_index(PlayerFeatures, "rel_dy"),
    _field_index(PlayerFeatures, "ball_rel_dy"),
    _field_index(PlayerFeatures, "ball_vel_rel_y"),
    _field_index(PlayerFeatures, "velocity_y"),
    _field_index(PlayerFeatures, "pos_y"),             # absolute y negated under flip_y
]

#: BallFeatures indices negated by flip_x (includes pseudovector spin)
BALL_FLIP_X_IDX: list[int] = [
    _field_index(BallFeatures, "pos_x"),   # absolute x negated under flip_x
    _field_index(BallFeatures, "velocity_x"),
    _field_index(BallFeatures, "spin_y"),  # pseudovector flip_x: -ω_y
    _field_index(BallFeatures, "spin_z"),  # pseudovector flip_x: -ω_z
]

#: BallFeatures indices negated by flip_y (includes pseudovector spin)
BALL_FLIP_Y_IDX: list[int] = [
    _field_index(BallFeatures, "pos_y"),   # absolute y negated under flip_y
    _field_index(BallFeatures, "velocity_y"),
    _field_index(BallFeatures, "spin_x"),  # pseudovector flip_y: -ω_x
    _field_index(BallFeatures, "spin_z"),  # pseudovector flip_y: -ω_z
]

# BC label column layout (see bc.py::BCLabel.to_array())
# NOTE: these are hardcoded magic numbers, NOT derived from bc.py's `_I_*`
# index constants. They happen to still be correct today because every
# bc.py layout change so far has only appended new fields after `valid`
# (currently the last-but-one field). If a FUTURE bc.py change ever
# inserts/reorders fields at or before index 11, these constants will
# silently go stale — cross-check them against `_I_DIR_X`/`_I_DIR_Y`/
# `_I_REGION_X`/`_I_REGION_Y` in bc.py whenever BC_LABEL_DIM's layout changes.
_BC_DIR_X_COL: int = 7   # move_direction x component
_BC_DIR_Y_COL: int = 8   # move_direction y component
_BC_REGION_X_COL: int = 10  # move_region_center x (absolute pitch metres — negate on flip_x)
_BC_REGION_Y_COL: int = 11  # move_region_center y (absolute pitch metres — negate on flip_y)
_BC_KICK_DIR_X_COL: int = 18  # kick_direction x component
_BC_KICK_DIR_Y_COL: int = 19  # kick_direction y component
_BC_KICK_DIR_Z_COL: int = 24  # kick_direction z component (non-contiguous with x/y; unaffected by x/y pitch flips)
# kick_spin (indices 21-23) is a pseudovector, same transform rules as
# BALL_FLIP_X_IDX/BALL_FLIP_Y_IDX above: flip_x negates spin_y/spin_z,
# flip_y negates spin_x/spin_z. kick_power (index 20) is a scalar and is
# invariant under both flips.
_BC_KICK_SPIN_X_COL: int = 21
_BC_KICK_SPIN_Y_COL: int = 22
_BC_KICK_SPIN_Z_COL: int = 23

# Action dict keys that are 2D direction/position vectors and need flipping.
# All other action keys are Bernoulli (0/1) or scalar — they are invariant.
_DIR_ACTION_KEYS: frozenset[str] = frozenset({
    "move_dir_raw",
    "kick_dir_raw",
    "move_region_center_raw",
})

# Geometric flip variants: (flip_x, flip_y)
_FLIP_VARIANTS: list[tuple[bool, bool]] = [
    (False, False),  # identity
    (True,  False),  # flip_x
    (False, True),   # flip_y
    (True,  True),   # flip_xy  (180° rotation)
]


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def augment_batch(
    batch: dict,
    n_slot_shuffles: int,
    rng: random.Random,
) -> dict:
    """Expand a PPO batch with geometric flips and opponent-slot permutations.

    Produces 4 × n_slot_shuffles augmented copies of every sample, including
    the original (identity flip, identity slot order) as one of the copies.

    All non-geometric fields (rewards, advantages, returns, dones, log_probs,
    values) are tiled unchanged.  Geometric obs/action fields are sign-flipped.
    Slot fields (other_feat, exists_mask) are permuted together consistently.

    BC labels (if present) have their direction components flipped to match.

    Args:
        batch:          CPU tensor dict from ``RolloutBuffer.as_tensors()``.
        n_slot_shuffles: Number of slot permutations per geometric variant.
                        Use ≥1 (1 = identity slot order only, giving 4× total).
        rng:            Random instance for reproducible slot permutations.

    Returns:
        New dict with batch dimension multiplied by (4 × n_slot_shuffles).
    """
    n_slot_shuffles = max(1, n_slot_shuffles)
    n_slots: int = batch["obs/other_feat"].shape[1]
    has_bc: bool = "bc_labels" in batch
    has_ai_type: bool = "obs/self_ai_type" in batch and "obs/other_ai_type" in batch
    has_head_log_probs: bool = "head_log_probs" in batch
    has_reward_comps: bool = "reward_comps_raw" in batch
    has_step_outcomes: bool = "step_outcomes" in batch

    # Pre-build index tensors once (avoids repeated Python list → tensor conv)
    _px = torch.tensor(PLAYER_FLIP_X_IDX, dtype=torch.long)
    _py = torch.tensor(PLAYER_FLIP_Y_IDX, dtype=torch.long)
    _bx = torch.tensor(BALL_FLIP_X_IDX,   dtype=torch.long)
    _by = torch.tensor(BALL_FLIP_Y_IDX,   dtype=torch.long)

    parts: list[dict] = []

    for flip_x, flip_y in _FLIP_VARIANTS:
        # ---- Flip observations ----
        sf = batch["obs/self_feat"].clone()
        of = batch["obs/other_feat"].clone()
        bf = batch["obs/ball_feat"].clone()
        # global_feat has no geometric content (pitch dims, score, time) — shared
        gf = batch["obs/global_feat"]
        # self_ai_type/other_ai_type are NOT geometric (not positional/directional)
        # — pass through unchanged under flip_x/flip_y, see ai/knowledge.md
        # "Opponent-AI-type (value-only)". other_ai_type still needs the SAME
        # slot permutation as other_feat (applied below, per shuffle variant).
        sat = batch["obs/self_ai_type"] if has_ai_type else None
        oat_base = batch["obs/other_ai_type"] if has_ai_type else None

        if flip_x:
            sf[:, _px]    *= -1.0
            of[:, :, _px] *= -1.0
            bf[:, _bx]    *= -1.0
        if flip_y:
            sf[:, _py]    *= -1.0
            of[:, :, _py] *= -1.0
            bf[:, _by]    *= -1.0

        # ---- Flip actions ----
        flipped_actions: dict = {}
        for k, v in batch.items():
            if not k.startswith("action/"):
                continue
            key = k[len("action/"):]
            if key in _DIR_ACTION_KEYS and v.ndim == 2 and v.shape[1] >= 2:
                v2 = v.clone()
                if flip_x:
                    v2[:, 0] *= -1.0
                if flip_y:
                    v2[:, 1] *= -1.0
                flipped_actions[k] = v2
            else:
                flipped_actions[k] = v  # unchanged (Bernoulli, scalar, etc.)

        # ---- Flip BC labels ----
        bc_labels_flipped: torch.Tensor | None = None
        if has_bc:
            bc_labels_flipped = batch["bc_labels"].clone()
            if flip_x:
                bc_labels_flipped[:, _BC_DIR_X_COL]      *= -1.0
                bc_labels_flipped[:, _BC_REGION_X_COL]   *= -1.0
                bc_labels_flipped[:, _BC_KICK_DIR_X_COL] *= -1.0
                bc_labels_flipped[:, _BC_KICK_SPIN_Y_COL] *= -1.0
                bc_labels_flipped[:, _BC_KICK_SPIN_Z_COL] *= -1.0
            if flip_y:
                bc_labels_flipped[:, _BC_DIR_Y_COL]      *= -1.0
                bc_labels_flipped[:, _BC_REGION_Y_COL]   *= -1.0
                bc_labels_flipped[:, _BC_KICK_DIR_Y_COL] *= -1.0
                bc_labels_flipped[:, _BC_KICK_SPIN_X_COL] *= -1.0
                bc_labels_flipped[:, _BC_KICK_SPIN_Z_COL] *= -1.0

        # ---- Slot permutation variants ----
        em_base = batch["obs/exists_mask"]

        for shuffle_i in range(n_slot_shuffles):
            if shuffle_i == 0:
                perm = torch.arange(n_slots, dtype=torch.long)
            else:
                perm = torch.tensor(
                    rng.sample(range(n_slots), k=n_slots),
                    dtype=torch.long,
                )

            # Build inverse permutation: old slot i → new position inv_perm[i]
            inv_perm = torch.argsort(perm)

            # Remap categorical target slot indices through the inverse permutation.
            # pass_target/tackle_target/mark_target store the slot index of the
            # target player; after permuting other_feat, that player now lives at
            # inv_perm[old_slot].
            _TARGET_KEYS = {"action/pass_target", "action/tackle_target", "action/mark_target"}
            remapped_actions: dict = {}
            for k, v in flipped_actions.items():
                if k in _TARGET_KEYS:
                    # v is (N, 1) long-ish float; map each index through inv_perm
                    remapped_actions[k] = inv_perm[v.long()].float()
                else:
                    remapped_actions[k] = v

            part: dict = {
                "obs/self_feat":   sf,
                "obs/other_feat":  of[:, perm, :],
                "obs/exists_mask": em_base[:, perm].clone(),
                "obs/ball_feat":   bf,
                "obs/global_feat": gf,
                # Scalar trajectory fields — copied unchanged
                "log_probs":      batch["log_probs"],
                "values":         batch["values"],
                "rewards":        batch["rewards"],
                "advantages":     batch["advantages"],
                "returns":        batch["returns"],
                "dones":          batch["dones"],
                "sample_weights": batch["sample_weights"],
            }
            if has_head_log_probs:
                # Per-head log_probs are not geometric (they're scalars per
                # head, not positional/directional) — pass through unchanged,
                # same as log_probs/values/rewards above. Without this,
                # augment_batch() silently dropped the key and any downstream
                # per-head KL diagnostic keyed on batch["head_log_probs"]
                # would KeyError once augmentation is enabled.
                part["head_log_probs"] = batch["head_log_probs"]
            if has_ai_type:
                # self_ai_type: pass-through (no geometric or slot content).
                # other_ai_type: SAME slot permutation as other_feat, so a
                # given real player's ai-type one-hot never desyncs from its
                # feature vector (see ai/knowledge.md).
                part["obs/self_ai_type"] = sat
                part["obs/other_ai_type"] = oat_base[:, perm, :]
            part.update(remapped_actions)
            if bc_labels_flipped is not None:
                part["bc_labels"] = bc_labels_flipped
            if has_reward_comps:
                part["reward_comps_raw"] = batch["reward_comps_raw"]
            if has_step_outcomes:
                part["step_outcomes"] = batch["step_outcomes"]

            parts.append(part)

    # Concatenate all variants along the batch dimension
    # List fields (reward_comps_raw, step_outcomes) are extended, not torch.cat'd.
    _list_keys = {"reward_comps_raw", "step_outcomes"}
    result: dict = {}
    for key in parts[0]:
        if key in _list_keys:
            combined: list = []
            for p in parts:
                combined.extend(p[key])
            result[key] = combined
        else:
            result[key] = torch.cat([p[key] for p in parts], dim=0)
    return result


def augment_obs_bc(
    obs_dict: dict,
    bc_labels: torch.Tensor,
    n_slot_shuffles: int,
    rng: random.Random,
) -> tuple[dict, torch.Tensor]:
    """Augment a single BC pretraining minibatch (obs + labels).

    ALWAYS APPLY THIS during BC pretraining — geometric augmentation during
    pretraining teaches the exact same pitch symmetries as during PPO and
    is strictly beneficial with no downsides.

    Args:
        obs_dict:        {self_feat, other_feat, exists_mask, ball_feat, global_feat}
                         as produced by DemonstrationDataset.iterate_minibatches().
        bc_labels:       (N, label_dim) float tensor.
        n_slot_shuffles: Number of slot-order permutations per flip variant.
        rng:             Random for slot permutations.

    Returns:
        (augmented_obs_dict, augmented_bc_labels) with batch dimension × (4 * n_slot_shuffles).
    """
    n_slot_shuffles = max(1, n_slot_shuffles)
    n_slots: int = obs_dict["other_feat"].shape[1]
    has_ai_type: bool = "self_ai_type" in obs_dict and "other_ai_type" in obs_dict

    _px = torch.tensor(PLAYER_FLIP_X_IDX, dtype=torch.long)
    _py = torch.tensor(PLAYER_FLIP_Y_IDX, dtype=torch.long)
    _bx = torch.tensor(BALL_FLIP_X_IDX,   dtype=torch.long)
    _by = torch.tensor(BALL_FLIP_Y_IDX,   dtype=torch.long)

    sf_parts, of_parts, em_parts, bf_parts, gf_parts, bc_parts = [], [], [], [], [], []
    sat_parts, oat_parts = [], []

    for flip_x, flip_y in _FLIP_VARIANTS:
        sf = obs_dict["self_feat"].clone()
        of = obs_dict["other_feat"].clone()
        bf = obs_dict["ball_feat"].clone()
        gf = obs_dict["global_feat"]  # invariant — no clone needed
        bc = bc_labels.clone()

        if flip_x:
            sf[:, _px]    *= -1.0
            of[:, :, _px] *= -1.0
            bf[:, _bx]    *= -1.0
            bc[:, _BC_DIR_X_COL]      *= -1.0
            bc[:, _BC_REGION_X_COL]   *= -1.0
            bc[:, _BC_KICK_DIR_X_COL] *= -1.0
            bc[:, _BC_KICK_SPIN_Y_COL] *= -1.0
            bc[:, _BC_KICK_SPIN_Z_COL] *= -1.0
        if flip_y:
            sf[:, _py]    *= -1.0
            of[:, :, _py] *= -1.0
            bf[:, _by]    *= -1.0
            bc[:, _BC_DIR_Y_COL]      *= -1.0
            bc[:, _BC_REGION_Y_COL]   *= -1.0
            bc[:, _BC_KICK_DIR_Y_COL] *= -1.0
            bc[:, _BC_KICK_SPIN_X_COL] *= -1.0
            bc[:, _BC_KICK_SPIN_Z_COL] *= -1.0

        em_base = obs_dict["exists_mask"]
        sat = obs_dict["self_ai_type"] if has_ai_type else None
        oat_base = obs_dict["other_ai_type"] if has_ai_type else None
        for shuffle_i in range(n_slot_shuffles):
            if shuffle_i == 0:
                perm = torch.arange(n_slots, dtype=torch.long)
            else:
                perm = torch.tensor(rng.sample(range(n_slots), k=n_slots), dtype=torch.long)
            sf_parts.append(sf)
            of_parts.append(of[:, perm, :])
            em_parts.append(em_base[:, perm].clone())
            bf_parts.append(bf)
            gf_parts.append(gf)
            bc_parts.append(bc)
            if has_ai_type:
                # self_ai_type: pass-through (not geometric). other_ai_type:
                # SAME slot permutation as other_feat (see ai/knowledge.md).
                sat_parts.append(sat)
                oat_parts.append(oat_base[:, perm, :])

    aug_obs = {
        "self_feat":   torch.cat(sf_parts, dim=0),
        "other_feat":  torch.cat(of_parts, dim=0),
        "exists_mask": torch.cat(em_parts, dim=0),
        "ball_feat":   torch.cat(bf_parts, dim=0),
        "global_feat": torch.cat(gf_parts, dim=0),
    }
    if has_ai_type:
        aug_obs["self_ai_type"] = torch.cat(sat_parts, dim=0)
        aug_obs["other_ai_type"] = torch.cat(oat_parts, dim=0)
    aug_bc = torch.cat(bc_parts, dim=0)
    return aug_obs, aug_bc
