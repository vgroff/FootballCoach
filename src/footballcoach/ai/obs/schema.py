"""Observation feature vector schemas (dataclasses).

These are the canonical definitions of what goes into each part of the
observation tensor fed to both the decision network and execution network.
See ai_design_doc.md sections 7.2-7.5 for full rationale.

Feature dimension constants are derived from these dataclasses so there is
one source of truth.  Use ``PLAYER_FEATURE_DIM``, ``BALL_FEATURE_DIM``, and
``GLOBAL_FEATURE_DIM`` when constructing network modules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, astuple

import numpy as np


@dataclass
class PlayerFeatures:
    """Per-player feature vector (see ``PLAYER_FEATURE_DIM`` for the
    authoritative count).

    Used for both the "self" slot and each of the up-to-21 "other player"
    slots.  For the self slot: ``rel_dx=0``, ``rel_dy=0``,
    ``distance_m=0``, ``is_self=1``.  For padded/absent slots: every field
    is 0 (including ``exists=0``, which distinguishes a padded slot from
    a real player standing at the same position as the observer).

    Velocity normalization: per-player ``effective_top_speed`` (not a
    global constant) so the feature is attribute-invariant across the full
    skill range.  Position normalization: relative offset divided by pitch
    half-dimensions so values stay ≈[-1, 1] across varied pitch sizes.
    """
    # --- Position (relative to observing player) ---
    rel_dx: float = 0.0          # (other.x - self.x) / (pitch.length_m / 2)
    rel_dy: float = 0.0          # (other.y - self.y) / (pitch.width_m / 2)
    distance_m: float = 0.0      # Euclidean 2D distance / pitch_half_diagonal, redundant but aids learning

    # --- Velocity ---
    velocity_x: float = 0.0      # world-frame vx / own_effective_top_speed
    velocity_y: float = 0.0      # world-frame vy / own_effective_top_speed
    speed_mps: float = 0.0       # |v_xy| / own_effective_top_speed, redundant

    # --- Heading (sin/cos avoids angle-wraparound discontinuity) ---
    heading_sin: float = 0.0
    heading_cos: float = 1.0

    # --- Stamina ---
    stamina: float = 1.0         # current stamina fraction [0, 1]

    # --- Attributes (all [0, 1] from PlayerAttributes) ---
    top_speed: float = 0.5
    acceleration: float = 0.5
    kick_power: float = 0.5
    kick_precision: float = 0.5
    dribbling: float = 0.5
    ball_control: float = 0.5
    tackling: float = 0.5
    stamina_attr: float = 0.5    # the stamina *attribute* (not current stamina)

    # --- Flags (0/1 floats for direct tensor packing) ---
    is_own_team: float = 0.0
    is_self: float = 0.0
    has_possession: float = 0.0
    is_inactive_tackled: float = 0.0
    is_controlling_ball: float = 0.0
    is_goalkeeper: float = 0.0
    attacking_direction: float = 0.0  # +1.0 if attacking +x, -1.0 if attacking -x

    # --- Existence mask ---
    exists: float = 0.0          # 1.0 for a real player, 0.0 for a padded slot

    # --- Absolute position (scaled by standard pitch half-dims) ---
    # Uses same axis convention as the engine: origin at pitch centre,
    # x in [-52.5, 52.5], y in [-34, 34].  Divided by standard half-dims
    # so values are ≈[-1, 1] on a standard pitch and scale gracefully on
    # smaller pitches.  Negated under flip_x / flip_y in augment.py.
    pos_x: float = 0.0           # player.position.x / 52.5
    pos_y: float = 0.0           # player.position.y / 34.0

    def to_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=np.float32)


@dataclass
class BallFeatures:
    """Ball feature vector (12 floats).

    Position relative to the observing player, normalized by pitch dimensions
    (same convention as PlayerFeatures).  Height uses a fixed divisor
    (``height_norm_m`` from ai_config.json, default 3.0m) since height
    does not scale with pitch size.
    """
    rel_dx: float = 0.0
    rel_dy: float = 0.0
    distance_m: float = 0.0      # 2D distance / pitch_half_diagonal
    height_m: float = 0.0        # ball.position.z / height_norm_m

    velocity_x: float = 0.0      # world-frame, normalized by pitch_half_diagonal / s (rough physical scale)
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    spin_x: float = 0.0          # normalized by ball_spin_norm_max_rad_s
    spin_y: float = 0.0
    spin_z: float = 0.0

    is_possessed: float = 0.0    # 1 if ball.possessed_by is not None
    is_loose: float = 1.0        # 1 - is_possessed (redundant but explicit)

    def to_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=np.float32)


MAX_TASK_IDS: int = 20  # curriculum phase/task one-hot width (see ai_config.json observation.max_task_ids)


@dataclass
class GlobalFeatures:
    """Match-context feature vector (see ``GLOBAL_FEATURE_DIM`` for the
    authoritative count).

    ``attack_defence_smoothed`` is technically per-player, but placed here
    (section 7.5 of ai_design_doc.md: "placed in self features in the actual
    tensor packing - listed here for narrative completeness").  In practice
    it is fed to the network's global_mlp branch alongside the rest of this
    vector.

    Time remaining: log1p-normalized (``log1p(t) / log1p(max_t)``) so
    the "urgent" 1-20s endgame scenarios are distinguishable from normal play
    (see ai_design_doc.md section 7.5's note on this).

    Task-id (``task_id_0`` .. ``task_id_19``): a MAX_TASK_IDS-wide one-hot
    identifying which curriculum phase/task is currently active. Fixed-width
    so the network architecture doesn't change as new phases are added —
    unused task slots stay zero. Task 0 = phase 1, task 1 = phase 2, etc.
    (index = phase_id - 1). NOT YET WIRED for mixed multi-phase training —
    see ai/knowledge.md "Task-id: scaffolded, not yet load-bearing" note.
    Individual scalar fields (rather than a tuple field) are used
    deliberately so ``astuple()``/``to_array()`` keep producing a flat
    float32 array without special-casing.
    """
    score_diff: float = 0.0         # own_goals - opp_goals (team-relative, not raw scores)
    time_remaining_norm: float = 1.0  # log1p(t_s) / log1p(7200), ~[0,1]

    # Pitch / goal / box dimensions, normalised by standard values so the
    # network receives ≈1.0 on a standard pitch and a fraction on smaller ones.
    # Standard values: length=105m, width=68m, goal_w=7.32m, goal_h=2.44m,
    # box_length=16.5m, box_width=40.32m  (from physics.json).
    pitch_length_norm: float = 1.0   # pitch.length_m / 105.0
    pitch_width_norm: float = 1.0    # pitch.width_m / 68.0
    goal_width_norm: float = 1.0     # pitch.goal_width_m / 7.32
    goal_height_norm: float = 1.0    # pitch.goal_height_m / 2.44
    box_length_norm: float = 1.0     # pitch.box_length_m / 16.5
    box_width_norm: float = 1.0      # pitch.box_width_m / 40.32

    ball_restitution_coefficient: float = 0.6
    rng_reduction: float = 0.3

    attack_defence_smoothed: float = 0.5  # EMA-smoothed attack/defence weighting [0,1]

    # --- Task identifier (one-hot, MAX_TASK_IDS wide) ---
    task_id_0: float = 0.0
    task_id_1: float = 0.0
    task_id_2: float = 0.0
    task_id_3: float = 0.0
    task_id_4: float = 0.0
    task_id_5: float = 0.0
    task_id_6: float = 0.0
    task_id_7: float = 0.0
    task_id_8: float = 0.0
    task_id_9: float = 0.0
    task_id_10: float = 0.0
    task_id_11: float = 0.0
    task_id_12: float = 0.0
    task_id_13: float = 0.0
    task_id_14: float = 0.0
    task_id_15: float = 0.0
    task_id_16: float = 0.0
    task_id_17: float = 0.0
    task_id_18: float = 0.0
    task_id_19: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=np.float32)


# ---------------------------------------------------------------------------
# Dimension constants (derived from the dataclasses - single source of truth)
# ---------------------------------------------------------------------------

PLAYER_FEATURE_DIM: int = len(fields(PlayerFeatures))
BALL_FEATURE_DIM: int = len(fields(BallFeatures))
GLOBAL_FEATURE_DIM: int = len(fields(GlobalFeatures))
MAX_OTHER_PLAYERS: int = 21  # full 11v11 minus self


# ---------------------------------------------------------------------------
# Observation batch
# ---------------------------------------------------------------------------

#: Width of each AI-type one-hot (rules / immobile / neural). See AI_TYPE_*
#: constants in ai/ppo/bc.py (kept in sync manually - this is a small, fixed
#: constant, not derived from bc.py to avoid a schema.py -> ppo.bc import).
AI_TYPE_ONE_HOT_DIM: int = 3


@dataclass
class ObservationBatch:
    """One observation for one player (unbatched arrays; batch dim added by
    the trainer when collecting rollouts from multiple envs/players).

    Shapes:
        self_feat:     (PLAYER_FEATURE_DIM,)
        other_feat:    (MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)
        exists_mask:   (MAX_OTHER_PLAYERS,)  - 1.0 for real, 0.0 for padded
        ball_feat:     (BALL_FEATURE_DIM,)
        global_feat:   (GLOBAL_FEATURE_DIM,)
        self_ai_type:  (AI_TYPE_ONE_HOT_DIM,)  - one-hot: [is_rules, is_immobile, is_neural]
        other_ai_type: (MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM) - permuted in lockstep
                        with other_feat/exists_mask (same slot shuffle).

    ``self_ai_type``/``other_ai_type`` are VALUE-ONLY side channels - see
    ai/knowledge.md "Opponent-AI-type (value-only)". They default to all-zero
    arrays (meaning: unknown/unset) so existing callers that don't populate
    them keep working; ``encode_observation()`` always populates them with a
    real one-hot.
    """
    self_feat: np.ndarray
    other_feat: np.ndarray
    exists_mask: np.ndarray
    ball_feat: np.ndarray
    global_feat: np.ndarray
    self_ai_type: np.ndarray = None
    other_ai_type: np.ndarray = None

    def __post_init__(self) -> None:
        if self.self_ai_type is None:
            self.self_ai_type = np.zeros(AI_TYPE_ONE_HOT_DIM, dtype=np.float32)
        if self.other_ai_type is None:
            self.other_ai_type = np.zeros(
                (self.other_feat.shape[0], AI_TYPE_ONE_HOT_DIM), dtype=np.float32
            )

    def to_torch_dict(self) -> dict:
        """Convert to a dict of 1-D (unbatched) torch tensors.

        The PPO loop adds the batch dimension when collating rollout steps.
        """
        import torch
        return {
            "self_feat": torch.from_numpy(self.self_feat),
            "other_feat": torch.from_numpy(self.other_feat),
            "exists_mask": torch.from_numpy(self.exists_mask),
            "ball_feat": torch.from_numpy(self.ball_feat),
            "global_feat": torch.from_numpy(self.global_feat),
            "self_ai_type": torch.from_numpy(self.self_ai_type),
            "other_ai_type": torch.from_numpy(self.other_ai_type),
        }
