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
from dataclasses import dataclass, fields

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

    Velocity normalization: FIXED reference pitch half-diagonal
    (``sqrt(52.5²+34.0²)``, standard 105m×68m pitch -- see
    ``ai/obs/encoder.py``'s ``_REF_HALF_DIAG_M``), NOT the live match's own
    pitch dimensions, so values represent absolute speed in a fixed
    pitch-scale unit per second regardless of episode pitch size, matching
    ball velocity normalization and the ``ai/physics_pretrain`` package's
    ``normalize_kinematics_by_base_pitch`` convention.  Position
    normalization (``rel_dx``/``rel_dy``/``pos_x``/``pos_y`` alike): also
    divided by the same fixed half-diagonal, NOT by the per-axis
    half-dimensions (52.5/34.0) and NOT by the live match's pitch (pitch-scale
    curriculum is not wired up today, but a live-pitch divisor would have
    silently made the same physical distance encode differently across
    episodes for no reason the network can see, once it is). The live
    pitch's actual size is still available to the network via
    ``GlobalFeatures.pitch_length_norm``/``pitch_width_norm``. Values stay
    ≈[-1, 1] on a standard pitch either way, so this only matters for anyone
    hand-computing a real metre distance back out of a raw feature value.
    """
    # --- Position (relative to observing player) ---
    rel_dx: float = 0.0          # (other.x - self.x) / (half_diag)
    rel_dy: float = 0.0          # (other.y - self.y) / (half_diag)
    distance_m: float = 0.0      # Euclidean 2D distance / pitch_half_diagonal, redundant but aids learning

    # --- Ball position and relative velocity (relative to this player) ---
    ball_rel_dx: float = 0.0        # (ball.x - player.x) / (half_diag)
    ball_rel_dy: float = 0.0        # (ball.y - player.y) / (half_diag)
    ball_distance_m: float = 0.0    # 2D distance to ball / pitch_half_diagonal
    ball_vel_rel_x: float = 0.0     # (ball.vx - player.vx) / pitch_half_diagonal
    ball_vel_rel_y: float = 0.0     # (ball.vy - player.vy) / pitch_half_diagonal
    ball_closing_speed: float = 0.0 # speed at which ball approaches player (flip-invariant)

    # --- Velocity ---
    velocity_x: float = 0.0      # world-frame vx / pitch_half_diagonal
    velocity_y: float = 0.0      # world-frame vy / pitch_half_diagonal
    speed_mps: float = 0.0       # |v_xy| / pitch_half_diagonal, redundant
    # heading_sin/heading_cos live at the end of this dataclass (see below) --
    # NOT dropped. velocity = (cos(heading)*speed, sin(heading)*speed) only
    # while speed > 0 (engine/movement.py constructs velocity FROM heading
    # every tick, so they're identical by construction whenever moving), but
    # at speed == 0 velocity collapses to (0,0,0) while heading_rad keeps its
    # last real value -- exactly the case a prior version of this comment
    # dismissed as "irrelevant." A standstill player's facing direction is
    # real signal, not noise.

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

    # --- Immobility flag ---
    is_immobile: float = 0.0     # 1.0 if this player has no AI and will not move

    # --- Absolute position (scaled by pitch half-diagonal) ---
    # Uses same axis convention as the engine: origin at pitch centre,
    # x in [-52.5, 52.5], y in [-34, 34].  Divided by the pitch
    # half-diagonal (sqrt(52.5^2 + 34^2), NOT separate per-axis 52.5/34.0
    # divisors -- see ai/obs/encoder.py's actual pos_x=.../half_diag and
    # this class's own docstring) so values are ≈[-1, 1] on a standard
    # pitch and scale gracefully on smaller pitches.  Negated under
    # flip_x / flip_y in augment.py.
    pos_x: float = 0.0           # player.position.x / half_diag
    pos_y: float = 0.0           # player.position.y / half_diag

    # --- Heading (facing direction) ---
    # Populated unconditionally from player.heading_rad, including at
    # standstill and for immobile players -- unlike velocity, a static facing
    # direction is real signal (see the velocity_x/velocity_y comment above).
    # Mirror rule (established by ai/ppo/bc.py's own heading_sin/heading_cos
    # BC label fields and physics_value_net.py's canonicalize_bc_labels()
    # hand-patch, now made systematic in obs/augment.py/obs/canonical.py):
    # flip_y negates heading_sin only; the canonical x-mirror negates
    # heading_cos only.
    heading_sin: float = 0.0     # sin(player.heading_rad)
    heading_cos: float = 1.0     # cos(player.heading_rad)

    # --- Previous-decision movement intent ---
    # The direction/speed-mode set by the player's LAST decision, still in
    # effect until the next decision tick overwrites it -- a cheap
    # acceleration/intent signal (ai/physics_pretrain already relies on the
    # equivalent quantity via BC labels; this exposes it as a live input
    # too). desired_dir_x/y come from player.desired_direction (world-frame
    # unit vector, safe to read directly -- never auto-cleared). The one-hot
    # comes from player.last_desired_speed_mode, NOT player.desired_speed_mode
    # (which match._apply_movement() unconditionally clears to None every
    # tick after consuming it -- see that field's own docstring on Player).
    # Zeroed (dir=0, one-hot=STANDSTILL) for immobile players and for a
    # player with no decision yet (last_desired_speed_mode is None) --
    # "no movement intent" is real signal here, same rationale as the
    # is_immobile velocity-zeroing above.
    desired_dir_x: float = 0.0
    desired_dir_y: float = 0.0
    desired_speed_standstill: float = 1.0
    desired_speed_jog: float = 0.0
    desired_speed_sprint: float = 0.0

    def to_array(self) -> np.ndarray:
        # vars(self).values() preserves field-declaration order (Python 3.7+ dict
        # ordering) same as astuple(), but skips its recursive nested-tuple-copy
        # machinery -- all fields here are plain floats, so that overhead is pure
        # waste (astuple() showed up as a major hot spot in match-step profiling).
        _vals = vars(self)
        return np.fromiter(_vals.values(), dtype=np.float32, count=len(_vals))


@dataclass
class BallFeatures:
    """Ball feature vector (12 floats).

    Absolute pitch position normalized by the FIXED reference pitch
    half-diagonal (``sqrt(52.5^2 + 34.0^2)``, standard 105m×68m pitch --
    the SAME divisor ``PlayerFeatures.pos_x``/``pos_y`` use, and NOT the
    live match's own pitch dimensions -- see ``ai/obs/encoder.py``'s
    ``_REF_HALF_DIAG_M`` and ``PlayerFeatures``'s own docstring), so values
    are ≈[-1, 1].  Height uses a fixed divisor
    (``height_norm_m`` from ai_config.json, default 3.0m).  Ball-to-player
    relative position is encoded per-player in ``PlayerFeatures.ball_rel_*``.
    """
    pos_x: float = 0.0           # ball.position.x / half_diag
    pos_y: float = 0.0           # ball.position.y / half_diag
    height_m: float = 0.0        # ball.position.z / height_norm_m

    velocity_x: float = 0.0      # world-frame, normalized by pitch_half_diagonal / s (rough physical scale)
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    spin_x: float = 0.0          # normalized by ball_spin_nn_norm_rad_s
    spin_y: float = 0.0
    spin_z: float = 0.0

    is_possessed: float = 0.0    # 1 if ball.possessed_by is not None
    is_loose: float = 1.0        # 1 - is_possessed (redundant but explicit)

    # Which team most recently GAINED possession (persists through loose-ball
    # periods -- see entities/ball.py's Ball.last_touched_by_player_id, whose
    # single write-path is Match._set_possession()). +1.0 if that player's
    # team attacks +x (Team.LEFT), -1.0 if Team.RIGHT, 0.0 if nobody has
    # touched the ball yet this episode. Matters for attributing an
    # out-of-bounds/goal outcome to a team (mirrors the engine's own existing
    # `own_team_touched_last` pattern in match.py's
    # _run_get_possession_behaviour -- this exposes that same fact to the
    # network instead of only using it inside reward shaping / rules logic).
    # Same sign convention as PlayerFeatures.attacking_direction; negated
    # under the canonical x-mirror in obs/augment.py's BALL_FLIP_X_IDX (NOT
    # flip_y -- a team-direction sign, not a y-coordinate).
    last_touch_team_direction: float = 0.0

    def to_array(self) -> np.ndarray:
        # See PlayerFeatures.to_array() -- same astuple() -> vars().values() swap.
        _vals = vars(self)
        return np.fromiter(_vals.values(), dtype=np.float32, count=len(_vals))


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
        # See PlayerFeatures.to_array() -- same astuple() -> vars().values() swap.
        _vals = vars(self)
        return np.fromiter(_vals.values(), dtype=np.float32, count=len(_vals))


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
