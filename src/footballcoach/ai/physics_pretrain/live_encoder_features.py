"""Reconstruct physics-pretrain encoder inputs from LIVE observation-schema
arrays (``PlayerFeatures``/``BallFeatures``/``GlobalFeatures``, as stored by
``DemonstrationDataset``), and load a frozen ``BallDynamicsEncoder``/
``PlayerDynamicsEncoder`` for inference-only use.

This module is standalone-friendly the same way the rest of
``ai/physics_pretrain/`` is (see this package's own ``__init__.py``): it never
imports anything from ``ai/models/``. It exists to support
``physics_value_net.py``'s diagnostic value network, which asks "do these two
independently-pretrained physics encoders already carry reward-relevant
signal on real match data?" -- see ``debug_value_network.py``'s
``--physics-encoder-value-net`` CLI flag.

The live observation encoding (``ai/obs/encoder.py``) and the physics-pretrain
episode generators (``ball_episode_gen.py``/``player_episode_gen.py``) were
built independently and use DIFFERENT normalization conventions for the same
underlying physical quantities (e.g. live position/velocity is normalized by
the LIVE match's own pitch half-diagonal; physics-pretrain normalizes by a
FIXED base-pitch half-diagonal when ``normalize_kinematics_by_base_pitch`` is
on). Every reconstruction function below is explicit about converting between
the two, rather than assuming they coincide numerically (they often do today,
since both trace back to the same ``physics.json`` defaults, but that is a
coincidence of current config values, not a guarantee).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from footballcoach.config import load_physics_config, require_section
from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsEncoder
from footballcoach.ai.physics_pretrain.player_dynamics_net import PlayerDynamicsEncoder
from footballcoach.ai.physics_pretrain.ball_episode_gen import (
    BALL_SPIN_NORM_DIVISOR_RAD_S,
    N_INPUT_FIELDS as BALL_N_INPUT_FIELDS,
    compute_engineered_features as ball_compute_engineered_features,
)
from footballcoach.ai.physics_pretrain.player_episode_gen import (
    N_INPUT_FIELDS as PLAYER_N_INPUT_FIELDS,
    compute_engineered_features as player_compute_engineered_features,
)
from footballcoach.ai.ppo.bc import _I_DIR_X, _I_DIR_Y, _I_EXEC_MOVE, _I_HEADING_COS, _I_HEADING_SIN, _I_SPRINT

# --- PlayerFeatures field indices (ai/obs/schema.py declaration order) -----
PF_VELOCITY_X = 9
PF_VELOCITY_Y = 10
PF_STAMINA = 12          # current stamina FRACTION (state)
PF_TOP_SPEED = 13        # attribute
PF_ACCELERATION = 14     # attribute
PF_KICK_POWER = 15       # attribute
PF_KICK_PRECISION = 16   # attribute
PF_DRIBBLING = 17        # attribute
PF_BALL_CONTROL = 18     # attribute
PF_TACKLING = 19         # attribute
PF_STAMINA_ATTR = 20     # attribute -- NOT the same as PF_STAMINA (index 12)
# All 8 PlayerAttributes fields are contiguous (schema.py declares them as
# one block, PF_TOP_SPEED..PF_STAMINA_ATTR) -- this slice is the single
# source of truth for "all attributes", used by PhysicsEncoderValueNet to
# pull the full set for both players as explicit MLP scalars, on top of
# whatever subset (PF_TOP_SPEED/PF_ACCELERATION/PF_BALL_CONTROL/
# PF_STAMINA_ATTR only) already reaches the frozen player encoder via
# player_live_to_physics_input below.
PF_ATTRIBUTES_SLICE = slice(PF_TOP_SPEED, PF_STAMINA_ATTR + 1)
PF_HAS_POSSESSION = 23
PF_POS_X = 30
PF_POS_Y = 31
# Added for the frozen physics-encoder integration (ai/knowledge.md "Frozen
# physics-dynamics encoders" / "Heading and previous-decision movement
# intent") -- appended after pos_y so every index above stays stable.
PF_HEADING_SIN = 32
PF_HEADING_COS = 33
PF_DESIRED_DIR_X = 34
PF_DESIRED_DIR_Y = 35
PF_DESIRED_SPEED_STANDSTILL = 36
PF_DESIRED_SPEED_JOG = 37
PF_DESIRED_SPEED_SPRINT = 38

# --- BallFeatures field indices ---------------------------------------------
BF_POS_X = 0
BF_POS_Y = 1
BF_HEIGHT_M = 2
BF_VELOCITY_X = 3
BF_VELOCITY_Y = 4
BF_VELOCITY_Z = 5
BF_SPIN_X = 6
BF_SPIN_Y = 7
BF_SPIN_Z = 8
BF_IS_POSSESSED = 9
BF_IS_LOOSE = 10
BF_LAST_TOUCH_TEAM_DIRECTION = 11

# --- GlobalFeatures field indices -------------------------------------------
GF_TIME_REMAINING_NORM = 1
GF_PITCH_LENGTH_NORM = 2
GF_PITCH_WIDTH_NORM = 3
GF_GOAL_WIDTH_NORM = 4
GF_GOAL_HEIGHT_NORM = 5
GF_RESTITUTION = 8

# ai/obs/encoder.py hardcodes these as the reference pitch/goal dims every
# GlobalFeatures ratio field is computed against (encoder.py:349-353) -- NOT
# necessarily the same object as physics-pretrain's own
# base_pitch_length_m/width_m/base_goal_width_m/height_m, even though both
# currently trace back to the same physics.json defaults.
_LIVE_PITCH_LENGTH_REF_M = 105.0
_LIVE_PITCH_WIDTH_REF_M = 68.0
_LIVE_GOAL_WIDTH_REF_M = 7.32
_LIVE_GOAL_HEIGHT_REF_M = 2.44


@dataclass(frozen=True)
class PhysicsPitchConstants:
    """The base/standard pitch dims physics-pretrain's episode generators
    normalize against (see ball_episode_gen.BallEpisodeGenParams.from_config's
    identical sourcing) -- NOT saved in a checkpoint's own config_snapshot
    (that's just ai_config.json's physics_pretrain.{ball,player} section
    verbatim, which has no pitch-dimension keys at all; pitch dims live in
    the separate physics.json config). Same physical pitch for both ball and
    player, so one instance is shared between the two reconstruction
    functions below."""
    base_pitch_length_m: float
    base_pitch_width_m: float
    base_goal_width_m: float
    base_goal_height_m: float


def load_physics_pitch_constants() -> PhysicsPitchConstants:
    """Reads physics.json's pitch section -- the same source
    BallEpisodeGenParams.from_config()/PlayerEpisodeGenParams.from_config()
    use for base_pitch_length_m/width_m/base_goal_width_m/height_m."""
    pitch_cfg = require_section(load_physics_config(), "pitch")
    return PhysicsPitchConstants(
        base_pitch_length_m=float(pitch_cfg["length_m"]),
        base_pitch_width_m=float(pitch_cfg["width_m"]),
        base_goal_width_m=float(pitch_cfg["goal_width_m"]),
        base_goal_height_m=float(pitch_cfg["goal_height_m"]),
    )


def _live_half_diag(global_feat: torch.Tensor) -> torch.Tensor:
    """(batch,1) live match's own pitch half-diagonal in metres, recovered
    from GlobalFeatures' pitch_length_norm/pitch_width_norm ratios."""
    length_m = global_feat[:, GF_PITCH_LENGTH_NORM:GF_PITCH_LENGTH_NORM + 1] * _LIVE_PITCH_LENGTH_REF_M
    width_m = global_feat[:, GF_PITCH_WIDTH_NORM:GF_PITCH_WIDTH_NORM + 1] * _LIVE_PITCH_WIDTH_REF_M
    return torch.hypot(length_m / 2.0, width_m / 2.0)


def _rescale_to_base_pitch(
    global_feat: torch.Tensor, base_pitch_length_m: float, base_pitch_width_m: float,
) -> torch.Tensor:
    """(batch,1) factor = live_half_diag / base_half_diag. Multiplying a
    live-normalized position/velocity value by this factor converts it to
    physics-pretrain's base-pitch-normalized convention (assumes
    ``normalize_kinematics_by_base_pitch=true``, verified true for both
    ``physics_pretrain.ball``/``physics_pretrain.player`` in the live config
    at the time this module was written -- re-check if that config ever
    changes, since the per-episode-pitch convention uses different divisors
    per axis and this simple scalar rescale would no longer be correct)."""
    base_half_diag = math.hypot(base_pitch_length_m / 2.0, base_pitch_width_m / 2.0)
    return _live_half_diag(global_feat) / base_half_diag


def time_remaining_and_elapsed_fraction(
    global_feat: torch.Tensor, time_norm_max_s: float, max_episode_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(time_remaining_frac, time_elapsed_frac), each (batch,1), both true
    LINEAR fractions of episode duration in [0, 1] with
    ``time_remaining_frac + time_elapsed_frac == 1.0`` exactly.

    Inverts GlobalFeatures.time_remaining_norm's log1p compression -- same
    exact pattern already used by debug_value_network.py's
    _run_linear_regression._features()/_lr_features_for_obs_list, reproduced
    here in torch rather than numpy. Deliberately NOT the same quantity as
    execution_network.py's ``elapsed_norm = 1 - time_remaining_norm``, which
    is an algebraic complement of the log-compressed value, not literally
    elapsed time."""
    norm = global_feat[:, GF_TIME_REMAINING_NORM:GF_TIME_REMAINING_NORM + 1]
    time_remaining_s = torch.expm1(norm * math.log1p(time_norm_max_s))
    time_remaining_frac = torch.clamp(time_remaining_s / max_episode_s, 0.0, 1.0)
    time_elapsed_frac = 1.0 - time_remaining_frac
    return time_remaining_frac, time_elapsed_frac


def ball_live_to_physics_input(
    ball_feat: torch.Tensor, global_feat: torch.Tensor, pitch_constants: PhysicsPitchConstants,
    live_ball_spin_nn_norm_rad_s: float = 55.0, live_height_norm_m: float = 3.0,
) -> torch.Tensor:
    """(N, 20) -- see ball_episode_gen.py's _encode_input for the target
    field layout this reproduces.

    ``live_ball_spin_nn_norm_rad_s``/``live_height_norm_m``: the LIVE game's
    ``observation.ball_spin_nn_norm_rad_s``/``observation.height_norm_m``
    config values (defaults match ai_config.json's own defaults, 55.0/3.0) --
    callers should read them from ``load_ai_config()["observation"]`` and pass
    them through explicitly rather than relying on these defaults drifting
    out of sync with the live config."""
    base_pitch_length_m = pitch_constants.base_pitch_length_m
    base_pitch_width_m = pitch_constants.base_pitch_width_m
    base_goal_width_m = pitch_constants.base_goal_width_m
    base_goal_height_m = pitch_constants.base_goal_height_m
    # Live spin is normalized by observation.ball_spin_nn_norm_rad_s while
    # physics-pretrain ALWAYS uses the fixed module constant
    # BALL_SPIN_NORM_DIVISOR_RAD_S=30.0 (deliberately different -- see that
    # constant's own docstring) -- these do NOT coincide (55.0 vs 30.0 at
    # current defaults), unlike position/velocity, so spin needs its own
    # rescale.
    spin_rescale = live_ball_spin_nn_norm_rad_s / BALL_SPIN_NORM_DIVISOR_RAD_S
    pos_vel_rescale = _rescale_to_base_pitch(global_feat, base_pitch_length_m, base_pitch_width_m)
    base_half_diag = math.hypot(base_pitch_length_m / 2.0, base_pitch_width_m / 2.0)

    n = ball_feat.shape[0]
    row = torch.zeros((n, BALL_N_INPUT_FIELDS), dtype=torch.float32, device=ball_feat.device)
    row[:, 0] = ball_feat[:, BF_POS_X] * pos_vel_rescale[:, 0]
    row[:, 1] = ball_feat[:, BF_POS_Y] * pos_vel_rescale[:, 0]
    # height_m is normalized by live_height_norm_m (obs/encoder.py:
    # "height_m=ball.position.z / height_norm_m"), a DIFFERENT divisor from
    # physics-pretrain's own (base_half_diag, under normalize_kinematics_by_
    # base_pitch=true) -- un-normalize to real metres first, then re-normalize
    # in physics-pretrain's own convention.
    row[:, 2] = (ball_feat[:, BF_HEIGHT_M] * live_height_norm_m) / base_half_diag
    row[:, 3] = ball_feat[:, BF_VELOCITY_X] * pos_vel_rescale[:, 0]
    row[:, 4] = ball_feat[:, BF_VELOCITY_Y] * pos_vel_rescale[:, 0]
    row[:, 5] = ball_feat[:, BF_VELOCITY_Z] * pos_vel_rescale[:, 0]
    row[:, 6] = ball_feat[:, BF_SPIN_X] * spin_rescale
    row[:, 7] = ball_feat[:, BF_SPIN_Y] * spin_rescale
    row[:, 8] = ball_feat[:, BF_SPIN_Z] * spin_rescale
    row[:, 9] = global_feat[:, GF_RESTITUTION]
    row[:, 10] = global_feat[:, GF_PITCH_LENGTH_NORM] * _LIVE_PITCH_LENGTH_REF_M / base_pitch_length_m
    row[:, 11] = global_feat[:, GF_PITCH_WIDTH_NORM] * _LIVE_PITCH_WIDTH_REF_M / base_pitch_width_m
    row[:, 12] = global_feat[:, GF_GOAL_WIDTH_NORM] * _LIVE_GOAL_WIDTH_REF_M / base_goal_width_m
    row[:, 13] = global_feat[:, GF_GOAL_HEIGHT_NORM] * _LIVE_GOAL_HEIGHT_REF_M / base_goal_height_m
    engineered = ball_compute_engineered_features(row[:, 0:9].detach().cpu().numpy())
    row[:, 14:20] = torch.as_tensor(engineered, dtype=torch.float32, device=ball_feat.device)
    return row


def player_live_to_physics_input(
    self_feat: torch.Tensor, global_feat: torch.Tensor, labels: torch.Tensor,
    pitch_constants: PhysicsPitchConstants,
) -> torch.Tensor:
    """(N, 24) -- see player_episode_gen.py's _encode_input for the target
    field layout this reproduces. ``labels`` must be the (N, BC_LABEL_DIM)
    array recorded alongside these rows (see ai/ppo/bc.py) -- supplies
    heading_sin/cos (recorded directly at demo-record time, see
    phase1_labels()) and desired_direction/desired_speed_mode (execution-
    level fields, correctly sourced from player.desired_direction/
    desired_speed_mode per the "Orders vs execution-network labels boundary"
    rule -- see ai/knowledge.md)."""
    base_pitch_length_m = pitch_constants.base_pitch_length_m
    base_pitch_width_m = pitch_constants.base_pitch_width_m
    base_goal_width_m = pitch_constants.base_goal_width_m
    base_goal_height_m = pitch_constants.base_goal_height_m
    pos_vel_rescale = _rescale_to_base_pitch(global_feat, base_pitch_length_m, base_pitch_width_m)

    n = self_feat.shape[0]
    row = torch.zeros((n, PLAYER_N_INPUT_FIELDS), dtype=torch.float32, device=self_feat.device)
    row[:, 0] = self_feat[:, PF_POS_X] * pos_vel_rescale[:, 0]
    row[:, 1] = self_feat[:, PF_POS_Y] * pos_vel_rescale[:, 0]
    row[:, 2] = self_feat[:, PF_VELOCITY_X] * pos_vel_rescale[:, 0]
    row[:, 3] = self_feat[:, PF_VELOCITY_Y] * pos_vel_rescale[:, 0]
    row[:, 4] = labels[:, _I_HEADING_SIN]
    row[:, 5] = labels[:, _I_HEADING_COS]
    row[:, 6] = self_feat[:, PF_STAMINA]
    row[:, 7] = self_feat[:, PF_TOP_SPEED]
    row[:, 8] = self_feat[:, PF_ACCELERATION]
    row[:, 9] = self_feat[:, PF_STAMINA_ATTR]
    row[:, 10] = self_feat[:, PF_BALL_CONTROL]
    row[:, 11] = self_feat[:, PF_HAS_POSSESSION]
    row[:, 12] = labels[:, _I_DIR_X]
    row[:, 13] = labels[:, _I_DIR_Y]
    exec_move = labels[:, _I_EXEC_MOVE]
    sprint = labels[:, _I_SPRINT]
    row[:, 14] = (exec_move < 0.5).float()                              # STANDSTILL
    row[:, 15] = ((exec_move >= 0.5) & (sprint < 0.5)).float()          # JOG
    row[:, 16] = ((exec_move >= 0.5) & (sprint >= 0.5)).float()         # SPRINT
    row[:, 17] = global_feat[:, GF_PITCH_LENGTH_NORM] * _LIVE_PITCH_LENGTH_REF_M / base_pitch_length_m
    row[:, 18] = global_feat[:, GF_PITCH_WIDTH_NORM] * _LIVE_PITCH_WIDTH_REF_M / base_pitch_width_m
    row[:, 19] = global_feat[:, GF_GOAL_WIDTH_NORM] * _LIVE_GOAL_WIDTH_REF_M / base_goal_width_m
    row[:, 20] = global_feat[:, GF_GOAL_HEIGHT_NORM] * _LIVE_GOAL_HEIGHT_REF_M / base_goal_height_m
    engineered = player_compute_engineered_features(
        row[:, 0:7].detach().cpu().numpy(), row[:, 12:14].detach().cpu().numpy(),
    )
    row[:, 21:24] = torch.as_tensor(engineered, dtype=torch.float32, device=self_feat.device)
    return row


def player_obs_to_physics_input(
    player_feat: torch.Tensor, global_feat: torch.Tensor, pitch_constants: PhysicsPitchConstants,
) -> torch.Tensor:
    """(..., 24) -- BC-label-free sibling of player_live_to_physics_input,
    for use inside a live forward pass (DecisionNetwork.forward()) where no
    BC labels exist. Sources heading/desired-direction/desired-speed-mode
    directly off live PlayerFeatures columns (heading_sin/cos,
    desired_dir_x/y, desired_speed_standstill/jog/sprint -- added to the
    live schema specifically to support this, see ai/knowledge.md's "Heading
    and previous-decision movement intent" section) instead of a
    BC-label tensor.

    Produces IDENTICAL field values to player_live_to_physics_input given
    the same underlying player state -- row[12]/row[13] here are
    player.desired_direction.x/.y directly (PF_DESIRED_DIR_X/_Y), matching
    that function's row[12]=labels[_I_DIR_X]/row[13]=labels[_I_DIR_Y] (both
    ultimately trace back to the same player.desired_direction.x/.y at
    record/observe time -- there is no sin/cos swap here despite
    player_physics_pretrain_plan.md §3.1's field names, which just label a
    2D unit vector's components, not literal trig functions of a separate
    angle).

    Works identically whether ``player_feat`` is self_feat
    ``(..., PLAYER_FEATURE_DIM)`` or other_feat
    ``(..., MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)`` -- every op broadcasts
    over leading dims; ``global_feat`` never carries the extra
    MAX_OTHER_PLAYERS dim, so it's unsqueezed once when needed."""
    base_pitch_length_m = pitch_constants.base_pitch_length_m
    base_pitch_width_m = pitch_constants.base_pitch_width_m
    base_goal_width_m = pitch_constants.base_goal_width_m
    base_goal_height_m = pitch_constants.base_goal_height_m
    pos_vel_rescale = _rescale_to_base_pitch(global_feat, base_pitch_length_m, base_pitch_width_m)

    has_slot_dim = player_feat.dim() == global_feat.dim() + 1
    if has_slot_dim:
        pos_vel_rescale = pos_vel_rescale.unsqueeze(1)
    gf = global_feat.unsqueeze(1) if has_slot_dim else global_feat

    shape = player_feat.shape[:-1]
    row = torch.zeros((*shape, PLAYER_N_INPUT_FIELDS), dtype=torch.float32, device=player_feat.device)
    row[..., 0] = player_feat[..., PF_POS_X] * pos_vel_rescale[..., 0]
    row[..., 1] = player_feat[..., PF_POS_Y] * pos_vel_rescale[..., 0]
    row[..., 2] = player_feat[..., PF_VELOCITY_X] * pos_vel_rescale[..., 0]
    row[..., 3] = player_feat[..., PF_VELOCITY_Y] * pos_vel_rescale[..., 0]
    row[..., 4] = player_feat[..., PF_HEADING_SIN]
    row[..., 5] = player_feat[..., PF_HEADING_COS]
    row[..., 6] = player_feat[..., PF_STAMINA]
    row[..., 7] = player_feat[..., PF_TOP_SPEED]
    row[..., 8] = player_feat[..., PF_ACCELERATION]
    row[..., 9] = player_feat[..., PF_STAMINA_ATTR]
    row[..., 10] = player_feat[..., PF_BALL_CONTROL]
    row[..., 11] = player_feat[..., PF_HAS_POSSESSION]
    row[..., 12] = player_feat[..., PF_DESIRED_DIR_X]
    row[..., 13] = player_feat[..., PF_DESIRED_DIR_Y]
    row[..., 14] = player_feat[..., PF_DESIRED_SPEED_STANDSTILL]
    row[..., 15] = player_feat[..., PF_DESIRED_SPEED_JOG]
    row[..., 16] = player_feat[..., PF_DESIRED_SPEED_SPRINT]
    row[..., 17] = gf[..., GF_PITCH_LENGTH_NORM] * _LIVE_PITCH_LENGTH_REF_M / base_pitch_length_m
    row[..., 18] = gf[..., GF_PITCH_WIDTH_NORM] * _LIVE_PITCH_WIDTH_REF_M / base_pitch_width_m
    row[..., 19] = gf[..., GF_GOAL_WIDTH_NORM] * _LIVE_GOAL_WIDTH_REF_M / base_goal_width_m
    row[..., 20] = gf[..., GF_GOAL_HEIGHT_NORM] * _LIVE_GOAL_HEIGHT_REF_M / base_goal_height_m
    flat_identity = row[..., 0:7].reshape(-1, 7).detach().cpu().numpy()
    flat_dir = row[..., 12:14].reshape(-1, 2).detach().cpu().numpy()
    engineered = player_compute_engineered_features(flat_identity, flat_dir)
    row[..., 21:24] = torch.as_tensor(
        engineered, dtype=torch.float32, device=player_feat.device
    ).reshape(*shape, 3)
    return row


# Auxiliary head output dims -- fixed architectural constants (NOT
# config-driven), hardcoded identically in ball_dynamics_net.py's
# BallDynamicsAutoencoder.__init__ (crossing_head/resting_head/
# position_head/event_head, ball_dynamics_net.py:480-483) and
# player_dynamics_net.py's PlayerDynamicsAutoencoder.__init__
# (crossing_head/goal_dist_delta_head/short_horizon_head_0_2s/
# short_horizon_head_1_0s, player_dynamics_net.py:333-336). Deliberately
# excludes the DECODER (the main per-horizon reconstruction task) -- these
# are every OTHER (auxiliary) supervised probe trained on the same latent.
BALL_AUX_HEAD_DIMS: dict[str, int] = {
    "crossing_head": 4, "resting_head": 2, "position_head": 2, "event_head": 2,
}
PLAYER_AUX_HEAD_DIMS: dict[str, int] = {
    "crossing_head": 4, "goal_dist_delta_head": 2,
    "short_horizon_head_0_2s": 4, "short_horizon_head_1_0s": 4,
}


def _load_frozen_heads(model_state_dict: dict, latent_dim: int, head_dims: dict[str, int]) -> nn.ModuleDict:
    """Build a frozen nn.Linear(latent_dim, out_dim) per (name, out_dim) in
    head_dims, loading its weight/bias from model_state_dict's unprefixed
    "<name>.weight"/"<name>.bias" keys (each head is a direct top-level
    submodule of the full Autoencoder, e.g. train_ball_dynamics.py calls
    them as `model.crossing_head(latent)` -- single argument, no horizon
    conditioning, raw Linear output, same convention every head below
    follows)."""
    heads = nn.ModuleDict()
    for name, out_dim in head_dims.items():
        head = nn.Linear(latent_dim, out_dim)
        head.load_state_dict({
            "weight": model_state_dict[f"{name}.weight"],
            "bias": model_state_dict[f"{name}.bias"],
        })
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        heads[name] = head
    return heads


def _build_encoder_kwargs(cfg: dict[str, Any], input_dim: int) -> dict[str, Any]:
    """Same config-key -> constructor-kwarg mapping widen_*_checkpoint.py's
    _build_model() uses for the encoder-relevant subset (see that function)."""
    return dict(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        bottleneck_dim=cfg.get("encoder_bottleneck_dim", 32),
        identity_shortcut=cfg.get("identity_shortcut_enabled", False),
        identity_shortcut_noise_std=cfg.get("identity_shortcut_noise_std", 0.0),
        concat_all_input_fields=cfg.get("encoder_concat_all_input_fields", False),
        leaky_relu_negative_slope=cfg.get("encoder_leaky_relu_negative_slope", 0.0),
    )


def load_frozen_ball_encoder(
    checkpoint_path: str | Path,
) -> tuple[BallDynamicsEncoder, nn.ModuleDict, dict[str, Any]]:
    """Load a ball_encoder_*.pt checkpoint: build the BallDynamicsEncoder
    from its saved config_snapshot + encoder_state_dict, AND the frozen
    auxiliary heads (BALL_AUX_HEAD_DIMS) from the checkpoint's full
    model_state_dict (encoder_state_dict alone doesn't carry them -- they're
    separate top-level submodules of the full Autoencoder, not part of the
    encoder itself). Everything returned is frozen/eval() for inference-only
    use. Returns (encoder, aux_heads, config_snapshot)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config_snapshot"]
    encoder = BallDynamicsEncoder(**_build_encoder_kwargs(cfg, BALL_N_INPUT_FIELDS))
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    aux_heads = _load_frozen_heads(ckpt["model_state_dict"], int(cfg["latent_dim"]), BALL_AUX_HEAD_DIMS)
    return encoder, aux_heads, cfg


def load_frozen_player_encoder(
    checkpoint_path: str | Path,
) -> tuple[PlayerDynamicsEncoder, nn.ModuleDict, dict[str, Any]]:
    """Mirrors load_frozen_ball_encoder for PlayerDynamicsEncoder (PLAYER_AUX_HEAD_DIMS)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config_snapshot"]
    encoder = PlayerDynamicsEncoder(**_build_encoder_kwargs(cfg, PLAYER_N_INPUT_FIELDS))
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    aux_heads = _load_frozen_heads(ckpt["model_state_dict"], int(cfg["latent_dim"]), PLAYER_AUX_HEAD_DIMS)
    return encoder, aux_heads, cfg


def _peek_physics_output_dim(checkpoint_path: str | Path, aux_head_dims: dict[str, int]) -> int:
    """latent_dim + sum(aux_head_dims) for a checkpoint, WITHOUT constructing
    the encoder/aux-head modules -- just enough of a read to size a sibling
    network's input layers (e.g. ExecutionNetwork.from_config(), which needs
    the same widened ball_dim/self_dim as DecisionNetwork but never builds
    its own BallPhysicsFeatureBlock/PlayerPhysicsFeatureBlock -- see
    models/physics_encoders.py)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    return int(ckpt["config_snapshot"]["latent_dim"]) + sum(aux_head_dims.values())


def peek_ball_physics_output_dim(checkpoint_path: str | Path) -> int:
    return _peek_physics_output_dim(checkpoint_path, BALL_AUX_HEAD_DIMS)


def peek_player_physics_output_dim(checkpoint_path: str | Path) -> int:
    return _peek_physics_output_dim(checkpoint_path, PLAYER_AUX_HEAD_DIMS)
