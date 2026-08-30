"""Observation encoder: Match + player_id -> ObservationBatch.

Converts the live match state into the normalized, padded, randomly-shuffled
tensor layout defined in obs/schema.py.  This is the bridge between the
engine's entities and the neural networks.

Key design choices (from ai_design_doc.md section 7):
- Positions encoded as (dx, dy) relative to the observing player, normalized
  by a FIXED reference pitch half-diagonal (see ``_REF_HALF_DIAG_M`` below) --
  NOT the live match's own pitch dimensions, so the same physical distance
  always maps to the same feature value regardless of episode pitch size.
- Velocities normalized by the same fixed pitch half-diagonal (absolute
  pitch-scale units).
- Other-player slots are shuffled randomly each call so the network learns
  permutation invariance (slot index carries no semantic meaning).
- Unused slots are zero-filled with exists=0.0.
- Time remaining: log1p-normalized (avoids squashing urgent endgame scenarios).
- Heading (heading_sin/heading_cos) and previous-decision movement intent
  (desired_dir_x/y + a STANDSTILL/JOG/SPRINT one-hot) are encoded for every
  player slot (self and others) -- see PlayerFeatures' own docstring.

``time_remaining_s`` must be passed in by the caller (the env wrapper tracks
the episode's remaining time; the engine only tracks elapsed ``time_s``).

This module ALWAYS encodes in raw world/engine-frame coordinates — it does
NOT apply the canonical-AI-frame mirror (observer's team always attacks
+x). That mirror is a thin wrapper applied at the network-forward boundary
instead — see ``ai/obs/canonical.py`` — so recorded/returned observations
stay in the same frame as the engine, match logs, and UI replays. Live
rollout code and BC dataset consumers both call ``canonical.py``'s
helpers immediately before/after using this encoder's output; this module
itself must never be changed to bake in team-conditioned mirroring.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.obs.schema import (
    AI_TYPE_ONE_HOT_DIM,
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_TASK_IDS,
    PLAYER_FEATURE_DIM,
    BallFeatures,
    GlobalFeatures,
    ObservationBatch,
    PlayerFeatures,
)
from footballcoach.engine.match import Match
from footballcoach.engine.movement import SpeedMode
from footballcoach.entities.player import Player, PlayerState, Team

MAX_OTHER_PLAYERS: int = 21  # full 11v11 minus self

# Reference/standard pitch dimensions used to normalize position and
# velocity features. Deliberately FIXED (not read from match.pitch) so the
# same physical distance/speed always maps to the same feature value
# regardless of the episode's pitch size -- pitch-scale curriculum
# (ai_config.json's curriculum.pitch_min_scale etc.) is not wired up yet,
# but WOULD silently corrupt this normalization if it ever is, since a
# live-pitch-derived divisor makes the same physical quantity encode
# differently across episodes for no reason the network can see (the value
# network in particular has no way to "explain away" that variance). Matches
# physics.json's pitch defaults, GlobalFeatures.pitch_length_norm/
# pitch_width_norm's own reference values just below, and
# ai/physics_pretrain's `normalize_kinematics_by_base_pitch` convention (see
# ai/physics_pretrain/live_encoder_features.py, which previously had to
# reconcile this live/fixed mismatch explicitly -- see ai/knowledge.md).
# The live pitch's actual size is still available to the network via
# GlobalFeatures.pitch_length_norm/pitch_width_norm (pitch.length_m/105.0,
# pitch.width_m/68.0) -- unaffected by this change.
_REF_PITCH_LENGTH_M: float = 105.0
_REF_PITCH_WIDTH_M: float = 68.0
_REF_HALF_DIAG_M: float = math.hypot(_REF_PITCH_LENGTH_M / 2.0, _REF_PITCH_WIDTH_M / 2.0)


def encode_observation(
    match: Match,
    player_id: str,
    time_remaining_s: float,
    attack_defence_smoothed: float = 0.5,
    rng: Optional[random.Random] = None,
    phase: Optional[int] = None,
) -> ObservationBatch:
    """Build the observation for a single player at the current match state.

    Args:
        match: The running Match instance.
        player_id: ID of the player whose perspective this observation is from.
        time_remaining_s: Seconds remaining in the episode (tracked by the
            env wrapper, not by the engine).
        attack_defence_smoothed: EMA-smoothed attack/defence weighting for
            this player (maintained externally, fed back as input per
            ai_design_doc.md section 2.7).
        rng: Optional Random for slot shuffling.  Pass a seeded instance for
            deterministic tests.
        phase: Active curriculum phase/task id (1-based, e.g. 1 for Phase 1).
            Populates the GlobalFeatures task-id one-hot at index
            ``phase - 1``.  ``None`` (default) encodes an all-zero task-id
            (scaffolding only, see ai/knowledge.md "Task-id: scaffolded, not
            yet load-bearing"). Out-of-range values (< 1 or > MAX_TASK_IDS)
            are clamped to an all-zero task-id rather than raising.

    Returns:
        ObservationBatch with all arrays ready for the neural network.
    """
    if rng is None:
        rng = random.Random()

    cfg = load_ai_config()
    obs_cfg = cfg["observation"]
    spin_norm = float(obs_cfg["ball_spin_nn_norm_rad_s"])
    time_norm_max = float(obs_cfg["time_remaining_norm_max_s"])
    height_norm_m = float(obs_cfg.get("height_norm_m", 3.0))

    self_player = _find_player(match, player_id)
    half_len = _REF_PITCH_LENGTH_M / 2.0
    half_wid = _REF_PITCH_WIDTH_M / 2.0
    half_diag = _REF_HALF_DIAG_M

    # Build self features
    self_feat = _player_features(
        player=self_player,
        observer=self_player,
        match=match,
        half_len=half_len,
        half_wid=half_wid,
        half_diag=half_diag,
        is_self=True,
    )

    # Collect other players (all except self), randomize slot assignment
    other_players = [p for p in match.players if p.player_id != player_id]
    n_other = min(len(other_players), MAX_OTHER_PLAYERS)

    slot_indices = rng.sample(range(MAX_OTHER_PLAYERS), k=n_other)

    other_feat = np.zeros((MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM), dtype=np.float32)
    exists_mask = np.zeros(MAX_OTHER_PLAYERS, dtype=np.float32)
    other_ai_type = np.zeros((MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM), dtype=np.float32)

    for slot_idx, other_player in zip(slot_indices, other_players[:MAX_OTHER_PLAYERS]):
        # A player with no AI won't move — zero their velocity to remove noise
        # from random initial heading/velocity, and set the is_immobile flag.
        other_immobile = other_player.ai is None
        feat = _player_features(
            player=other_player,
            observer=self_player,
            match=match,
            half_len=half_len,
            half_wid=half_wid,
            half_diag=half_diag,
            is_self=False,
            is_immobile=other_immobile,
        )
        other_feat[slot_idx] = feat
        exists_mask[slot_idx] = 1.0
        # Populated in the SAME loop iteration as other_feat/exists_mask so it
        # is structurally impossible for this to desync from the slot shuffle
        # (see ai/knowledge.md "Opponent-AI-type (value-only)").
        other_ai_type[slot_idx] = _ai_type_one_hot(other_player)

    self_ai_type = _ai_type_one_hot(self_player)

    # Ball features
    ball_feat = _ball_features(
        match=match,
        spin_norm=spin_norm,
        height_norm_m=height_norm_m,
    )

    # Global / match-context features
    global_feat = _global_features(
        match=match,
        observer=self_player,
        time_remaining_s=time_remaining_s,
        time_norm_max=time_norm_max,
        attack_defence_smoothed=attack_defence_smoothed,
        phase=phase,
    )

    return ObservationBatch(
        self_feat=self_feat,
        other_feat=other_feat,
        exists_mask=exists_mask,
        ball_feat=ball_feat,
        global_feat=global_feat,
        self_ai_type=self_ai_type,
        other_ai_type=other_ai_type,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ai_type_one_hot(player: Player) -> np.ndarray:
    """One-hot [is_rules, is_immobile, is_neural] for the AI controlling *player*.

    Value-only side channel (see ai/knowledge.md) - never fed to policy heads.
    ``player.ai is None`` is treated as immobile (matches the convention used
    throughout record_demonstrations.py / bc.py for the "no AI assigned"
    case). NeuralPlayerAI is checked by name rather than isinstance to avoid
    importing footballcoach.ai.env.scenario_env / rules_ai at module import
    time (would create a circular import: rules_ai -> ai.env -> ai.obs).
    """
    from footballcoach.rules_ai import Phase1RulesAI, NeuralPlayerAI

    one_hot = np.zeros(AI_TYPE_ONE_HOT_DIM, dtype=np.float32)
    if isinstance(player.ai, NeuralPlayerAI):
        one_hot[2] = 1.0  # is_neural
    elif isinstance(player.ai, Phase1RulesAI):
        one_hot[0] = 1.0  # is_rules
    else:
        one_hot[1] = 1.0  # is_immobile (player.ai is None, or any other/unknown AI)
    return one_hot


def _find_player(match: Match, player_id: str) -> Player:
    for p in match.players:
        if p.player_id == player_id:
            return p
    raise KeyError(f"Player '{player_id}' not found in match")


def _player_features(
    player: Player,
    observer: Player,
    match: Match,
    half_len: float,
    half_wid: float,
    half_diag: float,
    is_self: bool,
    is_immobile: bool = False,
) -> np.ndarray:
    """Encode one player's feature vector relative to the observer, in raw
    world/engine-frame coordinates (no canonical-frame mirroring — see
    module docstring).
    """
    dx = player.position.x - observer.position.x
    dy = player.position.y - observer.position.y
    dist = math.hypot(dx, dy)

    # Normalize velocity by pitch half-diagonal (absolute pitch-scale units).
    # For immobile players velocity is zeroed — their actual velocity is noise
    # (random initial heading, no movement intent), not a useful signal.
    vel_norm = max(half_diag, 1.0)
    if is_immobile:
        vel_x = 0.0
        vel_y = 0.0
        speed = 0.0
    else:
        vel_x = player.velocity.x / vel_norm
        vel_y = player.velocity.y / vel_norm
        speed = player.speed_mps / vel_norm

    ball = match.ball
    ball_dx = ball.position.x - player.position.x
    ball_dy = ball.position.y - player.position.y
    ball_dist = math.hypot(ball_dx, ball_dy)

    # Ball relative velocity and closing speed.
    # For immobile players player velocity is zero, so ball_vel_rel equals ball velocity.
    ball_vel_rel_x = (ball.velocity.x - player.velocity.x) / vel_norm
    ball_vel_rel_y = (ball.velocity.y - player.velocity.y) / vel_norm
    if ball_dist > 1e-6:
        # Positive = ball moving toward player; both numerator and direction flip together
        # under geometric augmentation so this scalar is flip-invariant.
        ball_closing_speed = -(ball.velocity.x - player.velocity.x) * (ball_dx / ball_dist) \
                             -(ball.velocity.y - player.velocity.y) * (ball_dy / ball_dist)
        ball_closing_speed /= vel_norm
    else:
        ball_closing_speed = 0.0

    # Team.LEFT attacks +x, Team.RIGHT attacks -x (per engine/offside.py convention)
    attacking_dir = +1.0 if player.team == Team.LEFT else -1.0

    # Heading: populated unconditionally, including at standstill/immobile --
    # a static facing direction is real signal, unlike velocity (which really
    # is noise for a never-moving player). See PlayerFeatures docstring.
    heading_sin = math.sin(player.heading_rad)
    heading_cos = math.cos(player.heading_rad)

    # Previous-decision movement intent: desired_direction is safe to read
    # directly (never auto-cleared); last_desired_speed_mode is the one that
    # survives match._apply_movement()'s per-tick reset of desired_speed_mode
    # (see Player.last_desired_speed_mode's docstring). Immobile players and
    # players with no decision yet both get the STANDSTILL/zero-direction
    # default -- "no movement intent" is real signal here, same rationale as
    # the is_immobile velocity-zeroing above.
    if is_immobile:
        desired_dir_x = 0.0
        desired_dir_y = 0.0
        speed_mode = SpeedMode.STANDSTILL
    else:
        d = player.desired_direction
        desired_dir_x = d.x
        desired_dir_y = d.y
        speed_mode = player.last_desired_speed_mode or SpeedMode.STANDSTILL
    desired_speed_standstill = 1.0 if speed_mode is SpeedMode.STANDSTILL else 0.0
    desired_speed_jog = 1.0 if speed_mode is SpeedMode.JOG else 0.0
    desired_speed_sprint = 1.0 if speed_mode is SpeedMode.SPRINT else 0.0

    feat = PlayerFeatures(
        rel_dx=dx / half_diag,
        rel_dy=dy / half_diag,
        distance_m=dist / half_diag,
        ball_rel_dx=ball_dx / half_diag,
        ball_rel_dy=ball_dy / half_diag,
        ball_distance_m=ball_dist / half_diag,
        ball_vel_rel_x=ball_vel_rel_x,
        ball_vel_rel_y=ball_vel_rel_y,
        ball_closing_speed=ball_closing_speed,
        velocity_x=vel_x,
        velocity_y=vel_y,
        speed_mps=speed,
        stamina=player.stamina,
        top_speed=player.attributes.top_speed,
        acceleration=player.attributes.acceleration,
        kick_power=player.attributes.kick_power,
        kick_precision=player.attributes.kick_precision,
        dribbling=player.attributes.dribbling,
        ball_control=player.attributes.ball_control,
        tackling=player.attributes.tackling,
        stamina_attr=player.attributes.stamina,
        is_own_team=1.0 if player.team == observer.team else 0.0,
        is_self=1.0 if is_self else 0.0,
        has_possession=1.0 if match.ball.possessed_by == player.player_id else 0.0,
        is_inactive_tackled=1.0 if player.state == PlayerState.INACTIVE_TACKLED else 0.0,
        is_controlling_ball=1.0 if player.state == PlayerState.CONTROLLING_BALL else 0.0,
        is_goalkeeper=1.0 if player.is_goalkeeper else 0.0,
        attacking_direction=attacking_dir,
        exists=1.0,
        is_immobile=1.0 if is_immobile else 0.0,
        pos_x=player.position.x / half_diag,
        pos_y=player.position.y / half_diag,
        heading_sin=heading_sin,
        heading_cos=heading_cos,
        desired_dir_x=desired_dir_x,
        desired_dir_y=desired_dir_y,
        desired_speed_standstill=desired_speed_standstill,
        desired_speed_jog=desired_speed_jog,
        desired_speed_sprint=desired_speed_sprint,
    )
    return feat.to_array()


def _ball_features(
    match: Match,
    spin_norm: float,
    height_norm_m: float,
) -> np.ndarray:
    ball = match.ball
    half_diag = _REF_HALF_DIAG_M
    vel_norm = max(half_diag, 1.0)

    is_possessed = 1.0 if ball.possessed_by is not None else 0.0

    # Team.LEFT attacks +x, Team.RIGHT attacks -x -- same convention as
    # PlayerFeatures.attacking_direction (engine/offside.py). 0.0 until the
    # first possession gain of the episode (Ball.last_touched_by_player_id
    # is None).
    last_touch_team_direction = 0.0
    if ball.last_touched_by_player_id is not None:
        last_toucher = match.player_by_id(ball.last_touched_by_player_id)
        last_touch_team_direction = +1.0 if last_toucher.team == Team.LEFT else -1.0

    feat = BallFeatures(
        pos_x=ball.position.x / half_diag,
        pos_y=ball.position.y / half_diag,
        height_m=ball.position.z / height_norm_m,
        velocity_x=ball.velocity.x / vel_norm,
        velocity_y=ball.velocity.y / vel_norm,
        velocity_z=ball.velocity.z / vel_norm,
        spin_x=ball.spin.x / max(spin_norm, 1e-3),
        spin_y=ball.spin.y / max(spin_norm, 1e-3),
        spin_z=ball.spin.z / max(spin_norm, 1e-3),
        is_possessed=is_possessed,
        is_loose=1.0 - is_possessed,
        last_touch_team_direction=last_touch_team_direction,
    )
    return feat.to_array()


def _global_features(
    match: Match,
    observer: Player,
    time_remaining_s: float,
    time_norm_max: float,
    attack_defence_smoothed: float,
    phase: Optional[int] = None,
) -> np.ndarray:
    pitch = match.pitch

    # Score diff from observer's team perspective
    sb = match.scoreboard
    if observer.team == Team.LEFT:
        score_diff = float(sb.left_goals - sb.right_goals)
    else:
        score_diff = float(sb.right_goals - sb.left_goals)

    # log1p normalization for time: spreads out the low end so "1-20s left"
    # scenarios are distinguishable from "2+ minutes left" (see design doc 7.5)
    t = max(time_remaining_s, 0.0)
    time_norm = math.log1p(t) / math.log1p(max(time_norm_max, 1.0))

    # Restitution coefficient: use the vertical bounce coefficient from params
    restitution = match.ball_physics_params.bounce_restitution_vertical

    task_id_kwargs = _task_id_one_hot_kwargs(phase)

    feat = GlobalFeatures(
        score_diff=score_diff,
        time_remaining_norm=time_norm,
        pitch_length_norm=pitch.length_m / 105.0,
        pitch_width_norm=pitch.width_m / 68.0,
        goal_width_norm=pitch.goal_width_m / 7.32,
        goal_height_norm=pitch.goal_height_m / 2.44,
        box_length_norm=pitch.box_length_m / 16.5,
        box_width_norm=pitch.box_width_m / 40.32,
        ball_restitution_coefficient=restitution,
        rng_reduction=match.rng_reduction,
        attack_defence_smoothed=attack_defence_smoothed,
        **task_id_kwargs,
    )
    return feat.to_array()


def _task_id_one_hot_kwargs(phase: Optional[int]) -> dict:
    """Build the ``task_id_N`` one-hot kwargs for GlobalFeatures.

    ``phase`` is 1-based (phase 1 -> index 0). None, or an out-of-range
    phase (< 1 or > MAX_TASK_IDS), yields an all-zero one-hot rather than
    raising — this is scaffolding, see ai/knowledge.md.
    """
    if phase is None or phase < 1 or phase > MAX_TASK_IDS:
        return {}
    return {f"task_id_{phase - 1}": 1.0}
