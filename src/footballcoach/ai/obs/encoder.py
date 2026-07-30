"""Observation encoder: Match + player_id -> ObservationBatch.

Converts the live match state into the normalized, padded, randomly-shuffled
tensor layout defined in obs/schema.py.  This is the bridge between the
engine's entities and the neural networks.

Key design choices (from ai_design_doc.md section 7):
- Positions encoded as (dx, dy) relative to the observing player, normalized
  by pitch half-dimensions.
- Velocities normalized by each player's own effective_top_speed.
- Other-player slots are shuffled randomly each call so the network learns
  permutation invariance (slot index carries no semantic meaning).
- Unused slots are zero-filled with exists=0.0.
- Time remaining: log1p-normalized (avoids squashing urgent endgame scenarios).

``time_remaining_s`` must be passed in by the caller (the env wrapper tracks
the episode's remaining time; the engine only tracks elapsed ``time_s``).
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    PLAYER_FEATURE_DIM,
    BallFeatures,
    GlobalFeatures,
    ObservationBatch,
    PlayerFeatures,
)
from footballcoach.engine.match import Match
from footballcoach.engine.movement import effective_top_speed
from footballcoach.entities.player import Player, PlayerState, Team

MAX_OTHER_PLAYERS: int = 21  # full 11v11 minus self


def encode_observation(
    match: Match,
    player_id: str,
    time_remaining_s: float,
    attack_defence_smoothed: float = 0.5,
    rng: Optional[random.Random] = None,
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

    Returns:
        ObservationBatch with all arrays ready for the neural network.
    """
    if rng is None:
        rng = random.Random()

    cfg = load_ai_config()
    obs_cfg = cfg["observation"]
    spin_norm = float(obs_cfg["ball_spin_norm_max_rad_s"])
    time_norm_max = float(obs_cfg["time_remaining_norm_max_s"])
    height_norm_m = float(obs_cfg.get("height_norm_m", 3.0))

    self_player = _find_player(match, player_id)
    pitch = match.pitch
    half_len = pitch.length_m / 2.0
    half_wid = pitch.width_m / 2.0
    half_diag = math.hypot(half_len, half_wid)

    mv_params = match.movement_params

    # Build self features
    self_feat = _player_features(
        player=self_player,
        observer=self_player,
        match=match,
        mv_params=mv_params,
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

    for slot_idx, other_player in zip(slot_indices, other_players[:MAX_OTHER_PLAYERS]):
        feat = _player_features(
            player=other_player,
            observer=self_player,
            match=match,
            mv_params=mv_params,
            half_len=half_len,
            half_wid=half_wid,
            half_diag=half_diag,
            is_self=False,
        )
        other_feat[slot_idx] = feat
        exists_mask[slot_idx] = 1.0

    # Ball features
    ball_feat = _ball_features(
        match=match,
        observer=self_player,
        half_len=half_len,
        half_wid=half_wid,
        half_diag=half_diag,
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
    )

    return ObservationBatch(
        self_feat=self_feat,
        other_feat=other_feat,
        exists_mask=exists_mask,
        ball_feat=ball_feat,
        global_feat=global_feat,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_player(match: Match, player_id: str) -> Player:
    for p in match.players:
        if p.player_id == player_id:
            return p
    raise KeyError(f"Player '{player_id}' not found in match")


def _player_features(
    player: Player,
    observer: Player,
    match: Match,
    mv_params,
    half_len: float,
    half_wid: float,
    half_diag: float,
    is_self: bool,
) -> np.ndarray:
    """Encode one player's feature vector relative to the observer."""
    dx = player.position.x - observer.position.x
    dy = player.position.y - observer.position.y
    dist = math.hypot(dx, dy)

    # Normalize this player's velocity by its own top speed (attribute-invariant)
    player_top_speed = effective_top_speed(
        mv_params,
        player.attributes.top_speed,
        player.stamina,
        has_ball=(match.ball.possessed_by == player.player_id),
        ball_control_attr=player.attributes.ball_control,
        is_goalkeeper=player.is_goalkeeper,
    )
    player_top_speed = max(player_top_speed, 1e-3)

    # Team.LEFT attacks +x, Team.RIGHT attacks -x (per engine/offside.py convention)
    attacking_dir = +1.0 if player.team == Team.LEFT else -1.0

    feat = PlayerFeatures(
        rel_dx=dx / half_len,
        rel_dy=dy / half_wid,
        distance_m=dist / half_diag,
        velocity_x=player.velocity.x / player_top_speed,
        velocity_y=player.velocity.y / player_top_speed,
        speed_mps=player.speed_mps / player_top_speed,
        heading_sin=math.sin(player.heading_rad),
        heading_cos=math.cos(player.heading_rad),
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
        pos_x=player.position.x / 52.5,
        pos_y=player.position.y / 34.0,
    )
    return feat.to_array()


def _ball_features(
    match: Match,
    observer: Player,
    half_len: float,
    half_wid: float,
    half_diag: float,
    spin_norm: float,
    height_norm_m: float,
) -> np.ndarray:
    ball = match.ball
    dx = ball.position.x - observer.position.x
    dy = ball.position.y - observer.position.y
    dist = math.hypot(dx, dy)

    # Velocity: normalize by half_diag (rough "pitch scale" per second)
    vel_norm = max(half_diag, 1.0)

    is_possessed = 1.0 if ball.possessed_by is not None else 0.0
    feat = BallFeatures(
        rel_dx=dx / half_len,
        rel_dy=dy / half_wid,
        distance_m=dist / half_diag,
        height_m=ball.position.z / height_norm_m,
        velocity_x=ball.velocity.x / vel_norm,
        velocity_y=ball.velocity.y / vel_norm,
        velocity_z=ball.velocity.z / vel_norm,
        spin_x=ball.spin.x / max(spin_norm, 1e-3),
        spin_y=ball.spin.y / max(spin_norm, 1e-3),
        spin_z=ball.spin.z / max(spin_norm, 1e-3),
        is_possessed=is_possessed,
        is_loose=1.0 - is_possessed,
    )
    return feat.to_array()


def _global_features(
    match: Match,
    observer: Player,
    time_remaining_s: float,
    time_norm_max: float,
    attack_defence_smoothed: float,
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
    )
    return feat.to_array()
