"""Diagnostic value network: predicts MC return from two FROZEN,
independently-pretrained physics encoders' latents (ball + the trainee's own
player state) plus time-remaining/elapsed, an is-possessed flag, the
opponent's relative position, and which team touched the ball last --
no entity attention (single fixed opponent slot, not attention over many),
no decision-network context.

The opponent's relative position was a late addition (see
compute_features()'s docstring): the original design excluded ALL opponent
info to isolate whether physics-only pretraining alone carries
reward-relevant signal. Kept as a small, targeted addition rather than a
full redesign -- two extra scalars, not entity attention -- to test whether
the immobile opponent's (verified genuinely varied) spawn position explains
some of the residual error the physics-only version couldn't close.

last_touch_team_direction (BallFeatures/Ball.last_touched_by_player_id) is a
similarly small, targeted addition: the FROZEN ball encoder itself has no
way to represent this (it was pretrained on pure free-flight ball dynamics,
which has no notion of team or possession history), so it's fed to the MLP
as a raw scalar alongside is_possessed/opp_rel_pos rather than through the
encoder. Motivated by the reward model's own touched-vs-untouched exit
distinction (ai/env/outcome.py's "miss"->ball_out vs "invalid": an untouched
exit scores 0, a touched one scores ball_out_penalty) -- which team touched
last changes whether a given loose-ball trajectory is heading for a
recoverable "invalid" or an already-locked-in "ball_out", something neither
physics encoder nor the opponent-position scalar can distinguish.

Both players' full 8-field PlayerAttributes block (self_attrs, opp_attrs) is
a third such addition: the frozen player encoder only ever runs on the
TRAINEE's own state (self_feat) and only exposes a subset of attributes to
it (top_speed/acceleration/ball_control/stamina_attr -- see
player_live_to_physics_input) -- kick_power/kick_precision/dribbling/
tackling never reach this net at all otherwise, and the opponent's
attributes never reach it through any path (only opp_rel_pos, a bare
position). Read directly off self_feat_c/other_feat_c (PF_ATTRIBUTES_SLICE)
rather than through either encoder, since attributes are static and cheap.

Isolates whether physics-only pretraining (never exposed to reward/task
supervision) already carries reward-relevant signal, on a rules-AI-vs-
immobile-opponent dataset. See ``debug_value_network.py``'s
``--physics-encoder-value-net`` CLI flag, the sole intended caller.

Deliberately does NOT import anything from ``ai/models/`` (matches this
package's own stated boundary, see ``ai/physics_pretrain/__init__.py``) --
``PhysicsValueHeads`` is a minimal local return type, not
``ai/action/schema.py``'s ``ExecutionHeadsRaw``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from footballcoach.ai.obs.canonical import canonicalize_bc_labels, canonicalize_obs, x_sign_of
from footballcoach.ai.physics_pretrain.live_encoder_features import (
    BF_IS_LOOSE,
    BF_IS_POSSESSED,
    BF_LAST_TOUCH_TEAM_DIRECTION,
    PF_ATTRIBUTES_SLICE,
    PhysicsPitchConstants,
    ball_live_to_physics_input,
    load_frozen_ball_encoder,
    load_frozen_player_encoder,
    load_physics_pitch_constants,
    player_live_to_physics_input,
    time_remaining_and_elapsed_fraction,
)
from footballcoach.ai.ppo.bc import _I_HEADING_COS


@dataclass
class PhysicsValueHeads:
    """Minimal .value-bearing return type -- see module docstring for why
    this isn't ai/action/schema.py's ExecutionHeadsRaw."""
    value: torch.Tensor  # (batch, 1)


class PhysicsEncoderValueNet(nn.Module):
    """ball_encoder/player_encoder must already be frozen (requires_grad_(False))
    and in .eval() mode -- see load_frozen_ball_encoder/load_frozen_player_encoder,
    or use from_checkpoints() below which handles that.

    Ball latent is masked to zero whenever the ball isn't loose (the ball
    encoder was trained purely on free-flight physics and never saw "ball
    glued to a moving carrier" -- feeding it held-ball state would be
    out-of-distribution). Player latent is NOT masked: has_possession is
    already one of the player encoder's own training inputs, so it
    generalizes across both possession states by construction. A separate
    is_possessed scalar is fed directly into the MLP so it can still tell
    *why* the ball latent went blank.

    No explicit ball-to-player distance input: both encoders have
    identity_shortcut_enabled=true, hand-initialized to preserve raw absolute
    position in the first latent dimensions, normalized against the SAME
    fixed base-pitch frame -- the downstream MLP can learn distance itself
    from the two latents' preserved positions.

    Canonicalized to the SAME "always attacking +x" frame the real
    ExecutionNetwork/DecisionNetwork train on (ai/obs/canonical.py), since
    attacking_direction varies genuinely across episodes (~50/50 in real
    recordings) and neither physics encoder has any way to know which raw-x
    direction means "progress" otherwise -- their identity-shortcut-preserved
    absolute pos_x/vel_x would mean opposite things for two otherwise-
    identical episodes with opposite attacking_direction. Reuses
    canonicalize_obs()/canonicalize_bc_labels() exactly as the production
    networks do (self_feat/ball_feat mirrored, including the ball spin
    pseudovector's flip_x rule: spin_x unchanged, spin_y/spin_z negated,
    NOT the naive polar-vector rule); heading_cos is negated by hand
    (heading_sin unchanged) since canonicalize_bc_labels doesn't cover it
    -- this codebase has never wired that up anywhere else either, since
    heading_sin/cos didn't exist as a stored field before this diagnostic
    added them. After canonicalization, attacking_direction is constant
    (+1.0) for every row, so it carries no information and is correctly
    never fed to the MLP as a separate scalar.

    Also concatenates every AUXILIARY head's raw output onto its encoder's
    latent -- crossing_head/resting_head/position_head/event_head for ball,
    crossing_head/goal_dist_delta_head/short_horizon_head_0_2s/
    short_horizon_head_1_0s for player (see live_encoder_features.py's
    BALL_AUX_HEAD_DIMS/PLAYER_AUX_HEAD_DIMS) -- deliberately excluding only
    the main per-horizon reconstruction DECODER. Each head is called as
    head(latent) (same convention the training scripts themselves use),
    raw/unactivated output, no extra processing. Ball's aux head outputs are
    masked by is_loose exactly like the ball latent itself (same
    out-of-distribution-while-held argument applies to every one of the
    ball network's outputs, not just its latent); player's aux head outputs
    are unmasked, same reasoning as the player latent.

    None of this (encoders, aux heads, canonicalization, time/possession
    scalars) depends on any trainable parameter -- self.mlp is the ONLY
    trainable part of this module. See compute_features()/
    precompute_and_cache_features(): the whole feature vector is a pure,
    deterministic function of a row's stored data, so it can be computed
    ONCE per dataset row and cached rather than recomputed every epoch.
    """

    def __init__(
        self,
        ball_encoder: nn.Module,
        player_encoder: nn.Module,
        ball_aux_heads: nn.ModuleDict,
        player_aux_heads: nn.ModuleDict,
        ball_cfg_snapshot: dict[str, Any],
        player_cfg_snapshot: dict[str, Any],
        mlp_hidden: int = 64,
        live_ball_spin_nn_norm_rad_s: float = 55.0,
        live_height_norm_m: float = 3.0,
        time_norm_max_s: float = 7200.0,
        max_episode_s: float = 60.0,
    ):
        super().__init__()
        self.ball_encoder = ball_encoder
        self.player_encoder = player_encoder
        self.ball_aux_heads = ball_aux_heads
        self.player_aux_heads = player_aux_heads
        self._ball_cfg = ball_cfg_snapshot
        self._player_cfg = player_cfg_snapshot
        self._live_ball_spin_nn_norm_rad_s = live_ball_spin_nn_norm_rad_s
        self._live_height_norm_m = live_height_norm_m
        self._time_norm_max_s = time_norm_max_s
        self._max_episode_s = max_episode_s
        # Shared base-pitch dims (105x68m, standard goal) physics-pretrain's
        # episode generators normalize against -- NOT part of a checkpoint's
        # own config_snapshot (that's ai_config.json's physics_pretrain
        # section, which has no pitch-dimension keys), sourced from
        # physics.json instead, same physical pitch either way.
        self._pitch_constants: PhysicsPitchConstants = load_physics_pitch_constants()
        # Populated by precompute_and_cache_features(); forward() uses it
        # (indexed by row_idx) instead of recomputing when available.
        self._feature_cache: torch.Tensor | None = None

        ball_latent_dim = int(ball_cfg_snapshot["latent_dim"])
        player_latent_dim = int(player_cfg_snapshot["latent_dim"])
        ball_aux_dim = sum(h.out_features for h in ball_aux_heads.values())
        player_aux_dim = sum(h.out_features for h in player_aux_heads.values())
        # + time_remaining, time_elapsed, is_possessed, opp_rel_dx, opp_rel_dy,
        # last_touch_team_direction, self attributes (8), opponent attributes (8)
        in_dim = ball_latent_dim + ball_aux_dim + player_latent_dim + player_aux_dim + 6 + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def compute_features(
        self,
        self_feat: torch.Tensor,
        other_feat: torch.Tensor,
        ball_feat: torch.Tensor,
        global_feat: torch.Tensor,
        labels: torch.Tensor,
        exists_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pure, autograd-free computation of the full MLP input vector --
        masked [ball latent + ball aux head outputs], [player latent +
        player aux head outputs], time_remaining, time_elapsed, is_possessed,
        opponent relative position. Nothing here depends on any trainable
        parameter (self.mlp is the only trainable part of this module), so
        it's deterministic given the inputs alone -- safe to compute once
        per dataset row and cache, see precompute_and_cache_features().

        Opponent relative position (rel_dx, rel_dy, already self-relative
        and canonicalized -- see class docstring) is a late addition: this
        net was originally "no entity attention, no opponent info" by
        design, to isolate whether the physics encoders alone carry
        reward-relevant signal. Added to test a concrete hypothesis instead
        of leaving it untested: the rules-vs-immobile matchup's immobile
        opponent spawns at a genuinely varied position every episode
        (verified against real data), which this net had no way to
        observe -- the same [ball latent, own-player latent, time] input
        could correspond to an opponent sitting directly in the trainee's
        path or nowhere near it, indistinguishable without this.

        last_touch_team_direction (see class docstring) is a second such
        late addition, same reasoning: a raw scalar the frozen encoders
        can't represent, appended after opp_rel_pos.

        self_attrs/opp_attrs (see class docstring) is a third such addition:
        both players' full PlayerAttributes block, appended last.

        All three REQUIRE a fresh --init-value-checkpoint-less run against
        a checkpoint trained before that field existed -- each one changed
        the MLP's input width, so an old checkpoint will NOT load into this
        shape (no warm start from a net trained on a narrower input)."""
        with torch.no_grad():
            # Canonicalize to "always attacking +x" -- see class docstring.
            # Mirrors self_feat/ball_feat exactly like the production
            # networks (PLAYER_FLIP_X_IDX/BALL_FLIP_X_IDX, including the
            # spin pseudovector rule), plus canonicalize_bc_labels() for
            # dir_x/kick fields, plus a manual heading_cos negation
            # (heading_sin unchanged) since that field didn't exist
            # anywhere in this codebase before this diagnostic added it.
            # other_feat_c (canonicalized/mirrored other_feat, same
            # PLAYER_FLIP_X_IDX rule as self_feat) was previously discarded
            # here -- now used below for the opponent relative position.
            x_sign = x_sign_of(self_feat)
            self_feat_c, other_feat_c, ball_feat_c, _ = canonicalize_obs(
                self_feat, other_feat, ball_feat, x_sign=x_sign,
            )
            labels_c = canonicalize_bc_labels(labels, x_sign)
            labels_c[:, _I_HEADING_COS] = labels_c[:, _I_HEADING_COS] * x_sign

            ball_in = ball_live_to_physics_input(
                ball_feat_c, global_feat, self._pitch_constants,
                live_ball_spin_nn_norm_rad_s=self._live_ball_spin_nn_norm_rad_s,
                live_height_norm_m=self._live_height_norm_m,
            )
            player_in = player_live_to_physics_input(self_feat_c, global_feat, labels_c, self._pitch_constants)

            is_loose = ball_feat[:, BF_IS_LOOSE:BF_IS_LOOSE + 1]
            ball_latent = self.ball_encoder(ball_in)
            ball_aux = [head(ball_latent) for head in self.ball_aux_heads.values()]
            ball_full = torch.cat([ball_latent, *ball_aux], dim=-1) * is_loose

            player_latent = self.player_encoder(player_in)
            player_aux = [head(player_latent) for head in self.player_aux_heads.values()]
            player_full = torch.cat([player_latent, *player_aux], dim=-1)

            time_rem, time_elapsed = time_remaining_and_elapsed_fraction(
                global_feat, self._time_norm_max_s, self._max_episode_s,
            )
            is_possessed = ball_feat[:, BF_IS_POSSESSED:BF_IS_POSSESSED + 1]

            # Phase 1 is 1v1 -- exactly one real slot in other_feat, at a
            # RANDOMIZED index per row (see ai/obs/encoder.py's slot
            # shuffle); exists_mask says which one. rel_dx/rel_dy are
            # PlayerFeatures fields 0/1 (declared first), already
            # self-relative ((other.x - self.x)/half_diag) and now
            # canonicalized via other_feat_c -- exactly the "attacking +x"
            # frame the rest of this net's inputs are already in.
            opp_slot = exists_mask.argmax(dim=-1)
            opp_rel_pos = other_feat_c[torch.arange(other_feat_c.shape[0]), opp_slot, 0:2]

            # Full 8-field PlayerAttributes block, both players, as explicit
            # MLP scalars -- a third late addition, same motivation as
            # opp_rel_pos/last_touch_team_direction above: this net's only
            # OTHER access to attributes is whatever subset reaches the
            # frozen player encoder via player_live_to_physics_input
            # (top_speed/acceleration/ball_control/stamina_attr, self only
            # -- see that function), which silently excludes kick_power/
            # kick_precision/dribbling/tackling entirely and never sees the
            # OPPONENT's attributes at all (only self_feat is ever run
            # through self.player_encoder -- the opponent contributes only
            # opp_rel_pos above, no encoder pass of its own). Attributes are
            # static and cheap to read directly off self_feat_c/other_feat_c
            # rather than requiring a second (unfrozen or duplicated) encoder
            # pass just to expose them.
            self_attrs = self_feat_c[:, PF_ATTRIBUTES_SLICE]
            opp_attrs = other_feat_c[torch.arange(other_feat_c.shape[0]), opp_slot, PF_ATTRIBUTES_SLICE]

            # Which team touched the ball last (0 until the first touch of
            # the episode, then +-1 and persists through loose-ball periods
            # -- see entities/ball.py's Ball.last_touched_by_player_id and
            # BallFeatures.last_touch_team_direction's own docstrings).
            # Direction-DEPENDENT (a team-direction sign, not a boolean like
            # is_possessed/is_loose above), so this MUST come from the
            # canonicalized ball_feat_c, not raw ball_feat -- it's already
            # in BALL_FLIP_X_IDX (obs/augment.py), so canonicalize_obs()
            # above has already negated it correctly for the "always
            # attacking +x" frame every other input here is in.
            last_touch_team_direction = ball_feat_c[
                :, BF_LAST_TOUCH_TEAM_DIRECTION:BF_LAST_TOUCH_TEAM_DIRECTION + 1
            ]

            return torch.cat(
                [ball_full, player_full, time_rem, time_elapsed, is_possessed, opp_rel_pos,
                 last_touch_team_direction, self_attrs, opp_attrs], dim=-1,
            )

    def precompute_and_cache_features(
        self, ds: Any, row_indices: np.ndarray, batch_size: int = 8192,
    ) -> torch.Tensor:
        """Runs compute_features() once over every row in row_indices
        (chunked to bound peak memory), caching the result in a
        (len(ds), in_dim) tensor -- ALSO assigned to self._feature_cache
        (forward() then looks features up by row_idx instead of recomputing
        the frozen encoder+heads pipeline every single epoch -- pure waste
        otherwise, since none of it depends on self.mlp's (the only
        trainable part) current weights) AND returned, so a caller juggling
        more than one cache (e.g. debug_value_network.py's
        --max-episodes-per-epoch, which swaps self._feature_cache between a
        static val cache and a per-epoch-rebuilt train cache) can hold onto
        both without them clobbering each other. Rows not included in
        row_indices are left as zero and must never be looked up (forward()
        falls back to computing on the fly whenever row_idx isn't given, so
        this is only a correctness requirement if you pass row_idx for a row
        outside what was precomputed here)."""
        from footballcoach.ai.bc.dataset import _to_tensor

        # Infer device from this module's own (already-.to(device)'d)
        # parameters, rather than taking a device argument -- matches how
        # forward()'s on-the-fly path already just inherits whatever device
        # its caller's tensors happen to be on.
        device = next(self.mlp.parameters()).device
        row_indices = np.asarray(row_indices)
        in_dim = self.mlp[0].in_features
        cache = torch.zeros((len(ds), in_dim), dtype=torch.float32, device=device)
        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start:start + batch_size]
            self_feat = _to_tensor(ds._self_feat[chunk], device)
            other_feat = _to_tensor(ds._other_feat[chunk], device)
            ball_feat = _to_tensor(ds._ball_feat[chunk], device)
            global_feat = _to_tensor(ds._global_feat[chunk], device)
            labels = _to_tensor(ds._labels[chunk], device)
            exists_mask = _to_tensor(ds._exists_mask[chunk], device)
            cache[chunk] = self.compute_features(self_feat, other_feat, ball_feat, global_feat, labels, exists_mask)
        self._feature_cache = cache
        return cache

    def compute_features_for_files_cached(
        self, ds: Any, files: list, file_row_counts: dict, file_feature_cache: dict,
    ) -> torch.Tensor:
        """Like precompute_and_cache_features(), but keyed PER-FILE rather
        than per-absolute-row-index -- for --max-episodes-per-epoch's
        resampled training working set, where ``ds`` gets rebuilt from a
        different file list every epoch (so a plain row-index cache would be
        meaningless from one epoch to the next). ``file_feature_cache``
        (owned and persisted ACROSS EPOCHS by the caller, not self) maps
        file path -> that file's own (n_rows_in_file, in_dim) feature
        tensor; only files not already in it get run through the frozen
        encoder+heads pipeline, one file at a time (file sizes are small,
        typically a few hundred rows -- see _peek_episode_count -- so this
        forgoes cross-file batching for simplicity, not for lack of a faster
        option). ``ds`` MUST have been built via
        ``DemonstrationDataset.from_files(files)`` (same order) so file
        boundaries line up with ``file_row_counts``.

        Returns a (len(ds), in_dim) tensor in ds's row order, built by
        concatenating each file's (cached-or-freshly-computed) tensor --
        assign it to value_net._feature_cache yourself (this method
        deliberately doesn't touch that attribute, since the caller also
        needs a SEPARATE static cache for the val set on the same module --
        see debug_value_network.py's per-epoch cache-swap)."""
        from footballcoach.ai.bc.dataset import _to_tensor

        device = next(self.mlp.parameters()).device
        offset = 0
        pieces = []
        for f in files:
            n = file_row_counts[f]
            if f not in file_feature_cache:
                chunk = np.arange(offset, offset + n)
                self_feat = _to_tensor(ds._self_feat[chunk], device)
                other_feat = _to_tensor(ds._other_feat[chunk], device)
                ball_feat = _to_tensor(ds._ball_feat[chunk], device)
                global_feat = _to_tensor(ds._global_feat[chunk], device)
                labels = _to_tensor(ds._labels[chunk], device)
                exists_mask = _to_tensor(ds._exists_mask[chunk], device)
                file_feature_cache[f] = self.compute_features(
                    self_feat, other_feat, ball_feat, global_feat, labels, exists_mask,
                )
            pieces.append(file_feature_cache[f])
            offset += n
        return torch.cat(pieces, dim=0)

    def forward(
        self,
        self_feat: torch.Tensor,
        other_feat: torch.Tensor,
        exists_mask: torch.Tensor,
        ball_feat: torch.Tensor,
        global_feat: torch.Tensor,
        decision_heads: Any,
        self_ai_type: torch.Tensor | None = None,
        other_ai_type: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        row_idx: torch.Tensor | np.ndarray | None = None,
    ) -> PhysicsValueHeads:
        """Matches the existing 8-positional-arg value_net(...) call-site
        contract used throughout debug_value_network.py exactly (all new/
        extra params are keyword-only with defaults). decision_heads/
        self_ai_type/other_ai_type are accepted and IGNORED -- decision-
        network context and full multi-entity attention are still
        deliberately excluded. other_feat/exists_mask are USED (opponent
        relative position -- see compute_features()'s docstring for why),
        despite the "no opponent info" framing in the class docstring
        predating that addition.

        If a feature cache has been built (precompute_and_cache_features())
        and row_idx is given, looks features up there instead of recomputing
        -- see compute_features()'s docstring for why that's always safe.
        Otherwise falls back to computing on the fly, requiring labels (see
        compute_features()) and raising ValueError if labels is None --
        fail fast rather than silently producing garbage. This is the
        enforcement point for incompatibility with
        --synthetic-ball-out-episodes/--synthetic-timeout-episodes, whose
        ObservationBatch rows have no BC-label equivalent at all (no
        heading/desired-direction/desired-speed-mode source) and are never
        passed a row_idx into a precomputed cache either."""
        if self._feature_cache is not None and row_idx is not None:
            features = self._feature_cache[row_idx]
        else:
            if labels is None:
                raise ValueError(
                    "PhysicsEncoderValueNet requires BC labels (heading/desired-direction/"
                    "speed-mode reconstruction) -- incompatible with "
                    "--synthetic-ball-out-episodes/--synthetic-timeout-episodes, whose rows "
                    "have no BC-label source."
                )
            features = self.compute_features(self_feat, other_feat, ball_feat, global_feat, labels, exists_mask)
        return PhysicsValueHeads(value=self.mlp(features))

    @classmethod
    def from_checkpoints(
        cls,
        ball_checkpoint_path: str | Path,
        player_checkpoint_path: str | Path,
        mlp_hidden: int = 64,
        live_ball_spin_nn_norm_rad_s: float = 55.0,
        live_height_norm_m: float = 3.0,
        time_norm_max_s: float = 7200.0,
        max_episode_s: float = 60.0,
    ) -> "PhysicsEncoderValueNet":
        ball_encoder, ball_aux_heads, ball_cfg = load_frozen_ball_encoder(ball_checkpoint_path)
        player_encoder, player_aux_heads, player_cfg = load_frozen_player_encoder(player_checkpoint_path)
        return cls(
            ball_encoder, player_encoder, ball_aux_heads, player_aux_heads, ball_cfg, player_cfg,
            mlp_hidden=mlp_hidden,
            live_ball_spin_nn_norm_rad_s=live_ball_spin_nn_norm_rad_s,
            live_height_norm_m=live_height_norm_m,
            time_norm_max_s=time_norm_max_s,
            max_episode_s=max_episode_s,
        )
