"""Entity encoder: shared per-entity MLP + multi-head attention pooling.

See ai_design_doc.md section 8.1 for the full architecture specification.

The same shared-weight MLP is run over:
  - The self-player feature vector (query for the attention)
  - Each of the up to 21 other-player feature vectors (keys/values)

Padded slots (exists=0) are masked out with -inf before the attention
softmax so they contribute exactly zero.

Edge case (noted in design doc 8.1): if ALL other-player slots are masked
(entirely zero exists_mask for a batch row), nn.MultiheadAttention produces
NaN (softmax over all -inf).  This won't occur in the current curriculum
(even a 1v1 scenario has 1 other player) but is documented and guarded.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EntityEncoder(nn.Module):
    """Shared per-entity MLP + multi-head attention over the other-player slots.

    Args:
        entity_feature_dim: Length of each player feature vector
            (PLAYER_FEATURE_DIM from obs/schema.py).
        embed_dim: Output embedding dimension per entity (and the attention
            query/key/value dimension).
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        entity_feature_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Shared-weight MLP applied identically to self and each other player.
        # The self slot's embedding becomes the attention query; the other-
        # player embeddings become keys and values.
        self.per_entity_mlp = nn.Sequential(
            nn.Linear(entity_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # batch_first=True: input shape (batch, seq, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        self_features: torch.Tensor,   # (batch, entity_feature_dim)
        other_features: torch.Tensor,  # (batch, MAX_OTHER_PLAYERS, entity_feature_dim)
        exists_mask: torch.Tensor,     # (batch, MAX_OTHER_PLAYERS), 1.0=real 0.0=padded
        return_embeds: bool = False,
    ):
        """
        Args:
            self_features: The observing player's own feature vector.
            other_features: All other-player slots (padded with zeros where
                exists_mask=0).
            exists_mask: Float mask, 1.0 for a real player, 0.0 for padding.
            return_embeds: If True, also return the pre-attention per-slot
                embeddings (self_embed, other_embed) alongside the pooled
                context. Callers that reuse these elsewhere (e.g. the
                value-only ai-type side channel, see
                ai/models/value_side_channel.py) MUST ``.detach()`` them
                first -- these embeddings are the live policy path's
                encoder output and are still attached to the policy's
                autograd graph here.

        Returns:
            Attention-pooled context vector (batch, embed_dim) summarising
            all relevant information about the other players, weighted by
            how relevant each is to the observing player's current state.
            If ``return_embeds=True``, returns
            ``(context, self_embed, other_embed)`` instead, where
            self_embed is (batch, embed_dim) and other_embed is
            (batch, MAX_OTHER_PLAYERS, embed_dim) -- both PRE-pooling,
            still attached to the policy graph (caller must detach).
        """
        # Embed self: (batch, embed_dim) -> (batch, 1, embed_dim) for attention query
        self_embed_raw = self.per_entity_mlp(self_features)
        self_embed = self_embed_raw.unsqueeze(1)

        # Embed all other players: (batch, MAX_OTHER_PLAYERS, embed_dim)
        other_embed = self.per_entity_mlp(other_features)

        # nn.MultiheadAttention uses key_padding_mask where True = "ignore this key"
        # i.e. the INVERSE of exists_mask.
        key_padding_mask = exists_mask < 0.5  # (batch, MAX_OTHER_PLAYERS), True = padded

        # Edge case (see module docstring): a batch row with ZERO real other
        # players (all slots padded, e.g. UI training mode's 1-player-only
        # match) makes that row's key_padding_mask all True -> softmax over
        # an all -inf row -> NaN, which then poisons every downstream head
        # (even ones unrelated to other-player info) since NaN propagates
        # through the shared trunk. Since other_embed is all zeros for a
        # fully-padded row anyway (per_entity_mlp applied to zero-padded
        # other_features -- not exactly zero after the MLP bias/ReLU, but
        # harmless either way), unmask those rows entirely so attention
        # degrades to "attend to the zero-padded slots" (a fixed, finite,
        # uninformative context) instead of NaN.
        fully_padded_rows = key_padding_mask.all(dim=1)  # (batch,)
        if fully_padded_rows.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded_rows] = False

        context, _ = self.attention(
            query=self_embed,
            key=other_embed,
            value=other_embed,
            key_padding_mask=key_padding_mask,
        )
        # context: (batch, 1, embed_dim) -> (batch, embed_dim)
        context = context.squeeze(1)
        if return_embeds:
            return context, self_embed_raw, other_embed
        return context
