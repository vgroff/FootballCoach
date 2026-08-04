"""Value-only opponent-AI-type side channel with a dedicated attention pool.

See ai/knowledge.md "Opponent-AI-type (value-only)" for the original design
rationale (feeds ONLY value_head, bypasses every policy head).

WHY THIS MODULE EXISTS (replacing a flatten+Linear side channel):
The original implementation flattened ``other_ai_type`` (batch,
MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM) into a single (batch, N*dim) vector
before a plain ``nn.Linear``. That gives each slot position its own learned
weight block, which means swapping which physical player sits in slot 3 vs
slot 7 changes the value output -- NOT permutation-invariant, unlike the
main entity encoder (shared per-slot MLP + attention pooling). Since
``ScenarioEnv``'s slot-shuffle augmentation is the only thing that was
papering over this (by showing many shuffles and training the value head
toward a consistent target across them), a dedicated pooling mechanism that
is exactly permutation-invariant by construction removes that dependency
for the value head specifically.

DESIGN: same pattern as ``EntityEncoder`` (shared per-slot MLP -> masked
attention pool), but a **completely separate, independent module** — no
weight sharing with ``EntityEncoder`` — so gradients from the value loss
never flow into the policy's entity encoder. Optionally enriched with a
**detached** copy of the main entity encoder's per-slot embeddings (richer
spatial/attribute context per opponent than a bare one-hot type flag alone),
passed in by the caller as ``self_embed``/``other_embed`` — the ``.detach()``
call is the caller's responsibility and is the load-bearing line that keeps
this value-only (see DecisionNetwork/ExecutionNetwork.forward()).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueAiTypeSideChannel(nn.Module):
    """Shared per-slot MLP + dedicated attention pool over opponent-ai-type
    (optionally concatenated with a detached entity embedding per slot).

    Exactly permutation-invariant: swapping which slot a real player
    occupies does not change the output, because the same per-slot MLP is
    applied identically to every slot and the attention-weighted sum over
    slots does not depend on slot order (same guarantee as EntityEncoder).
    """

    def __init__(
        self,
        ai_type_dim: int,
        entity_embed_dim: int = 0,
        hidden_dim: int = 16,
        num_heads: int = 1,
    ):
        super().__init__()
        self.entity_embed_dim = entity_embed_dim
        in_dim = ai_type_dim + entity_embed_dim

        # Shared-weight MLP applied identically to self and each other slot's
        # [ai_type_onehot; detached entity embedding] input.
        self.per_slot_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        self_ai_type: torch.Tensor,          # (batch, ai_type_dim)
        other_ai_type: torch.Tensor,         # (batch, MAX_OTHER_PLAYERS, ai_type_dim)
        exists_mask: torch.Tensor,           # (batch, MAX_OTHER_PLAYERS)
        self_embed: "torch.Tensor | None" = None,   # (batch, entity_embed_dim), already detached
        other_embed: "torch.Tensor | None" = None,  # (batch, MAX_OTHER_PLAYERS, entity_embed_dim), already detached
    ) -> torch.Tensor:  # (batch, hidden_dim)
        if self.entity_embed_dim > 0:
            assert self_embed is not None and other_embed is not None, (
                "ValueAiTypeSideChannel was constructed with entity_embed_dim>0 "
                "but no self_embed/other_embed was passed to forward()."
            )
            self_in = torch.cat([self_ai_type, self_embed], dim=-1)
            other_in = torch.cat([other_ai_type, other_embed], dim=-1)
        else:
            self_in = self_ai_type
            other_in = other_ai_type

        self_slot = self.per_slot_mlp(self_in).unsqueeze(1)   # (batch, 1, hidden_dim)
        other_slots = self.per_slot_mlp(other_in)             # (batch, N, hidden_dim)

        # True = ignore this key (inverse of exists_mask), matches EntityEncoder.
        key_padding_mask = exists_mask < 0.5

        context, _ = self.attention(
            query=self_slot,
            key=other_slots,
            value=other_slots,
            key_padding_mask=key_padding_mask,
        )
        return context.squeeze(1)  # (batch, hidden_dim)
