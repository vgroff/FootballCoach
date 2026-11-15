"""Prepare a trained checkpoint's input layers for phases it has never seen.

An input feature that was always 0 during training never gets a gradient, so
the first-layer weights reading it stay at their random initial values.
Measured on phase1_run321 (checkpoint1 vs checkpoint763): the task_id_1 and
is_goalkeeper columns are bit-identical across the whole run, with norms at
the median random-init column norm (e.g. global_mlp: task_id_1 0.61 vs median
0.61), while the trained task_id_0 column has norm 3.65. A phase-2 observation
(task_id_1 = 1, task_id_0 = 0) would therefore lose task_id_0's trained
contribution and gain a random one, and a keeper's flag would add random noise.

init_new_phase_inputs() fixes both, in place, on every network:
- copies the source phase's task-id column into each new phase's column, so a
  new phase's observations are first processed exactly like the source
  phase's (the columns then diverge through training);
- zeroes the named player-feature columns (e.g. is_goalkeeper), so the flag
  starts with no effect.

The physics encoders read only specific global fields (pitch size,
restitution, time), never the task ids, so they need nothing here. Layers are
named explicitly: some hidden layers have the same width as an input layer
(execution_net.value_head.1 takes 113 inputs, like self_mlp.0).
"""
from __future__ import annotations

import dataclasses

import torch

from footballcoach.ai.obs.schema import GLOBAL_FEATURE_DIM, MAX_TASK_IDS, PLAYER_FEATURE_DIM, GlobalFeatures, PlayerFeatures

# First layers reading global_feat directly (input width GLOBAL_FEATURE_DIM).
GLOBAL_INPUT_LAYERS: tuple[str, ...] = ("global_mlp.0.weight", "entity_encoder.global_query_proj.weight")
# First layers reading a player's own or another player's features (input =
# PlayerFeatures columns first, then any physics-encoder outputs).
PLAYER_INPUT_LAYERS: tuple[str, ...] = ("self_mlp.0.weight", "entity_encoder.per_entity_mlp.0.weight")


def _networks(trainer) -> dict:
    nets = {"decision_net": trainer.decision_net, "execution_net": trainer.execution_net}
    if getattr(trainer, "value_net", None) is not None:
        nets["value_net"] = trainer.value_net
    return nets


def _layers(net, suffixes) -> list[tuple[str, torch.nn.Parameter]]:
    """(layer name without wrapper prefixes, parameter) for each match; the
    networks are wrapped (e.g. ``_wrapped._wrapped.global_mlp.0.weight``)."""
    out = []
    for name, p in net.named_parameters():
        for s in suffixes:
            if name == s or name.endswith("." + s):
                out.append((s, p))
    return out


def init_new_phase_inputs(trainer, *, from_phase: int, to_phases, zero_player_features=()) -> list[dict]:
    """See the module docstring. Returns one report row per column touched:
    {network, layer, column, action, norm_before, norm_after}; ``layer`` is
    the name without wrapper prefixes."""
    g_names = [f.name for f in dataclasses.fields(GlobalFeatures)]
    p_names = [f.name for f in dataclasses.fields(PlayerFeatures)]
    for pid in (from_phase, *to_phases):
        if not 1 <= pid <= MAX_TASK_IDS:
            raise ValueError(f"phase {pid} has no task-id slot (1..{MAX_TASK_IDS})")
    src = g_names.index(f"task_id_{from_phase - 1}")
    dsts = [g_names.index(f"task_id_{p - 1}") for p in to_phases if p != from_phase]
    zero_cols = [p_names.index(n) for n in zero_player_features]
    report: list[dict] = []
    found_global = found_player = 0
    with torch.no_grad():
        for net_name, net in _networks(trainer).items():
            for layer, w in _layers(net, GLOBAL_INPUT_LAYERS):
                if w.shape[1] != GLOBAL_FEATURE_DIM:
                    raise ValueError(f"{net_name}.{layer}: {w.shape[1]} inputs, expected {GLOBAL_FEATURE_DIM}")
                found_global += 1
                for d in dsts:
                    before = float(w[:, d].norm())
                    w[:, d] = w[:, src]
                    report.append(dict(network=net_name, layer=layer, column=g_names[d], action=f"copy {g_names[src]}",
                                       norm_before=before, norm_after=float(w[:, d].norm())))
            for layer, w in _layers(net, PLAYER_INPUT_LAYERS):
                if w.shape[1] < PLAYER_FEATURE_DIM:
                    raise ValueError(f"{net_name}.{layer}: {w.shape[1]} inputs, fewer than {PLAYER_FEATURE_DIM} player features")
                found_player += 1
                for c in zero_cols:
                    before = float(w[:, c].norm())
                    w[:, c] = 0.0
                    report.append(dict(network=net_name, layer=layer, column=p_names[c], action="zero",
                                       norm_before=before, norm_after=0.0))
    if found_global == 0 or (zero_cols and found_player == 0):
        raise ValueError("no input layers found (network layout changed? update GLOBAL/PLAYER_INPUT_LAYERS)")
    return report
