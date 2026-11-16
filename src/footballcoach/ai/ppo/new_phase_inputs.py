"""Prepare a trained checkpoint's input layers for inputs it has never been
trained on, so switching them on changes nothing at first.

Two cases, both fixed by zeroing the first-layer weight columns that read the
input, on every network:

- Task ids. Until 2026-11-16 the rows PPO trains on never carried a task id
  (NeuralPlayerAI encoded its observation without a phase; now it always
  does), so a PPO-trained checkpoint plays at
  "all task ids 0". Its task-id columns are not neutral, though: measured on
  phase1_run321/latest (1240 phase-1 decisions), setting task_id_0 = 1 moves
  the kick logit by 3.8 on average (max 24) and the move logit by 1.3 (max
  8.8); pre-training, which encodes the env's own observation, did set it.
  Zeroing the columns of every phase that will now be switched on makes
  "task id = 1" add exactly nothing, so the checkpoint plays as it did.
- Player flags that were always 0 in training (e.g. is_goalkeeper before any
  goalkeeper role), whose columns are still at random init.

A zeroed column is not inert for learning: its gradient is the upstream
gradient times the input, and the input is 1 on the rows that set it. Only
the forward contribution starts at zero (tests/ai_scenario/
test_new_phase_inputs.py runs one optimizer step to show the column moves).
Within a single-phase run a task id is constant and acts like an extra bias;
it becomes a phase switch once phases are mixed.

(An earlier version, 2026-11-15, copied task_id_0's column into the new
phase's column instead. That reproduced "task_id_0 = 1", which is not how
the checkpoint plays, so it was replaced.)

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


def init_new_phase_inputs(trainer, *, task_phases=(), zero_player_features=()) -> list[dict]:
    """Zero, in place, the task-id columns of ``task_phases`` (1-based phase
    ids) and the named player-feature columns, on every network. Returns one
    report row per column touched: {network, layer, column, action,
    norm_before, norm_after}; ``layer`` is the name without wrapper prefixes.
    Sets ``trainer.task_id_inputs_ready`` when task phases were given."""
    g_names = [f.name for f in dataclasses.fields(GlobalFeatures)]
    p_names = [f.name for f in dataclasses.fields(PlayerFeatures)]
    for pid in task_phases:
        if not 1 <= pid <= MAX_TASK_IDS:
            raise ValueError(f"phase {pid} has no task-id slot (1..{MAX_TASK_IDS})")
    task_cols = sorted({g_names.index(f"task_id_{p - 1}") for p in task_phases})
    zero_cols = [p_names.index(n) for n in zero_player_features]
    report: list[dict] = []
    found_global = found_player = 0
    with torch.no_grad():
        for net_name, net in _networks(trainer).items():
            for layer, w in _layers(net, GLOBAL_INPUT_LAYERS):
                if w.shape[1] != GLOBAL_FEATURE_DIM:
                    raise ValueError(f"{net_name}.{layer}: {w.shape[1]} inputs, expected {GLOBAL_FEATURE_DIM}")
                found_global += 1
                for c in task_cols:
                    before = float(w[:, c].norm())
                    w[:, c] = 0.0
                    report.append(dict(network=net_name, layer=layer, column=g_names[c], action="zero",
                                       norm_before=before, norm_after=0.0))
            for layer, w in _layers(net, PLAYER_INPUT_LAYERS):
                if w.shape[1] < PLAYER_FEATURE_DIM:
                    raise ValueError(f"{net_name}.{layer}: {w.shape[1]} inputs, fewer than {PLAYER_FEATURE_DIM} player features")
                found_player += 1
                for c in zero_cols:
                    before = float(w[:, c].norm())
                    w[:, c] = 0.0
                    report.append(dict(network=net_name, layer=layer, column=p_names[c], action="zero",
                                       norm_before=before, norm_after=0.0))
    if (task_cols and found_global == 0) or (zero_cols and found_player == 0):
        raise ValueError("no input layers found (network layout changed? update GLOBAL/PLAYER_INPUT_LAYERS)")
    if task_cols:
        trainer.task_id_inputs_ready = True
    return report
