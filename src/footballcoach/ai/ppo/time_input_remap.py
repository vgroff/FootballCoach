"""Fold a linear change of the time-remaining input into a network's weights.

Every network reads GlobalFeatures.time_remaining_norm (global_feat[:, 1])
linearly: through the first global layers (GLOBAL_INPUT_LAYERS, both with a
bias) and, in the execution/value networks, through value_head's first
Linear, whose last input is elapsed_norm = 1 - time_remaining_norm. So if the
weights were trained on an input f_old and the input becomes f_new with
f_old = a + b * f_new, then replacing, in each of those layers,

    column w    by  b * w
    bias        by  bias + a * w            (first global layers, input f)
    bias        by  bias + (1 - a - b) * w  (value_head, input 1 - f)

makes the network compute exactly what it computed before on f_old. The fit
of (a, b) is where any approximation lives: see
scripts/time_input_remap_fit.py for the one used for the time-remaining fix
of 2026-11-16 (ai_config observation.legacy_time_remap).

Parameters shared between networks are changed once (deduplicated by
identity).
"""
from __future__ import annotations

import torch

from footballcoach.ai.obs.schema import GLOBAL_FEATURE_DIM
from footballcoach.ai.ppo.new_phase_inputs import GLOBAL_INPUT_LAYERS, _networks

TIME_REMAINING_COL = 1          # GlobalFeatures.time_remaining_norm (see schema.py)
VALUE_HEAD_FIRST_LINEAR = "value_head.1"


def _named(net) -> dict[str, torch.nn.Parameter]:
    """Parameters by name without wrapper prefixes (``_wrapped._wrapped.``)."""
    return {n.split("_wrapped.")[-1]: p for n, p in net.named_parameters()}


def remap_time_remaining_input(trainer, *, a: float, b: float) -> list[dict]:
    """In place, on every network: weights trained on f_old now see f_new with
    f_old = a + b * f_new (module docstring). Returns one report row per
    layer changed: {network, layer, input, column_norm_before}. Marks the
    trainer's weights as expecting the new input (true_time_input_ready)."""
    report: list[dict] = []
    seen: set[int] = set()
    with torch.no_grad():
        for net_name, net in _networks(trainer).items():
            params = _named(net)
            layers = [(s[: -len(".weight")], "time_remaining_norm") for s in GLOBAL_INPUT_LAYERS]
            if f"{VALUE_HEAD_FIRST_LINEAR}.weight" in params:
                layers.append((VALUE_HEAD_FIRST_LINEAR, "elapsed_norm"))
            found = 0
            for layer, inp in layers:
                w, bias = params.get(f"{layer}.weight"), params.get(f"{layer}.bias")
                if w is None:
                    continue
                if bias is None:
                    raise ValueError(f"{net_name}.{layer} has no bias to absorb the offset")
                found += 1
                if id(w) in seen:
                    continue
                seen.add(id(w))
                if inp == "time_remaining_norm":
                    if w.shape[1] != GLOBAL_FEATURE_DIM:
                        raise ValueError(f"{net_name}.{layer}: {w.shape[1]} inputs, expected {GLOBAL_FEATURE_DIM}")
                    col = TIME_REMAINING_COL
                    offset = a
                else:
                    col = w.shape[1] - 1     # elapsed_norm is value_head's last input
                    offset = 1.0 - a - b
                before = float(w[:, col].norm())
                bias += offset * w[:, col]
                w[:, col] *= b
                report.append(dict(network=net_name, layer=layer, input=inp, column_norm_before=before))
            if found == 0:
                raise ValueError(f"{net_name}: no time-reading layers found (network layout changed?)")
    trainer.true_time_input_ready = True
    return report
