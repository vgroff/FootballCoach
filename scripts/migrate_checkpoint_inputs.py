"""Migrate a checkpoint trained before 2026-11-16 to the current input
conventions, writing a NEW file (never overwrites the source):

- the true time-remaining input: fold ai_config observation.legacy_time_remap
  into the weights (ai/ppo/time_input_remap.py). Training refuses
  unmigrated weights.
- task ids in the training rows (always set since 2026-11-16): zero the
  task-id input columns of the given phases, and is_goalkeeper when a phase
  has a goalkeeper role (ai/ppo/new_phase_inputs.py), so setting them changes
  nothing at first. Training refuses unmigrated weights.

The output records true_time_input_ready and task_id_inputs_ready. Train it
WITHOUT --remap-time-input / --init-new-phase-inputs (those are for
unmigrated checkpoints; re-applying them after training would undo learning).

Verification (printed), on real states from seeded episodes of every given
phase, with the original checkpoint fed the OLD conventions (old clock, task
ids 0) and the migrated one the NEW (true clock, task id set):
- task ids: the migrated weights give bit-identical outputs with the task id
  set and unset (exact by construction);
- time: scripts/time_input_remap_fit.py's counterfactual comparison, each
  state placed across the clock, by true time left (approximate by nature).
Phase-2 states also differ by the zeroed goalkeeper flag, by design (its
weights were random init).

Usage (repo root):
    python scripts/migrate_checkpoint_inputs.py SRC.pt DST.pt [--phases 1,2] [--episodes 30] [--states 300]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_input_remap_fit import BINS, HEADS, collect, outputs  # noqa: E402

from footballcoach.ai.config import load_ai_config  # noqa: E402
from footballcoach.ai.curriculum.phases import PHASES_BY_ID  # noqa: E402
from footballcoach.ai.obs.schema import GlobalFeatures  # noqa: E402
from footballcoach.ai.ppo.new_phase_inputs import init_new_phase_inputs  # noqa: E402
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer  # noqa: E402
from footballcoach.ai.ppo.time_input_remap import remap_time_remaining_input  # noqa: E402

G_NAMES = [f.name for f in dataclasses.fields(GlobalFeatures)]


def _task_ids(obs, phase_id):
    """obs with every task id 0, and with phase_id's set."""
    off = {k: v.clone() for k, v in obs.items()}
    for i, n in enumerate(G_NAMES):
        if n.startswith("task_id_"):
            off["global_feat"][i] = 0.0
    on = {k: v.clone() for k, v in off.items()}
    on["global_feat"][G_NAMES.index(f"task_id_{phase_id - 1}")] = 1.0
    return off, on


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--phases", default="1,2")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--states", type=int, default=300)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        sys.exit(f"{dst} exists; refusing to overwrite")
    phases = [int(p) for p in args.phases.split(",")]

    orig = PPOTrainer.from_config(device=torch.device("cpu"))
    orig.load_checkpoint(src)
    if orig.true_time_input_ready or orig.task_id_inputs_ready:
        sys.exit(f"{src} is already marked migrated (time {orig.true_time_input_ready}, "
                 f"task ids {orig.task_id_inputs_ready})")
    migrated = copy.deepcopy(orig)
    remap = load_ai_config()["observation"]["legacy_time_remap"]
    remap_time_remaining_input(migrated, a=float(remap["a"]), b=float(remap["b"]))
    gk = ["is_goalkeeper"] if any("goalkeeper" in PHASES_BY_ID[p].roles.values() for p in phases) else []
    init_new_phase_inputs(migrated, task_phases=phases, zero_player_features=gk)
    print(f"time remap f_old = {remap['a']} + {remap['b']} * f_new; zeroed task ids of phases {phases}"
          + (" and is_goalkeeper" if gk else ""))

    rows = collect(orig, phases, args.episodes)
    states = random.Random(0).sample(rows, min(args.states, len(rows)))
    print(f"verification: {len(rows)} rows, {len(states)} states "
          + " ".join(f"phase{p}={sum(s[0] == p for s in states)}" for p in phases))

    # Task ids: exact.
    worst = 0.0
    for pid, obs, _fo, fn, *_ in states:
        off, on = _task_ids(obs, pid)
        a, b = outputs(migrated, off, fn), outputs(migrated, on, fn)
        worst = max(worst, max(abs(a[k] - b[k]) for k in a))
    print(f"task ids: max output change from setting the task id on the migrated weights = {worst:.1e}")
    if worst != 0.0:
        sys.exit("task-id zeroing is not exact; not writing")

    # Time: the original (old clock, task ids 0) vs the migrated (true clock,
    # task id set), each state across the clock, per phase.
    for p in phases:
        ps = [s for s in states if s[0] == p]
        if not ps:
            continue
        orig_in = [(pid, _task_ids(obs, pid)[0], fo, fn, h, dt) for pid, obs, fo, fn, h, dt in ps]
        new_in = [(pid, _task_ids(obs, pid)[1], fo, fn, h, dt) for pid, obs, fo, fn, h, dt in ps]
        res = _paired_counterfactual(orig, migrated, orig_in, new_in)
        print(f"-- phase {p}: migrated (new inputs) vs original (old inputs), probability changes mean/max")
        for label, _lo, _hi in BINS:
            r = res[label]
            if not r["value"]:
                continue
            parts = [f"{k[:-6]} {statistics.mean(r[k]):.4f}/{max(r[k]):.3f}" for k in HEADS]
            parts.append(f"value {statistics.mean(r['value']):.3f}/{max(r['value']):.3f}")
            print(f"   {label:>6}  " + "  ".join(parts))

    dst.parent.mkdir(parents=True, exist_ok=True)
    migrated._save_checkpoint_to(dst)
    check = PPOTrainer.from_config(device=torch.device("cpu"))
    check.load_checkpoint(dst)
    assert check.true_time_input_ready and check.task_id_inputs_ready
    print(f"wrote {dst} (true_time_input_ready, task_id_inputs_ready); {src} untouched")


def _paired_counterfactual(orig, migrated, orig_states, new_states):
    import math
    from time_input_remap_fit import CLOCK_POINTS_S, f_new_at, f_old_at
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    res = {b[0]: {k: [] for k in (*HEADS, "value")} for b in BINS}
    for (_p, obs_o, _a, _b, horizon, dt), (_q, obs_n, *_rest) in zip(orig_states, new_states):
        for t_left in CLOCK_POINTS_S:
            if t_left >= horizon:
                continue
            ticks = round((horizon - t_left) / dt)
            base = outputs(orig, obs_o, f_old_at(horizon, ticks))
            new = outputs(migrated, obs_n, f_new_at(horizon, ticks, dt))
            label = next(b[0] for b in BINS if b[1] <= horizon - ticks * dt < b[2])
            for k in HEADS:
                res[label][k].append(abs(sig(new[k]) - sig(base[k])))
            res[label]["value"].append(abs(new["value"] - base["value"]))
    return res


if __name__ == "__main__":
    main()
