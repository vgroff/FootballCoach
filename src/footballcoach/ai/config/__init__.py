"""AI configuration loader - mirrors engine/config/loader.py pattern."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "ai_config.json"


def _resolve_config_path() -> Path:
    """The live ai_config.json this process loads from. Defaults to the
    source-tree copy, but respects FOOTBALLCOACH_AI_CONFIG_PATH when set --
    train.py sets this to a run's own checkpoint-dir snapshot (see
    config_path()'s docstring) right after copying it there, BEFORE any
    worker process is spawned. multiprocessing's "spawn" context (the only
    one available on Windows, and what batched_rollout_worker.py uses)
    inherits the parent's environment at Process()-creation time, so every
    worker -- the first ones AND every later PPG-interleave-triggered
    respawn (ppo_trainer.py's _maybe_run_ppg_interleave closes and respawns
    the main loop's own persistent worker pool around every PPG burst,
    confirmed to be a frequent, normal occurrence, not an edge case) --
    inherits the SAME frozen path and reads that one run's own pinned
    snapshot for its entire lifetime, immune to any later edit to the live
    source-tree file. 2026-10-18: a live mid-run edit to ball_approach_bonus/
    ball_progress_scale was confirmed (via direct log inspection -- a sharp,
    exact-ratio step-change in the logged reward components at precisely a
    PPG-interleave respawn boundary) to have propagated into an
    already-running phase1_run321 this way, contrary to the assumption that
    config is fixed for a process's lifetime. This override exists so a run
    launched after it can't be affected the same way. Unset (the default) =
    byte-identical to before this existed."""
    override = os.environ.get("FOOTBALLCOACH_AI_CONFIG_PATH")
    return Path(override) if override else _DEFAULT_CONFIG_PATH


@functools.lru_cache(maxsize=1)
def load_ai_config() -> dict:
    with open(_resolve_config_path()) as f:
        return json.load(f)


def clear_ai_config_cache() -> None:
    load_ai_config.cache_clear()


def config_path() -> Path:
    """Path to the live ai_config.json this process loads from -- e.g. so a
    training run can snapshot a copy of it into its checkpoint dir at
    startup, to know exactly what config produced that run later. Respects
    FOOTBALLCOACH_AI_CONFIG_PATH (see _resolve_config_path()) -- once a run
    has pinned itself to its own snapshot, this correctly reports THAT path,
    not the original source-tree one."""
    return _resolve_config_path()
