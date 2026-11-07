"""pin_ai_config_to_run_dir(): a run must keep reading its own config snapshot,
even in workers spawned after the live file was edited (the 2026-10-18 leak)."""
from __future__ import annotations

import ast
import functools
import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

import footballcoach.ai.config as ai_config
from footballcoach.ai.config import clear_ai_config_cache, config_path, pin_ai_config_to_run_dir

ENV_VAR = "FOOTBALLCOACH_AI_CONFIG_PATH"


def _write(path: Path, marker: str) -> None:
    path.write_text(json.dumps({"marker": marker}))


def _read_marker_in_child(queue) -> None:
    queue.put(ai_config.load_ai_config()["marker"])


def _marker_seen_by_a_freshly_spawned_process() -> str:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_read_marker_in_child, args=(queue,))
    proc.start()
    try:
        return queue.get(timeout=60)
    finally:
        proc.join(timeout=60)


@pytest.fixture
def live_config(tmp_path, monkeypatch):
    """A stand-in for the live source-tree file, with a private config cache:
    tests/ai_scenario/conftest.py mutates the real cached dict for the whole
    session, so clearing that one here would silently undo its overrides."""
    live = tmp_path / "live.json"
    _write(live, "original")
    monkeypatch.setenv(ENV_VAR, str(live))
    monkeypatch.setattr(
        ai_config, "load_ai_config", functools.lru_cache(maxsize=1)(ai_config.load_ai_config.__wrapped__),
    )
    return live


def test_default_path_is_the_source_tree_file(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert config_path() == Path(ai_config.__file__).parent / "ai_config.json"


def test_snapshot_is_a_copy_of_the_live_config(live_config, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = pin_ai_config_to_run_dir(run_dir)
    assert snapshot == (run_dir / "ai_config.json").resolve()
    assert json.loads(snapshot.read_text()) == {"marker": "original"}


def test_this_process_reads_the_snapshot_even_with_a_stale_cache(live_config, tmp_path):
    assert ai_config.load_ai_config()["marker"] == "original"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pin_ai_config_to_run_dir(run_dir)
    _write(live_config, "edited")
    (run_dir / "ai_config.json").write_text(json.dumps({"marker": "snapshot"}))
    clear_ai_config_cache()
    assert ai_config.load_ai_config()["marker"] == "snapshot"


def test_pinning_drops_a_value_cached_before_the_pin(live_config, tmp_path):
    assert ai_config.load_ai_config()["marker"] == "original"
    _write(live_config, "edited after first read")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pin_ai_config_to_run_dir(run_dir)
    assert ai_config.load_ai_config()["marker"] == "edited after first read"


def test_spawned_worker_ignores_a_live_edit_made_after_the_pin(live_config, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pin_ai_config_to_run_dir(run_dir)
    _write(live_config, "edited mid-run")
    assert _marker_seen_by_a_freshly_spawned_process() == "original"


def test_an_unpinned_spawned_worker_does_see_a_live_edit(live_config):
    """Guards the test above: it must be the pin, not the harness, that blocks the edit."""
    _write(live_config, "edited mid-run")
    assert _marker_seen_by_a_freshly_spawned_process() == "edited mid-run"


def test_pinning_twice_to_the_same_dir_is_safe(live_config, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = pin_ai_config_to_run_dir(run_dir)
    assert pin_ai_config_to_run_dir(run_dir) == first
    assert ai_config.load_ai_config()["marker"] == "original"


def test_relative_run_dir_is_stored_as_an_absolute_path(live_config, tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    monkeypatch.chdir(tmp_path)
    pin_ai_config_to_run_dir(Path("run"))
    assert Path(os.environ[ENV_VAR]).is_absolute()


def test_train_pins_before_importing_anything_that_spawns_workers():
    """multiprocessing children inherit the environment at creation, so order matters."""
    import footballcoach.ai.scripts.train as train
    source = Path(train.__file__).read_text(encoding="utf-8")
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    pin_line = next(
        n.lineno for n in ast.walk(main)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "pin_ai_config_to_run_dir"
    )
    trainer_import_line = next(
        n.lineno for n in ast.walk(main)
        if isinstance(n, ast.ImportFrom) and n.module == "footballcoach.ai.ppo.ppo_trainer"
    )
    assert pin_line < trainer_import_line
