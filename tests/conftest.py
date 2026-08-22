"""Shared pytest fixtures: default entity factories and the balance-test
result reporting mechanism.
"""
from __future__ import annotations

import os

# Must happen before numpy/torch are imported anywhere in this process (BLAS
# thread pools are sized at first-use) -- pytest loads this conftest before
# any test module, so this is the earliest point available. Each pytest-xdist
# worker is its own process; without this, every worker's torch/numpy calls
# each try to use ALL cores for their own internal op-level parallelism, so
# (worker count) x (threads per worker) massively oversubscribes the machine
# -- the actual source of tests "hammering the CPU" far beyond what the
# xdist worker count alone would suggest. Capping to 1 thread per worker
# process trades a bit of single-op speed for eliminating that cross-worker
# contention, which in practice is a net wall-clock win too, not just a
# gentler one.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import random
from pathlib import Path

import pytest
import torch

from footballcoach.entities import Ball, Pitch, Player, PlayerAttributes, Team
from footballcoach.mathutils import Vector3

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already set / parallel work already started in this process -- not worth failing the whole test session over

RESULTS_DIR = Path(__file__).parent / "balance" / "results"


@pytest.fixture
def pitch() -> Pitch:
    return Pitch.standard()


@pytest.fixture
def seeded_rng() -> random.Random:
    return random.Random(12345)


def make_player(
    player_id: str = "p1",
    team: Team = Team.LEFT,
    attr_value: float = 0.5,
    position: Vector3 | None = None,
    is_goalkeeper: bool = False,
    **attr_overrides: float,
) -> Player:
    """Creates a player with all attributes set to `attr_value`, with any
    individual overrides applied (e.g. make_player(tackling=0.8))."""
    base = {
        "top_speed": attr_value,
        "acceleration": attr_value,
        "stamina": attr_value,
        "kick_precision": attr_value,
        "kick_power": attr_value,
        "dribbling": attr_value,
        "ball_control": attr_value,
        "tackling": attr_value,
    }
    base.update(attr_overrides)
    attrs = PlayerAttributes(**base)
    return Player.create(player_id, team, attrs, position=position, is_goalkeeper=is_goalkeeper)


class BalanceResultRecorder:
    """Collects (name -> stats dict) pairs during a balance test session and
    writes them to a JSON report at the end, as well as printing a readable
    table to stdout. Balance tests should report full statistics via this
    recorder rather than only asserting pass/fail, per project requirements.
    """

    def __init__(self) -> None:
        self.results: dict[str, dict] = {}

    def report(self, name: str, stats: dict) -> None:
        self.results[name] = stats
        print(f"\n=== Balance result: {name} ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def save(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "latest_results.json"
        existing = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing.update(self.results)
        out_path.write_text(json.dumps(existing, indent=2, default=str))


@pytest.fixture(scope="session")
def balance_recorder():
    recorder = BalanceResultRecorder()
    yield recorder
    recorder.save()
