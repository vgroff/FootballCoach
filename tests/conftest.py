"""Shared pytest fixtures: default entity factories and the balance-test
result reporting mechanism.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from footballcoach.entities import Ball, Pitch, Player, PlayerAttributes, Team
from footballcoach.mathutils import Vector3

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
