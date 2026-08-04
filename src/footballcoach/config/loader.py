"""Loads and caches the JSON configuration files that drive engine balance.

All tunable constants live in physics.json and attributes.json in this
package. Nothing in the engine should hardcode a balance constant directly -
it should be read from here, so users can tune the game by editing JSON
instead of code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).parent


def _load_json(name: str) -> dict[str, Any]:
    with open(_CONFIG_DIR / name, "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_physics_config() -> dict[str, Any]:
    """Returns the parsed contents of physics.json (cached)."""
    return _load_json("physics.json")


@lru_cache(maxsize=1)
def load_attributes_config() -> dict[str, Any]:
    """Returns the parsed contents of attributes.json (cached)."""
    return _load_json("attributes.json")


@lru_cache(maxsize=1)
def load_graphics_config() -> dict[str, Any]:
    """Returns the parsed contents of graphics.json (cached)."""
    return _load_json("graphics.json")


@lru_cache(maxsize=1)
def load_gameplay_config() -> dict[str, Any]:
    """Returns the parsed contents of gameplay.json (cached)."""
    return _load_json("gameplay.json")


@lru_cache(maxsize=1)
def load_orders_config() -> dict[str, Any]:
    """Returns the parsed contents of orders.json (cached)."""
    return _load_json("orders.json")


@lru_cache(maxsize=1)
def load_scenarios_config() -> dict[str, Any]:
    """Returns the parsed contents of scenarios.json (cached)."""
    return _load_json("scenarios.json")


def clear_config_cache() -> None:
    """Clears cached config, forcing a re-read from disk on next access.

    Useful in tests that want to monkeypatch config files.
    """
    load_physics_config.cache_clear()
    load_attributes_config.cache_clear()
    load_graphics_config.cache_clear()
    load_gameplay_config.cache_clear()
    load_orders_config.cache_clear()
    load_scenarios_config.cache_clear()


def require_section(config: dict[str, Any], section: str, file_name: str = "physics.json") -> dict[str, Any]:
    """Returns ``config[section]``, raising a ``KeyError`` that names both
    the config file and the missing section (rather than a bare
    ``KeyError('section')``) if it's absent.

    Every ``*Params.from_config()`` in the engine reads a section of
    ``physics.json`` this way instead of indexing the dict directly, so a
    missing/renamed section fails loudly with useful context rather than a
    generic exception or (worse) a silently-defaulted empty dict.
    """
    try:
        return config[section]
    except KeyError:
        raise KeyError(f"Missing section '{section}' in {file_name}") from None


@dataclass(frozen=True)
class PitchConfig:
    length_m: float
    width_m: float
    goal_width_m: float
    goal_height_m: float
    goal_depth_m: float
    box_length_m: float
    box_width_m: float
    six_yard_length_m: float
    six_yard_width_m: float
    penalty_spot_distance_m: float
    centre_circle_radius_m: float

    @staticmethod
    def from_config() -> "PitchConfig":
        d = require_section(load_physics_config(), "pitch")
        return PitchConfig(
            length_m=d["length_m"],
            width_m=d["width_m"],
            goal_width_m=d["goal_width_m"],
            goal_height_m=d["goal_height_m"],
            goal_depth_m=d.get("goal_depth_m", 2.45),
            box_length_m=d["box_length_m"],
            box_width_m=d["box_width_m"],
            six_yard_length_m=d["six_yard_length_m"],
            six_yard_width_m=d["six_yard_width_m"],
            penalty_spot_distance_m=d["penalty_spot_distance_m"],
            centre_circle_radius_m=d["centre_circle_radius_m"],
        )
