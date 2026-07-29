"""AI configuration loader - mirrors engine/config/loader.py pattern."""
from __future__ import annotations

import functools
import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "ai_config.json"


@functools.lru_cache(maxsize=1)
def load_ai_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def clear_ai_config_cache() -> None:
    load_ai_config.cache_clear()
