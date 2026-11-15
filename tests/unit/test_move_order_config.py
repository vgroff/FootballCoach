"""MoveOrder's arrival_tolerance_m and overshoot_timeout_s defaults come from
orders.json["move_order"], read at construction time; an explicit constructor
argument still wins, and a missing key fails loudly instead of falling back."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import footballcoach.orders as orders_mod
from footballcoach.config import load_orders_config
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder

_ORDERS_JSON = Path(orders_mod.__file__).parent / "config" / "orders.json"


def test_defaults_match_orders_json():
    section = json.loads(_ORDERS_JSON.read_text())["move_order"]
    order = MoveOrder(target_position=Vector3(1.0, 0.0, 0.0))
    assert order.arrival_tolerance_m == section["arrival_tolerance_m"]
    assert order.overshoot_timeout_s == section["overshoot_timeout_s"]


def test_values_unchanged_by_the_move_to_config():
    # The pre-config hardcoded defaults; the move was meant to be behaviour-neutral.
    order = MoveOrder(target_position=Vector3(1.0, 0.0, 0.0))
    assert (order.arrival_tolerance_m, order.overshoot_timeout_s) == (0.3, 0.5)


def test_explicit_arguments_win():
    order = MoveOrder(target_position=Vector3(1.0, 0.0, 0.0), arrival_tolerance_m=1.0, overshoot_timeout_s=0.1)
    assert (order.arrival_tolerance_m, order.overshoot_timeout_s) == (1.0, 0.1)


def test_default_follows_the_config(monkeypatch):
    cfg = copy.deepcopy(load_orders_config())
    cfg["move_order"]["arrival_tolerance_m"] = 0.75
    cfg["move_order"]["overshoot_timeout_s"] = 1.25
    monkeypatch.setattr(orders_mod, "load_orders_config", lambda: cfg)
    order = MoveOrder(target_position=Vector3(1.0, 0.0, 0.0))
    assert (order.arrival_tolerance_m, order.overshoot_timeout_s) == (0.75, 1.25)


@pytest.mark.parametrize("key", ["arrival_tolerance_m", "overshoot_timeout_s"])
def test_missing_key_raises(monkeypatch, key):
    cfg = copy.deepcopy(load_orders_config())
    del cfg["move_order"][key]
    monkeypatch.setattr(orders_mod, "load_orders_config", lambda: cfg)
    with pytest.raises(KeyError, match=key):
        MoveOrder(target_position=Vector3(1.0, 0.0, 0.0))
