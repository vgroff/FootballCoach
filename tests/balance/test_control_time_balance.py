"""Balance tests for the first-touch control-time model. These are mostly
monotonicity/ordering checks (the user did not give explicit numeric
targets for this one, unlike penalties/tackles), reported as full tables so
the shape of the model can be inspected and tuned.
"""
from __future__ import annotations

from footballcoach.engine.kicking import KickingParams, firsttime_difficulty_multiplier
from footballcoach.engine.possession import ControlTimeParams, compute_difficulty, control_time_s

RNG_REDUCTION = 0.3


def test_control_time_vs_ball_control_table(balance_recorder):
    params = ControlTimeParams.from_config()
    table = {}
    for bc in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        table[f"ball_control={bc}"] = round(
            control_time_s(params, ball_height_m=1.2, relative_speed_mps=8.0, player_speed_mps=2.0, ball_control_attr=bc),
            3,
        )
    balance_recorder.report("control_time_vs_ball_control_seconds", table)
    values = list(table.values())
    assert values == sorted(values, reverse=True)  # monotonically decreasing as ball_control rises


def test_control_time_vs_height_category_table(balance_recorder):
    params = ControlTimeParams.from_config()
    heights = {
        "ground_rolling (0.1m)": 0.1,
        "knee (0.49m)": 0.49,
        "waist (0.95m)": 0.95,
        "chest (1.4m)": 1.4,
        "head (1.8m)": 1.8,
        "above_head (2.2m)": 2.2,
    }
    table = {
        label: round(control_time_s(params, h, relative_speed_mps=3.0, player_speed_mps=1.0, ball_control_attr=0.5), 3)
        for label, h in heights.items()
    }
    balance_recorder.report("control_time_vs_height_seconds", table)
    values = list(table.values())
    assert values == sorted(values)  # monotonically increasing with height


def test_control_time_vs_relative_velocity_table(balance_recorder):
    params = ControlTimeParams.from_config()
    table = {}
    for v in (0.0, 2.0, 5.0, 10.0, 15.0):
        table[f"relative_speed={v}mps"] = round(
            control_time_s(params, ball_height_m=1.0, relative_speed_mps=v, player_speed_mps=1.0, ball_control_attr=0.5),
            3,
        )
    balance_recorder.report("control_time_vs_relative_velocity_seconds", table)
    values = list(table.values())
    assert values == sorted(values)


def test_goalkeeper_in_box_control_time_faster_for_high_balls(balance_recorder):
    params = ControlTimeParams.from_config()
    heights = {"waist (0.95m)": 0.95, "chest (1.4m)": 1.4, "head (1.8m)": 1.8}
    table = {}
    for label, h in heights.items():
        outfield = control_time_s(params, h, 5.0, 2.0, ball_control_attr=0.6, is_goalkeeper_in_box=False)
        gk = control_time_s(params, h, 5.0, 2.0, ball_control_attr=0.6, is_goalkeeper_in_box=True)
        table[label] = {"outfield_s": round(outfield, 3), "goalkeeper_s": round(gk, 3)}
    balance_recorder.report("control_time_gk_vs_outfield_seconds", table)
    for label, values in table.items():
        assert values["goalkeeper_s"] < values["outfield_s"], f"GK should be faster for {label}"


def test_firsttime_shot_accuracy_degrades_with_difficulty(balance_recorder):
    kicking_params = KickingParams.from_config()
    control_params = ControlTimeParams.from_config()

    table = {}
    for label, height, rel_v in (
        ("easy (ground, slow)", 0.1, 1.0),
        ("moderate (waist, medium)", 0.95, 5.0),
        ("hard (head, fast)", 1.8, 12.0),
    ):
        difficulty = compute_difficulty(control_params, height, rel_v, player_speed_mps=2.0)
        for precision in (0.3, 0.8):
            mult = firsttime_difficulty_multiplier(kicking_params, precision, difficulty)
            table[f"{label}_precision={precision}"] = round(mult, 3)

    balance_recorder.report("firsttime_shot_error_multiplier_table", table)
    # Harder situations should inflate error more than easy ones, at fixed precision.
    assert table["hard (head, fast)_precision=0.3"] > table["easy (ground, slow)_precision=0.3"]
    # Higher precision should reduce the inflation relative to lower precision, same difficulty.
    assert table["hard (head, fast)_precision=0.8"] < table["hard (head, fast)_precision=0.3"]
