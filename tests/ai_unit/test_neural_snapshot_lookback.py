"""_resolve_snapshot_lookback: turns one ppo.eval.neural_snapshot_lookbacks entry into a (label,
target checkpoint) pair. An int entry is a fixed number of rollouts back (unchanged, original
behaviour); a float entry is a FRACTION of the run's checkpoint count so far, so e.g. 0.5 always
points at the checkpoint sitting at the run's current halfway point -- a target that moves forward
as the run progresses, unlike a fixed lookback.
"""
import pytest

from footballcoach.ai.ppo.ppo_trainer import _resolve_snapshot_lookback


def test_int_lookback_unchanged_behaviour():
    assert _resolve_snapshot_lookback(1, 120) == ("1_back", 119)
    assert _resolve_snapshot_lookback(5, 120) == ("5_back", 115)


def test_int_lookback_not_enough_history_is_skipped():
    assert _resolve_snapshot_lookback(5, 4) is None
    assert _resolve_snapshot_lookback(1, 0) is None


def test_float_lookback_targets_the_fractional_point():
    assert _resolve_snapshot_lookback(0.5, 120) == ("0.5_back", 60)
    assert _resolve_snapshot_lookback(0.25, 120) == ("0.25_back", 90)


def test_float_lookback_target_moves_forward_as_run_progresses():
    # Unlike a fixed lookback, 0.5's TARGET checkpoint keeps advancing with the run.
    _, t1 = _resolve_snapshot_lookback(0.5, 100)
    _, t2 = _resolve_snapshot_lookback(0.5, 200)
    assert t2 > t1
    assert t1 == 50 and t2 == 100


def test_float_lookback_skipped_when_not_enough_history_yet():
    # current_count=1: k_effective = round(0.5*1) = 0 -> skip (would otherwise compare to itself).
    assert _resolve_snapshot_lookback(0.5, 1) is None
    assert _resolve_snapshot_lookback(0.5, 0) is None


def test_float_lookback_label_keeps_the_configured_fraction_not_the_drifting_checkpoint():
    label_a, target_a = _resolve_snapshot_lookback(0.5, 60)
    label_b, target_b = _resolve_snapshot_lookback(0.5, 120)
    assert label_a == label_b == "0.5_back"
    assert target_a != target_b


def test_float_that_is_actually_a_whole_number_still_formats_without_trailing_zero():
    # json.load would only hand this function a genuine float for a non-integral config entry
    # (make_run305_config.py-style generators aside), but the formatter itself should still be sane.
    assert _resolve_snapshot_lookback(2.0, 120) == ("2_back", 118)
