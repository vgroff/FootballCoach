"""``_value_loss_breakdown`` / ``_episode_phase_fraction`` (ai/ppo/ppo_trainer.py):
the pure-numpy value-error breakdown behind ppg_value_refit's opt-in
``bc.ppg_loss_diagnostics``."""
import numpy as np
import pytest

from footballcoach.ai.ppo.ppo_trainer import _episode_phase_fraction, _value_loss_breakdown


class TestEpisodePhaseFraction:
    def test_single_track_two_episodes_and_an_unfinished_tail(self):
        # episodes of length 2 and 4, then a 3-row unfinished tail (NaN mc)
        dones = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=float)
        mc = np.array([1, 1, 2, 2, 2, 2, np.nan, np.nan, np.nan], dtype=float)
        f = _episode_phase_fraction(["t"] * 9, dones, mc)
        assert f[:2].tolist() == pytest.approx([0.5, 1.0])
        assert f[2:6].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
        assert np.isnan(f[6:]).all()

    def test_buffer_seam_after_an_unfinished_tail_starts_a_new_episode(self):
        # env A: 1 finished ep (len 2) + 2-row tail; env B: finished ep of len 2.
        # No done marker at the seam -- the NaN->finite transition is the seam.
        dones = np.array([0, 1, 0, 0, 0, 1], dtype=float)
        mc = np.array([1, 1, np.nan, np.nan, 5, 5], dtype=float)
        f = _episode_phase_fraction(["t"] * 6, dones, mc)
        assert f[0:2].tolist() == pytest.approx([0.5, 1.0])
        assert f[4:6].tolist() == pytest.approx([0.5, 1.0]), "B's episode must not inherit A's tail counter"

    def test_tracks_are_segmented_independently_even_when_interleaved(self):
        dones = np.array([0, 0, 0, 1, 0, 1], dtype=float)
        mc = np.array([1, 2, 1, 2, 1, 2], dtype=float)
        tracks = ["a", "b", "a", "b", "a", "b"]  # a: rows 0,2,4 (done at 5? no)
        # track a rows: 0,2,4 -> dones 0,0,0 -> unfinished (but mc finite here); track b rows 1,3,5 -> dones 0,1,1
        f = _episode_phase_fraction(tracks, dones, mc)
        b = f[[1, 3, 5]]
        assert b.tolist() == pytest.approx([0.5, 1.0, 1.0])


def _rows():
    """12 finished rows: 6 'win' (target +4) and 6 'loss' (target -2), two
    tracks, one unfinished NaN row."""
    mc = np.array([4, 4, 4, 4, 4, 4, -2, -2, -2, -2, -2, -2, np.nan], dtype=float)
    outcomes = ["win"] * 6 + ["loss"] * 6 + [""]
    tracks = ["trainee"] * 8 + ["opponent"] * 4 + ["trainee"]
    dones = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=float)
    other = np.array([0] * 6 + [2] * 7)  # trainee vs rules for first 6, then neural
    return mc, outcomes, tracks, dones, other


class TestValueLossBreakdown:
    def test_perfect_predictions_give_zero_error_everywhere(self):
        mc, outcomes, tracks, dones, other = _rows()
        pred = np.where(np.isfinite(mc), mc, 0.0)
        text = "\n".join(_value_loss_breakdown({"current": pred}, mc, outcomes, tracks, dones, other, ret_std=1.0))
        assert "mse=0.0000" in text and "R2=1.000" in text
        assert "worst 1% of rows" in text and "calibration" in text

    def test_group_mse_and_nan_rows_excluded(self):
        mc, outcomes, tracks, dones, other = _rows()
        pred = np.where(np.isfinite(mc), mc, 999.0)
        pred[:6] += 1.0  # only the 'win' rows are off, by exactly 1
        lines = _value_loss_breakdown({"current": pred}, mc, outcomes, tracks, dones, other, ret_std=1.0)
        text = "\n".join(lines)
        assert "12 rows with MC targets" in text or "12 rows" in text  # NaN row not counted
        win = next(l for l in lines if l.strip().startswith("win "))
        loss = next(l for l in lines if l.strip().startswith("loss "))
        assert "1.0000" in win
        assert "0.0000" in loss
        assert "999" not in text.split("worst rows")[0].replace("999.0", "")  # NaN row's pred never scored

    def test_original_column_appears_only_when_given(self):
        mc, outcomes, tracks, dones, other = _rows()
        pred = np.where(np.isfinite(mc), mc, 0.0)
        without = "\n".join(_value_loss_breakdown({"current": pred}, mc, outcomes, tracks, dones, other, 1.0))
        with_ = "\n".join(_value_loss_breakdown({"current": pred, "original": pred + 1.0}, mc, outcomes, tracks, dones, other, 1.0))
        assert "orig mse" not in without and "orig mse" in with_ and "original mse=1.0000" in with_

    def test_within_outcome_variance_is_zero_when_returns_are_a_function_of_outcome(self):
        mc, outcomes, tracks, dones, other = _rows()
        pred = np.where(np.isfinite(mc), mc, 0.0)
        text = "\n".join(_value_loss_breakdown({"current": pred}, mc, outcomes, tracks, dones, other, 1.0))
        assert "would be 0.000 (0.0% of total)" in text

    def test_error_concentration_reports_a_single_dominant_row(self):
        mc = np.array([1.0] * 200)
        pred = np.array([1.0] * 199 + [11.0])  # one row carries ALL the error
        text = "\n".join(_value_loss_breakdown(
            {"current": pred}, mc, ["win"] * 200, ["t"] * 200, np.zeros(200), None, 1.0,
        ))
        assert "worst 1% of rows = 100%" in text

    def test_track_only_table_when_ai_type_unknown(self):
        mc, outcomes, tracks, dones, _ = _rows()
        pred = np.where(np.isfinite(mc), mc, 0.0)
        text = "\n".join(_value_loss_breakdown({"current": pred}, mc, outcomes, tracks, dones, None, 1.0))
        assert "by track (trainee / opponent)" in text
        assert "trainee vs rules" not in text

    def test_ai_type_table_names_the_opponent_types(self):
        mc, outcomes, tracks, dones, other = _rows()
        pred = np.where(np.isfinite(mc), mc, 0.0)
        text = "\n".join(_value_loss_breakdown({"current": pred}, mc, outcomes, tracks, dones, other, 1.0))
        assert "trainee vs rules" in text and "trainee vs neural" in text and "opponent vs neural" in text

    def test_no_finished_rows_is_handled(self):
        lines = _value_loss_breakdown(
            {"current": np.zeros(3)}, np.full(3, np.nan), ["", "", ""], ["t"] * 3, np.zeros(3), None, 1.0,
        )
        assert len(lines) == 1 and "no rows" in lines[0]


    def test_outcome_groups_are_split_by_row_owner(self):
        """An episode's outcome label is the trainee's view, so the opponent's rows
        in a 'win' episode must NOT be pooled with the trainee's."""
        mc = np.array([4.0, 4.0, -2.0, -2.0])
        pred = mc.copy()
        lines = _value_loss_breakdown(
            {"current": pred}, mc, ["win", "win", "win", "win"], ["trainee", "trainee", "opponent", "opponent"],
            np.array([0, 1, 0, 1.0]), None, 1.0,
        )
        text = "\n".join(lines)
        assert "win | trainee" in text and "win | opponent" in text
