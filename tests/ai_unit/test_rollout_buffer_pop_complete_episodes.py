"""Coverage for ``RolloutBuffer.pop_complete_episodes`` (ai/ppo/rollout_buffer.py)
-- the primitive behind ``BatchedEnvGroup``'s whole-episode chunk flushes.

Synthetic buffers only (no env/network): each row is ``(track_id, reward,
done, value)``. Rows for a tick are appended trainee-first, then secondary,
exactly like the real collection loops -- and every row of a TERMINAL tick
carries ``done=1``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer


def _make_buffer(rows: list[tuple]) -> RolloutBuffer:
    buf = RolloutBuffer()
    for i, (track, reward, done, value) in enumerate(rows):
        buf.add(
            obs={"x": np.full(2, i, dtype=np.float32)},
            action={"a": np.full(1, i, dtype=np.float32)},
            log_prob=-0.1 * i, value=value, reward=reward, done=done,
            reward_comps={"r": reward}, step_outcome="win" if done else "", track_id=track,
        )
    return buf


def _all_field_lengths(buf: RolloutBuffer) -> set[int]:
    return {len(getattr(buf, f.name)) for f in dataclasses.fields(buf)}


# Two complete single-track episodes (3 rows, 2 rows) + a 2-row unfinished tail.
_ROWS = [
    ("trainee", 1.0, 0.0, 0.5), ("trainee", 2.0, 0.0, 0.4), ("trainee", 3.0, 1.0, 0.3),
    ("trainee", 4.0, 0.0, 0.2), ("trainee", 5.0, 1.0, 0.1),
    ("trainee", 6.0, 0.0, 0.6), ("trainee", 7.0, 0.0, 0.7),
]


class TestPopCompleteEpisodes:
    def test_no_completed_episode_returns_none_and_leaves_buffer_untouched(self):
        buf = _make_buffer([("trainee", 1.0, 0.0, 0.0), ("trainee", 2.0, 0.0, 0.0)])
        assert buf.pop_complete_episodes() is None
        assert len(buf) == 2

    def test_empty_buffer_returns_none(self):
        assert RolloutBuffer().pop_complete_episodes() is None

    def test_splits_at_last_done_and_keeps_the_tail(self):
        buf = _make_buffer(_ROWS)
        head = buf.pop_complete_episodes()
        assert head is not None
        assert len(head) == 5 and len(buf) == 2
        assert head.dones[-1] == 1.0
        assert head.rewards == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert buf.rewards == [6.0, 7.0]
        assert buf.dones == [0.0, 0.0]

    def test_every_per_row_field_is_split_consistently(self):
        buf = _make_buffer(_ROWS)
        head = buf.pop_complete_episodes()
        assert _all_field_lengths(head) == {5}
        assert _all_field_lengths(buf) == {2}
        # row identity preserved end to end (obs carries the original row index)
        assert [int(o["x"][0]) for o in head.obs] == [0, 1, 2, 3, 4]
        assert [int(o["x"][0]) for o in buf.obs] == [5, 6]
        assert head.step_outcomes == ["", "", "win", "", "win"]

    def test_tail_finishing_later_is_handed_over_intact_by_a_second_pop(self):
        buf = _make_buffer(_ROWS)
        first = buf.pop_complete_episodes()
        assert len(first) == 5
        # the tail's episode now completes
        buf.add(
            obs={"x": np.full(2, 7, dtype=np.float32)}, action={"a": np.zeros(1, dtype=np.float32)},
            log_prob=0.0, value=0.0, reward=8.0, done=1.0,
        )
        second = buf.pop_complete_episodes()
        assert second.rewards == [6.0, 7.0, 8.0]
        assert len(buf) == 0

    def test_popping_twice_in_a_row_gives_none_the_second_time(self):
        buf = _make_buffer(_ROWS)
        buf.pop_complete_episodes()
        assert buf.pop_complete_episodes() is None  # tail has no done

    def test_returned_buffer_is_independent_of_later_mutation_of_the_source(self):
        buf = _make_buffer(_ROWS)
        head = buf.pop_complete_episodes()
        buf.clear()
        assert len(head) == 5 and head.rewards == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_multi_track_terminal_tick_stays_whole(self):
        # tick 0: trainee+opponent (mid), tick 1: trainee+opponent TERMINAL (both done=1),
        # tick 2: trainee+opponent (new episode, unfinished).
        rows = [
            ("trainee", 1.0, 0.0, 0.0), ("opponent", -1.0, 0.0, 0.0),
            ("trainee", 2.0, 1.0, 0.0), ("opponent", -2.0, 1.0, 0.0),
            ("trainee", 3.0, 0.0, 0.0), ("opponent", -3.0, 0.0, 0.0),
        ]
        buf = _make_buffer(rows)
        head = buf.pop_complete_episodes()
        assert head.track_ids == ["trainee", "opponent", "trainee", "opponent"]
        assert head.dones == [0.0, 0.0, 1.0, 1.0], "the secondary row of the terminal tick must travel with it"
        assert buf.track_ids == ["trainee", "opponent"], "new episode's rows stay together in the tail"

    def test_gae_on_popped_episodes_equals_gae_on_the_full_buffer_rows(self):
        gamma, lam = 0.99, 0.95
        full = _make_buffer(_ROWS)
        full_adv, full_ret = full.compute_gae(gamma, lam, 0.25)

        buf = _make_buffer(_ROWS)
        head = buf.pop_complete_episodes()
        # whole episodes end on done=1, so the bootstrap value is irrelevant
        head_adv, head_ret = head.compute_gae(gamma, lam, 0.0)
        assert head_adv == pytest.approx(full_adv[:5])
        assert head_ret == pytest.approx(full_ret[:5])
        head_adv_other_bootstrap, _ = head.compute_gae(gamma, lam, 123.0)
        assert head_adv_other_bootstrap == pytest.approx(head_adv)

    def test_mc_returns_on_popped_episodes_are_exact_without_truncation(self):
        gamma = 0.9
        buf = _make_buffer(_ROWS)
        head = buf.pop_complete_episodes()
        assert head.truncate_to_last_episode_end() == 0, "nothing left to truncate"
        rets = head.compute_mc_returns(gamma)
        # episode 1 = rewards [1,2,3]; episode 2 = [4,5]
        assert rets[2] == pytest.approx(3.0)
        assert rets[1] == pytest.approx(2.0 + gamma * 3.0)
        assert rets[0] == pytest.approx(1.0 + gamma * (2.0 + gamma * 3.0))
        assert rets[4] == pytest.approx(5.0)
        assert rets[3] == pytest.approx(4.0 + gamma * 5.0)
