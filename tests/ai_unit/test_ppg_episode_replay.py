"""Coverage for ``PPOTrainer._ppg_update_episode_replay`` (ai/ppo/ppo_trainer.py)
-- ``ppg_value_refit``'s OWN rollout-to-rollout episode-seed-replay loop
(``ppg_episode_replay_enabled``), modeled on but independent of the main PPO
loop's ``episode_replay_enabled``/``_update_episode_replay`` (see
test_episode_replay.py for that one). See ai/knowledge.md "PPG's own
episode-seed replay" for the full design writeup.

``_ppg_update_episode_replay`` is call-scoped (pending-state passed in and
returned, not read/written on ``self.``) specifically so it's testable like
this: pure function of its two arguments plus
``self._episode_replay_top_fraction``, no environment, no rollout, no
multiprocessing. The actual per-episode value-MSE EXTRACTION this method
consumes (``_collect_value_pretrain_rollout``'s new
``collect_episode_replay_stats``/``episode_seed_mse``) only activates on the
batched worker path, which -- like every other real-``ppg_value_refit``-call
test in this repo (see test_ppg_interleave.py's own docstring) -- needs a
live env + real rollout collection and is exercised via manual smoke test
(``--ppg-refit-only``), not here.
"""
from __future__ import annotations

import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

_PHASE = PHASES_BY_ID[1]


def _trainer(top_fraction: float = 0.25) -> PPOTrainer:
    t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=False)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    t._episode_replay_top_fraction = top_fraction
    return t


class TestConfigDefault:
    def test_code_level_default_is_off(self):
        # Opt-in convention: __init__ falls back to False when the key is absent from
        # config. NOT asserted against the live ai_config.json value (may be true there
        # while this feature is being live-tested) -- same reasoning test_ppg_interleave.py's
        # TestConfigDefaults gives.
        t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False)
        assert t._ppg_episode_replay_enabled is False


class TestFirstCycleNoPendingBefore:
    def test_empty_seed_mse_returns_empty_pending_and_none_seeds(self):
        t = _trainer()
        pending, next_seeds = t._ppg_update_episode_replay([], {})
        assert pending == {}
        assert next_seeds is None

    def test_selects_top_fraction_by_mse_descending_when_no_pending(self):
        t = _trainer(top_fraction=0.25)
        # 8 episodes -> ceil(0.25 * 8) = 2 selected, the two WORST (highest MSE).
        seed_mse = [(1, 0.1), (2, 0.9), (3, 0.3), (4, 0.05), (5, 0.7), (6, 0.2), (7, 0.4), (8, 0.6)]
        pending, next_seeds = t._ppg_update_episode_replay(seed_mse, {})
        assert pending == {2: 0.9, 5: 0.7}
        assert sorted(next_seeds) == [2, 5]

    def test_at_least_one_selected_even_with_tiny_fraction(self):
        t = _trainer(top_fraction=0.01)
        seed_mse = [(10, 0.5), (11, 0.2), (12, 0.9)]
        pending, next_seeds = t._ppg_update_episode_replay(seed_mse, {})
        assert len(pending) == 1
        assert next_seeds == [12]  # the single worst-predicted episode


class TestReportOnMatchedSeeds:
    def test_matched_seeds_report_before_after_and_clear_pending(self):
        t = _trainer(top_fraction=0.5)
        pending_before = {100: 0.8, 200: 0.6}
        this_cycle = [(100, 0.3), (200, 0.1), (300, 0.9)]  # both queued seeds improved

        pending_after, next_seeds = t._ppg_update_episode_replay(this_cycle, pending_before)

        # Report path consumed pending_before entirely before reselecting fresh.
        # Reselection (top 50% of 3 -> ceil(1.5) = 2) picks the two worst of THIS cycle.
        assert pending_after == {300: 0.9, 100: 0.3}
        assert sorted(next_seeds) == [100, 300]

    def test_unmatched_seeds_still_reselect_from_this_cycle(self):
        t = _trainer(top_fraction=1.0)
        pending_before = {999: 0.5}  # seed 999 never shows up this cycle (no reset reached it)
        this_cycle = [(1, 0.2), (2, 0.4)]

        pending_after, next_seeds = t._ppg_update_episode_replay(this_cycle, pending_before)

        assert pending_after == {2: 0.4, 1: 0.2}
        assert sorted(next_seeds) == [1, 2]

    def test_pending_before_but_no_episodes_this_cycle_clears_and_returns_none(self):
        t = _trainer()
        pending_after, next_seeds = t._ppg_update_episode_replay([], {42: 0.5})
        assert pending_after == {}
        assert next_seeds is None


class TestRanksByValueMseNotReward:
    def test_ranking_key_is_the_mse_value_not_seed_order(self):
        """Regression guard against accidentally sorting by seed or by
        insertion order instead of the actual MSE value (index 1 of each
        tuple) -- deliberately out-of-order input."""
        t = _trainer(top_fraction=0.34)  # ceil(0.34*3)=2
        seed_mse = [(3, 0.01), (1, 5.0), (2, 2.5)]
        _pending, next_seeds = t._ppg_update_episode_replay(seed_mse, {})
        assert sorted(next_seeds) == [1, 2]  # the two highest-MSE seeds, not seeds 1 and 3
