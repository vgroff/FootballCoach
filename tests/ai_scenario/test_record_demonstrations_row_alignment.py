"""Regression test: row/reward misattribution in record_demonstrations.py's
``record_episodes()`` when an on_kick/on_tackle callback fires SYNCHRONOUSLY
inside ``env.step()``.

Real bug this reproduces
-------------------------
The main recording loop, for each timed sample, used to do (paraphrased)::

    n_before = len(rewards)
    _recorded_ids = _record_now(reward=0.0, done=False)   # appends 2 rows
    n_appended = len(rewards) - n_before                   # = 2
    _obs, _reward, done, last_info = env.step()             # <- on_kick/on_tackle
                                                              #    callbacks fire
                                                              #    HERE, calling
                                                              #    _record_now(player_id=pid)
                                                              #    and appending
                                                              #    MORE rows
    for _offset, _pid in enumerate(_recorded_ids):
        _i = n_appended - _offset
        rewards[-_i] = ...    # STALE count + negative indexing

``n_appended`` was captured *before* ``env.step()`` and never refreshed, so
whenever a kick/tackle callback fired in the same decision interval as the
timed sample (confirmed common -- tackle_attempt rate ~0.7% of rows, and an
immobile player can passively receive the ball via the engine's generic
proximity-based pickup, `Match._update_loose_ball_pickup`, which has no AI
check at all -- see scenario_env.py's ``_can_score_box_terminal`` docstring),
the negative-index backfill silently wrote the wrong reward/reward_components
onto the wrong rows. This was invisible for the (usual) all-zero case and
only became glaringly visible when it shifted a real terminal reward --
confirmed in real recorded data: a trainee's genuine box-possession-terminal
reward ended up on the opponent's row (and the opponent's -- nonexistent --
loss penalty ended up on the trainee's row) whenever a tackle happened to
fire on the exact same tick as the win.

The fix: capture each timed-sample row's ABSOLUTE index right after
``_record_now()`` (before ``env.step()`` can insert anything else), and
backfill using those exact indices instead of a relative offset.
"""
from __future__ import annotations

import numpy as np

import footballcoach.ai.env.scenario_env as scenario_env_mod
from footballcoach.ai.curriculum.envs import bc_label_fn_for_phase, build_env
from footballcoach.ai.curriculum.phases import CurriculumPhase
from footballcoach.ai.scripts.record_demonstrations import record_episodes

_DISTINCTIVE_REWARD = 999.0


def test_trainee_reward_survives_mid_step_callback_row_insertion(monkeypatch):
    """Force an on_kick callback to fire synchronously inside env.step() (as
    on_kick/on_tackle genuinely do in real play), on a tick where the
    trainee's real reward is a distinctive, unmistakable value. The recorded
    dataset must attribute that value to a trainee row (is_trainee==1) --
    never to the opponent's row, and never lose it (leaving 0.0 behind)."""
    phase = CurriculumPhase(
        name="p1", phase_id=1, scenario_key="phase1_1v1", env_kwargs={"max_episode_s": 10.0}
    )
    env = build_env(phase)
    label_fn = bc_label_fn_for_phase(1)
    env.always_compute_secondary_reward = True

    orig_step = scenario_env_mod.ScenarioEnv.step
    call_count = {"n": 0}
    # Fire on a mid-episode tick (well after callbacks are wired by
    # record_episodes()'s per-episode setup, well before the episode's
    # natural end) so the injected row lands squarely inside a normal
    # timed-sample backfill, exactly like the real bug.
    target_call = 5

    def _patched_step(self):
        call_count["n"] += 1
        obs, reward, done, info = orig_step(self)
        if call_count["n"] == target_call:
            match = self._loop.match
            opp = match.player_by_id("opponent")
            if opp.on_kick is not None:
                # Simulates a kick/tackle callback firing synchronously
                # inside step() -- inserts one extra row via _record_now()
                # mid-call, exactly like the real on_kick/on_tackle wiring.
                opp.on_kick(opp)
            reward = _DISTINCTIVE_REWARD
            # This test forces an episode end on an artificial tick rather
            # than letting real ScenarioLoop outcome detection decide it, so
            # env.last_terminal_match (normally set by ScenarioEnv.step()
            # itself -- see its own docstring) was never populated for this
            # synthetic ending. self._loop.match is still the real, un-
            # rebuilt current state at this exact tick (nothing in the real
            # engine decided to end/rebuild anything), so it's the correct
            # stand-in -- matches ScenarioEnv.step()'s own fallback for its
            # non-ScenarioLoop-detected done paths (box_terminal/timeout).
            self.last_terminal_match = self._loop.match
            # End the episode on this exact tick. _reward_by_pid also
            # accrues into _pending_reward["trainee"] every tick (by
            # design -- see test_pending_reward_does_not_leak_across_
            # episodes), so if the episode kept running, a LATER natural
            # on_kick/on_tackle from the real (rules-driven) trainee could
            # legitimately pop _pending_reward and re-surface this same
            # 999.0 on a second row -- correct system behavior, but it
            # would make THIS test flaky (depends on whether the rules AI
            # happens to kick/tackle again before the episode ends). Ending
            # here removes that variable entirely.
            done = True
        return obs, reward, done, info

    monkeypatch.setattr(scenario_env_mod.ScenarioEnv, "step", _patched_step)

    result = record_episodes(
        env, label_fn, n_episodes=1, scenario_key="phase1_1v1", phase_id=1,
        sample_interval_s=0.5,
        opponent_rules_prob=0.0, opponent_immobile_prob=1.0,
    )

    rewards = result["rewards"]
    is_trainee = result["is_trainee"]

    assert call_count["n"] >= target_call, "test setup didn't reach the target step() call"

    hit_rows = np.nonzero(rewards == _DISTINCTIVE_REWARD)[0]
    assert len(hit_rows) == 1, (
        f"expected the distinctive reward to appear on exactly one row, found {len(hit_rows)} "
        f"(0 means it was lost/overwritten by the misindexed backfill; >1 means it was duplicated)"
    )
    assert is_trainee[hit_rows[0]] > 0.5, (
        "the trainee's own real reward landed on a non-trainee row -- this is the exact "
        "row-misattribution bug: a mid-step() callback (on_kick/on_tackle) inserted an extra "
        "row and the backfill used a stale row-count offset instead of each row's own "
        "absolute index."
    )


_DISTINCTIVE_REWARD_2 = 777.0


def test_kick_callback_row_does_not_duplicate_prior_reward(monkeypatch):
    """Regression test for the double-counting bug: a kick/tackle callback's
    own row must always carry reward=0.0, never a copy of a reward that was
    already attributed to an EARLIER timed-sample row.

    The removed ``_pending_reward`` mechanism accrued every timed sample's
    reward into a per-player dict, meant to be "claimed" by whichever
    kick/tackle callback fired next -- but the same reward had ALREADY been
    written onto its own timed-sample row immediately, unconditionally, so
    that claim was always a duplicate, not a deferred attribution. Confirmed
    in real recorded data: row N carried get_possession=+1.0 with a real
    component breakdown; row N+6, the kick-callback row where the trainee
    kicked the ball away, carried ANOTHER +1.0 with an empty breakdown.

    This reproduces that exact temporal pattern: a distinctive reward fires
    on tick 1 (a genuine timed sample), then an on_kick callback fires on
    tick 2, a SEPARATE later env.step() call. Fixed behaviour: the
    distinctive reward appears exactly once (on tick 1's row); the
    callback's own row carries reward=0.0.
    """
    phase = CurriculumPhase(
        name="p1", phase_id=1, scenario_key="phase1_1v1", env_kwargs={"max_episode_s": 10.0}
    )
    env = build_env(phase)
    label_fn = bc_label_fn_for_phase(1)
    env.always_compute_secondary_reward = True

    orig_step = scenario_env_mod.ScenarioEnv.step
    state = {"ticks": 0}

    def _patched_step(self):
        obs, reward, done, info = orig_step(self)
        state["ticks"] += 1
        if state["ticks"] == 1:
            reward = _DISTINCTIVE_REWARD_2
        elif state["ticks"] == 2:
            match = self._loop.match
            trainee = match.player_by_id(env.trainee_player_id)
            if trainee.on_kick is not None:
                trainee.on_kick(trainee)
            reward = 0.0
            done = True
            self.last_terminal_match = self._loop.match  # see other test's comment
        return obs, reward, done, info

    monkeypatch.setattr(scenario_env_mod.ScenarioEnv, "step", _patched_step)

    result = record_episodes(
        env, label_fn, n_episodes=1, scenario_key="phase1_1v1", phase_id=1,
        sample_interval_s=0.5,
        opponent_rules_prob=0.0, opponent_immobile_prob=1.0,
    )

    rewards = result["rewards"]
    assert state["ticks"] >= 2, "test setup didn't reach the callback tick"

    hit_rows = np.nonzero(rewards == _DISTINCTIVE_REWARD_2)[0]
    assert len(hit_rows) == 1, (
        f"expected the distinctive reward to appear on exactly one row, found {len(hit_rows)} "
        f"(>1 means the kick callback's row duplicated a reward already attributed to an "
        f"earlier timed-sample row -- the exact double-counting bug this test guards)"
    )

    # Every row from the callback tick onward must carry reward=0.0 -- none
    # of them should have "claimed" tick 1's reward for themselves.
    assert np.all(rewards[hit_rows[0] + 1:] == 0.0), (
        f"rows after the distinctive reward's row carry nonzero reward "
        f"{rewards[hit_rows[0] + 1:].tolist()} -- expected all zero (callback/terminal rows "
        f"never carry their own reward, see record_episodes()'s _record_now docstring)"
    )


def test_is_decision_step_flags_real_timed_samples_only(monkeypatch):
    """``is_decision_step`` must be 1.0 for genuine timed-sample rows and 0.0
    for kick/tackle-callback rows and the trailing true-terminal row --
    DemonstrationDataset.compute_returns() relies on this to only apply its
    per-row MC discount on rows that represent a real elapsed
    sample_interval_s, not once per row regardless of real time (see its own
    docstring)."""
    phase = CurriculumPhase(
        name="p1", phase_id=1, scenario_key="phase1_1v1", env_kwargs={"max_episode_s": 10.0}
    )
    env = build_env(phase)
    label_fn = bc_label_fn_for_phase(1)
    env.always_compute_secondary_reward = True

    orig_step = scenario_env_mod.ScenarioEnv.step
    state = {"ticks": 0}

    def _patched_step(self):
        obs, reward, done, info = orig_step(self)
        state["ticks"] += 1
        if state["ticks"] == 2:
            match = self._loop.match
            trainee = match.player_by_id(env.trainee_player_id)
            if trainee.on_kick is not None:
                trainee.on_kick(trainee)
        if state["ticks"] >= 4:
            done = True
            self.last_terminal_match = self._loop.match  # see other test's comment
        return obs, reward, done, info

    monkeypatch.setattr(scenario_env_mod.ScenarioEnv, "step", _patched_step)

    result = record_episodes(
        env, label_fn, n_episodes=1, scenario_key="phase1_1v1", phase_id=1,
        sample_interval_s=0.5,
        opponent_rules_prob=0.0, opponent_immobile_prob=1.0,
    )

    is_decision_step = result["is_decision_step"]
    is_trainee = result["is_trainee"]
    dones = result["dones"]

    assert state["ticks"] >= 4, "test setup didn't reach the forced episode end"
    assert is_decision_step.sum() > 0, "no rows flagged as real decision steps at all"
    assert is_decision_step.sum() < len(is_decision_step), (
        "every row flagged as a real decision step -- the on_kick callback row "
        "(and/or the trailing terminal row) should be flagged 0.0"
    )
    # The trailing terminal row (the true final row for each player, one
    # real physics tick after the last timed sample -- see
    # ScenarioLoop.last_completed_trial_match) must be flagged 0.0.
    trainee_rows = np.nonzero(is_trainee > 0.5)[0]
    last_trainee_row = trainee_rows[-1]
    assert dones[last_trainee_row] > 0.5, "last trainee row should carry the real dones=1 boundary"
    assert is_decision_step[last_trainee_row] < 0.5, (
        "the trailing terminal row must be flagged is_decision_step=0.0 -- it's one real "
        "physics tick after the last timed sample, not a new decision interval of its own"
    )
