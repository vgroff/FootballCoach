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


_LEAK_PER_TICK_REWARD = 100.0
_LEAK_THRESHOLD = 50.0  # anything at/above this can only be explained by a cross-episode leak


def test_pending_reward_does_not_leak_across_episodes(monkeypatch):
    """Regression test: ``_pending_reward`` (accrued per-tick, drained only
    when an on_kick/on_tackle callback fires for that player) must be reset
    at every ``env.reset()`` -- it represents "reward accrued since this
    player's last recorded sample", which can never legitimately span an
    episode boundary. Left uncleared, a player whose callback rarely fires
    (e.g. a neural-driven trainee that almost never kicks) silently
    accumulates reward across many episodes, then dumps the entire backlog
    onto a single row the next time a callback happens to fire -- confirmed
    in real recorded data: one row carried reward=1474.952, corrupting that
    episode's MC returns into the hundreds/thousands.

    Episode 1 here accrues a large, distinctive reward every tick with NO
    callback ever firing (so it would all sit in ``_pending_reward`` if
    uncleared). Episode 2 fires an on_kick callback on its very first tick,
    before any of its own reward could have accrued -- if episode 1's
    backlog leaked through, that callback's row would carry ~100 * n_ticks;
    if reset correctly, it carries ~0.
    """
    phase = CurriculumPhase(
        name="p1", phase_id=1, scenario_key="phase1_1v1", env_kwargs={"max_episode_s": 10.0}
    )
    env = build_env(phase)
    label_fn = bc_label_fn_for_phase(1)
    env.always_compute_secondary_reward = True

    orig_step = scenario_env_mod.ScenarioEnv.step
    orig_reset = scenario_env_mod.ScenarioEnv.reset
    state = {"episode": -1, "ticks_this_episode": 0}

    def _patched_reset(self):
        state["episode"] += 1
        state["ticks_this_episode"] = 0
        return orig_reset(self)

    def _patched_step(self):
        obs, reward, done, info = orig_step(self)
        state["ticks_this_episode"] += 1
        if state["episode"] == 0:
            # Episode 1: large distinctive per-tick reward, no callback ever
            # fires, force a clean end after a few ticks.
            reward = _LEAK_PER_TICK_REWARD
            if state["ticks_this_episode"] >= 4:
                done = True
        elif state["episode"] == 1:
            # Episode 2, first tick: fire the trainee's own on_kick callback
            # immediately, before any of THIS episode's reward has accrued.
            if state["ticks_this_episode"] == 1:
                match = self._loop.match
                trainee = match.player_by_id(env.trainee_player_id)
                if trainee.on_kick is not None:
                    trainee.on_kick(trainee)
            reward = 0.0
            done = True
        return obs, reward, done, info

    monkeypatch.setattr(scenario_env_mod.ScenarioEnv, "step", _patched_step)
    monkeypatch.setattr(scenario_env_mod.ScenarioEnv, "reset", _patched_reset)

    result = record_episodes(
        env, label_fn, n_episodes=2, scenario_key="phase1_1v1", phase_id=1,
        sample_interval_s=0.5,
        opponent_rules_prob=0.0, opponent_immobile_prob=1.0,
    )

    rewards = result["rewards"]
    dones = result["dones"]

    assert state["episode"] >= 1, "test setup didn't reach episode 2"

    # Episode 2 = every row after the first done==1 pair.
    first_done_pair_end = np.nonzero(dones > 0.5)[0][1]
    ep2_rewards = rewards[first_done_pair_end + 1:]
    assert len(ep2_rewards) > 0, "episode 2 recorded no rows"
    bad = ep2_rewards[np.abs(ep2_rewards) >= _LEAK_THRESHOLD]
    assert len(bad) == 0, (
        f"episode 2 contains reward value(s) {bad.tolist()} that can only be explained by "
        f"episode 1's un-drained _pending_reward backlog leaking across env.reset() -- "
        f"_pending_reward must be cleared at the start of every episode."
    )
