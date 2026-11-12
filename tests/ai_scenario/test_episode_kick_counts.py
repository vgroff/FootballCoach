"""Coverage for the per-episode trainee kick count ([PPO] `kicks` line): StepInfo.trainee_kicks_this_step /
trainee_armed_kicks_this_step summed per episode in BatchedEnvGroup.collect() (and the other rollout paths), then rendered by
PPOTrainer._log_rollout_summary. The count must include first-touch (armed) kicks and match an INDEPENDENT counter hooked on
Player._finish_kick, the bookkeeping tail of every kick. BC label generation (phase1_labels_for_player) also reaches
_finish_kick, for a hypothetical push-kick it then rolls back, so the reference counter skips that window."""
from __future__ import annotations

import logging

import pytest
import torch

from footballcoach.ai.ppo.batched_rollout_worker import BatchedEnvGroup
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3

from tests.ai_scenario.test_batched_rollout import _make_envs, trainer as _trainer  # noqa: F401  (fixture re-export)

trainer = _trainer


@pytest.fixture()
def always_kick(trainer):
    """Kick gate forced ~1 (bias +20, weights 0) so kicks -- in possession AND armed first-touch -- happen constantly."""
    lin = trainer.execution_net.kick_logit
    saved = {k: v.clone() for k, v in lin.state_dict().items()}
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.fill_(20.0)
    yield
    lin.load_state_dict(saved)


def _run_with_independent_counter(trainer, n_steps: int, seed: int):
    envs = _make_envs(trainer, 1, max_episode_s=3.0)
    group = BatchedEnvGroup(envs, trainer, seeds=[seed])
    env = envs[0]
    per_episode_all: list[int] = []
    per_episode_armed: list[int] = []
    cur = {"all": 0, "armed": 0}
    labelling = {"depth": 0, "sandbox_kicks": 0}

    def wrap_label_fn(ai):
        fn = getattr(ai, "bc_label_fn", None)
        if fn is None or getattr(fn, "_counts_sandbox", False):
            return

        def labelled(*a, **k):
            labelling["depth"] += 1
            try:
                return fn(*a, **k)
            finally:
                labelling["depth"] -= 1

        labelled._counts_sandbox = True
        ai.bc_label_fn = labelled

    orig_finish = Player._finish_kick
    orig_reset = env.reset

    def finish(self, *a, **k):
        if self.player_id == "trainee":
            if labelling["depth"]:
                labelling["sandbox_kicks"] += 1
            else:
                cur["all"] += 1
                cur["armed"] += int(bool(self.kick_armed))
        return orig_finish(self, *a, **k)

    def reset(*a, **k):
        per_episode_all.append(cur["all"])
        per_episode_armed.append(cur["armed"])
        cur["all"] = cur["armed"] = 0
        out = orig_reset(*a, **k)
        # A randomly-initialised trainee rarely reaches the ball inside a short episode, so put the ball on it: alternate
        # episodes start with the trainee in possession (first decision = in-possession kick) and with a LOOSE ball at the
        # trainee's feet (first decision = armed kick, fires at pickup = a first-touch kick).
        m = env._loop.match
        tr = m.player_by_id("trainee")
        m.ball.position = Vector3(tr.position.x, tr.position.y, 0.0)
        m.ball.velocity = Vector3.zero()
        # The scenario's random start velocity can carry the trainee straight off a loose ball (pickup needs
        # closing, or relative speed under ball_pickup.closing_speed_deadzone_mps), which left the armed path
        # unexercised in some runs. Start at rest so the loose-ball episodes always pick up on the first tick.
        tr.velocity = Vector3.zero()
        m._set_possession("trainee" if len(per_episode_all) % 2 else None)
        wrap_label_fn(tr.ai)
        return out

    Player._finish_kick = finish
    env.reset = reset
    wrap_label_fn(env._loop.match.player_by_id("trainee").ai)
    try:
        results = group.collect(n_steps=n_steps)
    finally:
        Player._finish_kick = orig_finish
        env.reset = orig_reset
    return results[0]["stats"], per_episode_all, per_episode_armed, labelling["sandbox_kicks"]


def test_counts_line_up_with_episodes_and_are_sane(trainer):
    envs = _make_envs(trainer, 2, max_episode_s=2.0)
    group = BatchedEnvGroup(envs, trainer, seeds=[11, 12])
    for r in group.collect(n_steps=300):
        s = r["stats"]
        assert len(s["episode_kick_counts"]) == len(s["episode_armed_kick_counts"]) == len(s["episode_rewards"]) > 0
        assert all(isinstance(x, int) and x >= 0 for x in s["episode_kick_counts"])
        assert all(a <= k for a, k in zip(s["episode_armed_kick_counts"], s["episode_kick_counts"]))


def test_matches_independent_counter_including_first_touch_kicks(trainer, always_kick):
    # Label-generation push-kicks (rolled back, so never real) only happen when the rules teacher would push-kick
    # at that moment, so some runs have none; the reference counter excludes them either way. That the sandbox
    # leaves no trace is pinned by test_phase1_labels_timing.py's repeated-calls test.
    stats, ref_all, ref_armed, _sandbox_kicks = _run_with_independent_counter(trainer, n_steps=400, seed=5)
    n = len(stats["episode_kick_counts"])
    assert n >= 3
    assert sum(stats["episode_kick_counts"]) > 0, "forced-gate run produced no kicks at all -- the test would be vacuous"
    assert sum(stats["episode_armed_kick_counts"]) > 0, "no armed/first-touch kicks -- first-touch path not exercised"
    assert sum(stats["episode_armed_kick_counts"]) < sum(stats["episode_kick_counts"]), "expected some in-possession kicks too"
    # ref lists include the initial reset entry (group construction happened before patching, so entry 0 is the FIRST episode's
    # counter flushed at its own end); compare the completed episodes one-for-one.
    assert stats["episode_kick_counts"] == ref_all[:n]
    assert stats["episode_armed_kick_counts"] == ref_armed[:n]


@pytest.fixture()
def full_trainer():
    torch.manual_seed(0)
    return PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False)


def _real_metrics(t):
    env = _make_envs(t, 1, max_episode_s=2.0)[0]
    batch, _ = t._collect_value_pretrain_rollout(env, 500, phase_id=None, use_gae=True)
    return t._ppo_update(batch, progress=0.0)


def _summary_text(t, metrics, caplog, **kw):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
        t._log_rollout_summary(
            metrics=metrics, steps_per_sec=1.0, episode_rewards=[0.0] * 4, secondary_episode_rewards=[],
            episode_outcomes_vs_rules=[], episode_outcomes_vs_immobile=[], episode_outcomes_vs_neural=[],
            rollout_components={}, episode_comp_list=[], episode_durations_s=[], comp_step_stats={}, n_reward_comp_steps=0, **kw,
        )
    return "\n".join(r.getMessage() for r in caplog.records)


def test_log_line_reports_totals_and_first_touch_share(full_trainer, caplog):
    text = _summary_text(full_trainer, _real_metrics(full_trainer), caplog,
                         episode_kick_counts=[0, 1, 2, 5], episode_armed_kick_counts=[0, 1, 0, 2])
    assert "kicks    2.000" in text
    assert "first-touch(armed)=0.750/ep (38% of kicks)" in text
    assert "0-kick eps=25%" in text and "max=5" in text and "(n=4)" in text


def test_log_line_omitted_when_stats_not_supplied(full_trainer, caplog):
    text = _summary_text(full_trainer, _real_metrics(full_trainer), caplog)
    assert "[PPO] step=" in text and "  kicks    " not in text
