"""Integration coverage for ``PPOTrainer.consistency_refit`` (flip_y consistency distillation).

Uses a real (tiny) rollout on short 1v1 episodes with an explicit rules opponent (config-independent,
same fixture pattern as test_ppg_value_refit.py). The properties that matter:
1. what must stay frozen stays bit-identical (value head, spread params) and requires_grad is restored;
2. the trunk really trains;
3. the student becomes more flip_y-consistent than it started (the whole point);
4. targets are bounded when a logit bound is given.
"""
from __future__ import annotations

import functools
import logging
import math

import pytest
import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

_PHASE = PHASES_BY_ID[1]


def _make_env(trainer) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="consistency_refit_test", label="consistency refit test",
        description="1v1, short episodes, rules-based opponent -- config-independent",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0),
        on_tick=phase1_training_on_tick,
    )
    env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=2.0)
    env.sample_action_fn = trainer._sample_action
    return env


@pytest.fixture()
def trainer():
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    return t


def _named(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def _run(trainer, **kw):
    env = _make_env(trainer)
    return trainer.consistency_refit(env, n_steps=900, phase_id=None, epochs=4, num_rollouts=1,
                                     val_episode_fraction=0.3, lr=1e-3, batch_size=256, **kw)


def test_frozen_stays_frozen_trunk_trains_and_requires_grad_is_restored(trainer):
    en, dn = trainer.execution_net, trainer.decision_net
    before_e, before_d = _named(en), _named(dn)
    flags_before = {n: p.requires_grad for n, p in list(en.named_parameters()) + list(dn.named_parameters())}
    _run(trainer, kappa_deg=22.0)
    after_e, after_d = _named(en), _named(dn)

    kappa = 1.0 / math.radians(22.0) ** 2
    assert en.move_dir_log_kappa.item() == pytest.approx(math.log(kappa), abs=1e-6)
    assert en.kick_dir_log_kappa.item() == pytest.approx(math.log(kappa), abs=1e-6)
    for name in after_e:
        if any(s in name for s in ("value_head", "value_ai_type_channel", "kick_dir_z_log_std", "kick_power_log_std", "kick_spin_log_std")):
            assert torch.equal(after_e[name], before_e[name]), f"{name} must stay frozen"
    trained = [n for n in after_e if "value" not in n and "log_" not in n and not torch.equal(after_e[n], before_e[n])]
    trained += [n for n in after_d if "physics" not in n and not torch.equal(after_d[n], before_d[n])]
    assert trained, "no trunk/head parameter changed -- the refit did not train anything"
    for n, p in list(en.named_parameters()) + list(dn.named_parameters()):
        assert p.requires_grad == flags_before[n], f"requires_grad not restored for {n}"


def test_student_becomes_more_flip_consistent_than_it_started(trainer, caplog):
    with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
        out = _run(trainer, kappa_deg=22.0)
    first, last = out["first_val"]["consistency"], out["last_val"]["consistency"]
    assert last["move_dir_err_p90_deg"] < first["move_dir_err_p90_deg"]
    assert out["last_val"]["kl"] < out["first_val"]["kl"]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[consistency epoch 0 (baseline)]" in text and "flip-consistency:" in text and "Consistency refit summary" in text


def test_bernoulli_targets_are_bounded_when_a_logit_bound_is_given(trainer):
    out = _run(trainer, kappa_deg=22.0, logit_bound=3.0)
    assert out["last_val"]["kl"] < out["first_val"]["kl"]
    # every anchor target the student was pulled toward was within +-3, so it cannot be systematically
    # more saturated than that on the primary rows once trained
    assert all(v > 50.0 for v in out["last_val"]["bounded_pct"].values())
