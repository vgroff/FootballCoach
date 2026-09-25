"""Coverage for ``PPOTrainer.ppg_value_refit`` and its ``_ppg_kl_penalty``
helper (ai/ppo/ppo_trainer.py) -- the PPG-style (Phasic Policy Gradient)
KL-anchored value refit: full, unfrozen gradient flow through the shared
trunk for the value loss, while an analytic per-head KL penalty anchors the
policy's output distributions to a snapshot taken right before the refit
starts. See ai/knowledge.md "PPG-style value refit" for the full design
writeup, and tests/ai_unit/test_von_mises_kl.py for the hand-derived von
Mises KL formula this leans on.

Three things are tested here:
1. ``_ppg_kl_penalty`` in isolation: current == anchor -> ~0 KL everywhere;
   perturbing one head's anchor moves only that head's breakdown entry.
2. An integration test: a real (tiny) ``ppg_value_refit`` call with
   ``kl_coef=0`` vs a large ``kl_coef``, asserting the large-coefficient run
   drifts the policy strictly less (measured via the same KL machinery) --
   the property that actually matters, not an absolute-threshold assertion.
   Also asserts trunk parameters actually change (proving real gradient
   reached the trunk, unlike ``pretrain_value``).
3. ``separate_value_net=True`` no-ops cleanly without crashing.

Mirrors test_episode_replay.py's fixture pattern (explicit
opponent_rules_prob, not ai_config.json's live ratios -- this suite runs
under pytest-xdist and mutating the shared config file mid-run is unsafe).
"""
from __future__ import annotations

import functools
import random

import pytest
import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _ai_types
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

_PHASE = PHASES_BY_ID[1]
_SHORT_MAX_EPISODE_S = 2.0


def _make_env(trainer) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="ppg_refit_test", label="PPG value refit test scenario",
        description="1v1, short episodes, rules-based opponent -- config-independent",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0),
        on_tick=phase1_training_on_tick,
    )
    env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=_SHORT_MAX_EPISODE_S)
    env.sample_action_fn = trainer._sample_action
    return env


@pytest.fixture()
def trainer():
    # A fresh, trainable (not inference_only) trainer per test -- ppg_value_refit
    # mutates decision_net/execution_net weights in place, so tests must not
    # share one trainer instance across cases.
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    return t


def _collect_tiny_minibatch(trainer, env, n_steps: int = 400):
    """Real rollout -> one minibatch's worth of (mb_obs, mb_actions, exists_mask),
    for feeding _ppg_kl_penalty directly without going through the full
    ppg_value_refit epoch loop."""
    batch, _stats = trainer._collect_value_pretrain_rollout(env, n_steps, phase_id=None)
    obs = {k.replace("obs/", ""): v for k, v in batch.items() if k.startswith("obs/")}
    actions = {k.replace("action/", ""): v for k, v in batch.items() if k.startswith("action/")}
    return obs, actions


class TestPPGKLPenaltyIsolated:
    def test_current_equals_anchor_gives_near_zero_kl_everywhere(self, trainer):
        env = _make_env(trainer)
        obs, actions = _collect_tiny_minibatch(trainer, env)
        anchor = trainer._ppg_snapshot_anchor(obs, batch_size=256)
        anchor_globals = {
            "move_dir_log_kappa": trainer.execution_net.move_dir_log_kappa.detach().clone(),
            "kick_dir_log_kappa": trainer.execution_net.kick_dir_log_kappa.detach().clone(),
            "kick_dir_z_log_std": trainer.execution_net.kick_dir_z_log_std.detach().clone(),
            "kick_power_log_std": trainer.execution_net.kick_power_log_std.detach().clone(),
        }
        sat, oat = _ai_types(obs)
        with torch.no_grad():
            d_heads = trainer.decision_net(
                obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                obs["ball_feat"], obs["global_feat"], sat, oat,
            )
            e_heads = trainer.execution_net(
                obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                obs["ball_feat"], obs["global_feat"], d_heads, sat, oat,
            )
        total_kl, per_head = trainer._ppg_kl_penalty(
            d_heads, e_heads, anchor, anchor_globals, actions, obs["exists_mask"],
        )
        assert total_kl.item() == pytest.approx(0.0, abs=1e-4)
        for name, v in per_head.items():
            assert v.item() == pytest.approx(0.0, abs=1e-4), f"head {name} not ~0: {v.item()}"

    def test_perturbing_one_head_only_moves_that_heads_breakdown_entry(self, trainer):
        env = _make_env(trainer)
        obs, actions = _collect_tiny_minibatch(trainer, env)
        anchor = trainer._ppg_snapshot_anchor(obs, batch_size=256)
        anchor_globals = {
            "move_dir_log_kappa": trainer.execution_net.move_dir_log_kappa.detach().clone(),
            "kick_dir_log_kappa": trainer.execution_net.kick_dir_log_kappa.detach().clone(),
            "kick_dir_z_log_std": trainer.execution_net.kick_dir_z_log_std.detach().clone(),
            "kick_power_log_std": trainer.execution_net.kick_power_log_std.detach().clone(),
        }
        sat, oat = _ai_types(obs)
        with torch.no_grad():
            d_heads = trainer.decision_net(
                obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                obs["ball_feat"], obs["global_feat"], sat, oat,
            )
            e_heads = trainer.execution_net(
                obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                obs["ball_feat"], obs["global_feat"], d_heads, sat, oat,
            )

        # "kick" is an execution-network Bernoulli head -- curriculum
        # freezing (set_frozen_heads) only ever targets the 7 DECISION heads
        # (_LP_HEAD_NAMES), never execution heads, so this is always active
        # regardless of curriculum phase -- a clean, always-safe single-head
        # probe (unlike a decision head, which phase 1 may freeze).
        inactive = trainer._inactive_head_lp_keys()
        assert "kick" not in inactive, "execution heads are never curriculum-frozen"

        perturbed_anchor = dict(anchor)
        perturbed_anchor["kick_logit"] = anchor["kick_logit"] + 5.0

        _, per_head_base = trainer._ppg_kl_penalty(
            d_heads, e_heads, anchor, anchor_globals, actions, obs["exists_mask"],
        )
        _, per_head_perturbed = trainer._ppg_kl_penalty(
            d_heads, e_heads, perturbed_anchor, anchor_globals, actions, obs["exists_mask"],
        )

        assert per_head_perturbed["kick"].item() > 0.5, "perturbed head's KL should be clearly nonzero"
        for name in per_head_base:
            if name == "kick":
                continue
            assert per_head_perturbed[name].item() == pytest.approx(
                per_head_base[name].item(), abs=1e-4,
            ), f"unrelated head {name} moved after perturbing only kick_logit"

    def test_frozen_heads_excluded_from_total(self, trainer):
        env = _make_env(trainer)
        obs, actions = _collect_tiny_minibatch(trainer, env)
        anchor = trainer._ppg_snapshot_anchor(obs, batch_size=256)
        anchor_globals = {
            "move_dir_log_kappa": trainer.execution_net.move_dir_log_kappa.detach().clone(),
            "kick_dir_log_kappa": trainer.execution_net.kick_dir_log_kappa.detach().clone(),
            "kick_dir_z_log_std": trainer.execution_net.kick_dir_z_log_std.detach().clone(),
            "kick_power_log_std": trainer.execution_net.kick_power_log_std.detach().clone(),
        }
        sat, oat = _ai_types(obs)
        with torch.no_grad():
            d_heads = trainer.decision_net(
                obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                obs["ball_feat"], obs["global_feat"], sat, oat,
            )
            e_heads = trainer.execution_net(
                obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                obs["ball_feat"], obs["global_feat"], d_heads, sat, oat,
            )
        inactive = trainer._inactive_head_lp_keys()
        # kick_spin is permanently frozen regardless of curriculum phase.
        assert "kick_spin" in inactive
        _, per_head = trainer._ppg_kl_penalty(
            d_heads, e_heads, anchor, anchor_globals, actions, obs["exists_mask"],
        )
        assert per_head["kick_spin"].item() == 0.0
        for name in inactive:
            if name in per_head:
                assert per_head[name].item() == 0.0


class TestPPGValueRefitIntegration:
    def test_separate_value_net_is_a_clean_noop(self):
        t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
        env = _make_env(t)
        result = t.ppg_value_refit(env, n_steps=200, phase_id=None, epochs=1)
        assert result == {}

    def test_val_fraction_zero_trains_on_everything_and_never_restores_best(self, trainer, caplog):
        """``bc.ppg_val_episode_fraction=0`` = no val set: the split log line
        shows no held-out episodes, per-epoch lines carry no val figures, and
        with nothing to early-stop on there is no best-weights restore."""
        import logging

        trainer._ppg_val_episode_fraction = 0.0
        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=600, phase_id=None, epochs=2, kl_coef=1.0, lr=1e-4)

        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "PPG refit split:" in text and "val eps" not in text
        assert "val_rmse" not in text
        assert "restored best-val weights" not in text
        assert "early stop" not in text
        # both epochs ran and were logged
        assert "PPG refit epoch 2/2" in text

    def test_kl_coef_actually_protects_the_policy(self, trainer):
        """The property that matters: a larger kl_coef must leave the policy
        CLOSER to where it started than kl_coef=0, measured via the same KL
        machinery the refit itself uses -- not an absolute-threshold check,
        since exact drift depends on network init/rollout noise."""
        import copy as _copy

        env = _make_env(trainer)
        # Snapshot BEFORE either refit so both runs start from the identical policy.
        dec_state0 = _copy.deepcopy(trainer.decision_net.state_dict())
        exec_state0 = _copy.deepcopy(trainer.execution_net.state_dict())

        def _measure_drift_from(dec_state, exec_state, kl_coef: float) -> float:
            t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False)
            if _PHASE.frozen_heads:
                t.set_frozen_heads(_PHASE.frozen_heads)
            t.decision_net.load_state_dict(dec_state)
            t.execution_net.load_state_dict(exec_state)
            env2 = _make_env(t)

            obs, actions = _collect_tiny_minibatch(t, env2, n_steps=600)
            anchor = t._ppg_snapshot_anchor(obs, batch_size=256)
            anchor_globals = {
                "move_dir_log_kappa": t.execution_net.move_dir_log_kappa.detach().clone(),
                "kick_dir_log_kappa": t.execution_net.kick_dir_log_kappa.detach().clone(),
                "kick_dir_z_log_std": t.execution_net.kick_dir_z_log_std.detach().clone(),
                "kick_power_log_std": t.execution_net.kick_power_log_std.detach().clone(),
            }
            t.ppg_value_refit(env2, n_steps=600, phase_id=None, epochs=3, kl_coef=kl_coef, lr=3e-3)

            sat, oat = _ai_types(obs)
            with torch.no_grad():
                d_heads = t.decision_net(
                    obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                    obs["ball_feat"], obs["global_feat"], sat, oat,
                )
                e_heads = t.execution_net(
                    obs["self_feat"], obs["other_feat"], obs["exists_mask"],
                    obs["ball_feat"], obs["global_feat"], d_heads, sat, oat,
                )
            total_kl, _ = t._ppg_kl_penalty(d_heads, e_heads, anchor, anchor_globals, actions, obs["exists_mask"])
            return total_kl.item(), t

        # Seed BOTH torch's and Python's global RNGs identically before each
        # run -- episode seeds are drawn via random.randint (see
        # batched_rollout_worker.py's "Episode-seed replay" docstring on why
        # that's statistically equivalent to bare env.reset()), which torch's
        # own seed does not control, so without this the two runs would see
        # different episode content and the comparison below could flake.
        torch.manual_seed(0)
        random.seed(0)
        drift_no_penalty, _t_zero = _measure_drift_from(dec_state0, exec_state0, kl_coef=0.0)
        torch.manual_seed(0)
        random.seed(0)
        drift_with_penalty, t_large = _measure_drift_from(dec_state0, exec_state0, kl_coef=50.0)

        assert drift_with_penalty < drift_no_penalty, (
            f"kl_coef=50 should drift the policy less than kl_coef=0: "
            f"{drift_with_penalty} vs {drift_no_penalty}"
        )

        # Trunk actually changed (proves real gradient reached it, unlike
        # pretrain_value's frozen-trunk mode) -- checked on the kl_coef=50 run.
        trunk_before = dec_state0["trunk.0.weight"] if "trunk.0.weight" in dec_state0 else next(
            v for k, v in dec_state0.items() if "trunk" in k and k.endswith("weight")
        )
        trunk_after = next(
            v for k, v in t_large.decision_net.state_dict().items() if "trunk" in k and k.endswith("weight")
        )
        assert not torch.allclose(trunk_before, trunk_after), "trunk weights should have moved after the refit"


class TestPPGCompareToOriginal:
    def test_compare_line_is_logged_and_first_cycle_delta_is_zero(self, trainer, caplog):
        """The refit starts from the very weights the frozen 'original' copy was
        taken from, so on cycle 1 original == current and delta must be ~0 --
        and the line must actually appear (i.e. mc_returns survived merge,
        split and augmentation)."""
        import logging
        import re

        trainer._ppg_compare_to_original = True
        trainer._ppg_val_episode_fraction = 0.0
        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=600, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4)
        text = "\n".join(r.getMessage() for r in caplog.records)
        m = re.search(r"\[ppg compare vs pure-MC returns\].*?delta\(original-current\)=([+-][0-9.]+)", text)
        assert m, f"compare line missing:\n{text[-1500:]}"
        assert abs(float(m.group(1))) < 1e-6

    def test_off_by_default_logs_nothing(self, trainer, caplog):
        import logging

        trainer._ppg_compare_to_original = False
        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=400, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4)
        assert "[ppg compare" not in "\n".join(r.getMessage() for r in caplog.records)


class TestPPGSeedAdamVariance:
    """bc.ppg_seed_adam_variance (off by default) -- one-directional seeding of
    ppg_value_refit's fresh Adam from the live PPO self.optimizer's per-parameter
    exp_avg_sq (variance) and step count, momentum (exp_avg) always zeroed,
    self.optimizer itself never written to. See its ai_config.json comment and
    the seeding block right after ppg_opt's construction in ppg_value_refit."""

    def test_off_by_default_no_seeding_and_no_log_line(self, trainer, caplog):
        import logging

        assert trainer._ppg_seed_adam_variance is False
        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=400, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4)
        assert "seeded Adam exp_avg_sq" not in "\n".join(r.getMessage() for r in caplog.records)

    def test_seeds_variance_and_step_zeros_momentum_read_only(self, trainer, caplog, monkeypatch):
        import logging

        trainer._ppg_seed_adam_variance = True
        # Give the LIVE optimizer distinctive, fake Adam state for one real
        # trainable param, as if PPO had already trained for a long time --
        # Adam's state dict is keyed by parameter object identity, not name.
        p = next(iter(trainer.decision_net.parameters()))
        fake_exp_avg_sq = torch.full_like(p, 7.0)
        fake_exp_avg = torch.full_like(p, 3.0)  # must NOT be copied into ppg_opt
        # Real Adam (this torch version) always stores step as a 0-dim tensor
        # internally (required by its functional API) -- match that here rather
        # than a plain python int, which would only be realistic for a version
        # of torch this repo doesn't use.
        fake_step = torch.tensor(123456.0)
        trainer.optimizer.state[p] = {
            "exp_avg_sq": fake_exp_avg_sq.clone(),
            "exp_avg": fake_exp_avg.clone(),
            "step": fake_step,
        }

        # Capture ppg_opt's per-parameter state for `p` at the moment its FIRST
        # .step() is called -- i.e. right after ppg_value_refit's seeding block
        # ran but before any real gradient update decays exp_avg_sq away from
        # the seeded value (Adam's own update, v = beta2*v_prev + ..., moves it
        # immediately on step 1, so checking post-hoc after the whole refit
        # call would be checking an already-decayed number, not what was seeded).
        created_opts = []
        pre_step_snapshot = {}
        _orig_adam = torch.optim.Adam

        def _capturing_adam(*args, **kwargs):
            opt = _orig_adam(*args, **kwargs)
            created_opts.append(opt)
            _orig_step = opt.step

            def _step_and_snapshot(*a, **kw):
                if not pre_step_snapshot and p in opt.state:
                    pre_step_snapshot["exp_avg_sq"] = opt.state[p]["exp_avg_sq"].clone()
                    pre_step_snapshot["exp_avg"] = opt.state[p]["exp_avg"].clone()
                    pre_step_snapshot["step"] = opt.state[p]["step"].clone()
                return _orig_step(*a, **kw)

            opt.step = _step_and_snapshot
            return opt

        monkeypatch.setattr(torch.optim, "Adam", _capturing_adam)

        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=400, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4)

        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "seeded Adam exp_avg_sq" in text

        assert len(created_opts) == 1, "ppg_value_refit should build exactly one fresh Adam"
        assert pre_step_snapshot, "ppg_opt.step() was never called -- refit didn't train"
        assert torch.allclose(pre_step_snapshot["exp_avg_sq"], fake_exp_avg_sq)
        assert torch.allclose(pre_step_snapshot["exp_avg"], torch.zeros_like(p)), "momentum must NOT be copied"
        assert pre_step_snapshot["step"].item() == fake_step.item()


class TestPPGFirstRolloutReplaySeeds:
    """ppg_value_refit's first_rollout_replay_seeds (forwarded to
    _collect_value_pretrain_rollout's replay_seeds) -- lets _maybe_run_ppg_interleave
    pass the main loop's queued episode-seed-replay seeds into the FIRST cycle's
    rollout only. phase_id=None (every test fixture here) means the single-process
    branch, which has no pool to inject seeds into and just logs-and-ignores them --
    a convenient way to prove the cycle-gating without a real multi-process pool."""

    def test_none_by_default_is_a_noop(self, trainer, caplog):
        import logging

        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=400, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4)
        assert "replay seed(s)" not in "\n".join(r.getMessage() for r in caplog.records)

    def test_only_forwarded_to_the_first_cycle_not_later_ones(self, trainer, caplog):
        """With num_rollouts=2, the ignore-log (single-process branch) must appear
        exactly once -- proving first_rollout_replay_seeds reaches cycle 1's
        collection call but NOT cycle 2's (which must see replay_seeds=None)."""
        import logging

        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(
                env, n_steps=400, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4,
                num_rollouts=2, first_rollout_replay_seeds=[111, 222, 333],
            )
        ignore_lines = [
            r.getMessage() for r in caplog.records if "replay seed(s) given but ignored" in r.getMessage()
        ]
        assert len(ignore_lines) == 1, f"expected exactly one ignore-log (cycle 1 only), got: {ignore_lines}"
        assert "3 replay seed(s)" in ignore_lines[0]


class TestPPGLossDiagnostics:
    def test_breakdown_is_logged_and_per_row_arrays_saved(self, trainer, caplog, tmp_path):
        """Cycle 1 prints the full tagged breakdown (with the original column
        when compare-to-original is on) and writes the per-row npz."""
        import logging

        import numpy as np

        trainer._ppg_loss_diagnostics = True
        trainer._ppg_compare_to_original = True
        trainer._ppg_val_episode_fraction = 0.0
        trainer.checkpoint_dir = tmp_path
        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=800, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "[ppg loss diag]" in text and "by episode outcome" in text
        assert "by position within the episode" in text and "calibration" in text
        assert "orig mse" in text, "original column expected when compare-to-original is on"
        f = tmp_path / "ppg_diag_cycle1.npz"
        assert f.exists()
        d = np.load(f, allow_pickle=False)
        n = len(d["mc"])
        assert n > 0 and len(d["pred_current"]) == n and len(d["phase"]) == n and len(d["outcome"]) == n
        assert len(d["pred_original"]) == n
        # current == original on cycle 1 (same weights)
        assert np.allclose(d["pred_current"], d["pred_original"], atol=1e-6)

    def test_off_by_default_logs_and_saves_nothing(self, trainer, caplog, tmp_path):
        import logging

        trainer._ppg_loss_diagnostics = False
        trainer._ppg_compare_to_original = False
        trainer.checkpoint_dir = tmp_path
        env = _make_env(trainer)
        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            trainer.ppg_value_refit(env, n_steps=400, phase_id=None, epochs=1, kl_coef=1.0, lr=1e-4)
        assert "[ppg loss diag]" not in "\n".join(r.getMessage() for r in caplog.records)
        assert not list(tmp_path.glob("ppg_diag_cycle*.npz"))
