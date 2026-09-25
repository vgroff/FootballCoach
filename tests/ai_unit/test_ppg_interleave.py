"""Coverage for ``PPOTrainer._maybe_run_ppg_interleave`` / ``_ppg_interleave_trigger_path``
(ai/ppo/ppo_trainer.py) -- the interleaved-PPG trigger mechanism that fires bursts of
``ppg_value_refit()`` mid-PPO instead of only via the standalone ``--ppg-refit-only`` entry
point. See ai/knowledge.md "Interleaved PPG" and memory project_ppg_interleave_design.

``ppg_value_refit`` itself is monkeypatched out in every test here (it's already covered by
test_ppg_value_refit.py, and a real call needs a live env + real rollout collection) -- this
suite is only about the trigger logic: does it fire at the right time, exactly once per manual
touch, with the right forwarded arguments, and does it save a checkpoint afterward. No
multiprocessing, no real rollout, no live training run -- purely in-process and fast.
"""
from __future__ import annotations

import pytest
import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

_PHASE = PHASES_BY_ID[1]


@pytest.fixture()
def trainer(tmp_path):
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    t.checkpoint_dir = tmp_path
    t._ppg_interleave_enabled = True
    t._ppg_interleave_trigger_file = None
    t._ppg_interleave_cadence_rollouts = 0
    t._ppg_interleave_num_rollouts = 3
    t._ppg_interleave_n_steps = 1234
    t._rollouts_since_ppg_interleave = 0

    calls = []

    def _fake_refit(env, n_steps, phase_id=None, num_rollouts=None, **kw):
        calls.append(dict(env=env, n_steps=n_steps, phase_id=phase_id, num_rollouts=num_rollouts))

    t.ppg_value_refit = _fake_refit
    t._refit_calls = calls

    saves = []
    t._save_checkpoint = lambda step: saves.append(step)
    t._checkpoint_saves = saves
    return t


class TestConfigDefaults:
    def test_code_level_default_is_off(self):
        # Opt-in convention: __init__ falls back to False when the key is absent from config
        # (ppo_trainer.py: `bool(bc_cfg.get("ppg_interleave_enabled", False))`). NOT asserted
        # against the live ai_config.json's ppg_interleave_enabled value here -- it's
        # deliberately True there while this feature is being live-tested (same reasoning
        # test_move_dir_entropy_weight.py etc. give for isolating against live non-default
        # config values rather than asserting on them directly).
        t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False)
        assert t._ppg_interleave_cadence_rollouts == 0
        assert t._ppg_interleave_trigger_file is None


class TestDisabledIsNoop:
    def test_disabled_never_fires_even_with_trigger_file_present(self, trainer):
        trainer._ppg_interleave_enabled = False
        path = trainer._ppg_interleave_trigger_path()
        path.touch()

        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)

        assert trainer._refit_calls == []
        assert trainer._checkpoint_saves == []
        # Never even looked at the trigger file, so it's still there.
        assert path.exists()


class TestManualTrigger:
    def test_fires_once_and_consumes_the_file(self, trainer):
        path = trainer._ppg_interleave_trigger_path()
        path.touch()

        trainer._maybe_run_ppg_interleave(env="ENV", phase_id=7)

        assert len(trainer._refit_calls) == 1
        call = trainer._refit_calls[0]
        assert call["env"] == "ENV"
        assert call["phase_id"] == 7
        assert call["n_steps"] == 1234
        assert call["num_rollouts"] == 3
        assert not path.exists(), "trigger file must be consumed (deleted) once noticed"
        assert trainer._checkpoint_saves == [trainer._total_steps]
        assert trainer._rollouts_since_ppg_interleave == 0

        # Calling again without recreating the file must not fire a second time.
        trainer._maybe_run_ppg_interleave(env="ENV", phase_id=7)
        assert len(trainer._refit_calls) == 1

    def test_custom_trigger_file_path_is_used_when_set(self, trainer, tmp_path):
        custom = tmp_path / "subdir" / "my_trigger"
        custom.parent.mkdir()
        trainer._ppg_interleave_trigger_file = str(custom)

        assert trainer._ppg_interleave_trigger_path() == custom

        custom.touch()
        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)

        assert len(trainer._refit_calls) == 1
        assert not custom.exists()

    def test_no_checkpoint_dir_and_no_explicit_file_disables_manual_trigger_only(self, trainer):
        trainer.checkpoint_dir = None
        assert trainer._ppg_interleave_trigger_path() is None

        # The cadence safety net must still work even with no manual trigger path available.
        trainer._ppg_interleave_cadence_rollouts = 1
        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)

        assert len(trainer._refit_calls) == 1
        # No checkpoint_dir -> no extra checkpoint save, and no crash from a None path.
        assert trainer._checkpoint_saves == []


class TestMainPoolCloseRespawn:
    """close_main_pools_fn/respawn_main_pools_fn: added 2026-09-23 after a real crash where the
    main loop's persistent worker pool (rollout + eval workers) stayed alive concurrently with
    ppg_value_refit()'s own freshly-spawned pool and exhausted RAM. Both are optional (None ->
    no-op, the single-process train() path has no persistent pool to manage)."""

    def test_none_callbacks_are_a_noop_default(self, trainer):
        path = trainer._ppg_interleave_trigger_path()
        path.touch()
        # No close_main_pools_fn/respawn_main_pools_fn passed -- must not raise.
        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)
        assert len(trainer._refit_calls) == 1

    def test_close_called_before_refit_and_respawn_called_after(self, trainer):
        events = []
        trainer.ppg_value_refit = lambda *a, **kw: events.append("refit")
        trainer._ppg_interleave_trigger_path().touch()

        trainer._maybe_run_ppg_interleave(
            env=None, phase_id=1,
            close_main_pools_fn=lambda: events.append("close"),
            respawn_main_pools_fn=lambda: events.append("respawn"),
        )

        assert events == ["close", "refit", "respawn"]

    def test_respawn_still_runs_if_refit_raises(self, trainer):
        events = []

        def _boom(*a, **kw):
            events.append("refit")
            raise RuntimeError("simulated refit failure")

        trainer.ppg_value_refit = _boom
        trainer._ppg_interleave_trigger_path().touch()

        with pytest.raises(RuntimeError, match="simulated refit failure"):
            trainer._maybe_run_ppg_interleave(
                env=None, phase_id=1,
                close_main_pools_fn=lambda: events.append("close"),
                respawn_main_pools_fn=lambda: events.append("respawn"),
            )

        # respawn must still happen (via try/finally) even though the burst itself failed --
        # otherwise the main loop would be left with no worker pool at all after a bad refit.
        assert events == ["close", "refit", "respawn"]
        # A raised refit means the interleave never "completed" -- no checkpoint save, and the
        # cadence counter should NOT have been reset (still mid-attempt, not a clean cycle).
        assert trainer._checkpoint_saves == []
        assert trainer._rollouts_since_ppg_interleave == 1


class TestCadenceSafetyNet:
    def test_fires_every_n_rollouts_and_resets_counter(self, trainer):
        trainer._ppg_interleave_cadence_rollouts = 3

        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=1
        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=2
        assert trainer._refit_calls == []

        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=3 -> fires
        assert len(trainer._refit_calls) == 1
        assert trainer._rollouts_since_ppg_interleave == 0

        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=1
        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=2
        assert len(trainer._refit_calls) == 1

        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=3 -> fires again
        assert len(trainer._refit_calls) == 2

    def test_manual_trigger_also_resets_the_cadence_counter(self, trainer):
        trainer._ppg_interleave_cadence_rollouts = 5

        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=1
        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # count=2

        trainer._ppg_interleave_trigger_path().touch()
        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)  # manual fires at count=3

        assert len(trainer._refit_calls) == 1
        assert trainer._rollouts_since_ppg_interleave == 0

        # Cadence timer restarted from the manual fire: needs 5 more calls, not 2 more.
        for _ in range(4):
            trainer._maybe_run_ppg_interleave(env=None, phase_id=1)
        assert len(trainer._refit_calls) == 1

        trainer._maybe_run_ppg_interleave(env=None, phase_id=1)
        assert len(trainer._refit_calls) == 2

    def test_cadence_zero_never_fires_on_its_own(self, trainer):
        trainer._ppg_interleave_cadence_rollouts = 0
        for _ in range(50):
            trainer._maybe_run_ppg_interleave(env=None, phase_id=1)
        assert trainer._refit_calls == []
