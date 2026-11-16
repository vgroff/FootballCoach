"""PPOTrainer checkpoint saving/loading, construction from config, and log-file rotation."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
import torch
from footballcoach.ai.config import load_ai_config
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.ppo.trainer_common import _load_state_dict_tolerant, log


def _migrate_direction_log_std_to_kappa(state_dict: dict) -> dict:
    """Backward-compat shim for the move_dir_log_std/kick_dir_log_std ->
    move_dir_log_kappa/kick_dir_log_kappa/kick_dir_z_log_std rename (see
    ai_trainer_knowledge.md "Direction heads: von Mises"). Unlike
    _migrate_crossing_head_state_dict's shape-only migrations, this is a
    same-shape RENAME plus a numeric transform, since log_kappa is not the
    same quantity as log_std (kappa ~= 1/sigma^2, an inverted, approximate
    relationship, not an exact one).

    Guarded against double-application: a checkpoint already saved under the
    new names is left untouched (checked BEFORE looking for old keys, same
    ordering rationale as _migrate_crossing_head_state_dict's shape-match
    guard -- a checkpoint in the current format must never be reinterpreted
    as if it were the old one). Returns a NEW dict (does not mutate
    state_dict); a no-op if neither old key is present (checkpoint predates
    these params entirely, or already uses the new names).
    """
    old_move_key, new_move_key = "move_dir_log_std", "move_dir_log_kappa"
    old_kick_key, new_kick_kappa_key, new_kick_z_key = (
        "kick_dir_log_std", "kick_dir_log_kappa", "kick_dir_z_log_std",
    )
    if new_move_key in state_dict or new_kick_kappa_key in state_dict:
        return state_dict
    if old_move_key not in state_dict and old_kick_key not in state_dict:
        return state_dict
    state_dict = dict(state_dict)
    if old_move_key in state_dict:
        old_log_std = state_dict.pop(old_move_key)
        sigma = torch.exp(old_log_std)
        new_log_kappa = torch.log(1.0 / (sigma * sigma))
        state_dict[new_move_key] = new_log_kappa
        log.info(
            f"Migrated checkpoint's move_dir_log_std={float(old_log_std):.4f} "
            f"(sigma={float(sigma):.4f}) to move_dir_log_kappa={float(new_log_kappa):.4f} "
            f"via kappa=1/sigma^2 (small-angle approximation)."
        )
    if old_kick_key in state_dict:
        old_log_std = state_dict.pop(old_kick_key)
        sigma = torch.exp(old_log_std)
        new_log_kappa = torch.log(1.0 / (sigma * sigma))
        state_dict[new_kick_kappa_key] = new_log_kappa
        # No analog for the new elevation-only z component in the old
        # (single, isotropic-3D) parameterization -- seed it from the SAME
        # old value directly (same numeric scale, an approximate starting
        # guess, not an exact conversion) rather than a fresh random init,
        # so kick_dir's exploration magnitude doesn't jump discontinuously
        # on resume.
        state_dict[new_kick_z_key] = old_log_std.clone()
        log.info(
            f"Migrated checkpoint's kick_dir_log_std={float(old_log_std):.4f} "
            f"(sigma={float(sigma):.4f}) to kick_dir_log_kappa={float(new_log_kappa):.4f} "
            f"(via kappa=1/sigma^2) and seeded kick_dir_z_log_std={float(old_log_std):.4f} "
            f"from the same old value (approximate starting guess -- the old "
            f"parameterization had no separate elevation component)."
        )
    return state_dict


class CheckpointIOMixin:
    """Checkpoint I/O for PPOTrainer. Mixed into PPOTrainer; relies on its attributes."""

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def _rotate_log_file(self) -> None:
        """(Re)attach a FileHandler writing to
        ``checkpoint_dir/training_log{N}.txt`` where N is one past the
        current ``_checkpoint_count`` -- e.g. before checkpoint1 is saved,
        logs go to ``training_log1.txt``; once checkpoint1 is saved, the
        handler rotates so subsequent logs (up through checkpoint2) go to
        ``training_log2.txt``, mirroring the ``checkpoint{N}.pt`` numbering.
        Attaches to the ``"footballcoach"`` root logger so every module's
        logs (train.py, ppo_trainer.py, bc.py, dataset.py, ...) land in the
        same file, not just this module's own logger.
        """
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger("footballcoach")
        if self._log_file_handler is not None:
            root.removeHandler(self._log_file_handler)
            self._log_file_handler.close()
        log_path = self.checkpoint_dir / f"training_log{self._checkpoint_count + 1}.txt"
        # utf-8 explicitly -- many diagnostic log lines use Unicode symbols
        # (Delta, sigma, box-drawing separators, degree signs); without this,
        # FileHandler defaults to the system locale (cp1252 on Windows),
        # which can't encode them and drops those lines with a logging error.
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(handler)
        self._log_file_handler = handler
        log.info(f"Logging to {log_path}")

    def _save_checkpoint(self, step: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_count += 1
        path = self.checkpoint_dir / f"checkpoint{self._checkpoint_count}.pt"
        ckpt = {
            "step": step,
            "checkpoint_count": self._checkpoint_count,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
            "task_id_inputs_ready": bool(getattr(self, "task_id_inputs_ready", True)),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.value_net is not None:
            ckpt["value_net"] = self.value_net.state_dict()
            ckpt["value_net_optimizer"] = self.value_net_optimizer.state_dict()
        torch.save(ckpt, path)
        # Update latest.pt -- prefer a symlink (cheap, no duplicate disk
        # usage), but Windows requires admin/Developer Mode privileges to
        # create one. Fall back to a plain copy so checkpoint saving never
        # crashes an otherwise-healthy training run over this.
        latest = self.checkpoint_dir / "latest.pt"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        try:
            latest.symlink_to(path.name)
        except OSError:
            shutil.copy2(path, latest)
        log.info(f"Saved checkpoint: {path}")
        self._rotate_log_file()

    def _save_checkpoint_to(self, path: Path) -> None:
        """Save a checkpoint to an explicit path (used for pre-trained snapshot).

        self.optimizer is None for a trainer built with inference_only=True
        (e.g. a standalone script training decision_net/execution_net with
        its own separate optimizer, never touching PPO's) -- omit the
        "optimizer" key in that case rather than crashing; load_checkpoint()
        already tolerates its absence the same way ("if self.optimizer is
        not None and 'optimizer' in ckpt:"), this just makes the save side
        consistent with that.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "step": self._total_steps,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
            "task_id_inputs_ready": bool(getattr(self, "task_id_inputs_ready", True)),
        }
        if self.optimizer is not None:
            ckpt["optimizer"] = self.optimizer.state_dict()
        if self.value_net is not None:
            ckpt["value_net"] = self.value_net.state_dict()
            if self.value_net_optimizer is not None:
                ckpt["value_net_optimizer"] = self.value_net_optimizer.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Path, reset_optimizer: bool = False) -> int:
        """Load network weights (and, unless reset_optimizer, optimizer
        state) from a checkpoint saved by _save_checkpoint()/_save_checkpoint_to().

        Args:
            reset_optimizer: If True, skip restoring Adam's state (per-param
                running m/v moment estimates + step count) for both
                self.optimizer and self.value_net_optimizer -- network
                weights still load normally either way. Adam's own state is
                otherwise carried forward unchanged across a resume (same as
                self._total_steps), which is usually what you want for a
                true continuation, but not when you've since changed
                hyperparameters enough (e.g. max_grad_norm, learning rates,
                added/removed param groups) that Adam's old running
                averages -- tuned for the previous regime -- would fight the
                new one rather than help it. Left in place (not
                auto-reset), same param-group count or not, unless this is
                explicitly set.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Tolerant load (not strict): decision_net.state_dict() deliberately
        # excludes ball_physics_encoder/player_physics_encoder keys (see
        # DecisionNetwork.state_dict()), so a checkpoint saved with this
        # feature enabled never has them either -- strict loading would
        # otherwise raise "missing keys" for a submodule that's already
        # correctly populated from its own physics_pretrain checkpoint path.
        _load_state_dict_tolerant(self.decision_net, ckpt["decision_net"], "decision_net")
        _execution_sd = _migrate_direction_log_std_to_kappa(ckpt["execution_net"])
        _load_state_dict_tolerant(self.execution_net, _execution_sd, "execution_net")
        if self.value_net is not None:
            if "value_net" in ckpt:
                # value_net reuses the whole ExecutionNetwork class (only its
                # .value output is ever read) so it carries the same stray
                # move_dir_log_std/kick_dir_log_std keys under an old
                # checkpoint -- migrate here too, purely for a clean load
                # (these params are never actually used by the critic).
                _value_sd = _migrate_direction_log_std_to_kappa(ckpt["value_net"])
                _load_state_dict_tolerant(self.value_net, _value_sd, "value_net")
                if not reset_optimizer and self.value_net_optimizer is not None and "value_net_optimizer" in ckpt:
                    self.value_net_optimizer.load_state_dict(ckpt["value_net_optimizer"])
            else:
                log.warning(
                    f"separate_value_net enabled but {path} has no 'value_net' key "
                    "(checkpoint predates this feature) -- value_net keeps its fresh "
                    "random init."
                )
        if reset_optimizer:
            log.info(f"--reset-optimizer: skipping optimizer state restore for {path} (network weights still loaded normally)")
        elif self.optimizer is not None and "optimizer" in ckpt:
            saved_n_groups = len(ckpt["optimizer"].get("param_groups", []))
            current_n_groups = len(self.optimizer.param_groups)
            if saved_n_groups == current_n_groups:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                # Optimizer shape changed (e.g. value-head param group added) since this
                # checkpoint was saved. Weights still load fine above; only the optimizer's
                # momentum/Adam state is skipped — harmless for --from-pretrained/--latest-
                # pretrain (a fresh PPO run builds its own optimizer state from step 0
                # anyway) but means a true PPO *resume* from an old-shape checkpoint will
                # restart Adam's running averages rather than continuing them exactly.
                log.warning(
                    f"Optimizer param group count changed ({saved_n_groups} -> "
                    f"{current_n_groups}); skipping optimizer state restore for {path} "
                    "(network weights still loaded normally)."
                )
        self._total_steps = ckpt["step"]
        # Absent = saved before task ids reached the training rows (2026-11-16):
        # those weights were trained at "all task ids 0". See
        # ai/ppo/new_phase_inputs.py and PPOTrainer._check_task_id_inputs.
        self.task_id_inputs_ready = bool(ckpt.get("task_id_inputs_ready", False))
        # checkpoint{N}.pt / training_log{N}.txt numbering is driven ENTIRELY by
        # what's physically on disk in THIS checkpoint_dir -- never by the loaded
        # checkpoint's own embedded "checkpoint_count" field. Two real incidents
        # this session, in order:
        #   1. load_checkpoint() didn't restore checkpoint_count at all, so
        #      resuming a killed/crashed run from a --checkpoint path (as opposed
        #      to never-interrupted continuous training, where this lives in
        #      memory the whole process lifetime) silently reset it to 0 -- the
        #      new process overwrote checkpoint1.pt, checkpoint2.pt, ... from
        #      scratch instead of continuing at the saved value.
        #   2. The fix for (1) (restoring the embedded value, cross-checked
        #      against on-disk files ONLY when self.checkpoint_dir.exists())
        #      still broke the OPPOSITE case: launching a genuinely NEW,
        #      not-yet-created checkpoint_dir from an OLD checkpoint (e.g.
        #      starting run318 from run317's final checkpoint592.pt, specifically
        #      to get a clean numbering restart after run317 had its own earlier
        #      resume incident). Since the new directory doesn't exist yet at
        #      this point, ``self.checkpoint_dir.exists()`` was False, so the
        #      whole cross-check was skipped and the stale embedded value (592)
        #      won by default -- silently carrying old numbering into a directory
        #      that should have started fresh at checkpoint1.pt, breaking every
        #      eval.neural_snapshot_lookbacks entry that depends on low-numbered
        #      files (_resolve_snapshot_lookback's "original" and large-fraction
        #      lookbacks) for the entire length of the new run, since those
        #      checkpoint numbers now point below the directory's actual range
        #      and will keep doing so until the count grows past where the old
        #      run left off.
        # The embedded field is NEVER read now -- on_disk_max alone (0 for an
        # empty or not-yet-existing directory, matching every "start a genuinely
        # new run" case the same way an empty tmp_path does in the regression
        # tests) is both simpler and correct for all three real scenarios: resume
        # in the SAME directory (on-disk files ARE the true history), a directory
        # with gaps from a prior cleanup (on-disk files are still the only
        # trustworthy source), and a brand-new directory (0, correctly resets).
        on_disk_max = 0
        if self.checkpoint_dir is not None and self.checkpoint_dir.exists():
            for p in self.checkpoint_dir.glob("checkpoint*.pt"):
                _n = p.stem.removeprefix("checkpoint")
                if _n.isdigit():
                    on_disk_max = max(on_disk_max, int(_n))
        if self.checkpoint_dir is not None:
            self._checkpoint_count = on_disk_max
        log.info(
            f"Loaded checkpoint: {path} (step {self._total_steps}, checkpoint_count "
            f"{self._checkpoint_count} -- from on-disk state in {self.checkpoint_dir}, "
            f"not the checkpoint's own embedded value)"
        )
        # __init__ already called _rotate_log_file() once, before this method ever
        # runs, using _checkpoint_count's constructor-time value of 0 -- so without
        # this second call, the just-restored count above only fixes FUTURE
        # checkpoint{N}.pt filenames (next _save_checkpoint() reads the correct
        # value); logging would stay pinned to whatever training_log1.txt-equivalent
        # __init__ already opened and keep appending there for the rest of the
        # process. Confirmed in practice: this was caught only after the
        # _checkpoint_count fix above still produced a fresh training_log1.txt on a
        # real resume, while checkpoint{N}.pt numbering (which only depends on the
        # NEXT _save_checkpoint() call, with no equivalent early-open step) had
        # already resumed correctly. Re-rotating here closes that one pre-restore
        # log file (which only ever got a couple of startup lines) and reopens at
        # the correct training_log{_checkpoint_count + 1}.txt.
        if self.checkpoint_dir is not None:
            self._rotate_log_file()
        return self._total_steps

    @classmethod
    def from_config(cls, **kwargs) -> "PPOTrainer":
        """Build a PPOTrainer with freshly-initialised networks from ai_config.json.

        Pass separate_value_net=True to enable the dedicated, fully independent
        critic network (see __init__ docstring / CLI --separate-value-net).
        """
        cfg = load_ai_config()
        decision_net = DecisionNetwork.from_config()
        net_cfg = cfg.get("network", {})
        if net_cfg.get("share_entity_encoder", False):
            execution_net = ExecutionNetwork.from_config(
                shared_entity_encoder=decision_net.entity_encoder,
                shared_ball_mlp=decision_net.ball_mlp,
                shared_global_mlp=decision_net.global_mlp,
            )
        else:
            execution_net = ExecutionNetwork.from_config()
        return cls(decision_net=decision_net, execution_net=execution_net, cfg=cfg, **kwargs)

    @classmethod
    def load_for_inference(cls, path: "Path | str") -> "PPOTrainer":
        """Load networks only — no optimizer created. Safe to call inside pygame/UI."""
        path = Path(path)
        cfg = load_ai_config()
        decision_net = DecisionNetwork.from_config()
        execution_net = ExecutionNetwork.from_config()
        # Auto-detect separate_value_net from the checkpoint itself so inference
        # (UI/evaluate.py) doesn't need to know which mode a given checkpoint was
        # trained with.
        _ckpt_peek = torch.load(path, map_location="cpu", weights_only=False)
        _separate_value_net = "value_net" in _ckpt_peek
        trainer = cls(
            decision_net=decision_net,
            execution_net=execution_net,
            cfg=cfg,
            inference_only=True,
            separate_value_net=_separate_value_net,
        )
        trainer.load_checkpoint(path)
        trainer.decision_net.eval()
        trainer.execution_net.eval()
        if trainer.value_net is not None:
            trainer.value_net.eval()
        return trainer
