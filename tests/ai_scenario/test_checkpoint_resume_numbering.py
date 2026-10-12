"""Regression tests for PPOTrainer.load_checkpoint()'s checkpoint{N}.pt /
training_log{N}.txt numbering restore.

Background: resuming a killed/crashed training run from a --checkpoint path
(as opposed to never-interrupted continuous training, where _checkpoint_count
lives in memory the whole process lifetime) used to silently reset the
counter to 0, so the new process overwrote the original early
checkpoint{N}.pt files in the same directory from scratch (plain
torch.save(), no append protection -- unlike logging.FileHandler, which
appends). Found and fixed in practice resuming a real training run, in two
layers:

  1. load_checkpoint() restored _total_steps from the checkpoint dict but
     never _checkpoint_count, despite it always being saved. Fixed by
     restoring it too.
  2. Restoring it in (1) only fixes the NEXT _save_checkpoint() call's
     numbering -- __init__ already calls _rotate_log_file() once, using
     _checkpoint_count's constructor-time value of 0, before load_checkpoint
     ever runs. So logging stayed pinned to the wrong (count=0-derived)
     training_log1.txt for the rest of the process even after (1). Fixed by
     calling _rotate_log_file() again at the end of load_checkpoint(), after
     the restore.
  3. A further, more subtle case: a resume from a checkpoint whose OWN
     embedded checkpoint_count is itself stale (e.g. it was saved by an
     earlier resume that started from a bad count before fix (1) existed) --
     restoring that stale value is "correct" per the loaded dict but still
     wrong in absolute terms, and resuming from it again collides with and
     overwrites real existing checkpoint{N}.pt history. Fixed by cross-
     checking the restored value against the highest checkpoint{N}.pt
     actually present in checkpoint_dir and using whichever is higher.

See agent_plans/more_players_extension_readiness.md for the original
incident writeup.
"""
from __future__ import annotations

import torch

from footballcoach.ai.ppo.ppo_trainer import PPOTrainer


class TestCheckpointCountRestore:
    def test_checkpoint_count_restored_on_resume(self, tmp_path):
        trainer = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        trainer._save_checkpoint(step=100)
        trainer._save_checkpoint(step=200)
        trainer._save_checkpoint(step=300)
        assert trainer._checkpoint_count == 3
        assert (tmp_path / "checkpoint3.pt").exists()

        resumed = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        resumed.load_checkpoint(tmp_path / "checkpoint3.pt")
        assert resumed._checkpoint_count == 3
        assert resumed._total_steps == 300

    def test_next_save_after_resume_continues_numbering(self, tmp_path):
        trainer = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        trainer._save_checkpoint(step=100)
        trainer._save_checkpoint(step=200)

        resumed = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        resumed.load_checkpoint(tmp_path / "checkpoint2.pt")
        resumed._save_checkpoint(step=300)

        # Must continue at checkpoint3.pt, not restart at checkpoint1.pt
        # (which would silently overwrite the real checkpoint1.pt from the
        # original trainer above).
        assert (tmp_path / "checkpoint3.pt").exists()
        original_checkpoint1 = torch.load(
            tmp_path / "checkpoint1.pt", map_location="cpu", weights_only=False
        )
        assert original_checkpoint1["step"] == 100

    def test_log_file_rotates_to_correct_number_after_resume(self, tmp_path):
        trainer = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        trainer._save_checkpoint(step=100)
        trainer._save_checkpoint(step=200)

        resumed = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        resumed.load_checkpoint(tmp_path / "checkpoint2.pt")

        # Without the _rotate_log_file() re-call fix, this would still point
        # at training_log1.txt (opened by __init__ with count=0, before
        # load_checkpoint ran) instead of training_log3.txt.
        assert resumed._log_file_handler is not None
        assert resumed._log_file_handler.baseFilename.endswith("training_log3.txt")

    def test_stale_embedded_checkpoint_count_corrected_from_disk(self, tmp_path):
        """Simulates resuming from a checkpoint whose own checkpoint_count is
        itself wrong (e.g. produced by an earlier buggy resume) -- the real
        history on disk (checkpoint1..5.pt) must win over the stale embedded
        value (2), not just be trusted at face value."""
        trainer = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        for step in (100, 200, 300, 400, 500):
            trainer._save_checkpoint(step=step)
        assert (tmp_path / "checkpoint5.pt").exists()

        # Hand-corrupt a copy to mimic a stale embedded count, as "latest.pt"
        # (genuinely-progressed weights, but inherited-bad bookkeeping).
        stale = torch.load(tmp_path / "checkpoint5.pt", map_location="cpu", weights_only=False)
        stale["checkpoint_count"] = 2
        stale_path = tmp_path / "latest.pt"
        torch.save(stale, stale_path)

        resumed = PPOTrainer.from_config(checkpoint_dir=tmp_path)
        resumed.load_checkpoint(stale_path)

        assert resumed._checkpoint_count == 5  # from disk, not the stale embedded 2
        assert resumed._total_steps == 500  # weights/step still trusted from the file

        resumed._save_checkpoint(step=600)
        assert (tmp_path / "checkpoint6.pt").exists()
        # checkpoint3.pt/4.pt (real history between the stale count and the
        # true on-disk max) must survive untouched.
        untouched = torch.load(tmp_path / "checkpoint3.pt", map_location="cpu", weights_only=False)
        assert untouched["step"] == 300
