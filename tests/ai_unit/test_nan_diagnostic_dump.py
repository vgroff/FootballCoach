"""_dump_nan_diagnostic() writes into the run's checkpoint dir and halts training."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from footballcoach.ai.ppo.ppo_trainer import PPOTrainer


@dataclass
class _Heads:
    logit: torch.Tensor


def _dump(checkpoint_dir) -> Path:
    heads = _Heads(torch.tensor([[1.0], [float("nan")]]))
    with pytest.raises(PPOTrainer._NaNReproFound) as exc:
        PPOTrainer._dump_nan_diagnostic(
            SimpleNamespace(checkpoint_dir=checkpoint_dir),
            stage="forward", epoch_i=1, mb_i=2, mb_idx=torch.tensor([0, 1]),
            track_ids_mb=["trainee", "secondary"], d_heads=heads, e_heads=heads,
            policy_loss=torch.tensor(float("nan")),
            value_loss=torch.tensor(0.5), entropy=torch.tensor(0.1),
        )
    return Path(exc.value.args[0])


def test_written_into_the_checkpoint_dir(tmp_path):
    out = _dump(tmp_path)
    assert out.parent == tmp_path and out.name.startswith("nan_diagnostic_")
    report = json.loads(out.read_text())
    assert report["policy_loss_finite"] is False and report["value_loss_finite"] is True
    assert report["d_heads"]["logit"]["n_nonfinite"] == 1
    assert report["d_heads"]["logit"]["n_nonfinite_secondary"] == 1


def test_missing_checkpoint_dir_is_created(tmp_path):
    assert _dump(tmp_path / "not" / "yet").is_file()


def test_no_checkpoint_dir_falls_back_to_the_working_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (tmp_path / _dump(None).name).is_file()
