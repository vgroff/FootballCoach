"""plot_training_curves.py run the documented way, on a small synthetic run."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ITERATION = """\
[PPO] step={step:,}  speed=9229/s  reward=2.02/-1.66
  loss     policy=-0.0056  value=0.0952(x3.0)=0.2857  val_pre=0.0980
           ent_coef=0.0030  entropy=-1.7309  kl=0.0248
  value    V=0.374±1.984  R=0.373±2.083  adv=-0.001±0.652  |adv|=0.340
  entropy  move=0.0000(+0.0000)  gp_extra=0.0000(+0.0000)  exec_move=0.0241(+0.0008)  kick=0.0040(-0.0002)  tackle_attempt=0.0000(-0.0000)  sprint=0.0649(+0.0016)
           move_dir=-1.7649(+0.0083)  kick_dir=-0.0427(+0.0007)  kick_dir_azimuth=-0.0172(+0.0003)  kick_dir_z=-0.0255(+0.0004)  kick_power=-0.0163(+0.0003)
  moves    mv_ls=[6.9079] (κ≈1000.11, ang std≈2°) g=0.00e+00  d_move=[+0.0000] (Δ(ang std)≈0.000°)
           kk_ls=[4.7053] (κ≈110.54, ang std≈5°)  d_kick=[-0.0002] (Δ(ang std)≈0.001°)
           kz_ls=[-2.8007] (σ≈0.0608, ang std≈3°)  d_kickz=[+0.0007] (Δ(ang std)≈0.003°)
           kp_ls=[-2.3029] (σ≈0.1000)  d_kickpower=[+0.0009] (Δσ≈0.0001)  | sampled power over 79,622 kicks: mean=0.405 std=0.197
  heads    move= 28 get_poss= 79 exec_move= 87 sprint= 86 kick= 11 tackle=100 shoot=  0 hold=
           0 tackle_prob=0.9985 kick_prob=0.1046
  kicks    0.804±1.237/ep (trainee, all real kicks)  first-touch(armed)=0.306/ep (38% of kicks)  0-kick eps=60%  max=10  (n=10106)
  [per-head KL] move=+0.0002  gp_extra=+0.0001  exec_move=+0.0006  sprint=+0.0006  kick=+0.0007  tackle_attempt=+0.0004  move_dir=+0.0222  kick_dir=+0.0174  kick_power=+0.0310
[eval vs rules] step={step:,} n=100 win=66%  mean_rew=1.300±0.500  (sem=0.050)
[eval vs neural:original] step={step:,} n=100 win=48%  mean_rew=0.300±0.500  (sem=0.050)
[eval baseline rules-vs-rules] n=100 win=46%  mean_rew=0.313±0.500  (sem=0.050)
"""

_STEPS = [100_000_000 + i * 50_000_000 for i in range(8)]  # only log 1 falls outside the 300M window
_PPG_LOGS = {1, 2, 3, 7}


def _run(run_dir: Path, out_prefix: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "footballcoach.ai.scripts.plot_training_curves", str(run_dir), str(out_prefix)],
        capture_output=True, encoding="utf-8", timeout=120,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )


@pytest.fixture
def result(tmp_path):
    run_dir = tmp_path / "phase1_run1"
    run_dir.mkdir()
    for i, step in enumerate(_STEPS):
        text = _ITERATION.format(step=step)
        if i + 1 in _PPG_LOGS:
            text += "[ppg interleave] triggered\n"
        (run_dir / f"training_log{i + 1}.txt").write_text(text, encoding="utf-8")
    return _run(run_dir, tmp_path / "out"), tmp_path


def test_writes_all_four_figures(result):
    proc, tmp_path = result
    assert proc.returncode == 0, proc.stderr
    for suffix in ("eval", "value", "kick", "entropy_kl"):
        assert (tmp_path / f"out_{suffix}.png").stat().st_size > 0


def test_ppg_rounds_are_counted_inside_the_window_only(result):
    proc, _ = result
    assert "7 iterations (last 300M steps)" in proc.stdout
    assert "3 PPG interleave round(s) in window" in proc.stdout


def test_parsed_values_reach_the_summary(result):
    proc, _ = result
    assert "reward first5 +1.300" in proc.stdout
    assert "kick_dir_z" in proc.stdout and "KL not logged for this head" in proc.stdout


def test_a_run_with_no_iterations_exits_with_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _run(empty, tmp_path / "out")
    assert proc.returncode == 1 and "No PPO iterations found." in proc.stdout
