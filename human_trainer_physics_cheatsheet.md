# Physics-pretrain cheatsheet (Windows / PowerShell)

Quick-reference commands for the ball/player dynamics offline pretraining
pipelines (`src/footballcoach/ai/physics_pretrain/`). All commands assume
you're in the repo root in PowerShell.

For multi-line PowerShell, use a backtick `` ` `` at the end of each line
(not `\` — that's bash syntax and silently breaks the line join in
PowerShell).

## Ball

**1. Generate dataset** (regenerate whenever engine/physics code changes):

```powershell
uv run python -m footballcoach.ai.physics_pretrain.ball_dataset --output physics_pretrain_data/ball --n-episodes 200000 --shard-size 10000 --n-workers 8
```

**2. Train:**

```powershell
uv run python -m footballcoach.ai.physics_pretrain.train_ball_dynamics --dataset physics_pretrain_data/ball --output checkpoints/physics_pretrain/ball_encoder.pt
```

**3. Visualiser** (ground-truth vs predicted trajectory/crossing/resting on a
pitch, per random episode — click "Next" or press n/space/right-arrow for a
new one; closing the window runs an error/input-correlation analysis in the
terminal):

```powershell
uv run python scripts/inspect_ball_pretrain.py --checkpoint checkpoints/physics_pretrain/ball_encoder.midtrain_latest.pt --dataset physics_pretrain_data/ball
```

Needs a **phase** checkpoint (`.midtrain_latest.pt`, `.after_training.pt`,
etc. — has `model_state_dict`), not the final `--output` artifact (encoder
weights only, decoder discarded).

## Player

**1. Generate dataset:**

```powershell
uv run python -m footballcoach.ai.physics_pretrain.player_dataset --output physics_pretrain_data/player --n-episodes 200000 --shard-size 1000 --n-workers 8
```

**2. Train:**

```powershell
uv run python -m footballcoach.ai.physics_pretrain.train_player_dynamics --dataset physics_pretrain_data/player --output checkpoints/physics_pretrain/player_encoder.pt
```

**3. Visualiser** (same idea as the ball one — trajectory + heading ticks +
stamina-over-horizon + crossing panel; no resting panel, players don't rest
like the ball does):

```powershell
uv run python scripts/inspect_player_pretrain.py --checkpoint checkpoints/physics_pretrain/player_encoder.midtrain_latest.pt --dataset physics_pretrain_data/player
```

## Common flags (both `train_*_dynamics.py` scripts)

- `--epochs` / `--batch-size` / `--lr` — default to `ai_config.json`'s
  `physics_pretrain.<ball|player>` values if omitted.
- `--val-frac` (default 0.15), `--seed` (default 0).
- `--device` — auto-picks `cuda` if available, else `cpu`.
- `--no-open-report` — suppress auto-opening the HTML report when training
  finishes (default opens it).
- `--init-checkpoint <path>` — warm-start from an existing checkpoint.
  **Skip this after an engine physics change** — the old weights were primed
  on stale dynamics, so training from scratch is cleaner.
- `--reset-optimizer-state` — only relevant with `--init-checkpoint`, when
  you've also changed a loss weight since that checkpoint was saved (Adam's
  stale `exp_avg_sq` otherwise throttles the effective step size for a long
  time post-resume).
- `--max-episodes N` — subsample the dataset for a quick sanity run before
  committing to the full one.
- `--linear-decoder` — force `linear_decoder_enabled=true` for this run
  regardless of config (no CLI way to force it back off).

Both `inspect_*_pretrain.py` scripts also take `--seed` (reproducible row
sequence) and `--linear-decoder` (force-build the linear decoder if a
checkpoint's `config_snapshot` is stale). The ball one additionally takes
`--alpha` (marker/line transparency) and `--error-samples` (episode count
for the on-close correlation analysis).

## Report file

Written once, at the very end of training (not periodically mid-run), to
`<output path>.report.html` — e.g. `--output
checkpoints/physics_pretrain/player_encoder.pt` writes
`checkpoints/physics_pretrain/player_encoder.report.html`. Auto-opens in
the browser unless `--no-open-report` was passed; reopen anytime with:

```powershell
start checkpoints/physics_pretrain/player_encoder.report.html
```

## linear_decoder_enabled + --init-checkpoint gotcha

The **inspector scripts** auto-detect this from the checkpoint's own
`config_snapshot['linear_decoder_enabled']` — no action needed, `--linear-
decoder` is only for overriding a stale snapshot.

**Training (`--init-checkpoint`) does NOT auto-detect it.** The model is
built from your CURRENT `ai_config.json` (or `--linear-decoder`), never
from the checkpoint's own setting. If it doesn't match what the checkpoint
was trained with, the encoder still resumes fine, but the decoder's
parameter names won't line up (`decoder.net.weight` vs `decoder.net.0.
weight`/`decoder.net.2.weight`) — `strict=False` loading silently drops it
to a fresh random init instead of erroring. Watch the log for `Checkpoint
missing N param(s)... left at fresh init` / `unexpected param(s), ignored`
right after `Resumed full model...` — that's the tell. Fix: make sure
`linear_decoder_enabled` in your live config matches whatever the
checkpoint you're resuming from was actually trained with.

## label_smoothing

Set via `ai_config.json`'s `physics_pretrain.ball.label_smoothing` /
`physics_pretrain.player.label_smoothing` — no CLI flag. `0.0` = off.
Diagnostics (console log, `.history.npz`, the HTML report) automatically
subtract the analytic smoothing floor, so the reported oob_bce/goal_bce/
crossing_crosses_loss/event_loss numbers stay comparable to an unsmoothed
run.
