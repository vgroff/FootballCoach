# Checkpoint archive: phase1_run321, checkpoint 763

A backup in git of the latest trained checkpoint (`checkpoints/` is gitignored,
so this is the only copy outside that folder). Copied 2026-11-16; every file
was checked byte-identical (md5) to its original, and the archived migrated
checkpoint was loaded with the archived physics encoders and gives bit-identical
outputs to the original.

| File | Original location | md5 |
|---|---|---|
| `checkpoint763.pt` | `checkpoints/phase1_run321/checkpoint763.pt` (= that run's `latest.pt`) | `57965e8f7eddf9770263309c118b6861` |
| `checkpoint763_migrated.pt` | `checkpoints/migrated/phase1_run321_checkpoint763_migrated.pt` | `0421499729b86af522f3c7b9721f4652` |
| `run321_ai_config.json` | `checkpoints/phase1_run321/ai_config.json` (the run's pinned config) | `f81728bc7f96a746495a53165205ad05` |
| `physics_pretrain/ball_encoder_75.midtrain_latest_train.pt` | `checkpoints/physics_pretrain/` | `115d7dfd5a7d37b0978093ae16d7387b` |
| `physics_pretrain/player_encoder_50.midtrain_latest_train.pt` | `checkpoints/physics_pretrain/` | `99f534886ca0ac83fde75117f52cc803` |

- **`checkpoint763.pt`**: as trained (pre-migration), with its optimizer state.
  Trained on the old input conventions (no task ids in its rows, the old
  time-remaining clock), so training refuses it until it is migrated.
- **`checkpoint763_migrated.pt`**: made by `scripts/migrate_checkpoint_inputs.py`
  (commit `6ddd263` and later): time-remaining remap folded in, task-id inputs of
  phases 1 and 2 and `is_goalkeeper` zeroed. No optimizer state (Adam starts
  fresh). The critic-only elapsed-fraction column is added as zeros when loaded
  by current code. Train from this one, without `--init-new-phase-inputs` /
  `--remap-time-input`.
- **The physics encoders**: the decision network's saved weights do NOT
  include its ball / player physics encoders; they are loaded from the files
  named by `network.ball_physics_encoder_checkpoint` /
  `network.player_physics_encoder_checkpoint` in ai_config.json. Without these
  two files neither checkpoint can be rebuilt.

To restore into `checkpoints/`: copy the files back to their original locations
in the table (copy, do not move), then check the md5s.
