# Human Trainer Cheatsheet

## Demonstrations

Demonstrations are used for BC pre-training (offline). Re-record whenever the reward
function, scenario config, or rules AI behaviour changes significantly.

### Re-record demonstrations (delete old + record fresh)
```bash
# Delete existing demos and record fresh ones (adjust --n-episodes as needed)
rm -f demonstrations/phase1/*.npz && \
uv run python -m footballcoach.ai.scripts.record_demonstrations \
    --phase 1 --n-episodes 6000 --episodes-per-file 8 \
    --output demonstrations/phase1/ --seed 42 >> /tmp/record_demos.log 2>&1
# tail -f /tmp/record_demos.log  # in another terminal to watch progress
```

| Flag | Effect |
|------|--------|
| `--n-episodes N` | Total episodes to record. 4000 ≈ 44k steps at 1s interval. 1000 = fast check (~11k steps). |
| `--episodes-per-file N` | Episodes per .npz file (default 8). Lower = more files, finer resume granularity. |
| `--output PATH` | Output directory (auto-created). |
| `--seed N` | Random seed for reproducibility. |
| `--info` | Print summary of existing .npz files in directory and exit (no recording). |
| `--opponent-rules-prob F` | Fraction of episodes where opponent is rules-based (default from config). |

### Inspect existing demonstrations
```bash
uv run python -m footballcoach.ai.scripts.record_demonstrations \
    --phase 1 --n-episodes 5000 --output demonstrations/phase1/ --info
```

---

## Core training commands

All commands append to `training_runs.log` via direct redirection (`>>`), not `tee`,
so Ctrl+C isn't swallowed by a pipe (see note below). Use `tail -f training_runs.log`
in another terminal to watch live progress.

### Default behaviour (no checkpoint flags)

With **no** `--checkpoint`, `--from-pretrained`, or `--pretrain-from-checkpoint` flag:
- Starts from a **fresh, randomly-initialised** network.
- Runs **full BC pre-training** (offline dataset if `--bc-dataset` is given, otherwise online rollout BC for `pretrain_steps` from ai_config.json).
- Then runs PPO for `--total-steps` (default: 500,000).
- Auto-creates a new `checkpoints/phase1_runN/` dir.

To continue from the most recent run without re-pretraining, use `--from-pretrained`
(see below). There is no "load latest and skip pretraining" automatic behaviour —
you must pass the checkpoint path explicitly.

### Fresh run (full BC pretraining from scratch)
```bash
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --seed 42 \
    --bc-dataset demonstrations/phase1/ \
    --verbose --total-steps 40000 2>&1 | tee -a training_runs.md
```

### Resume from the latest checkpoint automatically
```bash
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --seed 42 \
    --latest \
    --verbose --total-steps 100000 2>&1 | tee -a training_runs.md
```
Finds the most recent `latest.pt` (or highest-numbered checkpoint) across all
`checkpoints/phase1_run*/` dirs. Skips pretraining and continues the step counter.
Equivalent to `--checkpoint <that file>` but without needing to know the path.


### Continue from a pretrained checkpoint — skip BC pretraining entirely
```bash
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --seed 42 \
    --from-pretrained checkpoints/phase1_run35/checkpoint_pretrained.pt \
    --verbose --total-steps 40000 2>&1 | tee -a training_runs.md
# --total-steps 40000
```

### Re-run BC pretraining on an existing checkpoint, then PPO
```bash
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --seed 42 \
    --pretrain-from-checkpoint checkpoints/phase1_run41/checkpoint_00065000.pt \
    --bc-dataset demonstrations/phase1/ \
    --verbose --total-steps 40000 2>&1 | tee -a training_runs.md
# --total-steps 60000
```

### Re-run BC pretraining on the LATEST checkpoint automatically, then PPO
```bash
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --seed 42 \
    --latest-pretrain \
    --bc-dataset demonstrations/phase1/ \
    --verbose --total-steps 40000 2>&1 | tee -a training_runs.md
```
Finds the most recent checkpoint across all `checkpoints/phase{N}_run*/` dirs
(same resolution as `--latest`), loads its weights, then runs the full BC/value
pre-training loop again (equivalent to `--latest` + `--pretrain-from-checkpoint`
combined). **Resets the step counter to 0** (unlike `--latest`, which continues
it) since pretraining is being redone. Requires `--bc-dataset`.


### Resume a specific mid-run checkpoint (skips pretraining, continues step count)
```bash
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --seed 42 \
    --checkpoint checkpoints/phase1_run39/checkpoint_00016192.pt \
    --verbose >> training_runs.log 2>&1
# --total-steps 60000
```

### Checkpoint dir is auto-generated as `checkpoints/phase1_runN/` (next available N). Override with `--checkpoint-dir checkpoints/my_run/`.

---

## Useful flags

| Flag | Effect |
|------|--------|
| `--total-steps N` | PPO decision-steps to train (excl. pretraining). Try 20k–100k. |
| `--verbose` | Per-minibatch KL, ratio, and per-head log_prob diagnostics |
| `--seed 42` | Reproducibility |
| `--no-bc-aux` | Disable BC auxiliary loss during PPO (pure RL from step 1) |
| `--bc-pretrain-steps 0` | Skip online BC pretraining (only relevant without `--bc-dataset`) |
| `--no-head-freeze` | Train all decision-network heads during PPO (default freezes most) |
| `--pre-ppo-eval-trials 0` | Skip the 40-trial pre-PPO eval (saves ~30s) |
| `--device cuda` | GPU if available |

---

## What to watch in the log

### Per-rollout line
```
step=8,096 | rew=11.63/8.95 | pol=0.07 val=2.18 ent=0.52 kl=0.34 bc=2.3(x0.38) | 272sps
  act: mv=25 gp=75 emv=100 spr=80 kck=0 tk=0 sh=0 hld=0  vs_neural(148): 55%/45%
  [V=1.2\u00b10.4 R=1.3\u00b10.5 adv=0.0\u00b11.0]
```
- **rew** — mean episode reward / mean episode length in steps. Should climb over time.
- **[V=/R=/adv=]** — per-rollout value/return/advantage mean\u00b1std, from `execution_net.value_head` (the single trained critic — see "Single value head convention" in `src/footballcoach/ai/knowledge.md`). `V` and `R` should track each other reasonably closely once the critic is calibrated; a persistently large gap means the value head needs more pretraining epochs. DEBUG-level logs also show a per-minibatch d_val/e_val split — `d_val` is static since `decision_net.value_head` is frozen.
- **kl** — mean KL across the rollout. Should stay below `target_kl` (currently 0.4). Consistently near the ceiling = updates are too large.
- **bc(xN)** — BC aux loss × current anneal coefficient. Should drop to 0 by `aux_coeff_anneal_fraction` of training (currently 65%).
- **sps** — steps per second. ~280 is normal on CPU.
- **kck=0** — no kicks yet. Expected early; becomes non-zero once PPO learns to approach the ball.
- **vs_neural N%/N%** — trainee/opponent win rate in neural episodes. If both stay near 50/50 and don't drift, self-play is stable.

### Early-stop messages
```
[early stop e0 mb5]  KL=0.40190 > target=0.4  steps_this_update=6
```
- **2–6 minibatches through** = policy is moving too fast per gradient step. Lower `learning_rate` or `clip_range` in `ai_config.json`.
- **25+ minibatches through** (as in run39 rollout 1) = update is conservative, learning is proceeding healthily.
- If early stop fires at mb0 or mb1 every single rollout, the policy is collapsing.

### KL diagnostic block (printed when KL > 0.1)
```
[KL=0.34 > 0.1] ratio percentiles:  p5=0.38  p25=0.55  p50=0.76  p75=1.00  p95=1.00
```
- `p5` should be > 0.3 (policy isn't flipping wholesale).
- `p95` should be < 1.1 ideally; consistently 1.0 or above means the ratio is clipping.
- If `move_dir` log_std stays stuck at `[-1.0, -1.0]`, the execution network hasn't learned to narrow its direction distribution yet — normal early on.

### Warning signs
- **rew trending down** across rollouts → policy is regressing; load an earlier checkpoint.
- **kck=0 for 50k+ steps** → the agent is getting possession but not attempting kicks; check `get_possession` reward vs `ball_progress` reward balance.
- **vs_neural oscillates wildly** (e.g. 80%/20% then 20%/80% every rollout) → self-play cycling; add some rules-based opponent episodes via `phase1_opponent_rules_prob` in `ai_config.json`.
- **BC loss not dropping in pretraining** (`dir_cos` stuck < 0.9) → direction head isn't learning; raise `direction_loss_weight` in `bc` config.
- **val_loss RMSE >> returns std** after value pretraining → value head is poorly calibrated; increase `demo_value_pretrain_epochs`.
- **Baseline rules vs rules ≠ 50/50** → scenario is asymmetric (ball closer to trainee). The current config gives trainee ~100% baseline win — factor this in when reading neural vs rules win rates.

---

## Hyperparameter reference (`src/footballcoach/ai/config/ai_config.json`)

### PPO core

**`gamma` = 0.995** — Discount factor. How much a reward N steps in the future is worth: `gamma^N × face_value`. At 0.995, a reward 170 steps away (full episode) is still worth `0.995^170 ≈ 43%`. Use 0.99 for shorter episodes or if training is unstable; 0.999 if the agent seems short-sighted about long sequences.

**`lam` = 0.97** — GAE lambda. Balances bias vs variance in advantage estimates. Think of it as a mixing knob: `lam=1.0` = pure Monte Carlo (correct but noisy), `lam=0.0` = pure one-step TD (smooth but biased). 0.97 is a safe default. Go lower (0.9) if gradients are chaotic; higher (0.99) if advantage estimates seem too short-horizon.

**`clip_range` = 0.05** — PPO clipping. The ratio `new_prob / old_prob` is forced into the window `[0.95, 1.05]`. Any update that tries to change the policy more than 5% on a given sample is silently ignored. Standard is 0.2; we're at 0.05 because the KL was exploding with larger values. If 25+ minibatches are getting through comfortably, try 0.1.

**`target_kl` = 0.4** — Per-minibatch early-stop. KL divergence roughly measures "how different is the new policy from the one that collected this data?" We stop the epoch loop the moment any minibatch hits this threshold. 0.4 is very permissive (the real guard is `clip_range`). Lower to 0.1 to make training more conservative; useful if you see the policy regressing between rollouts.

**`learning_rate` = 5e-6** — Adam step size. Tiny compared to typical RL (3e-4) because: (a) 12× augmentation means each gradient step already sees 12× the data, and (b) BC pretraining puts us near a good starting point — large steps would undo it. Try 1e-5 if training feels glacially slow; 1e-6 if the policy keeps collapsing after pretraining.

**`ent_coef` = 0.1** — Entropy bonus coefficient. Adds `ent_coef × entropy` to the reward signal, incentivising exploration. High (0.1) early on to stop the agent collapsing to one action. Reduce to 0.01–0.02 once the policy has converged to sensible behaviour, or if the agent is still taking random actions after 50k steps.

**`ent_dir_weight` = 0.05** — Scales how much the continuous direction heads (move_dir, kick_dir) contribute to the entropy bonus. A Gaussian direction head's raw entropy is ~10× bigger than a Bernoulli's, so without downweighting it would drown out the binary heads (sprint, kick, tackle). At 0.05 the Bernoulli heads drive exploration. Raise toward 0.2 if kicks always go the same direction; lower toward 0.01 if the agent spins randomly.

**`max_grad_norm` = 0.5** — Gradient clipping. The entire gradient vector is scaled down if its L2 norm exceeds this. Safety net against catastrophic single-batch updates. Check `g=` in the rollout log: if it's pinned at exactly 0.5 every step, you're clipping every update — lower `learning_rate` instead. Raise to 1.0 if gradients feel overly constrained.

**`vf_coef` = 0.5** — How much the value loss contributes to the total loss vs the policy loss. Raise if value estimates are badly calibrated (RMSE much larger than returns std); lower (rarely needed) if you suspect the critic's gradients are destabilising the actor.

**`rollout_steps` = 8096** — Steps collected before each PPO update. Larger = more stable gradients, less frequent updates. With 12× augmentation this gives 97,152 effective samples per update. Lower to 4096 for faster iteration; raise to 16384 for more stable but slower learning.

**`n_epochs` = 4** — Maximum passes over the rollout data per update. In practice `target_kl` early-stops this to 1–2 passes currently. If early-stop fires after mb25+ (good), all 4 epochs may complete.

**`augment_n_slot_shuffles` = 3** — Slot permutations per flip variant = 3 → 4 flips × 3 shuffles = **12× batch expansion**. Set to 1 for 4× (flips only, faster); 0 to disable augmentation entirely (useful for debugging).

---

### Behavioural Cloning (`bc` section)

**`bc_pretrain_epochs` = 5** — How many full passes over the demonstration dataset during offline pretraining. More epochs = better BC fit, longer pretraining (~90s per epoch). Raise to 8–10 if `dir_cos` is still < 0.95 after pretraining.

**`aux_coeff_start` = 1.0 → `aux_coeff_end` = 0.0 over `aux_coeff_anneal_fraction` = 0.65** — During PPO, a BC loss is added to each update, weighted by a coefficient that starts at 1.0 and linearly decays to 0 by 65% of training. This keeps the policy near rules-AI behaviour early on while still allowing RL to take over. Raise `aux_coeff_start` to 2.0 if the policy drifts badly from BC early on; lower `anneal_fraction` to 0.3 to let RL take over sooner.

**`direction_loss_weight` = 4.0** — Multiplier on the move_direction cosine loss during BC. Needed because one continuous head competes with ~11 binary BCE heads. Raise if `dir_cos` isn't improving in pretraining logs.

**`demo_value_pretrain_epochs` = 1** — Epochs spent fitting the value head to demonstration returns before PPO starts. More = better critic initialisation = more stable early PPO. Currently 1 (fast); raise to 5 if `val_loss RMSE >> returns std` at PPO step 1.

---

### Curriculum / scenario (`curriculum` + `phase1_scenario`)

**`phase1_opponent_rules_prob` = 0.0** — Fraction of episodes where the opponent is rules-based. Currently 0 = pure self-play. Set to 0.3–0.5 if self-play is cycling (vs_neural oscillates wildly) — a stable rules-based opponent anchors training.

**`phase1_opponent_immobile_prob` = 0.0** — Fraction of episodes with a stationary opponent. Useful early in training when the agent can't beat anyone. The remaining fraction is neural self-play.

**`secondary_weight` = 1.0** — How much opponent (secondary player) transitions contribute to the PPO loss. 1.0 = equal to trainee transitions. Lower to 0.5 if you think the opponent perspective is hurting the trainee's learning; 0.0 to train on trainee only.

**`ball_max_dist_from_trainee_m` = 20.0** — Ball is placed within this distance of the trainee at episode start. Lower = easier (ball always nearby); higher = harder (trainee must search). Currently 20m — this is why baseline is ~100% trainee win.

**`opponent_min_dist_m` / `opponent_max_dist_m` = 13–50** — Opponent is placed between 13m and 50m from the trainee. Stops degenerate starts where they overlap or are instantly adjacent. Tighten the range (e.g. 10–20) to make training harder; set both to defaults (0 / 9999) for fully random placement.

**`trainee_tier` = `premier_league`, `opponent_tier` = `generic`** — Attribute tiers for generated player stats. `premier_league` trainee = faster, better control; `generic` opponent = weaker. Makes the task learnable early. Set both to `premier_league` for a harder, more realistic challenge.

---

### Reward shaping (`reward.phase1`)

| Parameter | Current | Effect |
|-----------|---------|--------|
| `ball_distance_shaping` | 0.05 | Small per-step bonus for closing distance to ball. Too high = agent circles the ball. |
| `gain_possession_bonus` | 1.0 | Awarded per **real turnover** gained (possession settling onto the trainee from a DIFFERENT player, e.g. a tackle or interception) within a decision interval — counted as an int, not just a one-off bool, so multiple gains in one interval each pay out. A "kick to yourself" (ball briefly loose/in-flight, e.g. during a push-kick dribble touch, then re-collected by the SAME player) does NOT count — see `src/footballcoach/ai/knowledge.md` "Possession gain/loss reward: real turnovers only". `loss_of_possession_penalty` mirrors this on the loss side. |
| `ball_progress_scale` | 0.5 | Per-step bonus × metres advanced toward opponent goal while in possession. Main driver of forward play. |
| `ball_out_penalty` | -1.0 | Penalises kicking out of bounds. |
| `illegal_action_penalty` | -0.2 | Penalises actions that aren't legal (e.g. kicking without possession). |
| `box_possession_terminal` | 5.0 | Episode-ending bonus for reaching the opponent box with the ball. Main terminal signal — **don't lower this**. |
