# Overnight 2026-11-20: getting the neural keeper to learn (option 1)

User's call (before bed): "Do option 1 until tomorrow morning ... just keep
trying to get the neural gk to learn". Option 1 = PPG value refit, then PPO,
starting from the keeper pretrain run40 (role-BC label-frame fix 6dcbec1).

Starting point (run40 final, see overnight_2026-11-19.md): keeper gates
learned from the rules keeper (exec_move / dive / tackle / move / get
possession agreement 61-95%), move direction not learned. Keeper evals
(115 x 4): vs rules attacker 44.1% conceded (rules keeper 7%), vs AI
attacker 25.2% (rules keeper 62%).

## Log

- ~01:5x: PPG value refit started: --ppg-refit-only --ppg-num-rollouts 6,
  --phase-mix "1:0.5,2:0.5", from checkpoints/phase2_run40/checkpoint_pretrained.pt
  (log scratchpad/ppg_refit_run41.log). Config unchanged: phase-2 controllers
  attacker rules:neural 1:4, defender 3:7, keeper 1:1; keeper save-target
  shaping 0.1; PPO n_epochs 10, rollout 500k, target_kl 0.5, replay 0.35
  with fresh engine dice; evals fully seeded.
- 02:34: PPG refit done -> checkpoints/phase2_run41/checkpoint_pretrained.pt
  (40 min, 6 cycles). Value MSE/var per cycle 0.40 -> 0.50 is the replayed
  hardest episodes (the ORIGINAL net's MSE on the same rows rose too:
  0.40 -> 0.51); the refit beats the original by a little each cycle
  (0.500 vs 0.509 on the last). Policy KL <= 0.007.
- 02:35: PPO started -> checkpoints/phase2_run42 (log scratchpad/ppo_run42.log),
  --checkpoint checkpoints/phase2_run41/checkpoint_pretrained.pt, config as above.
