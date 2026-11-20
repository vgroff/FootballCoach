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
