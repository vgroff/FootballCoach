"""
Diagnostic 1: Does BC pre-training actually produce a network that copies the rules-based AI?

Runs 50 episodes using the BC-pretrained network, and separately using the raw
rules-based AI labels, then compares actions side-by-side.

Run: uv run python debug_bc_fidelity.py
"""
import torch
import numpy as np
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.bc import phase1_labels, BCPretrainer
from footballcoach.ai.config import load_ai_config
from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition

torch.manual_seed(0)

cfg = load_ai_config()
bc_cfg = cfg.get("bc", {})
pretrain_steps = int(bc_cfg.get("pretrain_steps", 2000))

defn = ScenarioDefinition(key="dbg", label="dbg", description="dbg", build=build_1v1_scenario)
env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=30.0)
trainer = PPOTrainer.from_config()

print(f"=== BC PRE-TRAINING ({pretrain_steps} steps) ===")
pretrainer = BCPretrainer(trainer.decision_net, trainer.execution_net, cfg, torch.device("cpu"))
pretrainer.pretrain(env, n_steps=pretrain_steps, label_fn=phase1_labels)

print("\n=== FIDELITY CHECK: network vs rules-based labels over 200 steps ===")

obs = env.reset()
agree_move = 0
agree_gp = 0
agree_sprint = 0
n_valid = 0
move_dir_cos_sims = []

for i in range(200):
    obs_dict = obs.to_torch_dict()

    # Get BC label (what rules-based AI would do)
    label = phase1_labels(env)

    # Get network prediction
    result = trainer._sample_action(obs_dict)
    action = result[0]
    exec_phys = result[4]

    if label.valid:
        n_valid += 1
        net_move = action.move > 0.5
        net_gp   = action.get_possession_extra > 0.5
        net_sprint = exec_phys["sprint"]

        lbl_move   = label.move > 0.5
        lbl_gp     = label.get_possession_extra > 0.5
        lbl_sprint = label.sprint > 0.5

        if net_move == lbl_move:   agree_move += 1
        if net_gp == lbl_gp:       agree_gp += 1
        if net_sprint == lbl_sprint: agree_sprint += 1

        # Direction agreement (cosine similarity of move_direction)
        if label.move_direction is not None:
            net_dir = exec_phys["move_direction"]
            lbl_dir = label.move_direction
            cos = float(np.dot(net_dir, lbl_dir) /
                        (np.linalg.norm(net_dir) * np.linalg.norm(lbl_dir) + 1e-8))
            move_dir_cos_sims.append(cos)

    # Step env
    env_action = {
        "decision_probs": result[3],
        "execution_physical": result[4],
        "decision_physical": result[5],
        "target_slots": result[6],
        "slot_player_ids": [None] * 21,
        "decision": action,
        "execution": __import__("footballcoach.ai.action.schema",
                                fromlist=["ExecutionAction"]).ExecutionAction(),
    }
    next_obs, reward, done, _ = env.step(env_action)
    obs = env.reset() if done else next_obs

print(f"  Valid label steps: {n_valid}/200")
if n_valid > 0:
    print(f"  move agreement:    {agree_move}/{n_valid} = {agree_move/n_valid*100:.1f}%")
    print(f"  gp agreement:      {agree_gp}/{n_valid} = {agree_gp/n_valid*100:.1f}%")
    print(f"  sprint agreement:  {agree_sprint}/{n_valid} = {agree_sprint/n_valid*100:.1f}%")
if move_dir_cos_sims:
    print(f"  move_dir cosine:   mean={np.mean(move_dir_cos_sims):.3f}  "
          f"min={np.min(move_dir_cos_sims):.3f}  "
          f"(1.0=perfect, -1=opposite)")
