"""
Quick debug script: sample one action, store it, recompute log_prob,
print every component to understand the KL explosion.
"""
import torch
import numpy as np
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.action.distributions import IndependentBernoulli, DirectionHead
from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition

torch.manual_seed(42)

defn = ScenarioDefinition(key="dbg", label="dbg", description="dbg", build=build_1v1_scenario)
env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=30.0)
trainer = PPOTrainer.from_config()

obs = env.reset()
obs_dict = obs.to_torch_dict()

# --- Sample ---
result = trainer._sample_action(obs_dict)
action, log_prob_old, value, decision_probs, exec_phys, dec_phys, target_slots, raw_exec = result

print(f"\n=== SAMPLED ACTION ===")
print(f"log_prob_old (full, at collection time): {log_prob_old:.6f}")
print(f"raw_exec keys: {list(raw_exec.keys())}")
print(f"sprint={raw_exec['sprint']}  kick={raw_exec['kick']}  tackle_attempt={raw_exec['tackle_attempt']}")
print(f"move_dir_raw={raw_exec['move_dir_raw']}  kick_dir_raw={raw_exec['kick_dir_raw']}")

# --- Simulate storing + reloading (what PPO update does) ---
action_np = _action_to_numpy(action, raw_exec)
print(f"\n=== STORED ACTION NUMPY ===")
for k, v in action_np.items():
    print(f"  {k}: {v}")

# --- Now recompute log_prob the same way _recompute_log_prob does ---
dev = trainer.device
sf = obs_dict["self_feat"].unsqueeze(0).to(dev)
of = obs_dict["other_feat"].unsqueeze(0).to(dev)
em = obs_dict["exists_mask"].unsqueeze(0).to(dev)
bf = obs_dict["ball_feat"].unsqueeze(0).to(dev)
gf = obs_dict["global_feat"].unsqueeze(0).to(dev)

with torch.no_grad():
    d_heads = trainer.decision_net(sf, of, em, bf, gf)
    e_heads = trainer.execution_net(sf, of, em, bf, gf, d_heads)

# Convert stored numpy back to tensors (as the PPO update does via as_tensors)
mb_actions = {k: torch.tensor(v).unsqueeze(0).to(dev) for k, v in action_np.items()}

print(f"\n=== PER-HEAD LOG_PROB RECOMPUTE ===")
total_new = 0.0

def blp(logit, key, name):
    val = IndependentBernoulli(logit).log_prob(mb_actions[key]).sum().item()
    print(f"  {name}: logit={logit.mean().item():.4f}  stored={mb_actions[key].item():.1f}  lp={val:.6f}")
    return val

total_new += blp(d_heads.shoot_logit, "shoot", "shoot")
total_new += blp(d_heads.pass_logit, "pass_", "pass_")
total_new += blp(d_heads.move_logit, "move", "move")
total_new += blp(d_heads.tackle_logit, "tackle", "tackle")
total_new += blp(d_heads.get_possession_raw, "get_possession_extra", "gp_extra")
total_new += blp(d_heads.mark_logit, "mark", "mark")
total_new += blp(d_heads.hold_position_logit, "hold_position", "hold")
total_new += blp(e_heads.sprint_logit, "sprint", "sprint")
total_new += blp(e_heads.kick_logit, "kick", "kick")
total_new += blp(e_heads.tackle_attempt_logit, "tackle_attempt", "tackle_attempt")

log_std_move = trainer.execution_net.move_dir_log_std.to(dev)
log_std_kick = trainer.execution_net.kick_dir_log_std.to(dev)

move_dir_stored = mb_actions["move_dir_raw"]  # shape (1, 2)
kick_dir_stored = mb_actions["kick_dir_raw"]
print(f"  move_dir_raw stored: {move_dir_stored.tolist()}  network_out: {e_heads.move_direction.tolist()}")
print(f"  log_std_move: {log_std_move.tolist()}")
lp_movedir = DirectionHead(e_heads.move_direction, log_std_move).log_prob(move_dir_stored).sum().item()
print(f"  move_dir lp={lp_movedir:.6f}")
total_new += lp_movedir

print(f"  kick_dir_raw stored: {kick_dir_stored.tolist()}  network_out: {e_heads.kick_direction.tolist()}")
print(f"  log_std_kick: {log_std_kick.tolist()}")
lp_kickdir = DirectionHead(e_heads.kick_direction, log_std_kick).log_prob(kick_dir_stored).sum().item()
print(f"  kick_dir lp (ungated)={lp_kickdir:.6f}  kick_stored={raw_exec['kick'][0]:.1f}")
if raw_exec['kick'][0] > 0.5:
    total_new += lp_kickdir

print(f"\n  TOTAL new_log_prob (recomputed) = {total_new:.6f}")
print(f"  old_log_prob (at collection)    = {log_prob_old:.6f}")
print(f"  DIFF = {total_new - log_prob_old:.6f}")
print(f"  ratio = exp(diff) = {np.exp(total_new - log_prob_old):.6f}")

# --- Also show what _compute_log_prob gave at collection time, head by head ---
print(f"\n=== COLLECTION-TIME LOG_PROB (from _compute_log_prob) ===")
with torch.no_grad():
    d2 = trainer.decision_net(sf, of, em, bf, gf)
    e2 = trainer.execution_net(sf, of, em, bf, gf, d2)

    samples = {
        "shoot": torch.tensor([[action.shoot]]),
        "pass_": torch.tensor([[action.pass_]]),
        "move": torch.tensor([[action.move]]),
        "tackle": torch.tensor([[action.tackle]]),
        "gp_extra": torch.tensor([[action.get_possession_extra]]),
        "mark": torch.tensor([[action.mark]]),
        "hold": torch.tensor([[action.hold_position]]),
        "pass_tgt": torch.tensor([action.pass_target]),
        "tackle_tgt": torch.tensor([action.tackle_target]),
        "mark_tgt": torch.tensor([action.mark_target]),
        "sprint": torch.tensor(raw_exec["sprint"]).unsqueeze(0),
        "kick": torch.tensor(raw_exec["kick"]).unsqueeze(0),
        "tackle_attempt": torch.tensor(raw_exec["tackle_attempt"]).unsqueeze(0),
        "move_dir_raw": torch.tensor(raw_exec["move_dir_raw"]).unsqueeze(0),
        "kick_dir_raw": torch.tensor(raw_exec["kick_dir_raw"]).unsqueeze(0),
        "kick_power_raw": e2.kick_power,
    }

    lp_coll = trainer._compute_log_prob(d2, e2, samples, em)
    print(f"  _compute_log_prob result = {lp_coll.item():.6f}")
    print(f"  stored old_log_prob      = {log_prob_old:.6f}")
    print(f"  match: {abs(lp_coll.item() - log_prob_old) < 1e-4}")
