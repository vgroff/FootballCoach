"""The critic-only episode-elapsed fraction (obs value_side_feat,
schema.VALUE_SIDE_DIM): what it holds, that only the value head reads it,
that old checkpoints load with exact zero columns, and that every path that
feeds the networks carries it."""
from __future__ import annotations

import copy
import random

import pytest
import torch

from footballcoach.ai.curriculum.envs import build_env, build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.obs.augment import augment_batch, augment_obs_bc
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.ppo.checkpoint_io import pad_value_side_columns
from footballcoach.ai.ppo.trainer_common import _ai_types


def _trainer(separate_value_net=True, seed=5):
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(seed)
    return PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=separate_value_net)


@pytest.fixture(scope="module")
def trainer():
    return _trainer()


def _obs(time_remaining_s=9.0, episode_length_s=24.0):
    env = build_env(PHASES_BY_ID[1])
    env.reset(seed=12)
    return {k: v.unsqueeze(0) for k, v in encode_observation(
        match=env.match, player_id="trainee", time_remaining_s=time_remaining_s, attack_defence_smoothed=0.5,
        rng=random.Random(0), phase=1, episode_length_s=episode_length_s).to_torch_dict().items()}


def _all_outputs(t, obs):
    """Every policy output, and the critic's value, for obs."""
    sf, of, em, bf, gf = obs["self_feat"], obs["other_feat"], obs["exists_mask"], obs["ball_feat"], obs["global_feat"]
    sat, oat = _ai_types(obs)
    with torch.no_grad():
        d = t.decision_net(sf, of, em, bf, gf, sat, oat)
        e = t.execution_net(sf, of, em, bf, gf, d, sat, oat)
        value = t._get_value(obs)
    policy = {}
    for name, heads in (("d", d), ("e", e)):
        for k, v in vars(heads).items():
            if isinstance(v, torch.Tensor) and k != "value":
                policy[f"{name}.{k}"] = v.clone()
    return policy, value


def test_encoder_computes_the_elapsed_fraction():
    assert float(_obs(9.0, 24.0)["value_side_feat"][0, 0]) == pytest.approx(1 - 9.0 / 24.0)
    assert float(_obs(30.0, 24.0)["value_side_feat"][0, 0]) == 0.0      # clamped
    assert float(_obs(9.0, None)["value_side_feat"][0, 0]) == 0.0       # no episode length: unset


@pytest.mark.parametrize("phase_id", [1, 2])
def test_training_rows_and_env_observation_carry_the_same_fraction(trainer, phase_id):
    ph = PHASES_BY_ID[phase_id]
    env = build_env(ph) if not ph.is_role_based else build_role_env(
        ph, controller_weights={"attacker": {"neural": 1}, "defender": {"rules": 1}, "goalkeeper": {"rules": 1}})
    env.sample_action_fn = trainer._sample_action
    obs = env.reset(seed=3)
    for _ in range(8):
        ticks = env._episode_ticks
        env_side = float(obs.to_torch_dict()["value_side_feat"][0])
        obs, _, done, _ = env.step()
        row_side = float(env.last_trainee_transition["obs"]["value_side_feat"][0])
        assert row_side == pytest.approx(env_side, abs=1e-6)
        assert row_side == pytest.approx(ticks * env._dt_s / env.max_episode_s, abs=1e-6)
        if done:
            break


def test_no_policy_output_depends_on_it_and_the_value_does(trainer):
    a, b = _obs(), _obs()
    b["value_side_feat"] = torch.tensor([[0.95]])
    pa, va = _all_outputs(trainer, a)
    pb, vb = _all_outputs(trainer, b)
    assert pa.keys() == pb.keys() and len(pa) > 10
    for k in pa:
        assert torch.equal(pa[k], pb[k]), k
    assert va != vb


def test_the_value_gradient_reaches_its_column(trainer):
    t = copy.deepcopy(trainer)
    obs = _obs()
    sf, of, em, bf, gf = obs["self_feat"], obs["other_feat"], obs["exists_mask"], obs["ball_feat"], obs["global_feat"]
    sat, oat = _ai_types(obs)
    d = t.decision_net(sf, of, em, bf, gf, sat, oat)
    v = t.value_net(sf, of, em, bf, gf, d, sat, oat, value_only=True)
    v.sum().backward()
    w = dict(t.value_net.named_parameters())
    w = next(p for n, p in w.items() if n.endswith("value_head.1.weight"))
    side_col = w.shape[1] - 2          # just before elapsed_norm
    assert float(w.grad[:, side_col].abs().max()) > 0.0


@pytest.mark.parametrize("separate_value_net", [True, False])
def test_an_old_checkpoint_loads_with_an_exact_zero_column(tmp_path, separate_value_net):
    """Emulate a pre-change checkpoint by deleting the side column; it must
    load with a zero column (not the tolerant loader's random fallback) and
    give exactly the old critic's value whatever the side feature is."""
    src = _trainer(separate_value_net, seed=6)
    for net in [src.execution_net] + ([src.value_net] if src.value_net is not None else []):
        w = next(p for n, p in net.named_parameters() if n.endswith("value_head.1.weight"))
        with torch.no_grad():
            w[:, w.shape[1] - 2] = 0.0    # what the old network computes
    src._save_checkpoint_to(tmp_path / "new.pt")
    ck = torch.load(tmp_path / "new.pt", map_location="cpu", weights_only=False)
    for key in ("execution_net", "value_net"):
        if key not in ck:
            continue
        for k, v in list(ck[key].items()):
            if k.endswith("value_head.1.weight"):
                ck[key][k] = torch.cat([v[:, :-2], v[:, -1:]], dim=1)
    torch.save(ck, tmp_path / "old.pt")
    dst = _trainer(separate_value_net, seed=99)
    dst.load_checkpoint(tmp_path / "old.pt")
    for side in (0.0, 0.4, 1.0):
        o = _obs()
        o["value_side_feat"] = torch.tensor([[side]])
        assert _all_outputs(dst, o)[1] == _all_outputs(src, o)[1]
    # Nothing was left at random init.
    for a, b in zip(src.execution_net.state_dict().values(), dst.execution_net.state_dict().values()):
        assert torch.equal(a, b)


def test_padding_leaves_current_checkpoints_alone(trainer):
    sd = copy.deepcopy(trainer.execution_net.state_dict())
    assert not pad_value_side_columns(sd, trainer.execution_net, "x")
    assert all(torch.equal(a, b) for a, b in zip(sd.values(), trainer.execution_net.state_dict().values()))


def test_augmentation_carries_it(trainer):
    o = _obs()
    o["value_side_feat"] = torch.tensor([[0.3]])
    aug_obs, _ = augment_obs_bc(o, torch.zeros(1, 27), n_slot_shuffles=2, rng=random.Random(0))
    n = aug_obs["self_feat"].shape[0]
    assert aug_obs["value_side_feat"].shape == (n, 1) and torch.all(aug_obs["value_side_feat"] == 0.3)
    batch = {f"obs/{k}": v for k, v in o.items()}
    batch.update({"log_probs": torch.zeros(1), "values": torch.zeros(1), "rewards": torch.zeros(1),
                  "dones": torch.zeros(1), "advantages": torch.zeros(1), "returns": torch.zeros(1),
                  "sample_weights": torch.ones(1)})
    out = augment_batch(batch, n_slot_shuffles=2, rng=random.Random(0))
    assert torch.all(out["obs/value_side_feat"] == 0.3) and out["obs/value_side_feat"].shape[0] == out["obs/self_feat"].shape[0]


def test_the_sampled_value_uses_it(trainer):
    """The value recorded at rollout time (GAE) must see it like the update does."""
    a, b = _obs(), _obs()
    b["value_side_feat"] = torch.tensor([[0.95]])
    sq = {k: v.squeeze(0) for k, v in a.items()}
    sb = {k: v.squeeze(0) for k, v in b.items()}
    torch.manual_seed(0)
    va = trainer._sample_action(sq)[2]
    torch.manual_seed(0)
    vb = trainer._sample_action(sb)[2]
    assert va != vb
    assert va == pytest.approx(_all_outputs(trainer, a)[1], abs=1e-5)
