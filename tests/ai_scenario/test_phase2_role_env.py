"""Phase 2 (1v2) through the role-based env: outcome classification, controller
rolls, terminal rewards per role, trainee/secondary wiring for neural players,
and an all-rules playthrough that must produce goals, saves and defensive wins.
"""
from __future__ import annotations

import collections
import random

import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_env, build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.outcome import ATTACK_DEFENCE_OUTCOME_KEYS, classify_attack_defence_outcome
from footballcoach.ai.env.role_env import RoleScenarioEnv, roll_controller
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import NeuralPlayerAI
from footballcoach.ui.scenarios import build_1v2_scenario

PHASE2 = PHASES_BY_ID[2]
ALL_RULES = {r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")}
TABLE = load_ai_config()["reward"]["phase2"]["terminal"]


def _rules_env() -> RoleScenarioEnv:
    return build_role_env(PHASE2, controller_weights=ALL_RULES)


def _run_episode(env, seed):
    env.reset(seed=seed)
    done, info, rewards = False, None, []
    while not done:
        _, r, done, info = env.step()
        rewards.append(r)
    return info, rewards


# ----- classifier ---------------------------------------------------------------

def _loose_ball_at(match, x, y, toucher="attacker"):
    match.ball.possessed_by = None
    match.ball.last_touched_by_player_id = toucher
    match.ball.position = Vector3(x, y, match.ball.radius_m)
    return match


@pytest.fixture
def m():
    return build_1v2_scenario(seed=1)


def test_goal_and_timeout_pass_through(m):
    for raw in ("goal", "timeout"):
        assert classify_attack_defence_outcome(raw, m, attacking_team=Team.LEFT) == raw


@pytest.mark.parametrize("holder,expected", [("keeper", "saved"), ("defender", "defence_possession")])
@pytest.mark.parametrize("raw", ["saved", "dispossessed"])
def test_possession_by_the_defence_is_split_by_goalkeeper(m, holder, expected, raw):
    m.ball.possessed_by = holder
    assert classify_attack_defence_outcome(raw, m, attacking_team=Team.LEFT) == expected


def test_attacking_team_holding_the_ball_is_not_a_defensive_end(m):
    m.ball.possessed_by = "attacker"
    with pytest.raises(ValueError, match="not a defending player"):
        classify_attack_defence_outcome("saved", m, attacking_team=Team.LEFT)


def test_loose_ball_over_the_attacked_goal_line_is_off_target(m):
    _loose_ball_at(m, m.pitch.half_length + 1.0, 6.0)
    assert classify_attack_defence_outcome("miss", m, attacking_team=Team.LEFT) == "shot_off_target"


@pytest.mark.parametrize("where", ["touchline", "own_goal_line"])
def test_other_attacker_outs_are_attacker_out(m, where):
    x, y = (10.0, m.pitch.half_width + 1.0) if where == "touchline" else (-m.pitch.half_length - 1.0, 0.0)
    _loose_ball_at(m, x, y)
    assert classify_attack_defence_outcome("miss", m, attacking_team=Team.LEFT) == "attacker_out"


def test_dribbling_over_the_goal_line_is_attacker_out(m):
    _loose_ball_at(m, m.pitch.half_length + 0.5, 3.0)
    m.ball.possessed_by = "attacker"
    assert classify_attack_defence_outcome("miss", m, attacking_team=Team.LEFT) == "attacker_out"


@pytest.mark.parametrize("toucher,expected", [("keeper", "saved"), ("defender", "defence_possession")])
def test_ball_out_last_had_by_the_defence(m, toucher, expected):
    _loose_ball_at(m, m.pitch.half_length + 1.0, 6.0, toucher=toucher)
    assert classify_attack_defence_outcome("miss", m, attacking_team=Team.LEFT) == expected


def test_attacking_right_mirrors_the_goal_line(m):
    _loose_ball_at(m, -m.pitch.half_length - 1.0, 0.0)
    for p in m.players:
        p.team = Team.RIGHT if p.player_id == "attacker" else Team.LEFT
    assert classify_attack_defence_outcome("miss", m, attacking_team=Team.RIGHT) == "shot_off_target"


def test_untouched_ball_out_and_unknown_raw_outcomes_raise(m):
    _loose_ball_at(m, 0.0, m.pitch.half_width + 1.0, toucher=None)
    with pytest.raises(ValueError, match="nobody"):
        classify_attack_defence_outcome("miss", m, attacking_team=Team.LEFT)
    with pytest.raises(ValueError, match="box_possession"):
        classify_attack_defence_outcome("box_possession", m, attacking_team=Team.LEFT)


# ----- controllers --------------------------------------------------------------

def test_roll_controller_follows_the_weights():
    rng = random.Random(0)
    counts = collections.Counter(roll_controller({"rules": 1, "neural": 3}, rng) for _ in range(8000))
    assert counts["neural"] / 8000 == pytest.approx(0.75, abs=0.02)
    assert {roll_controller({"rules": 1}, rng) for _ in range(50)} == {"rules"}
    assert {roll_controller({"rules": 0, "neural": 2}, rng) for _ in range(50)} == {"neural"}


@pytest.mark.parametrize("weights,match", [
    ({"attacker": {"rules": 1}}, "no controller weights"),
    ({**ALL_RULES, "attacker": {"robot": 1}}, "unknown controller"),
    ({**ALL_RULES, "attacker": {"rules": 0, "neural": 0}}, "not all 0"),
])
def test_bad_controller_weights_are_rejected(weights, match):
    with pytest.raises(ValueError, match=match):
        build_role_env(PHASE2, controller_weights=weights)


# ----- env: all rules -------------------------------------------------------------

def test_build_env_dispatches_role_based_phases():
    assert type(build_env(PHASE2)) is RoleScenarioEnv
    assert not isinstance(build_env(PHASES_BY_ID[1]), RoleScenarioEnv)


def test_all_rules_episode_has_no_neural_player_and_pays_nothing_mid_episode():
    env = _rules_env()
    info, rewards = _run_episode(env, seed=3)
    assert env.trainee_player_id == "attacker" and env.secondary_player_ids == []
    assert env.episode_tag == "phase2_1v2[attacker=rules,defender=rules,goalkeeper=rules]"
    assert info.episode_tag == env.episode_tag
    assert info.trial_outcome in ATTACK_DEFENCE_OUTCOME_KEYS
    assert env.last_trainee_transition is None and env.last_secondary_results == []
    assert all(r == 0.0 for r in rewards[:-1])
    assert rewards[-1] == TABLE["attacker"][info.trial_outcome]


def test_seeded_episodes_replay_exactly():
    a, b = _rules_env(), _rules_env()
    for seed in range(5):
        ia, ra = _run_episode(a, seed)
        ib, rb = _run_episode(b, seed)
        assert (ia.trial_outcome, ia.ticks_elapsed, ra) == (ib.trial_outcome, ib.ticks_elapsed, rb)


def test_rules_playthrough_scores_saves_and_defends():
    env = _rules_env()
    outcomes = collections.Counter(_run_episode(env, seed)[0].trial_outcome for seed in range(150))
    assert outcomes["goal"] > 20 and outcomes["saved"] > 10 and outcomes["defence_possession"] > 10, outcomes


def test_episode_timeout_ends_with_the_timeout_reward():
    env = _rules_env()
    env._base_max_episode_s = 0.3
    info, rewards = _run_episode(env, seed=0)
    assert info.trial_outcome == "timeout" and rewards[-1] == TABLE["attacker"]["timeout"]


def test_rolled_neural_without_a_sampler_raises():
    env = build_role_env(PHASE2, controller_weights={**ALL_RULES, "defender": {"neural": 1}})
    with pytest.raises(ValueError, match="no sample_action_fn"):
        env.reset(seed=0)


def test_base_scenario_env_refuses_role_phases():
    env = ScenarioEnv(_rules_env().definition, trainee_player_id="attacker", phase=2, max_episode_s=5.0)
    env.reset(seed=0)
    with pytest.raises(NotImplementedError, match="phase 1 only"):
        env.step()


# ----- env: neural players ------------------------------------------------------

@pytest.fixture(scope="module")
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    return PPOTrainer.from_config(device=torch.device("cpu"))


def _neural_env(trainer, neural_roles):
    env = _rules_env()
    env.sample_action_fn = trainer._sample_action
    env.controller_overrides = {r: ("neural" if r in neural_roles else "rules") for r in ALL_RULES}
    return env


@pytest.mark.parametrize("neural_roles,trainee,secondaries", [
    (("attacker", "defender"), "attacker", ["defender"]),
    (("defender",), "defender", []),
    (("attacker", "defender", "goalkeeper"), "attacker", ["defender", "keeper"]),
])
def test_neural_players_are_wired_as_trainee_then_secondaries(trainer, neural_roles, trainee, secondaries):
    env = _neural_env(trainer, neural_roles)
    env.reset(seed=4)
    assert env.trainee_player_id == trainee and env.secondary_player_ids == secondaries
    for p in env.match.players:
        is_neural = env.roles[p.player_id] in neural_roles
        assert isinstance(p.ai, NeuralPlayerAI) == is_neural
        if is_neural:
            assert p.current_order is None


def test_neural_transitions_carry_each_roles_terminal_reward(trainer):
    env = _neural_env(trainer, ("attacker", "defender"))
    env.reset(seed=6)
    done, steps = False, 0
    while not done:
        _, reward, done, info = env.step()
        steps += 1
        assert env.last_trainee_transition is not None
        [sec] = env.last_secondary_results
        assert sec["player_id"] == "defender" and sec["done"] == (1.0 if done else 0.0)
        if not done:
            assert reward == 0.0 and sec["reward"] == 0.0 and sec["step_outcome"] == ""
    assert reward == TABLE["attacker"][info.trial_outcome]
    assert sec["reward"] == TABLE["defender"][info.trial_outcome]
    assert sec["step_outcome"] == info.trial_outcome
    assert env.episode_tag == "phase2_1v2[attacker=neural,defender=neural,goalkeeper=rules]"
