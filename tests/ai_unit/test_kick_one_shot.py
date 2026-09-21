"""action.kick_one_shot: one kick decision fires at most one kick.

NeuralPlayerAI re-applies the cached decision every tick of the decision interval. Without kick_one_shot an in-possession
kick released the ball and then re-armed (and, on re-touch, re-fired) for the rest of the interval: measured ~3 physical
kicks per decision and kick_armed set at the interval boundary. These tests drive a real Match with a stub network.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from footballcoach.ai.action.apply_nn_action import apply_action_to_player
from footballcoach.ai.action.gating import GatingResult, SelectedAction
from footballcoach.engine.match import Match
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import NeuralPlayerAI, StopWhenIdleAI
from tests.conftest import make_player

INTERVAL = 6


def _kick_sampler(kick: bool = True, tackle: bool = False):
    def sample(_obs):
        exec_phys = {
            "exec_move": False, "move_direction": np.array([0.0, 0.0]), "sprint": False,
            "kick_this_tick": kick, "kick_direction": np.array([1.0, 0.0, 0.0]), "kick_power_fraction": 0.5,
            "kick_spin": np.zeros(3), "tackle_attempt": tackle,
        }
        return ({}, 0.0, 0.0, {}, exec_phys, {}, {}, {}, {})
    return sample


def _match(ball_at: Vector3, possessed: bool):
    p1 = make_player("p1", Team.LEFT, attr_value=0.6, position=Vector3(0, 0, 0))
    p2 = make_player("p2", Team.RIGHT, attr_value=0.6, position=Vector3(0, -34, 0))
    p2.ai = StopWhenIdleAI()
    ball = Ball.at_rest(ball_at)
    if possessed:
        ball.possessed_by = "p1"
    return Match(pitch=Pitch.standard(), players=[p1, p2], ball=ball, rng_reduction=1.0, rng=random.Random(0)), p1


def _run_interval(match, player, one_shot: bool, sampler=None):
    player.ai = NeuralPlayerAI(sampler or _kick_sampler(), decision_interval_ticks=INTERVAL, max_episode_s=60.0,
                               rng=random.Random(1), kick_one_shot=one_shot)
    armed_ticks = 0
    for _ in range(INTERVAL):
        match.step()
        armed_ticks += int(player.kick_armed)
    return armed_ticks


class TestKickOneShot:
    def test_possession_kick_fires_once_and_does_not_rearm(self):
        match, p1 = _match(Vector3(0, 0, 0), possessed=True)
        armed_ticks = _run_interval(match, p1, one_shot=True)
        assert p1.kick_count == 1
        assert armed_ticks == 0
        assert not p1.kick_armed

    def test_possession_kick_old_behaviour_rearms_after_release(self):
        match, p1 = _match(Vector3(0, 0, 0), possessed=True)
        armed_ticks = _run_interval(match, p1, one_shot=False)
        assert p1.kick_count >= 1
        assert armed_ticks >= 1
        assert p1.kick_armed, "old behaviour: the released ball leaves the kick armed at the interval boundary"

    def test_armed_kick_fires_once_on_first_touch(self):
        match, p1 = _match(Vector3(0.3, 0, 0), possessed=False)
        p1.velocity = Vector3(7.0, 0, 0)
        _run_interval(match, p1, one_shot=True)
        assert p1.kick_count == 1
        assert not p1.kick_armed

    def test_armed_kick_keeps_chasing_until_the_ball_is_reachable(self):
        # Ball out of reach: the intent stays armed for the whole interval and nothing fires.
        match, p1 = _match(Vector3(15, 0, 0), possessed=False)
        armed_ticks = _run_interval(match, p1, one_shot=True)
        assert p1.kick_count == 0
        assert armed_ticks == INTERVAL
        assert p1.kick_armed

    def test_next_decision_can_kick_again(self):
        match, p1 = _match(Vector3(0, 0, 0), possessed=True)
        _run_interval(match, p1, one_shot=True)
        assert p1.kick_count == 1
        # Force the ball back to the player's feet; the NEXT interval's fresh decision fires again.
        match.ball.possessed_by = "p1"
        match.ball.position = Vector3(p1.position.x, p1.position.y, 0.0)
        match.ball.velocity = Vector3(0, 0, 0)
        for _ in range(INTERVAL):
            match.step()
        assert p1.kick_count == 2

    def test_no_kick_decision_never_kicks(self):
        match, p1 = _match(Vector3(0, 0, 0), possessed=True)
        _run_interval(match, p1, one_shot=True, sampler=_kick_sampler(kick=False))
        assert p1.kick_count == 0 and not p1.kick_armed


class TestDropFiredKick:
    def _ai(self, one_shot):
        ai = NeuralPlayerAI(_kick_sampler(), decision_interval_ticks=INTERVAL, kick_one_shot=one_shot)
        ai._last_gating = GatingResult(selected=SelectedAction.NONE, kick_this_tick=True, kick_direction=np.array([1.0, 0.0, 0.0]),
                                       kick_power_fraction=0.5)
        return ai

    def _player(self, kicks):
        p = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
        p.kick_count = kicks
        return p

    def test_drops_once_the_count_has_increased(self):
        ai = self._ai(True)
        ai._kick_count_at_decision = 3
        ai._drop_fired_kick(self._player(4))
        assert not ai._last_gating.kick_this_tick

    def test_keeps_intent_until_a_kick_fires(self):
        ai = self._ai(True)
        ai._kick_count_at_decision = 3
        ai._drop_fired_kick(self._player(3))
        assert ai._last_gating.kick_this_tick

    def test_disabled_flag_never_drops(self):
        ai = self._ai(False)
        ai._kick_count_at_decision = 0
        ai._drop_fired_kick(self._player(5))
        assert ai._last_gating.kick_this_tick

    def test_no_baseline_never_drops(self):
        ai = self._ai(True)
        ai._drop_fired_kick(self._player(5))
        assert ai._last_gating.kick_this_tick

    def test_reset_clears_baseline(self):
        ai = self._ai(True)
        ai._kick_count_at_decision = 2
        ai.reset()
        assert ai._kick_count_at_decision is None

    def test_default_reads_config(self, monkeypatch):
        import footballcoach.ai.action.apply_nn_action as ana
        monkeypatch.setattr(ana, "action_flag", lambda name: name == "kick_one_shot")
        assert NeuralPlayerAI(_kick_sampler()).kick_one_shot is True
        monkeypatch.setattr(ana, "action_flag", lambda name: False)
        assert NeuralPlayerAI(_kick_sampler()).kick_one_shot is False


class TestKickCount:
    def test_count_increments_per_executed_kick_only(self):
        match, p1 = _match(Vector3(0, 0, 0), possessed=True)
        assert p1.kick_count == 0
        gating = GatingResult(selected=SelectedAction.NONE, kick_this_tick=True, kick_direction=np.array([1.0, 0.0, 0.0]),
                              kick_power_fraction=0.5, kick_spin=np.zeros(3))
        apply_action_to_player(gating, p1, match, [None] * 21, {})
        assert p1.kick_count == 1
        # Ball released -> a second attempt only ARMS; no physical kick, no count.
        apply_action_to_player(gating, p1, match, [None] * 21, {})
        assert p1.kick_count == 1 and p1.kick_armed
