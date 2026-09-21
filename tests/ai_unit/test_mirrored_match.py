"""Physically mirrored ENGINE matches must give the network identical inputs and outputs after canonicalisation.

Unlike the hand-mirrored tensors used elsewhere (which reuse the very index lists the canonicalisers use, so a missing or
wrong y/x-signed feature would still pass), this builds a random real ``Match`` and its true mirror image in the simulator --
positions, velocities, headings, desired directions, ball spin (a pseudovector), team roles and scoreboard -- encodes both with
the real ``encode_observation`` and requires every feature to agree after canonicalisation.

Mirrors covered: x (the same match seen by the other team -- a team swap), y, and both (a 180 degree rotation); both pipelines:
x-canonical only (``canonicalize_obs``, always on) and y+x (``ppo.y_canonical``). Also checked through the real networks with the
frozen physics encoders enabled, including the encoder outputs and the latent vector.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import fields

import numpy as np
import pytest
import torch

from footballcoach.ai.obs.canonical import canonicalize_obs
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.obs.schema import BallFeatures, GlobalFeatures, PlayerFeatures
from footballcoach.ai.obs.y_canonical import flip_decision_heads_y, flip_exec_heads_y, mirror_y_obs, y_flip_mask
from footballcoach.engine.match import Match
from footballcoach.engine.movement import SpeedMode
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import Phase1RulesAI

from .test_y_canonical import _assert_heads_close, _physics_nets

_SPEED_MODES = list(SpeedMode)


def _random_match(seed: int, possessed_by: str | None) -> Match:
    r = random.Random(seed)
    players = []
    for pid, team in (("A1", Team.LEFT), ("A2", Team.LEFT), ("B1", Team.RIGHT), ("B2", Team.RIGHT)):
        attrs = PlayerAttributes(*[r.uniform(0.2, 0.9) for _ in range(8)])
        p = Player.create(pid, team, attrs, position=Vector3(r.uniform(-40, 40), r.uniform(-25, 25), 0.0))
        if pid == "B2":                                    # stationary player: heading is independent of velocity
            p.velocity = Vector3(0.0, 0.0, 0.0)
            p.heading_rad = r.uniform(-math.pi, math.pi)
        else:                                              # Match validates heading ~ velocity direction at construction
            p.velocity = Vector3(r.uniform(-6, 6), r.uniform(-6, 6), 0.0)
            p.heading_rad = math.atan2(p.velocity.y, p.velocity.x) + r.uniform(-0.05, 0.05)
        p.stamina = r.uniform(0.3, 1.0)
        a = r.uniform(-math.pi, math.pi)
        p.desired_direction = Vector3(math.cos(a), math.sin(a), 0.0)
        p.last_desired_speed_mode = r.choice(_SPEED_MODES)
        p.ai = Phase1RulesAI()
        players.append(p)
    ball = Ball(
        position=Vector3(r.uniform(-30, 30), r.uniform(-20, 20), r.uniform(0.0, 1.5)),
        velocity=Vector3(r.uniform(-15, 15), r.uniform(-15, 15), r.uniform(-3, 6)),
        spin=Vector3(r.uniform(-60, 60), r.uniform(-60, 60), r.uniform(-60, 60)),
    )
    match = Match(pitch=Pitch.standard(), players=players, ball=ball, rng_reduction=1.0, rng=random.Random(0))
    match.scoreboard.left_goals, match.scoreboard.right_goals = 2, 1
    if possessed_by is not None:
        ball.set_initial_possession(possessed_by)
    else:
        ball.last_touched_by_player_id = "B1"
    return match


def _mirror_vec(v: Vector3, mx: bool, my: bool) -> Vector3:
    return Vector3(-v.x if mx else v.x, -v.y if my else v.y, v.z)


def _mirror_spin(s: Vector3, mx: bool, my: bool) -> Vector3:
    """Angular velocity is a pseudovector: a reflection negates the components IN the mirror plane."""
    x, y, z = s.x, s.y, s.z
    if mx:
        y, z = -y, -z
    if my:
        x, z = -x, -z
    return Vector3(x, y, z)


def mirror_match(match: Match, mx: bool, my: bool) -> Match:
    m = copy.deepcopy(match)
    for p in m.players:
        p.position = _mirror_vec(p.position, mx, my)
        p.velocity = _mirror_vec(p.velocity, mx, my)
        p.desired_direction = _mirror_vec(p.desired_direction, mx, my)
        th = p.heading_rad
        if mx:
            th = math.pi - th
        if my:
            th = -th
        p.heading_rad = th
        if mx:                                            # seen from the other end: the teams swap sides
            p.team = Team.RIGHT if p.team == Team.LEFT else Team.LEFT
    m.ball.position = _mirror_vec(m.ball.position, mx, my)
    m.ball.velocity = _mirror_vec(m.ball.velocity, mx, my)
    m.ball.spin = _mirror_spin(m.ball.spin, mx, my)
    if mx:
        sb = m.scoreboard
        sb.left_goals, sb.right_goals = sb.right_goals, sb.left_goals
    return m


def _network_inputs(match: Match, observer: str, y_canonical: bool) -> dict[str, torch.Tensor]:
    obs = encode_observation(match, observer, time_remaining_s=42.0, attack_defence_smoothed=0.3, rng=random.Random(5), phase=1)
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(0)
    sf, of, bf, gf, em = t(obs.self_feat), t(obs.other_feat), t(obs.ball_feat), t(obs.global_feat), t(obs.exists_mask)
    if y_canonical:                                        # what YCanonicalNetworkWrapper does first
        sf, of, bf = mirror_y_obs(sf, of, bf, y_flip_mask(sf))
    sf, of, bf, _ = canonicalize_obs(sf, of, bf)          # then CanonicalNetworkWrapper's x mirror
    return dict(self_feat=sf, other_feat=of, ball_feat=bf, global_feat=gf, exists_mask=em,
                self_ai_type=t(obs.self_ai_type), other_ai_type=t(obs.other_ai_type))


def _assert_same_inputs(a: dict, b: dict, what: str, atol=1e-5):
    names = {"self_feat": [f.name for f in fields(PlayerFeatures)], "other_feat": [f.name for f in fields(PlayerFeatures)],
             "ball_feat": [f.name for f in fields(BallFeatures)], "global_feat": [f.name for f in fields(GlobalFeatures)]}
    for key in a:
        x, y = a[key], b[key]
        bad = (x - y).abs() > atol
        if bool(bad.any()):
            cols = sorted({int(i) for i in bad.nonzero()[:, -1]}) if key in names else []
            labels = [names[key][c] for c in cols] if cols else []
            raise AssertionError(f"{what}: {key} differs after canonicalisation in {labels or 'some entries'} "
                                 f"(max |diff| {float((x - y).abs().max()):.3e})")


_MIRRORS = {"x": (True, False), "y": (False, True), "xy": (True, True)}
_CASES = [(seed, poss) for seed in range(6) for poss in (None, "A2")]


@pytest.mark.parametrize("mirror", ["x", "y", "xy"])
@pytest.mark.parametrize("observer", ["A1", "B2"])
@pytest.mark.parametrize("seed,possessed_by", _CASES)
def test_mirrored_match_gives_identical_canonical_inputs_under_y_and_x_canonicalisation(mirror, observer, seed, possessed_by):
    base = _random_match(seed, possessed_by)
    mx, my = _MIRRORS[mirror]
    a = _network_inputs(base, observer, y_canonical=True)
    b = _network_inputs(mirror_match(base, mx, my), observer, y_canonical=True)
    _assert_same_inputs(a, b, f"mirror={mirror} observer={observer} seed={seed}")


@pytest.mark.parametrize("observer", ["A1", "B2"])
@pytest.mark.parametrize("seed,possessed_by", _CASES)
def test_x_mirrored_match_gives_identical_canonical_inputs_under_x_canonicalisation_alone(observer, seed, possessed_by):
    base = _random_match(seed, possessed_by)
    a = _network_inputs(base, observer, y_canonical=False)
    b = _network_inputs(mirror_match(base, True, False), observer, y_canonical=False)
    _assert_same_inputs(a, b, f"x-canonical only, observer={observer} seed={seed}")


def test_the_comparison_can_fail_a_y_mirrored_match_is_not_identical_without_y_canonicalisation():
    base = _random_match(0, None)
    a = _network_inputs(base, "A1", y_canonical=False)
    b = _network_inputs(mirror_match(base, False, True), "A1", y_canonical=False)
    with pytest.raises(AssertionError):
        _assert_same_inputs(a, b, "negative control")


def _run_nets(pair, inp):
    sf, of, bf, gf, em = inp["self_feat"], inp["other_feat"], inp["ball_feat"], inp["global_feat"], inp["exists_mask"]
    sat, oat = inp["self_ai_type"], inp["other_ai_type"]
    with torch.no_grad():
        d = pair[0](sf, of, em, bf, gf, sat, oat)
        return d, pair[1](sf, of, em, bf, gf, d, sat, oat)


@pytest.mark.parametrize("mirror", ["x", "y", "xy"])
@pytest.mark.parametrize("seed,possessed_by", [(0, None), (1, "A2"), (2, None), (3, "A2")])
def test_networks_with_physics_encoders_give_matching_outputs_for_a_mirrored_match(tmp_path, mirror, seed, possessed_by):
    """Raw (un-canonicalised) observations of the real match and its mirror image go through the wrapped networks with the frozen
    physics encoders enabled. Encoder outputs, latent vector, value and every head must agree; the y-signed action outputs agree
    after mirroring back (the wrapper returns each match in its own world frame). Under an x-mirror the outputs live in the canonical
    frame, so they are simply equal."""
    _, wrapped = _physics_nets(tmp_path)
    base = _random_match(seed, possessed_by)
    mx, my = _MIRRORS[mirror]
    mirrored = mirror_match(base, mx, my)

    def raw(match):
        o = encode_observation(match, "A1", time_remaining_s=42.0, attack_defence_smoothed=0.3, rng=random.Random(5), phase=1)
        t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(0)
        return dict(self_feat=t(o.self_feat), other_feat=t(o.other_feat), ball_feat=t(o.ball_feat), global_feat=t(o.global_feat),
                    exists_mask=t(o.exists_mask), self_ai_type=t(o.self_ai_type), other_ai_type=t(o.other_ai_type))

    da, ea = _run_nets(wrapped, raw(base))
    db, eb = _run_nets(wrapped, raw(mirrored))
    if my:                                                 # y-signed outputs come back in each match's own world frame
        flip = torch.ones(1, dtype=torch.bool)
        da, ea = flip_decision_heads_y(da, flip), flip_exec_heads_y(ea, flip)
    # (the ball/player physics encoder outputs, the latent and every other field are compared here too)
    _assert_heads_close(da, db, atol=1e-4)
    _assert_heads_close(ea, eb, atol=1e-4)
    assert da.ball_physics_full is not None and da.self_physics_full is not None
    assert torch.allclose(da.latent_vector, db.latent_vector, atol=1e-4)
