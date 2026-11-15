"""Engine-trace snapshot for engine refactors (e.g. splitting engine/match.py):
runs every registered UI scenario (rules/scripted AI, no network) through
ScenarioLoop for a fixed number of ticks with fixed seeds, and records a hash
of the full per-tick match state plus trial outcomes. Two runs of the same code
compare exactly, so a pure code move must give 0 differences.

Usage (from the repo root):
    python scripts/engine_trace_snapshot.py OUT.json
    python scripts/engine_trace_snapshot.py --compare A.json B.json

Each scenario also reports coverage counts (tackle attempts, kicks, possession
changes, outcomes), so it is visible which engine paths a "0 differences"
result actually exercised.
"""
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.getcwd())


class _CounterSeededRandom(random.Random):
    """Unseeded Random() takes the next counter value instead of OS entropy."""
    _created = 0

    def __init__(self, x=None):
        if x is None:
            _CounterSeededRandom._created += 1
            x = 300_000 + _CounterSeededRandom._created
        super().__init__(x)


random.Random = _CounterSeededRandom

_TICKS = 900
_SKIP = {"phase1_neural_ai"}   # needs a trained network; the env trace covers neural play


def _state(match) -> list:
    rows = []
    for p in sorted(match.players, key=lambda q: q.player_id):
        rows.append([p.player_id, repr(p.position.x), repr(p.position.y), repr(p.velocity.x), repr(p.velocity.y),
                     repr(p.heading_rad), repr(p.stamina), str(getattr(p, "state", "")),
                     p.kick_count, p.tackle_fire_count, type(p.current_order).__name__])
    b = match.ball
    rows.append(["ball", repr(b.position.x), repr(b.position.y), repr(b.position.z),
                 repr(b.velocity.x), repr(b.velocity.y), repr(b.velocity.z), b.possessed_by])
    rows.append(["score", match.scoreboard.left_goals, match.scoreboard.right_goals, repr(match.time_s)])
    return rows


def run() -> dict:
    from footballcoach.ui.scenarios import SCENARIOS, ScenarioLoop

    record = {}
    for si, defn in enumerate(SCENARIOS):
        if defn.key in _SKIP:
            continue
        random.seed(40 + si)
        np.random.seed(40 + si)
        loop = ScenarioLoop(definition=defn, max_trials=0, linger_s=0.0, timeout_ticks=300)
        h = hashlib.sha256()
        cov = {"tackles": 0, "kicks": 0, "possession_changes": 0, "trials_ended": 0}
        prev_owner = loop.match.ball.possessed_by
        prev_counts = {p.player_id: (p.kick_count, p.tackle_fire_count) for p in loop.match.players}
        for _ in range(_TICKS):
            ended = loop.step()
            m = loop.match
            if ended:
                cov["trials_ended"] += 1
                prev_counts = {p.player_id: (p.kick_count, p.tackle_fire_count) for p in m.players}
                prev_owner = m.ball.possessed_by
            else:
                for p in m.players:
                    k0, t0 = prev_counts.get(p.player_id, (0, 0))
                    cov["kicks"] += max(0, p.kick_count - k0)
                    cov["tackles"] += max(0, p.tackle_fire_count - t0)
                    prev_counts[p.player_id] = (p.kick_count, p.tackle_fire_count)
                if m.ball.possessed_by != prev_owner:
                    cov["possession_changes"] += 1
                    prev_owner = m.ball.possessed_by
            h.update(json.dumps(_state(m)).encode())
        record[defn.key] = {"hash": h.hexdigest(), "coverage": cov, "outcomes": dict(loop.outcomes)}
        print(f"{defn.key:28s} {cov} {dict(loop.outcomes)}")
    return record


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    bad = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print(f"{len(bad)} scenario(s) differ" + (": " + ", ".join(bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    rec = run()
    with open(sys.argv[1], "w") as f:
        json.dump(rec, f, sort_keys=True)
    print("wrote", sys.argv[1])
