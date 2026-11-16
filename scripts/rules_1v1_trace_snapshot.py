"""Rules-vs-rules 1v1 trace: both players Phase1RulesAI (which push-kicks on its
box run), 12 trials x 900 ticks, a hash of the full per-tick state plus the kick
count. Complements engine_trace_snapshot.py, whose scenarios fire only ~1
push-kick: this one fires ~57, so a change to the push-kick path (e.g. its
follow-through movement, engine/kick_intent.py) shows up here.

Usage (from the repo root):
    python scripts/rules_1v1_trace_snapshot.py OUT.json
    python scripts/rules_1v1_trace_snapshot.py --compare A.json B.json
"""
import functools
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())


def run() -> dict:
    sys_argv, sys.argv = sys.argv, ["x"]
    import scripts.engine_trace_snapshot as T   # installs the counter-seeded Random()
    sys.argv = sys_argv
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ui.scenarios import ScenarioDefinition, ScenarioLoop, build_1v1_scenario

    defn = ScenarioDefinition(
        key="rules_1v1", label="rules 1v1", description="rules 1v1",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0),
        box_possession_terminal=False,
    )
    h = hashlib.sha256()
    kicks = 0
    for trial_seed in range(12):
        random.seed(trial_seed)
        loop = ScenarioLoop(definition=defn, max_trials=0, linger_s=0.0, timeout_ticks=900)
        for p in loop.match.players:
            p.ai, p.current_order = Phase1RulesAI(), None
        for _ in range(900):
            if loop.step():
                for p in loop.match.players:
                    p.ai, p.current_order = Phase1RulesAI(), None
            m = loop.match
            kicks += sum(p.kicked_this_tick for p in m.players)
            h.update(json.dumps(T._state(m)).encode())
    return {"hash": h.hexdigest(), "kicks": kicks}


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
        print("0 differences" if a == b else f"DIFFERENT: {a} vs {b}")
        sys.exit(0 if a == b else 1)
    rec = run()
    json.dump(rec, open(sys.argv[1], "w"))
    print("wrote", sys.argv[1], rec)
