"""Module-level helpers shared by ppo_trainer.py and its PPOTrainer mixin modules."""
from __future__ import annotations

import logging
import torch


log = logging.getLogger("footballcoach.ai.ppo")

def _load_state_dict_tolerant(module: torch.nn.Module, ckpt_sd: dict, label: str) -> None:
    """Load ckpt_sd into module, skipping any param whose shape no longer
    matches (e.g. value_head grew an extra input column since the
    checkpoint was saved) instead of raising. Skipped params keep their
    fresh random init; everything else loads normally. Lets checkpoints
    saved under an older network architecture keep loading after additive
    changes to value_head/value_ai_type_channel sizing, at the cost of
    those specific params needing to be retrained (typically fine, since
    such changes are usually paired with --reset-value-weights anyway)."""
    own_sd = module.state_dict()
    compatible = {}
    skipped = []
    for k, v in ckpt_sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    module.load_state_dict(compatible, strict=False)
    if skipped:
        log.warning(
            f"{label}: skipped {len(skipped)} shape-mismatched param(s) "
            f"(network architecture changed since checkpoint was saved), "
            f"keeping fresh init for: {skipped}"
        )

# Reward component short-key → display label mapping (order = display order).
# Used by both the per-rollout log and the pre-training diagnostic in train.py.
REWARD_COMP_LABELS: list[tuple[str, str]] = [
    ("out",   "ball_out"),
    ("box",   "box_possession"),
    ("spd",   "speed_bonus"),
    ("lterm", "opponent_box"),
    ("tout",  "timeout"),
    ("sprint", "sprint_penalty"),
    ("appr",  "approach"),
    ("prog",  "progress"),
]

# All phase-1 StepInfo.trial_outcome values (see ScenarioEnv.step()) that
# outcome_breakdown() below always reports a percentage for, even when a
# given rollout/eval happens to have zero of them (0% rather than silently
# omitted) — win/loss stay first for backward-compat with old log-scrapers
# that split on "/".
_PHASE1_OUTCOME_KEYS: list[tuple[str, str]] = [
    ("box_possession", "win"),
    ("opponent_box_possession", "loss"),
    ("timeout", "tout"),
    ("miss", "miss"),
    ("invalid", "inval"),
]

def outcome_breakdown(outcomes: list[str]) -> str:
    """Format a list of StepInfo.trial_outcome strings as
    "win%/loss%/tout%/miss%/inval%[/other%]" -- a fuller breakdown than just
    win/loss so a swing in win% can be traced to (e.g.) more timeouts vs.
    more losses vs. more ball-out-of-play, instead of both being lumped into
    an invisible remainder. "inval" is a ball-out with no toucher at all
    (nobody's fault, e.g. a bad initial placement) as opposed to "miss"
    (the last toucher is blamed). 'other' only appears if some outcome value
    isn't one of the known keys above (e.g. a phase-2 "goal"/"dispossessed"
    leaking through, or an "unknown" from a missing info object).
    """
    n = len(outcomes)
    if n == 0:
        return "n/a"
    parts = [f"{outcomes.count(key) / n * 100:.1f}%" for key, _ in _PHASE1_OUTCOME_KEYS]
    known = sum(outcomes.count(key) for key, _ in _PHASE1_OUTCOME_KEYS)
    other = n - known
    if other > 0:
        parts.append(f"{other / n * 100:.0f}%")
    return "/".join(parts)

# Slash-joined short labels matching outcome_breakdown()'s column order, for
# a one-time "vs[...]:" legend prefix on log lines instead of repeating the
# key names on every vs_rules/vs_immobile/vs_neural segment.
_PHASE1_OUTCOME_LEGEND = "/".join(label for _, label in _PHASE1_OUTCOME_KEYS)
