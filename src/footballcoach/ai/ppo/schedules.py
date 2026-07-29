"""Learning-rate, clip-range, and rng_reduction schedule helpers.

All schedules are functions of ``progress`` in [0.0, 1.0] where 0.0 is the
start of training and 1.0 is the end.  This matches the CleanRL convention
(anneal_lr etc.) and keeps the schedules decoupled from the training loop.

See ai_design_doc.md section 4 (curriculum item 7) for the rng_reduction
schedule rationale (start slightly boosted ~0.55, anneal to default 0.3).
"""
from __future__ import annotations


def constant(value: float):
    """No schedule: always returns the same value."""
    def _f(progress: float) -> float:
        return value
    return _f


def linear_anneal(start: float, end: float):
    """Linear interpolation from start to end as progress goes 0 -> 1."""
    def _f(progress: float) -> float:
        return start + (end - start) * progress
    return _f


def cosine_anneal(start: float, end: float):
    """Cosine annealing from start to end."""
    import math
    def _f(progress: float) -> float:
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    return _f


def rng_reduction_schedule(cfg: dict):
    """Build an rng_reduction schedule from ai_config.json['curriculum'].

    Linearly anneals from ``rng_reduction_start`` (e.g. 0.55) to
    ``rng_reduction_end`` (e.g. 0.3) over the course of training.
    """
    start = float(cfg.get("rng_reduction_start", 0.55))
    end = float(cfg.get("rng_reduction_end", 0.3))
    return linear_anneal(start, end)


class TrainingSchedules:
    """Convenience bundle of all schedules used by the PPO trainer.

    Args:
        ppo_cfg: The 'ppo' section of ai_config.json.
        curriculum_cfg: The 'curriculum' section of ai_config.json.
    """

    def __init__(self, ppo_cfg: dict, curriculum_cfg: dict):
        self.learning_rate = constant(float(ppo_cfg.get("learning_rate", 3e-4)))
        self.clip_range = constant(float(ppo_cfg.get("clip_range", 0.2)))
        self.rng_reduction = rng_reduction_schedule(curriculum_cfg)

    def lr(self, progress: float) -> float:
        return self.learning_rate(progress)

    def clip(self, progress: float) -> float:
        return self.clip_range(progress)

    def rng(self, progress: float) -> float:
        return self.rng_reduction(progress)
