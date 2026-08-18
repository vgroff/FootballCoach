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

    def __init__(self, ppo_cfg: dict, curriculum_cfg: dict, bc_cfg: dict | None = None):
        self.learning_rate = constant(float(ppo_cfg.get("learning_rate", 3e-4)))
        self.value_learning_rate = constant(
            float(ppo_cfg.get("value_learning_rate", ppo_cfg.get("learning_rate", 3e-4)))
        )
        self.clip_range = constant(float(ppo_cfg.get("clip_range", 0.2)))
        self.rng_reduction = rng_reduction_schedule(curriculum_cfg)
        bc = bc_cfg or {}
        _bc_start = float(bc.get("aux_coeff_start", 0.0))
        _bc_end = float(bc.get("aux_coeff_end", 0.0))
        _bc_frac = float(bc.get("aux_coeff_anneal_fraction", 1.0))
        _bc_inner = linear_anneal(_bc_start, _bc_end)
        def _bc_schedule(progress: float) -> float:
            return _bc_inner(min(progress / _bc_frac, 1.0)) if _bc_frac > 0.0 else _bc_end
        self.bc_aux_coeff = _bc_schedule

        # Kick-group BC aux coefficient: an independent coefficient for the
        # kick_this_tick/kick_direction/kick_power/kick_spin loss group (see
        # bc_loss_from_tensor(split_kick=True) in bc.py), so kick-related BC
        # pressure can be tuned separately from everything else -- e.g. set
        # aux_coeff_start/end to 0.0 and kick_aux_coeff_start/end to 1.0 to
        # train on kicking only for a round. Reuses aux_coeff_anneal_fraction
        # (same anneal shape/timing as the "other" schedule above) so there's
        # only one new pair of values to set: kick_aux_coeff_start/end.
        _bc_kick_start = float(bc.get("kick_aux_coeff_start", 0.0))
        _bc_kick_end = float(bc.get("kick_aux_coeff_end", 0.0))
        _bc_kick_inner = linear_anneal(_bc_kick_start, _bc_kick_end)
        def _bc_kick_schedule(progress: float) -> float:
            return _bc_kick_inner(min(progress / _bc_frac, 1.0)) if _bc_frac > 0.0 else _bc_kick_end
        self.bc_kick_aux_coeff = _bc_kick_schedule

        # Tackle-group BC aux coefficient: same idea as kick above, but for
        # the single tackle_attempt BCE term (see
        # bc_loss_from_tensor(split_tackle=True) in bc.py). Also reuses
        # aux_coeff_anneal_fraction.
        _bc_tackle_start = float(bc.get("tackle_aux_coeff_start", 0.0))
        _bc_tackle_end = float(bc.get("tackle_aux_coeff_end", 0.0))
        _bc_tackle_inner = linear_anneal(_bc_tackle_start, _bc_tackle_end)
        def _bc_tackle_schedule(progress: float) -> float:
            return _bc_tackle_inner(min(progress / _bc_frac, 1.0)) if _bc_frac > 0.0 else _bc_tackle_end
        self.bc_tackle_aux_coeff = _bc_tackle_schedule

        # Entropy coefficient anneal, same shape as bc_aux_coeff above. A fixed
        # ent_coef has no self-limiting mechanism: the entropy bonus's pull on
        # the loss (ent_coef * entropy) is the same magnitude every rollout,
        # while the policy-gradient term's opposing pull shrinks as advantages
        # shrink over training -- so without an anneal, entropy can win the
        # "tug of war" by default late in training even at a small ent_coef.
        # See ai_trainer_knowledge.md's "Reading the training log" entropy
        # discussion for the run110 case study this was added for.
        # ent_coef_start defaults to the legacy single `ent_coef` key (constant
        # behaviour, unchanged) when the new keys aren't present; ent_coef_end
        # defaults to ent_coef_start (also no-op) so existing configs are
        # unaffected until ent_coef_start/end/anneal_fraction are set explicitly.
        _legacy_ent = float(ppo_cfg.get("ent_coef", 0.01))
        _ent_start = float(ppo_cfg.get("ent_coef_start", _legacy_ent))
        _ent_end = float(ppo_cfg.get("ent_coef_end", _ent_start))
        _ent_frac = float(ppo_cfg.get("ent_coef_anneal_fraction", 1.0))
        _ent_inner = linear_anneal(_ent_start, _ent_end)
        def _ent_schedule(progress: float) -> float:
            return _ent_inner(min(progress / _ent_frac, 1.0)) if _ent_frac > 0.0 else _ent_end
        self.ent_coef = _ent_schedule

    def lr(self, progress: float) -> float:
        return self.learning_rate(progress)

    def value_lr(self, progress: float) -> float:
        return self.value_learning_rate(progress)

    def clip(self, progress: float) -> float:
        return self.clip_range(progress)

    def rng(self, progress: float) -> float:
        return self.rng_reduction(progress)

    def bc(self, progress: float) -> float:
        """BC auxiliary loss coefficient (non-kick heads) — linearly anneals to 0.0."""
        return self.bc_aux_coeff(progress)

    def bc_kick(self, progress: float) -> float:
        """BC auxiliary loss coefficient for the kick head group — see bc.py split_kick."""
        return self.bc_kick_aux_coeff(progress)

    def bc_tackle(self, progress: float) -> float:
        """BC auxiliary loss coefficient for the tackle_attempt head — see bc.py split_tackle."""
        return self.bc_tackle_aux_coeff(progress)

    def ent(self, progress: float) -> float:
        """Entropy bonus coefficient — see ent_coef schedule built in __init__."""
        return self.ent_coef(progress)
