"""> **Documentation must stay in sync with code.** (see agent_plans/masked_action_training_plan.md)

Rollout-level statistics + invariant counters for the per-decision action-opportunity flags
(entities/action_opportunity.py), computed from the ``action/*`` tensors of a collected batch. Pure functions on
tensors so they are trivially testable and identical for every rollout path.

Keys read (all (N,) or (N,1) float 0/1): ``opp_kick`` / ``opp_tack`` (cut, i.e. independence rule applied),
``opp_kick_raw`` / ``opp_tack_raw``, ``fired_kick`` / ``fired_tack``, ``opp_partial`` and the sampled bits
``kick`` / ``tackle_attempt``.
"""
from __future__ import annotations

from typing import Optional

import torch


def has_opportunity_flags(batch: dict) -> bool:
    return "action/opp_kick" in batch and "action/opp_tack" in batch and "action/fired_kick" in batch


def _v(batch: dict, key: str) -> torch.Tensor:
    return batch[f"action/{key}"].reshape(-1).float()


def _ratio(num: float, den: float) -> Optional[float]:
    return None if den <= 0 else num / den


def opportunity_masks(batch: dict) -> dict[str, torch.Tensor]:
    """Per-head float masks used for training. Missing flag keys => all-ones (old behaviour)."""
    n = len(batch["log_probs"])
    if not has_opportunity_flags(batch):
        ones = torch.ones(n)
        return {"kick": ones, "tackle_attempt": ones, "kick_dir": ones, "kick_power": ones}
    return {
        "kick": _v(batch, "opp_kick"),
        "tackle_attempt": _v(batch, "opp_tack"),
        "kick_dir": _v(batch, "fired_kick"),
        "kick_power": _v(batch, "fired_kick"),
    }


def opportunity_stats(batch: dict) -> Optional[dict]:
    """None when the batch carries no flags. Otherwise a dict of rates / counts / invariant-violation counters."""
    if not has_opportunity_flags(batch):
        return None
    n = float(len(batch["log_probs"]))
    ok, okr, fk = _v(batch, "opp_kick"), _v(batch, "opp_kick_raw"), _v(batch, "fired_kick")
    ot, otr, ft = _v(batch, "opp_tack"), _v(batch, "opp_tack_raw"), _v(batch, "fired_tack")
    kick, tack = _v(batch, "kick"), _v(batch, "tackle_attempt")
    partial = _v(batch, "opp_partial") if "action/opp_partial" in batch else torch.zeros_like(ok)
    k1, t1 = kick > 0.5, tack > 0.5

    def s(x: torch.Tensor) -> float:
        return float(x.sum())

    out: dict = {
        "n": n,
        "p_opp_kick": s(ok) / n, "p_opp_kick_raw": s(okr) / n,
        "p_opp_tack": s(ot) / n, "p_opp_tack_raw": s(otr) / n,
        "p_both_raw": s(okr * otr) / n,
        "kick_cut_share": _ratio(s(okr) - s(ok), s(okr)),      # raw kick opportunities dropped by the cross-head rule
        "tack_cut_share": _ratio(s(otr) - s(ot), s(otr)),
        "p_partial": s(partial) / n,
        # kick gate
        "p_kick": s(kick) / n,
        "p_kick_given_opp": _ratio(s(kick * ok), s(ok)),
        "p_kick_given_no_opp": _ratio(s(kick * (1 - ok)), s(1 - ok)),
        "p_fired_given_kick": _ratio(s(fk * kick), s(kick)),
        "p_fired_given_kick_opp": _ratio(s(fk * kick * ok), s(kick * ok)),
        "p_fired_given_kick_no_opp": _ratio(s(fk * kick * (1 - ok)), s(kick * (1 - ok))),
        "frac_kick_rows_with_opp": _ratio(s(kick * ok), s(kick)),
        "n_kick_bit": s(kick), "n_fired_kick": s(fk),
        # tackle gate
        "p_tack": s(tack) / n,
        "p_tack_given_opp": _ratio(s(tack * ot), s(ot)),
        "p_tack_given_no_opp": _ratio(s(tack * (1 - ot)), s(1 - ot)),
        "p_fired_given_tack": _ratio(s(ft * tack), s(tack)),
        "frac_tack_rows_with_opp": _ratio(s(tack * ot), s(tack)),
        "n_tack_bit": s(tack), "n_fired_tack": s(ft),
        # rows that will carry policy-gradient signal per head under masking
        "n_train_kick": s(ok), "n_train_tack": s(ot), "n_train_kick_params": s(fk),
    }
    # invariants -- every one must be exactly 0
    out["viol_flag_not_binary"] = float(sum(
        int(((x != 0) & (x != 1)).sum()) for x in (ok, okr, fk, ot, otr, ft, partial)
    ))
    out["viol_fired_kick_without_bit"] = s(fk * (~k1).float())
    out["viol_fired_kick_without_raw_opp"] = s(fk * (1 - okr))
    out["viol_fired_tack_without_bit"] = s(ft * (~t1).float())
    out["viol_fired_tack_without_raw_opp"] = s(ft * (1 - otr))
    out["viol_cut_not_subset_of_raw"] = s(ok * (1 - okr)) + s(ot * (1 - otr))
    out["violations"] = sum(v for k, v in out.items() if k.startswith("viol_"))
    return out


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


def format_opportunity_block(st: dict) -> list[str]:
    """Log lines for the ``[action opp]`` block."""
    return [
        f"[action opp] rows={int(st['n']):,}  P(kick opp)={_pct(st['p_opp_kick'])} (raw {_pct(st['p_opp_kick_raw'])})  "
        f"P(tackle opp)={_pct(st['p_opp_tack'])} (raw {_pct(st['p_opp_tack_raw'])})  "
        f"both-raw={_pct(st['p_both_raw'])}  cross-head cut: kick {_pct(st['kick_cut_share'])} of raw, "
        f"tackle {_pct(st['tack_cut_share'])} of raw  partial-interval rows={_pct(st['p_partial'])}",
        f"[action opp] kick gate: P(kick=1)={_pct(st['p_kick'])}  P(kick=1|opp)={_pct(st['p_kick_given_opp'])}  "
        f"P(kick=1|no opp)={_pct(st['p_kick_given_no_opp'])}  share of kick=1 rows with an opportunity={_pct(st['frac_kick_rows_with_opp'])}  "
        f"P(fired|kick=1)={_pct(st['p_fired_given_kick'])} (opp: {_pct(st['p_fired_given_kick_opp'])}, no opp: {_pct(st['p_fired_given_kick_no_opp'])})  "
        f"kick=1 rows={int(st['n_kick_bit']):,} fired={int(st['n_fired_kick']):,}",
        f"[action opp] tackle gate: P(tackle=1)={_pct(st['p_tack'])}  P(tackle=1|opp)={_pct(st['p_tack_given_opp'])}  "
        f"P(tackle=1|no opp)={_pct(st['p_tack_given_no_opp'])}  share of tackle=1 rows with an opportunity={_pct(st['frac_tack_rows_with_opp'])}  "
        f"P(fired|tackle=1)={_pct(st['p_fired_given_tack'])}  tackle=1 rows={int(st['n_tack_bit']):,} fired={int(st['n_fired_tack']):,}",
        f"[action opp] rows that would train: kick gate {int(st['n_train_kick']):,}  tackle gate {int(st['n_train_tack']):,}  "
        f"kick dir/power {int(st['n_train_kick_params']):,}   invariant violations: {int(st['violations'])}"
        + ("" if st["violations"] == 0 else "  <-- BUG: " + ", ".join(
            f"{k}={int(v)}" for k, v in st.items() if k.startswith("viol_") and k != "violations" and v)),
    ]
