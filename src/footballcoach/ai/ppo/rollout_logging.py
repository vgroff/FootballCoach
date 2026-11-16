"""PPOTrainer's per-rollout [PPO] summary log block."""
from __future__ import annotations

import math
from typing import Optional
import numpy as np
from footballcoach.ai.ppo.trainer_common import REWARD_COMP_LABELS, _PHASE1_OUTCOME_LEGEND, log, outcome_breakdown



class RolloutLoggingMixin:
    """Per-rollout summary logging for PPOTrainer. Mixed into PPOTrainer; relies on its attributes."""

    def _log_rollout_summary(
        self,
        metrics: dict,
        steps_per_sec: float,
        episode_rewards: list[float],
        secondary_episode_rewards: list[float],
        episode_outcomes_vs_rules: list[str],
        episode_outcomes_vs_immobile: list[str],
        episode_outcomes_vs_neural: list[str],
        rollout_components: dict[str, float],
        episode_comp_list: list[dict[str, float]],
        episode_durations_s: list[float],
        comp_step_stats: dict[str, dict],
        n_reward_comp_steps: int,
        episode_kick_counts: Optional[list[int]] = None,
        episode_armed_kick_counts: Optional[list[int]] = None,
        episode_outcomes_by_tag: Optional[dict[str, list[str]]] = None,
        episode_rewards_by_tag: Optional[dict[str, list[float]]] = None,
    ) -> None:
        """Emit the full multi-line per-rollout summary (reward breakdown,
        per-episode reward stats, per-component GAE/TD stats, direction
        log_std, action-head probabilities, outcome breakdown).

        Shared by the single-process ``train()`` loop and ``_train_parallel()``
        so both paths produce IDENTICAL diagnostics -- the parallel path
        previously logged a stripped-down one-liner instead of this, which
        was a real regression (see ai_trainer_knowledge.md "Parallel rollout
        collection").
        """
        mean_ep_reward = (
            float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
        )
        mean_opp_reward = (
            float(np.mean(secondary_episode_rewards[-20:])) if secondary_episode_rewards else float('nan')
        )
        _bc_tk = metrics.get("bc_tackle_loss", 0.0)
        _bc_kick_coeff = metrics.get("bc_kick_coeff", 0.0)
        _bc_tackle_coeff = metrics.get("bc_tackle_coeff", 0.0)
        bc_str = (
            f"  bc={metrics['bc_loss']:.4f}(x{metrics['bc_coeff']:.2f}"
            + (f"/kick x{_bc_kick_coeff:.2f}" if _bc_kick_coeff > 0.0 else "")
            + (f"/tackle x{_bc_tackle_coeff:.2f}" if _bc_tackle_coeff > 0.0 else "")
            + f")[tk={_bc_tk:.3f}]"
            if metrics.get("bc_coeff", 0.0) > 0.0 or _bc_kick_coeff > 0.0 or _bc_tackle_coeff > 0.0 else ""
        )
        # Per-phase / per-role BC (ai/ppo/bc_sources.py): its coefficients and
        # its (other, kick, tackle) group losses, one entry per source.
        bc_lines = []
        for _src, _v in (metrics.get("bc_by_source") or {}).items():
            _c, _l = _v["coeff"], _v["loss"]
            from footballcoach.ai.ppo.bc import GROUP_ROW_COLUMNS
            bc_str += (f"  bc[{_src}]=" + "/".join(
                f"{g} {l:.3f}x{c:.2f}" for g, c, l in zip(GROUP_ROW_COLUMNS, _c, _l) if c > 0.0))
            _a = _v.get("agree")
            if _a:
                # The policy vs its expert on the kick group (ppo_update._kick_agreement).
                bc_lines.append(
                    f"  bc agree [{_src}] expert kick-now={100 * _a['label_kick_rate']:.0f}% of rows  "
                    f"P(kick)|kick-now={_a['p_kick_label_kick']:.3f}  P(kick)|else={_a['p_kick_label_no_kick']:.3f}  "
                    f"|power err|={_a['power_err']:.3f}  kick_dir err={_a['kick_dir_err_deg']:.1f}deg")
        for _ph, _dv in (metrics.get("distill") or {}).items():
            # Teacher distillation of a phase (ai/ppo/distill.py): its mean KL to
            # the teacher and the largest per-head terms.
            _top = sorted(_dv["by_head"].items(), key=lambda kv: -abs(kv[1]))[:4]
            bc_lines.append(f"  distill [{_ph}] kl={_dv['kl']:.4f} x{_dv['coeff']:.2f}  ("
                            + "  ".join(f"{h} {v:.4f}" for h, v in _top) + ")")
            _da = _dv.get("agree")
            if _da:
                # The same, readable (ppo_update._distill_agreement).
                bc_lines.append(
                    f"  distill agree [{_ph}] move_dir err={_da['move_dir_err_deg']:.1f}deg  "
                    f"P(kick) student/teacher={_da['p_kick_student']:.3f}/{_da['p_kick_teacher']:.3f}  "
                    f"|power diff|={_da['power_diff']:.3f}  exec_move agree={100 * _da['exec_move_agree']:.1f}%  "
                    f"sprint agree={100 * _da['sprint_agree']:.1f}%")
        _p = metrics.get("bc_pull") or {}
        if _p:
            # How hard BC pulls on the shared weights (ppo_update._measure_bc_pull):
            # its weighted gradient vs everything else's, and their alignment.
            bc_lines.append(
                f"  bc pull  main |g_bc|={_p['bc_main']:.2f} |g_rest|={_p['rest_main']:.2f} "
                f"(ratio {_p['bc_main'] / max(_p['rest_main'], 1e-12):.2f}, cos {_p['cos_main']:+.2f})  "
                f"dir |g_bc|={_p['bc_dir']:.2f} |g_rest|={_p['rest_dir']:.2f} "
                f"(ratio {_p['bc_dir'] / max(_p['rest_dir'], 1e-12):.2f}, cos {_p['cos_dir']:+.2f})")
        # Phase-1 outcome breakdown over this rollout's episodes, split by
        # opponent type -- win/loss/timeout/miss(+other), see
        # outcome_breakdown() for the win%/loss%/tout%/miss% format.
        outcome_parts = []
        if episode_outcomes_vs_rules:
            outcome_parts.append(
                f"vs_rules({len(episode_outcomes_vs_rules)}): {outcome_breakdown(episode_outcomes_vs_rules)}"
            )
        if episode_outcomes_vs_immobile:
            outcome_parts.append(
                f"vs_immobile({len(episode_outcomes_vs_immobile)}): {outcome_breakdown(episode_outcomes_vs_immobile)}"
            )
        if episode_outcomes_vs_neural:
            outcome_parts.append(
                f"vs_neural({len(episode_outcomes_vs_neural)}): {outcome_breakdown(episode_outcomes_vs_neural)}"
            )
        outcome_str = (
            f"  vs[{_PHASE1_OUTCOME_LEGEND}]  " + "  ".join(outcome_parts)
        ) if outcome_parts else ""
        mv_ls = metrics.get('move_log_std', [])
        kk_ls = metrics.get('kick_log_std', [])
        kz_ls = metrics.get('kick_z_log_std', [])
        kp_ls = metrics.get('kick_power_log_std', [])
        mv_ls_grad = metrics.get('mv_ls_grad', 0.0)
        # These are actually log_kappa now (von Mises concentration), not
        # log_std -- see ai_trainer_knowledge.md "Direction heads: von Mises".
        # kappa=exp(log_kappa); for large kappa a von Mises approaches a
        # wrapped normal with variance ~= 1/kappa, so degrees(sqrt(1/kappa))
        # is the (approximate) angular std this actually controls -- the
        # NUMBER that actually controls how tightly direction samples
        # cluster around the predicted mean, same role log_std's sigma
        # played before, just via the inverse relationship (larger kappa =
        # narrower). Delta vs the previous rollout's value shows whether
        # log_kappa is actually moving at all (it was observed to sit frozen
        # at its init value across many rollouts when the policy LR is tiny
        # and early-stop cuts gradient steps short).
        def _kappa_deg(ls_pair):
            if not ls_pair:
                return None
            kappa = [math.exp(v) for v in ls_pair]
            deg = [math.degrees(math.sqrt(1.0 / k)) if k > 0 else float("inf") for k in kappa]
            return kappa, deg
        # kick_dir_z_log_std is a PLAIN Gaussian log_std (elevation component,
        # not a von Mises kappa) -- larger = wider, same convention the old
        # log_std parameters used, so sigma=exp(log_std) directly (no
        # kappa->degrees inversion needed here). See ai_trainer_knowledge.md
        # "Direction heads: von Mises".
        def _sigma(ls_pair):
            if not ls_pair:
                return None
            return [math.exp(v) for v in ls_pair]
        _mv_sig = _kappa_deg(mv_ls)
        _kk_sig = _kappa_deg(kk_ls)
        _kz_sig = _sigma(kz_ls)
        _kp_sig = _sigma(kp_ls)
        _prev_mv_ls = self._prev_move_log_std if hasattr(self, "_prev_move_log_std") else None
        _prev_kk_ls = self._prev_kick_log_std if hasattr(self, "_prev_kick_log_std") else None
        _prev_kz_ls = self._prev_kick_z_log_std if hasattr(self, "_prev_kick_z_log_std") else None
        _prev_kp_ls = self._prev_kick_power_log_std if hasattr(self, "_prev_kick_power_log_std") else None
        _mv_delta_str = ""
        if mv_ls and _prev_mv_ls:
            _d = [b - a for a, b in zip(_prev_mv_ls, mv_ls)]
            # Show Δlog_kappa and the resulting change in angular std (degrees)
            _mv_dstd_deg = [
                abs(math.degrees(math.sqrt(1.0 / math.exp(b))) - math.degrees(math.sqrt(1.0 / math.exp(a))))
                for a, b in zip(_prev_mv_ls, mv_ls)
            ]
            _mv_delta_str = (
                f"  d_move=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δ(ang std)≈{','.join(f'{v:.3f}°' for v in _mv_dstd_deg)})"
            )
        _kk_delta_str = ""
        if kk_ls and _prev_kk_ls:
            _d = [b - a for a, b in zip(_prev_kk_ls, kk_ls)]
            _kk_dstd_deg = [
                abs(math.degrees(math.sqrt(1.0 / math.exp(b))) - math.degrees(math.sqrt(1.0 / math.exp(a))))
                for a, b in zip(_prev_kk_ls, kk_ls)
            ]
            _kk_delta_str = (
                f"  d_kick=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δ(ang std)≈{','.join(f'{v:.3f}°' for v in _kk_dstd_deg)})"
            )
        _kz_delta_str = ""
        if kz_ls and _prev_kz_ls:
            _d = [b - a for a, b in zip(_prev_kz_ls, kz_ls)]
            # Same small-angle sigma->degrees approximation as the non-delta
            # kz_ls line above (accurate near-horizontal, z_mean~=0) -- kept
            # analogous to d_move/d_kick's Δ(ang std) rather than a raw Δσ.
            _kz_dstd_deg = [
                abs(math.degrees(math.exp(b)) - math.degrees(math.exp(a)))
                for a, b in zip(_prev_kz_ls, kz_ls)
            ]
            _kz_delta_str = (
                f"  d_kickz=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δ(ang std)≈{','.join(f'{v:.3f}°' for v in _kz_dstd_deg)})"
            )
        _kp_delta_str = ""
        if kp_ls and _prev_kp_ls:
            _d = [b - a for a, b in zip(_prev_kp_ls, kp_ls)]
            # kick_power isn't an angle (it's a sigmoid-squashed [0,1] power
            # fraction) -- no ang-std-style geometric conversion applies here,
            # unlike kz_ls's z-component-of-a-direction-vector case. Report
            # the raw sigma delta directly.
            _kp_dsig = [abs(math.exp(b) - math.exp(a)) for a, b in zip(_prev_kp_ls, kp_ls)]
            _kp_delta_str = (
                f"  d_kickpower=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δσ≈{','.join(f'{v:.4f}' for v in _kp_dsig)})"
            )
        self._prev_move_log_std = list(mv_ls) if mv_ls else None
        self._prev_kick_log_std = list(kk_ls) if kk_ls else None
        self._prev_kick_z_log_std = list(kz_ls) if kz_ls else None
        self._prev_kick_power_log_std = list(kp_ls) if kp_ls else None
        mv_ls_str = ""
        if mv_ls:
            mv_ls_str = f"  mv_ls=[{','.join(f'{v:.4f}' for v in mv_ls)}]"
            if _mv_sig:
                _kap, _deg = _mv_sig
                mv_ls_str += f" (\u03ba\u2248{','.join(f'{k:.2f}' for k in _kap)}, ang std\u2248{','.join(f'{d:.0f}\u00b0' for d in _deg)})"
            mv_ls_str += f" g={mv_ls_grad:.2e}" + _mv_delta_str
        if kk_ls:
            mv_ls_str += f"\n  kk_ls=[{','.join(f'{v:.4f}' for v in kk_ls)}]"
            if _kk_sig:
                _kap, _deg = _kk_sig
                mv_ls_str += f" (\u03ba\u2248{','.join(f'{k:.2f}' for k in _kap)}, ang std\u2248{','.join(f'{d:.0f}\u00b0' for d in _deg)})"
            mv_ls_str += _kk_delta_str
        if kz_ls:
            mv_ls_str += f"\n  kz_ls=[{','.join(f'{v:.4f}' for v in kz_ls)}]"
            if _kz_sig:
                # kick_dir_z_log_std's sigma is a std on the raw z-COMPONENT of
                # the kick direction unit vector, not an angle -- elevation
                # angle = arcsin(z), and d(arcsin)/dz = 1/sqrt(1-z^2) = 1 at
                # z=0, so treating sigma directly as a small-angle radian
                # estimate is only accurate for near-horizontal kicks (mean
                # elevation near 0); it under-reports the true angular spread
                # for kicks with a large mean elevation (|z| closer to 1),
                # same caveat the move/kick_dir kappa->degrees conversion
                # already carries for its own small-angle approximation.
                _kz_deg = [math.degrees(s) for s in _kz_sig]
                mv_ls_str += (
                    f" (\u03c3\u2248{','.join(f'{s:.4f}' for s in _kz_sig)}"
                    f", ang std\u2248{','.join(f'{d:.0f}\u00b0' for d in _kz_deg)})"
                )
            mv_ls_str += _kz_delta_str
        if kp_ls:
            mv_ls_str += f"\n  kp_ls=[{','.join(f'{v:.4f}' for v in kp_ls)}]"
            if _kp_sig:
                # kick_power is a plain Gaussian log_std (same convention as
                # kz_ls -- larger = wider), but unlike kz_ls it's not any
                # kind of direction/angle component (SquashedNormalHead
                # sigmoid-squashes it to a [0,1] power fraction) -- no
                # geometric ang-std interpretation applies, so just report
                # sigma directly.
                mv_ls_str += f" (σ≈{','.join(f'{s:.4f}' for s in _kp_sig)})"
            mv_ls_str += _kp_delta_str
            _kpf = metrics.get("kick_power_frac")
            if _kpf and _kpf.get("n", 0) > 0:
                # The power fraction actually sampled on this rollout's kick rows
                # (sigmoid of the stored raw draw), not the log_std parameter.
                mv_ls_str += f"  | sampled power over {_kpf['n']:,} kicks: mean={_kpf['mean']:.3f} std={_kpf['std']:.3f}"
        ha = metrics.get("head_act", {})
        _ta_p = ha.get('ta_p', float('nan'))
        _kk_p = ha.get('kk_p', float('nan'))
        _prob_str = (
            (f" tackle_prob={_ta_p:.4f}" if _ta_p == _ta_p else "")
            + (f" kick_prob={_kk_p:.4f}" if _kk_p == _kk_p else "")
        )
        act_str = (
            f"  act: move={ha.get('mv','?'):>3} get_poss={ha.get('gp','?'):>3}"
            f" exec_move={ha.get('emv','?'):>3} sprint={ha.get('spr','?'):>3}"
            f" kick={ha.get('kck','?'):>3} tackle={ha.get('tk','?'):>3}"
            f" shoot={ha.get('sh','?'):>3} hold={ha.get('hld','?'):>3}"
            + _prob_str
        ) if ha else ""
        opp_rew_str = f"/{mean_opp_reward:.2f}" if not (mean_opp_reward != mean_opp_reward) else ""
        comp_parts = [
            f"{label}={rollout_components[k]:+.2f}"
            for k, label in REWARD_COMP_LABELS
            if abs(rollout_components.get(k, 0.0)) > 0.01
        ]
        comp_str = ("  rew: " + "  ".join(comp_parts)) if comp_parts else ""

        # Per-episode reward statistics table (mean/std/min/max per type).
        # Collect all keys that appeared in any episode this rollout,
        # then emit a compact aligned table.
        _rew_stats_lines: list[str] = []
        _n_ep_for_stats = len(episode_comp_list)
        if episode_comp_list:
            _all_keys = [k for k, _ in REWARD_COMP_LABELS
                         if any(k in ep for ep in episode_comp_list)]
            if _all_keys:
                _col_w = 14  # display-label column width
                _hdr = f"  {'component':<{_col_w}}  {'mean':>8}  {'std':>7}  {'min':>8}  {'max':>8}"
                _sep = "  " + "-" * _col_w + "  " + "-" * 8 + "  " + "-" * 7 + "  " + "-" * 8 + "  " + "-" * 8
                _rew_stats_lines.append(_hdr)
                _rew_stats_lines.append(_sep)
                _lbl_map = {k: lbl for k, lbl in REWARD_COMP_LABELS}
                for _k in _all_keys:
                    _vals = [ep[_k] for ep in episode_comp_list if _k in ep]
                    if not _vals:
                        continue
                    _arr = np.array(_vals)
                    # Same "never fired / always exactly zero this rollout"
                    # skip as the "rew/step" table's own "_cs['mean'] == 0.0
                    # and _cs['std'] == 0.0" check -- mean==0 and std==0
                    # together mean every episode's value for this
                    # component was exactly 0.0, so there's nothing to show.
                    # Component identity/whether it's config-disabled can
                    # change between rollouts (a coefficient retuned in
                    # ai_config.json, a curriculum phase change), so this is
                    # a per-rollout runtime check, not a permanent removal
                    # from REWARD_COMP_LABELS -- a component that starts
                    # firing again just reappears on its own next rollout,
                    # nothing to keep in sync by hand.
                    if _arr.mean() == 0.0 and _arr.std() == 0.0:
                        continue
                    _lbl = _lbl_map.get(_k, _k)
                    _rew_stats_lines.append(
                        f"  {_lbl:<{_col_w}}  {_arr.mean():>+8.3f}  {_arr.std():>7.3f}"
                        f"  {_arr.min():>+8.3f}  {_arr.max():>+8.3f}"
                    )

        _val_diag_str = (
            f"V={metrics['values_mean']:.3f}\u00b1{metrics['values_std']:.3f}  "
            f"R={metrics['returns_mean']:.3f}\u00b1{metrics['returns_std']:.3f}  "
            f"adv={metrics['adv_mean']:.3f}\u00b1{metrics['adv_std']:.3f}"
            f"  |adv|={metrics.get('adv_abs_mean', float('nan')):.3f}"
        )
        # Per-head entropy breakdown + \u0394 from the previous rollout \u2014 shows
        # which heads are actually driving a rising aggregate `entropy=`
        # scalar (e.g. a few Bernoulli heads saturating toward p=0.5, or a
        # continuous direction head's log_std climbing toward its clamp)
        # instead of only the summed total. |adv| above is the companion
        # number for the "tug of war" read: entropy's pull on the loss is a
        # FIXED ent_coef every step, while the policy-gradient term's pull
        # scales with |advantage| \u2014 as the policy converges and advantages
        # shrink, entropy's constant pull increasingly wins by default with
        # nothing here to anneal it back down. See ai_trainer_knowledge.md
        # "Reading the training log" / entropy-runaway discussion.
        # Masked/frozen/inactive heads (see _inactive_head_lp_keys) are
        # explicitly zeroed inside _compute_entropy -- a real, exact 0.0000,
        # not a residual -- but showing a wall of guaranteed-zero entries
        # here just buries the heads that are actually live. Dropped
        # entirely rather than displayed as 0.0000.
        _inactive_heads = self._inactive_head_lp_keys()
        _ent_bkdn = {
            k: v for k, v in metrics.get("entropy_breakdown", {}).items()
            if k not in _inactive_heads
        }
        if _ent_bkdn:
            _prev = self._prev_entropy_breakdown or {}
            _ent_parts = [
                f"{k}={v:.4f}({'+' if v - _prev.get(k, v) >= 0 else ''}{v - _prev.get(k, v):.4f})"
                if k in _prev else f"{k}={v:.4f}"
                for k, v in _ent_bkdn.items()
            ]
            self._prev_entropy_breakdown = dict(_ent_bkdn)
        # Tabulated multi-line rollout summary (readability refactor only —
        # every field present in the old single-line format is still here,
        # just grouped/aligned, with full-word labels and long lines wrapped
        # to avoid overflow. See
        # agent_plans/bc_execution_label_boundary_and_followups.md Part 4.
        _lines = [
            "\u2500" * 70,
            f"[PPO] step={self._total_steps:,}  speed={steps_per_sec:.0f}/s  "
            f"reward={mean_ep_reward:.2f}{opp_rew_str}",
            f"  loss     policy={metrics['policy_loss']:.4f}  "
            f"value={metrics['value_loss']:.4f}(x{self.vf_coef})={self.vf_coef * metrics['value_loss']:.4f}"
            f"  val_pre={metrics['pre_update_value_loss']:.4f}",
            f"           ent_coef={metrics['ent_coef']:.4f}  entropy={metrics['entropy']:.4f}"
            f"  kl={metrics['approx_kl']:.4f}"
            + (f"  {bc_str.strip()}" if bc_str else ""),
            f"  value    {_val_diag_str}",
        ] + bc_lines
        _fc = metrics.get("flip_consistency")
        if _fc:
            _m = _fc["metrics"]
            _inact = self._inactive_head_lp_keys()
            _lines.append(
                f"  flip     consistency KL={_fc['kl']:.4f}  move_dir err p50/p90={_m['move_dir_err_p50_deg']:.1f}/{_m['move_dir_err_p90_deg']:.1f} deg"
                f"  >90deg={_m['move_dir_err_gt90_pct']:.1f}%  hard-decision disagree%: "
                + "  ".join(f"{h}={_m.get(k, float('nan')):.1f}" for h, k in (
                    ("exec_move", "exec_move_sign_disagree_pct"), ("sprint", "sprint_sign_disagree_pct"),
                    ("kick", "kick_sign_disagree_pct"), ("tackle_att", "tackle_attempt_sign_disagree_pct")))
            )
            _lines.append(
                "           by head: " + "  ".join(f"{k}={v:.3f}" for k, v in _fc["per_head"].items()
                                                    if k not in _inact and v >= 0.0005)
                + f"  (n={_fc['n']:,} state pairs, pre-update weights)"
            )
        if _ent_bkdn:
            _mid_e = (len(_ent_parts) + 1) // 2
            _lines.append(f"  entropy  {'  '.join(_ent_parts[:_mid_e])}")
            if _ent_parts[_mid_e:]:
                _lines.append(f"           {'  '.join(_ent_parts[_mid_e:])}")
        if mv_ls_str:
            _mv_ls_sublines = mv_ls_str.strip().split("\n")
            _lines.append(f"  moves    {_mv_ls_sublines[0].strip()}")
            for _sub in _mv_ls_sublines[1:]:
                _lines.append(f"           {_sub.strip()}")
        if act_str:
            _act_body = act_str.strip().lstrip('act:').strip()
            _act_parts = _act_body.split("  ")
            _mid = (len(_act_parts) + 1) // 2
            _lines.append(f"  heads    {'  '.join(_act_parts[:_mid])}")
            if _act_parts[_mid:]:
                _lines.append(f"           {'  '.join(_act_parts[_mid:])}")
        if outcome_str:
            _lines.append(f"  vs       {outcome_str.strip()}")
        # Role-based phases (phase 2+): one line per phase/controller-mix tag
        # (StepInfo.episode_tag), outcomes in the phase's own vocabulary order,
        # plus the trainee's mean terminal reward (the trainee is the tag's
        # first neural role). Absent when no such episode ran, so phase-1-only
        # summaries are unchanged.
        for _tag in sorted(episode_outcomes_by_tag or {}):
            _lines.append("  " + format_tag_outcomes(
                _tag, episode_outcomes_by_tag[_tag], (episode_rewards_by_tag or {}).get(_tag, [])))
        if episode_durations_s:
            _dur_arr = np.array(episode_durations_s)
            _lines.append(
                f"  ep_len   {_dur_arr.mean():.1f}\u00b1{_dur_arr.std():.1f}s"
                f"  (n={len(episode_durations_s)}, min={_dur_arr.min():.1f}s, max={_dur_arr.max():.1f}s)"
            )
        if episode_kick_counts:
            _kk = np.array(episode_kick_counts, dtype=np.float64)
            _ka = np.array(episode_armed_kick_counts or [], dtype=np.float64)
            _armed_str = (
                f"  first-touch(armed)={_ka.mean():.3f}/ep ({100.0 * _ka.sum() / max(_kk.sum(), 1.0):.0f}% of kicks)"
                if len(_ka) == len(_kk) else ""
            )
            _lines.append(
                f"  kicks    {_kk.mean():.3f}±{_kk.std():.3f}/ep (trainee, all real kicks)"
                f"{_armed_str}  0-kick eps={100.0 * float((_kk == 0).mean()):.0f}%  max={int(_kk.max())}  (n={len(_kk)})"
            )
        if comp_str:
            _comp_body = comp_str.strip().lstrip('rew:').strip()
            _comp_parts = _comp_body.split("  ")
            _mid = (len(_comp_parts) + 1) // 2
            _lines.append(f"  reward   {'  '.join(_comp_parts[:_mid])}")
            if _comp_parts[_mid:]:
                _lines.append(f"           {'  '.join(_comp_parts[_mid:])}")
        if _rew_stats_lines:
            _lines.append(f"  rew/ep   (mean/std/min/max per episode, {_n_ep_for_stats} ep)")
            for _sl in _rew_stats_lines:
                _lines.append(_sl)
        if comp_step_stats:
            _cw = 14
            _lines.append(f"  rew/step (per-step stats, n={n_reward_comp_steps} steps; ret/gae/td at steps where component fired)")
            _lines.append(
                f"  {'component':<{_cw}}  {'count':>6}  {'mean':>8}  {'std':>7}"
                f"  {'mean_ret':>9}  {'std_ret':>8}  {'mean_gae':>9}"
                f"  {'mean_sq_td':>10}  {'mean|td|':>9}  {'p95|td|':>8}"
            )
            _lines.append("  " + "-" * _cw + "  " + "  ".join(["-"*6, "-"*8, "-"*7, "-"*9, "-"*8, "-"*9, "-"*10, "-"*9, "-"*8]))
            for _ck, _clabel in REWARD_COMP_LABELS:
                _cs = comp_step_stats.get(_ck)
                if _cs is None or (_cs["mean"] == 0.0 and _cs["std"] == 0.0):
                    continue
                def _fmt(v: float, w: int, prec: int = 3) -> str:
                    return f"{v:>+{w}.{prec}f}" if v == v else f"{'nan':>{w}}"
                def _fmtu(v: float, w: int, prec: int = 3) -> str:
                    return f"{v:>{w}.{prec}f}" if v == v else f"{'nan':>{w}}"
                _lines.append(
                    f"  {_clabel:<{_cw}}  {_cs['count']:>6d}  {_fmt(_cs['mean'],8)}  {_cs['std']:>7.3f}"
                    f"  {_fmt(_cs['mean_ret'],9)}  {_cs['std_ret']:>8.3f}  {_fmt(_cs['mean_gae'],9)}"
                    f"  {_fmtu(_cs['mean_sq_td'],10,4)}  {_fmtu(_cs['mean_abs_td'],9,3)}  {_fmtu(_cs['p95_td'],8,3)}"
                )
        _lines.append(
            f"  gae/td   mean_return={metrics['returns_mean']:+.3f}"
            f"  std_return={metrics['returns_std']:.3f}"
            f"  mean_gae={metrics['adv_mean']:+.3f}"
            f"  mean_sq_td={metrics['mean_sq_td']:.4f}"
        )
        _lines.append("\u2500" * 70)
        log.info("\n".join(_lines))


def format_tag_outcomes(tag: str, outcomes: list[str], trainee_rewards: list[float]) -> str:
    """"<tag>(n): goal=40.0% saved=...  trainee_rew=+1.234" for one episode tag."""
    from footballcoach.ai.curriculum.envs import outcome_keys_for_tag
    n = len(outcomes)
    keys = list(outcome_keys_for_tag(tag)) or sorted(set(outcomes))
    keys += sorted(set(outcomes) - set(keys))
    parts = " ".join(f"{k}={100.0 * outcomes.count(k) / n:.1f}%" for k in keys) if n else "n/a"
    rew = f"  trainee_rew={float(np.mean(trainee_rewards)):+.3f}" if trainee_rewards else ""
    return f"{tag}({n}): {parts}{rew}"
