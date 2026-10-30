"""The 4 standard training-run diagnostic plots, from a run's own
training_logN.txt files: eval reward/win%, value function (V vs R), kicking
behaviour, and per-head entropy/KL. Kept in the repo (not a scratch/temp
script) because it gets reused constantly across a run's lifetime -- a prior
copy living only in a session's scratch directory was silently wiped by a
routine temp-folder cleanup after an 11-day gap.

Usage: plot_training_curves.py RUN_DIR OUT_PREFIX
Writes OUT_PREFIX_eval.png, OUT_PREFIX_value.png, OUT_PREFIX_kick.png,
OUT_PREFIX_entropy_kl.png.
"""
import re
import sys
import types
from pathlib import Path

if "xml.parsers.expat" not in sys.modules:
    _fake_expat = types.ModuleType("xml.parsers.expat")
    _fake_expat.ParserCreate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stubbed"))
    _fake_expat.ExpatError = type("ExpatError", (Exception,), {})
    sys.modules["xml.parsers.expat"] = _fake_expat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run, out_prefix = Path(sys.argv[1]), sys.argv[2]
WINDOW = 300_000_000  # last-300M-steps window, matching this repo's own convention

# ---------------------------------------------------------------- regexes --
R_STEP = re.compile(r"^\[PPO\] step=([\d,]+)", re.M)
R_VALUE = re.compile(r"^  value\s+V=([-\d.]+)±([\d.]+)\s+R=([-\d.]+)±([\d.]+)", re.M)
R_VALPRE = re.compile(r"val_pre=([\d.]+)")
R_VALLOSS = re.compile(r"value=[\d.]+\(x[\d.]+\)=([\d.]+)")
R_ENT1 = re.compile(r"^  entropy\s+move=[-\d.]+\([-+\d.]+\)\s+gp_extra=[-\d.]+\([-+\d.]+\)\s+"
                     r"exec_move=([-\d.]+)\([-+\d.]+\)\s+kick=([-\d.]+)\([-+\d.]+\)\s+"
                     r"tackle_attempt=([-\d.]+)\([-+\d.]+\)\s+sprint=([-\d.]+)\([-+\d.]+\)", re.M)
R_ENT2 = re.compile(r"^\s+move_dir=([-\d.]+)\([-+\d.]+\)\s+kick_dir=([-\d.]+)\([-+\d.]+\)\s+"
                     r"kick_dir_azimuth=[-\d.]+\([-+\d.]+\)\s+kick_dir_z=([-\d.]+)\([-+\d.]+\)\s+"
                     r"kick_power=([-\d.]+)\([-+\d.]+\)", re.M)
R_KL = re.compile(r"\[per-head KL\] ((?:\w+=[-+\d.]+\s*)+)")
R_KK = re.compile(r"kk_ls=\[[-\d.]+\] \(κ≈([\d.]+)")
R_KZ = re.compile(r"kz_ls=\[[-\d.]+\] \(σ≈([\d.]+)")
R_KP = re.compile(r"kp_ls=\[[-\d.]+\] \(σ≈([\d.]+)")
R_HEADS = re.compile(r"^  heads\s+move=\s*(\d+)\s+get_poss=\s*(\d+)\s+exec_move=\s*(\d+)\s+sprint=\s*(\d+)", re.M)
R_KICKPROB = re.compile(r"kick_prob=([\d.]+)")
R_KICKS = re.compile(r"kicks\s+([\d.]+)±([\d.]+)/ep.*?0-kick eps=(\d+)%")

R_EVAL_RULES = re.compile(r"\[eval vs rules\] step=([\d,]+).*?win=(\d+)%\s+mean_rew=([-\d.]+)±([\d.]+)\s+\(sem=([\d.]+)\)")
R_EVAL_NEURAL = re.compile(r"\[eval vs neural:(\w+(?:\.\w+)?)\] step=([\d,]+).*?win=(\d+)%\s+mean_rew=([-\d.]+)±([\d.]+)\s+\(sem=([\d.]+)\)")
R_EVAL_BASE = re.compile(r"\[eval baseline rules-vs-rules\].*?win=(\d+)%\s+mean_rew=([-\d.]+)±([\d.]+)\s+\(sem=([\d.]+)\)")
R_PPG = re.compile(r"\[ppg interleave\] triggered")

# ---------------------------------------------------------------- parse ----
files = sorted(run.glob("training_log*.txt"), key=lambda f: int(re.search(r"(\d+)\.txt", f.name).group(1)))

TYPES = ["rules", "original", "1_back", "0.5_back", "baseline"]
LABELS = {"rules": "vs rules", "original": "vs neural:original", "1_back": "vs neural:1_back",
          "0.5_back": "vs neural:0.5_back", "baseline": "baseline rules-vs-rules"}
eval_data = {t: [] for t in TYPES}
rows = []
ppg_count = 0

for f in files:
    n = int(re.search(r"(\d+)\.txt", f.name).group(1))
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    ppg_count += len(R_PPG.findall(text))

    rules_step = None
    m = R_EVAL_RULES.search(text)
    if m:
        rules_step = int(m.group(1).replace(",", ""))
        eval_data["rules"].append((rules_step, int(m.group(2)), float(m.group(3)), float(m.group(5))))
    for m in R_EVAL_NEURAL.finditer(text):
        if m.group(1) in eval_data:
            eval_data[m.group(1)].append((int(m.group(2).replace(",", "")), int(m.group(3)), float(m.group(4)), float(m.group(6))))
    m = R_EVAL_BASE.search(text)
    if m and rules_step is not None:
        eval_data["baseline"].append((rules_step, int(m.group(1)), float(m.group(2)), float(m.group(4))))

    step_m = R_STEP.search(text)
    if not step_m:
        continue
    row = {"log": n, "step": int(step_m.group(1).replace(",", ""))}
    m = R_VALUE.search(text)
    if m:
        row["V"], row["V_std"], row["R"], row["R_std"] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    m = R_VALPRE.search(text)
    if m:
        row["val_pre"] = float(m.group(1))
    m = R_VALLOSS.search(text)
    if m:
        row["value_loss"] = float(m.group(1))
    m1, m2 = R_ENT1.search(text), R_ENT2.search(text)
    if m1 and m2:
        row["exec_move_ent"], row["kick_ent"], row["tackle_ent"], row["sprint_ent"] = (
            float(m1.group(1)), float(m1.group(2)), float(m1.group(3)), float(m1.group(4)))
        row["move_dir_ent"], row["kick_dir_ent"], row["kick_dir_z_ent"], row["kick_power_ent"] = (
            float(m2.group(1)), float(m2.group(2)), float(m2.group(3)), float(m2.group(4)))
    m = R_KL.search(text)
    if m:
        for km in re.finditer(r"(\w+)=([-+\d.]+)", m.group(1)):
            row[f"kl_{km.group(1)}"] = float(km.group(2))
    m = R_KK.search(text)
    if m:
        row["kick_dir_kappa"] = float(m.group(1))
    m = R_KZ.search(text)
    if m:
        row["kick_dir_z_sigma"] = float(m.group(1))
    m = R_KP.search(text)
    if m:
        row["kick_power_sigma"] = float(m.group(1))
    m = R_HEADS.search(text)
    if m:
        row["exec_move_pct"], row["sprint_pct"] = int(m.group(3)), int(m.group(4))
    m = R_KICKPROB.search(text)
    if m:
        row["kick_prob"] = float(m.group(1))
    m = R_KICKS.search(text)
    if m:
        row["kicks_per_ep"] = float(m.group(1))
        row["zero_kick_pct"] = int(m.group(3))
    rows.append(row)

if not rows:
    print("No PPO iterations found.")
    sys.exit(1)

max_step = rows[-1]["step"]
rows_w = [r for r in rows if r["step"] >= max_step - WINDOW]
for t in TYPES:
    eval_data[t] = sorted([p for p in eval_data[t] if p[0] >= max_step - WINDOW], key=lambda p: p[0])

print(f"{len(rows_w)} iterations (last 300M steps), log {rows_w[0]['log']}..{rows_w[-1]['log']}, "
      f"steps {rows_w[0]['step']/1e6:.1f}M..{rows_w[-1]['step']/1e6:.1f}M; {ppg_count} PPG interleave round(s) in window")


def smooth(y, k=5):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def series(key):
    return np.array([r.get(key, np.nan) for r in rows_w], dtype=float)


step = np.array([r["step"] for r in rows_w]) / 1e6
sx = step[4:] if len(step) >= 5 else step
title_suffix = f"({len(rows_w)} iterations, {step[0]:.0f}M to {step[-1]:.0f}M steps; thin = raw, thick = 5-iteration mean)"

# ============================================================== figure 1 ==
COL = {"rules": "#c0392b", "original": "#2980b9", "1_back": "#e67e22", "0.5_back": "#8e44ad", "baseline": "black"}
fig = plt.figure(figsize=(21, 10.5))
gs = fig.add_gridspec(2, 3)
ax_rew, ax_win = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_last5, ax_last10, ax_last20, ax_all = (fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]),
                                           fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))
print(f"\n{'':28s}n{'':>4s}reward first5 last5 best / win first5 last5")
for t in TYPES:
    pts = eval_data[t]
    if not pts:
        continue
    es = np.array([p[0] for p in pts]) / 1e6
    win = np.array([p[1] for p in pts], dtype=float)
    rew = np.array([p[2] for p in pts])
    sem = np.array([p[3] for p in pts])
    style = "--" if t == "baseline" else "-"
    ax_rew.plot(es, rew, style, color=COL[t], alpha=0.25, lw=1)
    ax_rew.fill_between(es, rew - sem, rew + sem, color=COL[t], alpha=0.08)
    sm = smooth(rew)
    ax_rew.plot(es[len(es) - len(sm):], sm, style, color=COL[t], lw=2.2, label=LABELS[t])
    smw = smooth(win)
    ax_win.plot(es, win, style, color=COL[t], alpha=0.25, lw=1)
    ax_win.plot(es[len(es) - len(smw):], smw, style, color=COL[t], lw=2.2, label=LABELS[t])

    first5, last5 = rew[:5].mean(), rew[-5:].mean()
    best_i = int(np.argmax(rew))
    print(f"{LABELS[t]:28s}n={len(pts):3d}  reward first5 {first5:+.3f} last5 {last5:+.3f} best "
          f"{rew[best_i]:+.3f}@{es[best_i]:.0f}M   win first5 {win[:5].mean():.1f}% last5 {win[-5:].mean():.1f}%")

    def summary_panel(ax, n_last):
        sub = rew[-n_last:] if len(rew) >= n_last else rew
        if len(sub) == 0:
            return
        mean, std = sub.mean(), sub.std()
        delta = (sub[-1] - sub[0]) / max(1, len(sub) - 1)
        x = TYPES.index(t)
        ax.errorbar(x, mean, yerr=std, fmt="o", color=COL[t], capsize=4)
        ax.annotate(f"{mean:+.3f}\n(Δ{delta:+.3f}/eval)", (x, mean), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=COL[t])

    summary_panel(ax_last5, 5); summary_panel(ax_last10, 10); summary_panel(ax_last20, 20); summary_panel(ax_all, len(rew))

ax_rew.set_title("Mean eval reward by type"); ax_rew.set_xlabel("env steps (millions)"); ax_rew.legend(fontsize=8)
ax_win.set_title("Eval win % by type"); ax_win.set_xlabel("env steps (millions)"); ax_win.legend(fontsize=8)
for ax, ttl in [(ax_last5, "Last 5 evals: mean reward (± std); Δ = mean eval-to-eval change within those 5"),
                (ax_last10, "Last 10 evals: mean reward (± std); Δ = mean eval-to-eval change within those 10"),
                (ax_last20, "Last 20 evals: mean reward (± std); Δ = mean eval-to-eval change within those 20"),
                (ax_all, "All evals THIS run: mean reward (± std); Δ = mean eval-to-eval change")]:
    ax.set_title(ttl, fontsize=10)
    ax.set_xticks(range(len(TYPES))); ax.set_xticklabels([LABELS[t].replace("vs ", "").replace("baseline ", "") for t in TYPES], rotation=20)
fig.suptitle(f"{run.name}: eval results over the run  (thin = raw per eval, thick = 5-eval mean; band = ±sem of a single eval)")
fig.tight_layout()
fig.savefig(f"{out_prefix}_eval.png", dpi=100)
print(f"saved {out_prefix}_eval.png")

# ============================================================== figure 2 ==
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
a = axes[0, 0]
vl, vp = series("value_loss"), series("val_pre")
a.plot(step, vp, color="tab:orange", alpha=0.3, lw=1); a.plot(sx, smooth(vp), color="tab:orange", lw=2, label="val_pre (pre-update, on the batch)")
a.plot(step, vl, color="tab:blue", alpha=0.3, lw=1); a.plot(sx, smooth(vl), color="tab:blue", lw=2, label="value loss (post-update)")
a.set_title(f"Value loss per iteration  (last: val_pre {vp[-1]:.3f}, loss {vl[-1]:.3f})"); a.legend(fontsize=8)

a = axes[0, 1]
V, R = series("V"), series("R")
a.plot(step, R, color="tab:orange", alpha=0.3, lw=1); a.plot(sx, smooth(R), color="tab:orange", lw=2, label="R  mean return")
a.plot(step, V, color="tab:blue", alpha=0.3, lw=1); a.plot(sx, smooth(V), color="tab:blue", lw=2, label="V  mean value prediction")
a.set_title("Mean V vs mean R (rollout batch)"); a.legend(fontsize=8)

a = axes[1, 1]
bias = V - R
a.axhline(0, color="gray", lw=0.8)
a.plot(step, bias, color="tab:red", alpha=0.3, lw=1); a.plot(sx, smooth(bias), color="tab:red", lw=2, label="mean(V) - mean(R)")
a.set_title("Value bias"); a.legend(fontsize=8)

a = axes[1, 0]
V_std, R_std = series("V_std"), series("R_std")
a.plot(step, R_std, color="tab:orange", alpha=0.3, lw=1); a.plot(sx, smooth(R_std), color="tab:orange", lw=2, label="std(R)  spread of returns")
a.plot(step, V_std, color="tab:blue", alpha=0.3, lw=1); a.plot(sx, smooth(V_std), color="tab:blue", lw=2, label="std(V)  spread of value predictions")
a.set_title("Spread of V vs spread of R (under-dispersion if std(V) << std(R))"); a.legend(fontsize=8)

for ax in axes.flat:
    ax.set_xlabel("env steps (millions)")
fig.suptitle(f"{run.name}: value function over the run  {title_suffix}")
fig.tight_layout()
fig.savefig(f"{out_prefix}_value.png", dpi=100)
print(f"saved {out_prefix}_value.png")

# ============================================================== figure 3 ==
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
a = axes[0, 0]
kpe = series("kicks_per_ep")
a.plot(step, kpe, color="tab:blue", alpha=0.3, lw=1); a.plot(sx, smooth(kpe), color="tab:blue", lw=2, label="real kicks / episode")
a.set_title("Real kicks per episode (trainee)"); a.set_ylabel("kicks / ep"); a.legend(fontsize=8, loc="upper left")
a2 = a.twinx()
zk = series("zero_kick_pct")
a2.plot(step, zk, "--", color="gray", alpha=0.5, lw=1)
a2.set_ylabel("% episodes with 0 kicks")

a = axes[0, 1]
kprob = series("kick_prob")
a.plot(step, kprob, color="tab:orange", alpha=0.3, lw=1); a.plot(sx, smooth(kprob), color="tab:orange", lw=2, label="mean kick gate prob (kick_prob)")
a.set_title("Kick gate probability"); a.legend(fontsize=8)

a = axes[1, 0]
for key, label, color in [("kick_ent", "kick gate entropy", "tab:blue"), ("kick_dir_ent", "kick_dir entropy", "tab:orange"),
                           ("kick_power_ent", "kick_power entropy", "tab:green")]:
    y = series(key)
    a.plot(step, y, color=color, alpha=0.3, lw=1); a.plot(sx, smooth(y), color=color, lw=2, label=label)
y = series("kick_dir_z_ent")
a.plot(step, y, "--", color="tab:purple", alpha=0.3, lw=1); a.plot(sx, smooth(y), "--", color="tab:purple", lw=2, label="kick_dir_z entropy")
a.set_yscale("symlog", linthresh=0.01)
a.set_title("Kick-head entropies (as logged)"); a.legend(fontsize=8)

a = axes[1, 1]
kk = series("kick_dir_kappa")
a.plot(step, kk, color="tab:blue", alpha=0.3, lw=1); a.plot(sx, smooth(kk), color="tab:blue", lw=2, label="kick_dir κ (azimuth)")
a.set_ylabel("κ (von Mises)"); a.legend(fontsize=8, loc="upper left")
a2 = a.twinx()
kp = series("kick_power_sigma")
a2.plot(step, kp, color="tab:orange", alpha=0.3, lw=1); a2.plot(sx, smooth(kp), color="tab:orange", lw=2, label="kick_power σ (right)")
kz = series("kick_dir_z_sigma")
a2.plot(step, kz, color="tab:green", alpha=0.3, lw=1); a2.plot(sx, smooth(kz), color="tab:green", lw=2, label="kick_dir_z σ (right, elevation)")
a2.set_ylabel("σ (Gaussian scale)")
a.set_title("Kick direction (azimuth κ + elevation σ) / power sharpness")
a2.legend(fontsize=8, loc="lower right")

for ax in axes.flat:
    ax.set_xlabel("env steps (millions)")
fig.suptitle(f"{run.name}: kicking over the run  {title_suffix}")
fig.tight_layout()
fig.savefig(f"{out_prefix}_kick.png", dpi=100)
print(f"saved {out_prefix}_kick.png")

# ============================================================== figure 4 ==
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
a = axes[0, 0]
for key, label, color in [("sprint_ent", "sprint", "tab:orange"), ("exec_move_ent", "exec_move", "tab:green"),
                           ("kick_ent", "kick", "tab:red"), ("tackle_ent", "tackle_attempt", "tab:purple"),
                           ("kick_dir_ent", "kick_dir", "#b8860b"), ("kick_power_ent", "kick_power", "tab:cyan")]:
    y = series(key)
    a.plot(step, y, color=color, alpha=0.25, lw=1)
    sm = smooth(y)
    a.plot(sx, sm, color=color, lw=2, label=f"{label}  ({y[0]:+.4f} -> {y[-1]:+.4f})")
y = series("kick_dir_z_ent")
a.plot(step, y, "--", color="tab:blue", alpha=0.25, lw=1)
a.plot(sx, smooth(y), "--", color="tab:blue", lw=2, label=f"kick_dir_z  ({y[0]:+.4f} -> {y[-1]:+.4f})")
a.set_yscale("symlog", linthresh=0.01)
a.set_title("Entropy per head (all except move_dir; log axis)"); a.set_ylabel("entropy"); a.legend(fontsize=7, ncol=2)

a = axes[0, 1]
md = series("move_dir_ent")
a.plot(step, md, alpha=0.3, lw=1, color="tab:blue"); a.plot(sx, smooth(md), color="tab:blue", lw=2.4)
a.set_title(f"move_dir entropy (differential; lower = sharper)   {md[0]:+.3f} -> {md[-1]:+.3f}"); a.set_ylabel("entropy")

a = axes[1, 0]
for key, label, color in [("kl_sprint", "sprint", "tab:orange"), ("kl_exec_move", "exec_move", "tab:green"),
                           ("kl_kick", "kick", "tab:red"), ("kl_tackle_attempt", "tackle_attempt", "tab:purple"),
                           ("kl_kick_dir", "kick_dir", "#b8860b"), ("kl_kick_power", "kick_power", "tab:cyan"),
                           ("kl_move", "move (frozen)", "lightgray"), ("kl_gp_extra", "gp_extra (frozen)", "darkgray")]:
    y = series(key)
    if np.all(np.isnan(y)):
        continue
    a.plot(step, y, color=color, alpha=0.2, lw=1)
    a.plot(sx, smooth(y), color=color, lw=1.8, label=f"{label}  ({y[0]:+.4f} -> {y[-1]:+.4f})")
a.set_yscale("log")
a.set_title("Per-head KL per iteration (all except move_dir; log axis)"); a.set_ylabel("KL"); a.legend(fontsize=7, ncol=2)

a = axes[1, 1]
mkl = series("kl_move_dir")
a.plot(step, mkl, alpha=0.3, lw=1, color="tab:blue"); a.plot(sx, smooth(mkl), color="tab:blue", lw=2.4)
a.set_title(f"move_dir KL   {mkl[0]:+.4f} -> {mkl[-1]:+.4f}"); a.set_ylabel("KL")

for ax in axes.flat:
    ax.set_xlabel("env steps (millions)")
fig.suptitle(f"{run.name}: entropy (top) and per-head KL (bottom)   thin = raw per iteration, thick = 5-iteration mean.")
fig.tight_layout()
fig.savefig(f"{out_prefix}_entropy_kl.png", dpi=100)
print(f"saved {out_prefix}_entropy_kl.png")

print("\nper-head summary (first5 -> last5):")
for key, kl_key, label in [("sprint_ent", "kl_sprint", "sprint"), ("exec_move_ent", "kl_exec_move", "exec_move"),
                            ("kick_ent", "kl_kick", "kick"), ("tackle_ent", "kl_tackle_attempt", "tackle_attempt"),
                            ("kick_dir_ent", "kl_kick_dir", "kick_dir"), ("kick_power_ent", "kl_kick_power", "kick_power"),
                            ("kick_dir_z_ent", "kl_kick_dir_z", "kick_dir_z"), ("move_dir_ent", "kl_move_dir", "move_dir")]:
    y = series(key)
    kly = series(kl_key)
    if np.all(np.isnan(kly)):
        print(f"{label:15s} entropy first5 {y[:5].mean():+.4f} last5 {y[-5:].mean():+.4f} | KL not logged for this head")
        continue
    print(f"{label:15s} entropy first5 {y[:5].mean():+.4f} last5 {y[-5:].mean():+.4f} | "
          f"KL first5 {kly[:5].mean():+.4f} last5 {kly[-5:].mean():+.4f} max {np.nanmax(kly):+.4f}")
