"""Diagnostic: encodes a batch of inputs through an encoder and reports
per-dimension and cross-dimension statistics of the resulting latent space
-- meant as an "is this latent well-behaved" snapshot, not a training
signal (nothing here is differentiated or used in any loss).

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

Generic over which encoder/dataset it's pointed at -- ``compute_latent_stats``
only needs a callable ``encoder`` (any ``nn.Module`` mapping ``(batch,
input_dim) -> (batch, latent_dim)``) and a plain ``(n_rows, input_dim)``
float array of inputs, so the same function serves both the ball and player
pipelines' encoders without either depending on the other's dataset/model
classes.

Called once at the start of ``train()`` (both ball and player), right after
the model is constructed and any ``--init-checkpoint`` weights are loaded,
BEFORE any training this run happens -- i.e. exactly the "epoch 0" state,
whether that's a fresh random init or a resumed checkpoint. Purely a logged
diagnostic snapshot (see ``format_latent_stats``); nothing here feeds back
into training.
"""
from __future__ import annotations

import numpy as np
import torch


def compute_latent_stats(
    encoder: torch.nn.Module, inputs: np.ndarray, device: torch.device,
    batch_size: int = 4096, max_rows: int = 50_000, seed: int = 0,
    dead_std_threshold: float = 1e-3, redundant_corr_threshold: float = 0.95,
) -> dict:
    """Encodes (a subsample of, if larger than ``max_rows``) ``inputs``
    through ``encoder`` in eval mode / no_grad, then computes a battery of
    latent-space diagnostics from the resulting ``(n, latent_dim)`` matrix
    ``Z``:

    Per-dimension (each a ``(latent_dim,)`` array): ``mean``, ``std``,
    ``min``, ``max``, ``mean_abs``.

    Whole-layer scalars: ``mean_per_dim_std`` (average of the per-dim
    stds -- "how much does a typical dimension move"), ``global_std_pooled``
    (std of ALL n*latent_dim values pooled together, ignoring which
    dimension each came from -- differs from ``mean_per_dim_std`` when
    dimensions have very different scales), ``mean_latent_norm``/
    ``std_latent_norm`` (mean/std of each row's L2 norm -- a proxy for
    "how big is a typical latent vector" and "how much does that size
    vary").

    Dead-unit detection: ``dead_dims`` (indices where ``std <
    dead_std_threshold`` -- a dimension that barely moves across the whole
    sample is either not being used or has collapsed) and ``n_dead_dims``.

    Correlation/covariance: ``cov`` and ``corr`` (both ``(latent_dim,
    latent_dim)``), plus summary scalars over the correlation matrix's
    OFF-DIAGONAL entries only (the diagonal is trivially 1.0) --
    ``mean_abs_offdiag_corr``, ``max_abs_offdiag_corr`` and the
    ``max_abs_offdiag_corr_pair`` (i, j) achieving it, and
    ``n_redundant_pairs`` (how many off-diagonal pairs exceed
    ``redundant_corr_threshold`` -- near-duplicate dimensions carrying
    almost the same information, wasted capacity).

    Spectral/effective-rank diagnostics (eigendecomposition of ``cov``,
    ascending-then-reversed so ``eigenvalues`` is DESCENDING):
    ``eigenvalues``, ``explained_variance_ratio`` (each eigenvalue divided
    by their sum), ``n_components_for_95pct_variance`` (how many of the
    TOP eigenvalues, cumulatively, are needed to reach 95% of total
    variance -- a low number relative to latent_dim means most dimensions
    are redundant/low-variance, i.e. the latent is using much less than its
    full nominal capacity), ``effective_rank_participation_ratio``
    (``(sum(eigenvalues))^2 / sum(eigenvalues^2)`` -- a smooth,
    threshold-free alternative "effective number of dimensions" measure,
    equal to latent_dim only if all eigenvalues are exactly equal, and
    dropping toward 1 as variance concentrates into fewer directions), and
    ``condition_number`` (largest eigenvalue / smallest eigenvalue,
    floored at a tiny epsilon to avoid a literal division by zero -- large
    values mean the latent's variance is extremely anisotropic, which can
    make anything reading it, e.g. an RL policy's own encoder-on-top,
    harder to optimize well).

    Returns a plain dict (all numpy arrays / Python scalars, no tensors) so
    it's trivially loggable/serializable. Raises no errors on n < 2 rows or
    latent_dim < 2 columns other than what numpy itself raises for a
    degenerate covariance -- callers are expected to pass a real dataset's
    worth of rows.
    """
    n_total = len(inputs)
    if n_total > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_rows, replace=False)
        inputs = inputs[idx]
    n = len(inputs)

    was_training = encoder.training
    encoder.eval()
    latents = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(inputs[start:start + batch_size].astype(np.float32, copy=False)).to(device)
            latents.append(encoder(batch).cpu().numpy())
    encoder.train(was_training)
    Z = np.concatenate(latents, axis=0).astype(np.float64)  # float64: covariance/eig accuracy matters more than speed here
    latent_dim = Z.shape[1]

    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    mn = Z.min(axis=0)
    mx = Z.max(axis=0)
    mean_abs = np.abs(Z).mean(axis=0)
    dead_dims = np.nonzero(std < dead_std_threshold)[0]

    row_norms = np.linalg.norm(Z, axis=1)

    cov = np.cov(Z, rowvar=False)
    if latent_dim == 1:
        cov = cov.reshape(1, 1)
    # A dead (zero-variance) dimension makes corrcoef divide 0/0 -- silence
    # the resulting (expected, handled below) RuntimeWarning rather than
    # letting it spam the log every time this runs on a model with dead
    # dims, which is exactly the case this diagnostic exists to surface.
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(Z, rowvar=False)
    if latent_dim == 1:
        corr = corr.reshape(1, 1)
    # NaN can appear in corrcoef's row/col for a dead (zero-variance)
    # dimension (0/0) -- treat as "no correlation" rather than propagating
    # NaN into every downstream summary stat.
    corr = np.nan_to_num(corr, nan=0.0)

    offdiag_mask = ~np.eye(latent_dim, dtype=bool)
    offdiag_abs = np.abs(corr) * offdiag_mask
    mean_abs_offdiag_corr = float(offdiag_abs.sum() / max(offdiag_mask.sum(), 1))
    flat_max_idx = int(np.argmax(offdiag_abs))
    max_i, max_j = np.unravel_index(flat_max_idx, offdiag_abs.shape)
    max_abs_offdiag_corr = float(offdiag_abs[max_i, max_j])
    n_redundant_pairs = int((offdiag_abs > redundant_corr_threshold).sum() // 2)

    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # eigvalsh: cov is symmetric; ascending -> reverse to descending
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # guard tiny negative numerical noise from a near-singular cov
    total_var = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(explained_variance_ratio)
    n_components_for_95pct_variance = int(np.searchsorted(cumulative, 0.95) + 1) if total_var > 0 else 0
    participation_ratio = float(total_var ** 2 / max((eigenvalues ** 2).sum(), 1e-30)) if total_var > 0 else 0.0
    condition_number = float(eigenvalues[0] / max(eigenvalues[-1], 1e-12))

    return {
        "n_rows": n,
        "latent_dim": latent_dim,
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "mean_abs": mean_abs,
        "mean_per_dim_std": float(std.mean()),
        "global_std_pooled": float(Z.std()),
        "mean_latent_norm": float(row_norms.mean()),
        "std_latent_norm": float(row_norms.std()),
        "dead_dims": dead_dims,
        "n_dead_dims": int(len(dead_dims)),
        "cov": cov,
        "corr": corr,
        "mean_abs_offdiag_corr": mean_abs_offdiag_corr,
        "max_abs_offdiag_corr": max_abs_offdiag_corr,
        "max_abs_offdiag_corr_pair": (int(max_i), int(max_j)),
        "n_redundant_pairs": n_redundant_pairs,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "n_components_for_95pct_variance": n_components_for_95pct_variance,
        "effective_rank_participation_ratio": participation_ratio,
        "condition_number": condition_number,
    }


def format_latent_stats(stats: dict, top_k_dims: int = 5, top_k_pairs: int = 5) -> str:
    """Renders ``compute_latent_stats``'s dict as a multi-line human-
    readable summary suitable for a single ``log.info(...)`` call --
    headline scalars first, then the ``top_k_dims`` smallest-std (most
    dead-adjacent) and largest-std dimensions individually, then the
    ``top_k_pairs`` most correlated off-diagonal pairs (redundancy
    candidates)."""
    d = stats
    lines = [
        f"Latent diagnostics ({d['n_rows']:,} rows, latent_dim={d['latent_dim']}):",
        f"    per-dim std: mean={d['mean_per_dim_std']:.4f}  pooled={d['global_std_pooled']:.4f}"
        f"  |  latent norm: mean={d['mean_latent_norm']:.4f} std={d['std_latent_norm']:.4f}",
        f"    dead dims (std < threshold): {d['n_dead_dims']}/{d['latent_dim']}"
        + (f"  {d['dead_dims'].tolist()}" if d["n_dead_dims"] else ""),
        f"    off-diagonal |corr|: mean={d['mean_abs_offdiag_corr']:.4f}"
        f"  max={d['max_abs_offdiag_corr']:.4f} (dims {d['max_abs_offdiag_corr_pair']})"
        f"  redundant pairs (|corr|>threshold): {d['n_redundant_pairs']}",
        f"    effective rank: {d['effective_rank_participation_ratio']:.2f}/{d['latent_dim']}"
        f" (participation ratio)  95%-variance components: {d['n_components_for_95pct_variance']}/{d['latent_dim']}"
        f"  condition number: {d['condition_number']:.3g}",
    ]

    order = np.argsort(d["std"])
    smallest = order[:top_k_dims]
    largest = order[::-1][:top_k_dims]
    lines.append(
        "    smallest-std dims: " + ", ".join(f"{i}(std={d['std'][i]:.4f},mean={d['mean'][i]:.4f})" for i in smallest)
    )
    lines.append(
        "    largest-std dims:  " + ", ".join(f"{i}(std={d['std'][i]:.4f},mean={d['mean'][i]:.4f})" for i in largest)
    )

    latent_dim = d["latent_dim"]
    if latent_dim > 1:
        corr = d["corr"]
        offdiag_mask = ~np.eye(latent_dim, dtype=bool)
        flat = np.abs(corr) * offdiag_mask
        pair_idx = np.dstack(np.unravel_index(np.argsort(flat, axis=None)[::-1], flat.shape))[0]
        seen = set()
        shown = []
        for i, j in pair_idx:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            shown.append(f"({i},{j})={corr[i, j]:.3f}")
            if len(shown) >= top_k_pairs:
                break
        lines.append("    most-correlated pairs: " + ", ".join(shown))

    return "\n".join(lines)
