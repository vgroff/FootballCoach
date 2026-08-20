"""Standalone HTML training-report generator for the ball-dynamics encoder.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

Renders per-epoch, per-head loss curves (position/velocity/spin RMSE,
out-of-bounds/goal-scored BCE) into a single self-contained ``.html`` file
next to the checkpoint -- no server, no external assets. ``train()`` in
``train_ball_dynamics.py`` calls ``write_report()`` automatically at the end
of every run and (unless disabled) opens it via ``open_in_browser()``.

This is a LOCAL file, not the hosted claude.ai artifact used earlier in
development -- publishing there requires a person-driven tool call in a
Claude Code conversation, which a training script has no way to trigger
itself. This local report is the fully-automatic equivalent: same page,
opened straight from disk after training finishes.
"""
from __future__ import annotations

import json
import logging
import webbrowser
from pathlib import Path
from typing import Any

log = logging.getLogger("footballcoach.ai.physics_pretrain.report")

_TEMPLATE_PATH = Path(__file__).parent / "report_template.html"
_PLACEHOLDER = "__HISTORY_JSON__"


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy arrays/scalars in a nested dict to plain
    Python types so ``json.dumps`` can serialize it directly."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def write_report(
    history_arrays: dict,
    dataset_stats: dict,
    config_snapshot: dict,
    normalization: dict,
    output_path: str | Path,
    val_examples: list[dict] | None = None,
    panel_defs: list[dict] | None = None,
    header_stat_defs: list[dict] | None = None,
    best_table_defs: list[dict] | None = None,
    title: str | None = None,
    config_namespace: str | None = None,
) -> Path:
    """Writes the self-contained HTML report to ``output_path``.

    ``history_arrays``: the same dict of per-epoch numpy arrays saved to
    ``<checkpoint>.history.npz`` (epoch, train_loss, val_loss,
    train_pos_rmse, val_goal_bce, horizons_s, ...) -- see
    ``train_ball_dynamics.py``'s ``train()``.

    ``val_examples``: optional list of dicts (``episode_idx``, ``horizon_s``,
    ``input``, ``pred``, ``target`` -- the latter 3 already denormalized,
    human-readable strings via ``describe_input_row``/``describe_target_
    row``) rendered as a table at the bottom of the report, so a reader can
    eyeball a handful of TYPICAL predictions directly rather than only
    aggregate metrics. Omitted (empty table) if ``None``/empty (e.g. no val
    split).

    ``panel_defs``/``header_stat_defs``/``best_table_defs``/``title``/
    ``config_namespace`` (all optional, ``None`` = use the ball pipeline's
    own hardcoded defaults baked into ``report_template.html`` -- passing
    nothing here reproduces the exact prior ball-only behaviour): generic
    hooks that let a DIFFERENT physics-pretrain pipeline (e.g.
    ``train_player_dynamics.py``) reuse this same template/renderer instead
    of writing a second bespoke report generator, per
    agent_plans/ball_physics_pretrain_plan.md §12.5. See
    ``report_template.html``'s ``PANEL_DEFS``/``buildHeader``/
    ``buildBestTable`` for the exact schema each expects:

    - ``panel_defs``: list of ``{key, title, note, tickDigits, unitScale,
      unitLabel, unitDigits, chanceLine, yRange}`` (``unitScale`` here is
      already a RESOLVED float, e.g. a caller's own ``pitch_half_diag_m``
      -- not a lookup key -- since a panel def is a plain JS object
      literal at chart-build time either way).
    - ``header_stat_defs``: list of ``{label, key, unit, scaleKey}`` --
      headline stat tiles at the top of the report; value is ``mean(DATA.
      val_<key>[bestIdx])``, multiplied by ``DATA.normalization[scaleKey]``
      if given.
    - ``best_table_defs``: list of ``{key, label, unit, unitDigits,
      scaleKey}`` -- rows of the best-epoch table at the bottom; same
      scaleKey-lookup convention as ``header_stat_defs``.
    """
    payload = {
        **_to_jsonable(history_arrays),
        "dataset_stats": _to_jsonable(dataset_stats),
        "config_snapshot": _to_jsonable(config_snapshot),
        "normalization": _to_jsonable(normalization),
        "val_examples": _to_jsonable(val_examples or []),
        "panel_defs": _to_jsonable(panel_defs) if panel_defs is not None else None,
        "header_stat_defs": _to_jsonable(header_stat_defs) if header_stat_defs is not None else None,
        "best_table_defs": _to_jsonable(best_table_defs) if best_table_defs is not None else None,
        "title": title,
        "config_namespace": config_namespace,
    }
    template = _TEMPLATE_PATH.read_text()
    if _PLACEHOLDER not in template:
        raise ValueError(f"{_TEMPLATE_PATH} is missing the {_PLACEHOLDER} placeholder")
    html = template.replace(_PLACEHOLDER, json.dumps(payload))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def open_in_browser(path: str | Path) -> bool:
    """Opens *path* in the OS default browser. Never raises -- a headless
    environment (no DISPLAY, no default browser configured) is a common,
    harmless case here, not a training failure, so this logs a warning and
    returns False instead of propagating."""
    path = Path(path).resolve()
    try:
        return webbrowser.open(path.as_uri())
    except Exception as e:
        log.warning(f"Could not open report in browser ({path}): {e}")
        return False
