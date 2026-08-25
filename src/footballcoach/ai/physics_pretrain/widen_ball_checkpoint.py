"""Widens a trained ``BallDynamicsAutoencoder`` checkpoint to bigger
``hidden_dim``/``encoder_bottleneck_dim``/``latent_dim``/``decoder_hidden_dim``
without losing the existing training.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

The core trick (see the identity-shortcut discovery in ``identity_shortcut.
py``'s ``_init_identity_shortcut_decoder`` docstring for the failure mode
this avoids): every widened layer's NEW output units get the bigger model's
own fresh (already-random, or hand-init for identity-shortcut rows) init --
never zeroed -- so they compute a real, non-degenerate, gradient-friendly
value from the first step. What DOES get zeroed is the corresponding NEW
*input* columns on whichever layer reads that widened output next, for
whichever OLD output rows must reproduce the OLD model exactly -- those
zeros mean the newly-added (live, random) units can't perturb an output
that has to stay bit-identical, while the zeroed weight itself still gets a
completely normal, immediate gradient (``d(loss)/d(zeroed weight) = d(loss)
/d(preactivation) * producing_unit's_output``, which doesn't depend on the
zeroed weight's own current value at all). Zeroing the wrong side --
the producing side, feeding directly into a ReLU -- was the actual bug
found and fixed in ``_init_identity_shortcut_decoder``: a unit whose
INCOMING weights are zero has pre-activation exactly 0, and ``ReLU'(0) ==
0`` by convention, so no gradient ever reaches it, a permanent dead unit
(verified empirically there: identical output for every input after 50
training steps). This module never repeats that mistake -- it only ever
zeros a *consuming* side.

One seam needs an EXTRA fixup on top of that rule: ``encoder.out``'s new
rows, under ``identity_shortcut_enabled``, are hand-zeroed by ``_init_
identity_shortcut_linear`` itself (it zeros the whole weight/bias before
hand-setting only the identity block) -- so `new`'s own construction already
leaves them at exact zero, not the random init "producing side, never
zero" assumes. Combined with THIS module zeroing the consuming side too,
both ends of that specific seam would land on zero simultaneously with
nothing to bootstrap from -- a real bug this module hit once (see
``_widen_encoder_out``'s docstring for the fix: explicitly re-randomizing
those rows).

Two seams need special handling because widening shifts where a
downstream, NEVER-widened block of columns lives: ``encoder.out``'s input
is ``[bottleneck features (widens) ‖ raw-input concat features (fixed
width)]`` -- growing the bottleneck moves the concat block's starting
column. ``decoder.net[0]``'s input is ``[latent (widens) ‖ 3 horizon
features (fixed width)]`` -- same shift for latent growth. Both are handled
explicitly below (``_widen_encoder_out``/``_widen_decoder_net0``) rather
than by the generic per-layer helpers.

Usage::

    # after raising hidden_dim/encoder_bottleneck_dim/latent_dim/
    # decoder_hidden_dim in ai_config.json's physics_pretrain.ball section:
    uv run python -m footballcoach.ai.physics_pretrain.widen_ball_checkpoint \\
        --checkpoint checkpoints/physics_pretrain/ball_encoder.midtrain_latest.pt \\
        --output checkpoints/physics_pretrain/ball_encoder.widened.pt \\
        --dataset physics_pretrain_data/ball/

The output is a normal phase checkpoint (same shape as ``_save_phase_
checkpoint`` writes in train_ball_dynamics.py) -- pass it straight back in
via ``--init-checkpoint`` to resume training with the new capacity.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsAutoencoder
from footballcoach.ai.physics_pretrain.train_ball_dynamics import (
    _migrate_crossing_head_state_dict,
    _physics_config_hash,
)

log = logging.getLogger("footballcoach.ai.physics_pretrain.widen_ball_checkpoint")

# cfg keys that must be IDENTICAL between old and new -- this module only
# widens the four numeric capacity dims below; changing whether/how the
# identity-shortcut mechanism is wired up (which columns/rows it hand-inits)
# at the same time as widening is out of scope, since the seam surgery
# below assumes those bits of layout are stable.
_STABLE_KEYS = (
    "horizons_s",
    "identity_shortcut_enabled",
    "identity_shortcut_noise_std",
    "encoder_concat_all_input_fields",
    "decoder_identity_shortcut_enabled",
    "linear_decoder_enabled",
)
_WIDEN_KEYS = ("hidden_dim", "encoder_bottleneck_dim", "latent_dim", "decoder_hidden_dim")
# Subset of the two lists above that matters when the DECODER isn't being
# touched at all (train_ball_dynamics.py's horizons_s-changed-AND-dims-
# changed fallback -- see widen_model_'s `widen_decoder` param): only the
# encoder's own layout (identity-shortcut wiring) has to stay stable for its
# seam surgery to be valid; horizons_s/decoder_identity_shortcut_enabled/
# linear_decoder_enabled are decoder-only concerns, and decoder_hidden_dim
# isn't relevant if the decoder is getting a fresh init regardless.
_ENCODER_STABLE_KEYS = ("identity_shortcut_enabled", "identity_shortcut_noise_std", "encoder_concat_all_input_fields")
_ENCODER_WIDEN_KEYS = ("hidden_dim", "encoder_bottleneck_dim", "latent_dim")


def _build_model(cfg: dict) -> BallDynamicsAutoencoder:
    return BallDynamicsAutoencoder(
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        horizons_s=cfg["horizons_s"],
        decoder_hidden_dim=cfg.get("decoder_hidden_dim", 32),
        encoder_bottleneck_dim=cfg.get("encoder_bottleneck_dim", 32),
        identity_shortcut=cfg.get("identity_shortcut_enabled", False),
        identity_shortcut_noise_std=cfg.get("identity_shortcut_noise_std", 0.0),
        encoder_concat_all_input_fields=cfg.get("encoder_concat_all_input_fields", False),
        decoder_identity_shortcut=cfg.get("decoder_identity_shortcut_enabled"),
        linear_decoder=cfg.get("linear_decoder_enabled", False),
        leaky_relu_negative_slope=cfg.get("encoder_leaky_relu_negative_slope", 0.0),
    )


def _concat_dim(cfg: dict) -> int:
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import N_IDENTITY_SHORTCUT_FIELDS
    from footballcoach.ai.physics_pretrain.ball_episode_gen import N_INPUT_FIELDS
    if not cfg.get("identity_shortcut_enabled", False):
        return 0
    return N_INPUT_FIELDS if cfg.get("encoder_concat_all_input_fields", False) else N_IDENTITY_SHORTCUT_FIELDS


def _validate_widen_cfgs(old_cfg: dict, new_cfg: dict, check_decoder: bool = True) -> None:
    """``check_decoder=False`` (see ``widen_model_``'s identical param):
    skips every decoder-specific check -- the ``horizons_s``/
    ``decoder_identity_shortcut_enabled``/``linear_decoder_enabled``
    stability requirements -- validating only what the ENCODER's own seam
    surgery needs (``_ENCODER_STABLE_KEYS``/``_ENCODER_WIDEN_KEYS``). Used
    by train_ball_dynamics.py's ``--init-checkpoint`` resume when
    horizons_s changed at the same time as encoder dims grew: the decoder
    is getting a fresh init regardless (its old weights are incompatible
    with the new horizons_s either way), so none of the decoder-shape
    assumptions below apply.

    ``linear_decoder_enabled`` (when ``check_decoder`` is True, i.e. BOTH
    configs must agree via ``_STABLE_KEYS`` below) is fully supported --
    ``BallDynamicsLinearDecoder.net`` is a single ``Linear(latent_dim, ...)``
    whose INPUT is the only thing that can widen (its output width depends
    only on ``horizons_s``, already required stable), so it gets the exact
    same ``_widen_linear_cols`` treatment as ``crossing_head``/
    ``resting_head`` -- see ``widen_model_``. ``decoder_hidden_dim`` is
    meaningless in this mode (``BallDynamicsLinearDecoder`` has no hidden
    layer at all) so it's excluded from the growth check below rather than
    comparing a value the model doesn't even use.
    """
    stable_keys = _ENCODER_STABLE_KEYS if not check_decoder else _STABLE_KEYS
    widen_keys = _ENCODER_WIDEN_KEYS if not check_decoder else _WIDEN_KEYS
    if check_decoder and new_cfg.get("linear_decoder_enabled", False):
        widen_keys = tuple(k for k in widen_keys if k != "decoder_hidden_dim")
    for key in stable_keys:
        old_v, new_v = old_cfg.get(key), new_cfg.get(key)
        if old_v != new_v:
            raise ValueError(
                f"Widening requires '{key}' to stay the same (old={old_v!r}, new={new_v!r}) -- "
                "this tool only widens hidden_dim/encoder_bottleneck_dim/latent_dim/decoder_hidden_dim, "
                "not shortcut wiring or horizons."
            )
    for key in widen_keys:
        old_v, new_v = old_cfg.get(key, 32), new_cfg.get(key, 32)
        if new_v < old_v:
            raise ValueError(f"'{key}' shrank ({old_v} -> {new_v}) -- this tool only supports growing dims.")
    if all(new_cfg.get(k, 32) == old_cfg.get(k, 32) for k in widen_keys):
        log.warning("No dims actually grew -- output will be a plain copy.")


def _widen_linear_rows(old: nn.Linear, new: nn.Linear, old_out: int) -> None:
    """Pure 'producing' widen: only the OUTPUT grows, input unchanged (e.g.
    the encoder's very first layer, whose input is the fixed-width raw
    input). Old rows copied exactly; new rows are left at `new`'s own
    (already-random) fresh init -- entirely new units, nothing to
    preserve."""
    new.weight.data[:old_out, :] = old.weight.data
    new.bias.data[:old_out] = old.bias.data


def _widen_linear_cols(old: nn.Linear, new: nn.Linear, old_in: int) -> None:
    """Pure 'consuming' widen: only the INPUT grows, output unchanged (e.g.
    crossing_head/resting_head reading a widened latent). Old columns
    copied exactly for every (unchanged-count) output row; new columns
    zeroed so those old outputs can't be perturbed by the newly-added,
    live/random input units."""
    new.weight.data[:, :old_in] = old.weight.data
    new.weight.data[:, old_in:] = 0.0
    new.bias.data[:] = old.bias.data


def _widen_linear_full(old: nn.Linear, new: nn.Linear, old_out: int, old_in: int) -> None:
    """Both axes grow at once (a hidden-to-hidden layer whose in/out share
    the same widened dim). OLD rows keep their old columns exactly and get
    ZERO on the new columns (protecting their preserved output from the
    newly-added, live input units); NEW rows are entirely fresh units with
    nothing to preserve, so every one of their columns (old and new alike)
    is left at `new`'s own fresh init."""
    new.weight.data[:old_out, :old_in] = old.weight.data
    new.weight.data[:old_out, old_in:] = 0.0
    new.bias.data[:old_out] = old.bias.data


def _widen_encoder_out(
    old: nn.Linear, new: nn.Linear, old_bneck: int, new_bneck: int, old_latent: int, new_latent: int,
    concat_dim: int, identity_shortcut_enabled: bool,
) -> None:
    """``encoder.out``'s input is ``[bottleneck ‖ raw-input concat]`` (see
    ``BallDynamicsEncoder.forward``) -- growing the bottleneck shifts where
    the (fixed-width, never-widened) concat block starts, so its weight
    columns must move with it, values unchanged, not just get appended
    after.

    NEW output rows (the newly-added latent dims, ``[old_latent:new_
    latent)``) need explicit handling when ``identity_shortcut_enabled``:
    ``_init_identity_shortcut_linear`` zeros this layer's ENTIRE weight/bias
    before hand-setting only the identity block's rows
    (``[0:N_IDENTITY_SHORTCUT_FIELDS)``) -- so `new`'s own fresh
    construction already leaves every "spare" row (which is exactly what a
    new latent row is, since old_latent is virtually always >=
    N_IDENTITY_SHORTCUT_FIELDS) at EXACT zero, not the random init the
    "producing side, never zero" rule assumes. In the ORIGINAL
    (non-widened) model that's harmless -- the decoder/crossing_head/
    resting_head/position_head/event_head consumers reading those spare latent columns were
    themselves freshly, randomly initialized from scratch, so gradient can
    reach encoder.out's zero row via the consumer's ALREADY-nonzero weight
    (``d(loss)/d(latent[row])`` doesn't depend on ``latent[row]``'s own
    current value, only on the consumer's weight and its own gradient).
    But here, the CONSUMING side's new columns were JUST zeroed too (by
    ``_widen_linear_cols``/``_widen_decoder_net0`` above, to protect old
    outputs from a brand-new, initially-meaningless latent dim) -- so both
    sides of the seam land on zero SIMULTANEOUSLY, a genuine two-sided dead
    fixed point with nothing left to bootstrap from (confirmed empirically:
    a real run's new latent rows were still EXACTLY zero, weight and bias,
    after many further epochs of training). Explicitly re-randomizing these
    rows breaks that lock the same way ``_init_identity_shortcut_decoder``'s
    own spare-unit fix already does for the decoder's analogous case.
    """
    new.weight.data[:old_latent, :old_bneck] = old.weight.data[:, :old_bneck]
    new.weight.data[:old_latent, old_bneck:new_bneck] = 0.0
    if concat_dim:
        new.weight.data[:old_latent, new_bneck:new_bneck + concat_dim] = old.weight.data[:, old_bneck:old_bneck + concat_dim]
    new.bias.data[:old_latent] = old.bias.data
    if identity_shortcut_enabled and new_latent > old_latent:
        nn.init.kaiming_uniform_(new.weight[old_latent:new_latent, :], a=5 ** 0.5)
        fan_in = new.weight.shape[1]
        bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0.0
        new.bias.data[old_latent:new_latent].uniform_(-bound, bound)


def _widen_decoder_net0(old: nn.Linear, new: nn.Linear, old_latent: int, new_latent: int, old_dhidden: int) -> None:
    """``decoder.net[0]``'s input is ``[latent ‖ 3 horizon features]`` (see
    ``BallDynamicsDecoder.forward``) -- growing latent_dim shifts where the
    3 (fixed-width) horizon-feature columns start, same shape of surgery as
    ``_widen_encoder_out``."""
    new.weight.data[:old_dhidden, :old_latent] = old.weight.data[:, :old_latent]
    new.weight.data[:old_dhidden, old_latent:new_latent] = 0.0
    new.weight.data[:old_dhidden, new_latent:new_latent + 3] = old.weight.data[:, old_latent:old_latent + 3]
    new.bias.data[:old_dhidden] = old.bias.data


def widen_model_(
    old_model: BallDynamicsAutoencoder, new_model: BallDynamicsAutoencoder, old_cfg: dict, new_cfg: dict,
    widen_decoder: bool = True,
) -> None:
    """Mutates ``new_model`` (already constructed with its own fresh/
    identity-shortcut init, per ``new_cfg``) in place so it computes the
    EXACT same function as ``old_model`` for its old capacity, plus fresh,
    immediately-trainable capacity for whatever grew. See module docstring
    for the general "zero the consuming side, never the producing side"
    rule and why the two seams below need to also handle a column shift.

    ``widen_decoder=False``: skips the decoder seam entirely, leaving
    ``new_model.decoder`` exactly as its own fresh construction left it --
    for train_ball_dynamics.py's ``--init-checkpoint`` resume when
    ``horizons_s`` also changed, making the old decoder's weights
    incompatible (shape and/or meaning) regardless of any widening. The
    encoder and every auxiliary head (crossing_head/resting_head/
    position_head/event_head, all pure latent consumers) still get widened
    normally -- only the decoder's own seam is conditional.
    """
    old_hidden, new_hidden = old_cfg["hidden_dim"], new_cfg["hidden_dim"]
    old_bneck = old_cfg.get("encoder_bottleneck_dim", 32)
    new_bneck = new_cfg.get("encoder_bottleneck_dim", 32)
    old_latent, new_latent = old_cfg["latent_dim"], new_cfg["latent_dim"]
    old_dhidden = old_cfg.get("decoder_hidden_dim", 32)
    concat_dim = _concat_dim(old_cfg)

    with torch.no_grad():
        _widen_linear_rows(old_model.encoder.trunk[0], new_model.encoder.trunk[0], old_hidden)
        _widen_linear_full(old_model.encoder.trunk[2], new_model.encoder.trunk[2], old_hidden, old_hidden)
        _widen_linear_full(old_model.encoder.trunk[4], new_model.encoder.trunk[4], old_bneck, old_hidden)
        _widen_encoder_out(
            old_model.encoder.out, new_model.encoder.out, old_bneck, new_bneck, old_latent, new_latent,
            concat_dim, old_cfg.get("identity_shortcut_enabled", False),
        )
        _widen_linear_cols(old_model.crossing_head, new_model.crossing_head, old_latent)
        _widen_linear_cols(old_model.resting_head, new_model.resting_head, old_latent)
        _widen_linear_cols(old_model.position_head, new_model.position_head, old_latent)
        _widen_linear_cols(old_model.event_head, new_model.event_head, old_latent)
        if widen_decoder:
            if old_model.linear_decoder:
                # BallDynamicsLinearDecoder.net is a single Linear(latent_
                # dim, n_horizons*6) -- output width depends only on
                # horizons_s (required stable, see _validate_widen_cfgs),
                # so growing latent_dim is a pure "consuming" widen, exactly
                # like crossing_head/resting_head above.
                _widen_linear_cols(old_model.decoder.net, new_model.decoder.net, old_latent)
            else:
                _widen_decoder_net0(old_model.decoder.net[0], new_model.decoder.net[0], old_latent, new_latent, old_dhidden)
                _widen_linear_cols(old_model.decoder.net[2], new_model.decoder.net[2], old_dhidden)


def verify_widened_model(old_model: BallDynamicsAutoencoder, new_model: BallDynamicsAutoencoder, x: torch.Tensor, atol: float = 1e-4) -> None:
    """Asserts ``new_model`` reproduces ``old_model`` EXACTLY on ``x`` --
    every decoder horizon, plus crossing_head/resting_head. Raises
    AssertionError (with the worst-offending max-diff) if not, so a bug in
    the seam surgery above is caught here rather than silently shipped into
    a training run."""
    old_model.eval()
    new_model.eval()
    with torch.no_grad():
        latent_old, decoder_old = old_model(x)
        latent_new, decoder_new = new_model(x)
        crossing_old, crossing_new = old_model.crossing_head(latent_old), new_model.crossing_head(latent_new)
        resting_old, resting_new = old_model.resting_head(latent_old), new_model.resting_head(latent_new)
        position_old, position_new = old_model.position_head(latent_old), new_model.position_head(latent_new)
        event_old, event_new = old_model.event_head(latent_old), new_model.event_head(latent_new)

    for h, (do, dn) in enumerate(zip(decoder_old, decoder_new)):
        diff = (do - dn).abs().max().item()
        assert diff < atol, f"decoder horizon {h} diverged after widening (max abs diff {diff:.6g})"
    diff = (crossing_old - crossing_new).abs().max().item()
    assert diff < atol, f"crossing_head diverged after widening (max abs diff {diff:.6g})"
    diff = (resting_old - resting_new).abs().max().item()
    assert diff < atol, f"resting_head diverged after widening (max abs diff {diff:.6g})"
    diff = (position_old - position_new).abs().max().item()
    assert diff < atol, f"position_head diverged after widening (max abs diff {diff:.6g})"
    diff = (event_old - event_new).abs().max().item()
    assert diff < atol, f"event_head diverged after widening (max abs diff {diff:.6g})"


def repair_dead_encoder_out_rows(
    checkpoint_path: str | Path, output_path: str | Path, row_start: int, row_end: int,
) -> None:
    """One-off repair for a checkpoint produced by the (now-fixed)
    ``_widen_encoder_out`` bug: re-randomizes ``encoder.out``'s weight/bias
    for latent rows ``[row_start:row_end)`` IN PLACE on an already-widened,
    already-further-trained checkpoint, leaving every other parameter --
    including that same checkpoint's crossing_head/resting_head/decoder.
    net[0] weights reading those latent columns, which were also stuck at
    zero for the same reason -- untouched. Those consuming weights don't
    need manual repair: once encoder.out's rows produce a real, input-
    dependent value again, their own gradient (``d(loss)/d(weight) = d(loss)
    /d(preactivation) * latent[row]``) stops being multiplied by zero, so
    they resume learning normally on the very next training step.

    Refuses to touch rows that AREN'T all-zero (real trained values would
    silently be destroyed otherwise) -- only for checkpoints predating the
    fix in ``_widen_encoder_out`` above; a freshly-``widen_checkpoint()``'d
    one never needs this.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no 'model_state_dict' (only an encoder) -- nothing to repair.")
    sd = ckpt["model_state_dict"]
    w, b = sd["encoder.out.weight"], sd["encoder.out.bias"]
    dead_w, dead_b = w[row_start:row_end, :], b[row_start:row_end]
    if dead_w.abs().sum().item() > 0 or dead_b.abs().sum().item() > 0:
        raise ValueError(
            f"encoder.out rows [{row_start}:{row_end}) aren't all-zero -- refusing to overwrite what "
            "might be real trained weights (this repair is only for the specific dead-row bug)."
        )
    with torch.no_grad():
        nn.init.kaiming_uniform_(w[row_start:row_end, :], a=5 ** 0.5)
        fan_in = w.shape[1]
        bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0.0
        b[row_start:row_end].uniform_(-bound, bound)
    sd["encoder.out.weight"], sd["encoder.out.bias"] = w, b
    ckpt["model_state_dict"] = sd
    ckpt["encoder_state_dict"] = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    log.info(f"Repaired encoder.out rows [{row_start}:{row_end}) and saved to {output_path}")


def widen_checkpoint(
    checkpoint_path: str | Path, output_path: str | Path,
    new_cfg: dict | None = None, dataset_dir: str | Path | None = None, verify_n: int = 16,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"{checkpoint_path} has no 'model_state_dict' (only an encoder). "
            "Pass a phase checkpoint instead, e.g. one ending in '.midtrain_latest.pt' or '.after_training.pt'."
        )
    old_cfg = ckpt["config_snapshot"]
    if new_cfg is None:
        from footballcoach.ai.config import load_ai_config
        new_cfg = load_ai_config()["physics_pretrain"]["ball"]
    _validate_widen_cfgs(old_cfg, new_cfg)
    # encoder_leaky_relu_negative_slope isn't part of the saved weights
    # (LeakyReLU has no learnable params) -- old_model only exists so
    # widen_model_ can copy FROM it, so build it with new_cfg's slope (the
    # value this checkpoint will actually run with going forward), not
    # whatever old_cfg says. This is also what makes resuming a checkpoint
    # trained under plain ReLU (slope 0.0) with a newly-nonzero slope "just
    # work" without any explicit migration step.
    old_cfg = {**old_cfg, "encoder_leaky_relu_negative_slope": new_cfg.get("encoder_leaky_relu_negative_slope", 0.0)}

    old_model = _build_model(old_cfg)
    missing, unexpected = old_model.load_state_dict(
        _migrate_crossing_head_state_dict(ckpt["model_state_dict"], old_model), strict=False,
    )
    if missing or unexpected:
        log.info(f"Loading old checkpoint: missing={missing} unexpected={unexpected}")
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)

    from footballcoach.ai.physics_pretrain.ball_episode_gen import N_INPUT_FIELDS
    if dataset_dir is not None:
        from footballcoach.ai.physics_pretrain.ball_dataset import BallDynamicsDataset
        ds = BallDynamicsDataset.from_directory(dataset_dir)
        n = min(verify_n, len(ds))
        idx = torch.randperm(len(ds))[:n].numpy()
        x = torch.from_numpy(ds.inputs[idx].astype("float32"))
    else:
        x = torch.randn(verify_n, N_INPUT_FIELDS) * 0.3
    verify_widened_model(old_model, new_model, x)
    log.info(f"Verified: widened model reproduces the old checkpoint exactly on {x.shape[0]} rows.")

    for key in _WIDEN_KEYS:
        old_v, new_v = old_cfg.get(key, 32), new_cfg.get(key, 32)
        if new_v != old_v:
            log.info(f"  {key}: {old_v} -> {new_v}")

    from footballcoach.ai.physics_pretrain.ball_episode_gen import BALL_SPIN_NORM_DIVISOR_RAD_S, BallEpisodeGenParams
    import math
    gen_params = BallEpisodeGenParams.from_config()
    pitch_half_diag_m = math.hypot(gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": new_model.state_dict(),
        "encoder_state_dict": new_model.encoder.state_dict(),
        "config_snapshot": new_cfg,
        "normalization": {
            "pitch_half_diag_m": pitch_half_diag_m,
            "height_norm_m": gen_params.height_norm_m,
            "ball_spin_norm_max_rad_s": BALL_SPIN_NORM_DIVISOR_RAD_S,
        },
        "physics_config_hash": _physics_config_hash(),
        "phase": "widened",
    }, output_path)
    log.info(f"Saved widened checkpoint to {output_path}")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="Full-model checkpoint to widen (has model_state_dict).")
    ap.add_argument("--output", required=True, type=Path, help="Where to write the widened checkpoint.")
    ap.add_argument("--dataset", type=Path, default=None, help="Optional dataset dir -- verifies on real rows instead of random noise.")
    ap.add_argument("--verify-n", type=int, default=16)
    args = ap.parse_args()
    widen_checkpoint(args.checkpoint, args.output, dataset_dir=args.dataset, verify_n=args.verify_n)


if __name__ == "__main__":
    _main()
