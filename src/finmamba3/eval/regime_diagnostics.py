"""Diagnose whether the RegimeFiLM head is learning meaningful regime structure.

Loads a pretrained checkpoint, runs the encoder + Mamba backbone over a
validation sequence, and extracts the soft regime-assignment logits produced by
the RegimeFiLM hypernetwork.  Reports four diagnostics that together indicate
whether the head is useful or has learned arbitrary labels:

1. **Temporal persistence** – fraction of adjacent steps that share the same
   argmax regime.  Real regimes are persistent; flickering (< 0.5) suggests the
   head is not learning stable structure.
2. **Usage distribution** – how evenly regimes are used across the sequence.
   The load-balance regulariser pushes toward uniform; if one regime still
   dominates (> 80 %), the regulariser may need tuning.
3. **Entropy over time** – rolling std of the per-step entropy.  A large std
   means regime assignments are time-varying, which is what we want.
4. **Vol-regime correlation** – Spearman rank correlation between the argmax
   regime index and the rolling realized volatility of the normalised midprice.
   A model that has learned volatility regimes should show |r| > 0.2.

The script also writes a JSON summary and, when matplotlib is available, a
two-panel plot of regime assignments and midprice over the sequence.

Example
-------
    python -m finmamba3.eval.regime_diagnostics \\
        --checkpoint saved_models/lob/LOB/<run>/ckpt/world_model_best.pth \\
        --config configs/lob.yaml \\
        --data-val data/validation \\
        --out-dir notes/regime_diag/
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime-FiLM assignment diagnostics")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", type=Path,
                   default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    p.add_argument("--market-slug", default=None)
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--context-len", type=int, default=16,
                   help="Number of ticks to encode before extracting regime assignments.")
    p.add_argument("--batch-len", type=int, default=128,
                   help="Sequence length per batch window.")
    p.add_argument("--vol-window", type=int, default=20,
                   help="Rolling window (ticks) for realized vol proxy.")
    p.add_argument("--out-dir", type=Path, default=Path("notes/regime_diag"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_regime_logits(
    checkpoint: Path,
    config_path: Path,
    val_flat: np.ndarray,
    device: torch.device,
    batch_len: int = 128,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return (regime_argmax, regime_entropy) arrays of shape (T,) over the sequence.

    Returns (None, None) if the checkpoint has RegimeFiLM disabled.
    """
    import yaml
    from finmamba3.config import DotDict, parse_args_and_update_config
    from finmamba3.models.world_model import WorldModel

    with open(config_path) as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)

    wm = WorldModel(action_dim=13, config=cfg, device=device).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = state.get("world_model", state.get("state_dict", state))
    wm.load_state_dict(sd, strict=False)
    wm.eval()

    if not wm.use_regime_film:
        logger.warning(
            "Checkpoint has RegimeFiLM disabled (RegimeFiLM.Enabled=False).  "
            "No regime logits to extract.  Enable RegimeFiLM in the config and "
            "re-train, or use this script on a treatment-arm checkpoint."
        )
        return None, None

    T, F = val_flat.shape
    all_argmax: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []

    # Slide non-overlapping windows of length batch_len over the sequence.
    starts = list(range(0, max(T - batch_len, 1), batch_len))
    for s in starts:
        window = val_flat[s : s + batch_len]
        if window.shape[0] < 2:
            continue
        obs = torch.from_numpy(window).float().unsqueeze(0).to(device)   # (1, L, F)
        action = torch.zeros((1, window.shape[0]), device=device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp
        ):
            embedding = wm.encoder(obs)
            post_logits = wm.dist_head.forward_post(embedding)
            sample = wm.straight_through_gradient(post_logits)
            flat_sample = wm.flatten_sample(sample)
            # return_regime=True asks the Mamba backbone to also return the
            # per-step soft regime logits from the FiLM hypernetwork.
            result = wm.sequence_model(flat_sample, action, return_regime=True)
            if isinstance(result, tuple):
                _, regime_logits = result  # (1, L, R)
            else:
                logger.warning("Backbone did not return regime logits despite use_regime_film=True.")
                return None, None

        rl = regime_logits.float().squeeze(0).cpu().numpy()  # (L, R)
        log_p = rl - np.log(np.exp(rl).sum(axis=-1, keepdims=True) + 1e-12)
        p = np.exp(log_p)
        entropy = -(p * log_p).sum(axis=-1)  # (L,)
        argmax = rl.argmax(axis=-1)          # (L,)
        all_argmax.append(argmax)
        all_entropy.append(entropy)

    if not all_argmax:
        return None, None

    return np.concatenate(all_argmax), np.concatenate(all_entropy)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def temporal_persistence(argmax: np.ndarray) -> float:
    """Fraction of consecutive pairs that share the same regime argmax."""
    if argmax.size < 2:
        return float("nan")
    return float((argmax[1:] == argmax[:-1]).mean())


def usage_distribution(argmax: np.ndarray, num_regimes: int) -> dict[int, float]:
    """Fraction of timesteps assigned to each regime."""
    counts = np.bincount(argmax, minlength=num_regimes)
    return {int(k): float(v / counts.sum()) for k, v in enumerate(counts)}


def entropy_variability(entropy: np.ndarray, vol_window: int = 20) -> float:
    """Rolling std of entropy — high means assignments vary meaningfully over time."""
    if entropy.size < vol_window:
        return float(entropy.std())
    stds = [
        float(entropy[max(0, t - vol_window) : t + 1].std())
        for t in range(len(entropy))
    ]
    return float(np.mean(stds))


def vol_regime_spearman(
    argmax: np.ndarray,
    val_flat: np.ndarray,
    mid_idx: int,
    vol_window: int = 20,
) -> float:
    """Spearman |r| between argmax regime and rolling realised vol of mid."""
    T = min(len(argmax), val_flat.shape[0])
    dmid = np.diff(val_flat[:T, mid_idx], prepend=val_flat[0, mid_idx])
    rolling_vol = np.array([
        float(dmid[max(0, t - vol_window) : t + 1].std())
        for t in range(T)
    ])
    # Spearman: rank correlation.
    from scipy.stats import spearmanr
    r, _ = spearmanr(argmax[:T], rolling_vol)
    return float(abs(r)) if np.isfinite(r) else float("nan")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _maybe_plot(
    argmax: np.ndarray,
    entropy: np.ndarray,
    val_flat: np.ndarray,
    mid_idx: int,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping regime plot.")
        return

    T = min(len(argmax), val_flat.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(val_flat[:T, mid_idx], linewidth=0.8, color="steelblue")
    axes[0].set_ylabel("mid_norm")
    axes[0].set_title("Normalised midprice")

    axes[1].step(np.arange(T), argmax[:T], linewidth=0.8, color="darkorange")
    axes[1].set_ylabel("regime argmax")
    axes[1].set_title("Regime assignment (argmax)")

    axes[2].plot(entropy[:T], linewidth=0.8, color="green")
    axes[2].set_ylabel("entropy (nats)")
    axes[2].set_title("Per-step regime entropy")
    axes[2].set_xlabel("tick")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    # Load validation sequence.
    import yaml
    from finmamba3.config import DotDict, parse_args_and_update_config
    from finmamba3.sequence_builder import build_sequences

    with open(args.config) as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)

    norm_clip = cfg.BasicSettings.get("NormClip", 8.0)
    aggregate_only = cfg.Models.WorldModel.Encoder.get("AggregateOnly", False)
    include_binary = cfg.Models.WorldModel.Encoder.get("BinaryMarketFeatures", False)
    val_seq, slug, _ = build_sequences(
        args.data_val, args.market_slug, args.hours_val, args.norm_path,
        fit_stats=False, norm_clip=norm_clip,
        aggregate_only=aggregate_only, include_binary_features=include_binary,
    )
    val_flat = val_seq.to_flat()
    logger.info(f"val market: {slug}, {val_flat.shape[0]} ticks, {val_flat.shape[1]} features")

    # Extract regime assignments.
    argmax, entropy = extract_regime_logits(
        args.checkpoint, args.config, val_flat, device, batch_len=args.batch_len,
    )
    if argmax is None:
        logger.error("No regime logits extracted.  See warnings above.")
        return 1

    # Derive mid_idx from the config schema (K * F_level).
    k = int(cfg.Models.WorldModel.Encoder.K)
    f_level = int(cfg.Models.WorldModel.Encoder.FeatureDimLevel)
    mid_idx = k * f_level   # tick.mid is the first tick-level feature

    num_regimes = int(cfg.Models.WorldModel.RegimeFiLM.get("NumRegimes", 8))
    persist = temporal_persistence(argmax)
    usage = usage_distribution(argmax, num_regimes)
    ent_var = entropy_variability(entropy, args.vol_window)

    try:
        vol_corr = vol_regime_spearman(argmax, val_flat, mid_idx, args.vol_window)
    except Exception:
        vol_corr = float("nan")

    # Interpretation guidance baked into the summary.
    persist_flag = "OK" if persist > 0.5 else "LOW – regime flickering, may be learning noise"
    dominant_regime = max(usage, key=usage.get)
    usage_flag = "OK" if usage[dominant_regime] < 0.8 else f"HIGH – regime {dominant_regime} dominates ({usage[dominant_regime]:.1%})"
    vol_corr_flag = "OK" if vol_corr > 0.2 else "LOW – head may not have learned volatility structure"

    summary = {
        "checkpoint": str(args.checkpoint),
        "market_slug": slug,
        "num_ticks_analysed": int(len(argmax)),
        "num_regimes": num_regimes,
        "temporal_persistence": persist,
        "temporal_persistence_flag": persist_flag,
        "usage_distribution": usage,
        "usage_flag": usage_flag,
        "entropy_variability": ent_var,
        "vol_regime_spearman_abs": vol_corr,
        "vol_corr_flag": vol_corr_flag,
    }

    logger.info(f"temporal persistence:   {persist:.4f}  [{persist_flag}]")
    logger.info(f"regime usage: " + "  ".join(f"r{k}={v:.2%}" for k, v in usage.items()))
    logger.info(f"usage flag:             {usage_flag}")
    logger.info(f"entropy variability:    {ent_var:.4f}")
    logger.info(f"vol-regime |Spearman|:  {vol_corr:.4f}  [{vol_corr_flag}]")

    out_json = args.out_dir / f"regime_diag_{slug}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info(f"summary written to {out_json}")

    _maybe_plot(argmax, entropy, val_flat, mid_idx,
                args.out_dir / f"regime_plot_{slug}.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
