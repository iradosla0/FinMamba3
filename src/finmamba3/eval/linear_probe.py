"""Linear probing of the world-model latent representation.

Extracts posterior latent vectors from a validation sequence and trains a
suite of linear classifiers on top of them.  A good latent representation
should support accurate linear prediction of properties that are present in
the input features; failure here reveals blind spots in what the world model
has compressed into the discrete bottleneck.

Probing targets (all derived from the same normalised flat features):

* ``vol_regime``        – current realised vol bucket (low / medium / high),
                          derived from rolling std of Δmid over a short window.
* ``imbalance_dir``     – whether the top-of-book bid/ask imbalance is
                          positive (bid-heavy) or negative (ask-heavy).
* ``spread_tier``       – whether the normalised spread is below / above the
                          median spread seen in the validation sequence.
* ``direction_next``    – 3-class next-tick midprice direction (uses the
                          same threshold as the Direction head in training).

For each target the script reports:

* Chance-level accuracy (majority class)
* Linear-probe accuracy (logistic regression on frozen latents, 5-fold CV)
* Δ accuracy = probe − chance  (the meaningful signal above baseline)

A Δ accuracy < 0.02 for a given target means the latent has discarded that
feature almost entirely.

Requirements: ``scikit-learn`` (``pip install scikit-learn``).

Example
-------
    python -m finmamba3.eval.linear_probe \\
        --checkpoint saved_models/lob/LOB/<run>/ckpt/world_model_best.pth \\
        --config configs/lob.yaml \\
        --data-val data/validation \\
        --out reports/linear_probe.json
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probing of world-model latents")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", type=Path,
                   default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    p.add_argument("--market-slug", default=None)
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--max-windows", type=int, default=512,
                   help="Number of random windows to sample for probe training.")
    p.add_argument("--window-len", type=int, default=32,
                   help="Length of each window (ticks).")
    p.add_argument("--vol-window", type=int, default=10,
                   help="Rolling window for vol-regime bucket derivation.")
    p.add_argument("--direction-threshold", type=float, default=0.01,
                   help="Bucket threshold for next-tick direction target (normalised units).")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("reports/linear_probe.json"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

def extract_latents(
    checkpoint: Path,
    config_path: Path,
    val_flat: np.ndarray,
    device: torch.device,
    max_windows: int,
    window_len: int,
) -> np.ndarray:
    """Return posterior latent vectors, shape (N, latent_dim).

    N = number of (window, timestep) pairs sampled.  The latent_dim is the
    flattened categorical sample dimension (CategoricalDim * ClassDim = 256
    in the default config).
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

    T = val_flat.shape[0]
    if T < window_len + 1:
        raise ValueError(f"Validation sequence too short ({T}) for window_len={window_len}.")

    rng = np.random.default_rng(42)
    n_windows = min(max_windows, T - window_len)
    starts = rng.integers(0, T - window_len, size=n_windows)

    all_latents: list[np.ndarray] = []
    batch_size = 32
    for batch_start in range(0, n_windows, batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        windows = np.stack([val_flat[s : s + window_len] for s in batch_starts], axis=0)
        obs = torch.from_numpy(windows).float().to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp
        ):
            embedding = wm.encoder(obs)                              # (B, L, enc_dim)
            post_logits = wm.dist_head.forward_post(embedding)      # (B, L, C, D)
            sample = wm.straight_through_gradient(post_logits)      # (B, L, C, D)
            flat_sample = wm.flatten_sample(sample)                  # (B, L, C*D)
        # Take the last timestep of each window as the representative latent.
        latents = flat_sample[:, -1, :].float().cpu().numpy()        # (B, C*D)
        all_latents.append(latents)

    return np.concatenate(all_latents, axis=0)   # (N, C*D)


# ---------------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------------

def _vol_regime_labels(
    val_flat: np.ndarray,
    starts: np.ndarray,
    window_len: int,
    mid_idx: int,
    vol_window: int,
) -> np.ndarray:
    """3-class vol-regime label for the last tick of each window."""
    labels = np.empty(len(starts), dtype=np.int64)
    for i, s in enumerate(starts):
        end = s + window_len
        chunk = val_flat[max(0, end - vol_window) : end, mid_idx]
        vol = float(np.diff(chunk).std()) if len(chunk) > 1 else 0.0
        labels[i] = int(vol > 0)  # temporarily binary; quantised below
    # Quantise into thirds relative to the observed distribution.
    q33, q67 = np.quantile(labels.astype(float), [0.33, 0.67])
    raw_vol = np.array([
        float(np.diff(val_flat[max(0, s + window_len - vol_window) : s + window_len, mid_idx]).std())
        for s in starts
    ])
    return np.where(raw_vol < q33, 0, np.where(raw_vol > q67, 2, 1))


def _imbalance_dir_labels(
    val_flat: np.ndarray,
    starts: np.ndarray,
    window_len: int,
    imbalance_idx: int,
) -> np.ndarray:
    """Binary: 0 = ask-heavy (imbalance < 0), 1 = bid-heavy (imbalance >= 0)."""
    imb = val_flat[starts + window_len - 1, imbalance_idx]
    return (imb >= 0.0).astype(np.int64)


def _spread_tier_labels(
    val_flat: np.ndarray,
    starts: np.ndarray,
    window_len: int,
    spread_idx: int,
) -> np.ndarray:
    """Binary: 0 = narrow spread (below median), 1 = wide spread."""
    spreads = val_flat[starts + window_len - 1, spread_idx]
    median = float(np.median(spreads))
    return (spreads >= median).astype(np.int64)


def _direction_next_labels(
    val_flat: np.ndarray,
    starts: np.ndarray,
    window_len: int,
    mid_idx: int,
    threshold: float,
) -> np.ndarray:
    """3-class next-tick midprice direction: 0=down, 1=flat, 2=up."""
    cur  = val_flat[starts + window_len - 1, mid_idx]
    nxt  = val_flat[np.minimum(starts + window_len, val_flat.shape[0] - 1), mid_idx]
    delta = nxt - cur
    return np.where(delta > threshold, 2, np.where(delta < -threshold, 0, 1))


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

class ProbeResult(NamedTuple):
    target: str
    chance_accuracy: float
    probe_accuracy: float
    delta_accuracy: float
    n_samples: int
    n_classes: int


def _probe(
    latents: np.ndarray,
    labels: np.ndarray,
    target_name: str,
    cv_folds: int,
) -> ProbeResult:
    """Train a logistic regression probe with cross-validation."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        logger.error("scikit-learn is required for linear probing: pip install scikit-learn")
        raise

    # Remove samples where label derivation failed (NaN encoded as -1).
    valid = labels >= 0
    X = latents[valid]
    y = labels[valid]
    if len(y) < cv_folds * 2:
        return ProbeResult(target_name, float("nan"), float("nan"), float("nan"), len(y), 0)

    classes, counts = np.unique(y, return_counts=True)
    chance = float(counts.max() / counts.sum())

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, multi_class="auto", solver="lbfgs"),
    )
    scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring="accuracy")
    probe_acc = float(scores.mean())
    return ProbeResult(
        target=target_name,
        chance_accuracy=chance,
        probe_accuracy=probe_acc,
        delta_accuracy=probe_acc - chance,
        n_samples=int(len(y)),
        n_classes=int(len(classes)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

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
    logger.info(f"val market: {slug}, {val_flat.shape[0]} ticks")

    # Feature indices.
    k = int(cfg.Models.WorldModel.Encoder.K)
    f_level = int(cfg.Models.WorldModel.Encoder.FeatureDimLevel)
    mid_idx       = k * f_level + 0   # tick.mid
    spread_idx    = k * f_level + 1   # tick.spread
    imbalance_idx = k * f_level + 3   # tick.imbalance

    # Extract latents.
    logger.info("extracting posterior latents …")
    latents = extract_latents(
        args.checkpoint, args.config, val_flat, device,
        args.max_windows, args.window_len,
    )
    logger.info(f"extracted {latents.shape[0]} latent vectors of dim {latents.shape[1]}")

    # Build window start indices to match latents (same rng seed as extract_latents).
    T = val_flat.shape[0]
    rng = np.random.default_rng(42)
    n_windows = min(args.max_windows, T - args.window_len)
    starts = rng.integers(0, T - args.window_len, size=n_windows)
    # Trim starts to match actual latent count (in case some batches were shorter).
    starts = starts[: latents.shape[0]]

    # Derive labels.
    vol_labels   = _vol_regime_labels(val_flat, starts, args.window_len, mid_idx, args.vol_window)
    imb_labels   = _imbalance_dir_labels(val_flat, starts, args.window_len, imbalance_idx)
    spr_labels   = _spread_tier_labels(val_flat, starts, args.window_len, spread_idx)
    dir_labels   = _direction_next_labels(
        val_flat, starts, args.window_len, mid_idx, args.direction_threshold,
    )

    # Run probes.
    results: list[ProbeResult] = []
    for target, labels in [
        ("vol_regime",   vol_labels),
        ("imbalance_dir", imb_labels),
        ("spread_tier",  spr_labels),
        ("direction_next", dir_labels),
    ]:
        logger.info(f"  probing {target} …")
        try:
            r = _probe(latents, labels, target, args.cv_folds)
        except ImportError:
            return 1
        results.append(r)
        flag = "OK" if r.delta_accuracy > 0.02 else "LOW – latent may have discarded this feature"
        logger.info(
            f"  {target:20s}: chance={r.chance_accuracy:.3f}  "
            f"probe={r.probe_accuracy:.3f}  Δ={r.delta_accuracy:+.3f}  [{flag}]"
        )

    # Print markdown table.
    print("\n| target | chance | probe | Δ accuracy | n_samples | note |")
    print("|--------|-------:|------:|-----------:|----------:|------|")
    for r in results:
        flag = "OK" if r.delta_accuracy > 0.02 else "⚠ below 0.02"
        print(
            f"| {r.target} | {r.chance_accuracy:.3f} | {r.probe_accuracy:.3f} "
            f"| {r.delta_accuracy:+.3f} | {r.n_samples} | {flag} |"
        )

    output = {
        "checkpoint": str(args.checkpoint),
        "market_slug": slug,
        "n_latents": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "direction_threshold": args.direction_threshold,
        "cv_folds": args.cv_folds,
        "results": [r._asdict() for r in results],
    }
    args.out.write_text(json.dumps(output, indent=2))
    logger.info(f"results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
