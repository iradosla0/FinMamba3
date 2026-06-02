"""Market-level regime gap evaluation.

Wires the regime_split infrastructure into the world-model validation metrics so
the in-regime / out-of-regime generalisation gap can be measured at the market
level — i.e. using whole contracts as the split unit rather than individual
windows.

Two split strategies are supported:

* ``time`` – markets resolved before a cutoff timestamp go to the in-regime
  (train) set; markets resolved after go to the out-of-regime (test) set.
* ``volatility`` – markets whose realised mid-vol is below the training-set
  median go to the in-regime set; markets above the median go to the
  out-of-regime set.

For each split the script loads the world-model checkpoint, samples random
windows from markets in each subset, runs ``_validation_metrics`` on each
subset independently, and reports every metric together with the gap
(out-of-regime minus in-regime).  Results are written as a JSON file and
printed as a markdown table.

This is the primary experiment for the RegimeFiLM paper claim: if
``Val/regime_gap/normalized_next_mse`` is smaller for the FiLM model than for
the unmodulated baseline, the hypothesis is supported.

Example
-------
    # Volatility split (recommended for the ablation):
    python -m finmamba3.eval.regime_gap_cli \\
        --checkpoint saved_models/lob/LOB/<run>/ckpt/world_model_best.pth \\
        --config configs/lob.yaml \\
        --data-train data/train \\
        --data-val data/validation \\
        --split-strategy volatility \\
        --out reports/regime_gap.json

    # Time split:
    python -m finmamba3.eval.regime_gap_cli \\
        ... \\
        --split-strategy time \\
        --cutoff-ts 1710000000
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
    p = argparse.ArgumentParser(description="Market-level regime gap evaluation")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-train", required=True, type=Path,
                   help="Training data directory (used to fit vol thresholds).")
    p.add_argument("--data-val", required=True, type=Path,
                   help="Validation data directory (markets to evaluate on).")
    p.add_argument("--norm-path", type=Path,
                   default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    p.add_argument("--split-strategy", choices=("time", "volatility"), default="volatility")
    p.add_argument("--cutoff-ts", type=float, default=None,
                   help="Unix-second cutoff for time split (required when --split-strategy=time).")
    p.add_argument("--vol-quantile", type=float, default=0.5,
                   help="Quantile for volatility split (default 0.5 = median).")
    p.add_argument("--hours-train", type=float, default=6.0)
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--batch-length", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("reports/regime_gap.json"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Per-split validation metrics
# ---------------------------------------------------------------------------

def _run_metrics_on_flat(
    world_model,
    flat: np.ndarray,
    stats,
    batch_size: int,
    batch_length: int,
) -> dict[str, float]:
    """Run _validation_metrics on an arbitrary flat feature array.

    Mirrors the logic in train._validation_metrics but accepts a pre-built
    numpy array so it can be called on arbitrary market subsets without
    rebuilding the sequence objects.
    """
    from finmamba3.train import validate
    from finmamba3.envs.lob_features import LOBSequence

    T = flat.shape[0]
    if T < batch_length + 1:
        logger.warning(f"Split has only {T} ticks; skipping (need >= {batch_length + 1}).")
        return {}

    # Wrap the flat array in a minimal LOBSequence so validate() can call to_flat().
    k = world_model.encoder.k_levels
    f_level = world_model.encoder.f_level
    f_tick = flat.shape[1] - k * f_level
    per_level = flat[:, : k * f_level].reshape(T, k, f_level)
    per_tick = flat[:, k * f_level :]
    seq = LOBSequence(
        market_slug="__subset__",
        per_level=per_level.astype(np.float32),
        per_tick=per_tick.astype(np.float32),
        midprice=np.zeros(T, dtype=np.float32),  # not used in validate()
        ts_sec=np.arange(T, dtype=np.int64),
        yes_outcome=None,
    )
    metrics, _ = validate(world_model, seq, stats, batch_size, batch_length)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    if args.split_strategy == "time" and args.cutoff_ts is None:
        logger.error("--cutoff-ts is required when --split-strategy=time")
        return 1

    import yaml
    from finmamba3.config import DotDict, parse_args_and_update_config
    from finmamba3.models.world_model import WorldModel
    from finmamba3.backtester.data_loader import build_timeline
    from finmamba3.envs.lob_features import (
        extract_features, apply_normalization, load_normalization,
        pick_longest_market, make_aggregate_only,
    )
    from finmamba3.eval.regime_split import (
        RegimeSplitResult, time_split, volatility_split, realized_vol_from_timeline,
    )

    with open(args.config) as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)
    norm_clip = cfg.BasicSettings.get("NormClip", 8.0)
    aggregate_only = cfg.Models.WorldModel.Encoder.get("AggregateOnly", False)
    include_binary = cfg.Models.WorldModel.Encoder.get("BinaryMarketFeatures", False)
    stats = load_normalization(args.norm_path)

    # Build world model.
    wm = WorldModel(action_dim=13, config=cfg, device=device).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = state.get("world_model", state.get("state_dict", state))
    wm.load_state_dict(sd, strict=False)
    wm.eval()
    logger.info(f"loaded checkpoint: {args.checkpoint}")

    # Build timelines.
    train_bt = build_timeline(data_dir=args.data_train, hours=args.hours_train)
    val_bt   = build_timeline(data_dir=args.data_val,   hours=args.hours_val)

    markets = list(train_bt.lifecycles) + list(val_bt.lifecycles)
    all_timeline = train_bt.timeline + val_bt.timeline

    # Produce split.
    if args.split_strategy == "time":
        split: RegimeSplitResult = time_split(markets, args.cutoff_ts)
    else:
        realized_vol = realized_vol_from_timeline(all_timeline)
        split = volatility_split(markets, realized_vol, quantile=args.vol_quantile)

    logger.info(
        f"split strategy={args.split_strategy}: "
        f"{len(split.train_markets)} in-regime, {len(split.test_markets)} out-of-regime markets"
    )
    if not split.train_markets or not split.test_markets:
        logger.error("One side of the split is empty.  Cannot compute a gap.")
        return 1

    def _build_flat_for_markets(lifecycles, bt) -> np.ndarray | None:
        """Concatenate normalised flat features for a list of market lifecycles."""
        slugs = {lc.market_slug for lc in lifecycles}
        chunks = []
        for slug in slugs:
            try:
                seq = extract_features(bt.timeline, slug)
            except Exception as e:
                logger.warning(f"  skipping {slug}: {e}")
                continue
            seq_norm = apply_normalization(seq, stats)
            if aggregate_only:
                seq_norm = make_aggregate_only(seq_norm)
            flat = seq_norm.to_flat()
            if flat.shape[0] >= args.batch_length + 1:
                chunks.append(flat)
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    # Collect flat arrays for each side of the split.
    # Markets may appear in train or val timeline; check both.
    def _get_bt_for_slug(slug: str):
        for lc in train_bt.lifecycles:
            if lc.market_slug == slug:
                return train_bt
        return val_bt

    in_regime_chunks = []
    out_regime_chunks = []
    for lc in split.train_markets:
        bt = _get_bt_for_slug(lc.market_slug)
        try:
            seq = extract_features(bt.timeline, lc.market_slug)
            seq_norm = apply_normalization(seq, stats)
            if aggregate_only:
                seq_norm = make_aggregate_only(seq_norm)
            flat = seq_norm.to_flat()
            if flat.shape[0] >= args.batch_length + 1:
                in_regime_chunks.append(flat)
        except Exception as e:
            logger.warning(f"  in-regime: skipping {lc.market_slug}: {e}")

    for lc in split.test_markets:
        bt = _get_bt_for_slug(lc.market_slug)
        try:
            seq = extract_features(bt.timeline, lc.market_slug)
            seq_norm = apply_normalization(seq, stats)
            if aggregate_only:
                seq_norm = make_aggregate_only(seq_norm)
            flat = seq_norm.to_flat()
            if flat.shape[0] >= args.batch_length + 1:
                out_regime_chunks.append(flat)
        except Exception as e:
            logger.warning(f"  out-of-regime: skipping {lc.market_slug}: {e}")

    if not in_regime_chunks or not out_regime_chunks:
        logger.error("Could not build both sides of the split after filtering.  Aborting.")
        return 1

    in_flat  = np.concatenate(in_regime_chunks,  axis=0)
    out_flat = np.concatenate(out_regime_chunks, axis=0)
    logger.info(f"in-regime:      {in_flat.shape[0]:,} ticks across {len(in_regime_chunks)} markets")
    logger.info(f"out-of-regime:  {out_flat.shape[0]:,} ticks across {len(out_regime_chunks)} markets")

    # Wrap stats in a minimal dummy object that validate() needs.
    from finmamba3.envs.lob_features import LOBSequence
    logger.info("running validation metrics on in-regime split …")
    in_metrics  = _run_metrics_on_flat(wm, in_flat,  stats, args.batch_size, args.batch_length)
    logger.info("running validation metrics on out-of-regime split …")
    out_metrics = _run_metrics_on_flat(wm, out_flat, stats, args.batch_size, args.batch_length)

    # Compute gap for every shared metric key.
    all_keys = sorted(set(in_metrics) & set(out_metrics))
    rows = []
    for key in all_keys:
        iv = in_metrics[key]
        ov = out_metrics[key]
        gap = (ov - iv) if (np.isfinite(iv) and np.isfinite(ov)) else float("nan")
        rows.append({"metric": key, "in_regime": iv, "out_of_regime": ov, "gap_out_minus_in": gap})

    # Print markdown table.
    header = "| metric | in-regime | out-of-regime | gap (out − in) |"
    sep    = "|--------|----------:|-------------:|---------------:|"
    print(header)
    print(sep)
    for r in rows:
        iv = r["in_regime"]
        ov = r["out_of_regime"]
        gp = r["gap_out_minus_in"]
        print(f"| {r['metric']} | {iv:.4f} | {ov:.4f} | {gp:+.4f} |")

    result = {
        "checkpoint": str(args.checkpoint),
        "split_strategy": args.split_strategy,
        "split_description": split.description,
        "in_regime_ticks": int(in_flat.shape[0]),
        "out_of_regime_ticks": int(out_flat.shape[0]),
        "in_regime_markets": len(in_regime_chunks),
        "out_of_regime_markets": len(out_regime_chunks),
        "metrics": rows,
    }
    args.out.write_text(json.dumps(result, indent=2))
    logger.info(f"results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
