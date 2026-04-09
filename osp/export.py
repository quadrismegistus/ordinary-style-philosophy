"""
Export derived data for publication.

Exports numeric/derived data that doesn't contain copyrighted text:
  - metadata.csv (bibliographic info only)
  - slice_features.csv (per-slice feature vectors)
  - predictions.csv (classifier predictions per slice)
  - feature_weights.csv (logistic regression weights per comparison)

Usage:
    python -m osp.export                    # export all
    python -m osp.export --output data/release/
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm


def export_metadata(output_dir):
    """Export corpus metadata (bibliographic info, no text)."""
    from .data_loaders import get_corpus_metadata

    df = get_corpus_metadata()
    # Drop any columns that might contain text content
    safe_cols = [
        "title", "author", "year", "journal", "volume", "issue",
        "discipline", "period", "decade", "publisher",
    ]
    safe_cols = [c for c in safe_cols if c in df.columns]
    out = df[safe_cols]
    path = os.path.join(output_dir, "metadata.csv")
    out.to_csv(path)
    print(f"  metadata: {len(out)} rows -> {path}")
    return out


def export_slice_features(output_dir):
    """Export the per-slice feature matrix (numeric only, no text)."""
    from .features import get_all_feats_stashed

    df = get_all_feats_stashed()
    path = os.path.join(output_dir, "slice_features.csv")
    df.to_csv(path)
    print(f"  slice_features: {len(df)} rows x {len(df.columns)} features -> {path}")
    return df


def export_predictions(output_dir):
    """Export classifier predictions and feature weights."""
    from .classify import get_preds_feats

    result = get_preds_feats()
    df_preds, df_feats = result[0], result[1]

    # Predictions: drop any text-containing columns, keep only IDs and probabilities
    pred_cols = [c for c in df_preds.columns if c.startswith("prob_") or c in (
        "true_label", "pred_label", "test_label", "confidence", "correct",
        "accuracy", "support", "run", "predict_type", "comparison",
    )]
    out_preds = df_preds[pred_cols].copy()
    path_preds = os.path.join(output_dir, "predictions.csv")
    out_preds.to_csv(path_preds)
    print(f"  predictions: {len(out_preds)} rows -> {path_preds}")

    # Feature weights
    path_feats = os.path.join(output_dir, "feature_weights.csv")
    df_feats.to_csv(path_feats, index=False)
    print(f"  feature_weights: {len(df_feats)} rows -> {path_feats}")

    return out_preds, df_feats


def main():
    parser = argparse.ArgumentParser(
        description="Export derived data for publication (no copyrighted text)."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/release",
        help="Output directory (default: data/release/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Exporting to {args.output}/\n")

    export_metadata(args.output)
    export_slice_features(args.output)
    export_predictions(args.output)

    print(f"\nDone. Files in {args.output}/")


if __name__ == "__main__":
    main()
