"""Part 4: Build Leak-Free Gold Feature Panel with CAMELS Ratios, Lags, and Target Labels."""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import polars as pl

from src.bfews.features import FEATURE_COLS_CAMELS, build_gold_features
from src.bfews.utils import get_logger, out_dir

log = get_logger("04_features")


def main():
    log.info("Starting Part 4: Gold Feature Engineering & Label Generation...")
    gold_df = build_gold_features()

    split_summary = (
        gold_df.group_by("split_group")
        .agg([
            pl.len().alias("total_rows"),
            pl.col("CERT").n_unique().alias("unique_banks"),
            pl.col("target_fail_4q").sum().alias("failure_labels"),
            (pl.col("target_fail_4q").mean() * 100).round(3).alias("failure_rate_pct"),
            pl.col("REPDTE").min().alias("min_quarter"),
            pl.col("REPDTE").max().alias("max_quarter"),
        ])
        .sort("min_quarter")
    )

    log.info("Gold Dataset Split Summary:")
    print(split_summary)

    out_csv = out_dir("tables") / "gold_split_summary.csv"
    split_summary.write_csv(out_csv)
    log.info(f"Saved gold split summary to {out_csv}")
    log.info("Part 4 complete.")


if __name__ == "__main__":
    main()
