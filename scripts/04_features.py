"""Part 4: build the Gold feature table (features + 4-quarter label + split) and summarise it."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl

from src.bfews.features import FEATURE_COLS_CAMELS, build_gold_features
from src.bfews.utils import get_logger, out_dir, save_json

log = get_logger("04_features")


def main() -> None:
    log.info("Part 4: Gold features and labels...")
    gold = build_gold_features()

    def summarise(frame: pl.DataFrame) -> pl.DataFrame:
        return (frame.group_by("split_group")
                .agg(pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks"),
                     pl.col("target_fail_4q").sum().alias("positive_rows"),
                     pl.col("CERT").filter(pl.col("target_fail_4q") == 1).n_unique().alias("failing_banks"),
                     (pl.col("target_fail_4q").mean() * 100).round(3).alias("positive_rate_pct"),
                     pl.col("REPDTE").min().alias("first_quarter"), pl.col("REPDTE").max().alias("last_quarter"))
                .sort("first_quarter"))

    all_rows = summarise(gold).with_columns(pl.lit("all rows").alias("scope"))
    scorable = summarise(gold.filter(pl.col("pit_scorable"))).with_columns(pl.lit("pit_scorable only (used for modelling)").alias("scope"))
    out = pl.concat([all_rows, scorable])
    print(out)
    out.write_csv(out_dir("tables") / "gold_split_summary.csv")

    not_scorable = gold.filter(~pl.col("pit_scorable"))
    save_json({"gold_rows": gold.height, "n_features": len(FEATURE_COLS_CAMELS),
               "non_scorable_rows": not_scorable.height,
               "non_scorable_positive_rows": int(not_scorable["target_fail_4q"].sum())},
              out_dir("tables") / "gold_summary.json")
    log.info("Part 4 complete.")


if __name__ == "__main__":
    main()
