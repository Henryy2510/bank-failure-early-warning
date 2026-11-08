"""Part 3 & 5: PySpark Core Pipeline, Execution Plan Analysis, and MLlib Modeling Demo."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl

from src.bfews.io import bronze_financials_path
from src.bfews.spark_jobs import run_spark_mllib_demo, run_spark_pipeline
from src.bfews.utils import data_dir, get_logger, out_dir, save_json

log = get_logger("06_spark")


def main():
    log.info("Starting Part 3 & 5: PySpark Pipeline & MLlib Demo...")
    bronze_p = bronze_financials_path()

    # 1. Compute Polars reference metrics on Bronze using identical formulas (low memory)
    log.info("Computing Polars reference summary on Bronze for parity verification...")
    eps = 1e-4
    polars_ref = (
        pl.scan_parquet(bronze_p, hive_partitioning=True)
        .select(["CERT", "REPDTE", "YEAR", "ASSET", "EQ", "LNLSGR", "NCLNLS", "ORE", "LNATRES", "NETINC"])
        .sort(["CERT", "REPDTE"])
        .with_columns(pl.col("REPDTE").dt.quarter().alias("quarter"))
        .with_columns([
            pl.when(pl.col("quarter") == 1)
            .then(pl.col("NETINC"))
            .otherwise(pl.col("NETINC") - pl.col("NETINC").shift(1).over(["CERT", "YEAR"]))
            .alias("NETINC_Q"),
            (pl.col("EQ") / (pl.col("ASSET") + eps)).alias("eq_asset_ratio"),
            (
                (pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0))
                / (pl.col("EQ").fill_null(0.0) + pl.col("LNATRES").fill_null(0.0) + eps)
            ).alias("texas_ratio"),
        ])
        .select([
            pl.len().alias("row_count"),
            pl.col("eq_asset_ratio").mean().round(5).alias("mean_eq_ratio"),
            pl.col("texas_ratio").mean().round(5).alias("mean_texas_ratio"),
            pl.col("NETINC_Q").drop_nulls().mean().round(2).alias("mean_netinc_q"),
        ])
        .collect()
    )
    polars_summary = {
        "row_count": int(polars_ref["row_count"].item()),
        "mean_eq_ratio": float(polars_ref["mean_eq_ratio"].item()),
        "mean_texas_ratio": float(polars_ref["mean_texas_ratio"].item()),
        "mean_netinc_q": float(polars_ref["mean_netinc_q"].item()),
    }
    log.info(f"Polars Summary:  {polars_summary}")

    # 2. Run PySpark Core Transformations
    log.info("Running PySpark Core Pipeline on Bronze Parquet...")
    spark_summary, plan_text = run_spark_pipeline(bronze_p)

    log.info(f"PySpark Summary: {spark_summary}")
    plan_out = out_dir("tables") / "spark_execution_plan.txt"
    plan_out.write_text(plan_text, encoding="utf-8")
    log.info(f"Saved PySpark formatted execution plan to {plan_out}")

    diff = {
        "row_count_diff": spark_summary["row_count"] - polars_summary["row_count"],
        "mean_eq_ratio_abs_diff": round(abs(spark_summary["mean_eq_ratio"] - polars_summary["mean_eq_ratio"]), 6),
        "mean_texas_ratio_abs_diff": round(abs(spark_summary["mean_texas_ratio"] - polars_summary["mean_texas_ratio"]), 6),
        "mean_netinc_q_abs_diff": round(abs(spark_summary["mean_netinc_q"] - polars_summary["mean_netinc_q"]), 4),
    }
    log.info(f"Engine Discrepancy Verification: {diff}")
    save_json({"spark": spark_summary, "polars": polars_summary, "difference": diff}, out_dir("tables") / "spark_polars_parity.json")

    # 3. Run PySpark MLlib Demo (Part 5 requirement)
    log.info("Running PySpark MLlib Logistic Regression on Gold Features...")
    mllib_results = run_spark_mllib_demo(data_dir("gold") / "features.parquet")
    log.info(f"MLlib Results: {mllib_results}")
    save_json(mllib_results, out_dir("tables") / "spark_mllib_results.json")

    log.info("Part 3 & 5 PySpark tasks completed successfully.")


if __name__ == "__main__":
    main()
