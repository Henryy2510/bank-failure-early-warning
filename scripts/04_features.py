"""Part 4: build the Gold feature table (features + 4-quarter label + split) and summarise it."""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["POLARS_MAX_THREADS"] = "4"

import numpy as np
import polars as pl

from src.bfews.features import FEATURE_COLS_CAMELS, build_gold_features, compute_camels_features
from src.bfews.io import FLOWS_YTD, bronze_financials_path
from src.bfews.panel import decumulate_ytd_flows
from src.bfews.utils import get_logger, out_dir, save_json

log = get_logger("04_features")


def run_full_panel_truncation_audit(cutoff: datetime.date = datetime.date(2015, 12, 31)) -> dict:
    """Prove exact causal invariance on real FDIC Bronze data (1984-2026 vs truncated <= cutoff) across all 26 features + 7 flows."""
    bronze = pl.read_parquet(bronze_financials_path(), hive_partitioning=True)
    full_feat = compute_camels_features(decumulate_ytd_flows(bronze).drop(["year", "quarter", "qidx"]))
    full_sub = full_feat.filter(pl.col("REPDTE") <= cutoff).sort(["CERT", "REPDTE"])

    trunc_raw = bronze.filter(pl.col("REPDTE") <= cutoff)
    trunc_feat = compute_camels_features(decumulate_ytd_flows(trunc_raw).drop(["year", "quarter", "qidx"])).sort(["CERT", "REPDTE"])

    assert full_sub.height == trunc_feat.height, f"Row count mismatch: {full_sub.height} vs {trunc_feat.height}"
    assert (full_sub["CERT"] == trunc_feat["CERT"]).all() and (full_sub["REPDTE"] == trunc_feat["REPDTE"]).all()

    check_cols = list(FEATURE_COLS_CAMELS) + [f"{c}_Q" for c in FLOWS_YTD]
    max_abs_diff = 0.0
    null_mismatches = 0
    for c in check_cols:
        a = full_sub[c]
        b = trunc_feat[c]
        null_mismatches += int((a.is_null() != b.is_null()).sum())
        both = a.is_not_null() & b.is_not_null()
        if int(both.sum()) > 0:
            diff = float((a.filter(both) - b.filter(both)).abs().max())
            if np.isfinite(diff):
                max_abs_diff = max(max_abs_diff, diff)

    payload = {
        "cutoff_date": str(cutoff),
        "full_bronze_rows": int(bronze.height),
        "truncated_rows_compared": int(trunc_feat.height),
        "future_rows_removed": int(bronze.height - trunc_feat.height),
        "features_checked": len(FEATURE_COLS_CAMELS),
        "ytd_flows_checked": len(FLOWS_YTD),
        "total_columns_verified": len(check_cols),
        "null_mask_mismatches": null_mismatches,
        "max_abs_diff_any_column": max_abs_diff,
        "exact_causal_invariance": bool(null_mismatches == 0 and max_abs_diff == 0.0),
    }
    save_json(payload, out_dir("tables") / "full_panel_truncation_audit.json")
    log.info(f"Full-panel causal truncation audit (<= {cutoff}): {payload}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truncation-audit-only", action="store_true", help="Only run the real-data 2015Q4 causal truncation audit.")
    args = ap.parse_args()
    if args.truncation_audit_only:
        res = run_full_panel_truncation_audit()
        if not res["exact_causal_invariance"]:
            raise AssertionError(f"Full-panel causal truncation check failed: {res}")
        return

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
    run_full_panel_truncation_audit()
    log.info("Part 4 complete.")


if __name__ == "__main__":
    main()
