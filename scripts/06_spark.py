"""Part 3 & 5: PySpark reproduction of the core steps, execution plan, parity with Polars, timing, MLlib demo."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl

from src.bfews.core import CORE_COLS, core_polars
from src.bfews.io import bronze_financials_path
from src.bfews.spark_jobs import (SPARK_OUT, get_spark_session, run_spark_mllib_demo, spark_core, spark_plan_text,
                                  spark_sample_rows, spark_yearly_summary)
from src.bfews.utils import get_logger, out_dir, save_json

log = get_logger("06_spark")
SAMPLE_MODULO = 50


def polars_reference(bronze_p: Path):
    """Same core steps in Polars: per-year fingerprint and the same sample of banks."""
    t0 = time.perf_counter()
    out = core_polars(pl.scan_parquet(bronze_p, hive_partitioning=True).select(CORE_COLS + ["YEAR"])).collect()
    yearly = (out.group_by("YEAR")
              .agg(pl.len().alias("rows"),
                   *[x for c in SPARK_OUT for x in (pl.col(c).sum().alias(f"{c}__sum"),
                                                    pl.col(c).null_count().alias(f"{c}__nulls"))])
              .sort("YEAR"))
    seconds = time.perf_counter() - t0
    sample = out.filter(pl.col("CERT") % SAMPLE_MODULO == 0).select(["CERT", "REPDTE"] + SPARK_OUT)
    return yearly, sample, seconds


def compare_yearly(spark_rows: list[dict], polars_yearly: pl.DataFrame) -> dict:
    sp = pl.DataFrame(spark_rows).sort("YEAR")
    assert sp["YEAR"].to_list() == polars_yearly["YEAR"].to_list(), "different set of years"
    worst, exact_counts = 0.0, True
    for c in sp.columns:
        if c == "YEAR":
            continue
        a, b = sp[c].cast(pl.Float64).to_numpy(), polars_yearly[c].cast(pl.Float64).to_numpy()
        if c == "rows" or c.endswith("__nulls"):
            exact_counts &= bool((a == b).all())
        else:
            worst = max(worst, float((abs(a - b) / pl.Series([max(1.0, abs(x)) for x in b]).to_numpy()).max()))
    return {"years_compared": sp.height, "row_and_null_counts_identical": exact_counts, "max_relative_diff_of_sums": worst}


def compare_rows(spark_rows: list[dict], polars_sample: pl.DataFrame) -> dict:
    sp = pl.DataFrame(spark_rows).with_columns(pl.col("CERT").cast(pl.Int32))
    j = sp.join(polars_sample.with_columns(pl.col("CERT").cast(pl.Int32)), on=["CERT", "REPDTE"], how="full", suffix="_pl")
    unmatched = int(j.filter(pl.col("CERT").is_null() | pl.col("CERT_pl").is_null()).height) if "CERT_pl" in j.columns else 0
    worst = 0.0
    for c in SPARK_OUT:
        both = j.filter(pl.col(c).is_not_null() & pl.col(f"{c}_pl").is_not_null())
        null_mismatch = int(j.filter(pl.col(c).is_null() != pl.col(f"{c}_pl").is_null()).height)
        if null_mismatch:
            return {"rows_compared": j.height, "null_pattern_mismatch": null_mismatch, "max_abs_diff": float("inf")}
        if both.height:
            worst = max(worst, float((both[c] - both[f"{c}_pl"]).abs().max()))
    return {"banks_sampled": sp["CERT"].n_unique(), "rows_compared": j.height, "rows_only_in_one_engine": unmatched,
            "max_abs_diff_any_column": worst}


def main() -> None:
    bronze_p = bronze_financials_path()
    log.info("Polars reference (same steps)...")
    pl_yearly, pl_sample, t_polars = polars_reference(bronze_p)
    log.info(f"Polars lazy collect + aggregate: {t_polars:.2f}s")

    t0 = time.perf_counter()
    spark = get_spark_session()
    t_session = time.perf_counter() - t0
    df = spark_core(spark, bronze_p)
    (out_dir("tables") / "spark_execution_plan.txt").write_text(spark_plan_text(df), encoding="utf-8")

    t0 = time.perf_counter()
    sp_yearly = spark_yearly_summary(df)
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    spark_yearly_summary(df)
    t_warm = time.perf_counter() - t0
    sp_rows = spark_sample_rows(df, SAMPLE_MODULO)
    spark.stop()
    log.info(f"Spark: session {t_session:.1f}s, job cold {t_cold:.1f}s, job warm {t_warm:.1f}s")

    parity = {"yearly_fingerprint": compare_yearly(sp_yearly, pl_yearly),
              "row_level_sample": compare_rows(sp_rows, pl_sample),
              "timing_seconds": {"polars_lazy_collect_plus_aggregate": round(t_polars, 2),
                                 "spark_session_startup": round(t_session, 2),
                                 "spark_job_cold": round(t_cold, 2), "spark_job_warm": round(t_warm, 2),
                                 "note": "single in-process runs on the full panel; Spark = local[2], 1.2 GB driver"}}
    log.info(f"Parity: {parity}")
    save_json(parity, out_dir("tables") / "spark_polars_parity.json")

    log.info("MLlib demo (development-only split)...")
    save_json(run_spark_mllib_demo(), out_dir("tables") / "spark_mllib_results.json")
    log.info("Part 3 / 5 Spark tasks complete.")


if __name__ == "__main__":
    main()
