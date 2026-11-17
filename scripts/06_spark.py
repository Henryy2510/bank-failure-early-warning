"""Part 3 & 5: PySpark reproduction of the core steps, execution plan, multi-scale JVM RSS, parity & MLlib demo."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl
import psutil

from src.bfews.core import CORE_COLS, core_polars
from src.bfews.io import bronze_financials_path
from src.bfews.spark_jobs import (SPARK_OUT, get_spark_session, run_spark_mllib_demo, spark_core, spark_plan_text,
                                  spark_sample_rows, spark_yearly_summary)
from src.bfews.utils import data_dir, get_logger, out_dir, save_json

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


def _run_spark_worker(src_p: Path) -> None:
    """Fresh-JVM worker computing total rows and checksum summary via Spark core transformations."""
    from pyspark.sql import functions as F

    t0 = time.perf_counter()
    spark = get_spark_session("BFT65B-ScaleWorker")
    t_startup = time.perf_counter() - t0
    t1 = time.perf_counter()
    df = spark_core(spark, src_p)
    agg_row = df.agg(
        F.count("*").alias("rows"),
        *[F.sum(c).alias(f"{c}__sum") for c in SPARK_OUT],
    ).collect()[0].asDict()
    t_compute = time.perf_counter() - t1
    spark.stop()
    print("SPARK_WORKER_JSON:" + json.dumps({
        "rows": int(agg_row["rows"]),
        "spark_startup_s": round(t_startup, 3),
        "compute_s": round(t_compute, 3),
        "total_s": round(t_startup + t_compute, 3),
    }))


def measure_spark_subprocess(src_p: Path, scale: int, repeat: int = 3) -> dict:
    """Measure fresh Python+JVM Spark execution while polling process-tree RSS (including child Java JVM)."""
    runs = []
    for _ in range(repeat):
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--spark-worker", str(src_p)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        p_util = psutil.Process(proc.pid)
        peak_bytes = 0
        while proc.poll() is None:
            try:
                procs = [p_util] + p_util.children(recursive=True)
                rss = sum(pr.memory_info().rss for pr in procs if pr.is_running())
                if rss > peak_bytes:
                    peak_bytes = rss
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Spark worker failed (scale={scale}x): {err}")
        payload_line = [ln for ln in out.splitlines() if ln.startswith("SPARK_WORKER_JSON:")]
        if not payload_line:
            raise RuntimeError(f"Missing SPARK_WORKER_JSON output: {out}\n{err}")
        d = json.loads(payload_line[-1].split("SPARK_WORKER_JSON:", 1)[1])
        d["peak_rss_tree_mb"] = round(peak_bytes / 1024**2, 1)
        runs.append(d)
    runs_sorted = sorted(runs, key=lambda r: r["total_s"])
    med = runs_sorted[len(runs_sorted) // 2]
    return {
        "scale": f"{scale}x",
        "scale_factor": scale,
        "rows": med["rows"],
        "engine": "PySpark local[2]",
        "spark_startup_s": med["spark_startup_s"],
        "compute_s": med["compute_s"],
        "total_s": med["total_s"],
        "peak_rss_tree_mb": max(r["peak_rss_tree_mb"] for r in runs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-worker", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.spark_worker is not None:
        _run_spark_worker(args.spark_worker)
        return

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

    log.info("Multi-scale PySpark benchmark (1x & 4x) with Python+JVM process-tree peak RSS...")
    from scripts import __path__ as _scripts_pkg  # noqa: F401
    import importlib
    bench_mod = importlib.import_module("scripts.05_benchmark")
    tmp_spark = data_dir("bronze") / "_scale_spark_tmp"
    shutil.rmtree(tmp_spark, ignore_errors=True)
    spark_rows = [measure_spark_subprocess(bronze_p, 1, repeat=2)]
    try:
        p4 = bench_mod.build_scaled_parquet(bronze_p, tmp_spark / "s4", 4)
        spark_rows.append(measure_spark_subprocess(p4, 4, repeat=2))
    finally:
        shutil.rmtree(tmp_spark, ignore_errors=True)
    spark_df = pl.DataFrame(spark_rows)
    spark_df.write_csv(out_dir("tables") / "spark_scaling_results.csv")
    log.info(f"PySpark multi-scale results:\n{spark_df}")

    p3_csv = out_dir("tables") / "part3_benchmark_results.csv"
    if p3_csv.exists():
        bench_mod.plot(pl.read_csv(p3_csv), spark_df)

    log.info("MLlib demo (development-only split)...")
    save_json(run_spark_mllib_demo(), out_dir("tables") / "spark_mllib_results.json")
    log.info("Part 3 / 5 Spark tasks complete.")


if __name__ == "__main__":
    main()
