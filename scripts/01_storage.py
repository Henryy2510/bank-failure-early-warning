"""Part 1: Storage Layer Ingestion & Format/Partitioning Benchmarking."""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pyarrow.parquet as pq

from src.bfews.io import bronze_financials_path, read_raw_all, write_bronze_financials
from src.bfews.utils import data_dir, get_logger, out_dir

log = get_logger("01_storage")


def run_storage_benchmark(df: pl.DataFrame) -> pl.DataFrame:
    """Benchmark raw vs partitioned Parquet vs CSV on size and read queries."""
    bench_dir = data_dir("bronze") / "_benchmark_temp"
    bench_dir.mkdir(parents=True, exist_ok=True)

    csv_path = bench_dir / "financials.csv"
    single_snappy = bench_dir / "single_snappy.parquet"
    single_zstd = bench_dir / "single_zstd.parquet"
    part_zstd = bronze_financials_path()

    log.info("Preparing benchmark files (single parquet, csv)...")
    # Take sample for CSV to avoid excessive disk bloat on USB drive
    sample_df = df.filter(pl.col("REPDTE").dt.year() >= 2020)
    sample_df.write_csv(csv_path)

    df.write_parquet(single_snappy, compression="snappy")
    df.write_parquet(single_zstd, compression="zstd", compression_level=3)

    # Size measurements
    def get_size_mb(p: Path) -> float:
        if p.is_file():
            return p.stat().st_size / (1024 * 1024)
        return sum(f.stat().st_size for f in p.glob("**/*") if f.is_file()) / (1024 * 1024)

    # Extrapolate full CSV size from sample ratio
    csv_extrapolated_mb = get_size_mb(csv_path) * (df.height / sample_df.height)
    single_snappy_mb = get_size_mb(single_snappy)
    single_zstd_mb = get_size_mb(single_zstd)
    part_zstd_mb = get_size_mb(part_zstd)

    # Query speed benchmarks
    log.info("Benchmarking query operations...")
    # 1. Full scan (all columns)
    t0 = time.perf_counter()
    _ = pl.read_parquet(single_zstd)
    t_full_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pl.read_parquet(part_zstd)
    t_full_part = time.perf_counter() - t0

    # 2. Projection pushdown (read only 3 columns: CERT, REPDTE, ASSET)
    cols = ["CERT", "REPDTE", "ASSET"]
    t0 = time.perf_counter()
    _ = pl.read_parquet(single_zstd, columns=cols)
    t_proj_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pl.read_parquet(part_zstd, columns=cols)
    t_proj_part = time.perf_counter() - t0

    # 3. Partition pruning (filter 1 year: 2023)
    t0 = time.perf_counter()
    _ = (
        pl.scan_parquet(single_zstd)
        .filter(pl.col("REPDTE").dt.year() == 2023)
        .collect()
    )
    t_prune_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = (
        pl.scan_parquet(part_zstd, hive_partitioning=True)
        .filter(pl.col("YEAR") == 2023)
        .collect()
    )
    t_prune_part = time.perf_counter() - t0

    # Clean up benchmark temp files
    shutil.rmtree(bench_dir, ignore_errors=True)

    results = pl.DataFrame([
        {
            "format": "Raw CSV (Uncompressed)",
            "disk_size_mb": round(csv_extrapolated_mb, 1),
            "full_read_s": None,
            "projection_read_s": None,
            "partition_filter_s": None,
            "pushdown_supported": False,
        },
        {
            "format": "Single Parquet (Snappy)",
            "disk_size_mb": round(single_snappy_mb, 1),
            "full_read_s": None,
            "projection_read_s": None,
            "partition_filter_s": None,
            "pushdown_supported": True,
        },
        {
            "format": "Single Parquet (Zstandard L3)",
            "disk_size_mb": round(single_zstd_mb, 1),
            "full_read_s": round(t_full_single, 3),
            "projection_read_s": round(t_proj_single, 3),
            "partition_filter_s": round(t_prune_single, 3),
            "pushdown_supported": True,
        },
        {
            "format": "Hive Partitioned Parquet (Year, Zstandard L3)",
            "disk_size_mb": round(part_zstd_mb, 1),
            "full_read_s": round(t_full_part, 3),
            "projection_read_s": round(t_proj_part, 3),
            "partition_filter_s": round(t_prune_part, 3),
            "pushdown_supported": True,
        },
    ])
    return results


def plot_storage_comparison(res: pl.DataFrame):
    """Plot storage size comparison chart."""
    df = res.to_pandas()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(df["format"], df["disk_size_mb"], color=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"])
    ax.set_xlabel("Disk Storage Footprint (MB)")
    ax.set_title("Part 1: Storage Layout & Compression Benchmark (1.68M Rows)")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.0f} MB", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center")
    plt.tight_layout()
    out_fig = out_dir("figures") / "part1_storage_comparison.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    log.info(f"Saved storage comparison figure to {out_fig}")


def main():
    log.info("Starting Part 1: Ingestion & Storage Optimization...")
    log.info("Reading raw quarterly files and enforcing canonical schema...")
    df = read_raw_all()
    log.info(f"Loaded {df.height:,} rows, {len(df.columns)} columns across {df['CERT'].n_unique():,} institutions.")

    log.info("Writing Bronze hive-partitioned Parquet (YEAR, zstd)...")
    out_p = write_bronze_financials(df)
    log.info(f"Bronze dataset successfully written to {out_p}")

    log.info("Running storage layout benchmarking...")
    bench_results = run_storage_benchmark(df)
    out_csv = out_dir("tables") / "part1_storage_benchmark.csv"
    bench_results.write_csv(out_csv)
    log.info(f"Saved benchmark results to {out_csv}")
    print(bench_results)

    plot_storage_comparison(bench_results)
    log.info("Part 1 complete.")


if __name__ == "__main__":
    main()
