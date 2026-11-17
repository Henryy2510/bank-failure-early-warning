"""Part 1: build the Bronze layer and measure storage layouts (every number comes from a real measurement)."""
from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import matplotlib.pyplot as plt
import polars as pl

from src.bfews.io import bronze_financials_path, read_raw_all, write_bronze_financials
from src.bfews.utils import data_dir, get_logger, measure, out_dir

log = get_logger("01_storage")
PROJ = ["CERT", "REPDTE", "ASSET"]
REPEAT = 3


def _mb(p: Path) -> float:
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1024**2


# ---- read workloads (module level so they can run in spawned processes) --------------------------------------
def csv_full(p): return pl.read_csv(p, try_parse_dates=True)
def csv_proj(p): return pl.read_csv(p, columns=PROJ, try_parse_dates=True)
def csv_year(p): return pl.read_csv(p, try_parse_dates=True).filter(pl.col("REPDTE").dt.year() == 2023)
def pd_csv_full(p):
    import pandas as pd
    return pd.read_csv(p)
def pq_full(p): return pl.read_parquet(p)
def pq_proj(p): return pl.read_parquet(p, columns=PROJ)
def pq_year(p): return pl.scan_parquet(p).filter(pl.col("REPDTE").is_between(date(2023, 1, 1), date(2023, 12, 31))).collect()
def pd_pq_full(p):
    import pandas as pd
    return pd.read_parquet(p)
def hive_full(p): return pl.read_parquet(p, hive_partitioning=True)
def hive_proj(p): return pl.read_parquet(p, columns=PROJ, hive_partitioning=True)
def hive_year(p): return pl.scan_parquet(p, hive_partitioning=True).filter(pl.col("YEAR") == 2023).collect()


def run_storage_benchmark(df: pl.DataFrame) -> pl.DataFrame:
    tmp = data_dir("bronze") / "_bench"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    d = df.sort(["REPDTE", "CERT"])
    csv_p, snappy_p, zstd_p, hive_tmp = tmp / "fin.csv", tmp / "fin_snappy.parquet", tmp / "fin_zstd.parquet", tmp / "fin_hive"
    log.info("Writing the CSV, single-file Parquet, and Hive Parquet variants for measurement...")
    t0 = time.perf_counter()
    d.write_csv(csv_p)
    w_csv = time.perf_counter() - t0
    t0 = time.perf_counter()
    d.write_parquet(snappy_p, compression="snappy", row_group_size=50_000, statistics=True)
    w_snappy = time.perf_counter() - t0
    t0 = time.perf_counter()
    d.write_parquet(zstd_p, compression="zstd", compression_level=3, row_group_size=50_000, statistics=True)
    w_zstd = time.perf_counter() - t0
    t0 = time.perf_counter()
    for yr in sorted(d["REPDTE"].dt.year().unique().to_list()):
        yp = hive_tmp / f"YEAR={yr}"
        yp.mkdir(parents=True, exist_ok=True)
        d.filter(pl.col("REPDTE").dt.year() == yr).write_parquet(
            yp / "part-0.parquet", compression="zstd", compression_level=3, statistics=True, row_group_size=50_000
        )
    w_hive = time.perf_counter() - t0
    hive_p = bronze_financials_path()

    plan = [
        ("CSV (uncompressed)", csv_p, w_csv, csv_full, csv_proj, csv_year),
        ("Parquet single file, Snappy", snappy_p, w_snappy, pq_full, pq_proj, pq_year),
        ("Parquet single file, Zstd-3", zstd_p, w_zstd, pq_full, pq_proj, pq_year),
        ("Parquet hive YEAR=, Zstd-3 (chosen)", hive_p, w_hive, hive_full, hive_proj, hive_year),
    ]
    rows = []
    for name, path, w_s, f_full, f_proj, f_year in plan:
        log.info(f"Measuring {name} (median of {REPEAT} runs, fresh process, warm page cache)...")
        rows.append({
            "format": name, "size_mb": round(_mb(path), 1), "write_s": round(w_s, 3),
            "full_read_s": round(measure(f_full, path, repeat=REPEAT)["median_s"], 3),
            "read_3_cols_s": round(measure(f_proj, path, repeat=REPEAT)["median_s"], 3),
            "filter_2023_s": round(measure(f_year, path, repeat=REPEAT)["median_s"], 3),
        })
    # pandas readers on the same files (the usual baseline)
    rows.append({"format": "pandas.read_csv (CSV)", "size_mb": rows[0]["size_mb"], "write_s": None,
                 "full_read_s": round(measure(pd_csv_full, csv_p, repeat=REPEAT)["median_s"], 3),
                 "read_3_cols_s": None, "filter_2023_s": None})
    rows.append({"format": "pandas.read_parquet (single Zstd-3)", "size_mb": rows[2]["size_mb"], "write_s": None,
                 "full_read_s": round(measure(pd_pq_full, zstd_p, repeat=REPEAT)["median_s"], 3),
                 "read_3_cols_s": None, "filter_2023_s": None})
    shutil.rmtree(tmp, ignore_errors=True)
    res = pl.DataFrame(rows).with_columns((pl.col("size_mb").first() / pl.col("size_mb")).round(2).alias("compression_vs_csv"))
    return res


def run_float32_precision_check(df: pl.DataFrame) -> pl.DataFrame:
    """Quantify float32 rounding error vs float64 canonical storage (justifies Float64 for USD stock amounts)."""
    cols = ["ASSET", "DEP", "EQ", "NETINC", "ROA", "ROE", "RBC1AAJ", "RBCRWAJ"]
    rows = []
    for c in cols:
        s = df[c].drop_nulls()
        s64 = s.cast(pl.Float64)
        s32 = s.cast(pl.Float32).cast(pl.Float64)
        err = (s64 - s32).abs()
        rel = err / s64.abs().clip(lower_bound=1.0)
        gt_2p24 = int((s64.abs() > float(2**24)).sum())
        rows.append({
            "column": c,
            "max_abs_value": round(float(s64.abs().max()), 2),
            "rows_exceeding_float32_exact_int_limit": gt_2p24,
            "max_abs_error": round(float(err.max()), 4),
            "max_rel_error": float(f"{float(rel.max()):.3e}"),
        })
    return pl.DataFrame(rows)


def plot_storage(res: pl.DataFrame) -> None:
    d = res.filter(pl.col("format").str.starts_with("pandas").not_()).to_pandas()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.barh(d["format"], d["size_mb"], color=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"])
    a1.set_xlabel("size on disk (MB)")
    a1.set_title("Storage footprint (1.68M rows)")
    for i, v in enumerate(d["size_mb"]):
        a1.text(v, i, f" {v:.0f}", va="center")
    w = 0.27
    x = range(len(d))
    a2.bar([i - w for i in x], d["full_read_s"], w, label="read all columns")
    a2.bar(list(x), d["read_3_cols_s"], w, label="read 3 columns")
    a2.bar([i + w for i in x], d["filter_2023_s"], w, label="filter year 2023")
    a2.set_xticks(list(x))
    a2.set_xticklabels(["CSV", "Parquet\nSnappy", "Parquet\nZstd", "Hive\nZstd"])
    a2.set_ylabel("seconds (median of 3)")
    a2.set_title("Read time")
    a2.legend()
    plt.tight_layout()
    p = out_dir("figures") / "part1_storage_comparison.png"
    plt.savefig(p, dpi=200)
    plt.close()
    log.info(f"Saved {p}")


def main() -> None:
    log.info("Part 1: Bronze layer and storage benchmark")
    df = read_raw_all()
    log.info(f"Loaded {df.height:,} rows, {len(df.columns)} columns, {df['CERT'].n_unique():,} banks.")
    write_bronze_financials(df)
    log.info(f"Bronze written to {bronze_financials_path()}")
    res = run_storage_benchmark(df)
    res.write_csv(out_dir("tables") / "part1_storage_benchmark.csv")
    print(res)
    f32 = run_float32_precision_check(df)
    f32.write_csv(out_dir("tables") / "part1_float32_check.csv")
    log.info(f"Float32 quantization check:\n{f32}")
    plot_storage(res)


if __name__ == "__main__":
    main()
