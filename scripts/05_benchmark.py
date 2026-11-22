"""Part 3: Multi-scale benchmark (1x..8x) comparing Polars (eager / lazy / streaming) vs pandas on identical work.

Fairness rules
* every engine runs the exact same six transformations (core.py) and materialises the full feature table;
* every run is a fresh spawned process (median of REPEAT runs, warm page cache) so peak memory peaks are isolated;
* every run verifies numeric parity against Polars lazy (max relative checksum difference <= 1e-6).
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ["POLARS_MAX_THREADS"] = "4"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import psutil

from src.bfews.core import CORE_COLS, checksum_pandas, checksum_polars, core_pandas, core_polars, max_rel_diff
from src.bfews.io import bronze_financials_path
from src.bfews.utils import available_memory_gb, data_dir, get_logger, measure, out_dir, save_json

log = get_logger("05_benchmark")
REPEAT = 3
COLS = list(CORE_COLS)


def bench_polars(src_p: Path, mode: str) -> dict:
    if mode == "eager":
        df = pl.read_parquet(src_p, columns=COLS, hive_partitioning=True)
        out = core_polars(df)
    else:
        lf = pl.scan_parquet(src_p, hive_partitioning=True).select(COLS)
        out = core_polars(lf).collect(engine="streaming" if mode == "streaming" else "in-memory")
    return checksum_polars(out)


def bench_pandas(src_p: Path, mode: str) -> dict:
    import pandas as pd_

    if mode == "naive":
        df = pd_.read_parquet(src_p)[COLS]
    else:
        df = pd_.read_parquet(src_p, columns=COLS)
    return checksum_pandas(core_pandas(df))


def build_scaled_parquet(bronze_p: Path, target_p: Path, scale: int, cols: list[str] | None = None) -> Path:
    """Create a year-partitioned k-times scaled panel with disjoint synthetic CERT offsets in low memory."""
    if scale == 1:
        return bronze_p
    shutil.rmtree(target_p, ignore_errors=True)
    target_p.mkdir(parents=True, exist_ok=True)
    read_cols = COLS if cols is None else cols
    for year_dir in sorted(bronze_p.glob("YEAR=*")):
        df_y = pl.read_parquet(year_dir / "part-0.parquet", columns=read_cols) if read_cols else pl.read_parquet(year_dir / "part-0.parquet")
        parts = [df_y.with_columns((pl.col("CERT") + r * 1_000_000).alias("CERT")) for r in range(scale)]
        out_y = target_p / year_dir.name
        out_y.mkdir(parents=True, exist_ok=True)
        pl.concat(parts, how="vertical").sort(["REPDTE", "CERT"]).write_parquet(
            out_y / "part-0.parquet", compression="zstd", compression_level=3, statistics=True, row_group_size=50_000
        )
    return target_p


def bench_full_schema_polars(src_p: Path, mode: str) -> dict:
    """End-to-end Bronze-to-Gold full-schema workload (all 34 cols -> 7 YTD flows -> 26 CAMELS + 4Q lags = 67 cols)."""
    from src.bfews.features import compute_camels_features
    from src.bfews.panel import decumulate_ytd_flows

    lf = pl.scan_parquet(src_p, hive_partitioning=True)
    lf = decumulate_ytd_flows(lf).drop(["year", "quarter", "qidx"])
    lf = compute_camels_features(lf)
    if mode == "collect":
        out = lf.collect(engine="streaming")
        return {"rows": int(out.height), "cols": len(out.columns)}
    with tempfile.TemporaryDirectory(prefix="bfews_sink_") as td:
        out_p = Path(td) / "gold_stream.parquet"
        lf.sink_parquet(out_p, compression="zstd", compression_level=3, engine="streaming")
        sch = pl.read_parquet_schema(out_p)
        n_rows = int(pl.scan_parquet(out_p).select(pl.len()).collect().item())
        return {"rows": n_rows, "cols": len(sch)}


def run_full_schema_benchmark(bronze_p: Path) -> dict:
    """Measure isolated 1x and 2x peak RSS & time on the full 34-col Bronze -> 67-col Gold pipeline (collect vs sink)."""
    tmp_root = data_dir("bronze") / "_scale_full_tmp"
    shutil.rmtree(tmp_root, ignore_errors=True)
    results = []
    try:
        for scale in (1, 2):
            src_p = bronze_p if scale == 1 else build_scaled_parquet(bronze_p, tmp_root / f"s{scale}", scale, cols=[])
            for label, mode in (
                ("Polars streaming (collect)", "collect"),
                ("Polars streaming (sink_parquet)", "sink"),
            ):
                log.info(f"Full-schema {scale}x: {label}")
                m = measure(bench_full_schema_polars, src_p, mode, repeat=REPEAT)
                results.append({
                    "scale": f"{scale}x",
                    "scale_factor": scale,
                    "rows": int(m["result"]["rows"]),
                    "cols": int(m["result"]["cols"]),
                    "mode": label,
                    "time_s": round(float(m["median_s"]), 3),
                    "peak_rss_mb": round(float(m["peak_rss_mb"]), 1),
                })
            if scale > 1:
                shutil.rmtree(tmp_root / f"s{scale}", ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    extrap = {}
    for label in ("Polars streaming (collect)", "Polars streaming (sink_parquet)"):
        pts = sorted([r for r in results if r["mode"] == label], key=lambda r: r["rows"])
        x = np.array([p["rows"] / 1e6 for p in pts], dtype=float)
        y_mem = np.array([p["peak_rss_mb"] for p in pts], dtype=float)
        y_t = np.array([p["time_s"] for p in pts], dtype=float)
        slope = float((y_mem[1] - y_mem[0]) / (x[1] - x[0]))
        icpt = float(y_mem[0] - slope * x[0])
        t_slope = float((y_t[1] - y_t[0]) / (x[1] - x[0]))
        extrap[label] = {
            "fit_points": "two-point (1x, 2x) linear estimate",
            "intercept_mb": round(icpt, 1),
            "mb_per_million_rows": round(slope, 1),
            "seconds_per_million_rows": round(t_slope, 3),
            "rows_millions_in_12gb_usable": round(float((12.0 * 1024 - icpt) / max(1.0, slope)), 1),
            "rows_millions_in_16gb_raw": round(float((16.0 * 1024 - icpt) / max(1.0, slope)), 1),
        }

    payload = {
        "workload": "End-to-end Bronze 34-col Hive scan -> sort(CERT, REPDTE) -> 7 YTD flows -> 26 CAMELS + 4Q lags (67 output cols)",
        "repeat": REPEAT,
        "results": results,
        "extrapolation": extrap,
    }
    save_json(payload, out_dir("tables") / "part3_full_schema_benchmark.json")
    log.info(f"Saved full-schema benchmark to outputs/tables/part3_full_schema_benchmark.json: {extrap}")
    return payload


def capture_polars_plan(bronze_p: Path) -> str:
    """Compare naive unprojected scan vs optimized projection+partition-pruned query plan."""
    naive_lf = core_polars(pl.scan_parquet(bronze_p, hive_partitioning=True)).filter(pl.col("year") >= 2008)
    opt_lf = core_polars(
        pl.scan_parquet(bronze_p, hive_partitioning=True).filter(pl.col("YEAR") >= 2008).select(COLS)
    )
    txt = "\n".join([
        "=== NAIVE LOGICAL PLAN (UNPROJECTED, POST-CORE FILTER ON COMPUTED `year`) ===",
        naive_lf.explain(optimized=False),
        "\n=== OPTIMIZED POLARS LAZY PLAN (PARTITION PRUNING YEAR>=2008 + PROJECT 8/34) ===",
        opt_lf.explain(optimized=True),
    ])
    (out_dir("tables") / "polars_naive_vs_optimized_plan.txt").write_text(txt, encoding="utf-8")
    (out_dir("tables") / "polars_optimized_plan.txt").write_text(txt, encoding="utf-8")
    return txt


def save_benchmark_environment() -> dict:
    env = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "ram_available_gb_at_start": round(available_memory_gb(), 2),
        "polars": pl.__version__,
        "pandas": pd.__version__,
        "polars_max_threads": os.environ.get("POLARS_MAX_THREADS", "default"),
    }
    save_json(env, out_dir("tables") / "benchmark_environment.json")
    return env


def run(bronze_p: Path) -> pl.DataFrame:
    tmp_root = data_dir("bronze") / "_scale_tmp"
    shutil.rmtree(tmp_root, ignore_errors=True)
    rows = []

    schedule = [
        (1, [("Polars lazy", bench_polars, "lazy"), ("Polars streaming", bench_polars, "streaming"),
             ("Polars eager", bench_polars, "eager"), ("pandas + pushdown", bench_pandas, "pushdown"),
             ("pandas naive", bench_pandas, "naive")]),
        (2, [("Polars lazy", bench_polars, "lazy"), ("Polars streaming", bench_polars, "streaming"),
             ("Polars eager", bench_polars, "eager"), ("pandas + pushdown", bench_pandas, "pushdown")]),
        (4, [("Polars lazy", bench_polars, "lazy"), ("Polars streaming", bench_polars, "streaming"),
             ("Polars eager", bench_polars, "eager"), ("pandas + pushdown", bench_pandas, "pushdown")]),
        (8, [("Polars lazy", bench_polars, "lazy"), ("Polars streaming", bench_polars, "streaming")]),
    ]

    try:
        for scale, eng_list in schedule:
            src_p = bronze_p if scale == 1 else build_scaled_parquet(bronze_p, tmp_root / f"s{scale}", scale)
            ref = None
            for name, fn, mode in eng_list:
                log.info(f"Scale {scale}x: {name}")
                m = measure(fn, src_p, mode, repeat=REPEAT)
                chk = m["result"]
                ref = ref or chk
                diff = max_rel_diff(chk, ref)
                if diff > 1e-6:
                    raise RuntimeError(f"engine outputs differ for {name} on {scale}x: {diff}")
                rows.append({
                    "scale": f"{scale}x",
                    "scale_factor": scale,
                    "rows": chk["rows"],
                    "engine": name,
                    "time_s": round(m["median_s"], 3),
                    "peak_rss_mb": round(m["peak_rss_mb"], 0),
                    "max_rel_diff_vs_polars_lazy": float(f"{diff:.1e}"),
                })
            if scale > 1:
                shutil.rmtree(tmp_root / f"s{scale}", ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return pl.DataFrame(rows)


def capacity_extrapolation(res: pl.DataFrame, ram_gb: float = 16.0) -> dict:
    """Linear fit of peak RSS and time against multi-scale measured rows (1x..8x) -> laptop 16GB RAM limit."""
    out = {}
    for eng in res["engine"].unique().to_list():
        d = res.filter(pl.col("engine") == eng).sort("rows")
        if d.height < 2:
            continue
        x = d["rows"].to_numpy() / 1e6
        mem_slope, mem_icpt = np.polyfit(x, d["peak_rss_mb"].to_numpy(), 1)
        t_slope, _ = np.polyfit(x, d["time_s"].to_numpy(), 1)
        out[eng] = {
            "measured_max_scale": d["scale"][-1],
            "mb_per_million_rows": round(float(mem_slope), 1),
            "seconds_per_million_rows": round(float(t_slope), 3),
            "rows_millions_until_16gb": round(float((ram_gb * 1024 - mem_icpt) / max(1.0, mem_slope)), 1),
        }
    return out


def plot(res: pl.DataFrame, spark_df: pl.DataFrame | None = None) -> None:
    d = res.to_pandas()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    sty = {
        "Polars lazy": ("#2b8cbe", "o", "-"),
        "Polars streaming": ("#1b9e77", "d", "-"),
        "Polars eager": ("#5cb85c", "^", "--"),
        "pandas + pushdown": ("#f0ad4e", "s", "-"),
        "pandas naive": ("#d9534f", "x", ":"),
    }
    for eng, (col, mk, ls) in sty.items():
        s = d[d["engine"] == eng].sort_values("rows")
        if s.empty:
            continue
        a1.plot(s["rows"] / 1e6, s["time_s"], marker=mk, linestyle=ls, color=col, lw=2, label=eng)
        a2.plot(s["rows"] / 1e6, s["peak_rss_mb"], marker=mk, linestyle=ls, color=col, lw=2, label=eng)
    if spark_df is not None and not spark_df.is_empty():
        sd_all = spark_df.to_pandas()
        sp_sty = {
            "PySpark local[4]": ("#756bb1", "*", "-.", 9),
            "PySpark local[2]": ("#9e9ac8", "P", ":", 7),
        }
        for sp_eng in sd_all["engine"].unique():
            sd = sd_all[sd_all["engine"] == sp_eng].sort_values("rows")
            col, mk, ls, ms = sp_sty.get(sp_eng, ("#756bb1", "*", "-.", 8))
            a1.plot(sd["rows"] / 1e6, sd["total_s"], marker=mk, markersize=ms, linestyle=ls, color=col, lw=2,
                    label=f"{sp_eng} (startup + compute)")
            a2.plot(sd["rows"] / 1e6, sd["peak_rss_tree_mb"], marker=mk, markersize=ms, linestyle=ls, color=col, lw=2,
                    label=f"{sp_eng} (Python + JVM RSS)")
    a1.set_xlabel("Panel Rows (Millions, Synthetic Replication 1x–8x)")
    a1.set_ylabel("Wall-Clock Time (seconds, median)")
    a1.set_title("Multi-Scale Execution Time Scaling")
    a2.set_xlabel("Panel Rows (Millions, Synthetic Replication 1x–8x)")
    a2.set_ylabel("Peak RSS Memory (MB)")
    a2.set_title("Multi-Scale Peak Process-Tree Memory")
    for a in (a1, a2):
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    p = out_dir("figures") / "part3_scaling_benchmark.png"
    plt.savefig(p, dpi=200)
    plt.close()
    log.info(f"Saved {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 3 multi-scale Polars vs pandas benchmark.")
    parser.add_argument("--full-schema-only", action="store_true", help="Run only the reproducible 1x/2x full-schema benchmark.")
    args = parser.parse_args()

    bronze_p = bronze_financials_path()
    if args.full_schema_only:
        run_full_schema_benchmark(bronze_p)
        return
    save_benchmark_environment()
    capture_polars_plan(bronze_p)
    res = run(bronze_p)
    print(res)
    res.write_csv(out_dir("tables") / "part3_benchmark_results.csv")
    cap = capacity_extrapolation(res)
    save_json(cap, out_dir("tables") / "part3_capacity_extrapolation.json")
    log.info(f"Capacity extrapolation (estimate): {cap}")
    run_full_schema_benchmark(bronze_p)
    plot(res)


if __name__ == "__main__":
    main()
