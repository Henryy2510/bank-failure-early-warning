"""Part 3: Polars (eager / lazy / streaming) vs pandas (naive / pushdown) on identical work, on three panel sizes.

Fairness rules
* every engine runs the same six transformations (core.py) and materialises the full feature table;
* every run is a fresh spawned process (median of REPEAT runs, warm page cache) so memory peaks are not shared;
* all engines return a checksum of their output, and the benchmark fails if the engines disagree.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.bfews.core import CORE_COLS, checksum_pandas, checksum_polars, core_pandas, core_polars, max_rel_diff
from src.bfews.io import bronze_financials_path
from src.bfews.utils import get_logger, measure, out_dir, save_json

log = get_logger("05_benchmark")
REPEAT = 3
COLS = [c for c in CORE_COLS]  # columns the core needs


def bench_polars(bronze_p: Path, min_year: int, mode: str) -> dict:
    if mode == "eager":
        df = pl.read_parquet(bronze_p, columns=COLS + ["YEAR"], hive_partitioning=True).filter(pl.col("YEAR") >= min_year)
        out = core_polars(df.drop("YEAR"))
    else:
        lf = (pl.scan_parquet(bronze_p, hive_partitioning=True).filter(pl.col("YEAR") >= min_year)
              .select(COLS))
        out = core_polars(lf).collect(engine="streaming" if mode == "streaming" else "in-memory")
    return checksum_polars(out)


def bench_pandas(bronze_p: Path, min_year: int, mode: str) -> dict:
    import pandas as pd

    if mode == "naive":  # read everything, filter afterwards (what most notebooks do)
        df = pd.read_parquet(bronze_p)
        df = df[df["YEAR"].astype(int) >= min_year][COLS]
    else:  # pyarrow column projection + partition filter pushed into the reader
        df = pd.read_parquet(bronze_p, columns=COLS, filters=[("YEAR", ">=", min_year)])
    return checksum_pandas(core_pandas(df))


def capture_polars_plan(bronze_p: Path) -> str:
    lf = (pl.scan_parquet(bronze_p, hive_partitioning=True)
          .filter(pl.col("YEAR") >= 2008).select(COLS))
    plan = core_polars(lf)
    text = ["=== UNOPTIMIZED LOGICAL PLAN ===", plan.explain(optimized=False),
            "\n=== OPTIMIZED PLAN ===", plan.explain(optimized=True)]
    (out_dir("tables") / "polars_optimized_plan.txt").write_text("\n".join(text), encoding="utf-8")
    return "\n".join(text)


def run(bronze_p: Path) -> pl.DataFrame:
    slices = [("2010-2026", 2010), ("1995-2026", 1995), ("1984-2026 (full)", 1984)]
    engines = [("Polars lazy", bench_polars, "lazy"), ("Polars streaming", bench_polars, "streaming"),
               ("Polars eager", bench_polars, "eager"), ("pandas + pushdown", bench_pandas, "pushdown"),
               ("pandas naive", bench_pandas, "naive")]
    rows = []
    for label, y in slices:
        ref = None
        for name, fn, mode in engines:
            log.info(f"{label}: {name}")
            m = measure(fn, bronze_p, y, mode, repeat=REPEAT)
            chk = m["result"]
            ref = ref or chk
            diff = max_rel_diff(chk, ref)
            if diff > 1e-6:
                raise RuntimeError(f"engine outputs differ for {name} on {label}: {diff}")
            rows.append({"slice": label, "rows": chk["rows"], "engine": name, "time_s": round(m["median_s"], 3),
                         "peak_rss_mb": round(m["peak_rss_mb"], 0), "max_rel_diff_vs_polars_lazy": float(f"{diff:.1e}")})
    return pl.DataFrame(rows)


def capacity_extrapolation(res: pl.DataFrame, ram_gb: float = 16.0) -> dict:
    """Linear fit of peak RSS and time against rows (3 measured sizes) -> rows at which RAM runs out. An ESTIMATE.

    'pandas naive' is excluded: it always reads the whole file, so its memory does not depend on the slice and a
    fit against the slice size would be meaningless.
    """
    out = {}
    for eng in res["engine"].unique().to_list():
        if eng == "pandas naive":
            continue
        d = res.filter(pl.col("engine") == eng).sort("rows")
        x = d["rows"].to_numpy() / 1e6
        mem_slope, mem_icpt = np.polyfit(x, d["peak_rss_mb"].to_numpy(), 1)
        t_slope, _ = np.polyfit(x, d["time_s"].to_numpy(), 1)
        out[eng] = {"mb_per_million_rows": round(float(mem_slope), 0), "seconds_per_million_rows": round(float(t_slope), 3),
                    "rows_millions_until_16gb": round(float((ram_gb * 1024 - mem_icpt) / mem_slope), 0)}
    return out


def plot(res: pl.DataFrame) -> None:
    d = res.to_pandas()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    sty = {"Polars lazy": ("#2b8cbe", "o"), "Polars streaming": ("#7bccc4", "d"), "Polars eager": ("#5cb85c", "^"),
           "pandas + pushdown": ("#f0ad4e", "s"), "pandas naive": ("#d9534f", "x")}
    for eng, (col, mk) in sty.items():
        s = d[d["engine"] == eng]
        a1.plot(s["rows"] / 1e6, s["time_s"], marker=mk, color=col, lw=2, label=eng)
        a2.plot(s["rows"] / 1e6, s["peak_rss_mb"], marker=mk, color=col, lw=2, label=eng)
    a1.set_xlabel("panel rows (millions)"); a1.set_ylabel("seconds (median of 3)"); a1.set_title("Time")
    a2.set_xlabel("panel rows (millions)"); a2.set_ylabel("peak RSS (MB)"); a2.set_title("Peak memory")
    for a in (a1, a2):
        a.grid(True, alpha=0.3)
        a.legend()
    plt.tight_layout()
    p = out_dir("figures") / "part3_scaling_benchmark.png"
    plt.savefig(p, dpi=200)
    plt.close()
    log.info(f"Saved {p}")


def main() -> None:
    bronze_p = bronze_financials_path()
    capture_polars_plan(bronze_p)
    res = run(bronze_p)
    print(res)
    res.write_csv(out_dir("tables") / "part3_benchmark_results.csv")
    cap = capacity_extrapolation(res)
    save_json(cap, out_dir("tables") / "part3_capacity_extrapolation.json")
    log.info(f"Capacity extrapolation (estimate): {cap}")
    plot(res)


if __name__ == "__main__":
    main()
