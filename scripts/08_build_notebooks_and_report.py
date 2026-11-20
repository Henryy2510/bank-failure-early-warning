"""Generate executed Jupyter notebooks and the complete English Report (Markdown + DOCX + LaTeX PDF)."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import subprocess
import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import polars as pl
import yaml

from src.bfews.utils import ROOT, get_logger, out_dir

log = get_logger("08_deliverables")


def load_student_info() -> dict[str, str]:
    info_file = ROOT / "report" / "student_info.yaml"
    if info_file.exists():
        try:
            return yaml.safe_load(info_file.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {
        "student_name": "[YOUR FULL NAME]",
        "student_id": "[YOUR STUDENT ID]",
        "github_url": "[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME]",
    }


def execute_and_save_nb(cells_spec: list[tuple[str, str]], out_path: Path) -> None:
    """Create a v4 Jupyter Notebook, execute every code cell via nbclient, and write to disk."""
    nb = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split()[0],
            },
        }
    )
    for kind, src in cells_spec:
        if kind == "md":
            nb.cells.append(new_markdown_cell(src))
        else:
            nb.cells.append(new_code_cell(src))

    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(out_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, out_path)
    log.info(f"Executed and saved notebook: {out_path.name}")


def build_notebooks():
    """Generate and execute 5 structured, self-contained notebooks in final/notebooks/."""
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    # 1. Part 1 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 1: FDIC Data Acquisition, Verification & Lakehouse Storage Architecture\n"
                "This notebook verifies the reproducible acquisition of 170 quarters (1984Q1–2026Q2) of FDIC Call Report data, "
                "audits manifest completeness against API totals, and benchmarks our Hive-partitioned Zstandard Parquet Bronze layer against raw CSV and unpartitioned layouts.",
            ),
            (
                "code",
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n"
                "import polars as pl\nfrom src.bfews.quality import audit_manifest\n\n"
                "manifest_check = audit_manifest()\nprint('Manifest Audit Summary:', manifest_check)",
            ),
            (
                "code",
                "bench_df = pl.read_csv('../outputs/tables/part1_storage_benchmark.csv')\n"
                "print(bench_df)\n"
                "f32_df = pl.read_csv('../outputs/tables/part1_float32_check.csv')\n"
                "print('\\n=== Float32 Precision Loss Check (Justifies Float64) ===')\nprint(f32_df)\n"
                "bronze_files = list(Path('../data/bronze/financials').rglob('*.parquet'))\n"
                "print(f'Bronze partition count: {len(bronze_files)} files across {len({p.parent for p in bronze_files})} year folders')",
            ),
        ],
        nb_dir / "01_acquisition_and_storage.ipynb",
    )

    # 2. Part 2 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 2: Automated Data Quality Audit & Point-in-Time Panel Construction\n"
                "Audits cross-quarter schema drift, structural missingness by regulatory era, balance-sheet invariants, "
                "YTD flow de-cumulation, and bank exit classification (FDIC failure vs. non-failure mergers).",
            ),
            (
                "code",
                "from pathlib import Path\nimport polars as pl\n\n"
                "dq_text = Path('../outputs/dq_report/data_quality_report.md').read_text()\n"
                "print(dq_text[:2500] + '\\n...[see outputs/dq_report/data_quality_report.md for full report]...')",
            ),
            (
                "code",
                "exit_df = pl.read_csv('../outputs/tables/bank_exit_summary.csv')\n"
                "print('=== Bank Exit Taxonomy ===')\nprint(exit_df)\n"
                "print('\\n=== Zero-as-Missing by Era (incl. 2020+ CBLR) ===')\n"
                "print(pl.read_csv('../outputs/tables/dq_zero_as_missing.csv'))\n"
                "print('\\n=== Empirical YTD Monotonicity ===')\n"
                "print(pl.read_csv('../outputs/tables/dq_ytd_monotonicity.csv'))\n"
                "print('\\n=== Top FDIC CHANGEC1 Exit Codes ===')\n"
                "print(pl.read_csv('../outputs/tables/exit_changec1_codes.csv').head(8))\n"
                "panel_schema = pl.read_parquet_schema('../data/silver/panel.parquet')\n"
                "print(f'Silver panel columns ({len(panel_schema)}):', list(panel_schema.keys())[:15], '...')",
            ),
        ],
        nb_dir / "02_data_quality_and_panel.ipynb",
    )

    # 3. Part 3 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 3: Scalable Processing — Polars Query Optimization, Pandas Benchmark & PySpark Parity\n"
                "Examines Polars lazy query optimization (predicate pushdown, projection pushdown, parallel window execution), "
                "benchmarks execution time and peak RSS against Pandas (naive vs column-pushdown), and verifies exact parity with PySpark.",
            ),
            (
                "code",
                "from pathlib import Path\nimport polars as pl\n\n"
                "print('=== Naive vs Optimized Polars Query Plan ===')\n"
                "print(Path('../outputs/tables/polars_naive_vs_optimized_plan.txt').read_text())",
            ),
            (
                "code",
                "bench = pl.read_csv('../outputs/tables/part3_benchmark_results.csv')\n"
                "print('=== Part 3 Multi-Scale (1x-8x) Benchmark ===')\nprint(bench)\n"
                "print('\\n=== PySpark Multi-Scale (1x & 4x) Python+JVM RSS ===')\n"
                "print(pl.read_csv('../outputs/tables/spark_scaling_results.csv'))",
            ),
            (
                "code",
                "import json\n"
                "parity = json.loads(Path('../outputs/tables/spark_polars_parity.json').read_text())\n"
                "print('=== PySpark vs Polars Parity Check ===')\n"
                "print(json.dumps(parity, indent=2))\n"
                "print('\\n=== PySpark Catalyst Execution Plan (Excerpt) ===')\n"
                "print(Path('../outputs/tables/spark_execution_plan.txt').read_text()[:1500])",
            ),
        ],
        nb_dir / "03_scalable_processing_polars_spark.ipynb",
    )

    # 4. Part 4 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 4: Leak-Free CAMELS Feature Engineering & Forward Failure Labeling\n"
                "Builds 26 point-in-time CAMELS, Texas Ratio, and 4-quarter historical lag features joined with the 4-quarter forward failure indicator $y_{i,q} = 1\\{\\text{fail in } (q, q+4]\\}$, and verifies the causal leak-free truncation property.",
            ),
            (
                "code",
                "import sys, datetime\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n"
                "import numpy as np\nimport polars as pl\nfrom src.bfews.features import FEATURE_COLS_CAMELS, compute_camels_features\n\n"
                "print(f'Total CAMELS features ({len(FEATURE_COLS_CAMELS)}):', FEATURE_COLS_CAMELS)\n"
                "print(pl.read_csv('../outputs/tables/gold_split_summary.csv'))",
            ),
            (
                "code",
                "# Live verification of leak-free causal truncation property\n"
                "dates = [datetime.date(2020, 3, 31), datetime.date(2020, 6, 30), datetime.date(2020, 9, 30), datetime.date(2020, 12, 31),\n"
                "         datetime.date(2021, 3, 31), datetime.date(2021, 6, 30), datetime.date(2021, 9, 30), datetime.date(2021, 12, 31)]\n"
                "n = len(dates)\n"
                "sample_df = pl.DataFrame({\n"
                "    'CERT': [101]*n, 'REPDTE': dates, 'ASSET': [1000.0 + i*50 for i in range(n)],\n"
                "    'EQ': [100.0 + i*5 for i in range(n)], 'LNLSGR': [600.0 + i*30 for i in range(n)],\n"
                "    'LNLSNET': [580.0 + i*30 for i in range(n)], 'NCLNLS': [10.0 + i for i in range(n)],\n"
                "    'ORE': [2.0]*n, 'LNATRES': [20.0]*n, 'DEP': [800.0]*n, 'CHBAL': [50.0]*n, 'SC': [100.0]*n,\n"
                "    'LNRE': [300.0]*n, 'LNCI': [150.0]*n, 'LNCON': [150.0]*n, 'ROA': [1.0]*n, 'ROE': [10.0]*n,\n"
                "    'NIMY': [3.5]*n, 'RBC1AAJ': [12.0]*n, 'RBCRWAJ': [14.0]*n, 'NONIX': [30.0]*n,\n"
                "    'INTINC': [40.0]*n, 'EINTEXP': [10.0]*n, 'NONII': [5.0]*n, 'ELNATR_Q': [2.0]*n,\n"
                "    'NETINC_Q': [10.0]*n, 'NAASSET': [5.0]*n, 'BRO': [10.0]*n\n"
                "})\n"
                "f_full = compute_camels_features(sample_df).filter(pl.col('REPDTE') == datetime.date(2020, 12, 31))\n"
                "f_trunc = compute_camels_features(sample_df.slice(0, 4)).filter(pl.col('REPDTE') == datetime.date(2020, 12, 31))\n"
                "diffs = {c: abs((f_full[c].item() or 0.0) - (f_trunc[c].item() or 0.0)) for c in ['eq_asset_ratio', 'texas_ratio', 'nclnls_ratio']}\n"
                "print('Max abs diff at 2020-12-31 between full (2020-2021) and truncated (2020 only) panel:', diffs)\n"
                "assert max(diffs.values()) == 0.0, 'Future leakage detected!'",
            ),
        ],
        nb_dir / "04_feature_pipeline.ipynb",
    )

    # 5. Part 5 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 5: Expanding Walk-Forward Validation, Prior-Odds Calibration, Operating Points & Held-Out Test Evaluation\n"
                "Compares supervisory baselines against regularized Logistic Regression, HistGradientBoosting, and PySpark MLlib, "
                "presents fixed-budget supervisory capacity operating points (Top 1%, 2%, 5% per quarter), cost-minimizing threshold $\\theta^*$, empirical leakage parameter shift, and a live operational 2026Q2 watchlist.",
            ),
            (
                "code",
                "import json\nfrom pathlib import Path\nimport polars as pl\n\n"
                "print('=== Walk-Forward Cross-Validation Comparison (DEV 1984-2007) ===')\n"
                "print(pl.read_csv('../outputs/tables/cv_model_comparison.csv'))\n"
                "print('\\n=== DEV Label Sensitivity (FAILURE+ASSISTANCE vs FAILURE-only) ===')\n"
                "print(pl.read_csv('../outputs/tables/dev_label_sensitivity.csv'))\n"
                "print('\\n=== Empirical Leakage Parameter Shift (DEV-only vs Naive Full) Top 8 ===')\n"
                "print(pl.read_csv('../outputs/tables/leak_parameter_shift.csv').head(8))",
            ),
            (
                "code",
                "res = json.loads(Path('../outputs/tables/test_results.json').read_text())\n"
                "cost_cfg = res['cost_threshold_tuning']\n"
                "t_eval = res['held_out_test_evaluation']\n"
                "print('=== DEV Cost-Optimal Threshold & Failing Bank Counts ===')\n"
                "print({\n"
                "    'theta_star': cost_cfg['theta'],\n"
                "    'theta_at_grid_edge': cost_cfg['theta_at_grid_edge'],\n"
                "    'dev_failing_banks_in_oof_blocks': cost_cfg['dev_failing_banks_in_oof_blocks'],\n"
                "    'dev_oof_banks_caught_at_theta': cost_cfg['dev_oof_banks_caught_at_theta'],\n"
                "    'dev_failing_banks_all_1984_2007': cost_cfg['dev_failing_banks_all'],\n"
                "})\n"
                "print('\\n=== Held-Out Test Overall Metrics (2009Q1-2025Q2) ===')\n"
                "print({k: t_eval[k] for k in ['pr_auc', 'roc_auc', 'recall_top_1pct', 'recall_top_5pct', 'brier_score', 'rows', 'positive_rows', 'failing_banks_total']})\n"
                "print('\\n=== Held-Out Test Capacity Operating Points ===')\n"
                "print(pl.DataFrame(t_eval['capacity_points']))\n"
                "print('\\n=== Held-Out Test at Cost-Optimal Threshold ===')\n"
                "print(t_eval['at_threshold'])\n"
                "print('\\n=== Top 10 Strictly-DEV Feature Importance (HGB Permutation + Logit Coef) ===')\n"
                "print(pl.read_csv('../outputs/tables/feature_importance.csv').head(10))\n"
                "print('\\n=== Post-hoc Historical Canary Case Studies (Held-Out TEST Failures) ===')\n"
                "print(pl.read_csv('../outputs/tables/canary_case_studies.csv'))\n"
                "print('\\n=== Operational 2026Q2 Top-10 Watchlist (Frozen DEV Model, Unlabelled Tail) ===')\n"
                "print(pl.read_csv('../outputs/tables/watchlist_2026q2.csv').head(10))",
            ),
        ],
        nb_dir / "05_modelling_and_validation.ipynb",
    )
    log.info(f"Generated and executed 5 notebooks in {nb_dir}")


def _replace_sync_block(text: str, key: str, body: str) -> str:
    start_tag = f"<!-- SYNC:START:{key} -->"
    end_tag = f"<!-- SYNC:END:{key} -->"
    pattern = re.compile(rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}", flags=re.DOTALL)
    replacement = f"{start_tag}\n{body.strip()}\n{end_tag}"
    return pattern.sub(lambda _m: replacement, text, count=1)


def _format_latex_sci(val: float) -> str:
    """Format a small float cleanly as LaTeX scientific notation (e.g., 4.9 \\times 10^{-16})."""
    if val == 0.0:
        return "0.0"
    mant_str, exp_str = f"{val:.1e}".split("e")
    return f"{mant_str} \\times 10^{{{int(exp_str)}}}"


def sync_benchmarks_into_docs() -> None:
    """Keep report.md and README.md wall-clock benchmark cells strictly aligned with latest measured CSV/JSON outputs."""
    tdir = out_dir("tables")
    p1 = {r["format"]: r for r in pl.read_csv(tdir / "part1_storage_benchmark.csv").iter_rows(named=True)}
    p3 = {(r["scale"], r["engine"]): r for r in pl.read_csv(tdir / "part3_benchmark_results.csv").iter_rows(named=True)}
    cap = json.loads((tdir / "part3_capacity_extrapolation.json").read_text(encoding="utf-8"))
    sp = {(r["scale"], r["engine"]): r for r in pl.read_csv(tdir / "spark_scaling_results.csv").iter_rows(named=True)}
    parity = json.loads((tdir / "spark_polars_parity.json").read_text(encoding="utf-8"))
    f32 = {r["column"]: r for r in pl.read_csv(tdir / "part1_float32_check.csv").iter_rows(named=True)}
    dq = json.loads((tdir / "dq_summary.json").read_text(encoding="utf-8"))
    ytd = {r["quarter"]: r for r in pl.read_csv(tdir / "dq_ytd_monotonicity.csv").iter_rows(named=True)}
    env = json.loads((tdir / "benchmark_environment.json").read_text(encoding="utf-8"))

    csv_r = p1["CSV (uncompressed)"]
    snappy_r = p1["Parquet single file, Snappy"]
    zstd_r = p1["Parquet single file, Zstd-3"]
    hive_r = p1["Parquet hive YEAR=, Zstd-3 (chosen)"]
    pd_csv_r = p1["pandas.read_csv (CSV)"]
    pd_pq_r = p1["pandas.read_parquet (single Zstd-3)"]

    max_diff = max(r["max_rel_diff_vs_polars_lazy"] for r in p3.values())
    ps_s = cap["Polars streaming"]
    pd_s = cap["pandas + pushdown"]
    ps8, pl8 = p3[("8x", "Polars streaming")], p3[("8x", "Polars lazy")]
    ps1, pl1, pe1, pd1, pn1 = (
        p3[("1x", "Polars streaming")],
        p3[("1x", "Polars lazy")],
        p3[("1x", "Polars eager")],
        p3[("1x", "pandas + pushdown")],
        p3[("1x", "pandas naive")],
    )
    ps2, pl2, pe2, pd2 = (
        p3[("2x", "Polars streaming")],
        p3[("2x", "Polars lazy")],
        p3[("2x", "Polars eager")],
        p3[("2x", "pandas + pushdown")],
    )
    ps4, pl4, pe4, pd4 = (
        p3[("4x", "Polars streaming")],
        p3[("4x", "Polars lazy")],
        p3[("4x", "Polars eager")],
        p3[("4x", "pandas + pushdown")],
    )
    sp1_2, sp4_2 = sp[("1x", "PySpark local[2]")], sp[("4x", "PySpark local[2]")]
    sp1_4, sp4_4 = sp[("1x", "PySpark local[4]")], sp[("4x", "PySpark local[4]")]
    parity_sum_sci = _format_latex_sci(float(parity["yearly_fingerprint"]["max_relative_diff_of_sums"]))

    q1_block = (
        f"With respect to the first research question—storage layout, laptop-scale processing, and the economic threshold for a distributed Spark cluster—"
        f"four decades of quarterly bank reports (1.68 million rows across 33 fields) compress from {csv_r['size_mb']:.1f} MB in raw uncompressed CSV "
        f"down to {hive_r['size_mb']:.1f} MB ({hive_r['compression_vs_csv']:.2f}x compression; write time {hive_r['write_s']:.3f} s) when stored as "
        f"Hive-partitioned Apache Parquet (YEAR=YYYY) with Zstandard level-3 compression, explicit canonical typing (Int32 keys, Date32 dates, Float64 measures), "
        f"and sorted (REPDTE, CERT) row groups. On a measured 1x–8x multi-scale stress test spanning 1.68 million up to 13.46 million rows ({env['cpu_logical']} logical CPUs, "
        f"{env['ram_total_gb']:.2f} GB RAM, 4 Polars threads), Polars Streaming completes 13.46 million rows in {ps8['time_s']:.3f} s ({int(ps8['peak_rss_mb']):,} MB peak RSS) and "
        f"Polars Lazy finishes in {pl8['time_s']:.3f} s ({int(pl8['peak_rss_mb']):,} MB), scaling at only {ps_s['mb_per_million_rows']:.1f} MB per million rows "
        f"(an extrapolated ceiling of {ps_s['rows_millions_until_16gb']:.1f} million rows in 16 GB RAM) compared with {pd_s['mb_per_million_rows']:.1f} MB per million rows for Pandas with pushdown "
        f"({pd_s['rows_millions_until_16gb']:.1f} million row ceiling) and {int(pn1['peak_rss_mb']):,} MB at 1x alone for naive Pandas. At equal 4-thread concurrency, local PySpark 4.3 (`local[4]`) reproduces "
        f"identical numeric results (${parity_sum_sci}$ yearly-sum parity, $0.000$ sample maximum difference) in {sp1_4['total_s']:.3f} s at 1x ({sp1_4['peak_rss_tree_mb']:,.1f} MB "
        f"Python+JVM tree RSS) and {sp4_4['total_s']:.3f} s at 4x (6.73 million rows, {sp4_4['peak_rss_tree_mb']:,.1f} MB tree RSS; `local[2]`: {sp4_2['total_s']:.3f} s). A distributed Spark cluster is therefore unnecessary "
        f"for quarterly bank-level Call Reports—panels below 30–50 million rows fit comfortably in Polars on a standard laptop—and becomes justified only when expanding to roughly 2,400 regulatory schedules, "
        f"borrower-level credit registries exceeding $10^8$ rows, or high-frequency interbank settlement streams exceeding 32–64 GB single-node memory."
    )

    p1_table = "\n".join([
        "**Table 2.** Storage layout, compression ratio, and query-latency benchmark across 170 quarters of FDIC financials.",
        "",
        "| Storage Format and Layout | Disk Size (MB) | Compression | Write (s) | Full Scan (s) | 3-Col Proj (s) | YEAR=2023 Filter (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| Raw CSV (Uncompressed) | {csv_r['size_mb']:.1f} | {csv_r['compression_vs_csv']:.2f}x | {csv_r['write_s']:.3f} | {csv_r['full_read_s']:.3f} | {csv_r['read_3_cols_s']:.3f} | {csv_r['filter_2023_s']:.3f} |",
        f"| Single Parquet (Snappy) | {snappy_r['size_mb']:.1f} | {snappy_r['compression_vs_csv']:.2f}x | {snappy_r['write_s']:.3f} | {snappy_r['full_read_s']:.3f} | {snappy_r['read_3_cols_s']:.3f} | {snappy_r['filter_2023_s']:.3f} |",
        f"| Single Parquet (Zstd L3) | {zstd_r['size_mb']:.1f} | {zstd_r['compression_vs_csv']:.2f}x | {zstd_r['write_s']:.3f} | {zstd_r['full_read_s']:.3f} | {zstd_r['read_3_cols_s']:.3f} | {zstd_r['filter_2023_s']:.3f} |",
        f"| **Hive Parquet (YEAR=, Zstd L3, Chosen)** | **{hive_r['size_mb']:.1f}** | **{hive_r['compression_vs_csv']:.2f}x** | **{hive_r['write_s']:.3f}** | **{hive_r['full_read_s']:.3f}** | **{hive_r['read_3_cols_s']:.3f}** | **{hive_r['filter_2023_s']:.3f}** |",
        f"| Pandas Baseline (read\\_csv / read\\_parquet) | {pd_csv_r['size_mb']:.1f} / {pd_pq_r['size_mb']:.1f} | {pd_csv_r['compression_vs_csv']:.2f}x / {pd_pq_r['compression_vs_csv']:.2f}x | — | {pd_csv_r['full_read_s']:.3f} / {pd_pq_r['full_read_s']:.3f} | — | — |",
        "",
        "*Note(s):* All timings report wall-clock medians across repeated reads of the 1,682,616-row panel in Polars (4 threads) unless designated as Pandas. Source measurements are stored in `outputs/tables/part1_storage_benchmark.csv`.",
    ])

    ram_pct_more = round((pd4["peak_rss_mb"] / ps4["peak_rss_mb"] - 1.0) * 100.0)
    slope_ratio = pd_s["mb_per_million_rows"] / ps_s["mb_per_million_rows"]
    spark_slowdown_4t = sp4_4["total_s"] / ps4["time_s"]
    spark_compute_slowdown_4t = sp4_4["compute_s"] / ps4["time_s"]
    p3_section = (
        f"### 4.2 Multi-Scale Benchmark (1x–8x, Up to 13.46 Million Rows)\n\n"
        f"To avoid unreliable linear extrapolation from sub-1.7-million-row slices, we benchmark Polars, Pandas, and PySpark on disjoint replicated panels scaled from "
        f"1x (1.68 million rows) up to 8x (13,460,928 rows) inside isolated `spawn` subprocesses on a Linux workstation equipped with {env['cpu_physical']} physical ({env['cpu_logical']} logical) CPUs "
        f"and {env['ram_total_gb']:.2f} GB RAM (Polars {env['polars']} capped at 4 threads; pandas {env['pandas']}). "
        f"For an equal-core comparison against 4-thread Polars, PySpark is benchmarked at `local[4]` alongside the 2-thread `local[2]` laptop-memory baseline.\n\n"
        f"**Table 5.** Multi-scale processing runtime and peak memory benchmark across Polars, Pandas, and PySpark (1x–8x).\n\n"
        f"| Scale (Rows) | Polars Streaming (4t) | Polars Lazy (4t) | Polars Eager (4t) | Pandas + Pushdown | PySpark `local[4]` (4t) *(vs. `local[2]`)* |\n"
        f"| :--- | :---: | :---: | :---: | :---: | :---: |\n"
        f"| 1x (1.68M) | {ps1['time_s']:.3f} s / {int(ps1['peak_rss_mb']):,} MB | {pl1['time_s']:.3f} s / {int(pl1['peak_rss_mb']):,} MB | {pe1['time_s']:.3f} s / {int(pe1['peak_rss_mb']):,} MB | {pd1['time_s']:.3f} s / {int(pd1['peak_rss_mb']):,} MB *(naive: {pn1['time_s']:.3f} s / {int(pn1['peak_rss_mb']):,} MB)* | {sp1_4['total_s']:.3f} s ({sp1_4['spark_startup_s']:.3f} s JVM) / {sp1_4['peak_rss_tree_mb']:,.1f} MB *(`local[2]`: {sp1_2['total_s']:.3f} s / {sp1_2['peak_rss_tree_mb']:,.1f} MB)* |\n"
        f"| 2x (3.37M) | {ps2['time_s']:.3f} s / {int(ps2['peak_rss_mb']):,} MB | {pl2['time_s']:.3f} s / {int(pl2['peak_rss_mb']):,} MB | {pe2['time_s']:.3f} s / {int(pe2['peak_rss_mb']):,} MB | {pd2['time_s']:.3f} s / {int(pd2['peak_rss_mb']):,} MB | — |\n"
        f"| 4x (6.73M) | **{ps4['time_s']:.3f} s / {int(ps4['peak_rss_mb']):,} MB** | {pl4['time_s']:.3f} s / {int(pl4['peak_rss_mb']):,} MB | {pe4['time_s']:.3f} s / {int(pe4['peak_rss_mb']):,} MB | {pd4['time_s']:.3f} s / {int(pd4['peak_rss_mb']):,} MB | {sp4_4['total_s']:.3f} s ({sp4_4['compute_s']:.3f} s compute) / {sp4_4['peak_rss_tree_mb']:,.1f} MB *(`local[2]`: {sp4_2['total_s']:.3f} s / {sp4_2['peak_rss_tree_mb']:,.1f} MB)* |\n"
        f"| 8x (13.46M) | **{ps8['time_s']:.3f} s / {int(ps8['peak_rss_mb']):,} MB** | {pl8['time_s']:.3f} s / {int(pl8['peak_rss_mb']):,} MB | *(skipped OOM risk)* | *(skipped eager overhead)* | — |\n\n"
        f"*Note(s):* Cells report wall-clock runtime / peak RSS (Python+JVM process-tree RSS for PySpark). Maximum relative checksum difference across all Polars, Pandas, and PySpark runs is $\\le {_format_latex_sci(max_diff)}$. "
        f"Sources: `outputs/tables/part3_benchmark_results.csv`, `spark_scaling_results.csv`, and `benchmark_environment.json`.\n\n"
        f"Measuring across 1x–8x exposes the structural memory advantage of streaming columnar execution (`outputs/tables/part3_capacity_extrapolation.json`). "
        f"Polars Streaming grows peak memory at only {ps_s['mb_per_million_rows']:.1f} MB per million rows ({ps_s['seconds_per_million_rows']:.3f} s per million rows, implying a {ps_s['rows_millions_until_16gb']:.1f}-million-row capacity ceiling inside 16 GB RAM), "
        f"whereas Pandas with column pushdown grows at {pd_s['mb_per_million_rows']:.1f} MB per million rows—a {slope_ratio:.2f}x steeper memory slope that exhausts 16 GB RAM by {pd_s['rows_millions_until_16gb']:.1f} million rows. "
        f"At 4x scale (6.73 million rows), Pandas with pushdown executes quickly on pre-sorted 8-column slices ({pd4['time_s']:.3f} s) but already consumes {ram_pct_more}% more memory than Polars Streaming "
        f"({int(pd4['peak_rss_mb']):,} MB vs. {int(ps4['peak_rss_mb']):,} MB), as depicted in Figure 3.\n\n"
        f"![Multi-scale execution time and peak RSS memory scaling across Polars, Pandas, and PySpark (1x–8x).](../outputs/figures/part3_scaling_benchmark.png)\n\n"
        f"### 4.3 PySpark Reproduction, Parity Verification, and Cluster Justification\n\n"
        f"In `scripts/06_spark.py` (1.2 GB driver memory), PySpark Catalyst reproduces Polars outputs identically across all 43 annual partitions, achieving a 0.0 maximum absolute difference across 33,042 sampled rows "
        f"and a ${parity_sum_sci}$ maximum relative difference in yearly aggregate checksums (`outputs/tables/spark_polars_parity.json`). Inspecting the Catalyst physical plan (`outputs/tables/spark_execution_plan.txt`) reveals two full `Exchange` shuffle barriers—"
        f"first partitioning by (CERT, YEAR) for YTD de-cumulation, then repartitioning by (CERT) for 4-quarter lags. Even on an equal 4-thread footing, PySpark `local[4]` incurs {sp4_4['spark_startup_s']:.3f} s of JVM startup overhead "
        f"and requires {sp4_4['total_s']:.3f} s total ({sp4_4['compute_s']:.3f} s compute, {sp4_4['peak_rss_tree_mb']:,.1f} MB Python+JVM tree RSS) at 6.73 million rows compared with {ps4['time_s']:.3f} s in 4-thread Polars Streaming—"
        f"{spark_slowdown_4t:.1f}x slower end-to-end and {spark_compute_slowdown_4t:.1f}x slower compute-only, while `local[2]` requires {sp4_2['total_s']:.3f} s and {sp4_2['peak_rss_tree_mb']:,.1f} MB tree RSS. "
        f"These measurements confirm that JVM initialization and shuffle barriers, rather than thread count, dominate local Spark latency. Consequently, a distributed Spark cluster is unwarranted for quarterly bank-level Call Reports below 30–50 million rows and becomes justified only when wide multi-schedule expansions or borrower-level credit registries overflow single-node RAM."
    )

    md_path = ROOT / "report" / "report.md"
    txt = md_path.read_text(encoding="utf-8")
    txt = _replace_sync_block(txt, "REPORT_Q1", q1_block)
    txt = _replace_sync_block(txt, "REPORT_PART1_TABLE", p1_table)
    txt = _replace_sync_block(txt, "REPORT_PART3_SECTION", p3_section)
    md_path.write_text(txt, encoding="utf-8")

    readme_bench = "\n".join([
        f"| **Part 1: Storage** | Hive-partitioned Parquet (`YEAR`, Zstd L3, write **{hive_r['write_s']:.3f} s**) reduces footprint from **{csv_r['size_mb']:.1f} MB** (CSV) to **{hive_r['size_mb']:.1f} MB** (**{hive_r['compression_vs_csv']:.2f}x compression**); `YEAR=2023` filter executes in **{hive_r['filter_2023_s']:.3f} s**. Quantifies `{f32['ASSET']['rows_exceeding_float32_exact_int_limit']:,}` `ASSET` rows exceeding Float32 exact-int limit (up to **$128k** quantization error), justifying `Float64`. |",
        f"| **Part 2: Quality & Panel** | Resolves {len(dq['schema_drift_columns'])}-column schema drift, `P3ASSET` structural missingness, pre-1990 Basel zeroes (`{dq['validity']['pre1990_rbc1_zero_pct']}%`), **post-2020 CBLR `RBCRWAJ==0` (`{dq['validity']['era2020_rbcrwa_zero_pct']}%`, `{dq['validity']['era2020_rbcrwa_zero']:,}/{dq['validity']['era2020_rows']:,}` rows)**, negative stock anomalies (`ORE<0` `{dq['validity']['ore_negative']}`, `SC<0` `{dq['validity']['sc_negative']}`), YTD `Q1` $\\rightarrow$ `Q4` flow step-up (`{ytd['Q1']['median_intinc_to_asset']:.4f}` $\\rightarrow$ `{ytd['Q4']['median_intinc_to_asset']:.4f}`), and FDIC `CHANGEC1` exit mapping (`16,892` censored mergers vs. `2,859` failures vs. `4,313` active). Excludes {dq['failures']['share_last_report_not_scorable']}% non-scorable final quarters (`pit_scorable=False`). |",
        f"| **Part 3: Processing** | Across **1x–8x (`1.68M–13.46M` rows)** (`POLARS_MAX_THREADS=4`): at `1x`, **Pandas + pushdown** (`{pd1['time_s']:.3f} s`, `{int(pd1['peak_rss_mb']):,} MB`) and **Polars Streaming** (`{ps1['time_s']:.3f} s`, `{int(ps1['peak_rss_mb']):,} MB`) lead; at **8x (`13.46M` rows)**, **Polars Streaming** completes in **`{ps8['time_s']:.3f} s`** (`{int(ps8['peak_rss_mb']):,} MB` RSS, scaling slope **`{ps_s['mb_per_million_rows']:.1f} MB / M rows`** $\\rightarrow$ `~{ps_s['rows_millions_until_16gb']:.1f}M` row limit in 16 GB RAM vs **`{pd_s['mb_per_million_rows']:.1f} MB / M rows`** $\\rightarrow$ `{pd_s['rows_millions_until_16gb']:.1f}M` for Pandas + pushdown). At equal 4 threads, **PySpark `local[4]`** matches Polars with **${parity_sum_sci}$ precision** (`{sp1_4['total_s']:.3f} s` / `{sp1_4['peak_rss_tree_mb']:,.1f} MB` tree RSS at `1x`; `{sp4_4['total_s']:.3f} s` / `{sp4_4['peak_rss_tree_mb']:,.1f} MB` at `4x`; `local[2]`: `{sp4_2['total_s']:.3f} s`). |",
    ])
    readme_path = ROOT / "README.md"
    readme_txt = readme_path.read_text(encoding="utf-8")
    readme_txt = _replace_sync_block(readme_txt, "README_BENCH", readme_bench)
    readme_path.write_text(readme_txt, encoding="utf-8")
    log.info("Synchronized report.md and README.md benchmark metrics from outputs/tables/.")


def update_report_metadata() -> Path:
    """Read report.md and inject student info if defined."""
    md_path = ROOT / "report" / "report.md"
    info = load_student_info()
    txt = md_path.read_text(encoding="utf-8")
    if info.get("student_name") and info["student_name"] != "[YOUR FULL NAME]":
        txt = txt.replace("[YOUR FULL NAME HERE]", info["student_name"])
    if info.get("student_id") and info["student_id"] != "[YOUR STUDENT ID]":
        txt = txt.replace("[YOUR STUDENT ID HERE]", info["student_id"])
    if info.get("github_url") and info["github_url"] != "[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME]":
        txt = txt.replace("[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME]", info["github_url"])
    md_path.write_text(txt, encoding="utf-8")
    log.info(f"Updated metadata in {md_path}")
    return md_path


_PANDOC_LUA_FILTER = r"""
local function latex_escape_code(s)
  local out = s:gsub("\\", "\\textbackslash{}")
  out = out:gsub("([%%$&#{}])", "\\%1")
  out = out:gsub("%^", "\\textasciicircum{}")
  out = out:gsub("~", "\\textasciitilde{}")
  out = out:gsub("_", "\\_\\allowbreak{}")
  out = out:gsub("/", "/\\allowbreak{}")
  return out
end

function Code(el)
  if FORMAT:match "latex" then
    return pandoc.RawInline("latex", "{\\normalfont\\ttfamily " .. latex_escape_code(el.text) .. "}")
  end
end

local function compute_col_widths(tbl)
  local ncols = #tbl.colspecs
  if ncols == 0 then return 0 end
  local weights = {}
  local min_chars = {}
  for j = 1, ncols do
    weights[j] = 4
    min_chars[j] = 4
  end
  local function inspect_rows(rows)
    for _, row in ipairs(rows or {}) do
      for j, cell in ipairs(row.cells or {}) do
        if j <= ncols then
          local txt = pandoc.utils.stringify(cell.contents)
          local len = math.min(#txt, 90)
          if len > weights[j] then weights[j] = len end
          for tok in txt:gmatch("%S+") do
            if #tok > min_chars[j] and #tok <= 26 then
              min_chars[j] = #tok
            end
          end
        end
      end
    end
  end
  if tbl.head then inspect_rows(tbl.head.rows) end
  for _, body in ipairs(tbl.bodies or {}) do
    inspect_rows(body.head)
    inspect_rows(body.body)
  end
  local sum_w = 0
  local sw = {}
  for j = 1, ncols do
    sw[j] = math.max(math.sqrt(weights[j]), min_chars[j] * 0.40)
    sum_w = sum_w + sw[j]
  end
  local min_w = (ncols <= 3) and 0.16 or 0.068
  local total = 0
  for j = 1, ncols do
    local w = math.max(min_w, (sw[j] / sum_w) * 0.96)
    sw[j] = w
    total = total + w
  end
  for j = 1, ncols do
    tbl.colspecs[j][2] = (sw[j] / total) * 0.96
  end
  return ncols
end

local function render_floating_table(cap_para, tbl, note_para)
  local ncols = compute_col_widths(tbl)
  local tbl_tex = pandoc.write(pandoc.Pandoc({tbl}), "latex")
  tbl_tex = tbl_tex:gsub("^%s*{%s*\\def\\LTcaptype{none}[^\n]*\n", "")
  tbl_tex = tbl_tex:gsub("\n}%s*$", "")
  tbl_tex = tbl_tex:gsub("\\begin{longtable}%[%]", "\\begin{tabular}")
  tbl_tex = tbl_tex:gsub("\\begin{longtable}", "\\begin{tabular}")
  tbl_tex = tbl_tex:gsub("\\endhead%s*\n\\bottomrule\\noalign{}%s*\n\\endlastfoot", "")
  tbl_tex = tbl_tex:gsub("\\endhead", "")
  tbl_tex = tbl_tex:gsub("\\endlastfoot", "")
  tbl_tex = tbl_tex:gsub("\\end{longtable}", "\\bottomrule\n\\end{tabular}")

  local sz = (ncols >= 6) and "\\footnotesize" or "\\small"
  local sep = (ncols >= 8) and "2.5pt" or "3.5pt"
  local cap_tex = cap_para and pandoc.write(pandoc.Pandoc({cap_para}), "latex") or ""
  local note_tex = note_para and pandoc.write(pandoc.Pandoc({note_para}), "latex") or ""

  local parts = {
    "\\begin{table}[htbp]",
    "\\centering",
    "\\begin{minipage}{\\linewidth}",
  }
  if cap_tex ~= "" then
    table.insert(parts, "{\\small\\noindent " .. cap_tex:gsub("%s+$", "") .. "\\par\\vspace{3pt}}")
  end
  table.insert(parts, "{\\centering" .. sz .. "\\renewcommand{\\arraystretch}{1.04}\\setlength{\\tabcolsep}{" .. sep .. "}")
  table.insert(parts, tbl_tex)
  table.insert(parts, "\\par}")
  if note_tex ~= "" then
    table.insert(parts, "{\\vspace{3pt}\\footnotesize\\noindent " .. note_tex:gsub("%s+$", "") .. "\\par}")
  end
  table.insert(parts, "\\end{minipage}")
  table.insert(parts, "\\end{table}")
  return pandoc.RawBlock("latex", table.concat(parts, "\n"))
end

function Table(tbl)
  if not FORMAT:match "latex" then
    compute_col_widths(tbl)
    return tbl
  end
  return tbl
end

function Blocks(blocks)
  if not FORMAT:match "latex" then return blocks end
  local out = {}
  local i = 1
  while i <= #blocks do
    local blk = blocks[i]
    if blk.t == "Para" and pandoc.utils.stringify(blk):match("^Table [A-Z]?%d+%.") and i < #blocks and blocks[i + 1].t == "Table" then
      local note_para = nil
      local step = 2
      if i + 2 <= #blocks and blocks[i + 2].t == "Para" then
        local ntxt = pandoc.utils.stringify(blocks[i + 2])
        if ntxt:match("^Note%(s%):") or ntxt:match("^Note:") then
          note_para = blocks[i + 2]
          step = 3
        end
      end
      table.insert(out, render_floating_table(blk, blocks[i + 1], note_para))
      i = i + step
    elseif blk.t == "Table" then
      table.insert(out, render_floating_table(nil, blk, nil))
      i = i + 1
    else
      table.insert(out, blk)
      i = i + 1
    end
  end
  return out
end

function Header(el)
  if FORMAT:match "latex" then
    if el.level == 1 then
      return {
        pandoc.RawBlock("latex", "\\thispagestyle{firstpage}\\vspace*{0.35in}\\begin{center}{\\fontsize{16}{21}\\selectfont\\bfseries " .. pandoc.utils.stringify(el) .. "}\\end{center}\\vspace{0.35em}"),
      }
    elseif el.level == 2 then
      local txt = pandoc.utils.stringify(el)
      if txt:match("^1%.%s+Introduction") or txt:match("^Appendix:") then
        return {
          pandoc.RawBlock("latex", "\\clearpage"),
          el,
        }
      end
    end
  end
  return el
end

function Div(el)
  if FORMAT:match "latex" then
    if el.classes:includes("titlemeta") then
      local out = { pandoc.RawBlock("latex", "\\begin{center}\\small\\setlength{\\parskip}{1.5pt}") }
      for _, blk in ipairs(el.content) do table.insert(out, blk) end
      table.insert(out, pandoc.RawBlock("latex", "\\end{center}\\vspace{0.25em}"))
      return out
    elseif el.classes:includes("abstract") then
      local out = { pandoc.RawBlock("latex", "\\begin{abstractbox}\\small\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{3pt}") }
      for _, blk in ipairs(el.content) do table.insert(out, blk) end
      table.insert(out, pandoc.RawBlock("latex", "\\end{abstractbox}\\vspace{0.35em}"))
      return out
    elseif el.classes:includes("references") then
      local out = { pandoc.RawBlock("latex", "\\begingroup\\small\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{3pt}") }
      for _, blk in ipairs(el.content) do
        if blk.t == "Para" then
          local ptex = pandoc.write(pandoc.Pandoc({blk}), "latex"):gsub("%s+$", "")
          table.insert(out, pandoc.RawBlock("latex", "\\noindent\\hangindent=1.5em\\hangafter=1 " .. ptex .. "\\par"))
        else
          table.insert(out, blk)
        end
      end
      table.insert(out, pandoc.RawBlock("latex", "\\endgroup"))
      return out
    end
  end
  return el
end

function Para(el)
  if FORMAT:match "latex" then
    local txt = pandoc.utils.stringify(el)
    if txt:match("^AI Tool Disclosure:") then
      return {
        pandoc.RawBlock("latex", "\\begingroup\\noindent\\footnotesize\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{2pt}"),
        el,
        pandoc.RawBlock("latex", "\\endgroup\\vspace{3pt}"),
      }
    end
  end
  return el
end

function HorizontalRule()
  if FORMAT:match "latex" then
    return pandoc.RawBlock("latex", "\\vspace{1pt}")
  end
  return {}
end

function Image(img)
  if FORMAT:match "latex" then
    img.attributes["width"] = "78%"
  elseif FORMAT:match "docx" then
    img.attributes["width"] = "5.8in"
  end
  return img
end
"""

_XELATEX_PREAMBLE = r"""
\usepackage{xcolor}
\definecolor{journalblue}{HTML}{003366}
\definecolor{abstractbg}{HTML}{F3F5F8}
\definecolor{rulegray}{HTML}{4A5568}
\usepackage{mdframed}
\newmdenv[
  backgroundcolor=abstractbg,
  linecolor=abstractbg,
  linewidth=0pt,
  leftmargin=0pt,
  rightmargin=0pt,
  innerleftmargin=10pt,
  innerrightmargin=10pt,
  innertopmargin=7pt,
  innerbottommargin=7pt,
  skipabove=3pt,
  skipbelow=4pt
]{abstractbox}
\usepackage{float}
\floatplacement{figure}{htbp}
\renewcommand{\topfraction}{0.90}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{9pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\usepackage{booktabs}
\usepackage{calc}
\usepackage{longtable}
\usepackage{array}
\usepackage{xurl}
\usepackage{fancyhdr}
\usepackage{caption}
\captionsetup{font={small},labelfont=bf,labelsep=period,justification=centering,skip=4pt}
\linespread{1.07}
\setlength{\parindent}{1.5em}
\setlength{\parskip}{1.5pt}
\emergencystretch=2em
\pagestyle{fancy}
\fancyhf{}
\lhead{\footnotesize\itshape\textcolor{rulegray}{BFT65B — A Big-Data Pipeline for Bank Failure Early Warning}}
\rhead{\footnotesize\textcolor{rulegray}{National Economics University}}
\cfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.45pt}
\fancypagestyle{firstpage}{
  \fancyhf{}
  \lhead{\footnotesize\itshape\textcolor{rulegray}{BFT65B — Big Data Analytics in Finance (Final Research Report)}}
  \rhead{\footnotesize\textcolor{rulegray}{National Economics University, Hanoi}}
  \cfoot{\small\thepage}
  \renewcommand{\headrulewidth}{0.75pt}
}
\usepackage{titlesec}
\titleformat{\section}{\normalfont\fontsize{13}{15}\selectfont\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalfont\fontsize{12}{14}\selectfont\bfseries}{}{0pt}{}
\titleformat{\subsubsection}{\normalfont\fontsize{12}{14}\selectfont\itshape\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{9pt plus 2pt minus 1pt}{3pt}
\titlespacing*{\subsection}{0pt}{7pt plus 1pt minus 1pt}{2.5pt}
\titlespacing*{\subsubsection}{0pt}{5pt}{2pt}
"""


def _prepare_pandoc_markdown(md_path: Path) -> str:
    """Convert semantic HTML block comments to Pandoc fenced divs while keeping report.md clean on GitHub."""
    text = md_path.read_text(encoding="utf-8")
    for block_name, div_class in (
        ("TITLEMETA", "titlemeta"),
        ("ABSTRACT", "abstract"),
        ("REFERENCES", "references"),
    ):
        text = text.replace(f"<!-- BLOCK:{block_name}:START -->", f"::: {{.{div_class}}}")
        text = text.replace(f"<!-- BLOCK:{block_name}:END -->", ":::")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = text.replace("<br>", "; ")
    return text


def _set_xml_font(rpr, font_name: str) -> None:
    """Ensure Word sets ASCII, HAnsi, EastAsia, and CS fonts consistently."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), font_name)


def _shade_paragraph(paragraph, fill_hex: str) -> None:
    """Apply subtle academic background shading to a Word paragraph (e.g., Abstract box)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    ppr = paragraph._p.get_or_add_pPr()
    for existing in ppr.findall(qn("w:shd")):
        ppr.remove(existing)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    ppr.append(shd)


def _apply_booktabs_borders(table) -> None:
    """Format a Word table with three horizontal booktabs rules and zero vertical rules."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tbl_pr = table._tbl.tblPr
    for existing in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for side, val, sz, color in (
        ("top", "single", "12", "1A202C"),
        ("bottom", "single", "12", "1A202C"),
        ("left", "none", "0", "auto"),
        ("right", "none", "0", "auto"),
        ("insideH", "none", "0", "auto"),
        ("insideV", "none", "0", "auto"),
    ):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), val)
        if val != "none":
            el.set(qn("w:sz"), sz)
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), color)
        borders.append(el)
    tbl_pr.append(borders)

    # Thin midrule beneath header row
    if table.rows:
        for cell in table.rows[0].cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            for existing in tc_pr.findall(qn("w:tcBorders")):
                tc_pr.remove(existing)
            tc_b = OxmlElement("w:tcBorders")
            bot = OxmlElement("w:bottom")
            bot.set(qn("w:val"), "single")
            bot.set(qn("w:sz"), "6")
            bot.set(qn("w:space"), "0")
            bot.set(qn("w:color"), "2D3748")
            tc_b.append(bot)
            tc_pr.append(tc_b)


def _style_docx_report(docx_path: Path) -> None:
    """Post-process Pandoc DOCX to match the Q1 finance journal PDF (Times New Roman 12pt, booktabs tables, shaded Abstract, APA hanging-indent references)."""
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    doc = Document(str(docx_path))

    for sec in doc.sections:
        sec.top_margin = Inches(0.82)
        sec.bottom_margin = Inches(0.82)
        sec.left_margin = Inches(0.82)
        sec.right_margin = Inches(0.82)

    body_style_names = {"Normal", "Body Text", "First Paragraph", "Compact", "List Paragraph"}
    heading_sizes = {"Title": 16, "Heading 1": 16, "Heading 2": 13, "Heading 3": 12, "Heading 4": 12}
    for st in doc.styles:
        if not hasattr(st, "font") or st.font is None:
            continue
        if st.name in ("Verbatim Char", "Source Code"):
            st.font.name = "Courier New"
            if st._element.rPr is not None:
                _set_xml_font(st._element.rPr, "Courier New")
            continue
        st.font.name = "Times New Roman"
        if st._element.rPr is not None:
            _set_xml_font(st._element.rPr, "Times New Roman")
        if st.name in body_style_names:
            st.font.size = Pt(12)
            if hasattr(st, "paragraph_format") and st.paragraph_format is not None:
                st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                st.paragraph_format.line_spacing = 1.15
                st.paragraph_format.space_after = Pt(3)
        elif st.name in heading_sizes:
            st.font.size = Pt(heading_sizes[st.name])
            st.font.bold = True

    in_abstract = False
    in_references = False
    after_heading = True
    fig_counter = 0

    for p in doc.paragraphs:
        sname = p.style.name if p.style else ""
        txt = p.text.strip()
        has_drawing = bool(p._element.findall(".//" + qn("w:drawing")) or p._element.findall(".//" + qn("w:pict")))

        if txt == "Abstract":
            in_abstract = True
        if sname.startswith("Heading") and txt == "References":
            in_references = True
        elif sname.startswith("Heading") and in_references:
            in_references = False

        if sname in ("Heading 1", "Title") or txt.startswith("Quản Hữu Nhật Minh") or txt.startswith("School of Banking") or txt.startswith("Student ID:") or txt.startswith("Repository:"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_after = Pt(2)
            after_heading = True
        elif in_abstract:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.left_indent = Inches(0.15)
            p.paragraph_format.right_indent = Inches(0.15)
            p.paragraph_format.line_spacing = 1.12
            p.paragraph_format.space_after = Pt(3.5)
            _shade_paragraph(p, "F3F5F8")
            for run in p.runs:
                run.font.size = Pt(11)
            if "JEL Classification:" in txt:
                in_abstract = False
                after_heading = True
        elif has_drawing:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Inches(0)
            after_heading = True
        elif sname in ("Caption", "Image Caption"):
            fig_counter += 1
            if not txt.startswith("Figure "):
                if p.runs:
                    first_run = p.runs[0]
                    first_run.text = f"Figure {fig_counter}. " + first_run.text
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_after = Pt(6)
            for run in p.runs:
                run.font.size = Pt(10.5)
            after_heading = True
        elif sname.startswith("Heading"):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            if txt.startswith("1. Introduction") or txt.startswith("Appendix:"):
                p.paragraph_format.page_break_before = True
            after_heading = True
        elif in_references and txt:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.35)
            p.paragraph_format.line_spacing = 1.12
            p.paragraph_format.space_after = Pt(3.5)
            for run in p.runs:
                run.font.size = Pt(11)
        elif re.match(r"^Table [A-Z]?\d+\.", txt):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            for run in p.runs:
                run.font.size = Pt(11)
            after_heading = True
        elif txt.startswith("Note(s):") or txt.startswith("AI Tool Disclosure:"):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            for run in p.runs:
                run.font.size = Pt(10)
            after_heading = True
        elif sname not in ("Source Code",):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.first_line_indent = Inches(0) if after_heading else Inches(0.25)
            after_heading = False

        for run in p.runs:
            rstyle = run.style.name if run.style else ""
            if rstyle in ("Verbatim Char", "Source Code"):
                run.font.name = "Courier New"
                _set_xml_font(run._r.get_or_add_rPr(), "Courier New")
                run.font.size = Pt(9.5)
                run.font.bold = False
                continue
            run.font.name = "Times New Roman"
            _set_xml_font(run._r.get_or_add_rPr(), "Times New Roman")
            if (sname in body_style_names or sname == "") and run.font.size is None:
                run.font.size = Pt(12)

    for tbl in doc.tables:
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        _apply_booktabs_borders(tbl)
        ncols = len(tbl.columns)
        cell_pt = Pt(9.5) if ncols >= 7 else Pt(10.5)
        for r_idx, row in enumerate(tbl.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.08
                    p.paragraph_format.space_before = Pt(1.5)
                    p.paragraph_format.space_after = Pt(1.5)
                    p.paragraph_format.first_line_indent = Inches(0)
                    for run in p.runs:
                        rstyle = run.style.name if run.style else ""
                        if rstyle in ("Verbatim Char", "Source Code"):
                            run.font.name = "Courier New"
                            _set_xml_font(run._r.get_or_add_rPr(), "Courier New")
                            run.font.size = Pt(8.5 if ncols >= 7 else 9.0)
                            run.font.bold = False
                            continue
                        run.font.name = "Times New Roman"
                        _set_xml_font(run._r.get_or_add_rPr(), "Times New Roman")
                        run.font.size = cell_pt
                        if r_idx == 0:
                            run.font.bold = True

    doc.save(str(docx_path))


def build_report_pdf_and_docx(md_path: Path) -> Path:
    """Render report.md to Q1-journal LaTeX PDF (Times New Roman 12pt + TeX Gyre Termes Math) and styled DOCX."""
    pdf_path = md_path.parent / "QuanHuuNhatMinh_11230998_Final_Report.pdf"
    docx_path = md_path.parent / "QuanHuuNhatMinh_11230998_Final_Report.docx"
    resource_path = f"{md_path.parent}:{out_dir('figures')}"

    with tempfile.TemporaryDirectory(prefix="bfews_pandoc_") as tmpdir:
        tmp = Path(tmpdir)
        clean_md = tmp / "report_pandoc.md"
        lua_filter = tmp / "filter.lua"
        preamble = tmp / "preamble.tex"
        clean_md.write_text(_prepare_pandoc_markdown(md_path), encoding="utf-8")
        lua_filter.write_text(_PANDOC_LUA_FILTER.lstrip(), encoding="utf-8")
        preamble.write_text(_XELATEX_PREAMBLE.lstrip(), encoding="utf-8")

        subprocess.run(
            [
                "pandoc",
                str(clean_md),
                "--from=gfm+tex_math_dollars+implicit_figures+fenced_divs",
                "--pdf-engine=xelatex",
                f"--lua-filter={lua_filter}",
                f"--include-in-header={preamble}",
                f"--resource-path={resource_path}",
                "-V", "indent=true",
                "-V", "geometry:margin=0.72in",
                "-V", "fontsize=12pt",
                "-V", "mainfont=Times New Roman",
                "-V", "sansfont=Times New Roman",
                "-V", "mathfont=texgyretermes-math.otf",
                "-V", "monofont=Courier New",
                "-V", "monofontoptions=Scale=0.82",
                "-V", "colorlinks=true",
                "-V", "linkcolor=journalblue",
                "-V", "urlcolor=journalblue",
                "-o", str(pdf_path),
            ],
            check=True,
        )
        log.info(f"Compiled Q1-journal LaTeX PDF report (Times New Roman 12pt + TeX Gyre Termes Math) to {pdf_path}")

        subprocess.run(
            [
                "pandoc",
                str(clean_md),
                "--from=gfm+tex_math_dollars+implicit_figures+fenced_divs",
                f"--lua-filter={lua_filter}",
                f"--resource-path={resource_path}",
                "-o", str(docx_path),
            ],
            check=True,
        )
        _style_docx_report(docx_path)
        log.info(f"Compiled and styled Q1-journal Word DOCX (Times New Roman 12pt, booktabs, shaded Abstract) to {docx_path}")

    return pdf_path


build_docx_report = build_report_pdf_and_docx


def main():
    build_notebooks()
    sync_benchmarks_into_docs()
    md_path = update_report_metadata()
    build_report_pdf_and_docx(md_path)


if __name__ == "__main__":
    main()
