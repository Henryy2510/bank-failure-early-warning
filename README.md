# BFT65B — A Big-Data Pipeline for Bank Failure Early Warning

**National Economics University (NEU)**  
**Course:** BFT65B – Big Data Analytics in Finance (Final Examination Individual Research Report)  

---

## 1. Overview & Architecture

This repository implements an end-to-end, reproducible, and memory-bounded **Medallion Lakehouse Pipeline** (`Bronze` $\rightarrow$ `Silver` $\rightarrow$ `Gold`) on 42 years of public US Federal Deposit Insurance Corporation (FDIC) Call Report data (`1984Q1–2026Q2`, **1,682,616 bank-quarters** across **24,064 institutions**) to support a point-in-time supervisory early-warning system (EWS) for bank failures.

```
data_raw/fdic (170 quarterly files, failures, institutions)
   │
   ▼  scripts/01_storage.py
Bronze Layer (data/bronze/financials)
   • Canonical Arrow schema (Int32 CERT, Date32 REPDTE, Float64 balances)
   • Hive-partitioned by YEAR=YYYY, sorted by (REPDTE, CERT), Zstandard L3 (154.1 MB vs 425.7 MB CSV)
   │
   ├──► scripts/02_quality.py ──► Automated DQ Audit (outputs/dq_report/data_quality_report.md)
   │
   ▼  scripts/03_panel.py
Silver Layer (data/silver/panel.parquet)
   • De-cumulated quarterly income statement flows (NETINC_Q, INTINC_Q, ...)
   • Point-in-time publication lag (available_date = REPDTE + 60 days)
   • Terminal exit classification: FAILED (2,859 banks) vs MERGED_OR_OTHER (16,892 censored) vs ACTIVE (4,313)
   │
   ├──► scripts/05_benchmark.py ──► Polars (Lazy/Streaming/Eager) vs Pandas Scaling Benchmarks
   ├──► scripts/06_spark.py     ──► PySpark Window Reproduction, Parity Check (0.0 diff) & MLlib Demo
   │
   ▼  scripts/04_features.py
Gold Layer (data/gold/features.parquet)
   • 26 leak-free CAMELS + Texas Ratio + 4Q historical lag features
   • 4-quarter forward failure label: y_{i,q} = 1{fail in (q, q+4]} (restricted to q <= 2025Q2)
   • Strict temporal split: DEV (1984Q1-2007Q4) | PURGE (2008Q1-2008Q4) | TEST (2009Q1-2025Q2)
   │
   ▼  scripts/07_model.py
Supervisory Modelling & Evaluation (outputs/tables/test_results.json)
   • In-fold sklearn Pipeline: RobustWinsorizer -> SimpleImputer(+Indicators) -> StandardScaler -> PriorOddsCalibrator
   • Expanding-window walk-forward CV on DEV (760,559 OOF rows, 1,522 OOF failing banks; 2,661 total DEV failing banks)
   • Prior-odds calibrated cost-optimal threshold (theta* = 0.0026) at C_FP = $250k + fixed capacity budgets (Top 1%, 2%, 5%)
   • Held-out TEST evaluation: PR-AUC = 0.5541, ROC-AUC = 0.9845, Recall@Top-1% = 82.75%, Bank Recall@theta* = 97.11% (471/485)
```

---

## 2. Repository Structure

```text
final/
├── README.md                          # End-to-end execution guide, results summary & citations
├── requirements.txt                   # Pinned Python 3.12 dependencies
├── config.yaml                        # Central configuration (paths, temporal splits, seed=42, costs)
├── run_all.sh                         # Master script executing all pipeline stages + pytest suite
├── src/bfews/                         # Core modular Python package
│   ├── __init__.py
│   ├── core.py                        # Canonical shared definitions (NETINC_Q, CAMELS formulas, checksums)
│   ├── utils.py                       # Config loader, isolated subprocess RSS/time profiler, markdown helpers
│   ├── io.py                          # Raw-to-Bronze ingestion, canonical schema casting, Hive partitioning
│   ├── quality.py                     # Automated DQ audit (schema drift, era missingness, invariants)
│   ├── panel.py                       # YTD flow de-cumulation, PIT lag, exit classification
│   ├── labels.py                      # Forward 4Q failure label construction & DEV/PURGE/TEST partitioning
│   ├── features.py                    # 26 point-in-time CAMELS, Texas Ratio & 4Q lag features
│   ├── spark_jobs.py                  # PySpark Window reproduction, physical plan dump & MLlib LR demo
│   └── modeling.py                    # Leak-free sklearn Pipeline, walk-forward CV, cost threshold & test eval
├── scripts/                           # Sequential CLI entrypoints
│   ├── 00_download.py                 # Resumable FDIC API downloader & offline manifest verifier
│   ├── 01_storage.py                  # Part 1: Build Bronze layer & benchmark CSV vs Parquet layouts
│   ├── 02_quality.py                  # Part 2: Generate automated DQ report & missingness-by-era plot
│   ├── 03_panel.py                    # Part 2: Build Silver panel & bank exit summary table
│   ├── 04_features.py                 # Part 4: Build Gold feature table & split summary
│   ├── 05_benchmark.py                # Part 3: Benchmark Polars (Lazy/Streaming/Eager) vs Pandas
│   ├── 06_spark.py                    # Part 3 & 5: Run PySpark pipeline, parity check & MLlib demo
│   ├── 07_model.py                    # Part 5: Run walk-forward CV, threshold tuning & held-out test evaluation
│   ├── 08_build_notebooks_and_report.py # Execute Jupyter notebooks via nbclient and build Report (MD + DOCX + PDF)
│   └── 09_check_report_numbers.py     # Automated consistency check verifying report/README against JSON/CSV outputs
├── notebooks/                         # Executed Jupyter notebooks (executed via nbclient with fixed seeds)
│   ├── 01_acquisition_and_storage.ipynb
│   ├── 02_data_quality_and_panel.ipynb
│   ├── 03_scalable_processing_polars_spark.ipynb
│   ├── 04_feature_pipeline.ipynb
│   └── 05_modelling_and_validation.ipynb
├── outputs/                           # Committed empirical outputs (tables, figures, DQ report)
│   ├── dq_report/data_quality_report.md
│   ├── figures/                       # 6 high-resolution PNG figures (200 DPI)
│   └── tables/                        # Benchmark CSVs, query plans, parity JSON, test_results.json, run_all_timing.txt
├── report/                            # Final Research Report deliverables
│   ├── report.md                      # Full English report in GitHub Markdown
│   ├── BFT65B_Final_Report.docx       # Formatted Word document with embedded tables and figures
│   └── BFT65B_Final_Report.pdf        # Final PDF report (<= 20 pages)
└── tests/
    └── test_pipeline.py               # Automated pytest suite (8 tests: completeness, de-YTD, leak-free, parity, threshold, capacity)
```

> **Note on Data Files:** Per course submission rules, `data/` (`bronze/`, `silver/`, `gold/`) and `*.parquet` files are excluded via `.gitignore` and are not committed to Git.

---

## 3. Reproducing the Entire Pipeline End-to-End

### 3.1 Prerequisites
* **OS:** Linux / macOS / Windows (WSL2)
* **Python:** Python `3.12+`
* **Java (for PySpark 4.3):** OpenJDK `17` or `21` (`java -version`)
* **Hardware:** Runs comfortably on a standard laptop (tested on 12-core CPU, 15 GB RAM; peak pipeline process RSS is capped below `1.4 GB` via `POLARS_MAX_THREADS=4` and Spark `driver.memory=1200m`).

### 3.2 Setup & Data Placement
1. Create a virtual environment and install the pinned dependencies:
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. **Obtain or Verify Raw FDIC Data:**
   * If raw data is already present at `./data_raw/fdic` (configurable in `config.yaml`), verify its integrity offline (takes ~1 second):
     ```bash
     .venv/bin/python scripts/00_download.py --verify-only
     ```
   * To download fresh data from the public FDIC API (`https://api.fdic.gov/banks`, no API key required; resumable, ~45–60 minutes for all 170 quarters `1984Q1–2026Q2`), run `data/fdic_download_starter.py --out ./data_raw/fdic`.

### 3.3 Run All Stages & Tests with One Command
```bash
bash run_all.sh
```
This script runs Steps `00` through `09`, logs per-step wall-clock times to `outputs/tables/run_all_timing.txt`, and executes `pytest -v tests/`:
* **Step 0 (`00_download.py`):** Verifies 170 quarters (`1,682,616` rows, `0` duplicate keys).
* **Step 1 (`01_storage.py`):** Builds `data/bronze/financials` and outputs `outputs/tables/part1_storage_benchmark.csv`.
* **Step 2 (`02_quality.py`):** Produces `outputs/dq_report/data_quality_report.md` and `outputs/figures/missingness_by_era.png`.
* **Step 3 (`03_panel.py`):** Builds `data/silver/panel.parquet` and `outputs/tables/bank_exit_summary.csv`.
* **Step 4 (`04_features.py`):** Builds `data/gold/features.parquet` and `outputs/tables/gold_split_summary.csv`.
* **Step 5 (`05_benchmark.py`):** Benchmarks Polars vs. Pandas and dumps `outputs/tables/polars_optimized_plan.txt`.
* **Step 6 (`06_spark.py`):** Runs PySpark pipeline, verifies `0.0` discrepancy against Polars, dumps `outputs/tables/spark_execution_plan.txt`, and runs the Spark MLlib Logistic Regression demo.
* **Step 7 (`07_model.py`):** Runs 5-fold expanding walk-forward CV on `DEV` (`760,559` OOF rows, `1,522` OOF failing banks; `2,661` total `DEV` failing banks), calibrates the prior-odds cost-optimal threshold ($\theta^* = 0.0026$), computes fixed-capacity operating points (`Top 1%`, `2%`, `5%`), and evaluates on `TEST`.
* **Step 8 (`08_build_notebooks_and_report.py`):** Executes `notebooks/*.ipynb` via `nbclient` and regenerates `report/BFT65B_Final_Report.docx` and `.pdf`.
* **Step 9 (`09_check_report_numbers.py`):** Verifies that all key metrics in `report/report.md` and `README.md` match the generated JSON/CSV files.
* **Tests (`pytest tests/`):** Runs 8 automated unit/integration tests verifying manifest completeness, YTD de-cumulation, strict point-in-time leak-free invariance, merger censoring, Polars–PySpark parity, non-degenerate cost threshold, capacity operating points, and test metrics.

---

## 4. Key Empirical Results Summary

| Pipeline Stage | Key Empirical Measurement |
| :--- | :--- |
| **Part 1: Storage** | Hive-partitioned Parquet (`YEAR`, Zstd L3) reduces disk footprint from **425.7 MB** (CSV) to **154.1 MB** (**2.76x compression**); single-year partition filter (`YEAR=2023`) executes in **0.035 s** with directory-level partition pruning. |
| **Part 2: Quality & Panel** | Resolves 14-column schema drift, pre-2001 `P3ASSET` structural missingness, pre-1990 Basel zeroes (`18.13%`), and YTD cumulative flows; classifies `16,892` merged banks as right-censored ($y=0$) vs. `2,859` failed banks and `4,313` active banks. Excludes 59.1% non-scorable final quarters (`pit_scorable=False`). |
| **Part 3: Processing** | On the full `1.68M`-row panel, **Polars Streaming** is fastest overall at **0.581 s** (RSS: **992 MB**), **Pandas + pushdown** finishes in **0.613 s** (RSS: **1,039 MB**), and **Polars Lazy** achieves the lowest memory footprint at **691 MB** (**1.271 s**) vs. **Pandas Naive** **1.353 s** (**1,947 MB** RSS, 2.8x higher RAM). Local **PySpark** matches Polars with **$2.24 \times 10^{-15}$ precision match** and identical row counts across 43 years. |
| **Part 4: Features** | 26 point-in-time CAMELS + Texas Ratio + 4Q lag features; zero look-ahead verified by automated truncation invariance test (`test_leak_free_property`). |
| **Part 5: Modelling** | Walk-forward CV PR-AUC (`760,559` OOF rows, `1,522` OOF failing banks): `HistGradientBoosting, 26 CAMELS` (**0.4699**, Brier **0.00337**) > `Logistic, 3 variables` (**0.4015**) > `Logistic, 26 CAMELS` (**0.3632**) > `Texas ratio rule` (**0.2642**). Held-out `TEST` (`2009Q1–2025Q2`, `400,332` rows, `485` failing banks): **PR-AUC = 0.5541**, **ROC-AUC = 0.9845**, **Brier = 0.00238**, **Recall@Top-1% = 82.75%**, **Recall@Top-5% = 95.46%**. Under fixed quarterly capacity **Top 1%/qtr**, catches **82.68%** of failing banks (`401 / 485`, precision **20.98%**, median lead **206 days**). At the prior-odds calibrated cost-optimal threshold $\theta^* = 0.0026$ ($C_{\text{FP}}=\$250\text{k}$ per episode), flags **3.657%** of test rows (`82.5` median alerts/qtr) and captures **97.11%** of failing banks (`471 / 485`) with **297.0 days** median lead time. Spark MLlib demo Development ROC-AUC = **0.9399**. |

---

## 5. Academic Integrity & Citations

1. **Primary Data Source:**  
   US Federal Deposit Insurance Corporation (FDIC) BankFind Suite Public API (`https://api.fdic.gov/banks`) and RISView Call Report Field Dictionary (`https://api.fdic.gov/banks/docs/`). Acquisition logic builds upon the optional course starter script `fdic_download_starter.py`.
2. **Academic Literature:**  
   * Wheelock, D. C., & Wilson, P. W. (2000). *Why do banks disappear? The determinants of U.S. bank failures and acquisitions.* Review of Economics and Statistics, 82(1), 127–138.  
   * Shumway, T. (2001). *Forecasting bankruptcy more accurately: A simple hazard model.* The Journal of Business, 74(1), 101–124.  
   * Cole, R. A., & White, L. J. (2012). *Déjà vu all over again: The causes of U.S. commercial bank failures this time around.* Journal of Financial Services Research, 42(1), 5–29.
3. **External Code & Reference Repositories:**  
   * Isik, B. (2026). *`batuhanisik751/bank-failure-early-warning`: Open, reproducible early-warning system for US bank failures built on FDIC Call Report data* (MIT License, `https://github.com/batuhanisik751/bank-failure-early-warning`). Consulted for domain feature concepts (Texas Ratio baseline and walk-forward validation structure). All Polars, PySpark, storage benchmarking, and scikit-learn pipeline implementations in this repository were written independently for the BFT65B rubric.  
   * Li, Z. *`zekaili/bank-loan-failure-prediction`* (`https://www.kaggle.com/code/zekaili/bank-loan-failure-prediction`). Evaluated during planning; focuses on consumer transaction loan default rather than institutional bank failure and was not used.
4. **AI Tools Disclosure:**  
   Google Antigravity AI coding assistant was utilized for code structuring, PySpark/Polars syntax optimization, and report formatting. All data processing steps, benchmarks, and statistical results were executed and verified locally on the FDIC dataset.
