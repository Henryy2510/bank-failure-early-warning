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
   • Hive-partitioned by YEAR=YYYY, sorted by (REPDTE, CERT), Zstandard L3 (154.1 MB vs 444.7 MB CSV)
   │
   ├──► scripts/02_quality.py ──► Automated DQ Audit (outputs/dq_report/data_quality_report.md)
   │
   ▼  scripts/03_panel.py
Silver Layer (data/silver/panel.parquet)
   • De-cumulated quarterly income statement flows (NETINC_Q, INTINC_Q, ...)
   • Point-in-time publication lag (available_date = REPDTE + 60 days)
   • Terminal exit classification: FAILED (2,856 banks) vs MERGED_OR_OTHER (16,850 censored) vs ACTIVE (4,358)
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
   • In-fold sklearn Pipeline: RobustWinsorizer -> SimpleImputer(+Indicators) -> StandardScaler -> Classifier
   • Expanding-window walk-forward CV on DEV (with 4-quarter purge buffer)
   • Data-driven alarm threshold (theta* = 0.3990) from median DIF loss ($9,814.7k) vs exam cost ($250k)
   • Single evaluation on held-out TEST set: PR-AUC = 0.6721, ROC-AUC = 0.9870, Recall@Top-1% = 85.09%
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
│   ├── utils.py                       # Config loader, isolated subprocess RSS/time profiler, markdown helpers
│   ├── io.py                          # Raw-to-Bronze ingestion, canonical schema casting, Hive partitioning
│   ├── quality.py                     # Automated DQ audit (schema drift, era missingness, invariants)
│   ├── panel.py                       # YTD flow de-cumulation, PIT lag, exit classification
│   ├── labels.py                      # Forward 4Q failure label construction & DEV/PURGE/TEST partitioning
│   ├── features.py                    # 26 point-in-time CAMELS, Texas Ratio & 4Q lag features
│   ├── pandas_baseline.py             # Pure Pandas baseline for fair Part 3 benchmarking
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
│   ├── 07_model.py                    # Part 5: Run walk-forward CV, threshold tuning & single test evaluation
│   └── 08_build_notebooks_and_report.py # Build executed Jupyter notebooks and Report (MD + DOCX)
├── notebooks/                         # Executed Jupyter notebooks (with outputs and fixed seeds)
│   ├── 01_acquisition_and_storage.ipynb
│   ├── 02_data_quality_and_panel.ipynb
│   ├── 03_scalable_processing_polars_spark.ipynb
│   ├── 04_feature_pipeline.ipynb
│   └── 05_modelling_and_validation.ipynb
├── outputs/                           # Committed empirical outputs (tables, figures, DQ report)
│   ├── dq_report/data_quality_report.md
│   ├── figures/                       # 5 high-resolution PNG figures (200 DPI)
│   └── tables/                        # Benchmark CSVs, query plans, parity JSON, test_results.json
├── report/                            # Final Research Report deliverables
│   ├── report.md                      # Full English report in GitHub Markdown
│   └── BFT65B_Final_Report.docx       # Formatted Word document with embedded tables and figures
└── tests/
    └── test_pipeline.py               # Automated pytest suite (completeness, de-YTD, leak-free, parity)
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
This script runs Steps `00` through `08` and executes `pytest -v tests/` in **~1.5 minutes total**:
* **Step 0 (`00_download.py`):** Verifies 170 quarters (`1,682,616` rows, `0` duplicate keys).
* **Step 1 (`01_storage.py`):** Builds `data/bronze/financials` and outputs `outputs/tables/part1_storage_benchmark.csv` (~3s).
* **Step 2 (`02_quality.py`):** Produces `outputs/dq_report/data_quality_report.md` and `outputs/figures/missingness_by_era.png` (~2s).
* **Step 3 (`03_panel.py`):** Builds `data/silver/panel.parquet` and `outputs/tables/bank_exit_summary.csv` (~3s).
* **Step 4 (`04_features.py`):** Builds `data/gold/features.parquet` and `outputs/tables/gold_split_summary.csv` (~2s).
* **Step 5 (`05_benchmark.py`):** Benchmarks Polars vs. Pandas and dumps `outputs/tables/polars_optimized_plan.txt` (~18s).
* **Step 6 (`06_spark.py`):** Runs PySpark pipeline, verifies `0.0` discrepancy against Polars, dumps `outputs/tables/spark_execution_plan.txt`, and runs the Spark MLlib Logistic Regression demo (~25s).
* **Step 7 (`07_model.py`):** Runs 5-fold expanding walk-forward CV on `DEV`, calibrates the cost-optimal threshold ($\theta^* = 0.3990$), and evaluates once on `TEST` (~50s).
* **Step 8 (`08_build_notebooks_and_report.py`):** Regenerates `notebooks/*.ipynb` and `report/BFT65B_Final_Report.docx` (~2s).
* **Tests (`pytest tests/`):** Runs 6 automated unit/integration tests verifying manifest completeness, YTD de-cumulation, strict point-in-time leak-free invariance, merger censoring, Polars–PySpark parity, and test metrics (~0.3s).

---

## 4. Key Empirical Results Summary

| Pipeline Stage | Key Empirical Measurement |
| :--- | :--- |
| **Part 1: Storage** | Hive-partitioned Parquet (`YEAR`, Zstd L3) reduces disk footprint from **444.7 MB** (CSV) to **154.1 MB** (**2.89x compression**); single-year partition filter (`YEAR=2023`) executes in **0.009 s** (9 ms). |
| **Part 2: Quality & Panel** | Resolves 14-column schema drift, pre-2001 `P3ASSET` structural missingness, pre-1990 Basel zeroes (`18.13%`), and YTD cumulative flows; classifies `16,850` merged banks as right-censored ($y=0$) vs. `2,856` failed banks. |
| **Part 3: Processing** | On the full `1.68M`-row panel, **Polars Lazy** executes in **0.572 s** (Polars Eager RSS: **815 MB**) vs. **Pandas Baseline** **1.211 s** (**2,231 MB** RSS); on temporal sub-slices (`0.25x`), Polars Lazy is **7.7x faster** (`0.158 s` vs. `1.221 s`) with **4.0x lower memory**. Local **PySpark** matches Polars with **`0.0` numerical discrepancy**. |
| **Part 4: Features** | 26 point-in-time CAMELS + Texas Ratio + 4Q lag features; zero look-ahead verified by automated truncation invariance test (`test_leak_free_property`). |
| **Part 5: Modelling** | Walk-forward CV PR-AUC: `HistGradientBoosting` (**0.5303**) > `Logistic_CAMELS` (**0.4587**) > `Logistic_3Var` (**0.4508**) > `Texas_Ratio` (**0.1238**). Held-out `TEST` (`2009Q1–2025Q2`): **PR-AUC = 0.6721**, **ROC-AUC = 0.9870**, **Recall@Top-1% = 85.09%**, **Recall@Top-5% = 95.95%**. At $\theta^* = 0.3990$ ($C_{\text{FN}}=\$9,814.7\text{k}$, $C_{\text{FP}}=\$250\text{k}$), captures **95.39%** of failure quarters (`1,695 / 1,777`). Spark MLlib demo Test ROC-AUC = **0.9769**. |

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
