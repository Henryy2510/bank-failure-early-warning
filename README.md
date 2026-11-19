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
   ├──► scripts/05_benchmark.py ──► Multi-Scale (1x-8x, up to 13.46M) Polars vs Pandas Scaling + Plan Diff
   ├──► scripts/06_spark.py     ──► PySpark Reproduction, 1x & 4x Python+JVM RSS, Parity (0.0 diff) & MLlib Demo
   │
   ▼  scripts/04_features.py
Gold Layer (data/gold/features.parquet)
   • 26 leak-free CAMELS + Texas Ratio + 4Q lag features (CBLR 2020+ zero-fix & negative stock sanitization)
   • 4-quarter forward failure label: y_{i,q} = 1{fail in (q, q+4]} (restricted to q <= 2025Q2)
   • Strict temporal split: DEV (1984Q1-2007Q4) | PURGE (2008Q1-2008Q4) | TEST (2009Q1-2025Q2) | POST_SAMPLE (2025Q3-2026Q2)
   │
   ▼  scripts/07_model.py
Supervisory Modelling & Evaluation (outputs/tables/test_results.json)
   • In-fold sklearn Pipeline: RobustWinsorizer → SimpleImputer(+Indicators) → StandardScaler → PriorOddsCalibrator
   • Expanding-window walk-forward CV on DEV (760,559 OOF rows, 1,522 OOF failing banks; 2,661 total DEV failing banks)
   • DEV label sensitivity (FAILURE vs FAILURE+ASSISTANCE), leakage shift table, & live 2026Q2 operational watchlist (4,306 banks)
   • Prior-odds calibrated cost-optimal threshold (theta* = 0.0039) at C_FP = $250k + fixed capacity budgets (Top 1%, 2%, 5%)
   • Held-out TEST evaluation: PR-AUC = 0.5478, ROC-AUC = 0.9854, Recall@Top-1% = 83.53%, Bank Recall@theta* = 96.70% (469/485)
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
│   ├── figures/                       # 9 high-resolution PNG figures (200 DPI, incl. PR/Cost U-curves & Canary trajectories)
│   └── tables/                        # Multi-scale & Spark benchmarks (local[2] & local[4]), Float32/CBLR/YTD/CHANGEC1 audits, leakage shift, cfp_stability_interval, frozen_protocol, watchlist_2026q2, test_results
├── report/                            # Final Research Report deliverables
│   ├── report.md                      # Full English report in GitHub Markdown
│   ├── BFT65B_Final_Report.docx       # Formatted Word document with embedded tables and figures
│   └── BFT65B_Final_Report.pdf        # Final PDF report (<= 20 pages)
└── tests/
    └── test_pipeline.py               # Automated pytest suite (20 tests: completeness, de-YTD, CBLR zero-fix, forbidden cols, post-failure filter, CV purge, leakage invariance, C_FP stability interval, frozen_protocol manifest, Spark local[2]/local[4] parity, non-degenerate cost threshold, README paths, PDF pages)
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
   * To download fresh data from the public FDIC API (`https://api.fdic.gov/banks`, no API key required; resumable, ~45–60 minutes for all 170 quarters `1984Q1–2026Q2`), run:
     ```bash
     .venv/bin/python scripts/00_download.py --download --out ./data_raw/fdic
     ```
     (which invokes the root `fdic_download_starter.py` script and runs offline completeness verification automatically).

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
* **Step 5 (`05_benchmark.py`):** Benchmarks Polars vs. Pandas across `1x–8x` (`1.68M–13.46M` rows) and dumps `outputs/tables/polars_naive_vs_optimized_plan.txt` & `benchmark_environment.json`.
* **Step 6 (`06_spark.py`):** Runs PySpark pipeline, verifies `0.0` discrepancy against Polars, measures `1x & 4x` Python+JVM tree RSS (`spark_scaling_results.csv`), dumps `spark_execution_plan.txt`, and runs the Spark MLlib demo.
* **Step 7 (`07_model.py`):** Runs 5-fold expanding walk-forward CV on `DEV` (`760,559` OOF rows, `1,522` OOF failing banks; `2,661` total `DEV` failing banks), audits leakage shifts & `FAILURE`-only label sensitivity, calibrates cost-optimal threshold ($\theta^* = 0.0039$), computes capacity budgets (`Top 1%, 2%, 5%`), evaluates once on `TEST`, scores `2026Q2` live operational watchlist (`watchlist_2026q2.csv`), and plots curves & trajectories.
* **Step 8 (`08_build_notebooks_and_report.py`):** Executes `notebooks/*.ipynb` via `nbclient` and regenerates `report/BFT65B_Final_Report.docx` and `.pdf`.
* **Step 9 (`09_check_report_numbers.py`):** Verifies that all key metrics in `report/report.md` and `README.md` match the generated JSON/CSV files.
* **Tests (`pytest tests/`):** Runs 20 automated unit/integration tests verifying manifest completeness, YTD de-cumulation, CBLR zero recoding & negative stock sanitization, forbidden outcome column exclusion, zero post-failure Gold rows, strict CV purges, preprocessor leakage invariance, `cfp_stability_interval.json` linear-cost decomposition, `frozen_protocol.json` audit manifest SHA-256 fingerprints, Polars–PySpark parity (`local[2]` & `local[4]`), non-degenerate cost threshold, capacity points, explainability, Canary studies, upgraded artifacts, README script paths, and the 20-page PDF limit.

---

## 4. Key Empirical Results Summary

| Pipeline Stage | Key Empirical Measurement |
| :--- | :--- |
<!-- SYNC:START:README_BENCH -->
| **Part 1: Storage** | Hive-partitioned Parquet (`YEAR`, Zstd L3, write **0.922 s**) reduces footprint from **425.7 MB** (CSV) to **154.1 MB** (**2.76x compression**); `YEAR=2023` filter executes in **0.040 s**. Quantifies `11,485` `ASSET` rows exceeding Float32 exact-int limit (up to **$128k** quantization error), justifying `Float64`. |
| **Part 2: Quality & Panel** | Resolves 14-column schema drift, `P3ASSET` structural missingness, pre-1990 Basel zeroes (`18.13%`), **post-2020 CBLR `RBCRWAJ==0` (`37.15%`, `45,865/123,461` rows)**, negative stock anomalies (`ORE<0` `646`, `SC<0` `96`), YTD `Q1` $\rightarrow$ `Q4` flow step-up (`0.0177` $\rightarrow$ `0.0689`), and FDIC `CHANGEC1` exit mapping (`16,892` censored mergers vs. `2,859` failures vs. `4,313` active). Excludes 59.1% non-scorable final quarters (`pit_scorable=False`). |
| **Part 3: Processing** | Across **1x–8x (`1.68M–13.46M` rows)** (`POLARS_MAX_THREADS=4`): at `1x`, **Pandas + pushdown** (`0.482 s`, `1,044 MB`) and **Polars Streaming** (`0.619 s`, `1,268 MB`) lead; at **8x (`13.46M` rows)**, **Polars Streaming** completes in **`4.271 s`** (`2,499 MB` RSS, scaling slope **`101.0 MB / M rows`** $\rightarrow$ `~151.1M` row limit in 16 GB RAM vs **`327.9 MB / M rows`** $\rightarrow$ `48.5M` for Pandas + pushdown). At equal 4 threads, **PySpark `local[4]`** matches Polars with **$7.4 \times 10^{-15}$ precision** (`10.551 s` / `1,262.9 MB` tree RSS at `1x`; `15.182 s` / `1,807.8 MB` at `4x`; `local[2]`: `16.163 s`). |
<!-- SYNC:END:README_BENCH -->
| **Part 4: Features** | 26 point-in-time CAMELS + Texas Ratio + 4Q lag features with CBLR zero-fix & negative stock cleaning; naive full-sample fit shifts upper quantiles up to `20.95%`; invariance & CV purge verified via `pytest`. |
| **Part 5: Modelling** | Walk-forward CV PR-AUC (`760,559` OOF rows, `1,522` OOF failing banks): `HistGradientBoosting, 26 CAMELS` (**0.4624**, ROC **0.9851**, Brier **0.00337**) > `Logistic, 3 variables` (**0.4015**) > `Logistic, 26 CAMELS` (**0.3601**) > `Texas ratio rule` (**0.2642**). Held-out `TEST` (`2009Q1–2025Q2`, `400,332` rows, `485` failing banks): **PR-AUC = 0.5478**, **ROC-AUC = 0.9854**, **Brier = 0.00238**, **Recall@Top-1% = 83.53%**, **Recall@Top-5% = 95.60%** (Modern `2015–2025` ROC-AUC lifted to **0.9193**). Top strictly-`DEV` risk drivers (`feature_importance.csv`): **`texas_ratio`** (`+0.26242` $\Delta\text{PR-AUC}$), **`d4_eq_ratio`** (`+0.06486`), **`eq_asset_ratio`** (`+0.06167`). Post-hoc **Canary Case Studies** (`canary_case_studies.csv`): catches **5/5 top GFC closures** (`129–304 d` early) plus *Doral Bank* (`333 d`) & *Republic Bank* (`301 d`). At calibrated **$\theta^* = 0.0039$** ($C_{\text{FP}}=\$250\text{k}$/episode), flags **2.894%** of test rows (`43.0` median alerts/qtr) and captures **96.70%** of failing banks (`469 / 485`, `297.0 d` lead); **Top 1%/qtr** catches **82.27%** (`399 / 485`, precision **20.95%**, `210 d` lead). Operational `2026Q2` scoring (`watchlist_2026q2.csv`): scores `4,306` active banks (`26` flagged >= $\theta^*$). Spark MLlib DEV ROC-AUC = **0.9388**. |

---

## 5. Academic Integrity & Citations

1. **Primary Data Source & Official Institutional References:**  
   US Federal Deposit Insurance Corporation (FDIC) BankFind Suite Public API (`https://api.fdic.gov/banks`), RISView Call Report Field Dictionary (`https://api.fdic.gov/banks/docs/`), FDIC 2024 Annual Report, and FDIC Proposed 2025 Operating Budget. Acquisition logic builds upon the optional course starter script `fdic_download_starter.py`.
2. **Academic Literature:**  
   * Wheelock, D. C., & Wilson, P. W. (2000). *Why do banks disappear? The determinants of U.S. bank failures and acquisitions.* Review of Economics and Statistics, 82(1), 127–138.  
   * Shumway, T. (2001). *Forecasting bankruptcy more accurately: A simple hazard model.* The Journal of Business, 74(1), 101–124.  
   * Cole, R. A., & White, L. J. (2012). *Déjà vu all over again: The causes of U.S. commercial bank failures this time around.* Journal of Financial Services Research, 42(1), 5–29.  
   * Elkan, C. (2001). *The foundations of cost-sensitive learning.* Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), 973–978.
3. **External Code & Reference Repositories (Cited & Critically Improved):**  
   * `batuhanisik751/bank-failure-early-warning` ([GitHub](https://github.com/batuhanisik751/bank-failure-early-warning)): Consulted for CAMELS/Texas Ratio baseline and walk-forward cost intuition; upgraded to Bronze/Silver/Gold Polars + PySpark, Elkan prior-odds calibration, episode cost, and capacity budgets.  
   * `Tajaddin/bank-early-warning` ([GitHub](https://github.com/Tajaddin/bank-early-warning)): R/SQL supervisory econometrics inspiration; upgraded to 1.68M-row Polars/Spark processing with HGB permutation importance paired with standardized Logit directionality.  
   * `vfabuluje/Canary` ([GitHub](https://github.com/vfabuluje/Canary)): Inspired *"Canary in the coal mine"* individual-bank tracking via our `canary_case_studies.csv` post-hoc analysis of top GFC & Modern receiverships.  
   * `martinogie/bank-health-data-platform` ([GitHub](https://github.com/martinogie/bank-health-data-platform)) & `saivardhan13-MSV/fdic-bank-health-pipeline` ([GitHub](https://github.com/saivardhan13-MSV/fdic-bank-health-pipeline)): Medallion/ETL engineering inspirations; extended with PyArrow Hive Zstd compression benchmarks, 42-year era-missingness audit, merger censoring, and Spark parity checks.  
   * `heltonwo/fdic-failed-banks-etl` ([GitHub](https://github.com/heltonwo/fdic-failed-banks-etl)): Evaluated; limited to failures-only ETL (survivorship bias), whereas our panel includes all `24,064` active/merged/failed FDIC institutions.  
   * `Zahivc/Bank-Bankruptcy-Predictions` ([GitHub](https://github.com/Zahivc/Bank-Bankruptcy-Predictions)): Evaluated; avoided its **SMOTE + random split** approach (which introduces temporal leakage) in favor of strict walk-forward splits and Bayes prior-odds calibration.  
   * `zekaili/bank-loan-failure-prediction` ([Kaggle](https://www.kaggle.com/code/zekaili/bank-loan-failure-prediction)): Addresses retail consumer loans rather than institutional bank failures; not used.
4. **AI Tools Disclosure:**  
   Google Antigravity AI coding assistant was utilized for code structuring, PySpark/Polars syntax optimization, and report formatting. All data processing steps, benchmarks, and statistical results were executed and verified locally on the FDIC dataset.
