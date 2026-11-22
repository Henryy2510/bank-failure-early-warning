# BFT65B — A Big-Data Pipeline for Bank Failure Early Warning

**National Economics University (NEU)**  
**Course:** BFT65B – Big Data Analytics in Finance (Final Examination Individual Research Report)  

---

## 1. Overview & Architecture

This repository implements an end-to-end, reproducible, and memory-bounded **Medallion Lakehouse Pipeline** (`Bronze` → `Silver` → `Gold`) on 42 years of public US Federal Deposit Insurance Corporation (FDIC) Call Report data (`1984Q1–2026Q2`, **1,682,616 bank-quarters** across **24,064 institutions**) to support a point-in-time supervisory early-warning system (EWS) for bank failures.

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
├── run_all.sh                         # Master Bash script (Linux/macOS/WSL2) executing all stages + pytest
├── run_all.ps1                        # Native Windows PowerShell runner (no WSL required) executing all stages + pytest
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
│   └── tables/                        # Multi-scale & Spark benchmarks (local[2] & local[4]), full-schema benchmark (part3_full_schema_benchmark.json), Float32/CBLR/YTD/CHANGEC1 audits, leakage shift, cfp_stability_interval, frozen_protocol, watchlist_2026q2, test_results
├── report/                            # Final Research Report deliverables
│   ├── report.md                      # Full English report in GitHub Markdown
│   ├── QuanHuuNhatMinh_11230998_Final_Report.docx       # Formatted Word document with embedded tables and figures
│   └── QuanHuuNhatMinh_11230998_Final_Report.pdf        # Final PDF report (20 main pages + 2-page appendix; <= 20 main + <= 10 appendix)
└── tests/
    └── test_pipeline.py               # Automated pytest suite (20 tests: completeness, de-YTD, CBLR zero-fix, forbidden cols, post-failure filter, CV purge, leakage invariance, C_FP stability interval, frozen_protocol manifest, Spark local[2]/local[4] parity, non-degenerate cost threshold, README paths, PDF pages)
```

> **Note on Data Files:** Per course submission rules, `data/` (`bronze/`, `silver/`, `gold/`) and `*.parquet` files are excluded via `.gitignore` and are not committed to Git.

---

## 3. Reproducing the Entire Pipeline End-to-End

### 3.1 Prerequisites
* **OS:** Linux, macOS, or **native Windows 10/11 (PowerShell — no WSL required)** (WSL2 also supported).
* **Python:** Python `3.12+`
* **Java (for PySpark 4.3):** OpenJDK `17` or `21` (`java -version`). On native Windows, PySpark also requires `JAVA_HOME` and Hadoop `winutils.exe` + `hadoop.dll` under `%HADOOP_HOME%\bin` (see Section 3.3).
* **Optional — Report PDF/DOCX Recompilation & PDF Audit Tools (Steps 8–9):**
  * `pandoc` (`>= 3.0`) and `xelatex` (TeX Live `texlive-xetex` + `texlive-latex-extra` for `mdframed`, `titlesec`, `fancyhdr`, `booktabs` on Linux/macOS; or MiKTeX on Windows), plus system fonts **`Times New Roman`**, **`Courier New`**, and **`TeX Gyre Termes Math`** (`texgyretermes-math.otf`; `Times New Roman` and `Courier New` are pre-installed on Windows and macOS; on Debian/Ubuntu install `ttf-mscorefonts-installer` and `fonts-texgyre` / `texlive-fonts-recommended`).
  * Optional `poppler-utils` (`pdfinfo`, `pdftotext`, `pdffonts`) for automated PDF page-count and embedded-font inspection in Step 9 and `pytest`.
  * **Graceful fallback:** If `pandoc`, `xelatex`, or the required TeX fonts/packages are missing on the host machine, Step 8 (`scripts/08_build_notebooks_and_report.py`) logs a warning and skips PDF/DOCX recompilation while preserving the pre-compiled files in `report/`, so Step 9 (`09_check_report_numbers.py`) and `pytest` always execute cleanly. Likewise, Step 9 and `pytest` skip only the `poppler` binary checks if `pdfinfo` is absent while still verifying all Markdown, JSON, CSV, and DOCX checks.
* **Hardware:** Runs comfortably on a standard laptop (tested on 6 physical / 12 logical CPU cores, `15.34 GB` RAM with `POLARS_MAX_THREADS=4` and Spark `driver.memory=1200m`; measured peak process RSS spans **`~0.7–2.7 GB`** across workloads — `1,597.7 MB` for the full 1x 34→67-column Bronze-to-Gold pipeline, `1,807.8 MB` for 4x PySpark `local[4]` Python+JVM tree RSS, `2,499 MB` for 8x Polars Streaming, and `2,684.9–2,689 MB` during 2x full-schema / 4x Pandas stress runs).

### 3.2 Setup & Execution — Linux / macOS (Bash)
1. Create a virtual environment and install the pinned dependencies:
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. **Obtain or Verify Raw FDIC Data:**
   * If raw data is already present at `./data_raw/fdic` (configurable in `config.yaml`), verify its integrity offline (~1 second):
     ```bash
     .venv/bin/python scripts/00_download.py --verify-only
     ```
   * To download fresh data from the public FDIC API (`https://api.fdic.gov/banks`, no API key required; resumable, ~45–60 minutes for all 170 quarters `1984Q1–2026Q2`), run:
     ```bash
     .venv/bin/python scripts/00_download.py --download --out ./data_raw/fdic
     ```
     (which invokes the root `fdic_download_starter.py` script and runs offline completeness verification automatically).
3. **Run All Stages & Tests with One Command:**
   ```bash
   bash run_all.sh
   ```

### 3.3 Setup & Execution — Native Windows 10/11 (PowerShell, No WSL Required)
All Python scripts, benchmarks (using `psutil` peak working-set RSS fallback when Unix `resource` is absent), PySpark jobs, and tests run natively on Windows without WSL:
1. **Install Python 3.12 and OpenJDK 17** (in PowerShell):
   ```powershell
   winget install Python.Python.3.12
   winget install EclipseAdoptium.Temurin.17.JDK
   ```
   Open a new PowerShell window and verify `py -3.12 --version` and `java -version`. Ensure `JAVA_HOME` points to the JDK directory (e.g., `C:\Program Files\Eclipse Adoptium\jdk-17...`).
2. **Configure Hadoop `winutils` (required by PySpark local Parquet I/O on native Windows):**
   * Create `C:\hadoop\bin` and place matching Hadoop 3.x `winutils.exe` and `hadoop.dll` inside `C:\hadoop\bin` (e.g., from `cdarlint/winutils` or `kontext-tech/winutils` for Hadoop `3.3.x`/`3.4.x`).
   * Set environment variables (current session or System Environment Variables):
     ```powershell
     $env:HADOOP_HOME = "C:\hadoop"
     $env:PATH = "$env:HADOOP_HOME\bin;$env:PATH"
     ```
3. **(Optional — only if you want Step 8 to recompile PDF/DOCX locally):**
   ```powershell
   winget install JohnMacFarlane.Pandoc
   winget install MiKTeX.MiKTeX
   winget install oschwartz10612.Poppler
   ```
   (`Times New Roman` and `Courier New` are pre-installed on Windows; in MiKTeX Console enable automatic on-the-fly package installation for `mdframed`, `titlesec`, `fancyhdr`, and `tex-gyre-math`. Skipping this step is safe — Step 8 will warn and keep the pre-compiled report files intact.)
4. **Create the virtual environment and install pinned packages** (inside `final\`):
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
5. **Verify or Download Raw FDIC Data:**
   ```powershell
   # Offline verification if .\data_raw\fdic already exists:
   .venv\Scripts\python.exe scripts\00_download.py --verify-only

   # Or download fresh from FDIC API (~45-60 min, resumable):
   .venv\Scripts\python.exe scripts\00_download.py --download --out .\data_raw\fdic
   ```
6. **Run All Stages & Tests with One Command:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
   ```

### 3.4 Pipeline Stage Breakdown (`run_all.sh` / `run_all.ps1`)
Both runners execute Steps `00` through `09`, log per-step wall-clock times to `outputs/tables/run_all_timing.txt`, and run `pytest -v tests/`:
* **Step 0 (`00_download.py`):** Verifies 170 quarters (`1,682,616` rows, `0` duplicate keys).
* **Step 1 (`01_storage.py`):** Builds `data/bronze/financials` and outputs `outputs/tables/part1_storage_benchmark.csv`.
* **Step 2 (`02_quality.py`):** Produces `outputs/dq_report/data_quality_report.md` and `outputs/figures/missingness_by_era.png`.
* **Step 3 (`03_panel.py`):** Builds `data/silver/panel.parquet` and `outputs/tables/bank_exit_summary.csv`.
* **Step 4 (`04_features.py`):** Builds `data/gold/features.parquet` and `outputs/tables/gold_split_summary.csv`.
* **Step 5 (`05_benchmark.py`):** Benchmarks Polars vs. Pandas across `1x–8x` (`1.68M–13.46M` rows), runs the `1x/2x` full 34→67-column Bronze-to-Gold schema benchmark (`outputs/tables/part3_full_schema_benchmark.json`), and dumps `outputs/tables/polars_naive_vs_optimized_plan.txt` & `benchmark_environment.json`.
* **Step 6 (`06_spark.py`):** Runs PySpark pipeline, verifies `0.0` discrepancy against Polars, measures `1x & 4x` Python+JVM tree RSS (`spark_scaling_results.csv`), dumps `spark_execution_plan.txt`, and runs the Spark MLlib demo.
* **Step 7 (`07_model.py`):** Runs 5-fold expanding walk-forward CV on `DEV` (`760,559` OOF rows, `1,522` OOF failing banks; `2,661` total `DEV` failing banks), audits leakage shifts & `FAILURE`-only label sensitivity, calibrates cost-optimal threshold (θ\* = 0.0039), computes capacity budgets (`Top 1%, 2%, 5%`), evaluates once on `TEST`, scores `2026Q2` live operational watchlist (`watchlist_2026q2.csv`), and plots curves & trajectories.
* **Step 8 (`08_build_notebooks_and_report.py`):** Executes `notebooks/*.ipynb` via `nbclient`, synchronizes `report/report.md` and `README.md` benchmark tables from `outputs/tables/`, and recompiles `report/QuanHuuNhatMinh_11230998_Final_Report.docx` and `.pdf` (automatically skipping PDF/DOCX compilation with a warning if `pandoc`/`xelatex` is absent, while keeping pre-compiled files intact).
* **Step 9 (`09_check_report_numbers.py`):** Verifies that all key metrics in `report/report.md` and `README.md` match the generated JSON/CSV files (and inspects PDF page limits and embedded fonts when `poppler-utils` is available).
* **Tests (`pytest tests/`):** Runs 20 automated unit/integration tests verifying manifest completeness, YTD de-cumulation, CBLR zero recoding & negative stock sanitization, forbidden outcome column exclusion, zero post-failure Gold rows, strict CV purges, preprocessor leakage invariance, `cfp_stability_interval.json` linear-cost decomposition, `frozen_protocol.json` audit manifest SHA-256 fingerprints, Polars–PySpark parity (`local[2]` & `local[4]`), non-degenerate cost threshold, capacity points, explainability, Canary studies, upgraded artifacts, README script paths (`run_all.sh` & `run_all.ps1`), and the PDF page ceiling.

---

## 4. Key Empirical Results Summary

<!-- SYNC:START:README_BENCH -->
| Pipeline Stage | Key Empirical Measurement |
| :--- | :--- |
| **Part 1: Storage** | Hive-partitioned Parquet (`YEAR`, Zstd L3, write **0.922 s**) reduces footprint from **425.7 MB** (CSV) to **154.1 MB** (**2.76x compression**); `YEAR=2023` filter executes in **0.040 s**. Quantifies `11,485` `ASSET` rows exceeding Float32 exact-int limit (up to **$128k** quantization error), justifying `Float64`. |
| **Part 2: Quality & Panel** | Resolves 14-column schema drift, `P3ASSET` structural missingness, pre-1990 Basel zeroes (`18.13%`), **post-2020 CBLR `RBCRWAJ==0` (`37.15%`, `45,865/123,461` rows)**, negative stock anomalies (`ORE<0` `646`, `SC<0` `96`), YTD `Q1` → `Q4` flow step-up (`0.0177` → `0.0689`), and FDIC `CHANGEC1` exit mapping (`16,892` censored mergers vs. `2,859` failures vs. `4,313` active). Excludes 59.1% non-scorable final quarters (`pit_scorable=False`). |
| **Part 3: Processing** | Across **1x–8x (`1.68M–13.46M` rows, 8-col core)** (`POLARS_MAX_THREADS=4`): at `1x`, **Pandas + pushdown** (`0.482 s`, `1,044 MB`) and **Polars Streaming** (`0.619 s`, `1,268 MB`) lead; at **8x (`13.46M` rows)**, **Polars Streaming** completes in **`4.271 s`** (`2,499 MB` RSS, slope **`101.0 MB / M rows`** → `~151.1M` row limit in 16 GB RAM vs **`327.9 MB / M rows`** → `48.5M` for Pandas + pushdown). On the **full 34-col Bronze → 67-col Gold schema** (`part3_full_schema_benchmark.json`, `1x/2x`), Polars Streaming runs in **`1.830 s` / `1,597.7 MB`** (`1x`) and **`3.475 s` / `2,631.7 MB`** (`2x`), scaling at **`614.5 MB / M rows`** (`.sink_parquet()`: `666.3 MB / M rows`), placing the practical 16 GB laptop ceiling (~12 GB usable) at **`~20M` full-schema rows** (`19.1M` collect / `17.8M` sink). At equal 4 threads, **PySpark `local[4]`** matches Polars with **7.4e-15 precision** (`10.551 s` / `1,262.9 MB` tree RSS at `1x`; `15.182 s` / `1,807.8 MB` at `4x`; `local[2]`: `16.163 s`). |
| **Part 4: Features** | 26 point-in-time CAMELS + Texas Ratio + 4Q lag features with CBLR zero-fix & negative stock cleaning; naive full-sample fit shifts upper quantiles up to `20.95%`; invariance & CV purge verified via `pytest`. |
| **Part 5: Modelling** | Walk-forward CV PR-AUC (`760,559` OOF rows, `1,522` OOF failing banks): `HistGradientBoosting, 26 CAMELS` (**0.4624**, ROC **0.9851**, Brier **0.00337**) > `Logistic, 3 variables` (**0.4015**) > `Logistic, 26 CAMELS` (**0.3601**) > `Texas ratio rule` (**0.2642**). Held-out `TEST` (`2009Q1–2025Q2`, `400,332` rows, `485` failing banks): **PR-AUC = 0.5478**, **ROC-AUC = 0.9854**, **Brier = 0.00238**, **Recall@Top-1% = 83.53%**, **Recall@Top-5% = 95.60%** (Modern `2015–2025` ROC-AUC lifted to **0.9193**). Top strictly-`DEV` risk drivers (`feature_importance.csv`): **`texas_ratio`** (`+0.26242` ΔPR-AUC), **`d4_eq_ratio`** (`+0.06486`), **`eq_asset_ratio`** (`+0.06167`). Post-hoc **Canary Case Studies** (`canary_case_studies.csv`): catches **5/5 top GFC closures** (`129–304 d` early) plus *Doral Bank* (`333 d`) & *Republic Bank* (`301 d`). At calibrated **θ\* = 0.0039** (`C_FP = $250k`/episode, invariant over `$231.1k–$519.9k`), flags **2.894%** of test rows (`43.0` median alerts/qtr) and captures **96.70%** of failing banks (`469 / 485`, `297.0 d` lead); **Top 1%/qtr** catches **82.27%** (`399 / 485`, precision **20.95%**, `210 d` lead). Operational `2026Q2` scoring (`watchlist_2026q2.csv`): scores `4,306` active banks (`26` flagged ≥ θ\*). Spark MLlib DEV ROC-AUC = **0.9388**. |
<!-- SYNC:END:README_BENCH -->

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
