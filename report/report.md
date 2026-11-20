# A Big-Data Pipeline for Bank Failure Early Warning: Storage Architecture, Point-in-Time Data Quality, and Cost-Sensitive Supervision

<!-- BLOCK:TITLEMETA:START -->
**Quản Hữu Nhật Minh**  
*School of Banking and Finance, National Economics University, Hanoi, Vietnam*  
Student ID: 11230998 | Course: BFT65B – Big Data Analytics in Finance | November 2026  
Repository: [https://github.com/Henryy2510/big-data-financial-analyst](https://github.com/Henryy2510/big-data-financial-analyst)
<!-- BLOCK:TITLEMETA:END -->

<!-- BLOCK:ABSTRACT:START -->
**Abstract**

**Purpose** – This study designs, benchmarks, and validates a reproducible point-in-time big-data pipeline for banking supervision using 42 years of U.S. FDIC Call Reports (1984Q1–2026Q2), answering how multi-decade regulatory panels should be stored and processed (Q1), how raw reporting defects can be remediated without look-ahead bias (Q2), and whether the output supports an honestly validated early-warning system (Q3).

**Design/methodology/approach** – We build a Medallion Lakehouse (Bronze, Silver, Gold) in Hive-partitioned Zstandard Parquet, implement automated data-quality audits and causal transformations in Polars, verify exact numerical parity against PySpark at equal 4-thread concurrency, and evaluate 26 CAMELS features under purged expanding walk-forward cross-validation in a development window (1984Q1–2007Q4) separated by a 4-quarter embargo (2008) from a single-use test partition (2009Q1–2025Q2).

**Findings** – Hive-partitioned Zstandard Parquet reduces storage by 2.76x (425.7 MB to 154.1 MB). Across 1x–8x scaling up to 13.46 million rows, Polars Streaming completes in 4.271 s (2,499 MB peak RSS; 101.0 MB per million rows) and runs 7.0x faster end-to-end than 4-thread PySpark at 4x scale (15.182 s vs. 2.173 s). Remediating eight structural data defects—including the post-2020 CBLR zero-as-missing regime affecting 37.15% of recent rows—cuts false-alarm episodes by 27%. On the held-out test set (485 failures across 400,332 bank-quarters), the champion HistGradientBoosting model attains a PR-AUC of 0.5478 (ROC-AUC of 0.9854) and flags 469 / 485 failing banks (96.70% recall, 297.0 days median lead) at the development-calibrated cost threshold ($\theta^* = 0.0039$).

**Regulatory implications** – Single-node columnar Polars suffices for bank-level Call Reports below 30–50 million rows; distributed Spark clusters become justified only when integrating granular loan-level credit registries or high-frequency interbank payment streams, such as Vietnam's CIC and CITAD systems under State Bank of Vietnam supervision.

**Keywords:** Bank failure early warning; Big-data lakehouse; Polars; PySpark; CAMELS; Point-in-time validation; Cost-sensitive thresholding  
**JEL Classification:** G21, G28, C53, C55
<!-- BLOCK:ABSTRACT:END -->

**Table 1.** Data acquisition, table row counts, temporal sample splits, and failure counts.

| Scope Element | Quantitative Specification |
| :--- | :--- |
| Acquisition & Date | FDIC BankFind Suite Public REST API (`https://api.fdic.gov/banks`) via resumable client (`scripts/00_download.py`); verified against API totals on 2026-11-05 |
| Raw Table Row Counts | Financials: 1,682,616 bank-quarters (170 quarters, 1984Q1–2026Q2, 24,064 CERTs, 0 duplicates); Failures: 4,117 resolutions (3,275 in 1984–2026); Institutions: 27,835 records |
| DEV & PURGE Splits | DEV (1984Q1–2007Q4): 1,224,815 rows (1,223,542 pit-scorable), 2,661 failing banks (1,522 in 5 walk-forward OOF blocks; 760,559 rows); PURGE (2008Q1–2008Q4): 33,966 rows (33,922 pit-scorable) |
| TEST & Unlabelled Tail | TEST (2009Q1–2025Q2): 400,698 rows (400,332 pit-scorable), 485 failing banks, 1,409 positive quarters (0.352%); POST\_SAMPLE (2025Q3–2026Q2): 17,501 rows (4,306 active banks scored at 2026-06-30) |

*Note(s):* Acquisition manifests and split invariants are archived in `manifest_financials.json`, `outputs/tables/gold_split_summary.csv`, and `outputs/tables/frozen_protocol.json`.

## 1. Introduction

Effective banking supervision depends on identifying fragile depository institutions early enough to intervene before capital depletion imposes severe losses on the deposit insurance fund. A long empirical literature demonstrates that accounting-based CAMELS ratios—measuring capital adequacy, asset quality, management efficiency, earnings, liquidity, and risk sensitivity—contain substantial predictive power for U.S. bank failures (Wheelock & Wilson, 2000; Shumway, 2001; Cole & White, 2012). In practice, however, moving from retrospective academic regressions to an operational central-bank early-warning system (EWS) is primarily a data-engineering and temporal-integrity problem. Over a 42-year reporting horizon, regulatory schemas evolve repeatedly, zero entries silently change meaning across legal regimes, income statements accumulate intra-year, mergers compete with failures as terminal exits, and Call Reports reach supervisors roughly two months after quarter-end. Any pipeline that overlooks these institutional mechanics risks severe look-ahead bias or systemic misclassification of healthy lenders.

Positioned from the perspective of a data engineer at Vietnam's central bank prototyping a supervisory pipeline on public U.S. Federal Deposit Insurance Corporation (FDIC) filings (1984Q1–2026Q2), this study designs, benchmarks, and audits an end-to-end Medallion Lakehouse architecture (Bronze, Silver, and Gold layers) implemented in Polars, PyArrow, PySpark, and scikit-learn. Rather than treating model accuracy as an isolated optimization target, the study investigates three interconnected operational questions: (Q1) how four decades of bank filings should be stored and processed on a laptop and when a distributed Spark cluster becomes economically necessary; (Q2) what structural quality defects exist in raw regulatory tables and how they can be corrected without look-ahead bias; and (Q3) whether the cleaned panel supports an honestly validated, cost-sensitive early-warning ranking under real-world supervisory budgets.

Three headline findings emerge from our measurements. First, columnar storage in Hive-partitioned Zstandard Parquet paired with Polars Streaming handles both the historical panel and replicated stress tests up to 13.46 million rows effortlessly on a single laptop, scaling at 101.0 MB per million rows and running 7.0x faster end-to-end than equal-core PySpark (`local[4]`) at 4x scale; distributed Spark clusters become justified only when expanding to borrower-level credit registries or high-frequency payment streams exceeding single-node RAM. Second, automated auditing uncovers eight systematic raw-data defects—most consequentially the post-2020 Community Bank Leverage Ratio (CBLR) zero-as-missing regime affecting 37.15% of recent rows and a 60-day publication lag under which 59.1% of failed banks' final filings are non-scorable ex ante. Third, evaluated strictly once on the held-out 2009Q1–2025Q2 test set (485 failures), a 26-feature HistGradientBoosting model delivers a PR-AUC of 0.5478 and ROC-AUC of 0.9854, catching 469 / 485 failing banks at a development-calibrated cost threshold ($\theta^* = 0.0039$) while clearly exposing its structural blind spot on sudden 2023 liquidity runs.

The remainder of this study is structured as follows. Section 2 details data acquisition, schema precision auditing, and lakehouse storage benchmarks. Section 3 presents the automated data-quality audit and point-in-time panel construction. Section 4 benchmarks Polars, Pandas, and PySpark across 1x–8x data scales and establishes the cluster-justification boundary. Section 5 describes leak-free CAMELS feature engineering and empirical anti-leakage verification. Section 6 reports walk-forward model comparison, cost-sensitive thresholding, held-out test evaluation, supervisory explainability, and post-hoc canary diagnostics. Section 7 synthesizes direct answers to Q1–Q3, states empirical limitations, and specifies institutional adaptations for the State Bank of Vietnam.

## 2. Data Acquisition and Lakehouse Storage Architecture

### 2.1 Field Selection and Reproducible Acquisition

The ingestion pipeline retrieves three relational FDIC tables linked by certificate number (CERT), summarized in Table 1: Financials (1,682,616 bank-quarter rows across 170 quarters from 1984Q1 through 2026Q2, comprising 3 identifiers and 30 CAMELS financial fields spanning capital, asset quality, liquidity, loan composition, and earnings), Failures (4,117 historical resolution records since 1934), and Institutions (27,835 structural bank records). Fields introduced only after 1994 or 2017—such as uninsured deposits (DEPUNINS) and held-to-maturity securities schedules (SCHA and SCHF)—are intentionally omitted from the uniform 42-year training schema to prevent historical availability bias across 1984–1993, a trade-off revisited in Section 7.2.

Acquisition is implemented as a resumable Python client (`scripts/00_download.py --download`, adapted from the course starter script). Each quarterly pull paginates deterministically by CERT, reconciles the returned row count against the API metadata total before atomically writing the quarterly file, logs completion in `manifest_financials.json`, skips already-verified quarters on restart, and retries transient HTTP 429 or 5xx responses with exponential backoff. Offline verification (`--verify-only`) confirms 170 of 170 complete quarters with zero duplicate (CERT, REPDTE) primary keys.

### 2.2 Storage Layout, Float64 Precision Audit, and Empirical Benchmarks

The Bronze layer (`data/bronze/financials`) enforces an explicit Apache Arrow schema: Int32 for CERT, Date32 for REPDTE, dictionary-encoded UTF-8 for state names (STNAME), and IEEE-754 Float64 for all monetary and ratio measures. Files are stored in Hive-style annual partitions (`YEAR=YYYY`), producing 43 compact directory files averaging roughly 3.6 MB per file rather than 170 sub-megabyte quarterly files that would inflate filesystem listing and Parquet footer overhead. Within each annual partition, rows are sorted by (REPDTE, CERT) with a 50,000-row group size and compressed using Zstandard level 3.

Retaining Float64 rather than downcasting monetary balances to Float32 is empirically required rather than merely conservative. Because IEEE-754 Float32 represents integers exactly only up to $2^{24} = 16,777,216$ (approximately $16.8 billion in thousands of dollars), our precision audit reveals that 11,485 ASSET rows (maximum $4.09 trillion), 8,706 DEP rows, and 1,652 EQ rows exceed exact Float32 integer resolution, introducing quantization rounding errors up to $128k on total assets and $120k on deposits (`outputs/tables/part1_float32_check.csv`).

<!-- SYNC:START:REPORT_PART1_TABLE -->
**Table 2.** Storage layout, compression ratio, and query-latency benchmark across 170 quarters of FDIC financials.

| Storage Format and Layout | Disk Size (MB) | Compression | Write (s) | Full Scan (s) | 3-Col Proj (s) | YEAR=2023 Filter (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Raw CSV (Uncompressed) | 425.7 | 1.00x | 0.770 | 0.683 | 0.220 | 0.988 |
| Single Parquet (Snappy) | 210.8 | 2.02x | 0.296 | 0.229 | 0.033 | 0.031 |
| Single Parquet (Zstd L3) | 153.8 | 2.77x | 0.589 | 0.210 | 0.037 | 0.036 |
| **Hive Parquet (YEAR=, Zstd L3, Chosen)** | **154.1** | **2.76x** | **0.922** | **0.233** | **0.042** | **0.040** |
| Pandas Baseline (read\_csv / read\_parquet) | 425.7 / 153.8 | 1.00x / 2.77x | — | 0.680 / 0.491 | — | — |

*Note(s):* All timings report wall-clock medians across repeated reads of the 1,682,616-row panel in Polars (4 threads) unless designated as Pandas. Source measurements are stored in `outputs/tables/part1_storage_benchmark.csv`.
<!-- SYNC:END:REPORT_PART1_TABLE -->

As reported in Table 2 and illustrated in Figure 1, Hive-partitioned Zstandard Parquet reduces disk footprint by 2.76x relative to raw CSV (154.1 MB vs. 425.7 MB) with negligible metadata overhead over single-file Zstandard (0.3 MB), while accelerating 3-column projections by 5.2x (0.042 s vs. 0.220 s) and single-year temporal slices by 24.7x (0.040 s vs. 0.988 s) through partition pruning and columnar projection pushdown.

![Storage layout, compression ratio, and query-latency benchmark across raw CSV, Snappy, Zstandard, and Hive-partitioned Parquet.](../outputs/figures/part1_storage_comparison.png)

## 3. Data Quality Audit and Point-in-Time Panel Construction

### 3.1 Automated Data Quality Audit Findings

Executing the automated audit (`scripts/02_quality.py`) generates a full diagnostic report (`outputs/dq_report/data_quality_report.md` and `outputs/tables/dq_summary.json`). Across the raw FDIC tables, we identify eight systematic structural defects—six table-level anomalies analyzed below alongside competing-exit ambiguity and publication-lag truncation in Section 3.2.

First, cross-quarter schema drift affects 14 numeric columns (LNLSGR, DEPDOM, NAASSET, RBCRWAJ, CHBAL, LNRE, P9ASSET, LNLSNET, NCLNLS, LNCON, LNCI, SC, P3ASSET, and DEP). Whenever a quarterly API response lacks fractional observations or omits a schedule, naive JSON-to-Parquet inference assigns Int64 or Null types instead of Float64, causing a direct wildcard scan (`pl.scan_parquet`) to fail with a schema mismatch error. Our Bronze ingestion eliminates this failure mode through deterministic per-file Arrow casting.

Second, missingness is strongly non-random across decades due to FFIEC Call Report form revisions rather than random non-response. As documented in Table 3 and plotted in Figure 2, 30–89 day past-due assets (P3ASSET) are 100.0% null prior to 2001, while 90+ day past-due assets (P9ASSET), nonaccrual assets (NAASSET), loan-loss provisions (ELNATR), and net charge-offs (NTLNLS) exhibit 18%–19% structural missingness during the 1984–1989 era.

**Table 3.** Structural missingness (%) of core FDIC Call Report variables across five-year regulatory eras.

| 5-Year Era | Panel Rows | P3ASSET (%) | P9ASSET (%) | NAASSET (%) | ELNATR (%) | NTLNLS (%) | RBC1AAJ (%) | EQ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1980–1984 | 71,760 | 100.0 | 19.2 | 19.2 | 18.5 | 18.5 | 0.0 | 0.2 |
| 1985–1989 | 347,654 | 100.0 | 18.4 | 18.4 | 7.6 | 18.2 | 0.0 | 0.2 |
| 1990–1994 | 286,105 | 100.0 | 0.0 | 0.0 | 0.3 | 0.3 | 0.0 | 0.3 |
| 1995–1999 | 225,775 | 100.0 | 0.0 | 0.0 | 0.3 | 0.3 | 0.0 | 0.3 |
| 2000–2004 | 192,271 | 21.1 | 0.0 | 0.0 | 0.2 | 0.2 | 0.0 | 0.2 |
| 2005–2009 | 172,995 | 0.0 | 0.0 | 0.0 | 0.2 | 0.2 | 0.0 | 0.2 |
| 2010–2014 | 145,726 | 0.0 | 0.0 | 0.0 | 0.2 | 0.2 | 0.0 | 0.2 |
| 2015–2019 | 116,869 | 0.0 | 0.0 | 0.0 | 0.3 | 0.3 | 0.0 | 0.3 |
| 2020–2024 | 96,902 | 0.0 | 0.0 | 0.0 | 0.4 | 0.4 | 0.0 | 0.4 |
| 2025–2026 | 26,559 | 0.0 | 0.0 | 0.0 | 0.4 | 0.4 | 0.0 | 0.4 |

*Note(s):* Values report the share of null entries per era across all 1,682,616 raw financial rows. Zero null rates for RBC1AAJ in 1984–1989 mask structural zero-as-missing coding analyzed in Section 3.1 (`outputs/tables/dq_summary.json`).

![Structural missingness of core Call Report fields across regulatory eras (1984–2026).](../outputs/figures/missingness_by_era.png)

Third, capital ratios exhibit two distinct zero-as-missing regimes (`outputs/tables/dq_zero_as_missing.csv`). Prior to the Basel I phase-in (1984–1989), total risk-based capital (RBCRWAJ) is coded as zero in 100.0% of rows and Tier 1 leverage (RBC1AAJ) is coded as zero in 18.13% of rows; after 1989, zero RBC1AAJ appears in 3,458 rows, of which 3,369 have null equity. Even more critically for modern supervision, zero-coded RBCRWAJ jumps from 0.22%–0.31% during 1990–2019 to 37.15% (45,865 of 123,461 rows) during 2020–2026, with 45,423 of those rows (36.79%) simultaneously reporting healthy positive Tier 1 leverage. Since 2020, qualifying community banks opting into the Community Bank Leverage Ratio (CBLR) framework are legally exempt from risk-weighted asset schedules; per the FDIC RISView API specification (`risview_properties.yaml`, `x-source-mapping`), API-derived percentage ratios default to literal zero rather than null whenever denominators are null, non-positive, or exempt. Left untreated, a tree model misclassifies over one-third of healthy modern banks as capital-depleted, inflating held-out false-alarm episodes from 2,707 to 3,721. Our feature pipeline (`src/bfews/features.py`) therefore recodes RBC1AAJ zeros (when pre-1990 or equity is null) and RBCRWAJ zeros (when pre-1990, equity is null, RBC1AAJ is at least 8%, or post-2020 with positive RBC1AAJ) to null while preserving genuine insolvency zeros where equity is negative.

Fourth, all seven income and expense flow variables are reported cumulatively year-to-date (`outputs/tables/dq_ytd_monotonicity.csv`). Raw median interest income scaled by assets rises nearly fourfold across fiscal quarters—from 0.0177 in Q1 to 0.0352 in Q2, 0.0526 in Q3, and 0.0689 in Q4—whereas annualized ROA remains flat across quarters (0.971%, 0.989%, 0.993%, and 0.944%). Leaving flows cumulative would confound quarterly seasonal effects with genuine profitability shocks; our Silver panel builder (`src/bfews/panel.py`) therefore de-cumulates flows strictly across consecutive same-year filings via a guarded quarter-continuity check (`ytd_prev_ok`), leaving non-consecutive gaps as null.

Fifth and sixth, raw balance sheets contain both negative stock anomalies and negative-equity denominator inversions. Specifically, we observe negative balances in ORE (646 rows), securities SC (96 rows), loan-loss allowances LNATRES (9 rows), noncurrent loans NCLNLS (3 rows), and cash balances CHBAL (2 rows), alongside 761 rows with gross loans exceeding total assets and 6,236 rows where NAASSET exceeds NCLNLS plus ORE. Negative stock balances are sanitized to null prior to ratio construction. Separately, negative equity occurs in 15,659 rows (0.931%, accompanied by 6,204 rows where deposits exceed assets); because negative equity inverts a naive Texas Ratio into a misleading negative number—making deeply insolvent banks look safer than healthy peers—we cap the Texas Ratio at 10.0 whenever the denominator is non-positive and bound continuous extremes via strictly in-fold winsorization.

### 3.2 Point-in-Time Exit Classification and Publication Lag Gate

Every distinct bank in the 42-year panel (24,064 CERTs) is assigned a terminal exit state and reconciled against FDIC CHANGEC1 closure codes in Table 4. Unassisted mergers and voluntary liquidations (16,892 banks, 70.20%) are treated as right-censored non-failures ($y_{i,q}=0$), while FDIC receiverships and open-bank assistance transactions (2,859 banks matched to the financial panel, 11.88%) define the forward failure outcome $y_{i,q}=1$ whenever failure occurs within the subsequent four quarters, $(q, q+4]$.

**Table 4.** Point-in-time bank exit taxonomy and FDIC CHANGEC1 resolution code reconciliation.

| Terminal Exit Status | Distinct Banks | Bank Share | Panel Rows | FDIC CHANGEC1 Codes and Label Treatment |
| :--- | :---: | :---: | :---: | :--- |
| MERGED\_OR\_OTHER | 16,892 | 70.20% | 923,166 | Unassisted mergers and voluntary exits (223: 14,183; 221: 808; 240: 504); right-censored ($y_{i,q}=0$) |
| ACTIVE (in 2026) | 4,313 | 17.92% | 671,777 | Active surviving institutions ($y_{i,q}=0$) |
| FAILED (FDIC Resolution) | 2,859 | 11.88% | 87,673 | Failures, assisted mergers, and payoffs (211: 2,393; 213: 337; 230: 191); $y_{i,q}=1$ in $(q, q+4]$ |
| **Total Panel** | **24,064** | **100.00%** | **1,682,616** | Institutions snapshot used solely for terminal audit, never as historical predictors |

*Note(s):* Counts reconcile `outputs/tables/bank_exit_summary.csv` and `outputs/tables/exit_changec1_codes.csv`.

Two additional point-in-time safeguards eliminate survival and timing leakage. First, across the 4,117 raw resolution records (3,524 formal failures and 593 assistance transactions), 44 CERTs appear multiple times (37 assistance-then-failure, 4 repeated assistance, and 3 repeated failure). Retaining each bank's earliest resolution date and removing all filings with $\text{REPDTE} \ge \text{FAILDATE}$ drops 5,636 post-event rows (0.335% across 471 banks: 4,428 following assistance and 1,208 on or after formal closure), leaving 1,676,980 strictly pre-event Gold observations. Second, Call Reports become publicly available approximately 60 days after quarter-end ($\text{available\_date} = \text{REPDTE} + 60\text{ days}$). Among failed banks, 59.1% failed within 60 days of their last filed quarter (median gap of 49 days); those filings could not have informed an ex-ante supervisory intervention and are marked non-scorable (`pit_scorable = False`), excluding them from both model fitting and performance evaluation.

## 4. Scalable Processing: Polars, Pandas, and PySpark

### 4.1 Query Optimization in Polars

Comparing a naive unprojected pipeline against our optimized lazy execution plan (`scripts/05_benchmark.py`; `outputs/tables/polars_naive_vs_optimized_plan.txt`) highlights two primary sources of speed and memory efficiency. First, partition pruning and projection pushdown eliminate wasted I/O: the naive plan reads all 34 columns across all 43 annual partitions (1,717,675 estimated rows) and sorts before filtering, whereas the optimized plan pushes the temporal filter ($\text{YEAR} \ge 2008$) directly into the Parquet scanner—skipping 24 annual directories and reading only 758,974 rows across 9 projected columns. Second, Polars performs automatic common subexpression elimination (extracting $\text{EQ} + \text{LNATRES}$ once) and evaluates (CERT, YEAR) and (CERT) window operators in parallel across Arrow buffers using 4 Rust worker threads.

<!-- SYNC:START:REPORT_PART3_SECTION -->
### 4.2 Multi-Scale Benchmark (1x–8x, Up to 13.46 Million Rows)

To avoid unreliable linear extrapolation from sub-1.7-million-row slices, we benchmark Polars, Pandas, and PySpark on disjoint replicated panels scaled from 1x (1.68 million rows) up to 8x (13,460,928 rows) inside isolated `spawn` subprocesses on a Linux workstation equipped with 6 physical (12 logical) CPUs and 15.34 GB RAM (Polars 2.1.0 capped at 4 threads; pandas 3.1.1). For an equal-core comparison against 4-thread Polars, PySpark is benchmarked at `local[4]` alongside the 2-thread `local[2]` laptop-memory baseline.

**Table 5.** Multi-scale processing runtime and peak memory benchmark across Polars, Pandas, and PySpark (1x–8x).

| Scale (Rows) | Polars Streaming (4t) | Polars Lazy (4t) | Polars Eager (4t) | Pandas + Pushdown | PySpark `local[4]` (4t) *(vs. `local[2]`)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1x (1.68M) | 0.619 s / 1,268 MB | 1.326 s / 777 MB | 1.281 s / 711 MB | 0.482 s / 1,044 MB *(naive: 0.671 s / 1,940 MB)* | 10.551 s (4.149 s JVM) / 1,262.9 MB *(`local[2]`: 10.617 s / 1,143.2 MB)* |
| 2x (3.37M) | 1.284 s / 1,532 MB | 2.665 s / 1,072 MB | 2.656 s / 1,029 MB | 0.882 s / 1,544 MB | — |
| 4x (6.73M) | **2.173 s / 1,734 MB** | 5.454 s / 1,613 MB | 5.254 s / 1,710 MB | 1.701 s / 2,689 MB | 15.182 s (11.117 s compute) / 1,807.8 MB *(`local[2]`: 16.163 s / 1,770.3 MB)* |
| 8x (13.46M) | **4.271 s / 2,499 MB** | 10.406 s / 2,569 MB | *(skipped OOM risk)* | *(skipped eager overhead)* | — |

*Note(s):* Cells report wall-clock runtime / peak RSS (Python+JVM process-tree RSS for PySpark). Maximum relative checksum difference across all Polars, Pandas, and PySpark runs is $\le 4.9 \times 10^{-16}$. Sources: `outputs/tables/part3_benchmark_results.csv`, `spark_scaling_results.csv`, and `benchmark_environment.json`.

A notable crossover appears in Table 5 and Figure 3: at 1x–4x on pre-sorted 8-column slices, Pandas with manual column pushdown edges Polars Streaming on raw wall time (0.482 s vs. 0.619 s at 1x; 1.701 s vs. 2.173 s at 4x) because PyArrow loads compact pre-sorted slices directly into contiguous NumPy arrays without morsel-scheduling overhead. Yet scaling to 4x–8x exposes why memory architecture governs laptop capacity (`outputs/tables/part3_capacity_extrapolation.json`). Because Pandas materializes full $O(N)$ intermediate column copies during `groupby` and `shift` transformations, its peak RSS grows at 327.9 MB per million rows—already consuming 55% more memory than Polars Streaming at 4x (2,689 MB vs. 1,734 MB) and exhausting 16 GB RAM by roughly 48.5 million rows. By contrast, Polars Streaming executes pipelined Rust operators over bounded morsels, flattening peak-memory growth to 101.0 MB per million rows (a 3.25x shallower slope; 0.305 s per million rows), raising the extrapolated single-laptop ceiling to 151.1 million rows inside 16 GB RAM.

![Multi-scale execution time and peak RSS memory scaling across Polars, Pandas, and PySpark (1x–8x).](../outputs/figures/part3_scaling_benchmark.png)

### 4.3 PySpark Reproduction, Parity Verification, and Cluster Justification

In `scripts/06_spark.py` (1.2 GB driver memory), PySpark Catalyst reproduces Polars outputs identically across all 43 annual partitions, achieving a 0.0 maximum absolute difference across 33,042 sampled rows and a $7.4 \times 10^{-15}$ maximum relative difference in yearly aggregate checksums (`outputs/tables/spark_polars_parity.json`). Inspecting the Catalyst physical plan (`outputs/tables/spark_execution_plan.txt`) explains why Spark lags on single-node workloads: Catalyst inserts two full `Exchange` shuffle barriers—first hash-partitioning by (CERT, YEAR) for YTD de-cumulation, then repartitioning by (CERT) for 4-quarter lags—serializing intermediate state across JVM boundaries. Even on an equal 4-thread footing, PySpark `local[4]` incurs 4.065 s of JVM startup overhead and requires 15.182 s total (11.117 s compute, 1,807.8 MB Python+JVM tree RSS) at 6.73 million rows compared with 2.173 s in 4-thread Polars Streaming—7.0x slower end-to-end and 5.1x slower compute-only, while `local[2]` requires 16.163 s and 1,770.3 MB tree RSS. Doubling Spark's local thread count trims total latency by only 6.1%, confirming that JVM startup and shuffle barriers rather than core count dominate local latency. Consequently, a distributed Spark cluster is unwarranted for quarterly bank-level Call Reports below 30–50 million rows and becomes justified only when wide multi-schedule expansions or borrower-level credit registries exceed single-node memory or require fault-tolerant disk-spilling execution.
<!-- SYNC:END:REPORT_PART3_SECTION -->

## 5. Leak-Free CAMELS Feature Engineering

### 5.1 Supervisory Feature Design

Guided by the U.S. Uniform Financial Institutions Rating System (CAMELS) and bank-failure econometrics (Wheelock & Wilson, 2000; Cole & White, 2012), we construct 26 strictly point-in-time features (`src/bfews/features.py`) covering capital adequacy, asset quality, management efficiency, earnings, liquidity, sectoral sensitivity, institution size, and 4-quarter balance-sheet momentum. Every ratio incorporates the post-2020 CBLR zero-as-missing remediation and negative-stock sanitization documented in Section 3.1. Table 6 summarizes the complete feature inventory.

**Table 6.** Point-in-time CAMELS feature inventory and supervisory rationale (26 features).

| CAMELS Pillar | Features Computed at Quarter $q$ | Economic and Supervisory Rationale |
| :--- | :--- | :--- |
| Capital (C) | `eq_asset_ratio`, `rbc1aaj_clean`, `rbcrwaj_clean`, `cap_distress_flag` ($\text{EQ}<0 \lor \text{EQ/ASSET}<2\%$) | Leverage and Basel capital buffers after recoding CBLR and pre-Basel zeros to null |
| Asset Quality (A) | `nclnls_ratio`, `naasset_ratio`, `ore_ratio`, `lnatres_ratio`, `texas_ratio` (capped at 10.0) | Nonperforming loan overhang and reserve coverage; non-positive equity denominators capped |
| Management (M) | `efficiency_ratio` ($\text{NONIX}/(\text{INTINC}-\text{EINTEXP}+\text{NONII})$) | Operating cost discipline relative to net operating revenue |
| Earnings (E) | `roa_ytd`, `nimy_clean`, `provision_asset_ratio`, `netinc_q_asset_ratio` | Annualized ROA and net interest margin paired with de-cumulated quarterly income and provisions |
| Liquidity (L) | `liquid_asset_ratio`, `brokered_dep_ratio`, `loan_to_deposit_ratio` | Liquid cash-and-securities cushion versus wholesale brokered funding reliance |
| Sensitivity (S) | `re_loan_share`, `ci_loan_share`, `con_loan_share` | Sectoral real-estate, commercial-and-industrial, and consumer credit concentration |
| Size and Dynamics | `log_asset`, `asset_growth_4q`, `loan_growth_4q`, `d4_texas_ratio`, `d4_nclnls_ratio`, `d4_eq_ratio` | Institution scale and guarded 4-quarter changes (`lag4_ok` prevents crossing reporting gaps) |

*Note(s):* All 26 features are computed exclusively from filings at or before quarter $q$ (`<= REPDTE`) in `src/bfews/features.py`.

### 5.2 Empirical Anti-Leakage Verification

Every panel lag and financial ratio uses strictly historical rows at or before the reporting quarter, and all ex-post exit metadata (ACTIVE, CHANGEC1, FAILDATE, COST) are structurally excluded via an explicit forbidden-column list. Within each walk-forward training fold, the scikit-learn preprocessing chain—RobustWinsorizer (1st–99th percentiles), median SimpleImputer with missingness indicators, StandardScaler, and the Elkan (2001) PriorOddsCalibrator—is fitted exclusively on training rows.

Quantifying what breaks under naive full-sample preprocessing demonstrates why strict in-fold fitting is not a formality (`outputs/tables/leak_parameter_shift.csv`). Fitting winsorizers and imputers across the combined 1984–2025 panel leaks four decades of secular banking trends (higher post-Basel capital ratios and modern wholesale funding structures), shifting 99th-percentile clip bounds by 20.95% for `brokered_dep_ratio`, 16.19% for `eq_asset_ratio`, and 16.00% for `loan_growth_4q`, while shifting imputation medians by 43.04% for `d4_texas_ratio` and 23.48% for `con_loan_share`. Automated pytest suites (`tests/test_pipeline.py`) verify causal truncation invariance, strict inter-fold purge gaps, and bit-exact preprocessor invariance even when held-out test features are artificially corrupted with values of $10^9$.

## 6. Modelling, Temporal Validation, and Cost-Sensitive Thresholding

### 6.1 Temporal Split Protocol and Walk-Forward Validation

We enforce a two-level temporal validation protocol (`config.yaml` and `src/bfews/modeling.py`) aligned with real-time supervisory deployment across distinct credit cycles. First, the development partition (DEV, 1984Q1–2007Q4) comprises 1,223,542 pit-scorable rows across 2,661 failing banks (8,844 pre-failure quarters), spanning the 1984–1993 savings-and-loan and commercial banking crisis. All model selection, hyperparameter choices, and threshold calibration occur exclusively within DEV via five expanding-window out-of-time validation blocks (1988, 1990, 1992, 1994–1999, and 2000–2007), each preceded by a 4-quarter purge gap between training and validation. Pooled across the five blocks, the out-of-fold (OOF) set contains 760,559 bank-quarters (3,547 positive quarters) and 1,522 failing banks; the remaining 1,139 DEV failures fall in the initial 1984–1987 warm-up window or inside the 1-year inter-fold purge years (1989, 1991, and 1993).

Second, a full 4-quarter embargo (PURGE, 2008Q1–2008Q4, 33,922 pit-scorable rows) separates DEV from the held-out test partition (TEST, 2009Q1–2025Q2, 400,332 pit-scorable rows, 485 failing banks, 1,409 pre-failure quarters), guaranteeing zero 4-quarter forward label overlap across the split boundary. Placing the 2009–2014 Global Financial Crisis inside TEST rather than DEV provides a large, demanding out-of-time failure sample (485 banks) while forcing the model to generalize across distinct regulatory regimes. The complete DEV-only specification—splits, 26 features, forbidden columns, champion rule, cost parameters, and $\theta^* = 0.0039$—is recorded as an auditable configuration-integrity manifest in `outputs/tables/frozen_protocol.json` (SHA-256 fingerprints of the DEV specification and `test_results.json`, verified by pytest; the manifest documents configuration integrity rather than proving pre-registration, while single-test discipline rests on DEV-only selection code and automated leakage tests).

### 6.2 Model Comparison, Non-Linear Interactions, and PySpark MLlib Check

Table 7 reports pooled out-of-fold walk-forward validation metrics across four progressively richer specifications inside DEV. Two structural patterns stand out. First, adding 23 additional CAMELS ratios to a regularized linear model actually *reduces* out-of-fold PR-AUC from 0.4015 (3-variable logistic regression) to 0.3601 (26-variable L2 logistic regression). Econometrically, bank balance-sheet ratios exhibit strong multicollinearity and sharp threshold non-linearities—for example, rapid loan growth or brokered-deposit reliance threatens solvency primarily when capital cushions are thin and noncurrent loans are accelerating. An additive linear decision boundary cannot represent these conditional interactions, so L2 shrinkage across 26 correlated inputs dilutes the core capital, delinquency, and earnings signals. Second, allowing depth-5 splits in HistGradientBoosting captures those non-linear threshold interactions directly, lifting OOF PR-AUC to 0.4624 (a +15.2% relative gain over 3-variable logistic regression and +75.0% over the univariate Texas Ratio rule of 0.2642) while lowering the Brier score to 0.00337.

**Table 7.** Out-of-fold walk-forward validation performance across candidate model architectures (DEV, 1984Q1–2007Q4).

| Model Architecture | Features Used | OOF PR-AUC | OOF ROC-AUC | Recall @ Top 1% | Recall @ Top 5% | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Texas ratio rule | 1 (Texas Ratio rank) | 0.2642 | 0.9734 | 0.6665 | 0.9290 | 0.00724 |
| Logistic, 3 variables | 3 (equity, NCLNLS, ROA) | 0.4015 | 0.9786 | 0.7277 | 0.9566 | 0.00368 |
| Logistic, 26 CAMELS | 26 CAMELS + indicators (L2, $C=0.1$) | 0.3601 | 0.9784 | 0.7291 | 0.9352 | 0.00393 |
| **HistGradientBoosting, 26 CAMELS** | **26 CAMELS (max\_depth=5, lr=0.05)** | **0.4624** | **0.9851** | **0.7753** | **0.9504** | **0.00337** |

*Note(s):* Metrics are evaluated on pooled out-of-fold predictions across 5 purged walk-forward blocks (760,559 bank-quarters; 1,522 failing banks). Source: `outputs/tables/cv_model_comparison.csv`.

Two auxiliary checks within DEV confirm robustness. First, restricting positive labels in DEV strictly to formal closures (excluding open-bank assistance transactions, reducing OOF failures from 1,522 to 1,361) yields an OOF PR-AUC of 0.3878, ROC-AUC of 0.9794, and Brier score of 0.00334 (`outputs/tables/dev_label_sensitivity.csv`), confirming ranking stability while retaining FDIC's standard supervisory definition (the 2009–2025 test set contains zero assistance transactions). Second, a PySpark MLlib logistic regression demonstration (`src/bfews/spark_jobs.py`) achieves a development validation ROC-AUC of 0.9388 and PR-AUC of 0.1893, matching scikit-learn (0.9394 ROC-AUC and 0.1896 PR-AUC). Figure 4 contrasts held-out precision-recall performance against the DEV expected-cost optimization curve.

![Held-out test precision-recall curves (left) and development out-of-fold expected supervisory cost curve (right).](../outputs/figures/pr_and_cost_curves.png)

### 6.3 Dual Supervisory Operating Regimes and Cost-Optimal Thresholding

In operational banking supervision, early-warning scores support two complementary triage regimes: fixed quarterly examination capacity budgets (inspecting the highest-risk 1%, 2%, or 5% of institutions each quarter) and an expected-cost minimizing probability threshold ($\theta^*$). Under the cost-sensitive formulation, we minimize expected supervisory loss on the 760,559 DEV out-of-fold rows (1,522 failing banks) over a 300-point log-spaced grid $\theta \in [10^{-4}, 0.90]$:

$$\mathcal{L}(\theta) = \sum_{b \in \text{FN}(\theta)} \ell_{\text{DIF}} \cdot \min(\text{ASSET}_b,\, A_{0.85}) \;+\; C_{\text{FP}} \cdot \text{FP}_{\text{episodes}}(\theta),$$

where $\ell_{\text{DIF}} = 0.1961$ is the empirical Deposit Insurance Fund loss rate estimated from historical failures, $A_{0.85} = \$474,926\text{k}$ is the 85th-percentile community-and-regional bank asset cap (preventing extreme mega-bank outliers from distorting community-bank calibration), and $C_{\text{FP}} = \$250\text{k}$ is the baseline cost of an unnecessary off-cycle examination episode. The unique interior minimum occurs at $\theta^* = 0.0039$ (expected DEV loss of $5,418,988k), capturing 1,434 of 1,522 DEV OOF failing banks (88 missed, 8,083 false-alarm episodes) and reducing expected supervisory loss by 88.56% relative to a no-alarm policy ($47,352.9M) and 86.61% relative to a naive 0.50 cutoff ($40,460.8M, which saves only 14.55% under 0.47% base-rate imbalance).

Because $\mathcal{L}(\theta; C_{\text{FP}}) = M(\theta) + C_{\text{FP}} \cdot E(\theta)$ is linear in $C_{\text{FP}}$, decomposing the DEV OOF loss curves (`outputs/tables/cfp_stability_interval.json`) separates missed-failure DIF loss $M(\theta)$ from false-alarm episodes $E(\theta)$ with 0.0 MUSD rounding error and proves that $\theta^* = 0.0039$ remains strictly optimal across the entire 2.25x interval $C_{\text{FP}} \in [\$231.1\text{k},\, \$519.9\text{k}]$. Relative to the empirical DEV OOF mean DIF loss per failing bank ($31.11M across 1,522 failures), this stability band corresponds to an implied false-negative to false-positive cost ratio of 60x–135x (124.4x at $250k). As an external institutional anchor, the FDIC Division of Risk Management Supervision's 2024 ongoing operating budget ($703.0M) across 1,200 Risk Management safety-and-soundness examinations implies a gross upper bound of $585.9k per full examination before netting specialty exams and off-site monitoring (FDIC, 2024, 2025), placing a targeted off-cycle review squarely inside the $250k–$500k band where $\theta^*$ is invariant.

**Table 8.** Sensitivity of the DEV-optimal probability threshold to the false-alarm episode cost ($C_{\text{FP}}$).

| False-Alarm Cost ($C_{\text{FP}}$) | Optimal $\theta^*$ (DEV) | Grid Edge? | DEV Missed (/ 1,522) | DEV FP Episodes | TEST Recall (/ 485) | TEST Flagged Rows | TEST Median Alerts / Qtr |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $100k | 0.0009 | False | 48 | 16,815 | 97.32% | 6.128% | 139.0 |
| **$250k (Baseline)** | **0.0039** | **False** | **88** | **8,083** | **96.70%** | **2.883%** | **43.0** |
| $500k | 0.0039 | False | 88 | 8,083 | 96.70% | 2.883% | 43.0 |
| $1,000k | 0.0139 | False | 150 | 4,383 | 95.67% | 1.700% | 26.5 |

*Note(s):* Table 8 evaluates rounded four-decimal grid cutoffs ($\theta = 0.0039 \rightarrow 2.883\%$), whereas Table 10 evaluates the exact unrounded log-grid optimum ($\theta^* = 0.003864 \rightarrow 2.894\%$, differing by 43 of 400,332 test rows). Source: `outputs/tables/test_results.json`.

### 6.4 Held-Out Test Evaluation and Operational 2026Q2 Watchlist

Evaluating the frozen DEV-trained HistGradientBoosting champion once on the 400,332 held-out test bank-quarters (2009Q1–2025Q2; 485 failing banks and 1,409 pre-failure quarters) confirms strong out-of-time generalization. Table 9 compares overall and regime-specific discrimination against the Texas Ratio baseline, while Table 10 and Figure 5 summarize supervisory operating trade-offs across fixed capacity budgets and the cost-optimal threshold.

**Table 9.** Held-out test evaluation and crisis-regime subperiod breakdown (2009Q1–2025Q2).

| Evaluation Partition and Model | Sample Rows | Failing Banks | Pre-Fail Qtrs | PR-AUC | ROC-AUC | Recall @ Top 1% | Recall @ Top 5% | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Test Set — Texas Ratio Baseline | 400,332 | 485 | 1,409 | 0.1507 | 0.9629 | 0.5877 | 0.9163 | 0.00625 |
| **Test Set — Champion HistGradientBoosting** | **400,332** | **485** | **1,409** | **0.5478** | **0.9854** | **0.8353** | **0.9560** | **0.00238** |
| Subperiod 1: GFC (2009–2014) | 177,923 | 453 | 1,303 | 0.5847 | 0.9881 | 0.6708 | 0.9601 | 0.00481 |
| Subperiod 2: Modern and SVB (2015–2025) | 222,409 | 35 | 106 | 0.1708 | 0.9193 | 0.6698 | 0.7547 | 0.00043 |

*Note(s):* Three banks contribute pre-failure observations spanning the 2014/2015 subperiod boundary, so subperiod failing-bank counts sum to 488 versus 485 unique failing banks overall. Source: `outputs/tables/test_results.json`.

**Table 10.** Held-out test supervisory operating points: fixed quarterly capacity budgets versus the cost-optimal threshold.

| Supervisory Operating Rule | Flagged Rows | Median Alerts / Qtr | Banks Caught (/ 485) | Bank Recall | Row Recall (/ 1,409) | Row Precision | FP Episodes (Rows) | Median Lead (Days) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Capacity Budget: Top 1% / Quarter | 1.000% | 58.5 | 399 / 485 | 82.27% | 59.55% (839) | 20.95% | 1,080 (3,165) | 210.0 |
| Capacity Budget: Top 2% / Quarter | 2.000% | 117.5 | 443 / 485 | 91.34% | 78.78% (1,110) | 13.86% | 2,157 (6,896) | 286.0 |
| **Cost-Optimal Threshold ($\theta^* = 0.0039$)** | **2.894%** | **43.0** | **469 / 485** | **96.70%** | **94.39% (1,330)** | **11.48%** | **2,707 (10,257)** | **297.0** |
| Capacity Budget: Top 5% / Quarter | 5.000% | 294.0 | 466 / 485 | 96.08% | 92.62% (1,305) | 6.52% | 5,601 (18,713) | 297.0 |

*Note(s):* Dynamic thresholding at $\theta^* = 0.0039$ concentrates inspection workload during crisis peaks (2009–2012) while keeping median quarterly alerts (43.0) below the Top-1% fixed-budget workload (58.5). Source: `outputs/tables/test_results.json`.

Table 10 and Figure 6 (left) highlight an important macroprudential contrast between fixed-budget and calibrated-threshold triage. Although $\theta^* = 0.0039$ flags a larger pooled share of rows over the 16-year test horizon (2.894% vs. 1.000%), its *median* quarterly workload is actually lower than the Top-1% rule (43.0 vs. 58.5 alerts per quarter) while lifting bank recall from 82.27% to 96.70% and median warning lead time from 210.0 to 297.0 days. Fixed percentage quotas force supervisors to inspect a constant fraction of the industry even during benign expansions (2015–2022), dissipating examination capacity on low-risk banks while capping coverage during systemic waves. Because our probabilities are calibrated to historical prior odds (Elkan, 2001), $\theta^*$ adapts countercyclically—surging above 400 alerts per quarter during the 2009–2011 GFC peak and contracting sharply during calm periods.

![Held-out test operating points across capacity budgets and crisis regime subperiods.](../outputs/figures/operating_points.png)

![Quarterly supervisory alerts over time (left) and risk trajectories for major historical failure cases (right).](../outputs/figures/alarms_and_canary_trajectories.png)

As a live operational demonstration on the right-censored tail (`outputs/tables/watchlist_2026q2.csv`), scoring all 4,306 active banks filing at 2026-06-30 with the frozen DEV-trained champion—without reusing test outcomes—flags 26 institutions above $\theta^* = 0.0039$ (cross-sectional mean predicted probability 0.00054, maximum 0.4253). The five highest-ranked institutions exhibit elevated Texas Ratios, thin GAAP equity, or heavy wholesale brokered funding: Tioga-Franklin Savings Bank (PA, $68.4M assets, $\hat{p}=0.4253$, Texas Ratio 2.570, equity 2.04%), Lamont Bank of St. John (WA, $51.1M, $\hat{p}=0.4030$, Texas Ratio 5.191, equity 1.80%), Summit National Bank (WY, $65.9M, $\hat{p}=0.2026$, equity 2.40%, brokered deposits 34.40%), Columbia Savings and Loan Association (WI, $20.6M, $\hat{p}=0.1890$, Texas Ratio 2.311), and Nano Banc (CA, $736.2M, $\hat{p}=0.1565$, Texas Ratio 2.098, brokered deposits 22.79%).

### 6.5 Supervisory Explainability: Top Ten CAMELS Risk Drivers

To verify that tree rankings reflect economically coherent solvency channels strictly without touching TEST, we compute weighted permutation importance ($\Delta\text{PR-AUC}$) on DEV walk-forward folds alongside standardized logistic regression coefficients ($\hat{\beta}$). Table 11 and Figure 7 show clear alignment with banking theory: nonperforming-asset overhang relative to loss-absorbing buffers (`texas_ratio`, $\Delta\text{PR-AUC} = 0.26242 \pm 0.00804$), rapid four-quarter equity erosion (`d4_eq_ratio`, $0.06486$), and thin book equity (`eq_asset_ratio`, $0.06167$, $\hat{\beta} = -1.1611$) dominate risk rankings, followed by Tier 1 leverage, earnings (`roa_ytd`, `nimy_clean`), wholesale brokered deposit reliance (`brokered_dep_ratio`, $\hat{\beta} = +0.2247$), and real-estate owned overhang (`ore_ratio`).

**Table 11.** Top ten CAMELS risk drivers ranked by strictly-DEV walk-forward permutation importance and standardized logistic coefficients.

| Rank | Feature (CAMELS Dimension) | HGB Perm. Importance ($\Delta\text{PR-AUC} \pm \text{SD}$) | Logit Std. Coef. ($\hat{\beta}$) | Econometric Direction |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `texas_ratio` (Asset Quality) | 0.26242 $\pm$ 0.00804 | +0.0606 | Increases risk |
| 2 | `d4_eq_ratio` (Size and Dynamics) | 0.06486 $\pm$ 0.00794 | -0.1646 | Reduces risk |
| 3 | `eq_asset_ratio` (Capital) | 0.06167 $\pm$ 0.00967 | -1.1611 | Reduces risk |
| 4 | `rbc1aaj_clean` (Capital) | 0.02975 $\pm$ 0.00717 | -0.7788 | Reduces risk |
| 5 | `roa_ytd` (Earnings) | 0.02555 $\pm$ 0.00174 | -0.0260 | Reduces risk |
| 6 | `nimy_clean` (Earnings) | 0.01491 $\pm$ 0.00173 | -0.1243 | Reduces risk |
| 7 | `brokered_dep_ratio` (Liquidity) | 0.00825 $\pm$ 0.00145 | +0.2247 | Increases risk |
| 8 | `ore_ratio` (Asset Quality) | 0.00646 $\pm$ 0.00215 | +0.2290 | Increases risk |
| 9 | `naasset_ratio` (Asset Quality) | 0.00546 $\pm$ 0.00176 | +0.0979 | Increases risk |
| 10 | `rbcrwaj_clean` (Capital) | 0.00466 $\pm$ 0.00390 | -0.0926 | Reduces risk |

*Note(s):* Permutation importance is averaged across DEV walk-forward out-of-fold validation blocks using 5 repeats per block (`outputs/tables/feature_importance.csv`).

![Top ten CAMELS risk drivers measured by strictly-DEV walk-forward permutation importance.](../outputs/figures/feature_importance.png)

### 6.6 Post-Hoc Historical Canary Case Studies

As a purely diagnostic post-hoc inspection of frozen test-set scores (`outputs/tables/canary_case_studies.csv`), Table 12 examines the five largest FDIC receivership closures by total assets during the Global Financial Crisis (2009–2014) and the five largest closures during the modern era (2015–2025).

**Table 12.** Post-hoc historical canary diagnostics for the five largest GFC and five largest modern FDIC failures.

| Bank Name (CERT, State) | Regime / Failure Date | Assets ($M) | Peak $\hat{p}$ (4Q) | Flagged at $\theta^*$ (Lead) | Top 1% (Lead) | Peak Texas / Min Eq% / Max Bro% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Colonial Bank (9609, AL) | GFC / 2009-08-14 | 25,455.1 | 0.0174 | Yes (2009-03-31, 136 d) | No | 0.572 / 5.60% / 12.51% |
| Guaranty Bank (32618, TX) | GFC / 2009-08-21 | 13,464.4 | 0.5524 | Yes (2009-03-31, 143 d) | Yes (143 d) | 10.000 / -5.60% / 4.49% |
| AmTrust Bank (29776, OH) | GFC / 2009-12-04 | 11,439.0 | 0.3743 | Yes (2009-03-31, 248 d) | Yes (157 d) | 2.677 / 2.42% / 4.11% |
| United Commercial (32469, CA) | GFC / 2009-11-06 | 10,895.3 | 0.0363 | Yes (2009-06-30, 129 d) | No | 0.569 / 9.30% / 14.39% |
| Westernbank PR (31027, PR) | GFC / 2010-04-30 | 10,797.3 | 0.0797 | Yes (2009-06-30, 304 d) | No | 1.767 / 3.92% / 75.65% |
| First Republic Bank (59017, CA) | Modern / 2023-05-01 | 212,638.9 | 0.0001 | No | No | 0.008 / 8.20% / 3.00% |
| Silicon Valley Bank (24735, CA) | Modern / 2023-03-10 | 209,026.0 | 0.0001 | No | No | 0.009 / 6.69% / 0.00% |
| Signature Bank (57053, NY) | Modern / 2023-03-12 | 110,363.6 | 0.0002 | No | No | 0.028 / 6.71% / 4.29% |
| Doral Bank (32102, PR) | Modern / 2015-02-27 | 5,898.5 | 0.0865 | Yes (2014-03-31, 333 d) | Yes (242 d) | 1.415 / 6.16% / 25.44% |
| Republic Bank (27332, PA) | Modern / 2024-04-26 | 5,866.2 | 0.0139 | Yes (2023-06-30, 301 d) | Yes (301 d) | 0.198 / 1.34% / 6.85% |

*Note(s):* Selection covers the top five FDIC formal closures (`RESTYPE == FAILURE`) by asset size (`QBFASSET`) in each subperiod. Lead days measure the interval from the first scorable flagged quarter to formal closure. Source: `outputs/tables/canary_case_studies.csv`.

Every top-five GFC credit-insolvency failure is flagged 129 to 304 days before closure, alongside modern credit-erosion failures such as Doral Bank (333 d lead) and Republic Bank (301 d lead). In sharp contrast, as shown in Figure 6 (right), Silicon Valley Bank, Signature Bank, and First Republic Bank remained near $\hat{p} \approx 5 \times 10^{-5}$ prior to their rapid 2023 collapses because their distress originated from concentrated uninsured deposit runs (DEPUNINS) and unrealized held-to-maturity bond duration losses (SCHA and SCHF) rather than credit delinquency—illustrating why accounting-based EWS models must be paired with real-time liquidity and interest-rate risk monitors.

## 7. Conclusions, Limitations, and Institutional Adaptation for Vietnam

### 7.1 Synthesis of Research Questions (Q1–Q3)

<!-- SYNC:START:REPORT_Q1 -->
On Q1 (storage, laptop processing, and Spark cluster justification), four decades of quarterly bank reports (1.68 million rows across 33 fields) are best stored as Hive-partitioned Apache Parquet (`YEAR=YYYY`) with Zstandard level-3 compression, canonical Arrow typing (Int32 keys, Date32 dates, Float64 measures), and sorted (REPDTE, CERT) row groups, compressing raw CSV storage by 2.76x (425.7 MB to 154.1 MB; write time 0.922 s). On a 1x–8x stress test up to 13.46 million rows (12 logical CPUs, 15.34 GB RAM, 4 Polars threads), Polars Streaming finishes in 4.271 s (2,499 MB peak RSS; Polars Lazy: 10.406 s / 2,569 MB), scaling at 101.0 MB per million rows (extrapolated 151.1-million-row ceiling in 16 GB RAM) versus 327.9 MB per million rows for Pandas with pushdown (48.5-million-row ceiling; 1,940 MB at 1x for naive Pandas). At equal 4-thread concurrency, PySpark 4.3 (`local[4]`) reproduces identical numeric outputs ($7.4 \times 10^{-15}$ yearly-sum parity, $0.000$ sample max diff) in 10.551 s at 1x (1,262.9 MB tree RSS) and 15.182 s at 4x (6.73 million rows, 1,807.8 MB tree RSS; `local[2]`: 16.163 s). A distributed Spark cluster is therefore unnecessary for bank-level Call Reports below 30–50 million rows and becomes justified only when expanding to thousands of regulatory schedules, borrower-level credit registries exceeding $10^8$ rows, or high-frequency interbank payment streams surpassing 32–64 GB single-node memory.
<!-- SYNC:END:REPORT_Q1 -->

On Q2 (raw data quality and look-ahead prevention), our automated audit documents eight structural defects—14-column schema drift, non-random era missingness (P3ASSET 100% null pre-2001), dual Basel/CBLR zero-as-missing regimes (18.13% pre-1990 RBC1AAJ; 37.15% post-2020 RBCRWAJ), negative stock balances, cumulative YTD income flows (0.0177 in Q1 to 0.0689 in Q4), negative-equity ratio inversions (15,659 rows), competing merger exits (16,892 censored vs. 2,859 failures), and a 60-day reporting lag excluding 59.1% of failed banks' final filings. Remediating every defect causally ($\le \text{REPDTE}$) and fitting all preprocessors strictly inside training folds prevents up to 20.95% quantile and 43.04% median parameter leakage while reducing modern false-alarm episodes by 27%.

On Q3 (honestly validated supervisory utility), the cleaned pipeline supports a reliable early-warning ranker for credit-driven insolvencies but not a stand-alone detector for sudden funding runs. On the single-use 2009Q1–2025Q2 test set (485 failures), HistGradientBoosting achieves a PR-AUC of 0.5478 (3.6x above the Texas Ratio baseline of 0.1507), ROC-AUC of 0.9854, and Brier score of 0.00238. At $\theta^* = 0.0039$ (invariant across $C_{\text{FP}} \in [\$231.1\text{k},\, \$519.9\text{k}]$), it flags 469 / 485 failing banks (96.70% recall, 297.0 days median lead, 43.0 median alerts per quarter), while Top-1% and Top-2% fixed capacity rules catch 399 / 485 (82.27%) and 443 / 485 (91.34%), respectively. Operational scoring of 4,306 unlabelled 2026Q2 banks flags 26 candidates for supervisory review without reusing test labels.

### 7.2 Empirical Limitations

Three limitations bound these conclusions. First, FDIC API history reflects restated Call Report snapshots rather than unrevised original vintage filings; although our 60-day gate enforces publication timing, historical accounting restatements cannot be reconstructed from the public REST endpoint. Second, trained on 1984–2007 credit crises, the model achieves 0.5847 PR-AUC (0.9881 ROC-AUC) during the 2009–2014 GFC and 0.1708 PR-AUC (0.9193 ROC-AUC; Recall@Top-1% of 66.98%) during 2015–2025, missing rapid uninsured-deposit runs and unrealized bond duration losses whose schedules (DEPUNINS, SCHA/SCHF) were absent in earlier decades. Third, private supervisory-brokered absorptions without formal FDIC resolution entries appear as censored mergers rather than observed failures.

### 7.3 Structural Adaptation for the State Bank of Vietnam (SBV)

While the lakehouse layout, point-in-time auditing rules, and temporal validation discipline transfer directly to the State Bank of Vietnam (SBV), four institutional adaptations are essential. First, regarding sample size and target-label definition, whereas the U.S. panel spans over 18,000 insured banks in 1984 and 4,117 historical closures, Vietnam's banking system consists of roughly 35 domestic commercial banks (four state-owned commercial banks and approximately 31 joint-stock commercial banks) plus foreign branches. Because formal liquidation is rare, distressed lenders enter SBV Early Intervention (*Can thiệp sớm*), Special Control (*Kiểm soát đặc biệt*), or mandatory zero-dong transfer (*Chuyển giao bắt buộc 0 đồng*, including OceanBank, CB, GPBank, DongA Bank, and SCB)—mechanisms now codified under the Law on Credit Institutions No. 32/2024/QH15. For Vietnam, $y_{i,q}=1$ must therefore represent supervisory intervention, emergency refinancing, or severe composite CAMELS downgrades rather than legal closure, and the loss function must reflect systemic rescue and contagion costs rather than Vietnam's statutory VND 125 million deposit insurance cap (Decision No. 32/2021/QD-TTg).

Second, regulatory accounting adjustments are critical under Vietnamese Accounting Standards (VAS) and Circular 11/2021/TT-NHNN. Historically, weak banks parked impaired loans at the Vietnam Asset Management Company (VAMC) in exchange for special bonds (*trái phiếu đặc biệt VAMC*) or restructured exposures under forbearance circulars (Circular 01/2020 and Circular 02/2023). An SBV feature pipeline must compute an Adjusted Texas Ratio combining Group 2 (Special Mention) loans, Group 3–5 nonperforming loans, and unprovisioned VAMC special bonds in the numerator, paired with Circular 41/2016/TT-NHNN (Basel II) capital adequacy ratios and the tightened related-party exposure limits introduced in Law 32/2024/QH15.

Third, because a cross-section of 35 banks cannot support complex tree ensembles on quarterly balance sheets alone—as underscored by our Table 7 finding that parsimonious models remain competitive even with thousands of U.S. banks—an SBV EWS should favor transparent, explainable risk indicators fusing monthly supervisory filings with granular Credit Information Center (CIC) borrower-level credit registries (tracking cross-bank NPL contagion and ecosystem concentration) and daily CITAD interbank payment flows (detecting sudden retail deposit runs in real time, as in October 2022).

Fourth, infrastructure sizing follows directly from Section 4's scaling boundary. Monthly bank-level panels for ~50 institutions require under 50 MB and execute instantaneously in single-node Polars. Integrating tens of millions of quarterly CIC borrower exposures alongside daily CITAD payment streams, however, expands the working set into hundreds of gigabytes—crossing the 30–50 million row threshold where the Bronze, Silver, and Gold Parquet lakehouse justifies an on-premise SBV Apache Spark cluster.

## Notes and Disclosures

1. *Reproducibility and data provenance:* The data-acquisition client was adapted from the instructor-provided starter script (`fdic_download_starter.py`). From a clean clone of the repository with raw data placed in `./data_raw/fdic` (or downloaded via `python scripts/00_download.py --download`), executing `bash run_all.sh` deterministically runs `01_storage.py` through `09_check_report_numbers.py` followed by `pytest tests/` (`seed = 42`), regenerating all tables, figures, notebooks, and this manuscript while logging per-step runtimes to `outputs/tables/run_all_timing.txt`.
2. *AI tool disclosure:* Google Antigravity AI coding assistant was used for pair-programming pipeline modularization, debugging schema drift and PySpark configurations, and typesetting documentation. All experimental designs, empirical measurements, statistical tests, and supervisory interpretations were independently verified against local data outputs.
3. *Open-source benchmarks:* All related public FDIC repositories reviewed during research design are documented alongside our methodological improvements in Appendix Table A1.

## References

<!-- BLOCK:REFERENCES:START -->
Cole, R.A. and White, L.J. (2012), “Déjà vu all over again: the causes of U.S. commercial bank failures this time around”, *Journal of Financial Services Research*, Vol. 42 No. 1, pp. 5–29.

Elkan, C. (2001), “The foundations of cost-sensitive learning”, *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 973–978.

Federal Deposit Insurance Corporation (FDIC) (2024), *2024 Annual Report*, Division of Risk Management Supervision examination counts, FDIC, Washington, DC.

Federal Deposit Insurance Corporation (FDIC) (2025), *Proposed 2025 Operating Budget*, Division of Risk Management Supervision Ongoing Operations budget, FDIC, Washington, DC.

Federal Deposit Insurance Corporation (FDIC) (2026), *BankFind Suite Public REST API and RISView Field Dictionary*, available at: [https://api.fdic.gov/banks/docs/](https://api.fdic.gov/banks/docs/) (accessed 5 November 2026).

National Assembly of Vietnam (2024), *Law on Credit Institutions No. 32/2024/QH15*, Hanoi, Vietnam.

Prime Minister of Vietnam (2021), *Decision No. 32/2021/QD-TTg on the Deposit Insurance Payment Limit*, Hanoi, Vietnam.

Shumway, T. (2001), “Forecasting bankruptcy more accurately: a simple hazard model”, *The Journal of Business*, Vol. 74 No. 1, pp. 101–124.

Wheelock, D.C. and Wilson, P.W. (2000), “Why do banks disappear? The determinants of U.S. bank failures and acquisitions”, *Review of Economics and Statistics*, Vol. 82 No. 1, pp. 127–138.
<!-- BLOCK:REFERENCES:END -->

## Appendix: Related Open-Source Repositories Consulted

**Table A1.** Comparative review of related public open-source repositories consulted during pipeline design.

| Open-Source Reference | Domain Focus and Inspiration Drawn | Methodological Improvements in Our Pipeline |
| :--- | :--- | :--- |
| [`batuhanisik751/bank-failure-early-warning`](https://github.com/batuhanisik751/bank-failure-early-warning) | CAMELS intuition, Texas Ratio baseline, walk-forward CV, FDIC cost logic | Re-implemented in Medallion Polars and PySpark (1.68M rows); added Bayes prior-odds calibration, episode cost, and capacity budgets |
| [`Tajaddin/bank-early-warning`](https://github.com/Tajaddin/bank-early-warning) | Classical R/SQL CAMELS econometrics and directional variable interpretation | Added columnar Parquet benchmarks, PySpark parity, and 42-year HGB permutation importance paired with standardized Logit coefficients |
| [`vfabuluje/Canary`](https://github.com/vfabuluje/Canary) | Canary-in-the-coal-mine supervisory framing and failure case tracking | Replaced heuristic dashboard with walk-forward calibrated ML and formal post-hoc Canary diagnostics across GFC and modern failures |
| [`martinogie/bank-health-data-platform`](https://github.com/martinogie/bank-health-data-platform) | Medallion lakehouse conceptual framing for FDIC Call Report engineering | Extended beyond BI ingestion to 42-year schema-drift audits, merger censoring, CBLR remediation, and forward failure modelling |
| [`saivardhan13-MSV/fdic-bank-health-pipeline`](https://github.com/saivardhan13-MSV/fdic-bank-health-pipeline) | Cloud/ETL ingestion workflow patterns over FDIC financial metrics | Added PyArrow Hive Zstd benchmarks, YTD flow de-cumulation tests, 60-day PIT lag enforcement, and Spark-Polars numerical parity |
| [`heltonwo/fdic-failed-banks-etl`](https://github.com/heltonwo/fdic-failed-banks-etl) | Lightweight ETL structure for historical FDIC closure records | Eliminated survivorship bias by linking closures to all 24,064 surviving and merged banks across 1.68M bank-quarters |
| [`Zahivc/Bank-Bankruptcy-Predictions`](https://github.com/Zahivc/Bank-Bankruptcy-Predictions) | ML/DL experimentation for bank bankruptcy classification on FDIC extracts | Avoided SMOTE and random splits (severe temporal leakage and distorted priors) in favor of strict temporal splits and Elkan (2001) calibration |
| [`zekaili/bank-loan-failure-prediction`](https://www.kaggle.com/code/zekaili/bank-loan-failure-prediction) | Reviewed during initial scoping for binary credit classification workflows | Targets retail consumer loan defaults rather than institutional bank Call Reports; excluded from pipeline implementation |

*Note(s):* Reviewed public repositories are documented in compliance with course academic integrity guidelines; all pipeline code, empirical benchmarks, and econometric evaluations in this study were authored independently.
