# Automated Data Quality Audit Report (Part 2)

## 1. Acquisition & Completeness Audit
- **Quarterly files verified**: 170 quarters (1984Q1 - 2026Q2)
- **Total panel records**: 1,682,616
- **Incomplete quarters**: None (100% verified)
- **API row mismatch quarters**: None

## 2. Schema Drift Across Quarters
The FDIC API returns inconsistent numeric types across quarters (e.g., Integer in some quarters, Double or Null in others).

| column | num_types | unique_types |
|---|---|---|
| LNLSGR | 2 | ['int64', 'double'] |
| DEPDOM | 2 | ['int64', 'double'] |
| NAASSET | 2 | ['double', 'int64'] |
| RBCRWAJ | 2 | ['int64', 'double'] |
| CHBAL | 2 | ['int64', 'double'] |
| LNRE | 2 | ['int64', 'double'] |
| P9ASSET | 2 | ['double', 'int64'] |
| LNLSNET | 2 | ['int64', 'double'] |
| NCLNLS | 2 | ['double', 'int64'] |
| LNCON | 2 | ['int64', 'double'] |
| LNCI | 2 | ['int64', 'double'] |
| SC | 2 | ['int64', 'double'] |
| P3ASSET | 2 | ['null', 'int64'] |
| DEP | 2 | ['int64', 'double'] |

> **Handling Strategy**: Explicit canonical typing is enforced during Bronze/Silver ingestion. All financial balances are unified as `Float64` and dates converted from string to ISO `Date`.

## 3. Structural Missingness by Era
Missingness is not missing completely at random (MCAR); it exhibits structural regulatory breaks:

| era | total_rows | P3ASSET_null_pct | P9ASSET_null_pct | NAASSET_null_pct | ELNATR_null_pct | NTLNLS_null_pct | RBC1AAJ_null_pct | ROA_null_pct | EQ_null_pct |
|---|---|---|---|---|---|---|---|---|---|
| 1980 | 71760 | 100 | 19.2 | 19.2 | 18.5 | 18.5 | 0 | 0 | 0.2 |
| 1985 | 347654 | 100 | 18.4 | 18.4 | 7.6 | 18.2 | 0 | 0 | 0.2 |
| 1990 | 286105 | 100 | 0 | 0 | 0.3 | 0.3 | 0 | 0 | 0.3 |
| 1995 | 225775 | 100 | 0 | 0 | 0.3 | 0.3 | 0 | 0 | 0.3 |
| 2000 | 192271 | 21.1 | 0 | 0 | 0.2 | 0.2 | 0 | 0 | 0.2 |
| 2005 | 172995 | 0 | 0 | 0 | 0.2 | 0.2 | 0 | 0 | 0.2 |
| 2010 | 145726 | 0 | 0 | 0 | 0.2 | 0.2 | 0 | 0 | 0.2 |
| 2015 | 116869 | 0 | 0 | 0 | 0.3 | 0.3 | 0 | 0 | 0.3 |
| 2020 | 96902 | 0 | 0 | 0 | 0.4 | 0.4 | 0 | 0 | 0.4 |
| 2025 | 26559 | 0 | 0 | 0 | 0.4 | 0.4 | 0 | 0 | 0.4 |

> **Key Findings**:
> - `P3ASSET` (loans past due 30-89 days) is **100% missing before 2001** due to reporting requirement changes.
> - `ELNATR` (provision for loan losses) exhibits ~18% missingness in 1984.
> - Core balance sheet items (`ASSET`, `DEP`, `EQ`) have near 0% missingness throughout 40+ years.

## 4. Domain Validity & Accounting Invariants
- **Non-positive total assets (`ASSET <= 0`)**: 0 records
- **Negative equity (`EQ < 0`)**: 15,659 (0.931%) — distressed banks with depleted capital.
- **Deposit exceeds asset (`DEP > ASSET`)**: 6204 records
- **Domestic deposit exceeds total (`DEPDOM > DEP`)**: 0 records
- **Extreme ROA (|ROA| > 50%)**: 1,056 records
- **Extreme ROE (|ROE| > 200%)**: 4,624 records
- **Pre-1990 RBC1AAJ == 0 (Basel I structural zero)**: 18.13% of pre-1990 rows

## 5. Referential Integrity & Exit Analysis
- **All-time FDIC failure events**: 4,117
- **Pre-1966 failures without CERT**: 488
- **Failures in panel period (1984-2026)**: 3,275
- **Matched with quarterly panel**: 3,218
- **Unmatched**: 22 (primarily institutions failing before their initial quarterly Call Report was captured)
- **Call Report to Failure Date Lag**: Median 48 days (95th percentile: 111 days)

## 6. Data Quality Decision Matrix
| Issue | Affected Rows | Remediation Strategy | Justification |
|---|---|---|---|
| Schema Drift | Multiple columns | Explicit schema casting in bronze ingestion | Prevents Polars union schema mismatch |
| Date String Types | `REPDTE`, `FAILDATE` | Convert to ISO Date `YYYY-MM-DD` | Enables date arithmetic and window joins |
| Structural Missing (`P3ASSET`) | 57.8% of all rows | Exclude `P3ASSET` or use with missingness indicator | Prevents historical survivorship/availability bias |
| YTD Cumulative Flows | All flow items (`NETINC`, etc.) | De-cumulate quarterly: `Q1=YTD1`, `Q_k=YTD_k - YTD_{k-1}` | Reflects true quarterly performance, removes seasonality trend |
| Pre-1990 Capital Zeroes | ~100% of pre-1990 RBC | Treat 0 as unobserved pre-Basel, impute median or use Equity/Assets | Avoids false insolvency signal before risk-weighted rules |
| Outliers / Distressed Values | ~1.5% of ratios | Robust median imputation + winsorization inside train pipeline | Prevents extreme gradients while avoiding look-ahead bias |
| Exits: Failure vs Merger | All non-active banks | Match with Failures table; non-failed exits censored (label=0) | Prevents labeling healthy acquired banks as failures |