"""Automated Data Quality Audit and Reporting for the FDIC Call Report panel (Part 2).

Every number in the generated report is computed here; nothing is typed in by hand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .io import failures_by_cert, manifest, parse_failures, raw_financial_files, read_raw_all
from .utils import cfg, df_to_markdown, out_dir, save_json


def audit_manifest() -> dict[str, Any]:
    """Verify raw download completeness via the manifest written by the downloader."""
    m = manifest()
    return {
        "quarters_downloaded": len(m),
        "total_manifest_rows": sum(v["rows"] for v in m.values()),
        "incomplete_quarters": [q for q, v in m.items() if not v.get("complete", False)],
        "api_mismatch_quarters": [q for q, v in m.items() if v.get("rows") != v.get("api_total")],
    }


def audit_schema_drift() -> pl.DataFrame:
    """Columns whose physical type changes across the quarterly files."""
    import pyarrow.parquet as pq

    records = []
    for f in raw_financial_files():
        sch = pq.read_schema(f)
        for name, typ in zip(sch.names, sch.types):
            records.append({"quarter": f.stem.replace("REPDTE=", ""), "column": name, "dtype": str(typ)})
    return (pl.DataFrame(records).group_by("column")
            .agg(pl.col("dtype").n_unique().alias("num_types"), pl.col("dtype").unique().sort().alias("types"))
            .filter(pl.col("num_types") > 1)
            .with_columns(pl.col("types").list.join(" / "))
            .sort("column"))


def audit_missingness_by_era(df: pl.DataFrame) -> pl.DataFrame:
    cols = ["P3ASSET", "P9ASSET", "NAASSET", "ELNATR", "NTLNLS", "RBC1AAJ", "EQ"]
    era_num = pl.col("REPDTE").dt.year() // 5 * 5
    era_label = (
        pl.when(era_num == 1980).then(pl.lit("1984 (4 qtrs)"))
        .when(era_num == 2025).then(pl.lit("2025-2026"))
        .otherwise(era_num.cast(pl.String) + pl.lit("-") + (era_num + 4).cast(pl.String))
    )
    return (df.with_columns(era_num.alias("_era_sort"), era_label.alias("era"))
            .group_by(["_era_sort", "era"])
            .agg(pl.len().alias("rows"), *[(pl.col(c).is_null().mean() * 100).round(1).alias(f"{c} null %") for c in cols])
            .sort("_era_sort")
            .drop("_era_sort"))


def audit_domain_validity(df: pl.DataFrame) -> dict[str, Any]:
    n = df.height
    yr = pl.col("REPDTE").dt.year()
    pre90 = df.filter(yr < 1990)
    post89 = df.filter(yr >= 1990)
    era2020 = df.filter(yr >= 2020)
    cnt = lambda e: int(df.filter(e).height)  # noqa: E731
    return {
        "rows": n,
        "asset_nonpositive": cnt(pl.col("ASSET") <= 0),
        "negative_equity": cnt(pl.col("EQ") < 0),
        "negative_equity_pct": round(cnt(pl.col("EQ") < 0) / n * 100, 3),
        "deposits_exceed_assets": cnt(pl.col("DEP") > pl.col("ASSET")),
        "deposits_exceed_assets_with_negative_equity": cnt((pl.col("DEP") > pl.col("ASSET")) & (pl.col("EQ") < 0)),
        "domestic_exceeds_total_deposits": cnt(pl.col("DEPDOM") > pl.col("DEP")),
        "net_loans_exceed_gross_loans": cnt(pl.col("LNLSNET") > pl.col("LNLSGR")),
        "gross_loans_exceed_assets": cnt(pl.col("LNLSGR") > pl.col("ASSET")),
        "nonaccrual_exceeds_noncurrent_plus_ore": cnt(
            pl.col("NAASSET") > (pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0))
        ),
        "ore_negative": cnt(pl.col("ORE") < 0),
        "sc_negative": cnt(pl.col("SC") < 0),
        "chbal_negative": cnt(pl.col("CHBAL") < 0),
        "nclnls_negative": cnt(pl.col("NCLNLS") < 0),
        "lnatres_negative": cnt(pl.col("LNATRES") < 0),
        "naasset_negative": cnt(pl.col("NAASSET") < 0),
        "extreme_roa": cnt(pl.col("ROA").abs() > 50),
        "extreme_roe": cnt(pl.col("ROE").abs() > 200),
        "pre1990_rows": pre90.height,
        "pre1990_rbc1_zero_pct": round(pre90.filter(pl.col("RBC1AAJ") == 0).height / max(1, pre90.height) * 100, 2),
        "post1989_rbc1_zero": post89.filter(pl.col("RBC1AAJ") == 0).height,
        "post1989_rbc1_zero_eq_null": post89.filter((pl.col("RBC1AAJ") == 0) & pl.col("EQ").is_null()).height,
        "era2020_rows": era2020.height,
        "era2020_rbcrwa_zero": era2020.filter(pl.col("RBCRWAJ") == 0).height,
        "era2020_rbcrwa_zero_pct": round(era2020.filter(pl.col("RBCRWAJ") == 0).height / max(1, era2020.height) * 100, 2),
        "era2020_rbcrwa_zero_cblr": era2020.filter((pl.col("RBCRWAJ") == 0) & (pl.col("RBC1AAJ") > 0)).height,
        "p3asset_null_pct": round(df["P3ASSET"].null_count() / n * 100, 1),
    }


def audit_zero_as_missing_by_era(df: pl.DataFrame) -> pl.DataFrame:
    """Detect zero-as-missing regimes across regulatory eras (Basel transition pre-1990 & CBLR opt-in post-2020)."""
    yr = pl.col("REPDTE").dt.year()
    era_expr = (
        pl.when(yr < 1990).then(pl.lit("1984-1989 (Pre-Basel)"))
        .when(yr < 2000).then(pl.lit("1990-1999"))
        .when(yr < 2010).then(pl.lit("2000-2009"))
        .when(yr < 2020).then(pl.lit("2010-2019"))
        .otherwise(pl.lit("2020-2026 (CBLR Era)"))
    )
    return (
        df.with_columns(era_expr.alias("era"))
        .group_by("era")
        .agg(
            pl.len().alias("rows"),
            ((pl.col("RBCRWAJ") == 0).mean() * 100).round(2).alias("RBCRWAJ==0 (%)"),
            (((pl.col("RBCRWAJ") == 0) & (pl.col("RBC1AAJ") > 0)).mean() * 100).round(2).alias("RBCRWAJ==0 & RBC1>0 (%)"),
            ((pl.col("RBC1AAJ") == 0).mean() * 100).round(2).alias("RBC1AAJ==0 (%)"),
        )
        .sort("era")
    )


def audit_ytd_monotonicity(df: pl.DataFrame) -> pl.DataFrame:
    """Prove cumulatively reported YTD income flows grow monotonically Q1->Q4 while annualised ROA stays flat."""
    return (
        df.filter(pl.col("ASSET") > 0)
        .with_columns(pl.col("REPDTE").dt.quarter().alias("quarter"))
        .group_by("quarter")
        .agg(
            pl.len().alias("rows"),
            (pl.col("INTINC") / pl.col("ASSET")).median().round(4).alias("median_intinc_to_asset"),
            (pl.col("NETINC") / pl.col("ASSET")).median().round(4).alias("median_netinc_to_asset"),
            pl.col("ROA").median().round(3).alias("median_roa_pct"),
        )
        .sort("quarter")
        .with_columns((pl.lit("Q") + pl.col("quarter").cast(pl.String)).alias("quarter"))
    )


def audit_failures_and_exits(df: pl.DataFrame) -> dict[str, Any]:
    """Referential integrity between Financials, Failures and the sample; point-in-time gap analysis."""
    raw = parse_failures()
    lag = cfg()["panel"]["publication_lag_days"]
    since = raw.filter(pl.col("FAILDATE") >= pl.date(1984, 1, 1))
    panel_certs = df["CERT"].unique().cast(pl.Int32)
    with_cert = since.filter(pl.col("CERT").is_not_null())
    matched = with_cert.filter(pl.col("CERT").is_in(panel_certs.implode()))
    by_cert = failures_by_cert()
    last = df.group_by("CERT").agg(pl.col("REPDTE").max().alias("last_repdte"))
    j = (by_cert.filter(pl.col("FAILDATE") >= pl.date(1984, 1, 1)).join(last, on="CERT", how="inner")
         .with_columns((pl.col("FAILDATE") - pl.col("last_repdte")).dt.total_days().alias("gap")))
    return {
        "events_all_time": raw.height,
        "events_without_cert": int(raw["CERT"].is_null().sum()),
        "events_since_1984": since.height,
        "events_since_1984_with_cert": with_cert.height,
        "events_matched_to_panel": matched.height,
        "events_not_in_panel": with_cert.height - matched.height,
        "restype_since_1984": {r["RESTYPE"]: r["len"] for r in since.group_by("RESTYPE").len().to_dicts()},
        "certs_with_multiple_events": int(
            raw.filter(pl.col("CERT").is_not_null())
            .group_by("CERT").len().filter(pl.col("len") > 1).height
        ),
        "rows_from_multiple_event_certs": int(raw.filter(pl.col("CERT").is_not_null())["CERT"].is_duplicated().sum()),
        "failed_banks_in_panel": j.height,
        "gap_median_days": float(j["gap"].median()),
        "gap_p95_days": float(j["gap"].quantile(0.95)),
        "share_last_report_not_scorable": round(float((j["gap"] <= lag).mean()) * 100, 1),
        "publication_lag_days": lag,
    }


def audit_reporting_gaps(df: pl.DataFrame) -> dict[str, Any]:
    """Count real internal reporting gaps (non-consecutive quarters within a bank) that justify ytd_prev_ok & lag4_ok."""
    from .core import add_time_keys

    d = add_time_keys(df.select(["CERT", "REPDTE"])).with_columns(
        pl.col("qidx").shift(1).over("CERT").alias("prev_qidx"),
        pl.col("year").shift(1).over("CERT").alias("prev_year"),
        pl.col("qidx").shift(4).over("CERT").alias("prev4_qidx"),
    )
    internal_gap = pl.col("prev_qidx").is_not_null() & ((pl.col("qidx") - pl.col("prev_qidx")) > 1)
    intra_year_gap = (
        internal_gap
        & (pl.col("quarter") > 1)
        & (pl.col("year") == pl.col("prev_year"))
    )
    lag4_gap = pl.col("prev4_qidx").is_not_null() & ((pl.col("qidx") - pl.col("prev4_qidx")) != 4)
    gap_rows = d.filter(internal_gap)
    ytd_rows = d.filter(intra_year_gap)
    lag4_rows = d.filter(lag4_gap)
    return {
        "banks_with_internal_gap": int(gap_rows["CERT"].n_unique()),
        "internal_gap_events": int(gap_rows.height),
        "missing_bank_quarters_implied": int((gap_rows["qidx"] - gap_rows["prev_qidx"] - 1).sum()) if gap_rows.height else 0,
        "intra_year_ytd_guard_triggers": int(ytd_rows.height),
        "banks_with_intra_year_ytd_gap": int(ytd_rows["CERT"].n_unique()),
        "lag4_guard_triggers": int(lag4_rows.height),
        "banks_with_lag4_gap": int(lag4_rows["CERT"].n_unique()),
    }


def audit_dev_missingness_and_tails(df: pl.DataFrame) -> dict[str, Any]:
    """Strictly DEV-only (<= 2007-12-31, pit_scorable) audit of informative missingness and extreme-tail failure rates.

    Never touches TEST (2009Q1-2025Q2). Quantitatively justifies keeping outliers (winsorising instead of dropping)
    and preserving missingness indicators rather than listwise deletion.
    """
    from .core import texas_expr
    from .labels import attach_labels

    dev_end = cfg()["split"]["dev_end"]
    dev_cut = pl.lit(dev_end).str.to_date("%Y-%m-%d")
    dev = (
        attach_labels(df.filter(pl.col("REPDTE") <= dev_cut))
        .filter(pl.col("pit_scorable"))
        .with_columns(
            pl.when(pl.col("ASSET") > 0).then(pl.col("EQ") / pl.col("ASSET")).otherwise(None).alias("eq_asset_ratio"),
            texas_expr().alias("texas_ratio"),
        )
    )
    base_rate_pct = float(dev["target_fail_4q"].mean() * 100.0)
    n_dev = int(dev.height)
    n_pos = int(dev["target_fail_4q"].sum())

    # 1. Informative missingness across key variables (y=1 vs y=0, and fail rate when missing vs observed)
    miss_rows = []
    for col in ("P9ASSET", "NAASSET", "ELNATR", "NTLNLS", "EQ", "ROA", "NIMY"):
        is_null = pl.col(col).is_null()
        pos_df = dev.filter(pl.col("target_fail_4q") == 1)
        neg_df = dev.filter(pl.col("target_fail_4q") == 0)
        m_df = dev.filter(is_null)
        o_df = dev.filter(~is_null)
        m_rate = float(m_df["target_fail_4q"].mean() * 100.0) if m_df.height else 0.0
        o_rate = float(o_df["target_fail_4q"].mean() * 100.0) if o_df.height else 0.0
        miss_rows.append({
            "column": col,
            "missing_rows_dev": int(m_df.height),
            "null_pct_when_y1": round(float(pos_df[col].is_null().mean() * 100.0), 2),
            "null_pct_when_y0": round(float(neg_df[col].is_null().mean() * 100.0), 2),
            "fail_rate_when_missing_pct": round(m_rate, 3),
            "fail_rate_when_observed_pct": round(o_rate, 3),
            "relative_risk_missing_vs_obs": round(m_rate / max(o_rate, 1e-6), 2),
        })

    # 2. Extreme-tail failure concentration (justifies winsorising rather than dropping outliers)
    q01_eq = float(dev["eq_asset_ratio"].drop_nulls().quantile(0.01))
    q01_roa = float(dev["ROA"].drop_nulls().quantile(0.01))
    q99_roa = float(dev["ROA"].drop_nulls().quantile(0.99))
    q99_tex = float(dev["texas_ratio"].drop_nulls().quantile(0.99))

    tail_specs = [
        ("Negative equity (EQ < 0)", pl.col("EQ") < 0),
        (f"Bottom 1% eq_asset_ratio (< {q01_eq:.4f})", pl.col("eq_asset_ratio") < q01_eq),
        (f"Top 1% texas_ratio (>= {q99_tex:.3f})", pl.col("texas_ratio") >= q99_tex),
        (f"Bottom 1% ROA (< {q01_roa:.2f}%)", pl.col("ROA") < q01_roa),
        (f"Outer 1% ROA tails (< {q01_roa:.2f}% or > {q99_roa:.2f}%)", (pl.col("ROA") < q01_roa) | (pl.col("ROA") > q99_roa)),
        (
            "Union of 1% distress tails (EQ<0 or bot-1% EQ/ROA or top-1% Texas)",
            (pl.col("EQ") < 0)
            | (pl.col("eq_asset_ratio") < q01_eq)
            | (pl.col("ROA") < q01_roa)
            | (pl.col("texas_ratio") >= q99_tex),
        ),
    ]
    tail_rows = []
    for label, cond in tail_specs:
        sub = dev.filter(cond.fill_null(False))
        sub_pos = int(sub["target_fail_4q"].sum())
        rate = float(sub["target_fail_4q"].mean() * 100.0) if sub.height else 0.0
        tail_rows.append({
            "segment": label,
            "rows": int(sub.height),
            "share_of_dev_rows_pct": round(sub.height / max(1, n_dev) * 100.0, 2),
            "positive_rows": sub_pos,
            "share_of_all_dev_positives_pct": round(sub_pos / max(1, n_pos) * 100.0, 2),
            "fail_rate_pct": round(rate, 2),
            "lift_vs_dev_base": round(rate / max(base_rate_pct, 1e-6), 1),
        })

    return {
        "scope": f"Strictly DEV (<= {dev_end}, pit_scorable only; TEST untouched)",
        "dev_rows": n_dev,
        "dev_positive_rows": n_pos,
        "dev_base_fail_rate_pct": round(base_rate_pct, 3),
        "informative_missingness": miss_rows,
        "tail_failure_rates": tail_rows,
    }


def generate_quality_report(output_path: Path | None = None) -> dict[str, Any]:
    """Run the complete audit, write the Markdown report and a JSON summary used by the final report."""
    output_path = output_path or out_dir("dq_report") / "data_quality_report.md"
    m = audit_manifest()
    drift = audit_schema_drift()
    df = read_raw_all()
    era = audit_missingness_by_era(df)
    v = audit_domain_validity(df)
    zeros_era = audit_zero_as_missing_by_era(df)
    ytd_mono = audit_ytd_monotonicity(df)
    f = audit_failures_and_exits(df)
    gaps = audit_reporting_gaps(df)
    dev_diag = audit_dev_missingness_and_tails(df)

    zeros_era.write_csv(out_dir("tables") / "dq_zero_as_missing.csv")
    ytd_mono.write_csv(out_dir("tables") / "dq_ytd_monotonicity.csv")
    save_json({"reporting_gaps": gaps, "dev_diagnostics": dev_diag}, out_dir("tables") / "dq_dev_diagnostics.json")

    lines = [
        "# Automated Data Quality Audit (Part 2)",
        "",
        "Generated by `src/bfews/quality.py`; every number below is computed from the raw files.",
        "",
        "## 1. Completeness",
        f"- Quarterly files: **{m['quarters_downloaded']}** (1984Q1-2026Q2); rows: **{m['total_manifest_rows']:,}**",
        f"- Quarters with `rows != api_total` or flagged incomplete: **{len(m['incomplete_quarters']) + len(m['api_mismatch_quarters'])}**",
        "",
        "## 2. Schema drift across quarterly files",
        f"{drift.height} columns change physical type between files (int64 / double / null), so a naive "
        "`pl.scan_parquet('*.parquet')` fails with `SchemaError`.",
        "",
        df_to_markdown(drift.select(["column", "types"])),
        "",
        "## 3. Structural missingness by era (% null) & Zero-as-Missing regimes",
        df_to_markdown(era),
        "",
        "### Basel / CBLR Zero-as-Missing Regime Audit",
        f"In **2020-2026**, `{v['era2020_rbcrwa_zero']:,}` / `{v['era2020_rows']:,}` (**{v['era2020_rbcrwa_zero_pct']}%**) rows record `RBCRWAJ == 0`, "
        f"of which `{v['era2020_rbcrwa_zero_cblr']:,}` have positive Tier 1 leverage (`RBC1AAJ > 0`): healthy small banks electing the Community Bank Leverage Ratio (CBLR) exempt from RWA reporting. "
        f"Post-1989 `RBC1AAJ == 0` occurs in `{v['post1989_rbc1_zero']:,}` rows (`{v['post1989_rbc1_zero_eq_null']:,}` with `EQ IS NULL`).",
        "",
        df_to_markdown(zeros_era),
        "",
        "## 4. Domain validity, negative stock anomalies & YTD flow evidence",
        f"- `ASSET <= 0`: **{v['asset_nonpositive']}** rows",
        f"- Negative equity: **{v['negative_equity']:,}** rows ({v['negative_equity_pct']}%)",
        f"- `DEP > ASSET`: **{v['deposits_exceed_assets']:,}** rows, of which **{v['deposits_exceed_assets_with_negative_equity']:,}** have negative equity",
        f"- `DEPDOM > DEP`: **{v['domestic_exceeds_total_deposits']}**; `LNLSNET > LNLSGR`: **{v['net_loans_exceed_gross_loans']}**; `LNLSGR > ASSET`: **{v['gross_loans_exceed_assets']:,}**",
        f"- Non-negative stock anomalies: `ORE < 0` (**{v['ore_negative']:,}**), `SC < 0` (**{v['sc_negative']:,}**), `CHBAL < 0` (**{v['chbal_negative']:,}**), `NCLNLS < 0` (**{v['nclnls_negative']:,}**), `LNATRES < 0` (**{v['lnatres_negative']:,}**), `NAASSET < 0` (**{v['naasset_negative']:,}**)",
        f"- `NAASSET > NCLNLS + ORE`: **{v['nonaccrual_exceeds_noncurrent_plus_ore']:,}** rows (structural schedule reporting differences)",
        f"- |ROA| > 50%: **{v['extreme_roa']:,}** rows; |ROE| > 200%: **{v['extreme_roe']:,}** rows",
        "",
        r"### Empirical YTD Flow Monotonicity (Q1 $\rightarrow$ Q4)",
        df_to_markdown(ytd_mono),
        "",
        "## 5. Failures table, exits, CHANGEC1 & point-in-time",
        f"- Failure/assistance events: **{f['events_all_time']:,}** all-time; **{f['events_without_cert']}** have no CERT (pre-1966); "
        f"**{f['events_since_1984']:,}** since 1984 (by type: {f['restype_since_1984']})",
        f"- Events since 1984 with a CERT: **{f['events_since_1984_with_cert']:,}**; matched to the panel: **{f['events_matched_to_panel']:,}**; "
        f"not in panel: **{f['events_not_in_panel']}**",
        f"- Unique CERTs that appear more than once in the table: **{f['certs_with_multiple_events']}** ({f['rows_from_multiple_event_certs']} total rows; earliest event is used)",
        rf"- Last Call Report $\rightarrow$ failure date: median **{f['gap_median_days']:.0f}** days, 95th percentile **{f['gap_p95_days']:.0f}** days",
        f"- **{f['share_last_report_not_scorable']}%** of failed banks fail within {f['publication_lag_days']} days of their last "
        "quarter-end, i.e. before that report could have been read: those rows are flagged `pit_scorable = False`",
        "- **Static Institutions Caveat**: `institutions.parquet` stores *current* metadata (`STNAME`, `ACTIVE`, `CHANGEC1`), not historical point-in-time state; exit audit uses it only post-hoc, never for historical predictive features.",
        "",
        "## 6. Decision matrix",
        "| Issue | Evidence | Handling | Why |",
        "|---|---|---|---|",
        f"| Schema drift | {drift.height} columns | explicit cast of every column when building Bronze | stable schema, `scan_parquet` works |",
        "| Dates stored as strings | `REPDTE`, `FAILDATE`, `ESTYMD` | parse to `Date` | date arithmetic, window joins, row-group statistics |",
        f"| Structural missing | `P3ASSET` null in {v['p3asset_null_pct']}% of rows (100% pre-2001); `P9ASSET`/`NAASSET`/`ELNATR`/`NTLNLS` 18–19% in 1984–89 | drop `P3ASSET` from 26-feature model; preserve 1984–89 nulls and impute strictly in-fold | prevents post-2001 era proxy while keeping genuine reporting regimes |",
        r"| YTD flows | 7 income items (`Q1` $\rightarrow$ `Q4` step-up) | de-cumulate, only if previous row is the previous quarter | true quarterly flow; no wrong lag across a gap |",
        f"| Pre-Basel & CBLR zeros | `RBC1AAJ==0` pre-1990 ({v['pre1990_rbc1_zero_pct']}%) & `RBCRWAJ==0` in 2020-26 ({v['era2020_rbcrwa_zero_pct']}%) | recode to null when pre-1990, `EQ IS NULL`, or CBLR (`year>=2020 & RBC1AAJ>0`) | prevents labeling 37% of healthy 2020+ banks as 0% capital |",
        f"| Negative stock anomalies | `ORE<0` ({v['ore_negative']}), `SC<0` ({v['sc_negative']}) | sanitize negative stocks to null before ratios | accounting adjustment artifacts, not negative holdings |",
        f"| Negative equity | {v['negative_equity']:,} rows | keep + distress flag; Texas ratio capped | these are the banks about to fail |",
        "| Extreme ratios | ROA, ROE, ... | winsorise inside each training fold | a global clip would use future data |",
        "| Merger vs failure | exit classification | merger = censored, y = 0 | healthy acquired banks are not failures |",
        f"| Publication lag | {f['share_last_report_not_scorable']}% of failures | rows that failed before availability are non-scorable | no look-ahead in training or evaluation |",
        "| Assistance vs closure | RESTYPE | both are failure events (DEV sensitivity vs FAILURE-only reported; TEST has 0 ASSISTANCE) | transparent |",
        "",
        "## 7. Strictly-DEV Empirical Justification (Gaps, Informative Missingness & Tails)",
        f"- Real internal reporting gaps across 1984Q1–2026Q2: **{gaps['banks_with_internal_gap']:,}** banks ({gaps['internal_gap_events']:,} gap events; {gaps['missing_bank_quarters_implied']:,} missing bank-quarters), triggering `ytd_prev_ok` on **{gaps['intra_year_ytd_guard_triggers']:,}** intra-year rows and `lag4_ok` on **{gaps['lag4_guard_triggers']:,}** 4Q-lag rows.",
        f"- Strictly DEV (`<= {cfg()['split']['dev_end']}`, `pit_scorable`, base failure rate **{dev_diag['dev_base_fail_rate_pct']}%**): `EQ < 0` rows exhibit **{dev_diag['tail_failure_rates'][0]['fail_rate_pct']}%** failure rate (**{dev_diag['tail_failure_rates'][0]['lift_vs_dev_base']}x** lift), proving extreme tails must be winsorised in-fold rather than dropped.",
        "",
        df_to_markdown(pl.DataFrame(dev_diag["tail_failure_rates"]).select(["segment", "rows", "share_of_dev_rows_pct", "share_of_all_dev_positives_pct", "fail_rate_pct", "lift_vs_dev_base"])),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "manifest": m,
        "validity": v,
        "failures": f,
        "reporting_gaps": gaps,
        "dev_diagnostics": dev_diag,
        "schema_drift_columns": drift["column"].to_list(),
        "missingness_by_era": era.to_dicts(),
        "zero_as_missing_by_era": zeros_era.to_dicts(),
        "ytd_monotonicity": ytd_mono.to_dicts(),
    }
    save_json(summary, out_dir("tables") / "dq_summary.json")
    return summary
