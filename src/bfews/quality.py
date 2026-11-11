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
    return (df.with_columns((pl.col("REPDTE").dt.year() // 5 * 5).alias("era"))
            .group_by("era")
            .agg(pl.len().alias("rows"), *[(pl.col(c).is_null().mean() * 100).round(1).alias(f"{c} null %") for c in cols])
            .sort("era"))


def audit_domain_validity(df: pl.DataFrame) -> dict[str, Any]:
    n = df.height
    pre90 = df.filter(pl.col("REPDTE").dt.year() < 1990)
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
        "extreme_roa": cnt(pl.col("ROA").abs() > 50),
        "extreme_roe": cnt(pl.col("ROE").abs() > 200),
        "pre1990_rows": pre90.height,
        "pre1990_rbc1_zero_pct": round(pre90.filter(pl.col("RBC1AAJ") == 0).height / max(1, pre90.height) * 100, 2),
        "p3asset_null_pct": round(df["P3ASSET"].null_count() / n * 100, 1),
    }


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
        "certs_with_multiple_events": int(raw.filter(pl.col("CERT").is_not_null())["CERT"].is_duplicated().sum()),
        "failed_banks_in_panel": j.height,
        "gap_median_days": float(j["gap"].median()),
        "gap_p95_days": float(j["gap"].quantile(0.95)),
        "share_last_report_not_scorable": round(float((j["gap"] <= lag).mean()) * 100, 1),
        "publication_lag_days": lag,
    }


def generate_quality_report(output_path: Path | None = None) -> dict[str, Any]:
    """Run the complete audit, write the Markdown report and a JSON summary used by the final report."""
    output_path = output_path or out_dir("dq_report") / "data_quality_report.md"
    m = audit_manifest()
    drift = audit_schema_drift()
    df = read_raw_all()
    era = audit_missingness_by_era(df)
    v = audit_domain_validity(df)
    f = audit_failures_and_exits(df)

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
        "## 3. Structural missingness by era (% null)",
        df_to_markdown(era),
        "",
        "## 4. Domain validity and accounting invariants",
        f"- `ASSET <= 0`: **{v['asset_nonpositive']}** rows",
        f"- Negative equity: **{v['negative_equity']:,}** rows ({v['negative_equity_pct']}%)",
        f"- `DEP > ASSET`: **{v['deposits_exceed_assets']:,}** rows, of which **{v['deposits_exceed_assets_with_negative_equity']:,}** have negative equity",
        f"- `DEPDOM > DEP`: **{v['domestic_exceeds_total_deposits']}**; `LNLSNET > LNLSGR`: **{v['net_loans_exceed_gross_loans']}**",
        f"- |ROA| > 50%: **{v['extreme_roa']:,}** rows; |ROE| > 200%: **{v['extreme_roe']:,}** rows",
        f"- `RBC1AAJ == 0` before 1990: **{v['pre1990_rbc1_zero_pct']}%** of {v['pre1990_rows']:,} rows (ratio not reported yet)",
        "",
        "## 5. Failures table, exits and point-in-time",
        f"- Failure/assistance events: **{f['events_all_time']:,}** all-time; **{f['events_without_cert']}** have no CERT (pre-1966); "
        f"**{f['events_since_1984']:,}** since 1984 (by type: {f['restype_since_1984']})",
        f"- Events since 1984 with a CERT: **{f['events_since_1984_with_cert']:,}**; matched to the panel: **{f['events_matched_to_panel']:,}**; "
        f"not in panel: **{f['events_not_in_panel']}**",
        f"- CERTs that appear more than once in the table: **{f['certs_with_multiple_events']}** (earliest event is used)",
        f"- Last Call Report -> failure date: median **{f['gap_median_days']:.0f}** days, 95th percentile **{f['gap_p95_days']:.0f}** days",
        f"- **{f['share_last_report_not_scorable']}%** of failed banks fail within {f['publication_lag_days']} days of their last "
        "quarter-end, i.e. before that report could have been read: those rows are flagged `pit_scorable = False`",
        "",
        "## 6. Decision matrix",
        "| Issue | Evidence | Handling | Why |",
        "|---|---|---|---|",
        f"| Schema drift | {drift.height} columns | explicit cast of every column when building Bronze | stable schema, `scan_parquet` works |",
        "| Dates stored as strings | `REPDTE`, `FAILDATE`, `ESTYMD` | parse to `Date` | date arithmetic, window joins, row-group statistics |",
        f"| Structural missing | `P3ASSET` null in {v['p3asset_null_pct']}% of rows | keep null, impute on training data only | missing is a reporting regime, not noise |",
        "| YTD flows | 7 income items | de-cumulate, only if previous row is the previous quarter | true quarterly flow; no wrong lag across a gap |",
        f"| Basel zeros | `RBC1AAJ == 0` in {v['pre1990_rbc1_zero_pct']}% of pre-1990 rows | recode to null | 0 would mean zero capital |",
        f"| Negative equity | {v['negative_equity']:,} rows | keep + distress flag; Texas ratio capped | these are the banks about to fail |",
        "| Extreme ratios | ROA, ROE, ... | winsorise inside each training fold | a global clip would use future data |",
        "| Merger vs failure | exit classification | merger = censored, y = 0 | healthy acquired banks are not failures |",
        f"| Publication lag | {f['share_last_report_not_scorable']}% of failures | rows that failed before availability are non-scorable | no look-ahead in training or evaluation |",
        "| Assistance vs closure | RESTYPE | both are failure events (course definition); counts by type reported | transparent |",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {"manifest": m, "validity": v, "failures": f, "schema_drift_columns": drift["column"].to_list(),
               "missingness_by_era": era.to_dicts()}
    save_json(summary, out_dir("tables") / "dq_summary.json")
    return summary
