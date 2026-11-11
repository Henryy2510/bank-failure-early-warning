"""Build clean, point-in-time, leak-free bank-quarter panel with de-cumulated flows and exit statuses."""
from __future__ import annotations

import polars as pl

from .core import add_time_keys, ytd_prev_ok
from .io import FLOWS_YTD, bronze_financials_path, failures_by_cert
from .utils import cfg, data_dir, get_logger

log = get_logger("panel")


def decumulate_ytd_flows(df: pl.DataFrame) -> pl.DataFrame:
    """Convert cumulative YTD Call Report income flows into true quarterly flows.

    FDIC income-statement items are reported year-to-date: Q1 = 3 months, Q2 = 6 months, Q3 = 9, Q4 = 12.
    Quarterly flow: Q1 -> YTD; Q2..Q4 -> YTD - previous row's YTD *only if the previous row is the previous
    quarter of the same bank and year* (otherwise null, instead of silently subtracting an older quarter).
    """
    df = add_time_keys(df)
    exprs = []
    for c in FLOWS_YTD:
        exprs.append(
            pl.when(pl.col("quarter") == 1).then(pl.col(c))
            .when(ytd_prev_ok()).then(pl.col(c) - pl.col(c).shift(1).over(["CERT", "year"]))
            .otherwise(None).alias(f"{c}_Q"))
    return df.with_columns(exprs)


def classify_bank_exits(df: pl.DataFrame) -> pl.DataFrame:
    """Classify the terminal status of every bank in the panel.

    * ACTIVE          : reports in the last quarter of the sample.
    * FAILED          : stopped reporting and the failures table has an event for this CERT whose date lies within
                        [min_gap, max_gap] days of the last report (config: panel.exit_failure_*_gap_days).
    * MERGED_OR_OTHER : stopped reporting for any other reason (merger, voluntary closure, ...): right-censored,
                        NEVER a failure.
    `has_failure_record` flags banks that have a failures-table event, whatever its date; a failure record whose
    date does not fit the window is reported as a data-quality finding.
    """
    p = cfg()["panel"]
    max_rep = df["REPDTE"].max()
    bank = df.group_by("CERT").agg(
        pl.col("REPDTE").min().alias("first_repdte"),
        pl.col("REPDTE").max().alias("last_repdte"),
        pl.len().alias("num_quarters"),
    )
    bank = bank.join(failures_by_cert().select(["CERT", "FAILDATE"]), on="CERT", how="left")
    gap = (pl.col("FAILDATE") - pl.col("last_repdte")).dt.total_days()
    return bank.with_columns(
        pl.col("FAILDATE").is_not_null().alias("has_failure_record"),
        gap.alias("fail_gap_days"),
        pl.when(pl.col("last_repdte") == max_rep).then(pl.lit("ACTIVE"))
        .when(pl.col("FAILDATE").is_not_null() & gap.is_between(p["exit_failure_min_gap_days"],
                                                                p["exit_failure_max_gap_days"]))
        .then(pl.lit("FAILED"))
        .otherwise(pl.lit("MERGED_OR_OTHER")).alias("exit_status"),
    )


def build_silver_panel() -> pl.DataFrame:
    """Bronze -> Silver: quarterly flows, availability date (publication lag), exit status."""
    log.info("Loading Bronze dataset...")
    df = pl.read_parquet(bronze_financials_path(), hive_partitioning=True)

    log.info("De-cumulating YTD flow variables (guarded by consecutive-quarter check)...")
    df = decumulate_ytd_flows(df).drop(["year", "quarter", "qidx"])

    lag = cfg()["panel"]["publication_lag_days"]
    log.info(f"Point-in-time: available_date = REPDTE + {lag} days")
    df = df.with_columns((pl.col("REPDTE") + pl.duration(days=lag)).alias("available_date"))

    log.info("Auditing bank exits...")
    exits = classify_bank_exits(df)
    df = df.join(exits.select(["CERT", "exit_status", "has_failure_record"]), on="CERT", how="left")
    exits.write_parquet(data_dir("silver") / "bank_exits.parquet")

    silver_path = data_dir("silver") / "panel.parquet"
    log.info(f"Writing clean panel to {silver_path} ({df.height:,} rows)...")
    df.write_parquet(silver_path, compression="zstd", compression_level=3)
    return df
