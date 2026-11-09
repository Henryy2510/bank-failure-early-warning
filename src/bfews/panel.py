"""Build clean, point-in-time, leak-free bank-quarter panel with de-cumulated flows and exit statuses."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .io import (
    FLOWS_YTD,
    bronze_financials_path,
    parse_failures,
    parse_institutions,
    scan_bronze,
)
from .utils import data_dir, get_logger

log = get_logger("panel")


def decumulate_ytd_flows(df: pl.DataFrame) -> pl.DataFrame:
    """Convert cumulative YTD Call Report income flows into true quarterly flows.

    FDIC Call Report income statement items are reported YTD:
    - Q1 (March): 3-month flow
    - Q2 (June): 6-month cumulative flow
    - Q3 (September): 9-month cumulative flow
    - Q4 (December): 12-month cumulative flow

    True quarterly flow:
    - Q1: flow_q = flow_ytd
    - Q2..Q4: flow_q = flow_ytd - lag(flow_ytd) within the same calendar year & CERT.
    """
    df = df.with_columns([
        pl.col("REPDTE").dt.year().alias("year"),
        pl.col("REPDTE").dt.quarter().alias("quarter"),
    ])

    exprs = []
    for c in FLOWS_YTD:
        c_q = f"{c}_Q"
        lag_c = pl.col(c).shift(1).over(["CERT", "year"])
        expr = (
            pl.when(pl.col("quarter") == 1)
            .then(pl.col(c))
            .otherwise(pl.col(c) - lag_c)
            .alias(c_q)
        )
        exprs.append(expr)

    return df.with_columns(exprs)


def classify_bank_exits(df: pl.DataFrame) -> pl.DataFrame:
    """Classify the terminal status of every bank in the panel.

    Exit categories:
    - ACTIVE: Bank continues reporting up to the end of the sample (2026).
    - FAILED: Bank ceased reporting and matched an FDIC resolution / receivership event.
    - MERGED_OR_OTHER: Bank ceased reporting due to merger, acquisition, or voluntary exit (censored).
    """
    failures = parse_failures().filter(pl.col("CERT").is_not_null())
    institutions = parse_institutions()

    bank_summary = (
        df.group_by("CERT")
        .agg([
            pl.col("REPDTE").min().alias("first_repdte"),
            pl.col("REPDTE").max().alias("last_repdte"),
            pl.len().alias("num_quarters"),
        ])
    )

    # Join failure info
    bf = (
        bank_summary.join(
            failures.select(["CERT", "FAILDATE", "RESTYPE", "COST"]).unique(subset=["CERT"]),
            on="CERT",
            how="left",
        )
        .join(
            institutions.select(["CERT", "ACTIVE", "CHANGEC1"]).unique(subset=["CERT"]),
            on="CERT",
            how="left",
        )
    )

    bf = bf.with_columns([
        (pl.col("FAILDATE") - pl.col("last_repdte")).dt.total_days().alias("fail_gap_days"),
        (
            pl.when(pl.col("last_repdte") >= pl.date(2026, 1, 1))
            .then(pl.lit("ACTIVE"))
            .when(
                pl.col("FAILDATE").is_not_null()
                & ((pl.col("FAILDATE") - pl.col("last_repdte")).dt.total_days() <= 270)
                & ((pl.col("FAILDATE") - pl.col("last_repdte")).dt.total_days() >= -90)
            )
            .then(pl.lit("FAILED"))
            .otherwise(pl.lit("MERGED_OR_OTHER"))
        ).alias("exit_status"),
    ])

    return bf


def build_silver_panel() -> pl.DataFrame:
    """Load Bronze dataset, compute de-cumulated quarterly flows, append point-in-time tags and exits."""
    log.info("Loading Bronze dataset...")
    df = pl.read_parquet(bronze_financials_path())

    log.info("De-cumulating YTD flow variables...")
    df = decumulate_ytd_flows(df)

    log.info("Point-in-time: Adding publication lag (60 days)...")
    # Publication date: Call Report is officially accessible ~60 days after quarter-end
    df = df.with_columns([
        (pl.col("REPDTE") + pl.duration(days=60)).alias("available_date"),
    ])

    log.info("Auditing bank exits...")
    exits = classify_bank_exits(df)

    df = df.join(
        exits.select(["CERT", "exit_status", "FAILDATE", "COST"]),
        on="CERT",
        how="left",
    )

    silver_path = data_dir("silver") / "panel.parquet"
    log.info(f"Writing clean panel to {silver_path} ({df.height:,} rows)...")
    df.write_parquet(silver_path, compression="zstd", compression_level=3)
    return df
