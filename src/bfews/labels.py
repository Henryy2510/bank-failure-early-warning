"""Leak-free label construction and time-ordered train/purge/test splitting."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .io import parse_failures
from .utils import cfg, data_dir, get_logger

log = get_logger("labels")


def attach_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Construct forward-looking binary failure label y_{i, q} = 1{bank i fails in (q, q+4 quarters]}.

    Rules:
    - Target: 1 if bank fails in (REPDTE, REPDTE + 366 days], 0 otherwise.
    - Mergers & voluntary exits: explicitly 0 (censored), NEVER 1.
    - Maximum label quarter: 2025-06-30 (later quarters lack full 4-quarter forward observation window).
    """
    failures = parse_failures().filter(pl.col("CERT").is_not_null())
    f_clean = (
        failures.filter(pl.col("FAILDATE").is_not_null())
        .select(["CERT", "FAILDATE"])
        .unique(subset=["CERT"])
    )

    df_labeled = df.join(f_clean, on="CERT", how="left")

    # Horizon: (REPDTE, REPDTE + 366 days]
    df_labeled = df_labeled.with_columns([
        (
            pl.col("FAILDATE").is_not_null()
            & (pl.col("FAILDATE") > pl.col("REPDTE"))
            & (pl.col("FAILDATE") <= pl.col("REPDTE") + pl.duration(days=366))
        )
        .cast(pl.Int8)
        .alias("target_fail_4q"),
        # Drop rows after failure if any bank miraculously filed post-failure
        (
            pl.col("FAILDATE").is_null() | (pl.col("REPDTE") < pl.col("FAILDATE"))
        ).alias("valid_pre_failure"),
    ]).filter(pl.col("valid_pre_failure")).drop("valid_pre_failure")

    # Mark modeling eligibility: quarters <= 2025-06-30
    last_lbl_q = pl.lit(cfg()["label"]["last_label_quarter"]).str.to_date("%Y-%m-%d")
    df_labeled = df_labeled.with_columns([
        (pl.col("REPDTE") <= last_lbl_q).alias("label_complete"),
    ])

    return df_labeled


def assign_splits(df: pl.DataFrame) -> pl.DataFrame:
    """Assign observations to DEV, PURGE, or TEST partitions.

    Partitioning scheme:
    - DEV: as-of quarters 1984-03-31 to 2007-12-31 (development period, contains S&L crisis).
    - PURGE: as-of quarters 2008-01-01 to 2008-12-31 (4 quarters gap to prevent label overlap).
    - TEST: as-of quarters 2009-03-31 to 2025-06-30 (held-out test period, contains GFC & 2023 crisis).
    - EXCLUDED: quarters > 2025-06-30 (incomplete labels).
    """
    dev_end = pl.lit(cfg()["split"]["dev_end"]).str.to_date("%Y-%m-%d")
    test_start = pl.lit(cfg()["split"]["test_start"]).str.to_date("%Y-%m-%d")
    test_end = pl.lit(cfg()["split"]["test_end"]).str.to_date("%Y-%m-%d")

    df_splits = df.with_columns([
        (
            pl.when(pl.col("REPDTE") <= dev_end)
            .then(pl.lit("DEV"))
            .when((pl.col("REPDTE") > dev_end) & (pl.col("REPDTE") < test_start))
            .then(pl.lit("PURGE"))
            .when((pl.col("REPDTE") >= test_start) & (pl.col("REPDTE") <= test_end))
            .then(pl.lit("TEST"))
            .otherwise(pl.lit("POST_SAMPLE"))
        ).alias("split_group"),
    ])
    return df_splits
