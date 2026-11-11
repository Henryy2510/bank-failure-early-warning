"""Leak-free label construction and time-ordered train/purge/test splitting."""
from __future__ import annotations

import polars as pl

from .io import failures_by_cert
from .utils import cfg


def attach_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Forward-looking label  y_{i,q} = 1{bank i fails in (q, q + 4 quarters]}  plus point-in-time flags.

    * y = 1 iff the bank's (earliest) failure event satisfies REPDTE < FAILDATE <= REPDTE + 1 year
      (exactly four quarter-ends; quarter-end dates + 1 year are quarter-end dates).
    * Mergers / voluntary exits have no failure event, so y = 0 (right-censored), NEVER 1.
    * Rows dated on/after the failure date are dropped (a failed bank cannot be scored any more).
    * `pit_scorable`: a Call Report only becomes available `publication_lag_days` after quarter-end. If the bank
      failed before that, nobody could have scored it in real time, so the row is flagged non-scorable and is
      excluded from training and evaluation (see modeling.py).
    * `label_complete`: the 4-quarter window is fully observed (REPDTE <= last_label_quarter).
    """
    lag = cfg()["panel"]["publication_lag_days"]
    out = df.join(failures_by_cert(), on="CERT", how="left")
    out = out.filter(pl.col("FAILDATE").is_null() | (pl.col("REPDTE") < pl.col("FAILDATE")))

    available = pl.col("REPDTE") + pl.duration(days=lag)
    last_lbl = pl.lit(cfg()["label"]["last_label_quarter"]).str.to_date("%Y-%m-%d")
    return out.with_columns(
        (pl.col("FAILDATE").is_not_null()
         & (pl.col("FAILDATE") > pl.col("REPDTE"))
         & (pl.col("FAILDATE") <= pl.col("REPDTE").dt.offset_by("1y"))).cast(pl.Int8).alias("target_fail_4q"),
        (pl.col("FAILDATE").is_null() | (pl.col("FAILDATE") > available)).alias("pit_scorable"),
        (pl.col("REPDTE") <= last_lbl).alias("label_complete"),
    )


def assign_splits(df: pl.DataFrame) -> pl.DataFrame:
    """DEV (<= dev_end) | PURGE (4-quarter gap) | TEST (test_start..test_end) | POST_SAMPLE (labels incomplete)."""
    s = cfg()["split"]
    dev_end = pl.lit(s["dev_end"]).str.to_date("%Y-%m-%d")
    test_start = pl.lit(s["test_start"]).str.to_date("%Y-%m-%d")
    test_end = pl.lit(s["test_end"]).str.to_date("%Y-%m-%d")
    return df.with_columns(
        pl.when(pl.col("REPDTE") <= dev_end).then(pl.lit("DEV"))
        .when(pl.col("REPDTE") < test_start).then(pl.lit("PURGE"))
        .when(pl.col("REPDTE") <= test_end).then(pl.lit("TEST"))
        .otherwise(pl.lit("POST_SAMPLE")).alias("split_group"))
