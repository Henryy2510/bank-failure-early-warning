"""Shared definitions of the "core" transformations, implemented identically in Polars and pandas.

The same six steps are reproduced in PySpark (`spark_jobs.py`) and are what Part 3 benchmarks and compares:

1. time keys: ``qidx = year * 4 + quarter`` (an integer that makes "consecutive quarter" checks exact)
2. ``NETINC_Q``: de-cumulated quarterly net income (YTD flow, guarded: only if the previous row is the previous quarter)
3. ``eq_asset_ratio = EQ / ASSET``
4. ``texas_ratio = (NCLNLS + ORE) / (EQ + LNATRES)``, capped; non-positive denominator = cap
5. ``asset_growth_4q`` (guarded: only if the row 4 positions back is exactly 4 quarters back)
6. ``d4_texas_ratio`` (same guard)

The guards matter: a bank with a missing quarter would otherwise get a "4-quarter lag" that is really 5 or 6 quarters.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .utils import cfg

CORE_COLS = ["CERT", "REPDTE", "ASSET", "EQ", "NCLNLS", "ORE", "LNATRES", "NETINC"]
CORE_OUT = ["NETINC_Q", "eq_asset_ratio", "texas_ratio", "asset_growth_4q", "d4_texas_ratio"]


def texas_cap() -> float:
    return float(cfg()["features"]["texas_cap"])


def texas_expr() -> pl.Expr:
    """Texas ratio = (non-current loans + OREO) / (equity + loan-loss reserves).

    A non-positive denominator means there is no loss-absorbing capital left, so the ratio is set to the cap
    (a negative ratio would wrongly rank an insolvent bank as the safest one).
    """
    num = pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0)
    den = pl.col("EQ").fill_null(0.0) + pl.col("LNATRES").fill_null(0.0)
    cap = texas_cap()
    return pl.when(den > 0).then((num / den).clip(upper_bound=cap)).otherwise(pl.lit(cap))


def ytd_prev_ok() -> pl.Expr:
    return pl.col("qidx").shift(1).over(["CERT", "year"]) == pl.col("qidx") - 1


def lag4_ok() -> pl.Expr:
    return pl.col("qidx").shift(4).over("CERT") == pl.col("qidx") - 4


def add_time_keys(frame):
    """Sort by (CERT, REPDTE) and add year / quarter / qidx. Works on DataFrame and LazyFrame."""
    return (frame.sort(["CERT", "REPDTE"])
            .with_columns(pl.col("REPDTE").dt.year().alias("year"), pl.col("REPDTE").dt.quarter().alias("quarter"))
            .with_columns((pl.col("year") * 4 + pl.col("quarter")).alias("qidx")))


def core_polars(frame):
    """Apply the core transformations. Accepts a pl.DataFrame (eager) or pl.LazyFrame (lazy/streaming)."""
    f = add_time_keys(frame)
    f = f.with_columns(
        pl.when(pl.col("quarter") == 1).then(pl.col("NETINC"))
        .when(ytd_prev_ok()).then(pl.col("NETINC") - pl.col("NETINC").shift(1).over(["CERT", "year"]))
        .otherwise(None).alias("NETINC_Q"),
        (pl.col("EQ") / pl.col("ASSET")).alias("eq_asset_ratio"),
        texas_expr().alias("texas_ratio"),
    )
    return f.with_columns(
        pl.when(lag4_ok()).then(pl.col("ASSET") / pl.col("ASSET").shift(4).over("CERT") - 1).otherwise(None)
        .alias("asset_growth_4q"),
        pl.when(lag4_ok()).then(pl.col("texas_ratio") - pl.col("texas_ratio").shift(4).over("CERT")).otherwise(None)
        .alias("d4_texas_ratio"),
    )


def core_pandas(df):
    """Same transformations in pandas (reference implementation for the Part 3 benchmark)."""
    import pandas as pd

    df = df.copy()
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    df = df.sort_values(["CERT", "REPDTE"])
    df["year"] = df["REPDTE"].dt.year
    df["quarter"] = df["REPDTE"].dt.quarter
    df["qidx"] = df["year"] * 4 + df["quarter"]

    g_year = df.groupby(["CERT", "year"], sort=False)
    prev_ok = g_year["qidx"].shift(1) == df["qidx"] - 1
    ytd_diff = df["NETINC"] - g_year["NETINC"].shift(1)
    df["NETINC_Q"] = np.where(df["quarter"] == 1, df["NETINC"], np.where(prev_ok, ytd_diff, np.nan))

    df["eq_asset_ratio"] = df["EQ"] / df["ASSET"]
    num = df["NCLNLS"].fillna(0.0) + df["ORE"].fillna(0.0)
    den = df["EQ"].fillna(0.0) + df["LNATRES"].fillna(0.0)
    cap = texas_cap()
    df["texas_ratio"] = np.where(den > 0, np.minimum(num / den.where(den > 0, 1.0), cap), cap)

    g_cert = df.groupby("CERT", sort=False)
    ok4 = g_cert["qidx"].shift(4) == df["qidx"] - 4
    df["asset_growth_4q"] = np.where(ok4, df["ASSET"] / g_cert["ASSET"].shift(4) - 1, np.nan)
    df["d4_texas_ratio"] = np.where(ok4, df["texas_ratio"] - g_cert["texas_ratio"].shift(4), np.nan)
    return df


def checksum_polars(df: pl.DataFrame) -> dict:
    """Order-independent summary used to prove that all engines produce the same output."""
    out = {"rows": int(df.height)}
    for c in CORE_OUT:
        out[c] = float(df[c].sum() or 0.0)
        out[c + "_nulls"] = int(df[c].null_count())
    return out


def checksum_pandas(df) -> dict:
    out = {"rows": int(len(df))}
    for c in CORE_OUT:
        out[c] = float(np.nansum(df[c].to_numpy(dtype=float)))
        out[c + "_nulls"] = int(df[c].isna().sum())
    return out


def max_rel_diff(a: dict, b: dict) -> float:
    worst = 0.0
    for k in a:
        if k.endswith("_nulls") or k == "rows":
            if a[k] != b[k]:
                return float("inf")
        else:
            worst = max(worst, abs(a[k] - b[k]) / max(1.0, abs(b[k])))
    return worst
