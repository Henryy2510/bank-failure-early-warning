"""Pandas baseline implementation of core transformations for fair benchmarking (Part 3)."""
from __future__ import annotations

import pandas as pd

from .io import FLOWS_YTD


def run_pandas_core_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Execute the exact same transformations as the Polars pipeline in pandas.

    Steps:
    1. Datetime parsing and year/quarter extraction.
    2. Groupby-shift flow de-cumulation within (CERT, year).
    3. CAMELS feature ratios & Texas Ratio.
    4. Groupby-shift 4-quarter historical lag dynamics within CERT.
    """
    df = df_raw.copy()

    # 1. Date extraction
    if not pd.api.types.is_datetime64_any_dtype(df["REPDTE"]):
        df["REPDTE"] = pd.to_datetime(df["REPDTE"], format="%Y%m%d")

    df["year"] = df["REPDTE"].dt.year
    df["quarter"] = df["REPDTE"].dt.quarter

    # Sort for deterministic window operations
    df = df.sort_values(["CERT", "REPDTE"])

    # 2. De-cumulate YTD flows
    for c in FLOWS_YTD:
        if c in df.columns:
            c_q = f"{c}_Q"
            lag_c = df.groupby(["CERT", "year"])[c].shift(1)
            df[c_q] = df[c].where(df["quarter"] == 1, df[c] - lag_c)

    # 3. CAMELS Ratios
    eps = 1e-4
    df["eq_asset_ratio"] = df["EQ"] / (df["ASSET"] + eps)
    df["nclnls_ratio"] = df["NCLNLS"] / (df["LNLSGR"] + eps)

    ore_fill = df["ORE"].fillna(0)
    ncl_fill = df["NCLNLS"].fillna(0)
    eq_fill = df["EQ"].fillna(0)
    lnat_fill = df["LNATRES"].fillna(0)
    df["texas_ratio"] = (ncl_fill + ore_fill) / (eq_fill + lnat_fill + eps)

    # 4. Groupby-shift 4Q dynamics
    g_cert = df.groupby("CERT")
    df["asset_growth_4q"] = (df["ASSET"] - g_cert["ASSET"].shift(4)) / (g_cert["ASSET"].shift(4) + eps)
    df["d4_texas_ratio"] = df["texas_ratio"] - g_cert["texas_ratio"].shift(4)

    return df
