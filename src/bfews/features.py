"""CAMELS + Texas Ratio + Dynamics Feature Engineering in Polars (Part 4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .utils import data_dir, get_logger

log = get_logger("features")

# Definitions of feature column groups for downstream modeling pipelines
FEATURE_COLS_CAMELS = [
    # Capital Adequacy
    "eq_asset_ratio",
    "rbc1aaj_clean",
    "rbcrwaj_clean",
    "cap_distress_flag",
    # Asset Quality
    "nclnls_ratio",
    "naasset_ratio",
    "ore_ratio",
    "lnatres_ratio",
    "texas_ratio",
    # Management Efficiency
    "efficiency_ratio",
    # Earnings & Profitability
    "roa_q",
    "nimy_clean",
    "provision_asset_ratio",
    "netinc_q_asset_ratio",
    # Liquidity & Funding
    "liquid_asset_ratio",
    "brokered_dep_ratio",
    "loan_to_deposit_ratio",
    # Sensitivity & Loan Concentration
    "re_loan_share",
    "ci_loan_share",
    "con_loan_share",
    # Size & Dynamics (Lags strictly from past quarters)
    "log_asset",
    "asset_growth_4q",
    "loan_growth_4q",
    "d4_texas_ratio",
    "d4_nclnls_ratio",
    "d4_eq_ratio",
]


def compute_camels_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute CAMELS and Texas Ratio features on the silver panel.

    All feature calculations are strictly point-in-time and use data available at or before
    the reporting date (t <= REPDTE). No future information is utilized.
    """
    log.info("Computing CAMELS ratios and Texas ratios...")

    # Safe denominator epsilon to avoid division by zero
    eps = 1e-4

    df = df.with_columns([
        # Capital Adequacy
        (pl.col("EQ") / (pl.col("ASSET") + eps)).alias("eq_asset_ratio"),
        # Regulatory capital: pre-1990 Basel zero is treated as unobserved/null
        (
            pl.when((pl.col("REPDTE").dt.year() < 1990) & (pl.col("RBC1AAJ") == 0))
            .then(None)
            .otherwise(pl.col("RBC1AAJ"))
        ).alias("rbc1aaj_clean"),
        (
            pl.when((pl.col("REPDTE").dt.year() < 1990) & (pl.col("RBCRWAJ") == 0))
            .then(None)
            .otherwise(pl.col("RBCRWAJ"))
        ).alias("rbcrwaj_clean"),
        # Severe capital distress indicator (Equity < 2% of Assets or negative equity)
        (
            (pl.col("EQ") < 0) | ((pl.col("EQ") / (pl.col("ASSET") + eps)) < 0.02)
        ).cast(pl.Float64).alias("cap_distress_flag"),

        # Asset Quality
        (pl.col("NCLNLS") / (pl.col("LNLSGR") + eps)).alias("nclnls_ratio"),
        (pl.col("NAASSET") / (pl.col("ASSET") + eps)).alias("naasset_ratio"),
        (pl.col("ORE") / (pl.col("ASSET") + eps)).alias("ore_ratio"),
        (pl.col("LNATRES") / (pl.col("LNLSGR") + eps)).alias("lnatres_ratio"),
        # Texas Ratio = (Nonperforming Loans + Other Real Estate Owned) / (Tangible Equity + Loan Loss Reserves)
        (
            (pl.col("NCLNLS").fill_null(0) + pl.col("ORE").fill_null(0))
            / (pl.col("EQ").fill_null(0) + pl.col("LNATRES").fill_null(0) + eps)
        ).alias("texas_ratio"),

        # Management Quality / Efficiency: Non-interest expense / Operating Net Revenue
        (
            pl.col("NONIX")
            / (pl.col("INTINC") - pl.col("EINTEXP") + pl.col("NONII") + eps)
        ).alias("efficiency_ratio"),

        # Earnings
        (pl.col("ROA") / 100.0).alias("roa_q"),
        (pl.col("NIMY") / 100.0).alias("nimy_clean"),
        (pl.col("ELNATR_Q") / (pl.col("ASSET") + eps)).alias("provision_asset_ratio"),
        (pl.col("NETINC_Q") / (pl.col("ASSET") + eps)).alias("netinc_q_asset_ratio"),

        # Liquidity
        ((pl.col("CHBAL") + pl.col("SC")) / (pl.col("ASSET") + eps)).alias("liquid_asset_ratio"),
        (pl.col("BRO") / (pl.col("DEP") + eps)).alias("brokered_dep_ratio"),
        (pl.col("LNLSNET") / (pl.col("DEP") + eps)).alias("loan_to_deposit_ratio"),

        # Sensitivity & Loan Composition
        (pl.col("LNRE") / (pl.col("LNLSGR") + eps)).alias("re_loan_share"),
        (pl.col("LNCI") / (pl.col("LNLSGR") + eps)).alias("ci_loan_share"),
        (pl.col("LNCON") / (pl.col("LNLSGR") + eps)).alias("con_loan_share"),

        # Size
        (pl.col("ASSET").clip(lower_bound=1).log()).alias("log_asset"),
    ])

    log.info("Computing 4-quarter historical lag dynamics per CERT...")
    # Lags are sorted by CERT and REPDTE, taking shift strictly backwards
    df = df.sort(["CERT", "REPDTE"])

    df = df.with_columns([
        (
            (pl.col("ASSET") - pl.col("ASSET").shift(4).over("CERT"))
            / (pl.col("ASSET").shift(4).over("CERT") + eps)
        ).alias("asset_growth_4q"),
        (
            (pl.col("LNLSGR") - pl.col("LNLSGR").shift(4).over("CERT"))
            / (pl.col("LNLSGR").shift(4).over("CERT") + eps)
        ).alias("loan_growth_4q"),
        (
            pl.col("texas_ratio") - pl.col("texas_ratio").shift(4).over("CERT")
        ).alias("d4_texas_ratio"),
        (
            pl.col("nclnls_ratio") - pl.col("nclnls_ratio").shift(4).over("CERT")
        ).alias("d4_nclnls_ratio"),
        (
            pl.col("eq_asset_ratio") - pl.col("eq_asset_ratio").shift(4).over("CERT")
        ).alias("d4_eq_ratio"),
    ])

    return df


def build_gold_features(silver_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Build the final leak-free Gold feature table with target labels and split assignments."""
    if silver_df is None:
        silver_path = data_dir("silver") / "panel.parquet"
        log.info(f"Reading silver panel from {silver_path}...")
        silver_df = pl.read_parquet(silver_path)

    from .labels import assign_splits, attach_labels

    log.info("Constructing target labels...")
    df_labeled = attach_labels(silver_df)

    log.info("Assigning Dev / Purge / Test splits...")
    df_split = assign_splits(df_labeled)

    log.info("Generating feature set...")
    df_features = compute_camels_features(df_split)

    gold_path = data_dir("gold") / "features.parquet"
    log.info(f"Writing Gold feature dataset to {gold_path} ({df_features.height:,} rows)...")
    df_features.write_parquet(gold_path, compression="zstd", compression_level=3)
    return df_features
