"""CAMELS + Texas ratio + 4-quarter dynamics, computed point-in-time in Polars (Part 4)."""
from __future__ import annotations

import polars as pl

from .core import add_time_keys, lag4_ok, texas_expr
from .utils import data_dir, get_logger

log = get_logger("features")

FEATURE_COLS_CAMELS = [
    # Capital adequacy
    "eq_asset_ratio", "rbc1aaj_clean", "rbcrwaj_clean", "cap_distress_flag",
    # Asset quality
    "nclnls_ratio", "naasset_ratio", "ore_ratio", "lnatres_ratio", "texas_ratio",
    # Management / efficiency
    "efficiency_ratio",
    # Earnings
    "roa_ytd", "nimy_clean", "provision_asset_ratio", "netinc_q_asset_ratio",
    # Liquidity & funding
    "liquid_asset_ratio", "brokered_dep_ratio", "loan_to_deposit_ratio",
    # Sensitivity / loan concentration
    "re_loan_share", "ci_loan_share", "con_loan_share",
    # Size and dynamics (lags only look backwards, and only across exactly 4 quarters)
    "log_asset", "asset_growth_4q", "loan_growth_4q", "d4_texas_ratio", "d4_nclnls_ratio", "d4_eq_ratio",
]


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num / den, or null when the denominator is not positive (imputed later, on training data only)."""
    return pl.when(den > 0).then(num / den).otherwise(None)


def compute_camels_features(df: pl.DataFrame) -> pl.DataFrame:
    """All features use data dated <= REPDTE of the same bank. No future information is used."""
    log.info("Computing CAMELS ratios and Texas ratio...")
    df = add_time_keys(df)
    pre_basel = pl.col("year") < 1990
    asset, loans, dep = pl.col("ASSET"), pl.col("LNLSGR"), pl.col("DEP")

    df = df.with_columns(
        # C
        safe_div(pl.col("EQ"), asset).alias("eq_asset_ratio"),
        # Basel risk-based ratios were not reported before 1990: a recorded 0 is "not available", not "zero capital"
        pl.when(pre_basel & (pl.col("RBC1AAJ") == 0)).then(None).otherwise(pl.col("RBC1AAJ")).alias("rbc1aaj_clean"),
        pl.when(pre_basel & (pl.col("RBCRWAJ") == 0)).then(None).otherwise(pl.col("RBCRWAJ")).alias("rbcrwaj_clean"),
        ((pl.col("EQ") < 0) | (safe_div(pl.col("EQ"), asset) < 0.02)).cast(pl.Float64).alias("cap_distress_flag"),
        # A
        safe_div(pl.col("NCLNLS"), loans).alias("nclnls_ratio"),
        safe_div(pl.col("NAASSET"), asset).alias("naasset_ratio"),
        safe_div(pl.col("ORE"), asset).alias("ore_ratio"),
        safe_div(pl.col("LNATRES"), loans).alias("lnatres_ratio"),
        texas_expr().alias("texas_ratio"),
        # M: non-interest expense / (net interest income + non-interest income), both year-to-date
        safe_div(pl.col("NONIX"), pl.col("INTINC") - pl.col("EINTEXP") + pl.col("NONII")).alias("efficiency_ratio"),
        # E  (ROA / NIMY as reported by FDIC: annualised, year-to-date, in percent)
        (pl.col("ROA") / 100.0).alias("roa_ytd"),
        (pl.col("NIMY") / 100.0).alias("nimy_clean"),
        safe_div(pl.col("ELNATR_Q"), asset).alias("provision_asset_ratio"),
        safe_div(pl.col("NETINC_Q"), asset).alias("netinc_q_asset_ratio"),
        # L
        safe_div(pl.col("CHBAL") + pl.col("SC"), asset).alias("liquid_asset_ratio"),
        safe_div(pl.col("BRO"), dep).alias("brokered_dep_ratio"),
        safe_div(pl.col("LNLSNET"), dep).alias("loan_to_deposit_ratio"),
        # S
        safe_div(pl.col("LNRE"), loans).alias("re_loan_share"),
        safe_div(pl.col("LNCI"), loans).alias("ci_loan_share"),
        safe_div(pl.col("LNCON"), loans).alias("con_loan_share"),
        asset.clip(lower_bound=1).log().alias("log_asset"),
    )

    log.info("Computing 4-quarter backward-looking dynamics (guarded: lag must be exactly 4 quarters)...")
    ok4 = lag4_ok()

    def d4(col: str) -> pl.Expr:
        return pl.when(ok4).then(pl.col(col) - pl.col(col).shift(4).over("CERT")).otherwise(None)

    def growth(col: str) -> pl.Expr:
        prev = pl.col(col).shift(4).over("CERT")
        return pl.when(ok4 & (prev > 0)).then(pl.col(col) / prev - 1).otherwise(None)

    return df.with_columns(
        growth("ASSET").alias("asset_growth_4q"),
        growth("LNLSGR").alias("loan_growth_4q"),
        d4("texas_ratio").alias("d4_texas_ratio"),
        d4("nclnls_ratio").alias("d4_nclnls_ratio"),
        d4("eq_asset_ratio").alias("d4_eq_ratio"),
    ).drop(["year", "quarter", "qidx"])


def build_gold_features(silver_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Silver -> Gold: label + split + features."""
    from .labels import assign_splits, attach_labels

    if silver_df is None:
        silver_path = data_dir("silver") / "panel.parquet"
        log.info(f"Reading silver panel from {silver_path}...")
        silver_df = pl.read_parquet(silver_path)

    log.info("Constructing target labels and point-in-time flags...")
    df = assign_splits(attach_labels(silver_df))
    log.info("Generating features...")
    df = compute_camels_features(df)

    gold_path = data_dir("gold") / "features.parquet"
    log.info(f"Writing Gold feature dataset to {gold_path} ({df.height:,} rows)...")
    df.write_parquet(gold_path, compression="zstd", compression_level=3)
    return df
