"""Unit and integration tests for data integrity, leak-free property, and model pipeline."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.bfews.features import compute_camels_features
from src.bfews.labels import assign_splits, attach_labels
from src.bfews.panel import decumulate_ytd_flows
from src.bfews.quality import audit_manifest


def test_manifest_completeness():
    """Verify that all downloaded quarters are complete without API count mismatches."""
    res = audit_manifest()
    assert res["quarters_downloaded"] >= 170
    assert len(res["incomplete_quarters"]) == 0
    assert len(res["api_mismatch_quarters"]) == 0


def test_decumulate_flows():
    """Test quarterly de-cumulation logic for Call Report income flows."""
    sample = pl.DataFrame({
        "CERT": [1, 1, 1, 1],
        "REPDTE": [
            datetime.date(2022, 3, 31),
            datetime.date(2022, 6, 30),
            datetime.date(2022, 9, 30),
            datetime.date(2022, 12, 31),
        ],
        "NETINC": [100.0, 250.0, 420.0, 600.0],
        "INTINC": [50.0, 110.0, 180.0, 260.0],
        "EINTEXP": [10.0, 25.0, 45.0, 70.0],
        "NONII": [5.0, 12.0, 20.0, 30.0],
        "NONIX": [20.0, 45.0, 75.0, 110.0],
        "ELNATR": [5.0, 15.0, 25.0, 40.0],
        "NTLNLS": [2.0, 6.0, 11.0, 18.0],
    })
    df_q = decumulate_ytd_flows(sample)
    # Q1 = 100, Q2 = 250-100=150, Q3 = 420-250=170, Q4 = 600-420=180
    assert df_q["NETINC_Q"].to_list() == [100.0, 150.0, 170.0, 180.0]


def test_leak_free_property():
    """Verify that feature engineering at quarter t does not depend on future records (t > REPDTE)."""
    # Create two datasets: one with 8 quarters, one truncated at quarter 4
    dates = [
        datetime.date(2020, 3, 31),
        datetime.date(2020, 6, 30),
        datetime.date(2020, 9, 30),
        datetime.date(2020, 12, 31),
        datetime.date(2021, 3, 31),
        datetime.date(2021, 6, 30),
        datetime.date(2021, 9, 30),
        datetime.date(2021, 12, 31),
    ]
    np.random.seed(42)
    n = len(dates)
    full_df = pl.DataFrame({
        "CERT": [101] * n,
        "REPDTE": dates,
        "ASSET": [1000.0 + i * 50 for i in range(n)],
        "EQ": [100.0 + i * 5 for i in range(n)],
        "LNLSGR": [600.0 + i * 30 for i in range(n)],
        "LNLSNET": [580.0 + i * 30 for i in range(n)],
        "NCLNLS": [10.0 + i for i in range(n)],
        "ORE": [2.0] * n,
        "LNATRES": [20.0] * n,
        "DEP": [800.0] * n,
        "CHBAL": [50.0] * n,
        "SC": [100.0] * n,
        "LNRE": [300.0] * n,
        "LNCI": [150.0] * n,
        "LNCON": [150.0] * n,
        "ROA": [1.0] * n,
        "ROE": [10.0] * n,
        "NIMY": [3.5] * n,
        "RBC1AAJ": [12.0] * n,
        "RBCRWAJ": [14.0] * n,
        "NONIX": [30.0] * n,
        "INTINC": [40.0] * n,
        "EINTEXP": [10.0] * n,
        "NONII": [5.0] * n,
        "ELNATR_Q": [2.0] * n,
        "NETINC_Q": [10.0] * n,
        "NAASSET": [5.0] * n,
        "BRO": [10.0] * n,
    })

    feat_full = compute_camels_features(full_df)
    # Truncate at Q4 (index 3)
    feat_trunc = compute_camels_features(full_df.slice(0, 4))

    # All features at Q4 must match exactly between full and truncated datasets
    for col in ["eq_asset_ratio", "texas_ratio", "nclnls_ratio", "asset_growth_4q"]:
        val_full = feat_full.filter(pl.col("REPDTE") == datetime.date(2020, 12, 31))[col].item()
        val_trunc = feat_trunc.filter(pl.col("REPDTE") == datetime.date(2020, 12, 31))[col].item()
        if val_full is not None or val_trunc is not None:
            assert np.isclose(val_full, val_trunc, equal_nan=True)


def test_merger_is_not_labeled_as_failure():
    """Verify that mergers and voluntary acquisitions are censored (label=0), never labeled as failures."""
    df = pl.DataFrame({
        "CERT": [99999],  # A bank that is not in the FDIC failures table
        "REPDTE": [datetime.date(2005, 6, 30)],
        "exit_status": ["MERGED_OR_OTHER"],
    })
    labeled = attach_labels(df)
    assert labeled["target_fail_4q"].item() == 0


def test_polars_spark_parity():
    """Verify zero discrepancy between Polars and PySpark outputs."""
    import json
    from src.bfews.utils import out_dir

    parity_file = out_dir("tables") / "spark_polars_parity.json"
    assert parity_file.exists(), "Run 06_spark.py first"
    data = json.loads(parity_file.read_text())
    assert data["difference"]["row_count_diff"] == 0
    assert data["difference"]["mean_eq_ratio_abs_diff"] < 1e-4
    assert data["difference"]["mean_texas_ratio_abs_diff"] < 1e-3


def test_held_out_test_results():
    """Verify held-out test evaluation metrics meet quality gates."""
    import json
    from src.bfews.utils import out_dir

    res_file = out_dir("tables") / "test_results.json"
    assert res_file.exists(), "Run 07_model.py first"
    res = json.loads(res_file.read_text())
    test_eval = res["held_out_test_evaluation"]
    assert test_eval["total_failures"] >= 50
    assert test_eval["roc_auc"] > 0.90
    assert test_eval["pr_auc"] > 0.50

