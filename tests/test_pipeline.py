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
    assert data["yearly_fingerprint"]["row_and_null_counts_identical"] is True
    assert data["yearly_fingerprint"]["max_relative_diff_of_sums"] < 1e-10
    assert data["row_level_sample"]["rows_only_in_one_engine"] == 0
    assert data["row_level_sample"]["max_abs_diff_any_column"] < 1e-6


def test_held_out_test_results():
    """Verify held-out test evaluation metrics meet quality gates."""
    import json
    from src.bfews.utils import out_dir

    res_file = out_dir("tables") / "test_results.json"
    assert res_file.exists(), "Run 07_model.py first"
    res = json.loads(res_file.read_text())
    test_eval = res["held_out_test_evaluation"]
    assert test_eval["failing_banks_total"] >= 50
    assert test_eval["roc_auc"] > 0.90
    assert test_eval["pr_auc"] > 0.50


def test_threshold_not_degenerate():
    """Verify that the cost-optimal threshold is strictly interior and sensitivity thresholds vary."""
    import json
    from src.bfews.utils import out_dir

    res_file = out_dir("tables") / "test_results.json"
    assert res_file.exists(), "Run 07_model.py first"
    res = json.loads(res_file.read_text())
    cost_opt = res["cost_threshold_tuning"]
    assert not cost_opt["theta_at_grid_edge"], "Optimal theta hit the boundary of theta_grid!"
    test_at = res["held_out_test_evaluation"]["at_threshold"]
    assert test_at["flagged_share_of_rows_pct"] < 20.0
    sens_thetas = [s["theta"] for s in cost_opt["sensitivity"].values()]
    assert len(set(sens_thetas)) >= 2, "Sensitivity thresholds are all identical!"
    assert cost_opt["dev_failing_banks_in_oof_blocks"] == 1522
    assert cost_opt["dev_failing_banks_all"] == 2661


def test_capacity_points_present():
    """Verify that fixed-budget capacity operating points (top 1%, 2%, 5%) are computed."""
    import json
    from src.bfews.utils import out_dir

    res_file = out_dir("tables") / "test_results.json"
    assert res_file.exists(), "Run 07_model.py first"
    res = json.loads(res_file.read_text())
    cap = {p["pct"]: p for p in res["held_out_test_evaluation"]["capacity_points"]}
    assert {1, 2, 5}.issubset(cap.keys())
    assert cap[1]["bank_recall"] > 0.70
    assert cap[5]["bank_recall"] > cap[1]["bank_recall"]


def test_notebooks_executed_and_report_numbers_consistent():
    """Verify all 5 notebooks have real nbclient execution metadata and report/README numbers match outputs."""
    import nbformat
    from src.bfews.utils import ROOT
    import subprocess
    import sys

    nb_dir = ROOT / "notebooks"
    nbs = sorted(nb_dir.glob("*.ipynb"))
    assert len(nbs) == 5, f"Expected 5 notebooks, found {len(nbs)}"
    for p in nbs:
        nb = nbformat.read(p, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) >= 2
        for c in code_cells:
            assert c.execution_count is not None and c.execution_count >= 1
            assert "execution" in c.metadata and "iopub.execute_input" in c.metadata["execution"]
            assert len(c.outputs) >= 1

    subprocess.run([sys.executable, str(ROOT / "scripts" / "09_check_report_numbers.py")], check=True)


def test_feature_importance_and_canary_outputs():
    """Verify strictly-DEV feature importance table and post-hoc Canary case studies CSV/JSON."""
    import json
    from src.bfews.features import FEATURE_COLS_CAMELS
    from src.bfews.utils import out_dir

    imp_path = out_dir("tables") / "feature_importance.csv"
    canary_path = out_dir("tables") / "canary_case_studies.csv"
    assert imp_path.exists() and canary_path.exists(), "Run 07_model.py first"

    imp_df = pl.read_csv(imp_path)
    assert imp_df.height == len(FEATURE_COLS_CAMELS)
    assert set(imp_df["feature"].to_list()) == set(FEATURE_COLS_CAMELS)
    assert imp_df["hgb_perm_pr_auc"].is_sorted(descending=True)
    assert imp_df.filter(pl.col("in_top10")).height == 10
    assert (out_dir("figures") / "feature_importance.png").exists()

    canary_df = pl.read_csv(canary_path)
    assert canary_df.height >= 8
    regimes = set(canary_df["regime"].to_list())
    assert "GFC (2009-2014)" in regimes and "Modern (2015-2025)" in regimes
    caught = canary_df.filter(pl.col("caught_theta"))
    assert caught.height >= 6
    assert (caught["lead_days_theta"] > 0).all()

    res = json.loads((out_dir("tables") / "test_results.json").read_text())
    assert len(res["feature_importance_top10"]) == 10
    assert len(res["held_out_test_evaluation"]["canary_case_studies"]) == canary_df.height


def test_pdf_page_limit():
    """Verify compiled PDF report exists and complies with the strict 20-page ceiling."""
    import re
    import subprocess
    from src.bfews.utils import ROOT

    pdf_path = ROOT / "report" / "BFT65B_Final_Report.pdf"
    assert pdf_path.exists(), "Run 08_build_notebooks_and_report.py first"
    out = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    assert m is not None
    pages = int(m.group(1))
    assert 15 <= pages <= 20, f"Expected 15-20 pages, got {pages}"


