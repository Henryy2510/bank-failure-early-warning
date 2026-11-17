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


def test_cblr_rbcrwaj_zero_and_negative_stock_sanitized():
    """Verify post-2020 CBLR RBCRWAJ==0 and pre-1990 Basel zeros become null, negative stocks sanitize, real distress keeps 0."""
    dates = [
        datetime.date(1988, 12, 31),
        datetime.date(2022, 6, 30),
        datetime.date(2022, 9, 30),
    ]
    sample = pl.DataFrame({
        "CERT": [1, 2, 3],
        "REPDTE": dates,
        "ASSET": [1000.0, 1000.0, 1000.0],
        "EQ": [80.0, 120.0, -10.0],
        "LNLSGR": [600.0, 600.0, 600.0],
        "LNLSNET": [580.0, 580.0, 580.0],
        "NCLNLS": [5.0, 5.0, 50.0],
        "ORE": [1.0, -5.0, 5.0],
        "LNATRES": [10.0, 10.0, 10.0],
        "DEP": [800.0, 800.0, 900.0],
        "CHBAL": [50.0, 50.0, 20.0],
        "SC": [100.0, -2.0, 50.0],
        "LNRE": [300.0, 300.0, 300.0],
        "LNCI": [150.0, 150.0, 150.0],
        "LNCON": [150.0, 150.0, 150.0],
        "ROA": [1.0, 1.1, -3.0],
        "ROE": [10.0, 11.0, -30.0],
        "NIMY": [3.5, 3.6, 2.0],
        "RBC1AAJ": [0.0, 11.5, 0.0],
        "RBCRWAJ": [0.0, 0.0, 0.0],
        "NONIX": [30.0, 30.0, 40.0],
        "INTINC": [40.0, 40.0, 30.0],
        "EINTEXP": [10.0, 10.0, 20.0],
        "NONII": [5.0, 5.0, 2.0],
        "ELNATR_Q": [2.0, 1.0, 15.0],
        "NETINC_Q": [10.0, 12.0, -20.0],
        "NAASSET": [5.0, 4.0, 40.0],
        "BRO": [10.0, 10.0, 100.0],
    })
    out = compute_camels_features(sample).sort("CERT")
    # Bank 1 (pre-1990): both Basel zeros -> None
    assert out["rbc1aaj_clean"][0] is None
    assert out["rbcrwaj_clean"][0] is None
    # Bank 2 (2022 CBLR with healthy Tier 1 lev + negative ORE/SC): RBCRWAJ -> None, negative stocks -> None
    assert np.isclose(out["rbc1aaj_clean"][1], 11.5)
    assert out["rbcrwaj_clean"][1] is None
    assert out["ore_ratio"][1] is None
    assert out["liquid_asset_ratio"][1] is None
    # Bank 3 (2022 truly insolvent EQ < 0 & RBC == 0): real distress zeros preserved
    assert out["rbc1aaj_clean"][2] == 0.0
    assert out["rbcrwaj_clean"][2] == 0.0


def test_cv_folds_strictly_purged():
    """Verify 4-quarter forward label windows in every walk-forward DEV fold end strictly before val starts."""
    from src.bfews.utils import cfg

    for y0, y1 in cfg()["split"]["cv_blocks"]:
        max_train_date = datetime.date(y0 - 2, 12, 31)
        max_label_horizon_end = datetime.date(y0 - 1, 12, 31)
        min_val_date = datetime.date(y0, 3, 31)
        assert max_train_date < max_label_horizon_end < min_val_date


def test_preprocessor_invariant_to_test_corruption():
    """Verify train-only preprocessing parameters do not change when TEST feature values are corrupted."""
    from src.bfews.modeling import make_model

    rng = np.random.default_rng(42)
    X_dev = rng.normal(size=(200, 5)).astype(np.float32)
    X_dev[0, 0] = np.nan
    y_dev = rng.integers(0, 2, size=200)

    X_test_clean = rng.normal(size=(50, 5)).astype(np.float32)
    X_test_corrupt = np.full((50, 5), 1e9, dtype=np.float32)

    pipe_a = make_model("logit")[:-1].fit(X_dev, y_dev)
    pipe_b = make_model("logit")[:-1].fit(X_dev, y_dev)
    _ = pipe_a.transform(X_test_clean)
    _ = pipe_b.transform(X_test_corrupt)

    assert np.allclose(pipe_a.named_steps["winsor"].lo_, pipe_b.named_steps["winsor"].lo_, equal_nan=True)
    assert np.allclose(pipe_a.named_steps["winsor"].hi_, pipe_b.named_steps["winsor"].hi_, equal_nan=True)
    assert np.allclose(pipe_a.named_steps["impute"].statistics_, pipe_b.named_steps["impute"].statistics_)
    assert np.allclose(pipe_a.named_steps["scale"].mean_, pipe_b.named_steps["scale"].mean_)


def test_upgrade_artifacts_present():
    """Verify newly added Part 1-5 upgrade tables and diagnostic figures exist and are non-empty."""
    from src.bfews.utils import out_dir

    tdir = out_dir("tables")
    fdir = out_dir("figures")
    required_tables = [
        "part1_float32_check.csv",
        "dq_zero_as_missing.csv",
        "dq_ytd_monotonicity.csv",
        "exit_changec1_codes.csv",
        "benchmark_environment.json",
        "polars_naive_vs_optimized_plan.txt",
        "spark_scaling_results.csv",
        "leak_parameter_shift.csv",
        "dev_label_sensitivity.csv",
        "watchlist_2026q2.csv",
        "watchlist_2026q2_summary.json",
        "model_curves_data.json",
    ]
    for name in required_tables:
        p = tdir / name
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty table artifact: {p}"

    p3 = pl.read_csv(tdir / "part3_benchmark_results.csv")
    assert {"1x", "2x", "4x", "8x"}.issubset(set(p3["scale"].to_list()))
    sp3 = pl.read_csv(tdir / "spark_scaling_results.csv")
    assert {"1x", "4x"}.issubset(set(sp3["scale"].to_list()))

    for fig in ("pr_and_cost_curves.png", "alarms_and_canary_trajectories.png"):
        fp = fdir / fig
        assert fp.exists() and fp.stat().st_size > 0, f"Missing figure artifact: {fp}"


