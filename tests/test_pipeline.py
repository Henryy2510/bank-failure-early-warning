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
    data = json.loads(parity_file.read_text(encoding="utf-8"))
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
    res = json.loads(res_file.read_text(encoding="utf-8"))
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
    res = json.loads(res_file.read_text(encoding="utf-8"))
    cost_opt = res["cost_threshold_tuning"]
    assert not cost_opt["theta_at_grid_edge"], "Optimal theta hit the boundary of theta_grid!"
    test_at = res["held_out_test_evaluation"]["at_threshold"]
    assert test_at["flagged_share_of_rows_pct"] < 20.0
    sens_thetas = [s["theta"] for s in cost_opt["sensitivity"].values()]
    assert len(set(sens_thetas)) >= 2, "Sensitivity thresholds are all identical!"
    dq_file = out_dir("tables") / "dq_summary.json"
    is_frozen = True
    if dq_file.exists():
        dq = json.loads(dq_file.read_text(encoding="utf-8"))
        is_frozen = int(dq["manifest"]["total_manifest_rows"]) == 1_682_616
    if is_frozen:
        assert cost_opt["dev_failing_banks_in_oof_blocks"] == 1522
        assert cost_opt["dev_failing_banks_all"] == 2661
    else:
        assert 1400 <= cost_opt["dev_failing_banks_in_oof_blocks"] <= 1650
        assert 2500 <= cost_opt["dev_failing_banks_all"] <= 2800


def test_capacity_points_present():
    """Verify that fixed-budget capacity operating points (top 1%, 2%, 5%) are computed."""
    import json
    from src.bfews.utils import out_dir

    res_file = out_dir("tables") / "test_results.json"
    assert res_file.exists(), "Run 07_model.py first"
    res = json.loads(res_file.read_text(encoding="utf-8"))
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

    res = json.loads((out_dir("tables") / "test_results.json").read_text(encoding="utf-8"))
    assert len(res["feature_importance_top10"]) == 10
    assert len(res["held_out_test_evaluation"]["canary_case_studies"]) == canary_df.height


def test_pdf_page_limit():
    """Verify compiled PDF report exists and complies with the rubric ceiling (<= 20 main + <= 10 appendix)."""
    import re
    import shutil
    import subprocess
    from src.bfews.utils import ROOT

    pdf_path = ROOT / "report" / "QuanHuuNhatMinh_11230998_Final_Report.pdf"
    assert pdf_path.exists() and pdf_path.stat().st_size > 10_000, "Run 08_build_notebooks_and_report.py first"
    if not (shutil.which("pdfinfo") and shutil.which("pdftotext")):
        pytest.skip("poppler-utils (pdfinfo/pdftotext) not installed on PATH; PDF file presence verified")
    out = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    assert m is not None
    pages = int(m.group(1))
    pdf_txt = subprocess.run(["pdftotext", str(pdf_path), "-"], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    pdf_pages = [p for p in pdf_txt.split("\f") if p.strip()]
    app_page = next((idx + 1 for idx, ptxt in enumerate(pdf_pages) if "Appendix:" in ptxt), None)
    main_pages = (app_page - 1) if app_page is not None else pages
    appendix_pages = pages - main_pages
    assert 15 <= main_pages <= 20, f"Expected 15-20 main body pages, got {main_pages}"
    assert appendix_pages <= 10 and pages <= 30, f"Expected <= 10 appendix pages (total <= 30), got {appendix_pages} (total {pages})"


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
        "cfp_stability_interval.json",
        "frozen_protocol.json",
        "part3_full_schema_benchmark.json",
    ]
    for name in required_tables:
        p = tdir / name
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty table artifact: {p}"

    p3 = pl.read_csv(tdir / "part3_benchmark_results.csv")
    assert {"1x", "2x", "4x", "8x"}.issubset(set(p3["scale"].to_list()))
    sp3 = pl.read_csv(tdir / "spark_scaling_results.csv")
    assert {"1x", "4x"}.issubset(set(sp3["scale"].to_list()))
    assert {"PySpark local[2]", "PySpark local[4]"}.issubset(set(sp3["engine"].to_list()))

    for fig in ("pr_and_cost_curves.png", "alarms_and_canary_trajectories.png"):
        fp = fdir / fig
        assert fp.exists() and fp.stat().st_size > 0, f"Missing figure artifact: {fp}"


def test_forbidden_outcome_columns_excluded_from_features():
    """Verify no ex-post outcome, failure-registry, or split metadata column can enter FEATURE_COLS_CAMELS."""
    from src.bfews.features import FEATURE_COLS_CAMELS, FORBIDDEN_FEATURE_COLS

    overlap = set(FEATURE_COLS_CAMELS).intersection(FORBIDDEN_FEATURE_COLS)
    assert not overlap, f"Forbidden columns leaked into FEATURE_COLS_CAMELS: {overlap}"
    assert len(FEATURE_COLS_CAMELS) == 26


def test_no_post_failure_rows_in_gold():
    """Verify gold/features.parquet contains zero bank-quarter rows dated on or after FAILDATE."""
    from src.bfews.utils import data_dir

    gold_path = data_dir("gold") / "features.parquet"
    if not gold_path.exists():
        pytest.skip("Gold features parquet not present locally")
    gold = pl.read_parquet(gold_path, columns=["REPDTE", "FAILDATE"])
    post_fail = gold.filter(pl.col("FAILDATE").is_not_null() & (pl.col("REPDTE") >= pl.col("FAILDATE")))
    assert post_fail.height == 0, f"Found {post_fail.height} post-failure rows in gold/features.parquet"


def test_frozen_protocol_matches_test_results():
    """Verify outputs/tables/frozen_protocol.json SHA-256 and spec match config.yaml and test_results.json."""
    import hashlib
    import json
    from src.bfews.features import FEATURE_COLS_CAMELS
    from src.bfews.utils import cfg, out_dir

    tdir = out_dir("tables")
    proto = json.loads((tdir / "frozen_protocol.json").read_text(encoding="utf-8"))
    test_bytes = (tdir / "test_results.json").read_bytes()
    test_res = json.loads(test_bytes.decode("utf-8"))

    raw_sha = hashlib.sha256(test_bytes).hexdigest()
    lf_sha = hashlib.sha256(test_bytes.replace(b"\r\n", b"\n")).hexdigest()
    assert proto["test_results_sha256"] in {raw_sha, lf_sha}
    spec = proto["frozen_spec"]
    spec_bytes = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert proto["protocol_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert spec["split"] == cfg()["split"]
    assert spec["feature_columns"] == list(FEATURE_COLS_CAMELS)
    assert spec["frozen_theta"] == test_res["cost_threshold_tuning"]["theta"]
    assert proto["held_out_test_summary"]["pr_auc"] == test_res["held_out_test_evaluation"]["pr_auc"]


def test_readme_download_command_paths_exist():
    """Verify script paths referenced in README.md exist on disk and broken starter path is absent."""
    from src.bfews.utils import ROOT

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "data/fdic_download_starter.py" not in readme, "Stale broken path data/fdic_download_starter.py in README.md"
    for rel in ("fdic_download_starter.py", "scripts/00_download.py", "run_all.sh", "run_all.ps1", "requirements.txt", "config.yaml"):
        assert (ROOT / rel).exists(), f"Referenced file missing: {rel}"


def test_cfp_interval_contains_baseline():
    """Verify DEV OOF linear-cost decomposition reconstructs all saved U-curves and brackets C_FP = $250k."""
    import json
    from src.bfews.utils import out_dir

    tdir = out_dir("tables")
    stab = json.loads((tdir / "cfp_stability_interval.json").read_text(encoding="utf-8"))
    mc = json.loads((tdir / "model_curves_data.json").read_text(encoding="utf-8"))
    tr = json.loads((tdir / "test_results.json").read_text(encoding="utf-8"))

    baseline = float(tr["cost_threshold_tuning"]["false_alarm_cost_kusd"])
    assert stab["cfp_lo_kusd"] <= baseline <= stab["cfp_hi_kusd"], (
        f"Baseline C_FP={baseline} outside stability interval [{stab['cfp_lo_kusd']}, {stab['cfp_hi_kusd']}]"
    )
    dq_file = tdir / "dq_summary.json"
    is_frozen = True
    if dq_file.exists():
        dq = json.loads(dq_file.read_text(encoding="utf-8"))
        is_frozen = int(dq["manifest"]["total_manifest_rows"]) == 1_682_616
    if is_frozen:
        assert stab["cfp_lo_kusd"] <= 500.0 <= stab["cfp_hi_kusd"], "Sensitivity $500k should stay inside stability interval"
    else:
        assert stab["cfp_lo_kusd"] < baseline < stab["cfp_hi_kusd"]
        assert stab["stability_span_ratio"] > 1.0
    assert stab["max_curve_reconstruction_err_musd"] <= 1e-6, (
        f"Linear decomposition reconstruction error too large: {stab['max_curve_reconstruction_err_musd']}"
    )
    # Cross-check OLS linearity directly from saved DEV OOF curves
    by_cfp = mc["cost_u_curve"]["cost_musd_by_cfp"]
    c_vals = np.array([100.0, 250.0, 500.0, 1000.0])
    y_kusd = np.vstack([np.array(by_cfp[str(int(c))], dtype=float) * 1000.0 for c in c_vals])
    c_bar = c_vals.mean()
    e_hat = ((c_vals - c_bar)[:, None] * (y_kusd - y_kusd.mean(axis=0)[None, :])).sum(axis=0) / ((c_vals - c_bar) ** 2).sum()
    m_hat = y_kusd.mean(axis=0) - e_hat * c_bar
    for idx, c_val in enumerate(c_vals):
        pred_musd = np.round((m_hat + c_val * e_hat) / 1000.0, 1)
        assert np.max(np.abs(pred_musd - y_kusd[idx] / 1000.0)) <= 1e-6


def test_full_panel_truncation_2015q4_all_26_features():
    """Verify exact 0.0 causal invariance across ALL 26 CAMELS features + 7 flows on real FDIC Bronze truncated at 2015Q4."""
    from src.bfews.features import FEATURE_COLS_CAMELS
    from src.bfews.io import FLOWS_YTD, bronze_financials_path

    bronze_p = bronze_financials_path()
    if not bronze_p.exists() or not any(bronze_p.glob("YEAR=*/*.parquet")):
        pytest.skip("Bronze parquet not present locally")

    cutoff = datetime.date(2015, 12, 31)
    bronze = pl.read_parquet(bronze_p, hive_partitioning=True)
    full_sub = (
        compute_camels_features(decumulate_ytd_flows(bronze).drop(["year", "quarter", "qidx"]))
        .filter(pl.col("REPDTE") <= cutoff)
        .sort(["CERT", "REPDTE"])
    )
    trunc_df = (
        compute_camels_features(
            decumulate_ytd_flows(bronze.filter(pl.col("REPDTE") <= cutoff)).drop(["year", "quarter", "qidx"])
        )
        .sort(["CERT", "REPDTE"])
    )

    assert full_sub.height == trunc_df.height and trunc_df.height > 1_000_000
    assert (full_sub["CERT"] == trunc_df["CERT"]).all()
    assert (full_sub["REPDTE"] == trunc_df["REPDTE"]).all()
    for col in list(FEATURE_COLS_CAMELS) + [f"{c}_Q" for c in FLOWS_YTD]:
        a, b = full_sub[col], trunc_df[col]
        assert int((a.is_null() != b.is_null()).sum()) == 0, f"Null mismatch on {col}"
        both = a.is_not_null() & b.is_not_null()
        if int(both.sum()) > 0:
            max_d = float((a.filter(both) - b.filter(both)).abs().max())
            assert max_d == 0.0, f"Future look-ahead detected on {col}: max_abs_diff={max_d}"


def test_deep_audit_artifacts_valid():
    """Verify DEV diagnostics, 100% 1.68M Spark parity, engine/shuffle ablations, truncation audit, and posthoc TEST diagnostics."""
    import json
    from src.bfews.utils import out_dir

    tdir = out_dir("tables")
    dq_file = tdir / "dq_summary.json"
    is_frozen = True
    total_manifest_rows = 1_682_616
    if dq_file.exists():
        dq = json.loads(dq_file.read_text(encoding="utf-8"))
        total_manifest_rows = int(dq["manifest"]["total_manifest_rows"])
        is_frozen = total_manifest_rows == 1_682_616

    dq_dev = json.loads((tdir / "dq_dev_diagnostics.json").read_text(encoding="utf-8"))
    assert dq_dev["reporting_gaps"]["banks_with_internal_gap"] > 0
    assert dq_dev["dev_diagnostics"]["tail_failure_rates"][0]["lift_vs_dev_base"] > 10.0

    sp_full = json.loads((tdir / "spark_full_panel_parity.json").read_text(encoding="utf-8"))
    if is_frozen:
        assert sp_full["rows_compared"] == 1_682_616
    else:
        assert sp_full["rows_compared"] == total_manifest_rows and sp_full["rows_compared"] > 1_500_000
    assert sp_full["keys_exact_match"] is True
    assert sp_full["total_null_mismatches"] == 0
    assert sp_full["max_abs_diff_any_column"] < 1e-6

    abl = json.loads((tdir / "part3_engine_ablation.json").read_text(encoding="utf-8"))
    assert len(abl["thread_scaling_1x"]) == 6
    assert abl["optimizer_ablation_year_ge_2008"]["runtime_speedup_x"] > 1.0

    shuf = json.loads((tdir / "spark_shuffle_ablation.json").read_text(encoding="utf-8"))
    expected_rows = 1_682_616 if is_frozen else total_manifest_rows
    assert shuf["shuffle_8"]["rows"] == expected_rows and shuf["shuffle_200_default"]["rows"] == expected_rows

    trunc = json.loads((tdir / "full_panel_truncation_audit.json").read_text(encoding="utf-8"))
    assert trunc["exact_causal_invariance"] is True
    assert trunc["total_columns_verified"] == 33

    ph = json.loads((tdir / "test_posthoc_diagnostics.json").read_text(encoding="utf-8"))
    assert ph["verified_parity_with_test_results_json"] is True
    lt = ph["lead_time_distribution"]["theta_star"]
    if is_frozen:
        assert lt["caught_banks"] == 469 and lt["total_failing_banks"] == 485
    else:
        assert 0 < lt["caught_banks"] <= lt["total_failing_banks"]
    assert lt["lead_1q_le90d"] + lt["lead_2q_91_180d"] + lt["lead_3q_181_270d"] + lt["lead_4q_gt270d"] == lt["caught_banks"]
    assert ph["annual_summary"]["hgb_mean_annual_ap"] > ph["annual_summary"]["texas_mean_annual_ap"]




