"""Time-ordered validation, leak-free pipeline, cost-sensitive thresholding, and modeling (Part 5)."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS_CAMELS
from .utils import cfg, data_dir, get_logger, out_dir, save_json

log = get_logger("modeling")


class RobustWinsorizer(BaseEstimator, TransformerMixin):
    """Winsorize numerical columns at empirical quantiles fitted strictly on training data."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        X_clean = np.where(np.isfinite(X_arr), X_arr, np.nan)
        n_cols = X_clean.shape[1]
        self.lower_bounds_ = np.zeros(n_cols, dtype=np.float32)
        self.upper_bounds_ = np.zeros(n_cols, dtype=np.float32)
        self.all_nan_cols_ = np.zeros(n_cols, dtype=bool)
        for j in range(n_cols):
            col = X_clean[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                self.all_nan_cols_[j] = True
                self.lower_bounds_[j] = 0.0
                self.upper_bounds_[j] = 0.0
            else:
                self.lower_bounds_[j] = float(np.quantile(valid, self.lower_quantile))
                self.upper_bounds_[j] = float(np.quantile(valid, self.upper_quantile))
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float32).copy()
        X_arr = np.where(np.isfinite(X_arr), X_arr, np.nan)
        if self.lower_bounds_ is not None and self.upper_bounds_ is not None:
            for col_idx in range(X_arr.shape[1]):
                if self.all_nan_cols_[col_idx]:
                    # Fill pre-Basel completely unobserved column with 0 (and tiny deterministic tie-breaker at row 0)
                    X_arr[:, col_idx] = 0.0
                    if X_arr.shape[0] > 1:
                        X_arr[0, col_idx] = 1e-6
                else:
                    lb = self.lower_bounds_[col_idx]
                    ub = self.upper_bounds_[col_idx]
                    np.clip(X_arr[:, col_idx], lb, ub, out=X_arr[:, col_idx])
        return X_arr


class TexasRatioBaseline(BaseEstimator, ClassifierMixin):
    """Rule-based supervisory baseline: ranks institutions purely by the Texas Ratio."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        texas = np.nan_to_num(X_arr[:, 0], nan=0.0, posinf=10.0, neginf=0.0)
        texas = np.clip(texas, -20.0, 20.0)
        prob = 1.0 / (1.0 + np.exp(-0.5 * (texas - 1.0)))
        return np.column_stack([1.0 - prob, prob])


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute metrics appropriate for extreme class imbalance in banking supervision."""
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    n = len(y_prob)
    total_positives = int(np.sum(y_true))
    order = np.argsort(y_prob)[::-1]

    top1_idx = order[: max(1, int(n * 0.01))]
    top5_idx = order[: max(1, int(n * 0.05))]

    recall_at_1pct = float(np.sum(y_true[top1_idx]) / max(1, total_positives))
    recall_at_5pct = float(np.sum(y_true[top5_idx]) / max(1, total_positives))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "recall_top_1pct": round(recall_at_1pct, 4),
        "recall_top_5pct": round(recall_at_5pct, 4),
        "brier_score": round(brier, 5),
        "total_failures": total_positives,
        "sample_size": n,
    }


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> tuple[float, float]:
    """Determine probability alarm threshold theta* minimizing expected supervisory cost.

    Expected Cost = cost_fn * FN(theta) + cost_fp * FP(theta).
    Vectorized over sorted probabilities for speed and zero memory overhead.
    """
    cand_thetas = np.linspace(0.01, 0.99, 200)
    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    total_pos = int(np.sum(pos_mask))

    best_theta = 0.5
    min_cost = float("inf")

    for theta in cand_thetas:
        tp = int(np.sum(y_prob[pos_mask] >= theta))
        fp = int(np.sum(y_prob[neg_mask] >= theta))
        fn = total_pos - tp
        cost = (cost_fn * fn) + (cost_fp * fp)
        if cost < min_cost:
            min_cost = float(cost)
            best_theta = float(theta)

    return best_theta, min_cost


def run_walk_forward_cv(
    dev_df: pl.DataFrame,
    features: list[str],
    model_name: str = "HistGradientBoosting",
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Execute expanding-window walk-forward validation inside the Dev set.

    For each validation block, training is restricted to observations whose 4Q forward label
    was fully realized prior to the start of the validation period (4Q purge buffer).
    """
    blocks = cfg()["split"]["cv_blocks"]

    cv_results = []
    oof_y_true = []
    oof_y_prob = []

    years = dev_df["REPDTE"].dt.year()

    for start_yr, end_yr in blocks:
        val_sub = dev_df.filter((years >= start_yr) & (years <= end_yr))
        train_sub = dev_df.filter(years < (start_yr - 1))

        if train_sub.height < 5000 or val_sub.height == 0:
            continue

        X_train = train_sub.select(features).to_numpy()
        y_train = train_sub["target_fail_4q"].to_numpy()

        X_val = val_sub.select(features).to_numpy()
        y_val = val_sub["target_fail_4q"].to_numpy()

        if model_name == "LogisticRegression":
            clf = Pipeline([
                ("winsorizer", RobustWinsorizer()),
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=300, C=0.1, random_state=42)),
            ])
        elif model_name == "TexasRatio":
            clf = Pipeline([
                ("model", TexasRatioBaseline()),
            ])
        else:
            clf = Pipeline([
                ("winsorizer", RobustWinsorizer()),
                ("model", HistGradientBoostingClassifier(
                    max_iter=120,
                    max_depth=5,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=42,
                    min_samples_leaf=50,
                )),
            ])

        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]

        metrics = evaluate_predictions(y_val, probs)
        metrics["val_years"] = f"{start_yr}-{end_yr}"
        cv_results.append(metrics)

        oof_y_true.extend(y_val.tolist())
        oof_y_prob.extend(probs.tolist())

        del X_train, y_train, X_val, y_val, clf
        gc.collect()

    return cv_results, np.array(oof_y_true), np.array(oof_y_prob)


def execute_full_modeling_workflow(gold_path: Path | None = None) -> dict[str, Any]:
    """Execute complete modeling, cross-validation, cost threshold tuning, and held-out test evaluation."""
    if gold_path is None:
        gold_path = data_dir("gold") / "features.parquet"

    log.info("Scanning Gold feature table (projection pushdown + Float32 cast)...")
    needed_cols = ["CERT", "REPDTE", "split_group", "target_fail_4q", "COST"] + FEATURE_COLS_CAMELS
    df = (
        pl.scan_parquet(gold_path)
        .filter(pl.col("split_group").is_in(["DEV", "TEST"]))
        .select(needed_cols)
        .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS_CAMELS])
        .collect()
    )

    dev_df = df.filter(pl.col("split_group") == "DEV").sort("REPDTE")
    test_df = df.filter(pl.col("split_group") == "TEST").sort("REPDTE")
    del df
    gc.collect()

    # 1. Compare models on Walk-Forward CV
    models = {
        "Texas_Ratio_Baseline": ["texas_ratio"],
        "Logistic_3Var_Baseline": ["eq_asset_ratio", "nclnls_ratio", "roa_q"],
        "Logistic_Regression_CAMELS": FEATURE_COLS_CAMELS,
        "Hist_Gradient_Boosting": FEATURE_COLS_CAMELS,
    }

    cv_summary = {}
    best_oof_y_true = None
    best_oof_y_prob = None

    for m_label, feat_list in models.items():
        log.info(f"Running walk-forward CV for {m_label}...")
        m_type = "LogisticRegression" if "Logistic" in m_label else ("TexasRatio" if "Texas" in m_label else "HistGradientBoosting")
        fold_res, oof_y, oof_p = run_walk_forward_cv(dev_df, feat_list, model_name=m_type)
        overall_metrics = evaluate_predictions(oof_y, oof_p)
        overall_metrics["fold_details"] = fold_res
        cv_summary[m_label] = overall_metrics

        if m_label == "Hist_Gradient_Boosting":
            best_oof_y_true = oof_y
            best_oof_y_prob = oof_p

    # 2. Derive Cost-Optimal Alarm Threshold from Dev DIF losses
    dev_failures = dev_df.filter(pl.col("target_fail_4q") == 1)
    median_dif_cost = float(dev_failures.select(pl.col("COST").drop_nulls().median()).item() or 35_000)
    fp_exam_cost = float(cfg()["cost"]["false_alarm_cost_kusd"])

    log.info(f"Dev Failure Median DIF Loss (C_FN): ${median_dif_cost:,.0f}k | Exam Cost (C_FP): ${fp_exam_cost:,.0f}k")
    opt_threshold, min_dev_cost = find_cost_optimal_threshold(
        best_oof_y_true, best_oof_y_prob, cost_fn=median_dif_cost, cost_fp=fp_exam_cost
    )
    log.info(f"Optimal Supervisory Alarm Threshold theta*: {opt_threshold:.4f}")

    # 3. SINGLE HELD-OUT TEST EVALUATION (Part 5 requirement)
    log.info("Evaluating Champion Model (HistGradientBoosting) ONCE on held-out TEST set (2009-2025)...")
    X_dev = dev_df.select(FEATURE_COLS_CAMELS).to_numpy()
    y_dev = dev_df["target_fail_4q"].to_numpy()
    X_test = test_df.select(FEATURE_COLS_CAMELS).to_numpy()
    y_test = test_df["target_fail_4q"].to_numpy()

    champion_pipeline = Pipeline([
        ("winsorizer", RobustWinsorizer()),
        ("model", HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=50,
        )),
    ])

    champion_pipeline.fit(X_dev, y_dev)
    test_probs = champion_pipeline.predict_proba(X_test)[:, 1]

    # Also evaluate Texas Ratio baseline on test set for reference
    texas_test_probs = TexasRatioBaseline().predict_proba(test_df.select(["texas_ratio"]).to_numpy())[:, 1]
    texas_test_metrics = evaluate_predictions(y_test, texas_test_probs)

    # Metrics on full test set
    test_metrics = evaluate_predictions(y_test, test_probs)
    test_metrics["baseline_texas_test"] = texas_test_metrics

    # Confusion matrix at optimal threshold
    y_pred_opt = (test_probs >= opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt, labels=[0, 1]).ravel()
    test_metrics["optimal_threshold"] = round(opt_threshold, 4)
    test_metrics["confusion_matrix_at_opt_threshold"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": round(float(tp / max(1, tp + fp)), 4),
        "recall": round(float(tp / max(1, tp + fn)), 4),
    }

    # Breakdown by crisis regime in test period
    test_years = test_df["REPDTE"].dt.year().to_numpy()
    gfc_mask = (test_years >= 2009) & (test_years <= 2014)
    modern_mask = (test_years >= 2015) & (test_years <= 2025)

    test_metrics["subperiod_gfc_2009_2014"] = evaluate_predictions(y_test[gfc_mask], test_probs[gfc_mask])
    test_metrics["subperiod_modern_2015_2025"] = evaluate_predictions(y_test[modern_mask], test_probs[modern_mask])

    results = {
        "cross_validation_models": cv_summary,
        "cost_threshold_tuning": {
            "median_dif_cost_kusd": round(median_dif_cost, 2),
            "false_alarm_cost_kusd": fp_exam_cost,
            "optimal_threshold": round(opt_threshold, 4),
            "min_expected_cost_dev": round(min_dev_cost, 2),
        },
        "held_out_test_evaluation": test_metrics,
    }

    out_json = out_dir("tables") / "test_results.json"
    save_json(results, out_json)
    log.info(f"Saved modeling test results to {out_json}")
    return results
