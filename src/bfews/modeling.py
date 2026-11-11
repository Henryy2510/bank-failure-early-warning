"""Leak-free validation, bank-level cost-based alarm threshold, and the single held-out test (Part 5)."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import texas_cap
from .features import FEATURE_COLS_CAMELS
from .io import parse_failures
from .utils import cfg, data_dir, get_logger, out_dir, save_json

log = get_logger("modeling")
SEED = 42


class RobustWinsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at training-set quantiles and drop columns that carry no information in the training set.

    Everything is learned in `fit` (training data only). Columns with fewer than two distinct observed values in
    the training data (for example the Basel ratios, which do not exist before 1990, in an early fold) are dropped
    in `transform`, so no dummy values have to be invented.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.where(np.isfinite(np.asarray(X, dtype=np.float32)), np.asarray(X, dtype=np.float32), np.nan)
        n = X.shape[1]
        self.keep_ = np.zeros(n, dtype=bool)
        self.lo_ = np.full(n, np.nan, dtype=np.float32)
        self.hi_ = np.full(n, np.nan, dtype=np.float32)
        for j in range(n):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if len(col) and np.unique(col).size >= 2:
                self.keep_[j] = True
                self.lo_[j], self.hi_[j] = np.quantile(col, [self.lower_quantile, self.upper_quantile])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = np.where(np.isfinite(X), X, np.nan)[:, self.keep_]
        return np.clip(X, self.lo_[self.keep_], self.hi_[self.keep_])  # NaN stays NaN (imputed or handled by the model)


class TexasRatioBaseline(BaseEstimator, ClassifierMixin):
    """Supervisory rule: rank banks by the (capped) Texas ratio. Nothing is fitted."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        t = np.nan_to_num(np.asarray(X, dtype=np.float32)[:, 0], nan=0.0)
        p = np.clip(t / texas_cap(), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def make_model(kind: str):
    if kind == "texas":
        return Pipeline([("model", TexasRatioBaseline())])
    if kind == "logit":
        return Pipeline([("winsor", RobustWinsorizer()), ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()),
                         ("model", LogisticRegression(class_weight="balanced", C=0.1, max_iter=300, random_state=SEED))])
    return Pipeline([("winsor", RobustWinsorizer()),
                     ("model", HistGradientBoostingClassifier(max_iter=120, max_depth=5, learning_rate=0.05,
                                                              min_samples_leaf=50, class_weight="balanced",
                                                              random_state=SEED))])


def evaluate_predictions(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Row-level metrics for rare events: PR-AUC (main), ROC-AUC, recall in the top 1% / 5% of scores, Brier."""
    both = len(np.unique(y)) > 1
    order = np.argsort(-p, kind="stable")
    n, pos = len(p), int(y.sum())
    top = lambda q: float(y[order[: max(1, int(n * q))]].sum() / max(1, pos))  # noqa: E731
    return {"pr_auc": round(float(average_precision_score(y, p)) if both else 0.0, 4),
            "roc_auc": round(float(roc_auc_score(y, p)) if both else 0.5, 4),
            "recall_top_1pct": round(top(0.01), 4), "recall_top_5pct": round(top(0.05), 4),
            "brier_score": round(float(brier_score_loss(y, p)), 5),
            "positive_rows": pos, "rows": n}


# ------------------------------------------------------------------------------------------------------------------
# Bank-level cost model
# ------------------------------------------------------------------------------------------------------------------
def resolution_loss_rate() -> float:
    """Median FDIC resolution loss as a share of the failed bank's assets, from development-period failures only.

    Using COST / QBFASSET (both in thousand USD) removes inflation and bank size, so it can be applied to any bank.
    """
    dev_end = cfg()["split"]["dev_end"]
    f = (parse_failures().filter(pl.col("FAILDATE") <= pl.lit(dev_end).str.to_date("%Y-%m-%d"))
         .filter((pl.col("COST") > 0) & (pl.col("QBFASSET") > 0)))
    return float((f["COST"] / f["QBFASSET"]).median())


class BankCostModel:
    """Expected cost of an alarm rule 'flag a bank-quarter if p >= theta'.

    * false alarm  : every flagged bank-quarter with y = 0 costs `fp_cost` (one targeted examination);
    * missed failure: a failing bank is caught if ANY of its positive rows (the 4 quarters before failure) is
      flagged; otherwise it costs `loss_rate x assets` (assets at its first positive row), one cost per BANK,
      not per row.
    """

    def __init__(self, cert, y, p, assets, loss_rate: float):
        pos = y == 1
        self.neg_sorted = np.sort(p[~pos])
        d = (pl.DataFrame({"cert": cert[pos], "p": p[pos], "a": assets[pos]})
             .group_by("cert", maintain_order=True).agg(pl.col("p").max().alias("pmax"), pl.col("a").first()))
        self.bank_pmax = d["pmax"].to_numpy()
        self.bank_cost = d["a"].to_numpy() * loss_rate
        self.n_banks = len(self.bank_pmax)

    def fp(self, theta: float) -> int:
        return int(len(self.neg_sorted) - np.searchsorted(self.neg_sorted, theta, side="left"))

    def missed(self, theta: float) -> tuple[int, float]:
        miss = self.bank_pmax < theta
        return int(miss.sum()), float(self.bank_cost[miss].sum())

    def total_cost(self, theta: float, fp_cost: float) -> float:
        return self.fp(theta) * fp_cost + self.missed(theta)[1]

    def best_theta(self, fp_cost: float, grid=None) -> tuple[float, float]:
        grid = np.linspace(0.01, 0.99, 197) if grid is None else grid
        costs = [self.total_cost(t, fp_cost) for t in grid]
        i = int(np.argmin(costs))
        return float(grid[i]), float(costs[i])


# ------------------------------------------------------------------------------------------------------------------
def load_modeling_frame(gold_path: Path | None = None) -> pl.DataFrame:
    """DEV and TEST rows that could have been scored in real time (pit_scorable), features as Float32."""
    gold_path = gold_path or data_dir("gold") / "features.parquet"
    cols = ["CERT", "REPDTE", "split_group", "target_fail_4q", "pit_scorable", "ASSET", "FAILDATE"] + FEATURE_COLS_CAMELS
    return (pl.scan_parquet(gold_path)
            .filter(pl.col("split_group").is_in(["DEV", "TEST"]) & pl.col("pit_scorable"))
            .select(cols)
            .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS_CAMELS])
            .collect())


def run_walk_forward_cv(dev: pl.DataFrame, features: list[str], kind: str):
    """Expanding-window validation in the development period.

    For a validation block starting in year Y the model is trained on quarters dated before Y-1; one year is purged
    so that no training label window (4 quarters ahead) reaches into the validation block.
    """
    year = dev["REPDTE"].dt.year()
    folds, oof = [], {"cert": [], "y": [], "p": [], "asset": []}
    for y0, y1 in cfg()["split"]["cv_blocks"]:
        tr, va = dev.filter(year < y0 - 1), dev.filter((year >= y0) & (year <= y1))
        if tr.height < 5000 or va.height == 0:
            continue
        model = make_model(kind).fit(tr.select(features).to_numpy(), tr["target_fail_4q"].to_numpy())
        p = model.predict_proba(va.select(features).to_numpy())[:, 1]
        y = va["target_fail_4q"].to_numpy()
        folds.append({**evaluate_predictions(y, p), "val_years": f"{y0}-{y1}",
                      "failing_banks": int(va.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique())})
        oof["cert"].append(va["CERT"].to_numpy()); oof["y"].append(y); oof["p"].append(p); oof["asset"].append(va["ASSET"].to_numpy())
        del tr, va, model
        gc.collect()
    return folds, {k: np.concatenate(v) for k, v in oof.items()}


def bank_level_test_summary(test: pl.DataFrame, p: np.ndarray, theta: float) -> dict[str, Any]:
    """Which failing banks were flagged at least once in their 4 pre-failure quarters, and how early."""
    d = test.select("CERT", "REPDTE", "FAILDATE", "target_fail_4q").with_columns(pl.Series("p", p))
    pos = d.filter(pl.col("target_fail_4q") == 1)
    per_bank = (pos.group_by("CERT")
                .agg(pl.col("FAILDATE").first(),
                     (pl.col("p") >= theta).any().alias("caught"),
                     pl.col("REPDTE").filter(pl.col("p") >= theta).min().alias("first_alarm")))
    caught = per_bank.filter(pl.col("caught"))
    lead = (caught["FAILDATE"] - caught["first_alarm"]).dt.total_days()
    flagged_neg = int(((d["p"] >= theta) & (d["target_fail_4q"] == 0)).sum())
    return {"failing_banks": per_bank.height, "failing_banks_caught": caught.height,
            "bank_recall": round(caught.height / max(1, per_bank.height), 4),
            "median_lead_days": float(lead.median()) if len(lead) else None,
            "false_alarm_rows": flagged_neg, "flagged_share_of_rows_pct": round(float((d["p"] >= theta).mean()) * 100, 3),
            "positive_rows_flagged": int(((d["p"] >= theta) & (d["target_fail_4q"] == 1)).sum())}


def execute_full_modeling_workflow(gold_path: Path | None = None) -> dict[str, Any]:
    df = load_modeling_frame(gold_path)
    dev = df.filter(pl.col("split_group") == "DEV").sort("REPDTE")
    test = df.filter(pl.col("split_group") == "TEST").sort("REPDTE")
    del df
    gc.collect()
    log.info(f"DEV rows {dev.height:,} | TEST rows {test.height:,} (pit_scorable only)")

    candidates = {
        "Texas ratio rule": (["texas_ratio"], "texas"),
        "Logistic, 3 variables": (["eq_asset_ratio", "nclnls_ratio", "roa_ytd"], "logit"),
        "Logistic, 26 CAMELS": (FEATURE_COLS_CAMELS, "logit"),
        "HistGradientBoosting, 26 CAMELS": (FEATURE_COLS_CAMELS, "hgb"),
    }
    cv, champion_oof = {}, None
    for name, (feats, kind) in candidates.items():
        log.info(f"Walk-forward CV: {name}")
        folds, oof = run_walk_forward_cv(dev, feats, kind)
        cv[name] = {**evaluate_predictions(oof["y"], oof["p"]), "folds": folds}
        if name == "HistGradientBoosting, 26 CAMELS":
            champion_oof = oof

    # ---- cost-based alarm threshold, decided on development out-of-fold predictions only -------------------------
    loss_rate = resolution_loss_rate()
    cm = BankCostModel(champion_oof["cert"], champion_oof["y"], champion_oof["p"], champion_oof["asset"], loss_rate)
    fp_cost = float(cfg()["cost"]["false_alarm_cost_kusd"])
    theta, dev_cost = cm.best_theta(fp_cost)
    miss_n, _ = cm.missed(theta)
    sens = {}
    for c in cfg()["cost"]["sensitivity_false_alarm_kusd"]:
        t, _ = cm.best_theta(float(c))
        sens[str(c)] = {"theta": round(t, 3), "dev_banks_missed": cm.missed(t)[0], "dev_false_alarm_rows": cm.fp(t)}
    log.info(f"loss rate {loss_rate:.3f}; theta* = {theta:.3f} (C_FP = {fp_cost:,.0f}k USD)")

    # ---- the one and only evaluation on the test period ------------------------------------------------------------
    log.info("Fitting the champion on all DEV rows and scoring TEST once...")
    feats = FEATURE_COLS_CAMELS
    champion = make_model("hgb").fit(dev.select(feats).to_numpy(), dev["target_fail_4q"].to_numpy())
    p_test = champion.predict_proba(test.select(feats).to_numpy())[:, 1]
    y_test = test["target_fail_4q"].to_numpy()
    p_texas = make_model("texas").fit(dev.select(["texas_ratio"]).to_numpy(), dev["target_fail_4q"].to_numpy()).predict_proba(test.select(["texas_ratio"]).to_numpy())[:, 1]

    years = test["REPDTE"].dt.year().to_numpy()
    gfc, modern = (years <= 2014), (years >= 2015)
    test_eval = {
        **evaluate_predictions(y_test, p_test),
        "texas_rule": evaluate_predictions(y_test, p_texas),
        "threshold": round(theta, 4),
        "at_threshold": bank_level_test_summary(test, p_test, theta),
        "sensitivity_to_false_alarm_cost": {
            c: {**v, **{k: bank_level_test_summary(test, p_test, v["theta"])[k]
                        for k in ("bank_recall", "false_alarm_rows", "flagged_share_of_rows_pct")}}
            for c, v in sens.items()},
        "gfc_2009_2014": {**evaluate_predictions(y_test[gfc], p_test[gfc]),
                          "failing_banks": int(test.filter(pl.col("target_fail_4q") == 1).filter(pl.col("REPDTE").dt.year() <= 2014)["CERT"].n_unique())},
        "modern_2015_2025": {**evaluate_predictions(y_test[modern], p_test[modern]),
                             "failing_banks": int(test.filter(pl.col("target_fail_4q") == 1).filter(pl.col("REPDTE").dt.year() >= 2015)["CERT"].n_unique())},
        "failing_banks_total": int(test.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique()),
        "dev_failing_banks_total": int(dev.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique()),
    }
    results = {
        "cross_validation_models": cv,
        "cost_threshold_tuning": {"resolution_loss_rate": round(loss_rate, 4), "false_alarm_cost_kusd": fp_cost,
                                  "theta": round(theta, 4), "expected_cost_dev_kusd": round(dev_cost, 0),
                                  "dev_banks_missed_at_theta": miss_n, "dev_failing_banks": cm.n_banks,
                                  "sensitivity": sens},
        "held_out_test_evaluation": test_eval,
    }
    save_json(results, out_dir("tables") / "test_results.json")
    log.info("Saved outputs/tables/test_results.json")
    return results
