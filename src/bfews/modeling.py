"""Leak-free validation, capacity & bank-level cost-based alarm thresholds, and the single held-out test (Part 5)."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
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


class PriorOddsCalibrator(BaseEstimator, ClassifierMixin):
    """Wrap a `class_weight='balanced'` classifier and map balanced posteriors back to the training prior (Elkan 2001).

    Under `class_weight='balanced'`, the training prior is shifted to 0.5. Given the empirical training prior
    `pi = mean(y_train)` learned strictly in `fit(X, y)`, Bayes' rule maps balanced probabilities `p_bal` to
    calibrated probabilities on the natural prior:
        p_cal = (pi * p_bal) / (pi * p_bal + (1 - pi) * (1 - p_bal))
    This transformation is strictly monotonic (preserving within-fold and test-set rankings) while restoring well-calibrated
    probabilities on the rare-event scale.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=np.float64)
        self.classes_ = np.array([0, 1])
        self.pi_ = float(np.clip(y_arr.mean(), 1e-6, 1.0 - 1e-6))
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict_proba(self, X):
        p_bal = np.clip(self.estimator_.predict_proba(X)[:, 1], 1e-9, 1.0 - 1e-9)
        num = self.pi_ * p_bal
        den = num + (1.0 - self.pi_) * (1.0 - p_bal)
        p_cal = num / den
        return np.column_stack([1.0 - p_cal, p_cal])


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
        base_lr = LogisticRegression(class_weight="balanced", C=0.1, max_iter=300, random_state=SEED)
        return Pipeline([("winsor", RobustWinsorizer()), ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()),
                         ("model", PriorOddsCalibrator(base_lr))])
    base_hgb = HistGradientBoostingClassifier(max_iter=120, max_depth=5, learning_rate=0.05,
                                              min_samples_leaf=50, class_weight="balanced",
                                              random_state=SEED)
    return Pipeline([("winsor", RobustWinsorizer()),
                     ("model", PriorOddsCalibrator(base_hgb))])


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
# Bank-level cost model & capacity operating points
# ------------------------------------------------------------------------------------------------------------------
def resolution_loss_rate() -> float:
    """Median FDIC resolution loss as a share of the failed bank's assets, from development-period failures only.

    Using COST / QBFASSET (both in thousand USD) removes inflation and bank size, so it can be applied to any bank.
    """
    dev_end = cfg()["split"]["dev_end"]
    f = (parse_failures().filter(pl.col("FAILDATE") <= pl.lit(dev_end).str.to_date("%Y-%m-%d"))
         .filter((pl.col("COST") > 0) & (pl.col("QBFASSET") > 0)))
    return float((f["COST"] / f["QBFASSET"]).median())


def build_theta_grid() -> np.ndarray:
    g = cfg()["cost"].get("theta_grid", {"min": 0.0001, "max": 0.90, "n": 300, "scale": "log"})
    lo, hi, n = float(g["min"]), float(g["max"]), int(g["n"])
    if g.get("scale", "log") == "log":
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def count_false_alarm_episodes(cert: np.ndarray, repdte: np.ndarray, y: np.ndarray, flag: np.ndarray) -> int:
    """Count distinct false-alarm episodes among negative rows (y == 0).

    For a non-failing bank, consecutive flagged quarters (qidx_t - qidx_{t-1} == 1) constitute a single continuous
    supervisory examination episode rather than independent off-cycle exam deployments.
    """
    neg = y == 0
    if not np.any(neg):
        return 0
    d = pl.DataFrame({
        "cert": cert[neg],
        "repdte": repdte[neg],
        "flag": flag[neg].astype(np.int8),
    }).sort(["cert", "repdte"]).with_columns(
        (pl.col("repdte").dt.year() * 4 + (pl.col("repdte").dt.month() - 1) // 3 + 1).alias("qidx"),
    ).with_columns(
        pl.col("flag").shift(1).over("cert").fill_null(0).alias("flag_prev"),
        pl.col("qidx").shift(1).over("cert").fill_null(-99).alias("q_prev"),
    )
    new_ep = (d["flag"] == 1) & ~((d["flag_prev"] == 1) & ((d["qidx"] - d["q_prev"]) == 1))
    return int(new_ep.sum())


class BankCostModel:
    """Expected cost of an alarm rule 'flag a bank-quarter if p >= theta'.

    * false alarm  : when `episode_cost=True`, a continuous run of flagged quarters on a non-failing bank (`y = 0`)
      costs `fp_cost` (one targeted examination per episode);
    * missed failure: a failing bank is caught if ANY of its positive rows (the 4 quarters before failure) is
      flagged; otherwise it costs `loss_rate x min(assets, cap_asset)` (assets capped at `asset_cap_quantile` of
      development failing banks so mega-banks with resident examiners do not distort the off-cycle threshold).
    """

    def __init__(self, cert, repdte, y, p, assets, loss_rate: float,
                 episode_cost: bool | None = None, asset_cap_quantile: float | None = None):
        cost_cfg = cfg()["cost"]
        self.episode_cost = bool(cost_cfg.get("episode_cost", True) if episode_cost is None else episode_cost)
        cap_q = float(cost_cfg.get("asset_cap_quantile", 0.85) if asset_cap_quantile is None else asset_cap_quantile)

        pos = y == 1
        self.neg_sorted = np.sort(p[~pos])
        neg_df = pl.DataFrame({
            "cert": cert[~pos],
            "repdte": repdte[~pos],
            "p": p[~pos],
        }).sort(["cert", "repdte"]).with_columns(
            (pl.col("repdte").dt.year() * 4 + (pl.col("repdte").dt.month() - 1) // 3 + 1).alias("qidx"),
        ).with_columns(
            pl.col("p").shift(1).over("cert").fill_null(-1.0).alias("p_prev"),
            pl.col("qidx").shift(1).over("cert").fill_null(-99).alias("q_prev"),
        )
        self.neg_p_curr = neg_df["p"].to_numpy()
        self.neg_p_prev = neg_df["p_prev"].to_numpy()
        self.neg_consec = ((neg_df["qidx"] - neg_df["q_prev"]) == 1).to_numpy()

        d = (pl.DataFrame({"cert": cert[pos], "p": p[pos], "a": assets[pos]})
             .group_by("cert", maintain_order=True).agg(pl.col("p").max().alias("pmax"), pl.col("a").first()))
        self.bank_pmax = d["pmax"].to_numpy()
        raw_assets = d["a"].to_numpy()
        self.asset_cap_kusd = float(np.quantile(raw_assets, cap_q)) if cap_q < 1.0 else float(raw_assets.max())
        self.bank_cost = np.minimum(raw_assets, self.asset_cap_kusd) * loss_rate
        self.n_banks = len(self.bank_pmax)

    def fp_rows(self, theta: float) -> int:
        return int(len(self.neg_sorted) - np.searchsorted(self.neg_sorted, theta, side="left"))

    def fp_episodes(self, theta: float) -> int:
        return int(((self.neg_p_curr >= theta) & ~(self.neg_consec & (self.neg_p_prev >= theta))).sum())

    def fp(self, theta: float) -> int:
        return self.fp_episodes(theta) if self.episode_cost else self.fp_rows(theta)

    def missed(self, theta: float) -> tuple[int, float]:
        miss = self.bank_pmax < theta
        return int(miss.sum()), float(self.bank_cost[miss].sum())

    def total_cost(self, theta: float, fp_cost: float) -> float:
        return self.fp(theta) * fp_cost + self.missed(theta)[1]

    def best_theta(self, fp_cost: float, grid=None) -> tuple[float, float, bool]:
        grid = build_theta_grid() if grid is None else grid
        costs = [self.total_cost(float(t), fp_cost) for t in grid]
        i = int(np.argmin(costs))
        at_edge = (i == 0) or (i == len(grid) - 1)
        if at_edge:
            log.warning(f"Optimal theta={grid[i]:.6f} hit grid boundary (index {i}/{len(grid) - 1}) for C_FP={fp_cost}")
        return float(grid[i]), float(costs[i]), bool(at_edge)


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
    folds, oof = [], {"cert": [], "repdte": [], "y": [], "p": [], "asset": []}
    for y0, y1 in cfg()["split"]["cv_blocks"]:
        tr, va = dev.filter(year < y0 - 1), dev.filter((year >= y0) & (year <= y1))
        if tr.height < 5000 or va.height == 0:
            continue
        model = make_model(kind).fit(tr.select(features).to_numpy(), tr["target_fail_4q"].to_numpy())
        p = model.predict_proba(va.select(features).to_numpy())[:, 1]
        y = va["target_fail_4q"].to_numpy()
        folds.append({**evaluate_predictions(y, p), "val_years": f"{y0}-{y1}",
                      "failing_banks": int(va.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique())})
        oof["cert"].append(va["CERT"].to_numpy())
        oof["repdte"].append(va["REPDTE"].to_numpy())
        oof["y"].append(y)
        oof["p"].append(p)
        oof["asset"].append(va["ASSET"].to_numpy())
        del tr, va, model
        gc.collect()
    return folds, {k: np.concatenate(v) for k, v in oof.items()}


def summarize_flag_rule(df: pl.DataFrame, p: np.ndarray, flag: np.ndarray) -> dict[str, Any]:
    """Summarize bank-level and row-level supervisory performance for any boolean alarm vector `flag`."""
    d = df.select("CERT", "REPDTE", "FAILDATE", "target_fail_4q").with_columns(
        pl.Series("p", p), pl.Series("flag", flag.astype(bool))
    )
    pos = d.filter(pl.col("target_fail_4q") == 1)
    per_bank = (pos.group_by("CERT")
                .agg(pl.col("FAILDATE").first(),
                     pl.col("flag").any().alias("caught"),
                     pl.col("REPDTE").filter(pl.col("flag")).min().alias("first_alarm")))
    caught = per_bank.filter(pl.col("caught"))
    lead = (caught["FAILDATE"] - caught["first_alarm"]).dt.total_days()

    y_arr = d["target_fail_4q"].to_numpy()
    flagged_neg = int((flag & (y_arr == 0)).sum())
    flagged_pos = int((flag & (y_arr == 1)).sum())
    total_flagged = flagged_neg + flagged_pos
    ep_fp = count_false_alarm_episodes(d["CERT"].to_numpy(), d["REPDTE"].to_numpy(), y_arr, flag)
    per_q_alerts = d.group_by("REPDTE").agg(pl.col("flag").sum().alias("alerts"))["alerts"]

    return {
        "failing_banks": per_bank.height,
        "failing_banks_caught": caught.height,
        "bank_recall": round(caught.height / max(1, per_bank.height), 4),
        "row_recall": round(flagged_pos / max(1, int(y_arr.sum())), 4),
        "precision_row": round(flagged_pos / max(1, total_flagged), 4),
        "median_lead_days": float(lead.median()) if len(lead) else None,
        "false_alarm_rows": flagged_neg,
        "false_alarm_episodes": ep_fp,
        "alerts_per_quarter_median": round(float(per_q_alerts.median()), 1),
        "flagged_share_of_rows_pct": round(float(flag.mean()) * 100, 3),
        "positive_rows_flagged": flagged_pos,
    }


def bank_level_test_summary(test: pl.DataFrame, p: np.ndarray, theta: float) -> dict[str, Any]:
    """Which failing banks were flagged at least once in their 4 pre-failure quarters, and how early."""
    return summarize_flag_rule(test, p, p >= theta)


def capacity_operating_points(df: pl.DataFrame, p: np.ndarray, budgets_pct: list[int] | None = None) -> list[dict[str, Any]]:
    """Evaluate fixed quarterly inspection budgets (flagging top k% highest-risk banks each quarter)."""
    budgets_pct = cfg()["cost"].get("alarm_budget_pct", [1, 2, 5]) if budgets_pct is None else budgets_pct
    d = df.select("REPDTE").with_columns(pl.Series("p", p), pl.arange(0, df.height).alias("idx"))
    # Rank within each quarter descending by p
    ranked = d.with_columns(
        pl.col("p").rank(method="ordinal", descending=True).over("REPDTE").alias("rk"),
        pl.len().over("REPDTE").alias("n_q"),
    ).sort("idx")
    rk = ranked["rk"].to_numpy()
    nq = ranked["n_q"].to_numpy()

    out = []
    for k in budgets_pct:
        quota = np.maximum(1, np.round(nq * (float(k) / 100.0)).astype(int))
        flag = rk <= quota
        out.append({"pct": int(k), **summarize_flag_rule(df, p, flag)})
    return out


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
    cm = BankCostModel(
        champion_oof["cert"], champion_oof["repdte"], champion_oof["y"],
        champion_oof["p"], champion_oof["asset"], loss_rate
    )
    fp_cost = float(cfg()["cost"]["false_alarm_cost_kusd"])
    theta, dev_cost, at_edge = cm.best_theta(fp_cost)
    miss_n, _ = cm.missed(theta)
    sens = {}
    sens_thetas = []
    for c in cfg()["cost"]["sensitivity_false_alarm_kusd"]:
        t, _, edge_c = cm.best_theta(float(c))
        sens_thetas.append(round(t, 4))
        sens[str(c)] = {
            "theta": round(t, 4),
            "theta_at_grid_edge": edge_c,
            "dev_banks_missed": cm.missed(t)[0],
            "dev_false_alarm_episodes": cm.fp_episodes(t),
            "dev_false_alarm_rows": cm.fp_rows(t),
        }
    if len(set(sens_thetas)) == 1:
        log.warning(f"All false-alarm costs yielded identical theta={sens_thetas[0]}")
    log.info(
        f"loss rate {loss_rate:.4f}; asset cap {cm.asset_cap_kusd:,.0f}k USD; "
        f"theta* = {theta:.4f} (C_FP = {fp_cost:,.0f}k USD, edge={at_edge})"
    )

    # Also compute capacity operating points on DEV OOF
    oof_df = pl.DataFrame({
        "CERT": champion_oof["cert"],
        "REPDTE": champion_oof["repdte"],
        "target_fail_4q": champion_oof["y"],
    }).join(dev.select(["CERT", "REPDTE", "FAILDATE"]), on=["CERT", "REPDTE"], how="left")
    dev_capacity = capacity_operating_points(oof_df, champion_oof["p"])

    # ---- the one and only evaluation on the test period ------------------------------------------------------------
    log.info("Fitting the champion on all DEV rows and scoring TEST once...")
    feats = FEATURE_COLS_CAMELS
    champion = make_model("hgb").fit(dev.select(feats).to_numpy(), dev["target_fail_4q"].to_numpy())
    p_test = champion.predict_proba(test.select(feats).to_numpy())[:, 1]
    y_test = test["target_fail_4q"].to_numpy()
    p_texas = (make_model("texas")
               .fit(dev.select(["texas_ratio"]).to_numpy(), dev["target_fail_4q"].to_numpy())
               .predict_proba(test.select(["texas_ratio"]).to_numpy())[:, 1])

    years = test["REPDTE"].dt.year().to_numpy()
    gfc, modern = (years <= 2014), (years >= 2015)
    dev_all_failing = int(dev.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique())
    test_eval = {
        **evaluate_predictions(y_test, p_test),
        "texas_rule": evaluate_predictions(y_test, p_texas),
        "threshold": round(theta, 4),
        "at_threshold": bank_level_test_summary(test, p_test, theta),
        "capacity_points": capacity_operating_points(test, p_test),
        "sensitivity_to_false_alarm_cost": {
            c: {**v, **{k: bank_level_test_summary(test, p_test, v["theta"])[k]
                        for k in ("bank_recall", "false_alarm_rows", "false_alarm_episodes",
                                  "alerts_per_quarter_median", "flagged_share_of_rows_pct")}}
            for c, v in sens.items()},
        "gfc_2009_2014": {**evaluate_predictions(y_test[gfc], p_test[gfc]),
                          "failing_banks": int(test.filter(pl.col("target_fail_4q") == 1).filter(pl.col("REPDTE").dt.year() <= 2014)["CERT"].n_unique())},
        "modern_2015_2025": {**evaluate_predictions(y_test[modern], p_test[modern]),
                             "failing_banks": int(test.filter(pl.col("target_fail_4q") == 1).filter(pl.col("REPDTE").dt.year() >= 2015)["CERT"].n_unique())},
        "failing_banks_total": int(test.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique()),
        "dev_failing_banks_total": dev_all_failing,
        "dev_failing_banks_all": dev_all_failing,
        "dev_failing_banks_in_oof_blocks": cm.n_banks,
    }
    results = {
        "cross_validation_models": cv,
        "cost_threshold_tuning": {
            "resolution_loss_rate": round(loss_rate, 4),
            "false_alarm_cost_kusd": fp_cost,
            "episode_cost": cm.episode_cost,
            "asset_cap_kusd": round(cm.asset_cap_kusd, 0),
            "theta": round(theta, 4),
            "theta_at_grid_edge": at_edge,
            "expected_cost_dev_kusd": round(dev_cost, 0),
            "dev_banks_missed_at_theta": miss_n,
            "dev_oof_banks_caught_at_theta": cm.n_banks - miss_n,
            "dev_failing_banks": cm.n_banks,
            "dev_failing_banks_in_oof_blocks": cm.n_banks,
            "dev_failing_banks_all": dev_all_failing,
            "dev_oof_capacity_points": dev_capacity,
            "sensitivity": sens,
        },
        "held_out_test_evaluation": test_eval,
    }
    save_json(results, out_dir("tables") / "test_results.json")
    log.info("Saved outputs/tables/test_results.json")
    return results
