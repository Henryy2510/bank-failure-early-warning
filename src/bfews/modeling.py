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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import texas_cap
from .features import FEATURE_COLS_CAMELS
from .io import parse_failures, parse_institutions
from .utils import cfg, data_dir, get_logger, out_dir, save_json

log = get_logger("modeling")
SEED = 42

CAMELS_DIMENSIONS: dict[str, str] = {
    "eq_asset_ratio": "C - Capital",
    "rbc1aaj_clean": "C - Capital",
    "rbcrwaj_clean": "C - Capital",
    "cap_distress_flag": "C - Capital",
    "nclnls_ratio": "A - Asset Quality",
    "naasset_ratio": "A - Asset Quality",
    "ore_ratio": "A - Asset Quality",
    "lnatres_ratio": "A - Asset Quality",
    "texas_ratio": "A - Asset Quality",
    "efficiency_ratio": "M - Management",
    "roa_ytd": "E - Earnings",
    "nimy_clean": "E - Earnings",
    "provision_asset_ratio": "E - Earnings",
    "netinc_q_asset_ratio": "E - Earnings",
    "liquid_asset_ratio": "L - Liquidity",
    "brokered_dep_ratio": "L - Liquidity",
    "loan_to_deposit_ratio": "L - Liquidity",
    "re_loan_share": "S - Sensitivity",
    "ci_loan_share": "S - Sensitivity",
    "con_loan_share": "S - Sensitivity",
    "log_asset": "Size & Dynamics",
    "asset_growth_4q": "Size & Dynamics",
    "loan_growth_4q": "Size & Dynamics",
    "d4_texas_ratio": "Size & Dynamics",
    "d4_nclnls_ratio": "Size & Dynamics",
    "d4_eq_ratio": "Size & Dynamics",
}


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
    cols = ["CERT", "REPDTE", "split_group", "target_fail_4q", "FAILTYPE", "pit_scorable", "ASSET", "FAILDATE"] + FEATURE_COLS_CAMELS
    return (pl.scan_parquet(gold_path)
            .filter(pl.col("split_group").is_in(["DEV", "TEST"]) & pl.col("pit_scorable"))
            .select(cols)
            .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS_CAMELS])
            .collect())


def _ap_scorer(estimator, X, y, sample_weight=None) -> float:
    """Weighted binary Average Precision (PR-AUC) scorer extracting positive-class probability."""
    p = estimator.predict_proba(X)[:, 1]
    return float(average_precision_score(y, p, sample_weight=sample_weight))


def _stratified_val_sample(X: np.ndarray, y: np.ndarray, max_neg: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep all positive rows and a deterministic negative subsample with inverse-probability weights."""
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(neg_idx) > max_neg:
        rng = np.random.default_rng(seed)
        neg_sub = rng.choice(neg_idx, size=max_neg, replace=False)
        w_neg = len(neg_idx) / float(max_neg)
    else:
        neg_sub = neg_idx
        w_neg = 1.0
    sub_idx = np.concatenate([pos_idx, neg_sub])
    w_sub = np.concatenate([np.ones(len(pos_idx), dtype=np.float64), np.full(len(neg_sub), w_neg, dtype=np.float64)])
    return X[sub_idx], y[sub_idx], w_sub


def run_walk_forward_cv(
    dev: pl.DataFrame,
    features: list[str],
    kind: str,
    explain_acc: list[dict[str, Any]] | None = None,
):
    """Expanding-window validation in the development period.

    For a validation block starting in year Y the model is trained on quarters dated before Y-1; one year is purged
    so that no training label window (4 quarters ahead) reaches into the validation block.
    """
    exp_cfg = cfg().get("explainability", {})
    perm_repeats = int(exp_cfg.get("perm_repeats", 5))
    max_neg = int(exp_cfg.get("max_neg_per_fold", 25000))

    year = dev["REPDTE"].dt.year()
    folds, oof = [], {"cert": [], "repdte": [], "y": [], "p": [], "asset": []}
    for y0, y1 in cfg()["split"]["cv_blocks"]:
        tr, va = dev.filter(year < y0 - 1), dev.filter((year >= y0) & (year <= y1))
        if tr.height < 5000 or va.height == 0:
            continue
        model = make_model(kind).fit(tr.select(features).to_numpy(), tr["target_fail_4q"].to_numpy())
        X_va = va.select(features).to_numpy()
        p = model.predict_proba(X_va)[:, 1]
        y = va["target_fail_4q"].to_numpy()
        n_pos = int(y.sum())
        folds.append({**evaluate_predictions(y, p), "val_years": f"{y0}-{y1}",
                      "failing_banks": int(va.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique())})
        oof["cert"].append(va["CERT"].to_numpy())
        oof["repdte"].append(va["REPDTE"].to_numpy())
        oof["y"].append(y)
        oof["p"].append(p)
        oof["asset"].append(va["ASSET"].to_numpy())

        if explain_acc is not None:
            fold_idx = len(folds)
            if kind == "logit":
                keep = model.named_steps["winsor"].keep_
                coefs = model.named_steps["model"].estimator_.coef_[0]
                row_coef = np.full(len(features), np.nan, dtype=np.float64)
                row_coef[keep] = coefs[: int(keep.sum())]
                explain_acc.append({"val_years": f"{y0}-{y1}", "pos": n_pos, "coef": row_coef})
            elif kind == "hgb":
                X_sub, y_sub, w_sub = _stratified_val_sample(X_va, y, max_neg=max_neg, seed=SEED + fold_idx)
                perm = permutation_importance(
                    model,
                    X_sub,
                    y_sub,
                    scoring=_ap_scorer,
                    n_repeats=perm_repeats,
                    n_jobs=1,
                    random_state=SEED + fold_idx,
                    sample_weight=w_sub,
                )
                explain_acc.append({
                    "val_years": f"{y0}-{y1}",
                    "pos": n_pos,
                    "imp_mean": perm.importances_mean,
                    "imp_std": perm.importances_std,
                })

        del tr, va, X_va, model
        gc.collect()
    return folds, {k: np.concatenate(v) for k, v in oof.items()}


def summarize_dev_feature_importance(
    features: list[str],
    hgb_folds_imp: list[dict[str, Any]],
    logit_folds_imp: list[dict[str, Any]],
) -> pl.DataFrame:
    """Aggregate strictly DEV out-of-fold HGB permutation importance and standardized Logit directionality."""
    hgb_weights = np.array([f["pos"] for f in hgb_folds_imp], dtype=np.float64)
    hgb_means = np.vstack([f["imp_mean"] for f in hgb_folds_imp])
    hgb_stds = np.vstack([f["imp_std"] for f in hgb_folds_imp])
    w_norm = hgb_weights / np.maximum(1.0, hgb_weights.sum())

    perm_mean = (hgb_means * w_norm[:, None]).sum(axis=0)
    perm_std = np.sqrt(((hgb_stds ** 2) * w_norm[:, None]).sum(axis=0))

    logit_weights = np.array([f["pos"] for f in logit_folds_imp], dtype=np.float64)
    logit_coefs = np.vstack([f["coef"] for f in logit_folds_imp])

    rows = []
    for j, feat in enumerate(features):
        col_coefs = logit_coefs[:, j]
        valid = np.isfinite(col_coefs)
        if np.any(valid):
            w_v = logit_weights[valid]
            beta = float(np.sum(col_coefs[valid] * w_v) / np.maximum(1.0, np.sum(w_v)))
            direction = "+ Increase Risk" if beta > 0 else "- Reduce Risk"
        else:
            beta = 0.0
            direction = "N/A"
        rows.append({
            "feature": feat,
            "camels_dim": CAMELS_DIMENSIONS.get(feat, "Other"),
            "hgb_perm_pr_auc": round(float(perm_mean[j]), 5),
            "hgb_perm_std": round(float(perm_std[j]), 5),
            "logit_std_coef": round(beta, 4),
            "logit_direction": direction,
        })

    df_imp = (
        pl.DataFrame(rows)
        .sort(["hgb_perm_pr_auc", "feature"], descending=[True, False])
        .with_row_index("rank", offset=1)
        .with_columns((pl.col("rank") <= 10).alias("in_top10"))
    )
    return df_imp


def extract_canary_case_studies(
    test: pl.DataFrame,
    p_test: np.ndarray,
    theta: float,
    top_gfc: int | None = None,
    top_modern: int | None = None,
) -> pl.DataFrame:
    """Post-hoc diagnostic report on the largest real receivership failures in the held-out TEST set."""
    cs_cfg = cfg().get("case_studies", {})
    top_gfc = int(cs_cfg.get("top_gfc", 5) if top_gfc is None else top_gfc)
    top_modern = int(cs_cfg.get("top_modern", 5) if top_modern is None else top_modern)

    ranked = (
        test.select("REPDTE")
        .with_columns(pl.Series("p", p_test), pl.arange(0, test.height).alias("idx"))
        .with_columns(
            pl.col("p").rank(method="ordinal", descending=True).over("REPDTE").alias("rk"),
            pl.len().over("REPDTE").alias("n_q"),
        )
        .sort("idx")
    )
    rk = ranked["rk"].to_numpy()
    nq = ranked["n_q"].to_numpy()
    flag_top1 = rk <= np.maximum(1, np.round(nq * 0.01).astype(int))
    flag_top2 = rk <= np.maximum(1, np.round(nq * 0.02).astype(int))
    flag_theta = p_test >= theta

    pos = (
        test.select("CERT", "REPDTE", "target_fail_4q", "texas_ratio", "eq_asset_ratio", "brokered_dep_ratio")
        .with_columns(
            pl.Series("p", p_test),
            pl.Series("flag_theta", flag_theta),
            pl.Series("flag_top1", flag_top1),
            pl.Series("flag_top2", flag_top2),
        )
        .filter(pl.col("target_fail_4q") == 1)
    )

    per_bank = pos.group_by("CERT").agg(
        pl.col("REPDTE").min().alias("min_pos_q"),
        pl.col("REPDTE").max().alias("max_pos_q"),
        pl.len().alias("pit_quarters"),
        pl.col("p").max().alias("max_prob"),
        pl.col("flag_theta").any().alias("caught_theta"),
        pl.col("flag_top1").any().alias("caught_top_1pct"),
        pl.col("flag_top2").any().alias("caught_top_2pct"),
        pl.col("REPDTE").filter(pl.col("flag_theta")).min().alias("first_alarm_theta"),
        pl.col("REPDTE").filter(pl.col("flag_top1")).min().alias("first_alarm_top1"),
        pl.col("texas_ratio").max().alias("max_texas_ratio"),
        (pl.col("eq_asset_ratio").min() * 100.0).alias("min_eq_asset_pct"),
        (pl.col("brokered_dep_ratio").max() * 100.0).alias("max_brokered_dep_pct"),
    )

    fail_meta = (
        parse_failures()
        .filter(pl.col("CERT").is_not_null() & pl.col("FAILDATE").is_not_null() & (pl.col("RESTYPE") == "FAILURE"))
        .sort(["CERT", "FAILDATE"])
        .group_by("CERT", maintain_order=True)
        .agg(
            pl.col("NAME").first().alias("bank_name"),
            pl.col("CITYST").first().alias("city_state"),
            pl.col("FAILDATE").first().alias("fail_date"),
            pl.col("QBFASSET").first().alias("qbf_asset_kusd"),
            pl.col("COST").first().fill_null(0.0).alias("dif_cost_kusd"),
        )
    )

    joined = (
        per_bank.join(fail_meta, on="CERT", how="inner")
        .with_columns(
            pl.when(pl.col("fail_date").dt.year() <= 2014)
            .then(pl.lit("GFC (2009-2014)"))
            .otherwise(pl.lit("Modern (2015-2025)"))
            .alias("regime"),
            (pl.col("fail_date") - pl.col("first_alarm_theta")).dt.total_days().alias("lead_days_theta"),
            (pl.col("fail_date") - pl.col("first_alarm_top1")).dt.total_days().alias("lead_days_top1"),
            (pl.col("qbf_asset_kusd") / 1000.0).round(1).alias("assets_musd"),
            (pl.col("dif_cost_kusd") / 1000.0).round(1).alias("dif_cost_musd"),
            pl.col("max_prob").round(4),
            pl.col("max_texas_ratio").round(3),
            pl.col("min_eq_asset_pct").round(2),
            pl.col("max_brokered_dep_pct").round(2),
        )
    )

    gfc_top = joined.filter(pl.col("regime") == "GFC (2009-2014)").sort("qbf_asset_kusd", descending=True).head(top_gfc)
    mod_top = joined.filter(pl.col("regime") == "Modern (2015-2025)").sort("qbf_asset_kusd", descending=True).head(top_modern)

    out = (
        pl.concat([gfc_top, mod_top])
        .with_columns(
            pl.col("first_alarm_theta").dt.strftime("%Y-%m-%d").fill_null("NOT_FLAGGED"),
            pl.col("first_alarm_top1").dt.strftime("%Y-%m-%d").fill_null("NOT_FLAGGED"),
            pl.col("fail_date").dt.strftime("%Y-%m-%d"),
            pl.col("min_pos_q").dt.strftime("%Y-%m-%d"),
            pl.col("max_pos_q").dt.strftime("%Y-%m-%d"),
        )
        .select([
            "CERT",
            "bank_name",
            "city_state",
            "regime",
            "fail_date",
            "assets_musd",
            "dif_cost_musd",
            "pit_quarters",
            "max_prob",
            "caught_theta",
            "first_alarm_theta",
            "lead_days_theta",
            "caught_top_1pct",
            "first_alarm_top1",
            "lead_days_top1",
            "caught_top_2pct",
            "max_texas_ratio",
            "min_eq_asset_pct",
            "max_brokered_dep_pct",
        ])
    )
    return out


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


def compute_leakage_parameter_shift(df: pl.DataFrame) -> pl.DataFrame:
    """Quantify median and upper-clip shifts if preprocessors were leaked across DEV+TEST instead of DEV-only."""
    dev = df.filter(pl.col("split_group") == "DEV")
    rows = []
    for col in FEATURE_COLS_CAMELS:
        s_dev = dev[col].drop_nulls().cast(pl.Float64).to_numpy()
        s_all = df[col].drop_nulls().cast(pl.Float64).to_numpy()
        s_dev = s_dev[np.isfinite(s_dev)]
        s_all = s_all[np.isfinite(s_all)]
        if len(s_dev) < 10 or len(s_all) < 10:
            continue
        med_d, q01_d, q99_d = float(np.median(s_dev)), float(np.quantile(s_dev, 0.01)), float(np.quantile(s_dev, 0.99))
        med_a, q01_a, q99_a = float(np.median(s_all)), float(np.quantile(s_all, 0.01)), float(np.quantile(s_all, 0.99))
        rel_med = abs(med_a - med_d) / max(abs(med_d), 1e-4) * 100.0
        rel_q99 = abs(q99_a - q99_d) / max(abs(q99_d), 1e-4) * 100.0
        rows.append({
            "feature": col,
            "dev_median": round(med_d, 5),
            "full_median": round(med_a, 5),
            "median_shift_pct": round(rel_med, 2),
            "dev_q01": round(q01_d, 5),
            "full_q01": round(q01_a, 5),
            "dev_q99": round(q99_d, 5),
            "full_q99": round(q99_a, 5),
            "q99_shift_pct": round(rel_q99, 2),
        })
    out = pl.DataFrame(rows).sort("q99_shift_pct", descending=True)
    out.write_csv(out_dir("tables") / "leak_parameter_shift.csv")
    return out


def run_dev_label_sensitivity(dev: pl.DataFrame, oof_primary: dict[str, np.ndarray]) -> pl.DataFrame:
    """Compare primary DEV CV target (FAILURE + ASSISTANCE) vs strict closures-only (FAILURE) sensitivity."""
    dev_fail_only = dev.with_columns(
        pl.when((pl.col("target_fail_4q") == 1) & (pl.col("FAILTYPE") != "FAILURE"))
        .then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.col("target_fail_4q"))
        .alias("target_fail_4q")
    )
    _, oof_closure = run_walk_forward_cv(dev_fail_only, FEATURE_COLS_CAMELS, "hgb", explain_acc=None)
    m_all = evaluate_predictions(oof_primary["y"], oof_primary["p"])
    m_cls = evaluate_predictions(oof_closure["y"], oof_closure["p"])
    rows = [
        {"label_scope": "FAILURE + ASSISTANCE (Primary)", "oof_failing_banks": int(pl.DataFrame({"c": oof_primary["cert"], "y": oof_primary["y"]}).filter(pl.col("y") == 1)["c"].n_unique()), **m_all},
        {"label_scope": "FAILURE only (Sensitivity)", "oof_failing_banks": int(pl.DataFrame({"c": oof_closure["cert"], "y": oof_closure["y"]}).filter(pl.col("y") == 1)["c"].n_unique()), **m_cls},
    ]
    tbl = pl.DataFrame(rows)
    tbl.write_csv(out_dir("tables") / "dev_label_sensitivity.csv")
    return tbl


def score_latest_quarter(champion, theta: float, gold_path: Path | None = None) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score the latest unlabelled POST_SAMPLE quarter (2026Q2) using the strictly DEV-trained champion."""
    gold_path = gold_path or data_dir("gold") / "features.parquet"
    lf = pl.scan_parquet(gold_path).filter(pl.col("split_group") == "POST_SAMPLE")
    latest_q = lf.select(pl.col("REPDTE").max()).collect().item()
    cols = ["CERT", "REPDTE", "STNAME", "ASSET", "pit_scorable"] + FEATURE_COLS_CAMELS
    latest = (
        lf.filter((pl.col("REPDTE") == latest_q) & pl.col("pit_scorable"))
        .select(cols)
        .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS_CAMELS])
        .sort("CERT")
        .collect()
    )
    probs = champion.predict_proba(latest.select(FEATURE_COLS_CAMELS).to_numpy())[:, 1]
    inst = parse_institutions().select(["CERT", "NAME", "BKCLASS", "ACTIVE"]).unique(subset=["CERT"], keep="first")

    n = latest.height
    top1_cut = max(1, int(round(n * 0.01)))
    top2_cut = max(1, int(round(n * 0.02)))
    scored = (
        latest.select(["CERT", "REPDTE", "STNAME", "ASSET", "texas_ratio", "eq_asset_ratio", "brokered_dep_ratio", "roa_ytd"])
        .with_columns(pl.Series("prob_fail_4q", probs))
        .sort("prob_fail_4q", descending=True)
        .with_row_index("rank", offset=1)
        .join(inst, on="CERT", how="left")
        .sort("rank")
        .with_columns(
            (pl.col("prob_fail_4q") >= theta).alias("flag_theta"),
            (pl.col("rank") <= top1_cut).alias("flag_top_1pct"),
            (pl.col("rank") <= top2_cut).alias("flag_top_2pct"),
            (pl.col("ASSET") / 1000.0).round(1).alias("assets_musd"),
            pl.col("prob_fail_4q").round(4),
            pl.col("texas_ratio").round(3),
            (pl.col("eq_asset_ratio") * 100.0).round(2).alias("eq_asset_pct"),
            (pl.col("brokered_dep_ratio") * 100.0).round(2).alias("brokered_dep_pct"),
            pl.col("REPDTE").dt.strftime("%Y-%m-%d").alias("repdte"),
        )
    )
    top25 = scored.head(25).select([
        "rank", "CERT", "NAME", "STNAME", "BKCLASS", "repdte", "assets_musd",
        "prob_fail_4q", "flag_theta", "flag_top_1pct", "texas_ratio", "eq_asset_pct", "brokered_dep_pct",
    ])
    top25.write_csv(out_dir("tables") / "watchlist_2026q2.csv")
    summary = {
        "quarter": str(latest_q),
        "banks_scored": n,
        "theta": round(theta, 4),
        "flagged_at_theta": int(scored.filter(pl.col("flag_theta")).height),
        "flagged_top_1pct": top1_cut,
        "flagged_top_2pct": top2_cut,
        "mean_prob": round(float(probs.mean()), 5),
        "max_prob": round(float(probs.max()), 4),
        "note": "Scored with strictly DEV-trained champion; labels >=2026Q3 are unobservable; NAME/BKCLASS from static institutions snapshot.",
    }
    save_json(summary, out_dir("tables") / "watchlist_2026q2_summary.json")
    return top25, summary


def _downsample_pr_curve(y: np.ndarray, p: np.ndarray, n_pts: int = 121) -> dict[str, list[float]]:
    """Compute a smooth Precision-Recall curve sampled uniformly along recall [0, 1]."""
    prec, rec, _ = precision_recall_curve(y, p)
    rec_asc = rec[::-1]
    prec_asc = prec[::-1]
    idx = np.unique(np.clip(np.searchsorted(rec_asc, np.linspace(0.0, 1.0, n_pts), side="left"), 0, len(rec_asc) - 1))
    return {
        "recall": [round(float(x), 4) for x in rec_asc[idx]],
        "precision": [round(float(x), 4) for x in prec_asc[idx]],
    }


def build_curves_payload(
    test: pl.DataFrame,
    y_test: np.ndarray,
    preds_by_model: dict[str, np.ndarray],
    cm: BankCostModel,
    theta: float,
    sens_costs: list[int],
) -> dict[str, Any]:
    """Export light JSON payload powering Part 5 PR curves, cost U-curve, alarms timeline & canary trajectories."""
    pr_curves = {name: _downsample_pr_curve(y_test, p) for name, p in preds_by_model.items()}
    p_hgb = preds_by_model["HistGradientBoosting, 26 CAMELS"]

    grid = build_theta_grid()
    cost_curves = {
        str(c): [round(cm.total_cost(float(t), float(c)) / 1e3, 1) for t in grid]  # million USD
        for c in sens_costs
    }

    test_q = (
        test.select("CERT", "REPDTE", "target_fail_4q")
        .with_columns(
            pl.Series("flag_theta", p_hgb >= theta),
        )
        .group_by("REPDTE")
        .agg(
            pl.col("target_fail_4q").sum().cast(int).alias("pos_banks"),
            pl.col("flag_theta").sum().cast(int).alias("alarms_theta"),
            pl.len().alias("total_banks"),
        )
        .sort("REPDTE")
        .with_columns(pl.col("REPDTE").dt.strftime("%Y-%m-%d").alias("repdte_str"))
    )

    canary_specs = [
        (29776, "AmTrust Bank (2009)"),
        (32102, "Doral Bank (2015)"),
        (27332, "Republic Bank (2024)"),
        (24735, "Silicon Valley Bank (2023)"),
        (59017, "First Republic Bank (2023)"),
    ]
    scored_test = test.select("CERT", "REPDTE", "FAILDATE").with_columns(pl.Series("p", p_hgb))
    trajectories = []
    for cert_id, label in canary_specs:
        sub = scored_test.filter(pl.col("CERT") == cert_id).sort("REPDTE").tail(8)
        if sub.height:
            q_before = list(range(-sub.height, 0))
            trajectories.append({
                "cert": cert_id,
                "label": label,
                "quarters_before_failure": q_before,
                "repdte": sub["REPDTE"].dt.strftime("%Y-%m-%d").to_list(),
                "prob": [round(float(x), 5) for x in sub["p"].to_list()],
            })

    payload = {
        "theta": round(theta, 4),
        "pr_curves": pr_curves,
        "cost_u_curve": {
            "theta_grid": [float(f"{t:.6g}") for t in grid],
            "cost_musd_by_cfp": cost_curves,
        },
        "quarterly_alarms": {
            "repdte": test_q["repdte_str"].to_list(),
            "pos_banks": test_q["pos_banks"].to_list(),
            "alarms_theta": test_q["alarms_theta"].to_list(),
        },
        "canary_trajectories": trajectories,
    }
    save_json(payload, out_dir("tables") / "model_curves_data.json")
    return payload


def execute_full_modeling_workflow(gold_path: Path | None = None) -> dict[str, Any]:
    df = load_modeling_frame(gold_path)
    log.info("Computing empirical train-only vs naive-full leakage parameter shift audit...")
    leak_shift_df = compute_leakage_parameter_shift(df)
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
    logit_explain: list[dict[str, Any]] = []
    hgb_explain: list[dict[str, Any]] = []
    for name, (feats, kind) in candidates.items():
        log.info(f"Walk-forward CV: {name}")
        acc = logit_explain if name == "Logistic, 26 CAMELS" else (hgb_explain if name == "HistGradientBoosting, 26 CAMELS" else None)
        folds, oof = run_walk_forward_cv(dev, feats, kind, explain_acc=acc)
        cv[name] = {**evaluate_predictions(oof["y"], oof["p"]), "folds": folds}
        if name == "HistGradientBoosting, 26 CAMELS":
            champion_oof = oof

    log.info("Running DEV label sensitivity check (FAILURE + ASSISTANCE vs FAILURE only)...")
    label_sens_df = run_dev_label_sensitivity(dev, champion_oof)

    feat_imp_df = summarize_dev_feature_importance(FEATURE_COLS_CAMELS, hgb_explain, logit_explain)
    feat_imp_csv = out_dir("tables") / "feature_importance.csv"
    feat_imp_df.write_csv(feat_imp_csv)
    log.info(f"Saved strictly DEV walk-forward feature importance to {feat_imp_csv}")

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
    log.info("Fitting DEV models once for held-out TEST evaluation & curve export...")
    feats = FEATURE_COLS_CAMELS
    X_dev_all, y_dev_all = dev.select(feats).to_numpy(), dev["target_fail_4q"].to_numpy()
    champion = make_model("hgb").fit(X_dev_all, y_dev_all)
    p_test = champion.predict_proba(test.select(feats).to_numpy())[:, 1]
    y_test = test["target_fail_4q"].to_numpy()
    p_texas = (make_model("texas")
               .fit(dev.select(["texas_ratio"]).to_numpy(), y_dev_all)
               .predict_proba(test.select(["texas_ratio"]).to_numpy())[:, 1])
    p_logit3 = (make_model("logit")
                .fit(dev.select(["eq_asset_ratio", "nclnls_ratio", "roa_ytd"]).to_numpy(), y_dev_all)
                .predict_proba(test.select(["eq_asset_ratio", "nclnls_ratio", "roa_ytd"]).to_numpy())[:, 1])
    p_logit26 = make_model("logit").fit(X_dev_all, y_dev_all).predict_proba(test.select(feats).to_numpy())[:, 1]

    build_curves_payload(
        test,
        y_test,
        {
            "HistGradientBoosting, 26 CAMELS": p_test,
            "Logistic, 26 CAMELS": p_logit26,
            "Logistic, 3 variables": p_logit3,
            "Texas ratio rule": p_texas,
        },
        cm,
        theta,
        cfg()["cost"]["sensitivity_false_alarm_kusd"],
    )

    log.info("Scoring unlabelled POST_SAMPLE latest quarter (2026Q2) operational watchlist...")
    watchlist_df, watchlist_summary = score_latest_quarter(champion, theta, gold_path)

    years = test["REPDTE"].dt.year().to_numpy()
    gfc, modern = (years <= 2014), (years >= 2015)
    dev_all_failing = int(dev.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique())

    canary_df = extract_canary_case_studies(test, p_test, theta)
    canary_csv = out_dir("tables") / "canary_case_studies.csv"
    canary_df.write_csv(canary_csv)
    log.info(f"Saved post-hoc Canary case studies ({canary_df.height} banks) to {canary_csv}")

    top_k = int(cfg().get("explainability", {}).get("top_k", 10))
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
        "canary_case_studies": canary_df.to_dicts(),
    }
    results = {
        "cross_validation_models": cv,
        "dev_label_sensitivity": label_sens_df.to_dicts(),
        "leakage_parameter_shift_top6": leak_shift_df.head(6).to_dicts(),
        "feature_importance_top10": feat_imp_df.head(top_k).to_dicts(),
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
        "operational_watchlist_2026q2": {
            "summary": watchlist_summary,
            "top10": watchlist_df.head(10).to_dicts(),
        },
    }
    save_json(results, out_dir("tables") / "test_results.json")
    log.info("Saved outputs/tables/test_results.json")
    return results
