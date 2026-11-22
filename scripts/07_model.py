"""Part 5: Model Training, Walk-Forward Validation, Cost-Sensitive Thresholding, and Test Set Evaluation."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from datetime import datetime

from src.bfews.modeling import execute_full_modeling_workflow
from src.bfews.utils import data_dir, df_to_markdown, get_logger, out_dir, save_json

log = get_logger("07_model")


def plot_model_curves(results: dict):
    """Plot PR and ROC curve comparisons and save figure."""
    cv_models = results["cross_validation_models"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Model comparison bar chart for PR-AUC and ROC-AUC
    names = list(cv_models.keys())
    clean_names = [n.replace("_", " ") for n in names]
    pr_aucs = [cv_models[m]["pr_auc"] for m in names]
    roc_aucs = [cv_models[m]["roc_auc"] for m in names]

    x = range(len(names))
    width = 0.35

    ax1.bar([i - width / 2 for i in x], pr_aucs, width, label="PR-AUC (Precision-Recall)", color="#5cb85c")
    ax1.bar([i + width / 2 for i in x], roc_aucs, width, label="ROC-AUC", color="#5bc0de")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(clean_names, rotation=25, ha="right")
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Walk-Forward Cross-Validation Comparison")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Recall @ Top 1% and Top 5% (Supervisory Inspection Capacity)
    rec1 = [cv_models[m]["recall_top_1pct"] for m in names]
    rec5 = [cv_models[m]["recall_top_5pct"] for m in names]

    ax2.bar([i - width / 2 for i in x], rec1, width, label="Recall @ Top 1% Ranked", color="#f0ad4e")
    ax2.bar([i + width / 2 for i in x], rec5, width, label="Recall @ Top 5% Ranked", color="#d9534f")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(clean_names, rotation=25, ha="right")
    ax2.set_ylabel("Recall Share of Failures")
    ax2.set_title("Supervisory Actionability: Failure Capture in Top Ranks")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_fig = out_dir("figures") / "model_evaluation_comparison.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    log.info(f"Saved model comparison figure to {out_fig}")


def plot_crisis_breakdown(results: dict):
    """Plot performance across test crisis subperiods (GFC vs Modern/SVB)."""
    t_res = results["held_out_test_evaluation"]
    gfc = t_res["gfc_2009_2014"]
    mod = t_res["modern_2015_2025"]

    subperiods = ["GFC (2009-2014)", "Post-GFC / SVB (2015-2025)"]
    pr_vals = [gfc["pr_auc"], mod["pr_auc"]]
    roc_vals = [gfc["roc_auc"], mod["roc_auc"]]
    fails = [gfc["failing_banks"], mod["failing_banks"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(subperiods))
    width = 0.35

    b1 = ax.bar([i - width / 2 for i in x], pr_vals, width, label="PR-AUC", color="#5cb85c")
    b2 = ax.bar([i + width / 2 for i in x], roc_vals, width, label="ROC-AUC", color="#5bc0de")

    for idx, (bar1, bar2, cnt) in enumerate(zip(b1, b2, fails)):
        ax.text(idx, 0.05, f"Failures: {cnt:,} banks", ha="center", va="bottom", fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xticks(list(x))
    ax.set_xticklabels(subperiods)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test Set Performance Across Crisis Regimes")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_fig = out_dir("figures") / "test_crisis_subperiods.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    log.info(f"Saved crisis subperiod figure to {out_fig}")


def plot_operating_points(results: dict):
    """Plot capacity-based vs cost-based operating points on held-out test set."""
    t_res = results["held_out_test_evaluation"]
    cap_by_pct = {p["pct"]: p for p in t_res["capacity_points"]}
    cost_pt = t_res["at_threshold"]

    labels = ["Top 1%/qtr", "Top 2%/qtr", f"Cost θ*={t_res['threshold']:.4f}", "Top 5%/qtr"]
    pts = [cap_by_pct[1], cap_by_pct[2], cost_pt, cap_by_pct[5]]

    bank_rec = [p["bank_recall"] for p in pts]
    row_rec = [p["row_recall"] for p in pts]
    prec = [p["precision_row"] for p in pts]
    flagged_pct = [p["flagged_share_of_rows_pct"] for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(labels))
    width = 0.28

    ax1.bar([i - width for i in x], bank_rec, width, label="Bank-Level Recall", color="#2b8cbe")
    ax1.bar(list(x), row_rec, width, label="Quarter-Row Recall", color="#7bccc4")
    ax1.bar([i + width for i in x], prec, width, label="Row Precision", color="#f0ad4e")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0, 1.22)
    ax1.set_ylabel("Rate")
    ax1.set_title("Held-Out Test Operating Points: Recall & Precision")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper center", ncol=3, fontsize=8.2, frameon=True)

    bars = ax2.bar(list(x), flagged_pct, width=0.45, color="#d9534f")
    for b, p in zip(bars, pts):
        caught = p.get("failing_banks_caught", p.get("banks_caught"))
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.15,
            f"{b.get_height():.2f}%\n({caught}/{p['failing_banks']} banks)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(0, max(flagged_pct) * 1.30)
    ax2.set_ylabel("Flagged Share of All Bank-Quarters (%)")
    ax2.set_title("Supervisory Inspection Burden (% Flagged Rows)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = out_dir("figures") / "operating_points.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    log.info(f"Saved operating points figure to {out_fig}")


def plot_feature_importance(results: dict):
    """Plot Top 10 DEV walk-forward permutation importance colored by Logistic directionality."""
    top10 = results["feature_importance_top10"]
    top10_rev = list(reversed(top10))

    labels = [f"{r['feature']} ({r['camels_dim'].split(' - ')[0]})" for r in top10_rev]
    vals = [r["hgb_perm_pr_auc"] for r in top10_rev]
    stds = [r["hgb_perm_std"] for r in top10_rev]
    colors = ["#d9534f" if r["logit_std_coef"] > 0 else "#2b8cbe" for r in top10_rev]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), vals, xerr=stds, color=colors, alpha=0.9, capsize=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("DEV Walk-Forward Permutation Importance (Δ PR-AUC)")
    ax.set_title("Top 10 CAMELS Risk Drivers (Leak-Free DEV CV, HGB + Logit Sign)")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d9534f", label="Logit β > 0 (+ Failure Risk)"),
        Patch(facecolor="#2b8cbe", label="Logit β < 0 (− Protective Buffer)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out_fig = out_dir("figures") / "feature_importance.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    log.info(f"Saved feature importance figure to {out_fig}")


def plot_pr_and_cost_curves(results: dict):
    """Plot TEST Precision-Recall curves across models and DEV OOF Cost U-curve over theta."""
    curves = json.loads((out_dir("tables") / "model_curves_data.json").read_text(encoding="utf-8"))
    t_eval = results["held_out_test_evaluation"]
    theta = curves["theta"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))
    styles = {
        "HistGradientBoosting, 26 CAMELS": ("#d9534f", "-", 2.4, f"HGB 26 (PR-AUC={t_eval['pr_auc']:.4f})"),
        "Logistic, 26 CAMELS": ("#2b8cbe", "--", 1.8, "Logit 26 CAMELS"),
        "Logistic, 3 variables": ("#5cb85c", "-.", 1.6, "Logit 3 variables"),
        "Texas ratio rule": ("#f0ad4e", ":", 1.8, f"Texas Ratio Rule (PR-AUC={t_eval['texas_rule']['pr_auc']:.4f})"),
    }
    for name, (col, ls, lw, lbl) in styles.items():
        if name in curves["pr_curves"]:
            c = curves["pr_curves"][name]
            ax1.plot(c["recall"], c["precision"], color=col, linestyle=ls, linewidth=lw, label=lbl)

    at_th = t_eval["at_threshold"]
    ax1.scatter(
        [at_th["row_recall"]],
        [at_th["precision_row"]],
        color="#d9534f",
        edgecolor="black",
        s=70,
        zorder=5,
        label=f"theta*={theta:.4f} (Rec={at_th['row_recall']:.2f}, Prec={at_th['precision_row']:.2f})",
    )
    ax1.set_xlabel("Quarter-Row Recall on Held-Out TEST (2009–2025)")
    ax1.set_ylabel("Precision")
    ax1.set_title("Held-Out TEST Precision-Recall Curves")
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    grid = curves["cost_u_curve"]["theta_grid"]
    cost_colors = {"100": "#7bccc4", "250": "#d9534f", "500": "#2b8cbe", "1000": "#6a51a3"}
    for c_str, costs_musd in curves["cost_u_curve"]["cost_musd_by_cfp"].items():
        lw = 2.4 if c_str == "250" else 1.3
        lbl = f"C_FP = ${c_str}k" + (" (baseline)" if c_str == "250" else "")
        ax2.plot(grid, costs_musd, color=cost_colors.get(c_str, "grey"), linewidth=lw, label=lbl)

    base_costs = curves["cost_u_curve"]["cost_musd_by_cfp"].get("250", [])
    if base_costs:
        min_idx = min(range(len(base_costs)), key=lambda i: base_costs[i])
        ax2.scatter([grid[min_idx]], [base_costs[min_idx]], color="#d9534f", edgecolor="black", s=70, zorder=5,
                    label=f"Min Cost @ theta*={theta:.4f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Alert Probability Threshold theta (log scale)")
    ax2.set_ylabel("DEV OOF Total Cost ($ Millions, capped DIF + Exams)")
    ax2.set_title("Cost-Sensitive Threshold Optimization (DEV OOF U-Curve)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper center", fontsize=7.8)

    plt.tight_layout()
    out_fig = out_dir("figures") / "pr_and_cost_curves.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    log.info(f"Saved PR & Cost U-curves figure to {out_fig}")


def plot_alarms_and_canary_trajectories():
    """Plot TEST quarterly alarms vs actual failure exposure and Canary bank pre-failure trajectories."""
    curves = json.loads((out_dir("tables") / "model_curves_data.json").read_text(encoding="utf-8"))
    qa = curves["quarterly_alarms"]
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in qa["repdte"]]
    theta = curves["theta"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax1.bar(dates, qa["alarms_theta"], width=70, color="#2b8cbe", alpha=0.65, label=f"Model Alarms (p >= {theta:.4f})")
    ax1_r = ax1.twinx()
    ax1_r.plot(dates, qa["pos_banks"], color="#d9534f", linewidth=2.0, label="Actual Failures in Next 4Q")
    ax1.xaxis.set_major_locator(mdates.YearLocator(3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlabel("Held-Out TEST Quarter")
    ax1.set_ylabel("Quarterly Alerts Flagged", color="#2b8cbe")
    ax1_r.set_ylabel("True Positive Pre-Failure Banks", color="#d9534f")
    ax1.set_title("Quarterly Model Alerts vs. True Failures (2009–2025)")
    ax1.grid(True, alpha=0.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    traj_colors = ["#d9534f", "#f0ad4e", "#2b8cbe", "#7f7f7f", "#9467bd"]
    for idx, tr in enumerate(curves["canary_trajectories"]):
        col = traj_colors[idx % len(traj_colors)]
        ls = "--" if "Valley" in tr["label"] or "First Republic" in tr["label"] else "-"
        ax2.plot(tr["quarters_before_failure"], tr["prob"], marker="o", markersize=4,
                 color=col, linestyle=ls, linewidth=2.0, label=tr["label"])
    ax2.axhline(theta, color="black", linestyle=":", linewidth=1.3, label=f"theta* = {theta:.4f}")
    ax2.set_yscale("log")
    ax2.set_ylim(2e-5, 1.0)
    ax2.set_xlabel("Quarters Relative to Receivership (-1 = Last PIT Report)")
    ax2.set_ylabel("Predicted 4Q Failure Probability (log scale)")
    ax2.set_title("Canary Failure Risk Trajectories: Credit vs. Liquidity Runs")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=7.8)

    plt.tight_layout()
    out_fig = out_dir("figures") / "alarms_and_canary_trajectories.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    log.info(f"Saved quarterly alarms & canary trajectories figure to {out_fig}")


def main():
    log.info("Starting Part 5: Full Modeling, Cross-Validation & Test Evaluation...")
    results = execute_full_modeling_workflow()

    # Save summary table
    cv_models = results["cross_validation_models"]
    rows = []
    for m, vals in cv_models.items():
        rows.append({
            "model": m,
            "pr_auc": vals["pr_auc"],
            "roc_auc": vals["roc_auc"],
            "recall_top_1pct": vals["recall_top_1pct"],
            "recall_top_5pct": vals["recall_top_5pct"],
            "brier_score": vals["brier_score"],
        })
    cv_df = pl.DataFrame(rows)
    cv_csv = out_dir("tables") / "cv_model_comparison.csv"
    cv_df.write_csv(cv_csv)
    log.info(f"Saved CV model comparison to {cv_csv}")
    print(cv_df)

    log.info("Held-out Test Evaluation Results:")
    t_eval = results["held_out_test_evaluation"]
    log.info(f"Overall Test PR-AUC: {t_eval['pr_auc']}, ROC-AUC: {t_eval['roc_auc']}")
    log.info(f"Optimal Threshold: {t_eval['threshold']}")
    log.info(f"At Threshold: {t_eval['at_threshold']}")
    log.info(f"Capacity Operating Points: {t_eval['capacity_points']}")
    log.info("Top 10 DEV Features:")
    print(pl.DataFrame(results["feature_importance_top10"]))
    log.info("Post-hoc Canary Case Studies on TEST:")
    print(pl.DataFrame(t_eval["canary_case_studies"]))

    plot_model_curves(results)
    plot_crisis_breakdown(results)
    plot_operating_points(results)
    plot_feature_importance(results)
    plot_pr_and_cost_curves(results)
    plot_alarms_and_canary_trajectories()
    log.info("Part 5 complete.")


if __name__ == "__main__":
    main()
