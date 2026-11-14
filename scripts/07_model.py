"""Part 5: Model Training, Walk-Forward Validation, Cost-Sensitive Thresholding, and Test Set Evaluation."""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import precision_recall_curve, roc_curve

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
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Rate")
    ax1.set_title("Held-Out Test Operating Points: Recall & Precision")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

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

    plot_model_curves(results)
    plot_crisis_breakdown(results)
    plot_operating_points(results)
    log.info("Part 5 complete.")


if __name__ == "__main__":
    main()
