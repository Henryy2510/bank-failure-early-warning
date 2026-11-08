"""Part 2: Automated Data Quality Audit and Visualization."""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import polars as pl

from src.bfews.io import read_raw_all
from src.bfews.quality import (
    audit_domain_validity,
    audit_missingness_by_era,
    audit_referential_integrity,
    generate_quality_report,
)
from src.bfews.utils import get_logger, out_dir

log = get_logger("02_quality")


def plot_missingness_era(era_df: pl.DataFrame):
    """Plot missingness trends by era showing regulatory structural breaks."""
    df = era_df.to_pandas()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(df["era"], df["P3ASSET_null_pct"], marker="o", label="P3ASSET (Past due 30-89d loans)", color="#d9534f", lw=2)
    ax.plot(df["era"], df["ELNATR_null_pct"], marker="s", label="ELNATR (Loan loss provision)", color="#f0ad4e", lw=2)
    ax.plot(df["era"], df["RBC1AAJ_null_pct"], marker="^", label="RBC1AAJ (Tier 1 cap ratio)", color="#5bc0de", lw=2)
    ax.plot(df["era"], df["EQ_null_pct"], marker="x", label="EQ (Equity capital)", color="#5cb85c", lw=2)

    ax.set_xlabel("Era (5-Year Windows)")
    ax.set_ylabel("Missingness (%)")
    ax.set_title("Part 2: Structural Missingness Across Eras in FDIC Call Reports")
    ax.axvline(x=2001, color="grey", linestyle="--", alpha=0.7, label="2001 Regulatory Reporting Revision")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_p = out_dir("figures") / "missingness_by_era.png"
    plt.savefig(out_p, dpi=200)
    plt.close()
    log.info(f"Saved era missingness plot to {out_p}")


def main():
    log.info("Starting Part 2: Automated Data Quality Audit...")
    report_path = out_dir("dq_report") / "data_quality_report.md"
    rep_text = generate_quality_report(report_path)
    log.info(f"Generated comprehensive DQ report at {report_path}")

    log.info("Generating DQ visualizations...")
    df = read_raw_all()
    era_df = audit_missingness_by_era(df)
    plot_missingness_era(era_df)
    log.info("Part 2 audit complete.")


if __name__ == "__main__":
    main()
