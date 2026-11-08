"""Part 2 (cont): Build Silver Bank-Quarter Panel with Flow De-cumulation and Exit Audit."""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import polars as pl

from src.bfews.panel import build_silver_panel
from src.bfews.utils import df_to_markdown, get_logger, out_dir

log = get_logger("03_panel")


def main():
    log.info("Starting Part 2: Building clean Silver Bank-Quarter Panel...")
    panel_df = build_silver_panel()

    # Summarize exit distribution
    exit_counts = (
        panel_df.group_by("exit_status")
        .agg([
            pl.col("CERT").n_unique().alias("unique_banks"),
            pl.len().alias("panel_rows"),
        ])
        .sort("unique_banks", descending=True)
    )

    log.info("Bank Exit Classification Summary:")
    print(exit_counts)

    out_tbl = out_dir("tables") / "bank_exit_summary.csv"
    exit_counts.write_csv(out_tbl)
    log.info(f"Saved exit summary table to {out_tbl}")
    log.info("Part 2 panel build complete.")


if __name__ == "__main__":
    main()
