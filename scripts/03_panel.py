"""Part 2 (cont.): build the Silver bank-quarter panel and summarise bank exits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl

from src.bfews.io import failures_by_cert
from src.bfews.panel import build_silver_panel
from src.bfews.utils import data_dir, get_logger, out_dir, save_json

log = get_logger("03_panel")


def main() -> None:
    log.info("Part 2: building the Silver panel...")
    panel = build_silver_panel()
    exits = pl.read_parquet(data_dir("silver") / "bank_exits.parquet")

    by_status = (exits.group_by("exit_status")
                 .agg(pl.len().alias("banks"), pl.col("num_quarters").sum().alias("bank_quarters"),
                      pl.col("has_failure_record").sum().alias("banks_with_failure_record"))
                 .sort("banks", descending=True))
    print(by_status)
    by_status.write_csv(out_dir("tables") / "bank_exit_summary.csv")

    odd = exits.filter(pl.col("has_failure_record") & (pl.col("exit_status") != "FAILED"))
    crosstab = (exits.filter(pl.col("has_failure_record"))
                .join(failures_by_cert().select("CERT", "FAILTYPE"), on="CERT", how="left")
                .group_by("exit_status", "FAILTYPE")
                .agg(pl.len().alias("banks"), pl.col("fail_gap_days").median().alias("median_gap_days"))
                .sort("exit_status", "FAILTYPE"))
    print(crosstab)
    crosstab.write_csv(out_dir("tables") / "failure_record_crosstab.csv")

    def n(status: str, ftype: str) -> int:
        r = crosstab.filter((pl.col("exit_status") == status) & (pl.col("FAILTYPE") == ftype))
        return int(r["banks"].sum()) if r.height else 0

    summary = {
        "rows": panel.height, "banks": exits.height,
        "failed_banks": int(exits.filter(pl.col("exit_status") == "FAILED").height),
        "failed_banks_by_type": {t: n("FAILED", t) for t in ("FAILURE", "ASSISTANCE")},
        # a failure record whose date is more than 90 days BEFORE the bank's last Call Report: the bank kept filing
        "failure_record_but_reports_long_after": {t: n("MERGED_OR_OTHER", t) for t in ("FAILURE", "ASSISTANCE")},
        "failure_record_but_still_active": int(odd.filter(pl.col("exit_status") == "ACTIVE").height),
    }
    save_json(summary, out_dir("tables") / "silver_summary.json")
    log.info(f"Silver summary: {summary}")


if __name__ == "__main__":
    main()
