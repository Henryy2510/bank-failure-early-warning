"""Part 2 (cont.): build the Silver bank-quarter panel and summarise bank exits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl

from src.bfews.io import failures_by_cert, parse_institutions
from src.bfews.panel import build_silver_panel
from src.bfews.utils import data_dir, get_logger, out_dir, save_json

log = get_logger("03_panel")

FDIC_CHANGEC1_MEANINGS = {
    211: "Failure - Whole Institution",
    213: "Merger - Assisted",
    216: "Bridge Bank Resolution",
    217: "Receivership / Conservatorship",
    221: "Absorption - Without Assistance",
    222: "Consolidation - Without Assistance",
    223: "Merger - Without Assistance",
    224: "Affiliated Institution Merger",
    225: "Partial P&A - Without Assistance",
    230: "Closing - Failure Payoff",
    235: "RTC Payoff / Liquidation",
    240: "Closing - Voluntary",
    470: "Charter / Agency Change",
}


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

    inst_codes = parse_institutions().select(["CERT", "CHANGEC1"])
    cc_tbl = (
        exits.filter(pl.col("exit_status") != "ACTIVE")
        .join(inst_codes, on="CERT", how="left")
        .group_by("CHANGEC1")
        .agg(
            pl.len().alias("exited_banks"),
            (pl.col("exit_status") == "FAILED").sum().alias("classified_failed"),
            pl.col("has_failure_record").sum().alias("in_failures_table"),
        )
        .sort("exited_banks", descending=True)
        .with_columns(
            pl.col("CHANGEC1").replace_strict(FDIC_CHANGEC1_MEANINGS, default="Other / Unspecified").alias("meaning")
        )
    )
    cc_tbl.write_csv(out_dir("tables") / "exit_changec1_codes.csv")
    log.info(f"Saved CHANGEC1 exit audit ({cc_tbl.height} codes) to outputs/tables/exit_changec1_codes.csv")

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
