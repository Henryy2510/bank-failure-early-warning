"""Automated consistency check verifying that report/report.md and README.md match generated JSON/CSV outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from src.bfews.utils import ROOT, get_logger, out_dir

log = get_logger("09_check_numbers")


def main() -> None:
    report_txt = (ROOT / "report" / "report.md").read_text(encoding="utf-8")
    readme_txt = (ROOT / "README.md").read_text(encoding="utf-8")

    test_res = json.loads((out_dir("tables") / "test_results.json").read_text(encoding="utf-8"))
    cv_df = pl.read_csv(out_dir("tables") / "cv_model_comparison.csv")
    exit_df = pl.read_csv(out_dir("tables") / "bank_exit_summary.csv")
    storage_df = pl.read_csv(out_dir("tables") / "part1_storage_benchmark.csv")
    bench_df = pl.read_csv(out_dir("tables") / "part3_benchmark_results.csv")
    full_bench = {
        r["engine"]: r
        for r in bench_df.filter(pl.col("slice") == "1984-2026 (full)").iter_rows(named=True)
    }

    cost_cfg = test_res["cost_threshold_tuning"]
    t_eval = test_res["held_out_test_evaluation"]
    at_th = t_eval["at_threshold"]
    cap_by_pct = {p["pct"]: p for p in t_eval["capacity_points"]}

    # 1. Required exact strings in report.md
    required_in_report = [
        f"{cost_cfg['theta']:.4f}",  # 0.0026
        f"{cost_cfg['dev_failing_banks_in_oof_blocks']:,}",  # 1,522
        f"{cost_cfg['dev_failing_banks_all']:,}",  # 2,661
        f"{cost_cfg['dev_oof_banks_caught_at_theta']:,}",  # 1,448
        f"{t_eval['pr_auc']:.4f}",  # 0.5541
        f"{t_eval['roc_auc']:.4f}",  # 0.9845
        f"{t_eval['brier_score']:.5f}",  # 0.00238
        f"{at_th['failing_banks_caught']} / {at_th['failing_banks']}",  # 471 / 485
        f"{at_th['bank_recall'] * 100:.2f}%",  # 97.11%
        f"{at_th['flagged_share_of_rows_pct']:.3f}%",  # 3.657%
        f"{cap_by_pct[1]['failing_banks_caught']} / {cap_by_pct[1]['failing_banks']}",  # 401 / 485
        f"{cap_by_pct[1]['bank_recall'] * 100:.2f}%",  # 82.68%
        f"{cap_by_pct[2]['failing_banks_caught']} / {cap_by_pct[2]['failing_banks']}",  # 441 / 485
        f"{cap_by_pct[5]['failing_banks_caught']} / {cap_by_pct[5]['failing_banks']}",  # 465 / 485
        f"{full_bench['pandas + pushdown']['time_s']:.3f} s",
        f"{full_bench['Polars lazy']['time_s']:.3f} s",
        f"{full_bench['Polars streaming']['time_s']:.3f} s",
    ]

    for row in cv_df.iter_rows(named=True):
        required_in_report.append(f"{row['pr_auc']:.4f}")
        required_in_report.append(f"{row['roc_auc']:.4f}")
        required_in_report.append(f"{row['brier_score']:.5f}")

    for row in exit_df.iter_rows(named=True):
        required_in_report.append(f"{int(row['banks']):,}")

    missing_report = [tok for tok in required_in_report if tok not in report_txt]
    if missing_report:
        raise AssertionError(f"Missing expected tokens in report/report.md: {missing_report}")

    # 2. Required exact strings in README.md
    required_in_readme = [
        f"{cost_cfg['theta']:.4f}",  # 0.0026
        f"{cost_cfg['dev_failing_banks_in_oof_blocks']:,}",  # 1,522
        f"{cost_cfg['dev_failing_banks_all']:,}",  # 2,661
        f"{t_eval['pr_auc']:.4f}",  # 0.5541
        f"{t_eval['roc_auc']:.4f}",  # 0.9845
        f"{at_th['bank_recall'] * 100:.2f}%",  # 97.11%
        "425.7 MB",
        "154.1 MB",
        "2,859",
        "16,892",
        "4,313",
        f"{full_bench['pandas + pushdown']['time_s']:.3f} s",
    ]
    missing_readme = [tok for tok in required_in_readme if tok not in readme_txt]
    if missing_readme:
        raise AssertionError(f"Missing expected tokens in README.md: {missing_readme}")

    # 3. Forbidden stale numbers in report.md and README.md
    stale_tokens = [
        "0.3990",
        "0.6721",
        "444.7 MB",
        "2,856",
        "16,850",
        "4,358",
        "2,650",
        "483 / 485",
        "99.59%",
        "theta* = 0.010",
        "\\theta^* = 0.010",
    ]
    stale_found = [
        tok for tok in stale_tokens if (tok in report_txt or tok in readme_txt)
    ]
    if stale_found:
        raise AssertionError(f"Found stale tokens in report.md or README.md: {stale_found}")

    log.info("All report.md and README.md numbers verified against JSON/CSV outputs with zero stale tokens!")


if __name__ == "__main__":
    main()
