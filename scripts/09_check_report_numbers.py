"""Automated consistency check verifying that report/report.md and README.md match generated JSON/CSV outputs."""
from __future__ import annotations

import json
import re
import subprocess
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
    spark_bench_df = pl.read_csv(out_dir("tables") / "spark_scaling_results.csv")
    feat_imp_df = pl.read_csv(out_dir("tables") / "feature_importance.csv")
    canary_df = pl.read_csv(out_dir("tables") / "canary_case_studies.csv")
    watchlist_summary = json.loads((out_dir("tables") / "watchlist_2026q2_summary.json").read_text(encoding="utf-8"))
    dq_summary = json.loads((out_dir("tables") / "dq_summary.json").read_text(encoding="utf-8"))
    cfp_stab = json.loads((out_dir("tables") / "cfp_stability_interval.json").read_text(encoding="utf-8"))
    full_bench = {
        r["engine"]: r
        for r in bench_df.filter(pl.col("scale") == "1x").iter_rows(named=True)
    }
    bench_8x = {
        r["engine"]: r
        for r in bench_df.filter(pl.col("scale") == "8x").iter_rows(named=True)
    }
    spark_by_key = {(r["scale"], r["engine"]): r for r in spark_bench_df.iter_rows(named=True)}
    for req_k in (("1x", "PySpark local[2]"), ("4x", "PySpark local[2]"), ("1x", "PySpark local[4]"), ("4x", "PySpark local[4]")):
        if req_k not in spark_by_key:
            raise AssertionError(f"Missing expected PySpark benchmark row {req_k} in spark_scaling_results.csv")

    cost_cfg = test_res["cost_threshold_tuning"]
    t_eval = test_res["held_out_test_evaluation"]
    at_th = t_eval["at_threshold"]
    cap_by_pct = {p["pct"]: p for p in t_eval["capacity_points"]}

    # 1. Required exact strings in report.md
    required_in_report = [
        f"{cost_cfg['theta']:.4f}",
        f"{cost_cfg['dev_failing_banks_in_oof_blocks']:,}",
        f"{cost_cfg['dev_failing_banks_all']:,}",
        f"{cost_cfg['dev_oof_banks_caught_at_theta']:,}",
        f"{t_eval['pr_auc']:.4f}",
        f"{t_eval['roc_auc']:.4f}",
        f"{t_eval['brier_score']:.5f}",
        f"{at_th['failing_banks_caught']} / {at_th['failing_banks']}",
        f"{at_th['bank_recall'] * 100:.2f}%",
        f"{at_th['flagged_share_of_rows_pct']:.3f}%",
        f"{cap_by_pct[1]['failing_banks_caught']} / {cap_by_pct[1]['failing_banks']}",
        f"{cap_by_pct[1]['bank_recall'] * 100:.2f}%",
        f"{cap_by_pct[2]['failing_banks_caught']} / {cap_by_pct[2]['failing_banks']}",
        f"{cap_by_pct[5]['failing_banks_caught']} / {cap_by_pct[5]['failing_banks']}",
        f"{full_bench['pandas + pushdown']['time_s']:.3f} s",
        f"{bench_8x['Polars streaming']['time_s']:.3f} s",
        f"{spark_by_key[('4x', 'PySpark local[4]')]['total_s']:.3f} s",
        f"{spark_by_key[('4x', 'PySpark local[2]')]['total_s']:.3f} s",
        "local[4]",
        f"{dq_summary['validity']['era2020_rbcrwa_zero_pct']}%",
        f"{watchlist_summary['banks_scored']:,}",
        f"{watchlist_summary['flagged_at_theta']}",
        "2026-11-05",
        "frozen_protocol.json",
        "cfp_stability_interval.json",
        f"${cfp_stab['cfp_lo_kusd']:.1f}\\text{{k}}",
        f"${cfp_stab['cfp_hi_kusd']:.1f}\\text{{k}}",
        "88.56%",
        "5,636",
        "1984 (4 qtrs)",
        "2,320",
        "3,241",
        "3,192",
        "Table A2.",
        "2007-12-31",
        "663.7 MB",
        "16.1%",
        "CERT 33802",
    ]

    for row in cv_df.iter_rows(named=True):
        required_in_report.append(f"{row['pr_auc']:.4f}")
        required_in_report.append(f"{row['roc_auc']:.4f}")
        required_in_report.append(f"{row['brier_score']:.5f}")

    for row in exit_df.iter_rows(named=True):
        required_in_report.append(f"{int(row['banks']):,}")

    for row in feat_imp_df.head(3).iter_rows(named=True):
        required_in_report.append(row["feature"])
        required_in_report.append(f"{row['hgb_perm_pr_auc']:.5f}")
        required_in_report.append(f"{row['logit_std_coef']:+.4f}")

    for row in canary_df.iter_rows(named=True):
        if row["lead_days_theta"] is not None:
            required_in_report.append(f"{int(row['lead_days_theta'])} d")

    repo_handles = [
        "batuhanisik751/bank-failure-early-warning",
        "Tajaddin/bank-early-warning",
        "martinogie/bank-health-data-platform",
        "saivardhan13-MSV/fdic-bank-health-pipeline",
        "vfabuluje/Canary",
        "heltonwo/fdic-failed-banks-etl",
        "Zahivc/Bank-Bankruptcy-Predictions",
    ]
    required_in_report.extend(repo_handles)

    missing_report = [tok for tok in required_in_report if tok not in report_txt]
    if missing_report:
        raise AssertionError(f"Missing expected tokens in report/report.md: {missing_report}")

    report_lower = report_txt.lower()
    missing_banks = [
        name for name in ["colonial bank", "guaranty bank", "silicon valley bank", "first republic bank", "doral bank", "republic bank"]
        if name not in report_lower
    ]
    if missing_banks:
        raise AssertionError(f"Missing Canary case-study bank names in report/report.md: {missing_banks}")

    # 2. Required exact strings in README.md
    required_in_readme = [
        f"{cost_cfg['theta']:.4f}",
        f"{cost_cfg['dev_failing_banks_in_oof_blocks']:,}",
        f"{cost_cfg['dev_failing_banks_all']:,}",
        f"{t_eval['pr_auc']:.4f}",
        f"{t_eval['roc_auc']:.4f}",
        f"{at_th['bank_recall'] * 100:.2f}%",
        "425.7 MB",
        "154.1 MB",
        "2,859",
        "16,892",
        "4,313",
        f"{full_bench['pandas + pushdown']['time_s']:.3f} s",
        f"{bench_8x['Polars streaming']['time_s']:.3f} s",
        f"{watchlist_summary['banks_scored']:,}",
        "Elkan, C. (2001)",
        "frozen_protocol.json",
        *feat_imp_df.head(3)["feature"].to_list(),
        *repo_handles,
    ]
    missing_readme = [tok for tok in required_in_readme if tok not in readme_txt]
    if missing_readme:
        raise AssertionError(f"Missing expected tokens in README.md: {missing_readme}")

    # 3. Verify compiled PDF exists and complies with rubric page limits (<= 20 main body + <= 10 appendix)
    pdf_path = ROOT / "report" / "QuanHuuNhatMinh_11230998_Final_Report.pdf"
    if not pdf_path.exists():
        raise AssertionError(f"Missing compiled PDF: {pdf_path}")
    info = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", info)
    if not m:
        raise AssertionError("Could not parse page count from pdfinfo output")
    pages = int(m.group(1))
    pdf_txt = subprocess.run(["pdftotext", str(pdf_path), "-"], check=True, capture_output=True, text=True).stdout
    pdf_pages = [p for p in pdf_txt.split("\f") if p.strip()]
    app_page = next((idx + 1 for idx, ptxt in enumerate(pdf_pages) if "Appendix:" in ptxt), None)
    main_pages = (app_page - 1) if app_page is not None else pages
    appendix_pages = pages - main_pages
    if main_pages > 20:
        raise AssertionError(f"Compiled PDF main body exceeds 20-page limit: {main_pages} main pages (total {pages})")
    if appendix_pages > 10 or pages > 30:
        raise AssertionError(f"Compiled PDF appendix exceeds 10-page limit: {appendix_pages} appendix pages (total {pages})")
    raw_latex_leaks = [tok for tok in ("\\theta", "\\rightarrow", "\\mathcal{", "$y_{") if tok in pdf_txt]
    if raw_latex_leaks:
        raise AssertionError(f"Found unrendered LaTeX markup in compiled PDF: {raw_latex_leaks}")

    fonts_out = subprocess.run(["pdffonts", str(pdf_path)], check=True, capture_output=True, text=True).stdout
    fonts_lower = fonts_out.lower()
    if "timesnewroman" not in fonts_lower:
        raise AssertionError(f"Expected Times New Roman embedded in compiled PDF; pdffonts:\n{fonts_out}")
    if "texgyretermesmath" not in fonts_lower:
        raise AssertionError(f"Expected TeX Gyre Termes Math embedded in compiled PDF; pdffonts:\n{fonts_out}")
    if "notosansmono" in fonts_lower:
        raise AssertionError(f"Found unexpected heavy NotoSansMono font in Q1-styled PDF; pdffonts:\n{fonts_out}")

    backtick_nums = re.findall(r"`[\$~\+\-]?\d[\d,\.%kMGBsdx/\-\s]*`", report_txt)
    if backtick_nums:
        raise AssertionError(f"Found backtick-wrapped numeric tokens in report.md (violates Q1 typography): {backtick_nums[:10]}")

    docx_path = ROOT / "report" / "QuanHuuNhatMinh_11230998_Final_Report.docx"
    if not docx_path.exists():
        raise AssertionError(f"Missing compiled DOCX: {docx_path}")
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    doc = Document(str(docx_path))
    normal_st = doc.styles["Normal"]
    if normal_st.font.name != "Times New Roman" or normal_st.font.size != Pt(12):
        raise AssertionError(
            f"DOCX Normal style mismatch: font={normal_st.font.name}, size={normal_st.font.size}"
        )
    if doc.paragraphs[0].alignment != WD_ALIGN_PARAGRAPH.CENTER:
        raise AssertionError("DOCX main title paragraph is not centered")
    justified_body = sum(
        1 for p in doc.paragraphs
        if (p.style and p.style.name in ("Body Text", "First Paragraph", "Normal"))
        and len(p.text.strip()) > 80
        and p.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY
    )
    if justified_body < 10:
        raise AssertionError(f"Expected justified body paragraphs in DOCX, found only {justified_body}")

    # 4. Forbidden stale numbers in report.md and README.md
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
        "0.5541",
        "0.9845",
        "0.0026",
        "471 / 485",
        "97.11%",
        "3.657%",
        "0.4699",
        "917,074",
        "678,370",
        "87,172",
        "data/fdic_download_starter.py",
        "cryptographically locked",
    ]
    stale_found = [
        tok for tok in stale_tokens if (tok in report_txt or tok in readme_txt)
    ]
    if stale_found:
        raise AssertionError(f"Found stale tokens in report.md or README.md: {stale_found}")

    log.info(
        f"All report.md and README.md numbers verified against JSON/CSV outputs "
        f"(PDF = {main_pages} main <= 20 + {appendix_pages} appendix <= 10; total {pages} pages) with zero stale tokens!"
    )


if __name__ == "__main__":
    main()
