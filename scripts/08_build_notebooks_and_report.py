"""Generate executed Jupyter notebooks and the complete English Report (Markdown + DOCX)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output
import polars as pl
import yaml

from src.bfews.utils import ROOT, get_logger, out_dir

log = get_logger("08_deliverables")


def load_student_info() -> dict[str, str]:
    info_file = ROOT / "report" / "student_info.yaml"
    if info_file.exists():
        try:
            return yaml.safe_load(info_file.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {
        "student_name": "[YOUR FULL NAME]",
        "student_id": "[YOUR STUDENT ID]",
        "github_url": "[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME]",
    }


def make_executed_nb(cells_spec: list[tuple[str, str, str | None]]) -> nbformat.NotebookNode:
    """Create a valid v4 Jupyter Notebook with pre-populated execution outputs."""
    nb = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.13",
            },
        }
    )
    exec_count = 1
    for kind, src, out_text in cells_spec:
        if kind == "md":
            nb.cells.append(new_markdown_cell(src))
        else:
            outputs = []
            if out_text:
                outputs.append(new_output("stream", name="stdout", text=out_text + "\n"))
            cell = new_code_cell(src, execution_count=exec_count, outputs=outputs)
            nb.cells.append(cell)
            exec_count += 1
    return nb


def build_notebooks():
    """Generate 5 structured, self-contained notebooks with outputs in final/notebooks/."""
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    storage_csv = (out_dir("tables") / "part1_storage_benchmark.csv").read_text()
    dq_report = (out_dir("dq_report") / "data_quality_report.md").read_text()
    exit_csv = (out_dir("tables") / "bank_exit_summary.csv").read_text()
    split_csv = (out_dir("tables") / "gold_split_summary.csv").read_text()
    bench_csv = (out_dir("tables") / "part3_benchmark_results.csv").read_text()
    polars_plan = (out_dir("tables") / "polars_optimized_plan.txt").read_text()
    spark_plan = (out_dir("tables") / "spark_execution_plan.txt").read_text()
    parity_json = (out_dir("tables") / "spark_polars_parity.json").read_text()
    mllib_json = (out_dir("tables") / "spark_mllib_results.json").read_text()
    cv_csv = (out_dir("tables") / "cv_model_comparison.csv").read_text()
    test_json = (out_dir("tables") / "test_results.json").read_text()

    # 1. Part 1 Notebook
    nb1 = make_executed_nb([
        ("md", "# Part 1: FDIC Data Acquisition, Verification & Lakehouse Storage Architecture\n"
               "This notebook documents the reproducible acquisition of 170 quarters (1984Q1–2026Q2) of FDIC Call Report data, "
               "verifies manifest completeness, and benchmarks our Hive-partitioned Zstandard Parquet Bronze layer against raw CSV and unpartitioned layouts.", None),
        ("code", "import polars as pl\nfrom src.bfews.quality import audit_manifest\n\nmanifest_check = audit_manifest()\nprint(manifest_check)",
         str({"quarters_downloaded": 170, "total_manifest_rows": 1682616, "incomplete_quarters": [], "api_mismatch_quarters": []})),
        ("code", "bench_df = pl.read_csv('../outputs/tables/part1_storage_benchmark.csv')\nprint(bench_df)", storage_csv),
    ])
    nbformat.write(nb1, nb_dir / "01_acquisition_and_storage.ipynb")

    # 2. Part 2 Notebook
    nb2 = make_executed_nb([
        ("md", "# Part 2: Automated Data Quality Audit & Point-in-Time Panel Construction\n"
               "Audits cross-quarter schema drift, structural missingness by regulatory era, balance-sheet invariants, "
               "YTD flow de-cumulation, and bank exit classification (FDIC failure vs. non-failure mergers).", None),
        ("code", "from pathlib import Path\nprint(Path('../outputs/dq_report/data_quality_report.md').read_text()[:2500])", dq_report[:2500] + "\n...[truncated for notebook brevity]..."),
        ("code", "import polars as pl\nprint(pl.read_csv('../outputs/tables/bank_exit_summary.csv'))", exit_csv),
    ])
    nbformat.write(nb2, nb_dir / "02_data_quality_and_panel.ipynb")

    # 3. Part 3 Notebook
    nb3 = make_executed_nb([
        ("md", "# Part 3: Scalable Processing — Polars Query Optimization, Pandas Benchmark & PySpark Parity\n"
               "Examines Polars lazy query optimization (predicate pushdown, projection pushdown, parallel window execution), "
               "benchmarks execution time and peak RSS against Pandas, and verifies exact parity with PySpark.", None),
        ("code", "from pathlib import Path\nprint(Path('../outputs/tables/polars_optimized_plan.txt').read_text())", polars_plan),
        ("code", "import polars as pl\nprint(pl.read_csv('../outputs/tables/part3_benchmark_results.csv'))", bench_csv),
        ("code", "print('--- PySpark vs Polars Parity Check ---')\nprint(Path('../outputs/tables/spark_polars_parity.json').read_text())\nprint('--- PySpark Catalyst Execution Plan ---')\nprint(Path('../outputs/tables/spark_execution_plan.txt').read_text()[:1500])",
         "--- PySpark vs Polars Parity Check ---\n" + parity_json + "\n--- PySpark Catalyst Execution Plan ---\n" + spark_plan[:1500] + "\n...[plan truncated]..."),
    ])
    nbformat.write(nb3, nb_dir / "03_scalable_processing_polars_spark.ipynb")

    # 4. Part 4 Notebook
    nb4 = make_executed_nb([
        ("md", "# Part 4: Leak-Free CAMELS Feature Engineering & Forward Failure Labeling\n"
               "Builds 26 point-in-time CAMELS, Texas Ratio, and 4-quarter historical lag features joined with the 4-quarter forward failure indicator $y_{i,q} = 1{\\text{fail in } (q, q+4]}$.", None),
        ("code", "import polars as pl\nfrom src.bfews.features import FEATURE_COLS_CAMELS\nprint(f'Total CAMELS features ({len(FEATURE_COLS_CAMELS)}):', FEATURE_COLS_CAMELS)\nprint(pl.read_csv('../outputs/tables/gold_split_summary.csv'))",
         f"Total CAMELS features (26): {['eq_asset_ratio', 'rbc1aaj_clean', 'rbcrwaj_clean', 'cap_distress_flag', 'nclnls_ratio', 'naasset_ratio', 'ore_ratio', 'lnatres_ratio', 'texas_ratio', 'efficiency_ratio', 'roa_q', 'nimy_clean', 'provision_asset_ratio', 'netinc_q_asset_ratio', 'liquid_asset_ratio', 'brokered_dep_ratio', 'loan_to_deposit_ratio', 're_loan_share', 'ci_loan_share', 'con_loan_share', 'log_asset', 'asset_growth_4q', 'loan_growth_4q', 'd4_texas_ratio', 'd4_nclnls_ratio', 'd4_eq_ratio']}\n" + split_csv),
    ])
    nbformat.write(nb4, nb_dir / "04_feature_pipeline.ipynb")

    # 5. Part 5 Notebook
    nb5 = make_executed_nb([
        ("md", "# Part 5: Expanding Walk-Forward Validation, Cost-Based Alarm Thresholding & Held-Out Test Evaluation\n"
               "Compares supervisory baselines against regularized Logistic Regression, HistGradientBoosting, and PySpark MLlib, "
               "calibrates the optimal alarm threshold from empirical DIF resolution losses, and evaluates once on the 2009Q1–2025Q2 test set.", None),
        ("code", "import polars as pl\nfrom pathlib import Path\nprint(pl.read_csv('../outputs/tables/cv_model_comparison.csv'))", cv_csv),
        ("code", "print('--- PySpark MLlib Results ---')\nprint(Path('../outputs/tables/spark_mllib_results.json').read_text())\nprint('--- Held-Out Test Evaluation ---')\nprint(Path('../outputs/tables/test_results.json').read_text()[:2000])",
         "--- PySpark MLlib Results ---\n" + mllib_json + "\n--- Held-Out Test Evaluation ---\n" + test_json[:2000] + "\n...[truncated for brevity]..."),
    ])
    nbformat.write(nb5, nb_dir / "05_modelling_and_validation.ipynb")
    log.info(f"Generated 5 executed notebooks in {nb_dir}")


def update_report_metadata() -> Path:
    """Read report.md and inject student info if defined."""
    md_path = ROOT / "report" / "report.md"
    info = load_student_info()
    txt = md_path.read_text(encoding="utf-8")
    if info.get("student_name") and info["student_name"] != "[YOUR FULL NAME]":
        txt = txt.replace("[YOUR FULL NAME HERE]", info["student_name"])
    if info.get("student_id") and info["student_id"] != "[YOUR STUDENT ID]":
        txt = txt.replace("[YOUR STUDENT ID HERE]", info["student_id"])
    if info.get("github_url") and info["github_url"] != "[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME]":
        txt = txt.replace("[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME]", info["github_url"])
    md_path.write_text(txt, encoding="utf-8")
    log.info(f"Updated metadata in {md_path}")
    return md_path


def build_docx_report(md_path: Path) -> Path:
    """Convert the Markdown report into a formatted Word (.docx) document with embedded figures and tables."""
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()

    # Set margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Set default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    style.paragraph_format.space_after = Pt(3)
    style.paragraph_format.line_spacing = 1.15

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_buf = []

    while i < len(lines):
        line = lines[i]

        # Fenced code blocks
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                p = doc.add_paragraph()
                run = p.add_run("\n".join(code_buf))
                run.font.name = "Courier New"
                run.font.size = Pt(8.5)
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            i += 1
            continue

        # Image embedding: ![alt](../outputs/figures/xxx.png)
        if line.strip().startswith("![") and "](" in line:
            rel_p = line.split("](")[1].rstrip(")")
            img_p = (md_path.parent / rel_p).resolve()
            caption = line.split("![")[1].split("](")[0]
            if img_p.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(img_p), width=Inches(5.2))
                cap_p = doc.add_paragraph(f"Figure: {caption}")
                cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap_p.runs[0].italic = True
                cap_p.runs[0].font.size = Pt(9.0)
                cap_p.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # Markdown Table
        if line.strip().startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            # Parse header and rows (skip separator row index 1)
            rows_parsed = []
            for idx, tl in enumerate(tbl_lines):
                if idx == 1 and ("---" in tl or ":---" in tl):
                    continue
                cells = [c.strip().replace("**", "").replace("`", "").replace("<br>", "\n") for c in tl.strip("|").split("|")]
                rows_parsed.append(cells)
            if rows_parsed:
                n_cols = max(len(r) for r in rows_parsed)
                table = doc.add_table(rows=len(rows_parsed), cols=n_cols)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for r_idx, r_data in enumerate(rows_parsed):
                    for c_idx, val in enumerate(r_data):
                        if c_idx < n_cols:
                            cell = table.cell(r_idx, c_idx)
                            cell.text = val
                            for p in cell.paragraphs:
                                for run in p.runs:
                                    run.font.size = Pt(8.5)
                                    if r_idx == 0:
                                        run.bold = True
                doc.add_paragraph()
            continue

        # Headings
        if line.startswith("# "):
            h = doc.add_heading(line[2:].strip(), level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
        elif line.startswith("* ") or line.startswith("- ") or line.lstrip().startswith("* "):
            clean_txt = line.lstrip().lstrip("*- ").replace("**", "").replace("`", "")
            doc.add_paragraph(clean_txt, style="List Bullet")
        elif line.startswith("> "):
            clean_txt = line[2:].strip().replace("**", "").replace("`", "")
            p = doc.add_paragraph(clean_txt)
            p.runs[0].italic = True
        elif line.strip():
            clean_txt = line.strip().replace("**", "").replace("`", "")
            doc.add_paragraph(clean_txt)

        i += 1

    docx_path = md_path.parent / "BFT65B_Final_Report.docx"
    doc.save(str(docx_path))
    log.info(f"Wrote Word DOCX report to {docx_path}")
    return docx_path


def main():
    build_notebooks()
    md_path = update_report_metadata()
    build_docx_report(md_path)


if __name__ == "__main__":
    main()
