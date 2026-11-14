"""Generate executed Jupyter notebooks and the complete English Report (Markdown + DOCX)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["POLARS_MAX_THREADS"] = "4"

import subprocess
import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
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


def execute_and_save_nb(cells_spec: list[tuple[str, str]], out_path: Path) -> None:
    """Create a v4 Jupyter Notebook, execute every code cell via nbclient, and write to disk."""
    nb = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split()[0],
            },
        }
    )
    for kind, src in cells_spec:
        if kind == "md":
            nb.cells.append(new_markdown_cell(src))
        else:
            nb.cells.append(new_code_cell(src))

    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(out_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, out_path)
    log.info(f"Executed and saved notebook: {out_path.name}")


def build_notebooks():
    """Generate and execute 5 structured, self-contained notebooks in final/notebooks/."""
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    # 1. Part 1 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 1: FDIC Data Acquisition, Verification & Lakehouse Storage Architecture\n"
                "This notebook verifies the reproducible acquisition of 170 quarters (1984Q1–2026Q2) of FDIC Call Report data, "
                "audits manifest completeness against API totals, and benchmarks our Hive-partitioned Zstandard Parquet Bronze layer against raw CSV and unpartitioned layouts.",
            ),
            (
                "code",
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n"
                "import polars as pl\nfrom src.bfews.quality import audit_manifest\n\n"
                "manifest_check = audit_manifest()\nprint('Manifest Audit Summary:', manifest_check)",
            ),
            (
                "code",
                "bench_df = pl.read_csv('../outputs/tables/part1_storage_benchmark.csv')\n"
                "print(bench_df)\n"
                "bronze_files = list(Path('../data/bronze/financials').rglob('*.parquet'))\n"
                "print(f'Bronze partition count: {len(bronze_files)} files across {len({p.parent for p in bronze_files})} year folders')",
            ),
        ],
        nb_dir / "01_acquisition_and_storage.ipynb",
    )

    # 2. Part 2 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 2: Automated Data Quality Audit & Point-in-Time Panel Construction\n"
                "Audits cross-quarter schema drift, structural missingness by regulatory era, balance-sheet invariants, "
                "YTD flow de-cumulation, and bank exit classification (FDIC failure vs. non-failure mergers).",
            ),
            (
                "code",
                "from pathlib import Path\nimport polars as pl\n\n"
                "dq_text = Path('../outputs/dq_report/data_quality_report.md').read_text()\n"
                "print(dq_text[:2500] + '\\n...[see outputs/dq_report/data_quality_report.md for full report]...')",
            ),
            (
                "code",
                "exit_df = pl.read_csv('../outputs/tables/bank_exit_summary.csv')\n"
                "print('=== Bank Exit Taxonomy ===')\nprint(exit_df)\n"
                "panel_schema = pl.read_parquet_schema('../data/silver/panel.parquet')\n"
                "print(f'Silver panel columns ({len(panel_schema)}):', list(panel_schema.keys())[:15], '...')",
            ),
        ],
        nb_dir / "02_data_quality_and_panel.ipynb",
    )

    # 3. Part 3 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 3: Scalable Processing — Polars Query Optimization, Pandas Benchmark & PySpark Parity\n"
                "Examines Polars lazy query optimization (predicate pushdown, projection pushdown, parallel window execution), "
                "benchmarks execution time and peak RSS against Pandas (naive vs column-pushdown), and verifies exact parity with PySpark.",
            ),
            (
                "code",
                "from pathlib import Path\nimport polars as pl\n\n"
                "print('=== Polars Optimized Query Plan ===')\n"
                "print(Path('../outputs/tables/polars_optimized_plan.txt').read_text())",
            ),
            (
                "code",
                "bench = pl.read_csv('../outputs/tables/part3_benchmark_results.csv')\n"
                "print('=== Part 3 Scaling & Memory Benchmark ===')\nprint(bench)",
            ),
            (
                "code",
                "import json\n"
                "parity = json.loads(Path('../outputs/tables/spark_polars_parity.json').read_text())\n"
                "print('=== PySpark vs Polars Parity Check ===')\n"
                "print(json.dumps(parity, indent=2))\n"
                "print('\\n=== PySpark Catalyst Execution Plan (Excerpt) ===')\n"
                "print(Path('../outputs/tables/spark_execution_plan.txt').read_text()[:1500])",
            ),
        ],
        nb_dir / "03_scalable_processing_polars_spark.ipynb",
    )

    # 4. Part 4 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 4: Leak-Free CAMELS Feature Engineering & Forward Failure Labeling\n"
                "Builds 26 point-in-time CAMELS, Texas Ratio, and 4-quarter historical lag features joined with the 4-quarter forward failure indicator $y_{i,q} = 1\\{\\text{fail in } (q, q+4]\\}$, and verifies the causal leak-free truncation property.",
            ),
            (
                "code",
                "import sys, datetime\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n"
                "import numpy as np\nimport polars as pl\nfrom src.bfews.features import FEATURE_COLS_CAMELS, compute_camels_features\n\n"
                "print(f'Total CAMELS features ({len(FEATURE_COLS_CAMELS)}):', FEATURE_COLS_CAMELS)\n"
                "print(pl.read_csv('../outputs/tables/gold_split_summary.csv'))",
            ),
            (
                "code",
                "# Live verification of leak-free causal truncation property\n"
                "dates = [datetime.date(2020, 3, 31), datetime.date(2020, 6, 30), datetime.date(2020, 9, 30), datetime.date(2020, 12, 31),\n"
                "         datetime.date(2021, 3, 31), datetime.date(2021, 6, 30), datetime.date(2021, 9, 30), datetime.date(2021, 12, 31)]\n"
                "n = len(dates)\n"
                "sample_df = pl.DataFrame({\n"
                "    'CERT': [101]*n, 'REPDTE': dates, 'ASSET': [1000.0 + i*50 for i in range(n)],\n"
                "    'EQ': [100.0 + i*5 for i in range(n)], 'LNLSGR': [600.0 + i*30 for i in range(n)],\n"
                "    'LNLSNET': [580.0 + i*30 for i in range(n)], 'NCLNLS': [10.0 + i for i in range(n)],\n"
                "    'ORE': [2.0]*n, 'LNATRES': [20.0]*n, 'DEP': [800.0]*n, 'CHBAL': [50.0]*n, 'SC': [100.0]*n,\n"
                "    'LNRE': [300.0]*n, 'LNCI': [150.0]*n, 'LNCON': [150.0]*n, 'ROA': [1.0]*n, 'ROE': [10.0]*n,\n"
                "    'NIMY': [3.5]*n, 'RBC1AAJ': [12.0]*n, 'RBCRWAJ': [14.0]*n, 'NONIX': [30.0]*n,\n"
                "    'INTINC': [40.0]*n, 'EINTEXP': [10.0]*n, 'NONII': [5.0]*n, 'ELNATR_Q': [2.0]*n,\n"
                "    'NETINC_Q': [10.0]*n, 'NAASSET': [5.0]*n, 'BRO': [10.0]*n\n"
                "})\n"
                "f_full = compute_camels_features(sample_df).filter(pl.col('REPDTE') == datetime.date(2020, 12, 31))\n"
                "f_trunc = compute_camels_features(sample_df.slice(0, 4)).filter(pl.col('REPDTE') == datetime.date(2020, 12, 31))\n"
                "diffs = {c: abs((f_full[c].item() or 0.0) - (f_trunc[c].item() or 0.0)) for c in ['eq_asset_ratio', 'texas_ratio', 'nclnls_ratio']}\n"
                "print('Max abs diff at 2020-12-31 between full (2020-2021) and truncated (2020 only) panel:', diffs)\n"
                "assert max(diffs.values()) == 0.0, 'Future leakage detected!'",
            ),
        ],
        nb_dir / "04_feature_pipeline.ipynb",
    )

    # 5. Part 5 Notebook
    execute_and_save_nb(
        [
            (
                "md",
                "# Part 5: Expanding Walk-Forward Validation, Prior-Odds Calibration, Operating Points & Held-Out Test Evaluation\n"
                "Compares supervisory baselines against regularized Logistic Regression, HistGradientBoosting, and PySpark MLlib, "
                "presents both fixed-budget supervisory capacity operating points (Top 1%, 2%, 5% per quarter) and the prior-odds calibrated cost-minimizing threshold $\\theta^* = 0.0026$, and summarizes the 2009Q1–2025Q2 held-out test set.",
            ),
            (
                "code",
                "import json\nfrom pathlib import Path\nimport polars as pl\n\n"
                "print('=== Walk-Forward Cross-Validation Comparison (DEV 1984-2007) ===')\n"
                "print(pl.read_csv('../outputs/tables/cv_model_comparison.csv'))",
            ),
            (
                "code",
                "res = json.loads(Path('../outputs/tables/test_results.json').read_text())\n"
                "cost_cfg = res['cost_threshold_tuning']\n"
                "t_eval = res['held_out_test_evaluation']\n"
                "print('=== DEV Cost-Optimal Threshold & Failing Bank Counts ===')\n"
                "print({\n"
                "    'theta_star': cost_cfg['theta'],\n"
                "    'theta_at_grid_edge': cost_cfg['theta_at_grid_edge'],\n"
                "    'dev_failing_banks_in_oof_blocks': cost_cfg['dev_failing_banks_in_oof_blocks'],\n"
                "    'dev_oof_banks_caught_at_theta': cost_cfg['dev_oof_banks_caught_at_theta'],\n"
                "    'dev_failing_banks_all_1984_2007': cost_cfg['dev_failing_banks_all'],\n"
                "})\n"
                "print('\\n=== Held-Out Test Overall Metrics (2009Q1-2025Q2) ===')\n"
                "print({k: t_eval[k] for k in ['pr_auc', 'roc_auc', 'recall_top_1pct', 'recall_top_5pct', 'brier_score', 'rows', 'positive_rows', 'failing_banks_total']})\n"
                "print('\\n=== Held-Out Test Capacity Operating Points ===')\n"
                "print(pl.DataFrame(t_eval['capacity_points']))\n"
                "print('\\n=== Held-Out Test at Cost-Optimal Threshold (theta* = 0.0026) ===')\n"
                "print(t_eval['at_threshold'])",
            ),
        ],
        nb_dir / "05_modelling_and_validation.ipynb",
    )
    log.info(f"Generated and executed 5 notebooks in {nb_dir}")


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
    try:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "-env:UserInstallation=file:///tmp/lo_profile",
                "--convert-to",
                "pdf",
                str(docx_path),
                "--outdir",
                str(md_path.parent),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log.info(f"Converted DOCX to PDF at {md_path.parent / 'BFT65B_Final_Report.pdf'}")
    except Exception as exc:
        log.warning(f"LibreOffice PDF conversion skipped or failed: {exc}")
    return docx_path


def main():
    build_notebooks()
    md_path = update_report_metadata()
    build_docx_report(md_path)


if __name__ == "__main__":
    main()
