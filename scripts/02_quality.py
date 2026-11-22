"""Part 2: automated data-quality audit (report + missingness figure)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ["POLARS_MAX_THREADS"] = "4"

import matplotlib.pyplot as plt
import polars as pl

from src.bfews.quality import audit_missingness_by_era, generate_quality_report
from src.bfews.io import read_raw_all
from src.bfews.utils import get_logger, out_dir

log = get_logger("02_quality")


def plot_missingness_era(era: pl.DataFrame) -> None:
    d = era.to_pandas()
    x = list(range(len(d)))
    x_labels = [
        lbl.replace(" (4 qtrs)", "\n(4 qtrs)").replace("-", "–")
        for lbl in d["era"].astype(str).tolist()
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    for col, mk in [("P3ASSET null %", "o"), ("ELNATR null %", "s"), ("NTLNLS null %", "^"), ("P9ASSET null %", "d")]:
        ax.plot(x, d[col], marker=mk, lw=2, label=col.replace(" null %", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.set_xlabel("Regulatory Era (sample begins 1984Q1)")
    ax.set_ylabel("Share of rows that are null (%)")
    ax.set_title("Structural Missingness in FDIC Call Reports (1984Q1–2026Q2)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p = out_dir("figures") / "missingness_by_era.png"
    plt.savefig(p, dpi=200)
    plt.close()
    log.info(f"Saved {p}")


def main() -> None:
    log.info("Part 2: automated data-quality audit...")
    summary = generate_quality_report()
    log.info(f"Report written; failures summary: {summary['failures']}")
    plot_missingness_era(audit_missingness_by_era(read_raw_all()))


if __name__ == "__main__":
    main()
