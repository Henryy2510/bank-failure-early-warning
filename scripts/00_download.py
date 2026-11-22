"""Part 1: Reproducible & Resumable FDIC Data Acquisition and Verification Script.

Adapted from the BFT65B course starter (`fdic_download_starter.py`).
Supports:
- `--verify-only`: Offline verification of already-downloaded raw files against manifest totals.
- Full resumable API download from `https://api.fdic.gov/banks` (1984Q1 -> 2026Q2).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["POLARS_MAX_THREADS"] = "4"

import polars as pl
import requests

from src.bfews.utils import ROOT, raw_dir

BASE = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000

FIN_FIELDS = [
    "CERT", "REPDTE", "STNAME",
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    "ROA", "ROE", "NIMY",
    "RBC1AAJ", "RBCRWAJ",
]
FAIL_FIELDS = ["CERT", "NAME", "CITYST", "FAILDATE", "FAILYR", "COST", "QBFASSET", "QBFDEP", "RESTYPE"]
INST_FIELDS = ["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "ESTYMD", "ENDEFYMD", "CHANGEC1", "INSDATE"]


def verify_completeness(out: Path) -> bool:
    """Verify local raw dataset against manifest totals and check primary-key uniqueness."""
    manifest_p = out / "manifest_financials.json"
    if not manifest_p.exists():
        print(f"Manifest not found at {manifest_p}")
        return False

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    files = sorted((out / "financials").glob("REPDTE=*.parquet"))
    total_rows = 0
    dup_count = 0
    certs = set()

    for f in files:
        df = pl.read_parquet(f, columns=["CERT", "REPDTE"])
        total_rows += df.height
        dup_count += int(df.is_duplicated().sum())
        certs.update(df["CERT"].to_list())

    bad_quarters = [q for q, m in manifest.items() if not m.get("complete", False)]
    fails = pl.read_parquet(out / "failures.parquet")
    insts = pl.read_parquet(out / "institutions.parquet")

    print("=== FDIC Raw Dataset Completeness Check ===")
    print(f"Quarters in manifest    : {len(manifest)}")
    print(f"Quarterly parquet files : {len(files)}")
    print(f"Total panel rows        : {total_rows:,}")
    print(f"Distinct banks (CERT)   : {len(certs):,}")
    print(f"Duplicate (CERT,REPDTE) : {dup_count}")
    print(f"Incomplete quarters     : {bad_quarters if bad_quarters else 'none'}")
    print(f"Failures table rows     : {fails.height:,}")
    print(f"Institutions table rows : {insts.height:,}")
    return dup_count == 0 and not bad_quarters and len(files) == len(manifest)


def run_resumable_download(out: Path, start: str, end: str) -> None:
    """Invoke the resumable FDIC BankFind API downloader (`fdic_download_starter.py`)."""
    starter = ROOT / "fdic_download_starter.py"
    if not starter.exists():
        raise FileNotFoundError(f"Resumable downloader script not found at {starter}")
    cmd = [sys.executable, str(starter), "--start", start, "--end", end, "--out", str(out)]
    print(f"Launching resumable FDIC API download: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(raw_dir()), help="Raw FDIC directory")
    ap.add_argument("--download", action="store_true", default=False,
                    help="Run full resumable API download from FDIC BankFind before verifying")
    ap.add_argument("--start", default="1984-03-31", help="Start quarter-end YYYY-MM-DD (used with --download)")
    ap.add_argument("--end", default="2026-06-30", help="End quarter-end YYYY-MM-DD (used with --download)")
    ap.add_argument("--verify-only", action="store_true", default=False,
                    help="Verify local raw data without hitting network (default unless --download is set)")
    args = ap.parse_args()

    out = Path(args.out)
    if args.download and not args.verify_only:
        run_resumable_download(out, args.start, args.end)
    ok = verify_completeness(out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
