"""Raw -> bronze ingestion: canonical schema, typing, hive-partitioned Parquet layout."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .utils import data_dir, raw_dir

KEYS = ["CERT", "REPDTE", "STNAME"]

# Stocks (end-of-quarter balances, thousands USD)
STOCKS = ["ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
          "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
          "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "ORE"]
# Flows reported YEAR-TO-DATE (cumulative within the calendar year, thousands USD)
FLOWS_YTD = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]
# Ratios in percent (ROA/ROE/NIMY are annualised YTD ratios)
RATIOS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]
NUMERIC = STOCKS + FLOWS_YTD + RATIOS

# Canonical schema. The raw quarterly files disagree on dtypes (int64 / double / null for the
# same column, see the quality report), so every column is cast explicitly.
#  * CERT  -> Int32  (max ~ 60k, halves the key size vs int64)
#  * REPDTE-> Date   (raw: string "YYYYMMDD")
#  * amounts and ratios -> Float64 (nullable; float32 would lose precision on ASSET in kUSD > 16.7M)
#  * STNAME -> String (Parquet dictionary-encodes it: 56 distinct values)
SCHEMA = {"CERT": pl.Int32, "REPDTE": pl.Date, "STNAME": pl.String, **{c: pl.Float64 for c in NUMERIC}}


def raw_financial_files() -> list[Path]:
    return sorted((raw_dir() / "financials").glob("REPDTE=*.parquet"))


def read_raw_quarter(f: Path) -> pl.DataFrame:
    """Read one raw quarterly file and cast it to the canonical schema."""
    df = pl.read_parquet(f)
    exprs = []
    for c, t in SCHEMA.items():
        if c not in df.columns:
            exprs.append(pl.lit(None, dtype=t).alias(c))
        elif c == "REPDTE":
            exprs.append(pl.col(c).cast(pl.String).str.strptime(pl.Date, "%Y%m%d"))
        else:
            exprs.append(pl.col(c).cast(t, strict=True))
    return df.select(exprs)


def read_raw_all() -> pl.DataFrame:
    return pl.concat([read_raw_quarter(f) for f in raw_financial_files()], how="vertical")


def raw_dtype_table() -> pl.DataFrame:
    """Physical dtype of every column in every raw file (schema-drift audit)."""
    rows = []
    for f in raw_financial_files():
        sch = pq.read_schema(f)
        for name, typ in zip(sch.names, sch.types):
            rows.append({"file": f.name, "column": name, "dtype": str(typ)})
    return pl.DataFrame(rows)


def manifest() -> dict:
    return json.loads((raw_dir() / "manifest_financials.json").read_text())


# ---------------------------------------------------------------------------
# Bronze layer
# ---------------------------------------------------------------------------
def bronze_financials_path() -> Path:
    return data_dir("bronze") / "financials"


def write_bronze_financials(df: pl.DataFrame, compression: str = "zstd", level: int = 3) -> Path:
    """Hive-partitioned by YEAR, sorted by (REPDTE, CERT) inside each file so that row-group
    min/max statistics allow predicate pushdown on dates and banks."""
    out = bronze_financials_path()
    if out.exists():
        import shutil
        shutil.rmtree(out)
    (df.with_columns(pl.col("REPDTE").dt.year().alias("YEAR"))
       .sort(["REPDTE", "CERT"])
       .write_parquet(out, compression=compression, compression_level=level,
                      statistics=True, row_group_size=50_000, partition_by=["YEAR"]))
    return out


def scan_bronze() -> pl.LazyFrame:
    return pl.scan_parquet(bronze_financials_path(), hive_partitioning=True)


def parse_failures() -> pl.DataFrame:
    return (pl.read_parquet(raw_dir() / "failures.parquet")
              .with_columns(pl.col("CERT").cast(pl.Int32),
                            pl.col("FAILDATE").str.strptime(pl.Date, "%m/%d/%Y"),
                            pl.col("FAILYR").cast(pl.Int32)))


def parse_institutions() -> pl.DataFrame:
    def d(c):
        return pl.col(c).str.strptime(pl.Date, "%m/%d/%Y", strict=False)
    return (pl.read_parquet(raw_dir() / "institutions.parquet")
              .with_columns(pl.col("CERT").cast(pl.Int32), pl.col("CHANGEC1").cast(pl.Int32),
                            pl.col("ACTIVE").cast(pl.Int8),
                            d("ESTYMD"), d("ENDEFYMD"), d("INSDATE")))
