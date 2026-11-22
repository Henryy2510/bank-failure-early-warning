"""PySpark: core transformations (same logic as core.py), execution plan, per-year parity, MLlib demo."""
from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from pathlib import Path

from .core import texas_cap
from .utils import cfg, data_dir, get_logger

log = get_logger("spark")


def get_spark_session(app_name: str = "BFT65B-BankFailurePipeline", master: str | None = None):
    """Local SparkSession with a conservative memory configuration for a laptop."""
    import os
    import re
    import sys
    import tempfile
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    derby_home = (Path(tempfile.gettempdir()) / "derby").as_posix()
    java_opts = (
        f"-XX:+UseG1GC -Dderby.system.home={derby_home}"
        if " " not in derby_home
        else f'-XX:+UseG1GC "-Dderby.system.home={derby_home}"'
    )
    s = cfg()["spark"]
    m = master or s["master"]
    match = re.fullmatch(r"local\[(\d+)\]", m.strip())
    default_par = match.group(1) if match else "4"
    return (SparkSession.builder.appName(app_name).master(m)
            .config("spark.driver.memory", s["driver_memory"])
            .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
            .config("spark.default.parallelism", default_par)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.extraJavaOptions", java_opts)
            .getOrCreate())


SPARK_OUT = ["NETINC_Q", "eq_asset_ratio", "texas_ratio", "asset_growth_4q", "d4_texas_ratio"]


def spark_core(spark, bronze_path: Path):
    """The six core steps from core.py, written with Spark window functions."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    cols = ["CERT", "REPDTE", "YEAR", "ASSET", "EQ", "NCLNLS", "ORE", "LNATRES", "NETINC"]
    df = spark.read.parquet(str(bronze_path)).select(*cols)
    cap = F.lit(texas_cap())
    w_year = Window.partitionBy("CERT", "YEAR").orderBy("REPDTE")
    w_cert = Window.partitionBy("CERT").orderBy("REPDTE")

    df = df.withColumn("quarter", F.quarter("REPDTE")).withColumn("qidx", F.col("YEAR") * 4 + F.col("quarter"))
    prev_ok = F.lag("qidx", 1).over(w_year) == F.col("qidx") - 1
    num = F.coalesce(F.col("NCLNLS"), F.lit(0.0)) + F.coalesce(F.col("ORE"), F.lit(0.0))
    den = F.coalesce(F.col("EQ"), F.lit(0.0)) + F.coalesce(F.col("LNATRES"), F.lit(0.0))
    df = (df.withColumn("NETINC_Q", F.when(F.col("quarter") == 1, F.col("NETINC"))
                        .when(prev_ok, F.col("NETINC") - F.lag("NETINC", 1).over(w_year)))
            .withColumn("eq_asset_ratio", F.col("EQ") / F.col("ASSET"))
            .withColumn("texas_ratio", F.when(den > 0, F.least(num / den, cap)).otherwise(cap)))
    lag4_ok = F.lag("qidx", 4).over(w_cert) == F.col("qidx") - 4
    return (df.withColumn("asset_growth_4q", F.when(lag4_ok, F.col("ASSET") / F.lag("ASSET", 4).over(w_cert) - 1))
              .withColumn("d4_texas_ratio", F.when(lag4_ok, F.col("texas_ratio") - F.lag("texas_ratio", 4).over(w_cert))))


def spark_yearly_summary(df) -> list[dict]:
    """Per-YEAR row count, column sums and null counts: a 43-row fingerprint of the full output."""
    from pyspark.sql import functions as F

    aggs = [F.count("*").alias("rows")]
    for c in SPARK_OUT:
        aggs += [F.sum(c).alias(f"{c}__sum"), F.sum(F.col(c).isNull().cast("int")).alias(f"{c}__nulls")]
    return [r.asDict() for r in df.groupBy("YEAR").agg(*aggs).orderBy("YEAR").collect()]


def spark_sample_rows(df, modulo: int = 50) -> list[dict]:
    """Row-level output for a deterministic subset of banks (CERT % modulo == 0)."""
    cols = ["CERT", "REPDTE"] + SPARK_OUT
    return [r.asDict() for r in df.filter(df.CERT % modulo == 0).select(*cols).collect()]


def spark_plan_text(df) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        df.explain(mode="formatted")
    return buf.getvalue()


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


MLLIB_FEATURES = ["eq_asset_ratio", "nclnls_ratio", "texas_ratio", "efficiency_ratio",
                  "roa_ytd", "liquid_asset_ratio", "loan_to_deposit_ratio", "log_asset"]
# demo split lives entirely inside the development period (the test period is never touched here)
MLLIB_TRAIN_END, MLLIB_VAL_START, MLLIB_VAL_END = "1988-12-31", "1990-03-31", "1992-12-31"


def run_spark_mllib_demo(gold_path: Path | None = None) -> dict:
    """MLlib logistic regression on a development-only split, compared with scikit-learn on the same rows.

    Train: <= 1988Q4; purge: 1989; validation: 1990Q1-1992Q4. Rows that could not have been scored in real time
    (`pit_scorable == False`) are excluded, as in the main model.
    """
    import numpy as np
    import polars as pl
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.sql import functions as F
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression as SkLR
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler as SkScaler

    gold_path = gold_path or data_dir("gold") / "features.parquet"
    spark = get_spark_session()
    f = MLLIB_FEATURES
    sdf = (spark.read.parquet(str(gold_path))
           .select(*f, "target_fail_4q", "split_group", "pit_scorable", "REPDTE")
           .filter((F.col("split_group") == "DEV") & F.col("pit_scorable"))
           .withColumn("label", F.col("target_fail_4q").cast("double")))
    for c in f:  # same clamp as the sklearn side: keeps ratios with near-zero denominators finite
        sdf = sdf.withColumn(c, F.greatest(F.least(F.col(c).cast("double"), F.lit(100.0)), F.lit(-100.0)))
    train = sdf.filter(F.col("REPDTE") <= F.lit(MLLIB_TRAIN_END).cast("date"))
    val = sdf.filter(F.col("REPDTE").between(F.lit(MLLIB_VAL_START).cast("date"), F.lit(MLLIB_VAL_END).cast("date")))

    imp = Imputer(inputCols=f, outputCols=[f"{c}_i" for c in f]).setStrategy("median").fit(train)
    asm = VectorAssembler(inputCols=[f"{c}_i" for c in f], outputCol="raw", handleInvalid="keep")
    prep_train = asm.transform(imp.transform(train))
    scaler = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True).fit(prep_train)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.0).fit(
        scaler.transform(prep_train))
    pred = lr.transform(scaler.transform(asm.transform(imp.transform(val))))
    ev = lambda m: BinaryClassificationEvaluator(labelCol="label", metricName=m).evaluate(pred)  # noqa: E731
    out = {"mllib_val_roc_auc": round(ev("areaUnderROC"), 4), "mllib_val_pr_auc": round(ev("areaUnderPR"), 4),
           "train_rows": int(train.count()), "val_rows": int(val.count()),
           "val_failure_rows": int(val.filter("label = 1").count()),
           "split": f"train <= {MLLIB_TRAIN_END}; validate {MLLIB_VAL_START}..{MLLIB_VAL_END}; 1989 purged"}

    # scikit-learn on exactly the same rows and features
    def to_np(sdf_):
        rows = sdf_.select(*f, "label").collect()
        arr = np.array([[np.nan if r[c] is None else r[c] for c in f] + [r["label"]] for r in rows], dtype=float)
        return arr[:, :-1], arr[:, -1]

    xtr, ytr = to_np(train)
    xva, yva = to_np(val)
    sk_imp = SimpleImputer(strategy="median").fit(xtr)
    sk_sc = SkScaler().fit(sk_imp.transform(xtr))
    sk = SkLR(C=1e6, max_iter=500).fit(sk_sc.transform(sk_imp.transform(xtr)), ytr)
    p = sk.predict_proba(sk_sc.transform(sk_imp.transform(xva)))[:, 1]
    out["sklearn_val_roc_auc"] = round(float(roc_auc_score(yva, p)), 4)
    out["sklearn_val_pr_auc"] = round(float(average_precision_score(yva, p)), 4)
    spark.stop()
    log.info(f"MLlib vs scikit-learn on the same development split: {out}")
    return out
