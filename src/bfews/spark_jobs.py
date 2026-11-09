"""PySpark implementations of core transformations, execution plan audit, and MLlib demo (Part 3 & 5)."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from .utils import cfg, data_dir, get_logger

log = get_logger("spark")


def get_spark_session(app_name: str = "BFT65B-BankFailurePipeline"):
    """Create local SparkSession with conservative memory configuration for laptops."""
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[2]")
        .config("spark.driver.memory", "1200m")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -Dderby.system.home=/tmp/derby")
    )
    return builder.getOrCreate()


def run_spark_pipeline(bronze_path: Path | None = None) -> tuple[dict, str]:
    """Execute core bank pipeline in PySpark, extract formatted execution plan, and collect metrics."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    if bronze_path is None:
        bronze_path = data_dir("bronze") / "financials"

    spark = get_spark_session()
    log.info(f"Reading bronze Parquet via PySpark from {bronze_path}...")
    cols = ["CERT", "REPDTE", "YEAR", "ASSET", "EQ", "LNLSGR", "NCLNLS", "ORE", "LNATRES", "NETINC", "INTINC"]
    df = spark.read.parquet(str(bronze_path)).select(*cols)

    # 1. Window specs
    w_year = Window.partitionBy("CERT", "YEAR").orderBy("REPDTE")
    w_cert = Window.partitionBy("CERT").orderBy("REPDTE")

    # 2. De-cumulate quarterly flows
    quarter_col = F.quarter("REPDTE")
    df = df.withColumn("quarter", quarter_col)

    df_trans = (
        df.withColumn(
            "NETINC_Q",
            F.when(quarter_col == 1, F.col("NETINC")).otherwise(
                F.col("NETINC") - F.lag("NETINC", 1).over(w_year)
            ),
        )
        .withColumn(
            "INTINC_Q",
            F.when(quarter_col == 1, F.col("INTINC")).otherwise(
                F.col("INTINC") - F.lag("INTINC", 1).over(w_year)
            ),
        )
    )

    # 3. CAMELS and Texas Ratio
    eps = 1e-4
    df_trans = (
        df_trans.withColumn("eq_asset_ratio", F.col("EQ") / (F.col("ASSET") + eps))
        .withColumn("nclnls_ratio", F.col("NCLNLS") / (F.col("LNLSGR") + eps))
        .withColumn(
            "texas_ratio",
            (F.coalesce(F.col("NCLNLS"), F.lit(0.0)) + F.coalesce(F.col("ORE"), F.lit(0.0)))
            / (F.coalesce(F.col("EQ"), F.lit(0.0)) + F.coalesce(F.col("LNATRES"), F.lit(0.0)) + eps),
        )
    )

    # 4. Lag dynamics
    df_trans = df_trans.withColumn("d4_texas_ratio", F.col("texas_ratio") - F.lag("texas_ratio", 4).over(w_cert))

    # Capture physical execution plan
    f = io.StringIO()
    with redirect_stdout(f):
        df_trans.explain(mode="formatted")
    plan_str = f.getvalue()

    # Single aggregation pass to compute count + means simultaneously
    agg_row = df_trans.select([
        F.count("*").alias("row_count"),
        F.round(F.mean("eq_asset_ratio"), 5).alias("mean_eq_ratio"),
        F.round(F.mean("texas_ratio"), 5).alias("mean_texas_ratio"),
        F.round(F.mean("NETINC_Q"), 2).alias("mean_netinc_q"),
    ]).collect()[0]

    summary = {
        "row_count": int(agg_row["row_count"]),
        "mean_eq_ratio": float(agg_row["mean_eq_ratio"]),
        "mean_texas_ratio": float(agg_row["mean_texas_ratio"]),
        "mean_netinc_q": float(agg_row["mean_netinc_q"]),
    }

    return summary, plan_str


def run_spark_mllib_demo(gold_path: Path | None = None) -> dict:
    """Train a PySpark MLlib Logistic Regression model on the Gold feature dataset (Part 5 requirement)."""
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.sql import functions as F

    if gold_path is None:
        gold_path = data_dir("gold") / "features.parquet"

    spark = get_spark_session()
    log.info(f"Reading Gold features for MLlib from {gold_path}...")

    feature_cols = [
        "eq_asset_ratio", "nclnls_ratio", "texas_ratio", "efficiency_ratio",
        "roa_q", "liquid_asset_ratio", "loan_to_deposit_ratio", "log_asset"
    ]

    df_model = (
        spark.read.parquet(str(gold_path))
        .select(*(feature_cols + ["target_fail_4q", "split_group"]))
        .filter(F.col("split_group").isin("DEV", "TEST"))
        .dropna(subset=["target_fail_4q"])
        .withColumn("target_fail_4q", F.col("target_fail_4q").cast("double"))
    )
    for c in feature_cols:
        df_model = df_model.withColumn(
            c,
            F.when(F.col(c).isNull() | F.isnan(F.col(c)), F.lit(float("nan")))
            .otherwise(F.greatest(F.least(F.col(c).cast("double"), F.lit(100.0)), F.lit(-100.0))),
        )

    imputer = Imputer(inputCols=feature_cols, outputCols=[f"{c}_imp" for c in feature_cols]).setStrategy("median")
    assembler = VectorAssembler(inputCols=[f"{c}_imp" for c in feature_cols], outputCol="raw_features", handleInvalid="keep")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

    train_df = df_model.filter(F.col("split_group") == "DEV")
    test_df = df_model.filter(F.col("split_group") == "TEST")

    imp_model = imputer.fit(train_df)
    imputed_train = imp_model.transform(train_df)
    assembled_train = assembler.transform(imputed_train)
    scaler_model = scaler.fit(assembled_train)
    scaled_train = scaler_model.transform(assembled_train)

    lr = LogisticRegression(featuresCol="features", labelCol="target_fail_4q", maxIter=20, regParam=0.01)
    lr_model = lr.fit(scaled_train)

    imputed_test = imp_model.transform(test_df)
    assembled_test = assembler.transform(imputed_test)
    scaled_test = scaler_model.transform(assembled_test)
    predictions = lr_model.transform(scaled_test)

    evaluator_roc = BinaryClassificationEvaluator(labelCol="target_fail_4q", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_pr = BinaryClassificationEvaluator(labelCol="target_fail_4q", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    roc_auc = evaluator_roc.evaluate(predictions)
    pr_auc = evaluator_pr.evaluate(predictions)

    n_train = int(train_df.count())
    n_test = int(test_df.count())
    spark.stop()

    log.info(f"Spark MLlib Test ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    return {
        "mllib_test_roc_auc": round(roc_auc, 4),
        "mllib_test_pr_auc": round(pr_auc, 4),
        "train_rows": n_train,
        "test_rows": n_test,
    }
