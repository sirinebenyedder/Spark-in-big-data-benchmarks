import json
import time
from pyspark.sql import functions as F
from bench_common import (
    PARQUET_PATH,
    RESULTS_DIR,
    build_spark,
    build_summary_metrics,
    save_metrics,
)


def _get_b0_duration() -> float | None:
    b0_file = RESULTS_DIR / "b0_pandas_chunked.json"
    if b0_file.exists():
        return float(json.loads(b0_file.read_text(encoding="utf-8"))["duration_s"])
    return None


def run_b1() -> None:
    nodes_p = 1
    partitions_p = 16
    aqe_enabled = False

    spark = build_spark(
        app_name="metropt3-b1-mono-noeud",
        benchmark_slug="b1_mono_noeud",
        nodes_p=nodes_p,
        shuffle_partitions=partitions_p,
        aqe_enabled=aqe_enabled,
    )
    try:
        t0 = time.perf_counter()
        df = (
            spark.read.parquet(str(PARQUET_PATH))
            .select(
                "equipment_id",
                "win_start",
                "n_obs",
                "TP2_mean_5m",
                "TP3_mean_5m",
                "Oil_temperature_mean_5m",
                "Motor_current_mean_5m",
            )
            .repartition(partitions_p)
        )
        agg_df = (
            df.withColumn("hour_ts", F.date_trunc("hour", F.col("win_start")))
            .groupBy("equipment_id", "hour_ts")
            .agg(
                F.count("*").alias("windows_5m"),
                F.sum("n_obs").alias("samples"),
                (F.sum(F.col("TP2_mean_5m") * F.col("n_obs")) / F.sum("n_obs")).alias("avg_tp2"),
                F.max("TP3_mean_5m").alias("max_tp3"),
                (F.sum(F.col("Motor_current_mean_5m") * F.col("n_obs")) / F.sum("n_obs")).alias("avg_motor_current"),
            )
        )
        row_count = agg_df.count()
        elapsed = time.perf_counter() - t0

        b0_s = _get_b0_duration()
        metrics = build_summary_metrics(
            code="B1",
            nom="PySpark Mono-nœud (5min)",
            type_exec="PySpark Baseline",
            nodes_p=nodes_p,
            partitions_p=partitions_p,
            aqe_enabled=aqe_enabled,
            duration_s=elapsed,
            baseline_b0_s=b0_s,
            extra={"hourly_groups": row_count},
        )
        save_metrics(metrics, "b1_spark_baseline.json")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_b1()