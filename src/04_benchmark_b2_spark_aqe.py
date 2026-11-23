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


def run_b2() -> None:
    nodes_p = 4
    partitions_p = 64
    aqe_enabled = True

    spark = build_spark(
        app_name="metropt3-b2-config-fixe-aqe",
        benchmark_slug="b2_config_fixe_aqe",
        nodes_p=nodes_p,
        shuffle_partitions=partitions_p,
        aqe_enabled=aqe_enabled,
    )
    try:
        t0 = time.perf_counter()
        df = (
            spark.read.parquet(str(PARQUET_PATH))
            .select("equipment_id", "event_ts", "TP2", "TP3", "Oil_temperature", "Motor_current")
            .repartition(partitions_p)
        )
        agg_df = (
            df.withColumn("hour_ts", F.date_trunc("hour", F.col("event_ts")))
            .groupBy("equipment_id", "hour_ts")
            .agg(
                F.count("*").alias("samples"),
                F.avg("TP2").alias("avg_tp2"),
                F.max("TP3").alias("max_tp3"),
                F.avg("Motor_current").alias("avg_motor_current"),
            )
        )
        row_count = agg_df.count()
        elapsed = time.perf_counter() - t0

        b0_s = _get_b0_duration()
        metrics = build_summary_metrics(
            code="B2",
            nom="PySpark Configuration Fixe + AQE",
            type_exec="PySpark Operational",
            nodes_p=nodes_p,
            partitions_p=partitions_p,
            aqe_enabled=aqe_enabled,
            duration_s=elapsed,
            baseline_b0_s=b0_s,
            extra={"hourly_groups": row_count},
        )
        save_metrics(metrics, "b2_spark_aqe.json")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_b2()