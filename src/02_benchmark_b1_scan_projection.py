import time
from pyspark.sql import functions as F
from bench_common import PARQUET_PATH, build_spark, save_metrics


def run_b1() -> None:
    spark = build_spark("metropt3-b1-scan-projection", "b1_scan_projection")
    try:
        df = spark.read.parquet(str(PARQUET_PATH))

        # Variante A : lecture large sans projection sélective
        t0 = time.perf_counter()
        wide_df = df.select("*").filter((F.col("TP2") > 0) & (F.col("COMP") == 1))
        wide_count = wide_df.count()
        dt_wide = time.perf_counter() - t0

        # Variante B : projection ciblée + pushdown de filtres
        projected_cols = ["event_ts", "equipment_id", "TP2", "TP3", "Motor_current", "COMP"]
        t1 = time.perf_counter()
        pruned_df = (
            df.select(*projected_cols)
            .filter((F.col("TP2") > 0) & (F.col("COMP") == 1))
            .agg(
                F.count("*").alias("rows"),
                F.avg("TP2").alias("avg_tp2"),
                F.avg("Motor_current").alias("avg_motor_current"),
            )
        )
        summary = pruned_df.collect()[0].asDict()
        dt_pruned = time.perf_counter() - t1

        save_metrics(
            {
                "benchmark": "B1",
                "name": "scan_projection_pushdown",
                "parquet_path": str(PARQUET_PATH),
                "projected_columns": projected_cols,
                "wide_filtered_rows": wide_count,
                "duration_wide_s": round(dt_wide, 4),
                "pruned_filtered_rows": summary["rows"],
                "avg_tp2_active": round(float(summary["avg_tp2"] or 0.0), 4),
                "avg_motor_current_active": round(float(summary["avg_motor_current"] or 0.0), 4),
                "duration_pruned_s": round(dt_pruned, 4),
                "speedup_factor": round(dt_wide / dt_pruned, 4) if dt_pruned > 0 else None,
            },
            "b1_scan_projection.json",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    run_b1()