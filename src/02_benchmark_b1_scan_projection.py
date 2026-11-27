import time
from pyspark.sql import functions as F
from bench_common import PARQUET_PATH, build_spark, save_metrics


def run_b1() -> None:
    spark = build_spark("metropt3-b1-scan-projection", "b1_scan_projection")
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.read.parquet(str(PARQUET_PATH))
        active_cond = (F.col("TP2_mean_5m") > 0) & (F.col("COMP_mean_5m") > 0.5)

        # Variante A : lecture large sans projection sélective
        t0 = time.perf_counter()
        wide_count = df.select("*").filter(active_cond).count()
        dt_wide = time.perf_counter() - t0

        # Variante B : projection ciblée + pushdown de filtres + moyenne pondérée par n_obs
        projected_cols = [
            "win_start",
            "equipment_id",
            "n_obs",
            "TP2_mean_5m",
            "TP3_mean_5m",
            "Motor_current_mean_5m",
            "COMP_mean_5m",
        ]
        t1 = time.perf_counter()
        pruned_df = (
            df.select(*projected_cols)
            .filter(active_cond)
            .agg(
                F.count("*").alias("windows_5m"),
                F.sum("n_obs").alias("total_obs"),
                (
                    F.sum(F.col("TP2_mean_5m") * F.col("n_obs")) / F.sum("n_obs")
                ).alias("weighted_avg_tp2"),
                (
                    F.sum(F.col("Motor_current_mean_5m") * F.col("n_obs")) / F.sum("n_obs")
                ).alias("weighted_avg_motor_current"),
            )
        )
        summary = pruned_df.collect()[0].asDict()
        dt_pruned = time.perf_counter() - t1

        save_metrics(
            {
                "benchmark": "B1",
                "name": "scan_projection_pushdown_5min",
                "parquet_path": str(PARQUET_PATH),
                "projected_columns": projected_cols,
                "wide_filtered_windows": wide_count,
                "duration_wide_s": round(dt_wide, 4),
                "pruned_filtered_windows": summary["windows_5m"],
                "pruned_total_obs": summary["total_obs"],
                "weighted_avg_tp2_active": round(float(summary["weighted_avg_tp2"] or 0.0), 4),
                "weighted_avg_motor_current_active": round(
                    float(summary["weighted_avg_motor_current"] or 0.0), 4
                ),
                "duration_pruned_s": round(dt_pruned, 4),
                "speedup_factor": round(dt_wide / dt_pruned, 4) if dt_pruned > 0 else None,
            },
            "b1_scan_projection.json",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    run_b1()