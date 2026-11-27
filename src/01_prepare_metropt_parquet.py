import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from pyspark.sql import SparkSession, functions as F, types as T


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_spark(app_name: str, event_log_dir: Path) -> SparkSession:
    event_log_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_log_dir.as_uri())
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main() -> None:
    base_dir = Path("C:/Users/HP/Desktop/bigdata/projet/SparkCode")
    raw_csv = base_dir / "data/MetroPT3(AirCompressor).csv"
    clean_parquet = base_dir / "data/prepared/metropt3_clean_base.parquet"
    out_5m_parquet = base_dir / "data/prepared/metropt3_5min_base.parquet"
    audit_path = base_dir / "results/metropt3_audit_spark.json"
    event_log_dir = base_dir / "results/spark_events"

    spark = build_spark("metropt3-prepare-5min-parquet", event_log_dir)
    spark.sparkContext.setLogLevel("WARN")

    try:
        t0 = perf_counter()
        file_hash = sha256_file(raw_csv)
        raw_size_bytes = raw_csv.stat().st_size

        df_raw = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(str(raw_csv))
        )

        ts_col = "timestamp" if "timestamp" in df_raw.columns else df_raw.columns[0]

        df_typed = (
            df_raw.withColumn(
                "event_ts",
                F.coalesce(
                    F.to_timestamp(F.col(ts_col)),
                    F.to_timestamp(F.col(ts_col), "yyyy-MM-dd HH:mm:ss"),
                ),
            )
            .withColumn("equipment_id", F.lit("metropt3_unit_01"))
            .withColumn("is_valid_ts", F.col("event_ts").isNotNull())
        )

        # Audit sur les lignes typées
        total_rows = df_typed.count()
        distinct_rows = df_typed.dropDuplicates().count()
        bounds = (
            df_typed.filter(F.col("is_valid_ts"))
            .agg(
                F.min("event_ts").alias("min_ts"),
                F.max("event_ts").alias("max_ts"),
                F.count("*").alias("valid_ts_rows"),
            )
            .collect()[0]
        )

        df_clean = df_typed.filter(F.col("is_valid_ts")).drop("is_valid_ts", ts_col)

        # Détection explicite des colonnes numériques capteurs (hors event_ts et equipment_id)
        numeric_types = {"int", "bigint", "smallint", "tinyint", "double", "float"}
        sensor_cols = [
            c for c, dt in df_clean.dtypes
            if c not in {"event_ts", "equipment_id"} and dt in numeric_types
        ]

        # Cast explicite en DoubleType pour éviter tout écart de type flottant
        for c in sensor_cols:
            df_clean = df_clean.withColumn(c, F.col(c).cast(T.DoubleType()))

        # Optionnel : conserver la base propre ligne à ligne
        clean_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_clean.write.mode("overwrite").parquet(str(clean_parquet))

        # Agrégation par paquets de 5 minutes : effectif, moyenne et écart-type par capteur
        agg_exprs = [F.count("*").alias("n_obs")]
        for c in sensor_cols:
            agg_exprs.append(F.avg(F.col(c)).alias(f"{c}_mean_5m"))
            agg_exprs.append(F.stddev_pop(F.col(c)).alias(f"{c}_std_5m"))

        df_5m = (
            df_clean.groupBy(
                F.window(F.col("event_ts"), "5 minutes").alias("win_5m"),
                F.col("equipment_id"),
            )
            .agg(*agg_exprs)
            .withColumn("win_start", F.col("win_5m.start"))
            .withColumn("win_end", F.col("win_5m.end"))
            .drop("win_5m")
            .orderBy("win_start")
        )

        out_5m_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_5m.write.mode("overwrite").parquet(str(out_5m_parquet))

        prep_duration_s = perf_counter() - t0
        prepared_5m_bytes = sum(p.stat().st_size for p in out_5m_parquet.rglob("*") if p.is_file())
        rows_5m = df_5m.count()

        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_payload = {
            "dataset": "MetroPT-3 (UCI)",
            "raw_file": str(raw_csv),
            "sha256": file_hash,
            "source_bytes": raw_size_bytes,
            "prepared_5m_parquet": str(out_5m_parquet),
            "prepared_5m_bytes": prepared_5m_bytes,
            "prep_duration_s": round(prep_duration_s, 4),
            "total_raw_rows": total_rows,
            "valid_ts_rows": int(bounds["valid_ts_rows"]),
            "invalid_ts_rows": total_rows - int(bounds["valid_ts_rows"]),
            "exact_duplicate_rows": total_rows - distinct_rows,
            "windows_5m_rows": rows_5m,
            "std_convention": "stddev_pop",
            "sensor_columns": sensor_cols,
            "min_timestamp": bounds["min_ts"].isoformat() if bounds["min_ts"] else None,
            "max_timestamp": bounds["max_ts"].isoformat() if bounds["max_ts"] else None,
            "columns_5m": df_5m.columns,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
        print(json.dumps(audit_payload, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()