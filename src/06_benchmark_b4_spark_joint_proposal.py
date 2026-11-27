import json
import time
import pandas as pd
from pyspark.sql import functions as F
from sklearn.tree import DecisionTreeRegressor
from bench_common import (
    PARQUET_PATH,
    RESULTS_DIR,
    build_spark,
    build_summary_metrics,
    save_metrics,
)

DEADLINE_S = 180.0


def _get_b0_duration() -> float | None:
    b0_file = RESULTS_DIR / "b0_pandas_chunked.json"
    if b0_file.exists():
        return float(json.loads(b0_file.read_text(encoding="utf-8"))["duration_s"])
    return None


def _dataset_size_mb() -> float:
    total_bytes = sum(p.stat().st_size for p in PARQUET_PATH.rglob("*") if p.is_file())
    return total_bytes / (1024 * 1024)


def select_b4_dt_config(safety_ratio: float = 0.90) -> tuple[int, int, dict]:
    target_mb = _dataset_size_mb()
    rows = []

    # Collecte les exécutions Spark historiques déjà mesurées (hors B0 et hors ancien B4)
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        if jf.name in {"b0_pandas_chunked.json", "b4_spark_joint_proposal.json", "metropt3_audit_spark.json"}:
            continue
        d = json.loads(jf.read_text(encoding="utf-8"))
        if all(k in d for k in ("duration_s", "nodes_p", "partitions_P")):
            rows.append(
                {
                    "data_mb": float(d.get("data_mb", target_mb)),
                    "nodes_p": int(d["nodes_p"]),
                    "partitions_P": int(d["partitions_P"]),
                    "duration_s": float(d["duration_s"]),
                }
            )

    if len(rows) < 2:
        return 4, 128, {"fallback": True, "reason": "insufficient_history", "samples": len(rows)}

    train_df = pd.DataFrame(rows)
    features = ["data_mb", "nodes_p", "partitions_P"]
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=1, random_state=42)
    model.fit(train_df[features], train_df["duration_s"])

    candidates = [(1, 16), (2, 32), (4, 64), (4, 128), (8, 64), (8, 128)]
    cand_df = pd.DataFrame(
        [{"data_mb": target_mb, "nodes_p": p, "partitions_P": part} for p, part in candidates]
    )
    cand_df["pred_T_s"] = model.predict(cand_df[features])
    cand_df["pred_cost_min"] = (cand_df["nodes_p"] * cand_df["pred_T_s"]) / 60.0

    safe_limit = DEADLINE_S * safety_ratio
    feasible = cand_df[cand_df["pred_T_s"] <= safe_limit]
    if feasible.empty:
        feasible = cand_df[cand_df["pred_T_s"] <= DEADLINE_S]
    pool = feasible if not feasible.empty else cand_df.sort_values("pred_T_s")
    best = pool.sort_values(["pred_cost_min", "pred_T_s"]).iloc[0]

    info = {
        "selected_nodes_p": int(best["nodes_p"]),
        "selected_partitions_P": int(best["partitions_P"]),
        "pred_T_s": round(float(best["pred_T_s"]), 4),
        "pred_cost_min": round(float(best["pred_cost_min"]), 4),
        "safe_deadline_s": safe_limit,
        "train_samples": len(train_df),
    }
    return int(best["nodes_p"]), int(best["partitions_P"]), info


def run_b4() -> None:
    nodes_p, partitions_p, dt_info = select_b4_dt_config(safety_ratio=0.90)
    aqe_enabled = True
    print(f"[B4-DT MetroPT-3] Sélection : p={nodes_p}, P={partitions_p} | détails={dt_info}")

    spark = build_spark(
        app_name="metropt3-b4-choix-conjoint-dt",
        benchmark_slug="b4_choix_conjoint",
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
            code="B4",
            nom="PySpark Choix Conjoint Proposé (Arbre de Décision, 5min)",
            type_exec="PySpark Joint Proposal (DT)",
            nodes_p=nodes_p,
            partitions_p=partitions_p,
            aqe_enabled=aqe_enabled,
            duration_s=elapsed,
            baseline_b0_s=b0_s,
            extra={"hourly_groups": row_count, "decision_tree": dt_info},
        )
        save_metrics(metrics, "b4_spark_joint_proposal.json")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_b4()