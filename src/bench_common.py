from datetime import datetime, timezone
from pathlib import Path
import json
from pyspark.sql import SparkSession

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = BASE_DIR / "data" / "prepared" / "metropt3_5min_base.parquet"
RESULTS_DIR = BASE_DIR / "results" / "benchmarks"
EVENTS_BASE_DIR = BASE_DIR / "results" / "spark_events"
DEADLINE_S = 180.0  # D = 3 minutes

SENSOR_COLS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]


def build_spark(
    app_name: str,
    benchmark_slug: str,
    nodes_p: int,
    shuffle_partitions: int,
    aqe_enabled: bool,
) -> SparkSession:
    event_dir = EVENTS_BASE_DIR / benchmark_slug
    event_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    return (
        SparkSession.builder.appName(app_name)
        .master(f"local[{nodes_p}]")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_dir.as_uri())
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true" if aqe_enabled else "false")
        .getOrCreate()
    )


def build_summary_metrics(
    code: str,
    nom: str,
    type_exec: str,
    nodes_p: int,
    partitions_p: int,
    aqe_enabled: bool,
    duration_s: float,
    baseline_b0_s: float | None = None,
    extra: dict | None = None,
) -> dict:
    speedup = (baseline_b0_s / duration_s) if (baseline_b0_s and duration_s > 0) else 1.0
    efficiency = (speedup / nodes_p) if nodes_p > 0 else 0.0
    cost_node_s = nodes_p * duration_s
    cost_node_min = cost_node_s / 60.0

    payload = {
        "code": code,
        "nom": nom,
        "type": type_exec,
        "nodes_p": nodes_p,
        "partitions_P": partitions_p,
        "aqe": aqe_enabled,
        "duration_s": round(duration_s, 4),
        "speedup_S": round(speedup, 4),
        "efficiency_E": round(efficiency * 100.0, 2),  # en %
        "cost_node_s": round(cost_node_s, 4),
        "cost_node_min": round(cost_node_min, 4),
        "deadline_s": DEADLINE_S,
        "respect_delai": duration_s <= DEADLINE_S,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload["details"] = extra
    return payload


def save_metrics(payload: dict, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / filename
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Métriques enregistrées : {out_file}")
    return out_file