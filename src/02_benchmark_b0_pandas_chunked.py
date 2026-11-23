from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from bench_common import PARQUET_PATH, build_summary_metrics, save_metrics


def run_b0(chunk_rows: int = 250_000) -> float:
    cols = ["equipment_id", "event_ts", "TP2", "TP3", "Oil_temperature", "Motor_current"]
    dataset = ds.dataset(str(PARQUET_PATH), format="parquet")

    t0 = time.perf_counter()
    total_rows = 0
    sum_tp2 = 0.0
    count_tp2 = 0
    max_tp3 = -np.inf
    sum_motor = 0.0
    count_motor = 0

    for batch in dataset.to_batches(columns=cols, batch_size=chunk_rows):
        pdf = batch.to_pandas()
        total_rows += len(pdf)

        tp2_vals = pd.to_numeric(pdf["TP2"], errors="coerce").to_numpy(dtype=np.float64)
        tp2_valid = tp2_vals[~np.isnan(tp2_vals)]
        sum_tp2 += float(np.sum(tp2_valid))
        count_tp2 += int(tp2_valid.size)

        tp3_vals = pd.to_numeric(pdf["TP3"], errors="coerce").to_numpy(dtype=np.float64)
        tp3_valid = tp3_vals[~np.isnan(tp3_vals)]
        if tp3_valid.size > 0:
            max_tp3 = max(max_tp3, float(np.max(tp3_valid)))

        motor_vals = pd.to_numeric(pdf["Motor_current"], errors="coerce").to_numpy(dtype=np.float64)
        motor_valid = motor_vals[~np.isnan(motor_vals)]
        sum_motor += float(np.sum(motor_valid))
        count_motor += int(motor_valid.size)

    elapsed = time.perf_counter() - t0

    metrics = build_summary_metrics(
        code="B0",
        nom="Python NumPy/pandas Chunked",
        type_exec="Local Python",
        nodes_p=1,
        partitions_p=1,
        aqe_enabled=False,
        duration_s=elapsed,
        baseline_b0_s=elapsed,
        extra={
            "chunk_rows": chunk_rows,
            "total_rows": total_rows,
            "avg_tp2": round(sum_tp2 / count_tp2, 4) if count_tp2 else None,
            "max_tp3": None if max_tp3 == -np.inf else round(max_tp3, 4),
            "avg_motor_current": round(sum_motor / count_motor, 4) if count_motor else None,
        },
    )
    save_metrics(metrics, "b0_pandas_chunked.json")
    return elapsed


if __name__ == "__main__":
    run_b0()