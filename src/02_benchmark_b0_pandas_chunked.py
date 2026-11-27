import time
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from bench_common import PARQUET_PATH, build_summary_metrics, save_metrics


def _weighted_sum_and_count(series_mean: pd.Series, series_n: pd.Series) -> tuple[float, int]:
    m = pd.to_numeric(series_mean, errors="coerce").to_numpy(dtype=np.float64)
    n = pd.to_numeric(series_n, errors="coerce").to_numpy(dtype=np.float64)
    mask = ~np.isnan(m) & ~np.isnan(n) & (n > 0)
    if not np.any(mask):
        return 0.0, 0
    return float(np.sum(m[mask] * n[mask])), int(np.sum(n[mask]))


def run_b0(chunk_rows: int = 250_000) -> float:
    cols = [
        "equipment_id",
        "win_start",
        "n_obs",
        "TP2_mean_5m",
        "TP2_std_5m",
        "TP3_mean_5m",
        "Motor_current_mean_5m",
    ]
    dataset = ds.dataset(str(PARQUET_PATH), format="parquet")

    t0 = time.perf_counter()
    total_windows = 0
    total_obs = 0

    wsum_tp2, wcnt_tp2 = 0.0, 0
    max_tp3_mean = -np.inf
    wsum_motor, wcnt_motor = 0.0, 0

    for batch in dataset.to_batches(columns=cols, batch_size=chunk_rows):
        pdf = batch.to_pandas()
        total_windows += len(pdf)

        obs_vals = pd.to_numeric(pdf["n_obs"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        total_obs += int(np.sum(obs_vals))

        s2, c2 = _weighted_sum_and_count(pdf["TP2_mean_5m"], pdf["n_obs"])
        wsum_tp2 += s2
        wcnt_tp2 += c2

        tp3_vals = pd.to_numeric(pdf["TP3_mean_5m"], errors="coerce").to_numpy(dtype=np.float64)
        tp3_valid = tp3_vals[~np.isnan(tp3_vals)]
        if tp3_valid.size > 0:
            max_tp3_mean = max(max_tp3_mean, float(np.max(tp3_valid)))

        sm, cm = _weighted_sum_and_count(pdf["Motor_current_mean_5m"], pdf["n_obs"])
        wsum_motor += sm
        wcnt_motor += cm

    elapsed = time.perf_counter() - t0

    metrics = build_summary_metrics(
        code="B0",
        nom="Python NumPy/pandas Chunked (5min Parquet)",
        type_exec="Local Python",
        nodes_p=1,
        partitions_p=1,
        aqe_enabled=False,
        duration_s=elapsed,
        baseline_b0_s=elapsed,
        extra={
            "chunk_rows": chunk_rows,
            "total_windows_5m": total_windows,
            "total_obs": total_obs,
            "weighted_avg_tp2": round(wsum_tp2 / wcnt_tp2, 4) if wcnt_tp2 else None,
            "max_tp3_mean_5m": None if max_tp3_mean == -np.inf else round(max_tp3_mean, 4),
            "weighted_avg_motor_current": round(wsum_motor / wcnt_motor, 4) if wcnt_motor else None,
        },
    )
    save_metrics(metrics, "b0_pandas_chunked.json")
    return elapsed


if __name__ == "__main__":
    run_b0()