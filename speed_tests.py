"""Benchmark King-function normalization implementations for speed and accuracy."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any

from implementations import (
    compute_normalization_gauss_legendre,
    compute_normalization_mpmath,
    compute_normalization_qags,
    compute_normalization_vectorized,
)
import numpy as np
import pandas as pd
import pyperf
from tqdm import tqdm

BenchmarkFunc = Callable[..., float]


def time_function(
    func: BenchmarkFunc,
    *args: Any,
    warmups: int = 7,
    samples: int = 21,
    min_sample_window_s: float = 0.05,
    **kwargs: Any,
) -> dict[str, float]:
    """Measure function execution time using a ``pyperf`` clock.

    Parameters
    ----------
    func : BenchmarkFunc
        Function to benchmark.
    *args : Any
        Positional arguments forwarded to ``func``.
    warmups : int, optional
        Number of warm-up invocations before measurement, by default 7.
    samples : int, optional
        Number of measured samples, by default 21.
    min_sample_window_s : float, optional
        Minimum duration for each sample window in seconds, by default 0.05.
    **kwargs : Any
        Keyword arguments forwarded to ``func``.

    Returns
    -------
    dict[str, float]
        Timing statistics in microseconds.
    """
    for _ in range(warmups):
        func(*args, **kwargs)

    iters = 1
    while True:
        start = pyperf.perf_counter()
        for _ in range(iters):
            func(*args, **kwargs)
        elapsed = pyperf.perf_counter() - start
        if elapsed >= min_sample_window_s or iters >= 100_000:
            break
        iters *= 2

    sample_times_us: list[float] = []
    for _ in range(samples):
        start = pyperf.perf_counter()
        for _ in range(iters):
            func(*args, **kwargs)
        elapsed = pyperf.perf_counter() - start
        sample_times_us.append((elapsed / iters) * 1e6)

    arr = np.asarray(sample_times_us, dtype=float)
    return {
        "Time_us": float(np.mean(arr)),
        "Time_Mean_us": float(np.mean(arr)),
        "Time_Min_us": float(np.min(arr)),
        "Time_Q1_us": float(np.percentile(arr, 25)),
        "Time_Median_us": float(np.median(arr)),
        "Time_Q3_us": float(np.percentile(arr, 75)),
        "Time_Max_us": float(np.max(arr)),
    }


def benchmark() -> None:
    """Run the full benchmark grid and save CSV plus summary table."""
    alphas = np.logspace(-6, 0, 16)
    betas = np.linspace(1, 10, 16)

    precisions: dict[str, dict[str, float | int]] = {
        "64-bit": {
            "rel_tol": float(np.finfo(np.float64).eps),
            "n_terms": 14,
        },
        "32-bit": {
            "rel_tol": float(np.finfo(np.float32).eps),
            "n_terms": 9,
        },
    }

    configs: list[tuple[str, BenchmarkFunc, str]] = [
        ("Series", compute_normalization_vectorized, "n_terms"),
        ("Gauss-Legendre", compute_normalization_gauss_legendre, "rel_tol"),
        ("QAGS", compute_normalization_qags, "rel_tol"),
    ]

    print(f"Running benchmark grid: 16x16 ({16 * 16} parameter combos)...")
    results: list[dict[str, float | str]] = []

    for alpha, beta in tqdm(list(product(alphas, betas)), desc="Benchmarking"):
        truth = compute_normalization_mpmath(float(alpha), float(beta))

        for prec_label, params in precisions.items():
            for label, func, param_name in configs:
                kwargs = {param_name: params[param_name]}

                res = func(float(alpha), float(beta), **kwargs)
                rel_error = abs(res - truth) / abs(truth) if truth != 0 else 0.0

                timing_stats = time_function(func, float(alpha), float(beta), **kwargs)
                results.append(
                    {
                        "Precision": prec_label,
                        "Implementation": label,
                        "Alpha": float(alpha),
                        "Beta": float(beta),
                        "Rel_Error": rel_error,
                        **timing_stats,
                    }
                )

    df = pd.DataFrame(results)
    csv_filename = "benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved raw data to: {csv_filename}\n")

    summary = (
        df.groupby(["Precision", "Implementation"])
        .agg(
            Max_Rel_Error=("Rel_Error", "max"),
            Med_Rel_Error=("Rel_Error", "median"),
            Avg_Time_us=("Time_Mean_us", "mean"),
            Med_Time_us=("Time_Median_us", "median"),
            Max_Time_us=("Time_Max_us", "max"),
            Min_Time_us=("Time_Min_us", "min"),
        )
        .reset_index()
    )
    summary = summary.sort_values(by=["Precision", "Avg_Time_us"], ascending=[False, True])

    print("=== Benchmark Summary ===")
    print(summary.to_string(index=False, float_format="{:.3e}".format))


if __name__ == "__main__":
    benchmark()
