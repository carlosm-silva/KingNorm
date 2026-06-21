r"""Benchmark evaluators for $\mathcal{N}(\alpha,\beta)$ for speed and accuracy."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from itertools import product
import json
from pathlib import Path
import platform
from typing import Any

import mpmath as mp
import numpy as np
import pandas as pd
import pyperf
import scipy
from tqdm import tqdm

from implementations import (
    compute_normalization_gauss_legendre,
    compute_normalization_qags,
    compute_normalization_series,
    compute_reference_integral,
)

BenchmarkFunc = Callable[..., float]
TARGET_REL_TOL = 1e-6
REFERENCE_PRECISION = 80
ALPHA_GRID = np.logspace(-6, 0, 16)
BETA_GRID = np.linspace(1, 10, 16)
CSV_PATH = Path("benchmark_results.csv")
METADATA_PATH = Path("benchmark_metadata.json")


def read_cpu_model() -> str:
    """Read a human-readable CPU model name."""
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or platform.machine()


def write_metadata() -> None:
    """Write benchmark provenance to a sidecar JSON file."""
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "description": r"Time-to-target benchmark for the King's-function normalization \mathcal{N}(\alpha,\beta) over the IceCube parameter grid.",
        "target_rel_tol": TARGET_REL_TOL,
        "reference": {
            "method": r"mpmath direct quadrature of the defining integral for \mathcal{N}(\alpha,\beta)",
            "precision_decimal_digits": REFERENCE_PRECISION,
        },
        "work_units": {
            "Series": "terms summed; the stopping-test candidate term is evaluated but not included",
            "Gauss-Legendre": "integrand evaluations",
            "QAGS": "integrand evaluations reported by QUADPACK neval",
        },
        "grid": {
            "scope": "IceCube-relevant parameter range",
            "alpha": {
                "kind": "logspace",
                "min": float(ALPHA_GRID[0]),
                "max": float(ALPHA_GRID[-1]),
                "count": len(ALPHA_GRID),
            },
            "beta": {
                "kind": "linspace",
                "min": float(BETA_GRID[0]),
                "max": float(BETA_GRID[-1]),
                "count": len(BETA_GRID),
            },
        },
        "libraries": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mpmath": mp.__version__,
            "pandas": pd.__version__,
            "pyperf": getattr(pyperf, "__version__", "unknown"),
        },
        "machine": {
            "cpu_model": read_cpu_model(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def result_work_done(result: Any) -> int:
    """Extract the implementation-specific work count from a result object."""
    return int(result.work_done)


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
        "Time_Mean_us": float(np.mean(arr)),
        "Time_Min_us": float(np.min(arr)),
        "Time_Q1_us": float(np.percentile(arr, 25)),
        "Time_Median_us": float(np.median(arr)),
        "Time_Q3_us": float(np.percentile(arr, 75)),
        "Time_Max_us": float(np.max(arr)),
    }


def benchmark() -> None:
    r"""Run the full $(\alpha,\beta)$ benchmark grid and save CSV plus summary table."""
    configs: list[tuple[str, BenchmarkFunc, str]] = [
        ("Series", compute_normalization_series, "terms"),
        ("Gauss-Legendre", compute_normalization_gauss_legendre, "evaluations"),
        ("QAGS", compute_normalization_qags, "evaluations"),
    ]

    write_metadata()

    parameter_grid = list(product(ALPHA_GRID, BETA_GRID))
    print(f"Running benchmark grid: 16x16 ({len(parameter_grid)} IceCube-scope parameter combos)...")
    print(f"Target relative tolerance: {TARGET_REL_TOL:.1e}")
    print(f"Writing provenance to: {METADATA_PATH}")
    print("Work units: Series terms summed; quadrature integrand evaluations.")
    results: list[dict[str, bool | float | int | str]] = []

    for alpha, beta in tqdm(parameter_grid, desc="Benchmarking"):
        alpha_float = float(alpha)
        beta_float = float(beta)
        truth = compute_reference_integral(alpha_float, beta_float, precision=REFERENCE_PRECISION)

        for label, func, work_unit in configs:
            result = func(alpha_float, beta_float, rel_tol=TARGET_REL_TOL)
            rel_error = abs(float(result) - truth) / abs(truth) if truth != 0 else 0.0

            timing_stats = time_function(func, alpha_float, beta_float, rel_tol=TARGET_REL_TOL)
            results.append(
                {
                    "Implementation": label,
                    "Alpha": alpha_float,
                    "Beta": beta_float,
                    "Target_Rel_Tol": TARGET_REL_TOL,
                    "Reference": truth,
                    "Estimate": float(result),
                    "Rel_Error": rel_error,
                    "Converged": result.converged,
                    "Work_Done": result_work_done(result),
                    "Work_Unit": work_unit,
                    **timing_stats,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved raw data to: {CSV_PATH}\n")

    summary_all = (
        df.groupby(["Implementation", "Work_Unit"])
        .agg(
            Max_Rel_Error=("Rel_Error", "max"),
            Med_Rel_Error=("Rel_Error", "median"),
            Converged_Rate=("Converged", "mean"),
            Med_Work_Done=("Work_Done", "median"),
        )
        .reset_index()
    )
    summary_converged = (
        df[df["Converged"]]
        .groupby(["Implementation", "Work_Unit"])
        .agg(
            Max_Rel_Error_Converged=("Rel_Error", "max"),
            Med_Work_Done_Converged=("Work_Done", "median"),
            Med_Time_us_Converged=("Time_Median_us", "median"),
            Min_Time_us_Converged=("Time_Min_us", "min"),
            Mean_Time_us_Converged=("Time_Mean_us", "mean"),
            Max_Time_us_Converged=("Time_Max_us", "max"),
        )
        .reset_index()
    )
    summary = summary_all.merge(summary_converged, on=["Implementation", "Work_Unit"], how="left")
    summary = summary[
        [
            "Implementation",
            "Work_Unit",
            "Converged_Rate",
            "Max_Rel_Error",
            "Max_Rel_Error_Converged",
            "Med_Rel_Error",
            "Med_Work_Done",
            "Med_Work_Done_Converged",
            "Med_Time_us_Converged",
            "Min_Time_us_Converged",
            "Mean_Time_us_Converged",
            "Max_Time_us_Converged",
        ]
    ]
    summary = summary.sort_values(by="Med_Time_us_Converged", ascending=True)

    print("=== Benchmark Summary ===")
    print(summary.to_string(index=False, float_format="{:.3e}".format))


if __name__ == "__main__":
    benchmark()
