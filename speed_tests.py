import numpy as np
import pandas as pd
import pyperf
from itertools import product
from tqdm import tqdm
from implementations import (
    compute_normalization_vectorized,
    compute_normalization_mpmath,
    compute_normalization_gauss_legendre,
    compute_normalization_qags
)

def time_function(func, *args, warmups=7, samples=21, min_sample_window_s=0.05, **kwargs):
    """
    Measure execution time with a pyperf-backed clock.
    Returns summary statistics (in microseconds) over repeated samples.
    """
    # Warm multiple times to stabilize caches / branch predictors.
    for _ in range(warmups):
        func(*args, **kwargs)

    # Calibrate loop count so each sample has enough wall-clock signal.
    iters = 1
    while True:
        start = pyperf.perf_counter()
        for _ in range(iters):
            func(*args, **kwargs)
        elapsed = pyperf.perf_counter() - start
        if elapsed >= min_sample_window_s:
            break
        iters *= 2
        if iters >= 100_000:
            break

    sample_times_us = []
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

def benchmark():
    # Grid specification
    alphas = np.logspace(-6, 0, 16)
    betas = np.linspace(1, 10, 16)
    
    # Precision setups
    precisions = {
        "64-bit": {
            "rel_tol": float(np.finfo(np.float64).eps),
            "n_terms": 14,
        },
        "32-bit": {
            "rel_tol": float(np.finfo(np.float32).eps),
            "n_terms": 9,
        }
    }
    
    configs = [
        ("Series", compute_normalization_vectorized, "n_terms"),
        ("Gauss-Legendre", compute_normalization_gauss_legendre, "rel_tol"),
        ("QAGS", compute_normalization_qags, "rel_tol")
    ]
    
    print(f"Running benchmark grid: 16x16 ({16 * 16} parameter combos)...")
    results = []
    
    for alpha, beta in tqdm(list(product(alphas, betas)), desc="Benchmarking"):
        # Ground truth
        truth = compute_normalization_mpmath(alpha, beta)
        
        for prec_label, params in precisions.items():
            for label, func, param_name in configs:
                kwargs = {param_name: params[param_name]}
                
                # Accuracy
                res = func(alpha, beta, **kwargs)
                rel_error = abs(res - truth) / abs(truth) if truth != 0 else 0.0
                
                # Performance
                timing_stats = time_function(func, alpha, beta, **kwargs)
                
                results.append({
                    "Precision": prec_label,
                    "Implementation": label,
                    "Alpha": alpha,
                    "Beta": beta,
                    "Rel_Error": rel_error,
                    # Keep Time_us for compatibility with existing plotting scripts.
                    **timing_stats
                })

    # Save and output
    df = pd.DataFrame(results)
    csv_filename = "benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved raw data to: {csv_filename}\n")
    
    # Generate neat summary by grouping
    summary = df.groupby(["Precision", "Implementation"]).agg(
        Max_Rel_Error=("Rel_Error", "max"),
        Med_Rel_Error=("Rel_Error", "median"),
        Avg_Time_us=("Time_Mean_us", "mean"),
        Med_Time_us=("Time_Median_us", "median"),
        Max_Time_us=("Time_Max_us", "max"),
        Min_Time_us=("Time_Min_us", "min")
    ).reset_index()
    
    # Sort for consistent display
    summary = summary.sort_values(by=["Precision", "Avg_Time_us"], ascending=[False, True])
    
    print("=== Benchmark Summary ===")
    print(summary.to_string(index=False, float_format="{:.3e}".format))

    
if __name__ == "__main__":
    benchmark()
