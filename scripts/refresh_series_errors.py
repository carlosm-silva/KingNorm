r"""Refresh only the series $\epsilon_{rel}$ values in ``benchmark_results.csv``."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "benchmark_results.csv"
REFERENCE_PRECISION = 80
REFERENCE_TERMS = 80

def main() -> int:
    r"""Recompute series $\epsilon_{rel}$ values without rerunning timing benchmarks."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from implementations import compute_normalization_mpmath, compute_normalization_series

    df = pd.read_csv(CSV_PATH)
    series_mask = df["Implementation"] == "Series"
    series_df = df[series_mask]

    truths: dict[tuple[float, float], float] = {}
    parameter_pairs = list(series_df[["Alpha", "Beta"]].drop_duplicates().itertuples(index=False, name=None))
    for alpha, beta in tqdm(parameter_pairs, desc="Computing references"):
        truths[(float(alpha), float(beta))] = compute_normalization_mpmath(
            float(alpha),
            float(beta),
            precision=REFERENCE_PRECISION,
            n_terms=REFERENCE_TERMS,
        )

    old_errors = series_df["Rel_Error"].copy()
    for index, row in tqdm(series_df.iterrows(), total=len(series_df), desc="Refreshing series errors"):
        alpha = float(row["Alpha"])
        beta = float(row["Beta"])
        n_terms = int(row["Work_Done"])
        truth = truths[(alpha, beta)]
        estimate = compute_normalization_series(alpha, beta, n_terms=n_terms)
        df.loc[index, "Rel_Error"] = abs(float(estimate) - truth) / abs(truth) if truth != 0 else 0.0

    df.to_csv(CSV_PATH, index=False)

    new_errors = df.loc[series_mask, "Rel_Error"]
    print("Updated series epsilon_rel values only.")
    print(f"  rows updated: {len(series_df)}")
    print(f"  old max: {old_errors.max():.3e}")
    print(f"  new max: {new_errors.max():.3e}")
    print(f"  old median: {old_errors.median():.3e}")
    print(f"  new median: {new_errors.median():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
