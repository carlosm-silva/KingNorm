import csv
import inspect
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import unittest

# Ensure the parent directory is in the path so we can import implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import implementations

if TYPE_CHECKING:
    from collections.abc import Callable


class TestKingNorm(unittest.TestCase):
    """Validate King-function normalization implementations."""

    def test_mathematica_values_relative_error(self) -> None:
        """Test every implementation against Mathematica reference values."""
        # Find all implementation functions dynamically
        impl_funcs: list[Callable[[float, float], float]] = []
        for name, obj in inspect.getmembers(implementations, inspect.isfunction):
            if name.startswith("compute_"):
                impl_funcs.append(obj)

        csv_path = Path(__file__).with_name("mathematica_vals.csv")

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                alpha = float(row["alpha"])
                beta = float(row["beta"])
                expected_n = float(row["N"])

                for func in impl_funcs:
                    with self.subTest(func=func.__name__, alpha=alpha, beta=beta):
                        actual_n = float(func(alpha, beta))
                        rel_error = abs(actual_n - expected_n) / abs(expected_n)
                        self.assertLessEqual(
                            rel_error,
                            1e-10,
                            f"{func.__name__} failed for alpha={alpha}, beta={beta}. "
                            f"Expected: {expected_n}, Actual: {actual_n}, Rel Error: {rel_error}",
                        )


if __name__ == "__main__":
    unittest.main()
