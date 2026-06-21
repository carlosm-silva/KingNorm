import csv
import inspect
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import unittest

# Add the repository root so the tests import the local implementation module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import implementations

if TYPE_CHECKING:
    from collections.abc import Callable

REFERENCE_REL_TOL = 1e-11


class TestKingNorm(unittest.TestCase):
    r"""Validate implementations of $\mathcal{N}(\alpha,\beta)$."""

    def _evaluate_for_reference(self, func: "Callable[[float, float], float]", alpha: float, beta: float) -> float:
        r"""Evaluate an implementation of $\mathcal{N}(\alpha,\beta)$ for reference-value tests."""
        if func in (implementations.compute_normalization_mpmath, implementations.compute_reference_integral):
            return float(func(alpha, beta))
        return float(func(alpha, beta, rel_tol=REFERENCE_REL_TOL))

    def test_series_large_lambda_beta_one(self) -> None:
        r"""Check the stable $n=0$ path for large $\lambda$ at $\beta=1$."""
        alpha = 1e-6
        beta = 1.0

        expected_n = implementations.compute_normalization_mpmath(alpha, beta, precision=80, n_terms=80)
        result = implementations.compute_normalization_series(alpha, beta, n_terms=14)
        rel_error = abs(float(result) - expected_n) / abs(expected_n)

        self.assertLessEqual(rel_error, 1e-12)
        self.assertEqual(result.n_terms_used, 14)

    def test_adaptive_series_reports_convergence_metadata(self) -> None:
        r"""Check adaptive $S_N$ metadata and accuracy against the high-precision reference."""
        alpha = 0.2
        beta = 3.0

        expected_n = implementations.compute_normalization_mpmath(alpha, beta, precision=80, n_terms=80)
        result = implementations.compute_normalization_series(alpha, beta, rel_tol=1e-6)
        rel_error = abs(float(result) - expected_n) / abs(expected_n)

        self.assertTrue(result.converged)
        self.assertLessEqual(result.n_terms_used, 30)
        self.assertLessEqual(rel_error, 1e-6)

    def test_quadrature_methods_report_convergence_metadata(self) -> None:
        r"""Check quadrature baselines for $\mathcal{N}(\alpha,\beta)$ expose convergence metadata."""
        alpha = 0.4
        beta = 2.5
        expected_n = implementations.compute_normalization_mpmath(alpha, beta, precision=80, n_terms=80)

        for func in (implementations.compute_normalization_gauss_legendre, implementations.compute_normalization_qags):
            with self.subTest(func=func.__name__):
                result = func(alpha, beta, rel_tol=1e-6)
                rel_error = abs(float(result) - expected_n) / abs(expected_n)

                self.assertTrue(result.converged)
                self.assertGreater(result.work_done, 0)
                self.assertLessEqual(rel_error, 1e-6)

    def test_mathematica_values_relative_error(self) -> None:
        r"""Test every implementation against Mathematica values for $\mathcal{N}(\alpha,\beta)$."""
        # Collect all evaluators of the normalization constant exposed by the module.
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
                        actual_n = self._evaluate_for_reference(func, alpha, beta)
                        rel_error = abs(actual_n - expected_n) / abs(expected_n)
                        self.assertLessEqual(
                            rel_error,
                            1e-10,
                            f"{func.__name__} failed for alpha={alpha}, beta={beta}. "
                            f"Expected: {expected_n}, Actual: {actual_n}, epsilon_rel: {rel_error}",
                        )


if __name__ == "__main__":
    unittest.main()
