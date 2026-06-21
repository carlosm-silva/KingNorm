r"""Check numerical claims quoted in ``paper/main.tex``.

The paper's main accuracy claims come from the uniform absolute and relative
truncation bounds for $R_N=\mathcal{N}(\alpha,\beta)-S_N$. This script
recomputes those values from the displayed formulas, then performs a small
implementation sanity check using the existing normalization routines.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from math import factorial, isclose, pi
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ABSOLUTE_THRESHOLD = 1e-16
FLOAT32_EPSILON = 1.1920928955078125e-7
FLOAT64_EPSILON = 2.220446049250313e-16
CLAIM_REL_TOL = 5e-3


@dataclass(frozen=True)
class NumericClaim:
    """A quoted paper value and the exact value recomputed by this script."""

    label: str
    quoted: float
    computed: float

    @property
    def relative_error(self) -> float:
        r"""Return the relative error $\epsilon_{rel}$ of the quoted rounded value."""
        return abs(self.quoted - self.computed) / abs(self.computed)

    @property
    def is_ok(self) -> bool:
        """Return whether the quoted value matches the recomputed value."""
        return isclose(self.quoted, self.computed, rel_tol=CLAIM_REL_TOL)


def absolute_remainder_bound(partial_sum_index: int) -> float:
    r"""Compute the uniform absolute bound from Eq. ``\eqref{eq:err_bound}``.

    Parameters
    ----------
    partial_sum_index : int
        Index $N$ of the partial sum $S_N$.

    Returns
    -------
    float
        Uniform upper bound on $|R_N|$.
    """
    return pi ** (2 * partial_sum_index + 4) / (2 * (partial_sum_index + 2) * factorial(2 * partial_sum_index + 3))


def relative_remainder_bound(partial_sum_index: int) -> float:
    r"""Compute the uniform relative bound from Eq. ``\eqref{eq:rel_err_bound}``.

    Parameters
    ----------
    partial_sum_index : int
        Index $N$ of the partial sum $S_N$.

    Returns
    -------
    float
        Uniform upper bound on $\epsilon_{rel}=|R_N|/\mathcal{N}(\alpha,\beta)$.
    """
    chebyshev_factor = 1.0 / (1.0 - pi**2 / 12.0)
    return chebyshev_factor * pi ** (2 * partial_sum_index + 2) / factorial(2 * partial_sum_index + 3)


def first_partial_sum_index_below(bound: Callable[[int], float], threshold: float) -> int:
    r"""Return the first partial-sum index $N$ whose bound is below ``threshold``."""
    partial_sum_index = 0
    while bound(partial_sum_index) >= threshold:
        partial_sum_index += 1
    return partial_sum_index


def paper_claims() -> list[NumericClaim]:
    """Return the rounded numerical values quoted in ``paper/main.tex``."""
    chebyshev_factor = 1.0 / (1.0 - pi**2 / 12.0)
    return [
        NumericClaim("absolute bound, N=0", 4.06, absolute_remainder_bound(0)),
        NumericClaim("absolute bound, N=2", 0.235, absolute_remainder_bound(2)),
        NumericClaim("absolute bound, N=5", 1.05e-4, absolute_remainder_bound(5)),
        NumericClaim("absolute bound, N=12", 2.73e-16, absolute_remainder_bound(12)),
        NumericClaim("absolute bound, N=13", 3.10e-18, absolute_remainder_bound(13)),
        NumericClaim("relative prefactor", 5.63, chebyshev_factor),
        NumericClaim("relative bound, N=5", 8.36e-4, relative_remainder_bound(5)),
        NumericClaim("relative bound, N=12", 4.36e-15, relative_remainder_bound(12)),
        NumericClaim("relative bound, N=8", 4.11e-8, relative_remainder_bound(8)),
        NumericClaim("relative bound, N=13", 5.30e-17, relative_remainder_bound(13)),
        NumericClaim("float32 machine epsilon", 1.19e-7, FLOAT32_EPSILON),
        NumericClaim("float64 machine epsilon", 2.22e-16, FLOAT64_EPSILON),
    ]


def check_paper_claims() -> bool:
    """Print recomputed values for the paper's quoted numerical claims."""
    print("Paper numerical claims")
    print("----------------------")
    all_ok = True
    for claim in paper_claims():
        status = "ok" if claim.is_ok else "FAIL"
        print(
            f"{status:>4}  {claim.label:<32} quoted={claim.quoted:.6e}  "
            f"computed={claim.computed:.12e}  rel_diff={claim.relative_error:.2e}"
        )
        all_ok &= claim.is_ok

    first_abs_n = first_partial_sum_index_below(absolute_remainder_bound, ABSOLUTE_THRESHOLD)
    first_float32_n = first_partial_sum_index_below(relative_remainder_bound, FLOAT32_EPSILON)
    first_float64_n = first_partial_sum_index_below(relative_remainder_bound, FLOAT64_EPSILON)

    threshold_checks = [
        ("absolute < 1e-16", first_abs_n, 13),
        ("relative < float32 eps", first_float32_n, 8),
        ("relative < float64 eps", first_float64_n, 13),
    ]

    print("\nThreshold term counts")
    print("---------------------")
    for label, computed_n, quoted_n in threshold_checks:
        status = "ok" if computed_n == quoted_n else "FAIL"
        print(f"{status:>4}  {label:<24} first_N={computed_n}  terms={computed_n + 1}  quoted_N={quoted_n}")
        all_ok &= computed_n == quoted_n

    return all_ok


def check_existing_implementation_grid() -> None:
    """Spot-check current implementations against the high-precision reference.

    This is a diagnostic finite-precision check of the current SciPy-backed
    implementation, not a check of the analytical truncation bounds.  The
    paper's guarantees control $R_N$; the implementation can still lose accuracy
    in the hypergeometric evaluation.
    """
    try:
        from implementations import compute_normalization_mpmath, compute_normalization_series
    except ModuleNotFoundError as exc:
        print(f"\nImplementation grid skipped: missing dependency while importing implementations.py ({exc}).")
        return

    parameter_grid = [
        (1e-6, 1.0),
        (1e-4, 2.2),
        (1e-2, 5.8),
        (1.0, 10.0),
    ]
    configs = [
        ("32-bit terms", 9, FLOAT32_EPSILON),
        ("64-bit terms", 14, FLOAT64_EPSILON),
    ]

    print("\nExisting implementation spot-check")
    print("----------------------------------")
    for alpha, beta in parameter_grid:
        truth = compute_normalization_mpmath(alpha, beta, precision=80, n_terms=80)
        for label, n_terms, tolerance in configs:
            actual = compute_normalization_series(alpha, beta, n_terms=n_terms)
            relative_error = abs(float(actual) - truth) / abs(truth)
            status = "ok" if relative_error <= tolerance else "warn"
            print(
                f"{status:>4}  {label:<12} alpha={alpha:.1e} beta={beta:>4.1f}  "
                f"rel_error={relative_error:.3e}  tolerance={tolerance:.3e}"
            )


def parse_args() -> bool:
    """Return whether to skip the slow finite-precision implementation spot-check."""
    parser = ArgumentParser(description="Check numerical claims quoted in paper/main.tex.")
    parser.add_argument(
        "--claims-only",
        action="store_true",
        help="Only check analytical paper claims; skip the slower implementation spot-check.",
    )
    args = parser.parse_args()
    return bool(args.claims_only)


def main() -> int:
    """Run the selected numerical checks."""
    claims_only = parse_args()
    claims_ok = check_paper_claims()
    if claims_only:
        print("\nExisting implementation spot-check skipped (--claims-only).")
    else:
        check_existing_implementation_grid()
    return 0 if claims_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
