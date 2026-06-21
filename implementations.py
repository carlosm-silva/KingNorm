r"""Numerical implementations for the King-function normalization $\mathcal{N}(\alpha,\beta)$."""

from __future__ import annotations

from typing import NamedTuple
import warnings

from mpmath import fac, mp, mpf
from mpmath import hyp2f1 as mp_hyp2f1
from mpmath import pi as mpi
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gamma, hyp2f1

DEFAULT_REL_TOL = 1e-6
DEFAULT_MAX_SERIES_TERMS = 30
GAUSS_LEGENDRE_ORDER = 16
GAUSS_LEGENDRE_MAX_NODES = 8192


class SeriesResult(NamedTuple):
    r"""Series estimate of $\mathcal{N}(\alpha,\beta)$ with convergence metadata."""

    value: float
    n_terms_used: int
    converged: bool

    def __float__(self) -> float:
        r"""Return the estimate of $\mathcal{N}(\alpha,\beta)$."""
        return self.value

    @property
    def work_done(self) -> int:
        """Return the number of terms included in the partial sum."""
        return self.n_terms_used


class QuadratureResult(NamedTuple):
    r"""Quadrature estimate of $\mathcal{N}(\alpha,\beta)$ with work metadata."""

    value: float
    work_done: int
    converged: bool

    def __float__(self) -> float:
        r"""Return the estimate of $\mathcal{N}(\alpha,\beta)$."""
        return self.value

    @property
    def nodes_used(self) -> int:
        """Return the quadrature work count for backward compatibility."""
        return self.work_done


_GAUSS_LEGENDRE_NODES, _GAUSS_LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(GAUSS_LEGENDRE_ORDER)


def _zeroth_order_term(alpha: float, beta: float) -> float:
    r"""Evaluate the $n=0$ term $a_0$ without the hypergeometric backend.

    Parameters
    ----------
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.

    Returns
    -------
    float
        The signed $n=0$ contribution to $\mathcal{N}(\alpha,\beta)$.
    """
    lambda_param = np.pi**2 / (2.0 * beta * alpha**2)
    log_factor = np.log1p(lambda_param)
    beta_offset = 1.0 - beta

    if beta_offset == 0.0:
        integral = log_factor / lambda_param
    else:
        integral = np.expm1(beta_offset * log_factor) / (beta_offset * lambda_param)

    return float((np.pi**2 / 2.0) * integral)


def _series_term(alpha: float, beta: float, n: int) -> float:
    r"""Evaluate the signed $n$-th term in the series for $\mathcal{N}(\alpha,\beta)$.

    Parameters
    ----------
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.
    n : int
        Series index.

    Returns
    -------
    float
        The signed contribution $(-1)^n a_n$.
    """
    if n == 0:
        return _zeroth_order_term(alpha, beta)

    z = -(np.pi**2) / (2.0 * beta * alpha**2)
    coefficient = ((-1.0) ** n * np.pi ** (2.0 * n + 2.0)) / (2.0 * (n + 1.0) * gamma(2.0 * n + 2.0))
    hyper_val = hyp2f1(n + 1.0, beta, n + 2.0, z)
    return float(coefficient * hyper_val)


def _integrand(x: float, alpha: float, beta: float) -> float:
    r"""Evaluate the integrand defining $\mathcal{N}(\alpha,\beta)$ at one point.

    Parameters
    ----------
    x : float
        Angular separation $x$.
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.

    Returns
    -------
    float
        Value of $\left(1+x^2/(2\beta\alpha^2)\right)^{-\beta}\sin x$.
    """
    return (1.0 + x**2 / (2.0 * beta * alpha**2)) ** (-beta) * np.sin(x)


def _gauss_legendre_interval(left: float, right: float, alpha: float, beta: float) -> float:
    """Evaluate one Gauss-Legendre panel."""
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    points = midpoint + half_width * _GAUSS_LEGENDRE_NODES
    values = _integrand(points, alpha, beta)
    return float(half_width * np.dot(_GAUSS_LEGENDRE_WEIGHTS, values))


def compute_normalization_series(
    alpha: float,
    beta: float,
    rel_tol: float = DEFAULT_REL_TOL,
    n_terms: int | None = None,
    max_terms: int = DEFAULT_MAX_SERIES_TERMS,
) -> SeriesResult:
    r"""Compute $\mathcal{N}(\alpha,\beta)$ with an adaptive alternating series.

    Notes
    -----
    The implementation evaluates the series

    $$
    \mathcal{N}(\alpha, \beta) = \sum_{n=0}^{n_{\mathrm{terms}}-1}
    \frac{(-1)^n\pi^{2n+2}}{2(n+1)(2n+1)!}
    \,{}_2F_1(n+1, \beta; n+2; -\lambda),
    $$

    where

    $$
    \lambda = \frac{\pi^2}{2\beta\alpha^2}.
    $$

    The stopping rule adds terms until the next omitted term is at most
    ``rel_tol`` times the current partial sum $S_N$. The alternating-series
    remainder $R_N=\mathcal{N}(\alpha,\beta)-S_N$ then gives a rigorous relative
    error bound, provided each term is evaluated accurately. The explicit $n=0$
    evaluation avoids the known large-negative-argument hypergeometric failure
    mode in that first term.

    Parameters
    ----------
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.
    rel_tol : float, optional
        Relative tolerance for adaptive truncation, by default ``1e-6``.
    n_terms : int | None, optional
        If supplied, evaluate exactly this many terms and report whether the next
        term satisfies the adaptive stopping rule. By default, use adaptive
        truncation.
    max_terms : int, optional
        Maximum number of terms for adaptive truncation, by default 30.

    Returns
    -------
    SeriesResult
        Estimate of $\mathcal{N}(\alpha,\beta)$, included terms, and convergence flag.
    """
    if n_terms is not None:
        total = float(sum(_series_term(alpha, beta, n) for n in range(n_terms)))
        next_term = abs(_series_term(alpha, beta, n_terms))
        converged = next_term <= rel_tol * abs(total)
        return SeriesResult(total, n_terms, converged)

    total = _series_term(alpha, beta, 0)
    terms_used = 1
    while terms_used < max_terms:
        next_term = _series_term(alpha, beta, terms_used)
        if abs(next_term) <= rel_tol * abs(total):
            return SeriesResult(total, terms_used, True)
        total += next_term
        terms_used += 1

    next_term = abs(_series_term(alpha, beta, max_terms))
    converged = next_term <= rel_tol * abs(total)
    return SeriesResult(total, max_terms, converged)


def compute_normalization_mpmath(alpha: float, beta: float, precision: int = 30, n_terms: int = 30) -> float:
    r"""Compute a high-precision series reference for $\mathcal{N}(\alpha,\beta)$ with mpmath.

    Parameters
    ----------
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.
    precision : int, optional
        Decimal precision used by mpmath, by default 30.
    n_terms : int, optional
        Number of terms in the series, by default 30.

    Returns
    -------
    float
        High-precision estimate of $\mathcal{N}(\alpha,\beta)$ cast to ``float``.
    """
    mp.dps = precision
    alpha_mp, beta_mp = mpf(alpha), mpf(beta)
    z = -(mpi**2) / (2.0 * beta_mp * alpha_mp**2)

    total = mpf(0)
    for n in range(n_terms):
        term = ((-1) ** n * mpi ** (2 * n + 2)) / (2 * (n + 1) * fac(2 * n + 1)) * mp_hyp2f1(n + 1, beta_mp, n + 2, z)
        total += term

    return float(total)


def compute_reference_integral(alpha: float, beta: float, precision: int = 80) -> float:
    r"""Compute an independent high-precision reference from Eq. $\mathcal{N}(\alpha,\beta)$.

    Parameters
    ----------
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.
    precision : int, optional
        Decimal precision used by mpmath, by default 80.

    Returns
    -------
    float
        High-precision direct quadrature estimate of $\mathcal{N}(\alpha,\beta)$ cast to ``float``.
    """
    with mp.workdps(precision):
        alpha_mp = mpf(alpha)
        beta_mp = mpf(beta)

        def integrand(x: mpf) -> mpf:
            return (1 + x**2 / (2 * beta_mp * alpha_mp**2)) ** (-beta_mp) * mp.sin(x)

        return float(mp.quad(integrand, [0, mp.pi]))


def compute_normalization_gauss_legendre(
    alpha: float,
    beta: float,
    rel_tol: float = DEFAULT_REL_TOL,
    max_nodes: int = GAUSS_LEGENDRE_MAX_NODES,
) -> QuadratureResult:
    r"""Compute $\mathcal{N}(\alpha,\beta)$ using adaptive Gauss-Legendre quadrature.

    Parameters
    ----------
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.
    rel_tol : float, optional
        Relative convergence tolerance, by default ``1e-6``.
    max_nodes : int, optional
        Maximum integrand evaluations before reporting non-convergence, by
        default 8192.

    Returns
    -------
    QuadratureResult
        Estimate of $\mathcal{N}(\alpha,\beta)$, number of nodes evaluated, and convergence flag.
    """
    parent_value = _gauss_legendre_interval(0.0, np.pi, alpha, beta)
    midpoint = 0.5 * np.pi
    left_value = _gauss_legendre_interval(0.0, midpoint, alpha, beta)
    right_value = _gauss_legendre_interval(midpoint, np.pi, alpha, beta)
    nodes_used = 3 * GAUSS_LEGENDRE_ORDER

    intervals: list[tuple[float, float, float, float]] = [
        (0.0, midpoint, left_value, abs(left_value + right_value - parent_value) / 2.0),
        (midpoint, np.pi, right_value, abs(left_value + right_value - parent_value) / 2.0),
    ]
    total_value = left_value + right_value
    total_error = abs(total_value - parent_value)

    while total_error > rel_tol * abs(total_value) and nodes_used + 2 * GAUSS_LEGENDRE_ORDER <= max_nodes:
        worst_index = max(range(len(intervals)), key=lambda index: intervals[index][3])
        left, right, interval_value, interval_error = intervals.pop(worst_index)
        midpoint = 0.5 * (left + right)
        left_child = _gauss_legendre_interval(left, midpoint, alpha, beta)
        right_child = _gauss_legendre_interval(midpoint, right, alpha, beta)
        nodes_used += 2 * GAUSS_LEGENDRE_ORDER

        refined_value = left_child + right_child
        refined_error = abs(refined_value - interval_value)
        total_value += refined_value - interval_value
        total_error += refined_error - interval_error
        child_error = 0.5 * refined_error
        intervals.extend(
            [
                (left, midpoint, left_child, child_error),
                (midpoint, right, right_child, child_error),
            ]
        )

    converged = total_error <= rel_tol * abs(total_value)
    return QuadratureResult(float(total_value), nodes_used, converged)


def compute_normalization_qags(alpha: float, beta: float, rel_tol: float = DEFAULT_REL_TOL) -> QuadratureResult:
    r"""Compute $\mathcal{N}(\alpha,\beta)$ using QUADPACK's QAGS integration routine.

    Parameters
    ----------
    alpha : float
        Angular scale parameter $\alpha$.
    beta : float
        Shape parameter $\beta$.
    rel_tol : float, optional
        Relative tolerance for the QAGS solver, by default ``1e-6``.

    Returns
    -------
    QuadratureResult
        Estimate of $\mathcal{N}(\alpha,\beta)$, function evaluations, and convergence flag.
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", category=IntegrationWarning)
        result = quad(
            _integrand,
            0.0,
            np.pi,
            args=(alpha, beta),
            epsabs=0.0,
            epsrel=rel_tol,
            limit=200,
            full_output=1,
        )

    value = float(result[0])
    info = result[2]
    evaluations_used = int(info["neval"])
    has_integration_warning = any(issubclass(warning.category, IntegrationWarning) for warning in caught_warnings)
    converged = len(result) == 3 and not has_integration_warning
    return QuadratureResult(value, evaluations_used, converged)
