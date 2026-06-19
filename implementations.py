"""Numerical implementations for King-function normalization constants."""

from __future__ import annotations

import warnings

from mpmath import fac, mp, mpf
from mpmath import hyp2f1 as mp_hyp2f1
from mpmath import pi as mpi
import numpy as np
from scipy.integrate import IntegrationWarning, fixed_quad, quad
from scipy.special import gamma, hyp2f1


def _integrand(x: float, alpha: float, beta: float) -> float:
    """Evaluate the King integrand at a single point.

    Parameters
    ----------
    x : float
        Integration variable.
    alpha : float
        Core scale parameter.
    beta : float
        Concentration parameter.

    Returns
    -------
    float
        Integrand value.
    """
    return (1.0 + x**2 / (2.0 * beta * alpha**2)) ** (-beta) * np.sin(x)


def compute_normalization_vectorized(alpha: float, beta: float, n_terms: int = 14) -> float:
    r"""Compute normalization via a vectorized finite-term series expansion.

    Notes
    -----
    The implementation evaluates the truncated series

    $$
    N(\alpha, \beta) = \sum_{n=0}^{n_{\mathrm{terms}}-1}
    \frac{(-1)^n\pi^{2n+2}}{2(n+1)(2n+1)!}
    \,{}_2F_1(n+1, \beta; n+2; z),
    $$

    where

    $$
    z = -\frac{\pi^2}{2\beta\alpha^2}
    $$

    Parameters
    ----------
    alpha : float
        Core scale parameter.
    beta : float
        Concentration parameter.
    n_terms : int, optional
        Number of series terms to include, by default 14.

    Returns
    -------
    float
        Normalization constant estimate.
    """
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    n = np.arange(n_terms, dtype=np.float64)
    z = -(np.pi**2) / (2.0 * beta * alpha_arr**2)

    coef = ((-1.0) ** n * np.pi ** (2.0 * n + 2.0)) / (2.0 * (n + 1.0) * gamma(2.0 * n + 2.0))
    hyper_val = hyp2f1(n + 1.0, beta, n + 2.0, z)
    return float(np.sum(coef * hyper_val))


def compute_normalization_mpmath(alpha: float, beta: float, precision: int = 30, n_terms: int = 30) -> float:
    """Compute a high-precision normalization reference with mpmath.

    Parameters
    ----------
    alpha : float
        Core scale parameter.
    beta : float
        Concentration parameter.
    precision : int, optional
        Decimal precision used by mpmath, by default 30.
    n_terms : int, optional
        Number of terms in the series, by default 30.

    Returns
    -------
    float
        High-precision normalization constant cast to ``float``.
    """
    mp.dps = precision
    alpha_mp, beta_mp = mpf(alpha), mpf(beta)
    z = -(mpi**2) / (2.0 * beta_mp * alpha_mp**2)

    total = mpf(0)
    for n in range(n_terms):
        term = ((-1) ** n * mpi ** (2 * n + 2)) / (2 * (n + 1) * fac(2 * n + 1)) * mp_hyp2f1(n + 1, beta_mp, n + 2, z)
        total += term

    return float(total)


def compute_normalization_gauss_legendre(
    alpha: float,
    beta: float,
    rel_tol: float = np.finfo(float).eps,
) -> float:
    """Compute normalization using adaptive Gauss-Legendre quadrature.

    Parameters
    ----------
    alpha : float
        Core scale parameter.
    beta : float
        Concentration parameter.
    rel_tol : float, optional
        Relative convergence tolerance, by default machine epsilon.

    Returns
    -------
    float
        Quadrature estimate of normalization.
    """
    val = 0.0
    for n in range(10, 501, 10):
        new_val, _ = fixed_quad(_integrand, 0.0, np.pi, args=(alpha, beta), n=n)
        if n > 10 and abs(new_val - val) <= rel_tol * abs(new_val):
            return float(new_val)
        val = float(new_val)
    return val


def compute_normalization_qags(alpha: float, beta: float, rel_tol: float = np.finfo(float).eps) -> float:
    """Compute normalization using QUADPACK's QAGS integration routine.

    Parameters
    ----------
    alpha : float
        Core scale parameter.
    beta : float
        Concentration parameter.
    rel_tol : float, optional
        Relative tolerance for the QAGS solver, by default machine epsilon.

    Returns
    -------
    float
        Quadrature estimate of normalization.
    """
    epsrel_safe = max(rel_tol, 1.12e-14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        val, _ = quad(_integrand, 0.0, np.pi, args=(alpha, beta), epsabs=0.0, epsrel=epsrel_safe, limit=200)
    return float(val)
