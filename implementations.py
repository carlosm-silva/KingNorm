import warnings
import numpy as np
from scipy.special import gamma, hyp2f1
from scipy.integrate import quad, fixed_quad, IntegrationWarning
from mpmath import mp, mpf, hyp2f1 as mp_hyp2f1, fac, pi as mpi

def _integrand(x, alpha, beta):
    return (1.0 + x**2 / (2.0 * beta * alpha**2))**(-beta) * np.sin(x)


def compute_normalization_vectorized(alpha, beta, n_terms=14):
    """
    Compute the normalization constant for the King function using a vectorized
    finite-term series expansion with SciPy's hypergeometric function.

    The implementation utilizes the expansion:
    N(α, β) = \sum_{n=0}^{n_terms-1} \frac{(-1)^n π^{2n+2}}{2(n+1)(2n+1)!} ₂F₁(n+1, β; n+2; z)
    where z = -π² / (2β α²).

    Parameters:
    -----------
    alpha : float or array-like
        The core scale parameter (α > 0).
    beta : float
        The concentration parameter (β > 0).
    n_terms : int, optional
        Number of terms to include in the expansion. Default is 14,
        which is universally sufficient to reach float64 machine
        precision across the typical stable parameter domain.

    Returns:
    --------
    float or array-like
        The computed normalization constant.
    """
    # Create the term index array
    n = np.arange(n_terms, dtype=np.float64)

    # Calculate the argument for the hypergeometric function
    z = -np.pi**2 / (2 * beta * alpha**2)

    # Precompute the series coefficients: (-1)^n * π^(2n+2) / (2 * (n+1) * (2n+1)!)
    # Note: gamma(2n + 2) = (2n+1)!
    coef = ((-1) ** n * np.pi ** (2 * n + 2)) / (2 * (n + 1) * gamma(2 * n + 2))

    # Evaluate the Gaussian hypergeometric function ₂F₁(a, b; c; z)
    # SciPy's implementation is natively vectorized over 'a' and 'z'
    hyper_val = hyp2f1(n + 1, beta, n + 2, z)

    # Sum all terms to obtain the final normalization value
    return np.sum(coef * hyper_val)


def compute_normalization_mpmath(alpha, beta, precision=30, n_terms=30):
    """
    High-precision evaluation of the King normalization constant using mpmath.
    Used as a ground-truth reference for validating numerical accuracy.

    Parameters:
    -----------
    alpha : float
        The scale parameter α.
    beta : float
        The parameter β.
    precision : int, optional
        Digits of precision (dps) for the mpmath context. Default is 30.
    n_terms : int, optional
        Number of terms in the series summation. Default is 30.

    Returns:
    --------
    float
        The normalization constant cast to standard float64.
    """
    mp.dps = precision
    alpha_mp, beta_mp = mpf(alpha), mpf(beta)
    z = -(mpi**2) / (2 * beta_mp * alpha_mp**2)

    total = mpf(0)
    for n in range(n_terms):
        term = (
            ((-1) ** n * mpi ** (2 * n + 2))
            / (2 * (n + 1) * fac(2 * n + 1))
            * mp_hyp2f1(n + 1, beta_mp, n + 2, z)
        )
        total += term

    return float(total)


def compute_normalization_gauss_legendre(alpha, beta, rel_tol=np.finfo(float).eps):
    """
    Compute the normalization constant for the King function using adaptive
    Gauss-Legendre quadrature.

    Parameters:
    -----------
    alpha : float
        The core scale parameter (α > 0).
    beta : float
        The concentration parameter (β > 0).
    rel_tol : float, optional
        Relative error tolerance. Defaults to machine precision.

    Returns:
    --------
    float
        The computed normalization constant.
    """
    val = 0.0
    for n in range(10, 501, 10):
        new_val, _ = fixed_quad(_integrand, 0.0, np.pi, args=(alpha, beta), n=n)
        if n > 10:
            if abs(new_val - val) <= rel_tol * abs(new_val):
                return new_val
        val = new_val
    return val


def compute_normalization_qags(alpha, beta, rel_tol=np.finfo(float).eps):
    """
    Compute the normalization constant for the King function using adaptive
    quadrature (QUADPACK's QAGS).

    Parameters:
    -----------
    alpha : float
        The core scale parameter (α > 0).
    beta : float
        The concentration parameter (β > 0).
    rel_tol : float, optional
        Relative error tolerance. Defaults to machine precision.

    Returns:
    --------
    float
        The computed normalization constant.
    """
    # QUADPACK epsrel must be >= 50 * machine_epsilon when epsabs=0.0
    epsrel_safe = max(rel_tol, 1.12e-14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        val, err = quad(_integrand, 0.0, np.pi, args=(alpha, beta), epsabs=0.0, epsrel=epsrel_safe, limit=200)
    return val
