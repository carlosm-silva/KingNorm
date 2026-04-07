import numpy as np
from scipy.special import gamma, hyp2f1
import time
from mpmath import mp, mpf, hyp2f1 as mp_hyp2f1, fac, pi as mpi


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


# --- Performance and Accuracy Benchmark ---

if __name__ == "__main__":
    alpha, beta = 0.01, 2.5

    print(f"Benchmarking King Normalization (α={alpha}, β={beta})")

    # Calculate ground truth
    reference_value = compute_normalization_mpmath(alpha, beta)

    benchmark_configs = [
        ("mpmath (Ground Truth)", compute_normalization_mpmath),
        ("SciPy (Vectorized)", compute_normalization_vectorized),
    ]

    print(
        f"{'Implementation':<25} | {'Result':<22} | {'Rel. Error':<12} | {'Avg Time (µs)':<14}"
    )
    print("-" * 80)

    for label, func in benchmark_configs:
        # Final value check
        result = func(alpha, beta)
        rel_error = (
            abs(result - reference_value) / abs(reference_value)
            if reference_value != 0
            else 0
        )

        # Timing
        # Preliminary call to ensure any lazy loading/caching is handled
        _ = func(alpha, beta)

        iterations = 100_000 if "SciPy" in label else 100
        start_time = time.perf_counter()
        for _ in range(iterations):
            func(alpha, beta)
        end_time = time.perf_counter()

        avg_latency = (end_time - start_time) / iterations * 1e6

        print(
            f"{label:<25} | {result:<22.17e} | {rel_error:<12.2e} | {avg_latency:<14.2f}"
        )
