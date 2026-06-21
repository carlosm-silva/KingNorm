# King's Function Normalization

This project accompanies the paper *"An exact series representation for the normalization constant of King's function"* and provides reproducible numerical evaluators for the polar normalization constant

$$
\mathcal{N}(\alpha,\beta) := \int_{0}^{\pi}
\left(1 + \frac{x^2}{2\beta\alpha^2}\right)^{-\beta}\sin(x)\,dx.
$$

Here $x$ is the angular separation, $\alpha>0$ is the angular scale, and $\beta>0$ is the shape parameter. The full solid-angle normalization is $2\pi\,\mathcal{N}(\alpha,\beta)$.

The default evaluator is the adaptive series [compute_normalization_series][implementations.compute_normalization_series].
Quadrature implementations are kept as independent baselines for accuracy and timing comparisons.

## Minimal Example

```python
from implementations import compute_normalization_series

result = compute_normalization_series(alpha=0.1, beta=2.0, rel_tol=1e-6)
print(float(result), result.n_terms_used, result.converged)
```

The result object converts to `float` for the estimate of $\mathcal{N}(\alpha,\beta)$ and also reports the number of terms included in the partial sum.

## What Is Included

- [Getting Started](usage.md) shows the local workflow for evaluating the normalization, running tests, regenerating benchmarks, and previewing the docs.
- [Theory and Numerics](theory.md) summarizes the notation, truncation guarantees, implementation caveats, and benchmark scope used by the paper.
- [API Reference](api.md) is generated from the Python docstrings with `mkdocstrings`.

## Main Interfaces

- [compute_normalization_series][implementations.compute_normalization_series]: adaptive finite-term series using SciPy `hyp2f1`.
- [compute_normalization_gauss_legendre][implementations.compute_normalization_gauss_legendre]: adaptive Gauss-Legendre quadrature baseline.
- [compute_normalization_qags][implementations.compute_normalization_qags]: QUADPACK QAGS quadrature baseline.
- [compute_normalization_mpmath][implementations.compute_normalization_mpmath]: high-precision series reference for accuracy checks.
- [compute_reference_integral][implementations.compute_reference_integral]: independent high-precision quadrature of the defining integral.

## Notation

The paper writes

$$
\lambda = \frac{\pi^2}{2\beta\alpha^2},
$$

and expresses the exact series using the argument $-\lambda$ of the Gauss hypergeometric function. The $N$-th partial sum is $S_N$, the truncation remainder is $R_N=\mathcal{N}(\alpha,\beta)-S_N$, and the relative truncation error is $\epsilon_{rel}=|R_N|/\mathcal{N}(\alpha,\beta)$.

If math is rendering correctly, the equations above should appear as formatted symbols (not raw LaTeX text).
