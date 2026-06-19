# King Normalization

This project studies numerical implementations for the normalization constant of the King-function-like integral:

$$
N(\alpha,\beta) = \int_{0}^{\pi}
\left(1 + \frac{x^2}{2\beta\alpha^2}\right)^{-\beta}\sin(x)\,dx.
$$

## Series Formulation

The vectorized implementation evaluates the truncated series:

$$
N(\alpha,\beta) \approx \sum_{n=0}^{n_{\mathrm{terms}}-1}
\frac{(-1)^n\pi^{2n+2}}{2(n+1)(2n+1)!}\,
{}_2F_1(n+1,\beta;n+2;z),
$$

with

$$
z = -\frac{\pi^2}{2\beta\alpha^2}.
$$

## Implementations

- `compute_normalization_vectorized`: finite-term series using SciPy `hyp2f1`
- `compute_normalization_gauss_legendre`: adaptive Gauss-Legendre quadrature
- `compute_normalization_qags`: QUADPACK QAGS adaptive quadrature
- `compute_normalization_mpmath`: high-precision reference for accuracy checks

## Benchmark Scripts

- `speed_tests.py`: runs timing and accuracy sweeps, writes `benchmark_results.csv`
- `plot_benchmarks.py`: generates publication-ready plots in `plots/`

## Quick Start

```bash
python speed_tests.py
python plot_benchmarks.py
mkdocs serve
```

If math is rendering correctly, the equations above should appear as formatted symbols (not raw LaTeX text).

## API Reference (Auto-generated)

The sections below are built directly from Python docstrings using `mkdocstrings`.

### `implementations.py`

::: implementations
    options:
      members_order: source

### `speed_tests.py`

::: speed_tests
    options:
      members_order: source

### `plot_benchmarks.py`

::: plot_benchmarks
    options:
      members_order: source
