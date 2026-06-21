# Getting Started

This page shows the common local workflows for evaluating $\mathcal{N}(\alpha,\beta)$, checking the implementation, regenerating benchmark outputs, and previewing the documentation site.

Run commands from the repository root. The repository is currently organized as a flat research-code project, so running from the root keeps imports such as `import implementations` unambiguous.

## Evaluate the Series

```python
from implementations import compute_normalization_series

result = compute_normalization_series(alpha=0.1, beta=2.0, rel_tol=1e-6)
normalization = float(result)

print(normalization)
print(result.n_terms_used)
print(result.converged)
```

Use [compute_normalization_series][implementations.compute_normalization_series] for normal floating-point evaluation. It returns a [SeriesResult][implementations.SeriesResult] with the estimate of $\mathcal{N}(\alpha,\beta)$, the number of included terms, and a convergence flag for the adaptive stopping rule.

## Compare Against References

The repository includes two independent reference routes:

- [compute_normalization_mpmath][implementations.compute_normalization_mpmath] evaluates the same exact series with arbitrary precision.
- [compute_reference_integral][implementations.compute_reference_integral] evaluates the defining integral directly with high-precision quadrature.

```python
from implementations import compute_normalization_series, compute_reference_integral

alpha = 0.1
beta = 2.0

reference = compute_reference_integral(alpha, beta, precision=80)
estimate = float(compute_normalization_series(alpha, beta, rel_tol=1e-6))
epsilon_rel = abs(estimate - reference) / abs(reference)

print(epsilon_rel)
```

## Run Checks

The test suite uses `unittest`:

```bash
python -m unittest discover -s tests
```

The paper-claim checker recomputes the quoted truncation-bound values and spot-checks the current implementation:

```bash
python scripts/check_paper_numerics.py
```

## Regenerate Benchmarks

The benchmark script evaluates the adaptive series, adaptive Gauss-Legendre quadrature, and QUADPACK QAGS over the application grid, then writes `benchmark_results.csv` and `benchmark_metadata.json`.

```bash
python speed_tests.py
python plot_benchmarks.py
```

`plot_benchmarks.py` writes publication-oriented figures and table source data under `plots/`.

## Preview the Docs

Build the site strictly before serving it:

```bash
python -m mkdocs build --strict
python -m mkdocs serve
```

The local preview is usually available at `http://127.0.0.1:8000`.

## Release Reproducibility

`environment.yml` is the maintained dependency specification for development and CI. For a release, create a clean environment from that file, run the checks above, then freeze the exact Python packages used for the release:

```bash
python -m pip freeze > requirements-lock.txt
```

If the release needs exact conda package builds as well as Python package versions, also keep an explicit conda export from the same environment:

```bash
conda list --explicit > conda-lock.txt
```
