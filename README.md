# King's Function Normalization

This repository contains numerical implementations and benchmarks for the exact series representation of the King's-function normalization constant $\mathcal{N}(\alpha,\beta)$, accompanying the paper *"An exact series representation for the normalization constant of King's function"*.

```python
from implementations import compute_normalization_series

result = compute_normalization_series(alpha=0.1, beta=2.0, rel_tol=1e-6)
print(float(result), result.n_terms_used, result.converged)
```

## Organization

- **`implementations.py`**: The core mathematical routines. Contains the adaptive SciPy-backed series for $\mathcal{N}(\alpha,\beta)$, quadrature baselines, and arbitrary-precision references.
- **`speed_tests.py`**: A benchmarking script designed to compare execution latency and $\epsilon_{rel}$ of the available implementations over the $(\alpha,\beta)$ grid.
- **`tests/`**: Directory for test cases.
  - `test_king_norm.py`: A starting suite for unit tests to ensure that the numerical methods remain accurate and stable across the parameter space.

## Usage

Run commands from the repository root so local modules import correctly.

### Benchmarking
To run the performance and accuracy benchmarks:
```bash
python speed_tests.py
python plot_benchmarks.py
```

### Running Tests
To run the test suite:
```bash
python -m unittest discover -s tests
```

### Local Documentation
To inspect the MkDocs site locally:
```bash
python -m mkdocs serve
```

### Release Reproducibility
Use `environment.yml` as the maintained development and CI specification. For a release, create a clean environment from it, run the checks, then freeze exact package versions with:
```bash
python -m pip freeze > requirements-lock.txt
```
