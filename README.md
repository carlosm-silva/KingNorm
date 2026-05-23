# King's Function Normalization

This repository contains numerical implementations and benchmarks for the exact series representation of the normalization constant of King's function, accompanying the paper *"An exact series representation for the normalization constant of King’s function"*.

## Organization

- **`implementations.py`**: The core mathematical routines. Contains a fast, vectorized SciPy implementation for float64 precision and an arbitrary-precision reference implementation using `mpmath`.
- **`speed_tests.py`**: A benchmarking script designed to compare the execution latency and relative error of the available implementations.
- **`tests/`**: Directory for test cases.
  - `test_king_norm.py`: A starting suite for unit tests to ensure that the numerical methods remain accurate and stable across the parameter space.

## Usage

### Benchmarking
To run the performance and accuracy benchmarks:
```bash
python speed_tests.py
```

### Running Tests
To run the test suite:
```bash
python -m unittest discover -s tests
```
