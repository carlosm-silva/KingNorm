# Theory and Numerics

This page records the notation and numerical choices that are easy to lose when reading only the API reference. The full derivation lives in `paper/main.tex`; this page summarizes the parts that determine how the implementation is used and checked.

## Normalization Constant

King's function is used as a spherical point-spread function with angular separation $x$, angular scale $\alpha>0$, and shape parameter $\beta>0$. The polar normalization constant is

$$
\mathcal{N}(\alpha,\beta) := \int_{0}^{\pi}
\left(1 + \frac{x^2}{2\beta\alpha^2}\right)^{-\beta}\sin(x)\,dx.
$$

The full solid-angle normalization includes the azimuthal factor $2\pi$. The code in this repository evaluates only the polar factor $\mathcal{N}(\alpha,\beta)$.

## Exact Series

With

$$
\lambda = \frac{\pi^2}{2\beta\alpha^2},
$$

the paper proves the exact representation

$$
\mathcal{N}(\alpha,\beta) =
\sum_{n=0}^{\infty}
\frac{(-1)^n\pi^{2n+2}}{2(n+1)(2n+1)!}
{}_2F_1(n+1,\beta;n+2;-\lambda).
$$

The implementation [compute_normalization_series][implementations.compute_normalization_series] evaluates partial sums of this expression. If $S_N$ is the $N$-th partial sum and $R_N=\mathcal{N}(\alpha,\beta)-S_N$, the paper establishes the uniform absolute bound

$$
|R_N| \le \frac{\pi^{2N+4}}{2(N+2)(2N+3)!},
$$

and the uniform relative bound

$$
\epsilon_{rel} \le
\left(1 - \frac{\pi^2}{12}\right)^{-1}
\frac{\pi^{2N+2}}{(2N+3)!}.
$$

These bounds are independent of $\alpha$ and $\beta$ over the domain $\alpha,\beta>0$.

## Adaptive Stopping Rule

The code accumulates terms until the next omitted term is no larger than the requested tolerance times the current partial sum. The alternating-series remainder theorem makes this stopping rule meaningful because the proof shows that the non-alternating terms $a_n$ are positive, strictly decreasing, and converge to zero.

The `work_done` value for [SeriesResult][implementations.SeriesResult] is the number of terms included in the partial sum. A candidate next term may be evaluated to test convergence but is not counted as included work.

## Numerical Caveat

For large $\lambda$, general-purpose evaluations of the leading hypergeometric factor can lose accuracy. The implementation avoids this for $n=0$ by using the closed form

$$
{}_2F_1(1,\beta;2;-\lambda) =
\frac{(1+\lambda)^{1-\beta}-1}{\lambda(1-\beta)},
$$

with limiting value $\lambda^{-1}\log(1+\lambda)$ at $\beta=1$. This is why the implementation uses stable elementary functions for the leading term and SciPy's `hyp2f1` for the remaining terms.

## Benchmark Scope

The benchmark grid is intentionally narrower than the theorem:

- $\alpha\in[10^{-6},1]$, logarithmically spaced.
- $\beta\in[1,10]$, linearly spaced.
- Target relative tolerance $10^{-6}$.

The exact series and bounds apply for all $\alpha,\beta>0$. The grid describes the application regime used for runtime and accuracy comparisons against direct quadrature.
