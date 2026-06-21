r"""Generate comparative visualizations for $\mathcal{N}(\alpha,\beta)$ benchmarks."""

from __future__ import annotations

from math import factorial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib

from implementations import (
    compute_normalization_series,
    compute_reference_integral,
)

matplotlib.use("Agg")

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
    }
)

IMPL_COLORS = {
    "Series": "#2563eb",
    "QAGS": "#dc2626",
    "Gauss-Legendre": "#16a34a",
}
IMPL_MARKERS = {
    "Series": "o",
    "QAGS": "s",
    "Gauss-Legendre": "^",
}
IMPL_LABELS = {
    "Series": "Adaptive series",
    "Gauss-Legendre": "Gauss-Legendre (adaptive)",
    "QAGS": "QAGS (SciPy default)",
}
IMPLEMENTATIONS = ("Series", "Gauss-Legendre", "QAGS")
PLOTS_DIR = Path("plots")
CONVERGENCE_TERMS = np.arange(1, 31)
CONVERGENCE_CASES = (
    (1e-6, 1.0, r"$\alpha=10^{-6},\,\beta=1$"),
    (1e-2, 5.8, r"$\alpha=10^{-2},\,\beta=5.8$"),
    (1.0, 10.0, r"$\alpha=1,\,\beta=10$"),
)
REQUIRED_COLUMNS = {
    "Implementation",
    "Alpha",
    "Beta",
    "Target_Rel_Tol",
    "Reference",
    "Estimate",
    "Rel_Error",
    "Converged",
    "Work_Done",
    "Work_Unit",
    "Time_Mean_us",
    "Time_Min_us",
    "Time_Q1_us",
    "Time_Median_us",
    "Time_Q3_us",
    "Time_Max_us",
}
REFERENCE_PRECISION = 80
REFERENCE_TERMS = 80
LOG_ERROR_FLOOR = 1e-20


def absolute_remainder_bound(partial_sum_index: int) -> float:
    r"""Compute the uniform absolute truncation bound for $S_N$.

    Parameters
    ----------
    partial_sum_index : int
        Partial-sum index $N$.

    Returns
    -------
    float
        Parameter-independent upper bound on $|R_N|$.
    """
    return float(
        np.pi ** (2 * partial_sum_index + 4) / (2 * (partial_sum_index + 2) * factorial(2 * partial_sum_index + 3))
    )


def relative_remainder_bound(partial_sum_index: int) -> float:
    r"""Compute the uniform relative truncation bound for $S_N$.

    Parameters
    ----------
    partial_sum_index : int
        Partial-sum index $N$.

    Returns
    -------
    float
        Parameter-independent upper bound on $\epsilon_{rel}=|R_N|/\mathcal{N}(\alpha,\beta)$.
    """
    chebyshev_factor = 1.0 / (1.0 - np.pi**2 / 12.0)
    return float(chebyshev_factor * np.pi ** (2 * partial_sum_index + 2) / factorial(2 * partial_sum_index + 3))


def _as_axes_list(axes: object) -> list[Axes]:
    """Return Matplotlib axes as a flat typed list."""
    return [cast("Axes", ax) for ax in np.asarray(axes, dtype=object).ravel()]


def _save(fig: Figure, name: str) -> None:
    """Save and close a figure."""
    output_path = PLOTS_DIR / name
    fig.savefig(output_path, bbox_inches="tight")
    print(f"  -> saved {output_path}")
    plt.close(fig)


def _validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if benchmark results do not match the expected schema."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        message = "benchmark_results.csv does not match the expected schema; missing columns: " + ", ".join(missing)
        raise ValueError(message)
    if "Precision" in df.columns:
        raise ValueError("benchmark_results.csv still has a Precision column; rerun speed_tests.py first.")


def _target_rel_tol(df: pd.DataFrame) -> float:
    r"""Return the common target tolerance for $\epsilon_{rel}$ from the benchmark data."""
    values = df["Target_Rel_Tol"].dropna().unique()
    if len(values) != 1:
        raise ValueError("Expected one common Target_Rel_Tol value in benchmark_results.csv.")
    return float(values[0])


def _timing_col(df: pd.DataFrame) -> str:
    """Return the headline timing column."""
    return "Time_Median_us" if "Time_Median_us" in df.columns else "Time_Min_us"


def _positive_log_values(values: np.ndarray, floor: float = LOG_ERROR_FLOOR) -> np.ndarray:
    """Floor non-positive values for log-scale plotting."""
    return np.maximum(values.astype(float), floor)


def _converged_mask(values: pd.Series) -> np.ndarray:
    """Return a boolean convergence mask robust to string-loaded CSV values."""
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.lower().str.strip()
    return normalized.isin({"true", "1", "yes"}).to_numpy(dtype=bool)


def _line_with_convergence_markers(
    ax: Axes,
    sample: pd.DataFrame,
    x_col: str,
    y_col: str,
    impl: str,
    label: str | None = None,
    *,
    line: bool = True,
) -> None:
    """Plot filled markers for converged rows and hollow markers for failures."""
    sample = sample.sort_values(x_col)
    color = IMPL_COLORS[impl]
    marker = IMPL_MARKERS[impl]
    x_values = sample[x_col].to_numpy(dtype=float)
    y_values = _positive_log_values(sample[y_col].to_numpy(dtype=float))

    if line:
        ax.plot(x_values, y_values, color=color, lw=1.2, alpha=0.9)

    converged = _converged_mask(sample["Converged"])
    if np.any(converged):
        ax.scatter(
            x_values[converged],
            y_values[converged],
            color=color,
            marker=marker,
            s=26,
            edgecolors="black",
            linewidths=0.35,
            label=label,
            zorder=4,
        )
    if np.any(~converged):
        ax.scatter(
            x_values[~converged],
            y_values[~converged],
            facecolors="none",
            edgecolors=color,
            marker=marker,
            s=42,
            linewidths=1.2,
            label=f"{label} (not converged)" if label else None,
            zorder=5,
        )


def _grid_axes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted grid coordinates and cell edges for heatmaps."""
    alphas = np.sort(df["Alpha"].unique().astype(float))
    betas = np.sort(df["Beta"].unique().astype(float))
    log_alphas = np.log10(alphas)
    log_alpha_edges = _cell_edges(log_alphas)
    beta_edges = _cell_edges(betas)
    return alphas, betas, log_alpha_edges, beta_edges


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Compute heatmap cell edges from sorted centers."""
    if len(centers) == 1:
        delta = abs(centers[0]) * 0.05 if centers[0] != 0 else 0.5
        return np.asarray([centers[0] - delta, centers[0] + delta])
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _heatmap_grid(
    sample: pd.DataFrame,
    alphas: np.ndarray,
    betas: np.ndarray,
    value_col: str,
    *,
    transform_log10: bool = True,
) -> np.ndarray:
    r"""Convert a method sample into a $(\beta,\alpha)$ grid."""
    grid = np.full((len(betas), len(alphas)), np.nan)
    for alpha, beta, value in zip(sample["Alpha"], sample["Beta"], sample[value_col], strict=True):
        i = int(np.searchsorted(betas, float(beta)))
        j = int(np.searchsorted(alphas, float(alpha)))
        value_float = float(value)
        grid[i, j] = np.log10(max(value_float, LOG_ERROR_FLOOR)) if transform_log10 else value_float
    return grid


def _overlay_nonconverged_cells(
    ax: Axes,
    sample: pd.DataFrame,
    alphas: np.ndarray,
    betas: np.ndarray,
    log_alpha_edges: np.ndarray,
    beta_edges: np.ndarray,
) -> None:
    """Outline heatmap cells whose benchmark row did not converge."""
    failed = sample[~_converged_mask(sample["Converged"])]
    for alpha, beta in zip(failed["Alpha"], failed["Beta"], strict=True):
        i = int(np.searchsorted(betas, float(beta)))
        j = int(np.searchsorted(alphas, float(alpha)))
        rectangle = Rectangle(
            (log_alpha_edges[j], beta_edges[i]),
            log_alpha_edges[j + 1] - log_alpha_edges[j],
            beta_edges[i + 1] - beta_edges[i],
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            hatch="///",
        )
        ax.add_patch(rectangle)


def _target_line(ax: Axes, target_rel_tol: float, *, label: str = "target") -> None:
    r"""Draw the target $\epsilon_{rel}$ line."""
    ax.axhline(target_rel_tol, color="black", linestyle="--", linewidth=1.0, alpha=0.85, label=label)


def _float64_line(ax: Axes) -> None:
    r"""Draw a faint float64 machine-$\epsilon$ reference."""
    ax.axhline(np.finfo(np.float64).eps, color="0.55", linestyle=":", linewidth=0.9, alpha=0.5, label="float64 eps")


def _converged_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that self-reported convergence."""
    return df[_converged_mask(df["Converged"])]


def plot_accuracy_vs_speed(df: pd.DataFrame) -> None:
    r"""Plot $\epsilon_{rel}$ and runtime trade-offs among converged runs."""
    _validate_schema(df)
    target_rel_tol = _target_rel_tol(df)
    time_col = _timing_col(df)
    converged_df = _converged_rows(df)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for impl in IMPLEMENTATIONS:
        sample = df[df["Implementation"] == impl]
        converged_sample = converged_df[converged_df["Implementation"] == impl]
        if converged_sample.empty:
            continue
        median_time = float(converged_sample[time_col].median())
        median_error = max(float(converged_sample["Rel_Error"].median()), LOG_ERROR_FLOOR)
        max_error = max(float(converged_sample["Rel_Error"].max()), median_error)
        ax.vlines(median_time, median_error, max_error, color=IMPL_COLORS[impl], linewidth=2.0, alpha=0.28)
        ax.scatter(
            median_time,
            median_error,
            color=IMPL_COLORS[impl],
            marker=IMPL_MARKERS[impl],
            s=150,
            edgecolors="black",
            linewidths=0.65,
            zorder=5,
            label=IMPL_LABELS[impl],
        )

        if impl == "QAGS":
            failed_fraction = 1.0 - float(sample["Converged"].mean())
            ax.annotate(
                f"{failed_fraction:.0%} non-converged\n(excluded)",
                (median_time, median_error),
                xytext=(-14, 6),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8,
                color=IMPL_COLORS[impl],
            )

    _target_line(ax, target_rel_tol)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Median converged runtime ($\mu$s)")
    ax.set_ylabel(r"$\epsilon_{rel}$ (converged rows)")
    ax.set_title("Accuracy and Runtime at Target Tolerance")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "plot_accuracy_vs_speed.pdf")


def plot_timing_boxplots(df: pd.DataFrame) -> None:
    """Plot timing distributions among converged rows."""
    _validate_schema(df)
    time_col = _timing_col(df)
    converged_df = _converged_rows(df)
    data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    for impl in IMPLEMENTATIONS:
        vals = converged_df[converged_df["Implementation"] == impl][time_col].to_numpy(dtype=float)
        data.append(vals)
        labels.append(IMPL_LABELS[impl])
        colors.append(IMPL_COLORS[impl])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    box_props = dict(marker=".", ms=3, alpha=0.3)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True, flierprops=box_props)
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    ax.set_yscale("log")
    ax.set_ylabel("Median runtime per call (us), converged rows only")
    ax.set_title("Timing Distribution Among Converged Runs")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "plot_timing_boxplots.pdf")


def plot_error_heatmaps(df: pd.DataFrame) -> None:
    r"""Plot log-scaled $\epsilon_{rel}$ heatmaps over $(\alpha,\beta)$."""
    _validate_schema(df)
    target_rel_tol = _target_rel_tol(df)
    alphas, betas, log_alpha_edges, beta_edges = _grid_axes(df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    axes_list = _as_axes_list(axes)
    floor = -17
    vmin, vmax = floor, 0
    images: list[AxesImage] = []
    log_alphas = np.log10(alphas)
    log_alpha_grid, beta_grid = np.meshgrid(log_alphas, betas)

    for ax, impl in zip(axes_list, IMPLEMENTATIONS, strict=True):
        sample = df[df["Implementation"] == impl]
        grid = _heatmap_grid(sample, alphas, betas, "Rel_Error")
        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
            extent=[log_alpha_edges[0], log_alpha_edges[-1], beta_edges[0], beta_edges[-1]],
        )
        images.append(image)
        if np.nanmin(grid) <= np.log10(target_rel_tol) <= np.nanmax(grid):
            ax.contour(
                log_alpha_grid,
                beta_grid,
                grid,
                levels=[np.log10(target_rel_tol)],
                colors="black",
                linewidths=1.0,
            )
        _overlay_nonconverged_cells(ax, sample, alphas, betas, log_alpha_edges, beta_edges)
        ax.set_title(IMPL_LABELS[impl], fontweight="bold")
        ax.set_xlabel(r"$\log_{10}\alpha$")
        if impl == "Series":
            ax.set_ylabel(r"$\beta$")

    cbar = fig.colorbar(images[-1], ax=axes_list, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\log_{10}\epsilon_{rel}$")
    fig.suptitle(r"$\epsilon_{rel}$ Across the IceCube Parameter Grid", fontweight="bold", y=1.03)
    _save(fig, "plot_error_heatmaps.pdf")


def plot_series_error_heatmap(df: pd.DataFrame) -> None:
    r"""Plot observed series $\epsilon_{rel}$ divided by the analytic bound at achieved $N$."""
    _validate_schema(df)
    series_df = df[df["Implementation"] == "Series"].copy()
    series_df["Bound_Ratio"] = [
        rel_error / relative_remainder_bound(int(work_done) - 1)
        for rel_error, work_done in zip(series_df["Rel_Error"], series_df["Work_Done"], strict=True)
    ]
    alphas, betas, log_alpha_edges, beta_edges = _grid_axes(series_df)
    grid = _heatmap_grid(series_df, alphas, betas, "Bound_Ratio")
    vmax = max(0.0, float(np.nanmax(grid)))

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=-16,
        vmax=vmax,
        extent=[log_alpha_edges[0], log_alpha_edges[-1], beta_edges[0], beta_edges[-1]],
    )
    _overlay_nonconverged_cells(ax, series_df, alphas, betas, log_alpha_edges, beta_edges)
    ax.set_xlabel(r"$\log_{10}\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"Series $\epsilon_{rel}$ Relative to Proven Bound")
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label(r"$\log_{10}$(observed $\epsilon_{rel}$ / bound)")
    # fig.text(
    #     0.5,
    #     0.01,
    #     "Figure 2 role: uniformity of the proved bound across the IceCube grid at the operational tolerance.",
    #     ha="center",
    #     fontsize=8.5,
    #     color="0.25",
    # )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "plot_series_error_heatmap.pdf")


def plot_error_vs_alpha(df: pd.DataFrame) -> None:
    r"""Plot $\epsilon_{rel}$ versus $\alpha$ for selected $\beta$ slices."""
    _validate_schema(df)
    target_rel_tol = _target_rel_tol(df)
    betas_sel = [1.0, 4.0, 7.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes_list = _as_axes_list(axes)
    for ax, beta in zip(axes_list, betas_sel, strict=True):
        for impl in IMPLEMENTATIONS:
            sample = df[(df["Implementation"] == impl) & (np.isclose(df["Beta"], beta))]
            _line_with_convergence_markers(ax, sample, "Alpha", "Rel_Error", impl, IMPL_LABELS[impl])
        _target_line(ax, target_rel_tol)
        _float64_line(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 1)
        ax.set_title(rf"$\beta = {beta:.1f}$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=6.8, framealpha=0.85)

    fig.supxlabel(r"$\alpha$", fontsize=12)
    fig.supylabel(r"$\epsilon_{rel}$", fontsize=12)
    fig.suptitle(r"$\epsilon_{rel}$ vs $\alpha$ at Selected $\beta$ Slices", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "plot_error_vs_alpha.pdf")


def plot_time_vs_alpha(df: pd.DataFrame) -> None:
    r"""Plot runtime versus $\alpha$ for selected $\beta$ slices."""
    _validate_schema(df)
    betas_sel = [1.0, 4.0, 7.0, 10.0]
    time_col = _timing_col(df)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes_list = _as_axes_list(axes)
    for ax, beta in zip(axes_list, betas_sel, strict=True):
        for impl in IMPLEMENTATIONS:
            sample_all = df[(df["Implementation"] == impl) & (np.isclose(df["Beta"], beta))].sort_values("Alpha")
            sample = sample_all[_converged_mask(sample_all["Converged"])]
            if sample.empty:
                continue
            alphas = sample["Alpha"].to_numpy(dtype=float)
            widths = alphas * 0.08
            for width, alpha, min_time, q1_time, q3_time, max_time in zip(
                widths,
                sample["Alpha"],
                sample["Time_Min_us"],
                sample["Time_Q1_us"],
                sample["Time_Q3_us"],
                sample["Time_Max_us"],
                strict=True,
            ):
                ax.vlines(alpha, min_time, max_time, color=IMPL_COLORS[impl], alpha=0.45, linewidth=1.0)
                ax.bar(
                    alpha,
                    q3_time - q1_time,
                    bottom=q1_time,
                    width=width,
                    color=IMPL_COLORS[impl],
                    edgecolor="black",
                    linewidth=0.25,
                    alpha=0.25,
                )
            ax.plot(
                alphas,
                sample[time_col],
                color=IMPL_COLORS[impl],
                marker=IMPL_MARKERS[impl],
                ms=3.6,
                lw=1.1,
                label=IMPL_LABELS[impl],
            )
            failed = sample_all[~_converged_mask(sample_all["Converged"])]
            if not failed.empty:
                ax.scatter(
                    failed["Alpha"],
                    failed[time_col],
                    facecolors="none",
                    edgecolors=IMPL_COLORS[impl],
                    marker=IMPL_MARKERS[impl],
                    s=42,
                    linewidths=1.2,
                    label=f"{IMPL_LABELS[impl]} give-up runs",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\beta = {beta:.1f}$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=6.6, framealpha=0.85)

    fig.supxlabel(r"$\alpha$", fontsize=12)
    fig.supylabel("Runtime among converged rows (us)", fontsize=12)
    fig.suptitle(r"Execution Time vs $\alpha$ at Selected $\beta$ Slices", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "plot_time_vs_alpha.pdf")


def plot_time_heatmaps(df: pd.DataFrame) -> None:
    r"""Plot timing and work heatmaps over $(\alpha,\beta)$."""
    _validate_schema(df)
    time_col = _timing_col(df)
    alphas, betas, log_alpha_edges, beta_edges = _grid_axes(df)
    converged_df = _converged_rows(df)

    all_times = np.log10(converged_df[time_col].to_numpy(dtype=float))
    time_vmin, time_vmax = float(np.nanmin(all_times)), float(np.nanmax(all_times))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    axes_list = _as_axes_list(axes)
    images: list[AxesImage] = []
    for ax, impl in zip(axes_list, IMPLEMENTATIONS, strict=True):
        sample_all = df[df["Implementation"] == impl]
        sample = converged_df[converged_df["Implementation"] == impl]
        grid = _heatmap_grid(sample, alphas, betas, time_col)
        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            vmin=time_vmin,
            vmax=time_vmax,
            extent=[log_alpha_edges[0], log_alpha_edges[-1], beta_edges[0], beta_edges[-1]],
        )
        images.append(image)
        _overlay_nonconverged_cells(ax, sample_all, alphas, betas, log_alpha_edges, beta_edges)
        ax.set_title(IMPL_LABELS[impl], fontweight="bold")
        ax.set_xlabel(r"$\log_{10}\alpha$")
        if impl == "Series":
            ax.set_ylabel(r"$\beta$")
    cbar = fig.colorbar(images[-1], ax=axes_list, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\log_{10}$(runtime / $\mu$s), converged rows")
    fig.suptitle("Execution Time Across Parameter Space", fontweight="bold", y=1.03)
    _save(fig, "plot_time_heatmaps.pdf")

    all_work = np.log10(df["Work_Done"].to_numpy(dtype=float))
    work_vmin, work_vmax = float(np.nanmin(all_work)), float(np.nanmax(all_work))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    axes_list = _as_axes_list(axes)
    work_images: list[AxesImage] = []
    for ax, impl in zip(axes_list, IMPLEMENTATIONS, strict=True):
        sample = df[df["Implementation"] == impl]
        grid = _heatmap_grid(sample, alphas, betas, "Work_Done")
        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=work_vmin,
            vmax=work_vmax,
            extent=[log_alpha_edges[0], log_alpha_edges[-1], beta_edges[0], beta_edges[-1]],
        )
        work_images.append(image)
        _overlay_nonconverged_cells(ax, sample, alphas, betas, log_alpha_edges, beta_edges)
        ax.set_title(IMPL_LABELS[impl], fontweight="bold")
        ax.set_xlabel(r"$\log_{10}\alpha$")
        if impl == "Series":
            ax.set_ylabel(r"$\beta$")
    cbar = fig.colorbar(work_images[-1], ax=axes_list, shrink=0.85, pad=0.02)
    cbar.set_label(r"log$_{10}$(work done)")
    fig.suptitle("Implementation Work Across Parameter Space", fontweight="bold", y=1.03)
    _save(fig, "plot_work_heatmaps.pdf")


def plot_summary_bars(df: pd.DataFrame) -> None:
    """Plot private aggregate sanity bars."""
    _validate_schema(df)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes_list = _as_axes_list(axes)
    x_pos = np.arange(len(IMPLEMENTATIONS))
    labels = [IMPL_LABELS[impl] for impl in IMPLEMENTATIONS]

    max_all = [
        max(float(df[df["Implementation"] == impl]["Rel_Error"].max()), LOG_ERROR_FLOOR) for impl in IMPLEMENTATIONS
    ]
    max_conv = [
        max(
            float(_converged_rows(df)[_converged_rows(df)["Implementation"] == impl]["Rel_Error"].max()),
            LOG_ERROR_FLOOR,
        )
        for impl in IMPLEMENTATIONS
    ]
    width = 0.36
    axes_list[0].bar(x_pos - width / 2, max_all, width, label="all rows", color="0.65")
    axes_list[0].bar(
        x_pos + width / 2,
        max_conv,
        width,
        label="converged rows",
        color=[IMPL_COLORS[impl] for impl in IMPLEMENTATIONS],
    )
    axes_list[0].set_yscale("log")
    axes_list[0].set_title(r"Maximum $\epsilon_{rel}$")
    axes_list[0].legend(fontsize=7.5)

    convergence_rates = [float(df[df["Implementation"] == impl]["Converged"].mean()) for impl in IMPLEMENTATIONS]
    axes_list[1].bar(x_pos, convergence_rates, color=[IMPL_COLORS[impl] for impl in IMPLEMENTATIONS])
    axes_list[1].set_ylim(0, 1.05)
    axes_list[1].set_title("Convergence rate")

    time_col = _timing_col(df)
    converged_df = _converged_rows(df)
    median_times = [
        float(converged_df[converged_df["Implementation"] == impl][time_col].median()) for impl in IMPLEMENTATIONS
    ]
    axes_list[2].bar(x_pos, median_times, color=[IMPL_COLORS[impl] for impl in IMPLEMENTATIONS])
    axes_list[2].set_yscale("log")
    axes_list[2].set_title("Median runtime, converged")

    for ax in axes_list:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", ls=":", alpha=0.35)
    fig.suptitle("Aggregate Benchmark Sanity Checks", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "plot_summary_bars.pdf")


def plot_series_convergence() -> None:
    r"""Plot float64 implementation $\epsilon_{rel}$ versus series length and the analytic bound."""
    target_rel_tol = 1e-6
    partial_sum_indices = CONVERGENCE_TERMS - 1
    relative_bounds = np.asarray([relative_remainder_bound(int(n)) for n in partial_sum_indices])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONVERGENCE_CASES)))

    for color, (alpha, beta, label) in zip(colors, CONVERGENCE_CASES, strict=True):
        truth = compute_reference_integral(alpha, beta, precision=REFERENCE_PRECISION)
        estimates = np.asarray(
            [float(compute_normalization_series(alpha, beta, n_terms=int(n_terms))) for n_terms in CONVERGENCE_TERMS]
        )
        implementation_errors = np.maximum(np.abs(estimates - truth) / abs(truth), LOG_ERROR_FLOOR)
        ax.plot(CONVERGENCE_TERMS, implementation_errors, color=color, marker="o", ms=3.5, lw=1.2, label=label)

        adaptive_result = compute_normalization_series(alpha, beta, rel_tol=target_rel_tol)
        adaptive_terms = int(adaptive_result.work_done)
        marker_index = np.where(adaptive_terms == CONVERGENCE_TERMS)[0]
        if len(marker_index) == 1:
            index = int(marker_index[0])
            ax.scatter(
                [adaptive_terms],
                [implementation_errors[index]],
                marker="*",
                s=130,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )

    ax.plot(
        CONVERGENCE_TERMS,
        relative_bounds,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Uniform relative bound",
    )
    _target_line(ax, target_rel_tol)
    _float64_line(ax)
    ax.set_xlabel("Number of series terms")
    ax.set_ylabel(r"$\epsilon_{rel}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-20, 10)
    ax.set_xticks([1, 5, 9, 14, 20, 30])
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "plot_series_convergence.pdf")


def plot_why_series(df: pd.DataFrame, mode: Literal["slice", "worst"] = "slice", beta: float = 10.0) -> None:
    r"""Plot the reliability/cost argument for using the adaptive series for $\mathcal{N}(\alpha,\beta)$."""
    _validate_schema(df)
    target_rel_tol = _target_rel_tol(df)
    time_col = _timing_col(df)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True, height_ratios=[1.15, 1.0])
    axes_list = _as_axes_list(axes)

    if mode == "slice":
        figure_df = df[np.isclose(df["Beta"], beta)].copy()
        title_detail = rf"$\beta={beta:.1f}$"
    else:
        error_rows: list[pd.DataFrame] = []
        time_rows: list[pd.DataFrame] = []
        for impl in IMPLEMENTATIONS:
            impl_df = df[df["Implementation"] == impl]
            error_rows.append(impl_df.loc[impl_df.groupby("Alpha")["Rel_Error"].idxmax()])
            converged_impl = impl_df[_converged_mask(impl_df["Converged"])]
            median_time = converged_impl.groupby("Alpha", as_index=False)[time_col].median()
            template = impl_df.drop_duplicates("Alpha")[["Implementation", "Alpha", "Target_Rel_Tol"]]
            merged = template.merge(median_time, on="Alpha", how="left")
            time_rows.append(merged)
        figure_df = pd.concat(error_rows, ignore_index=True)
        time_df = pd.concat(time_rows, ignore_index=True)
        title_detail = r"worst error over $\beta$"

    for impl in IMPLEMENTATIONS:
        if mode == "slice":
            sample = figure_df[figure_df["Implementation"] == impl]
            _line_with_convergence_markers(axes_list[0], sample, "Alpha", "Rel_Error", impl, IMPL_LABELS[impl])
            time_sample = sample[_converged_mask(sample["Converged"])]
            axes_list[1].plot(
                time_sample["Alpha"],
                time_sample[time_col],
                color=IMPL_COLORS[impl],
                marker=IMPL_MARKERS[impl],
                ms=4,
                lw=1.2,
                label=IMPL_LABELS[impl],
            )
            failed = sample[~_converged_mask(sample["Converged"])]
            if not failed.empty:
                axes_list[1].scatter(
                    failed["Alpha"],
                    failed[time_col],
                    facecolors="none",
                    edgecolors=IMPL_COLORS[impl],
                    marker=IMPL_MARKERS[impl],
                    s=44,
                    linewidths=1.2,
                )
        else:
            sample = figure_df[figure_df["Implementation"] == impl]
            _line_with_convergence_markers(axes_list[0], sample, "Alpha", "Rel_Error", impl, IMPL_LABELS[impl])
            time_sample = time_df[time_df["Implementation"] == impl].sort_values("Alpha")
            axes_list[1].plot(
                time_sample["Alpha"],
                time_sample[time_col],
                color=IMPL_COLORS[impl],
                marker=IMPL_MARKERS[impl],
                ms=4,
                lw=1.2,
                label=IMPL_LABELS[impl],
            )

    _target_line(axes_list[0], target_rel_tol)
    _float64_line(axes_list[0])
    axes_list[0].set_yscale("log")
    axes_list[0].set_ylim(1e-18, 1)
    axes_list[0].set_ylabel(r"$\epsilon_{rel}$")
    axes_list[0].set_title(rf"Reliability and Cost vs $\alpha$ ({title_detail})")
    axes_list[0].grid(True, which="both", ls=":", alpha=0.35)
    axes_list[0].legend(loc="best", fontsize=7.4, framealpha=0.9)

    # if mode == "slice":
    #     inset = axes_list[0].inset_axes([0.6, 0.66, 0.38, 0.30])
    #     inset_alpha = 1e-4
    #     x_values = np.linspace(0.0, 25 * inset_alpha, 300)
    #     y_values = _integrand(x_values, inset_alpha, beta)
    #     inset.plot(x_values / inset_alpha, y_values / np.max(y_values), color="0.15", lw=1.0)
    #     inset.axvline(10.0, color="0.45", linestyle="--", linewidth=0.8)
    #     inset.set_title(r"localized peak near $x=0$", fontsize=7)
    #     inset.set_xlabel(r"$x/\alpha$", fontsize=7)
    #     inset.set_ylabel("scaled integrand", fontsize=7)
    #     inset.tick_params(labelsize=6)

    # To show interpreter-independent cost instead of time, replace ``time_col``
    # with ``Work_Done`` in the bottom-panel data above.
    axes_list[1].set_xscale("log")
    axes_list[1].set_yscale("log")
    axes_list[1].set_xlabel(r"$\alpha$")
    axes_list[1].set_ylabel(r"Median converged runtime ($\mu$s)")
    axes_list[1].grid(True, which="both", ls=":", alpha=0.35)
    axes_list[1].legend(loc="best", fontsize=7.4, framealpha=0.9)

    fig.tight_layout()
    suffix = "" if mode == "slice" else "_worst"
    _save(fig, f"plot_why_series{suffix}.pdf")


def write_benchmark_summary_table(df: pd.DataFrame) -> None:
    """Write the Table I source data as CSV and LaTeX."""
    _validate_schema(df)
    summary_all = (
        df.groupby(["Implementation", "Work_Unit"])
        .agg(
            Max_Rel_Error=("Rel_Error", "max"),
            Med_Rel_Error=("Rel_Error", "median"),
            Converged_Rate=("Converged", "mean"),
            Med_Work_Done=("Work_Done", "median"),
        )
        .reset_index()
    )
    summary_converged = (
        df[_converged_mask(df["Converged"])]
        .groupby(["Implementation", "Work_Unit"])
        .agg(
            Max_Rel_Error_Converged=("Rel_Error", "max"),
            Med_Time_us_Converged=("Time_Median_us", "median"),
            Min_Time_us_Converged=("Time_Min_us", "min"),
        )
        .reset_index()
    )
    table = summary_all.merge(summary_converged, on=["Implementation", "Work_Unit"], how="left")
    table["Implementation"] = table["Implementation"].map(IMPL_LABELS)
    table = table.sort_values("Med_Time_us_Converged")
    csv_path = PLOTS_DIR / "table_benchmark_summary.csv"
    tex_path = PLOTS_DIR / "table_benchmark_summary.tex"
    table.to_csv(csv_path, index=False)
    table.to_latex(tex_path, index=False, float_format="{:.3e}".format)
    print(f"  -> saved {csv_path}")
    print(f"  -> saved {tex_path}")


def main() -> None:
    """Load benchmark data and generate all plots."""
    PLOTS_DIR.mkdir(exist_ok=True)
    csv_path = "benchmark_results.csv"
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    _validate_schema(df)
    print(f"  {len(df)} rows  |  {df['Implementation'].nunique()} implementations\n")

    print("Generating plots ...")
    plot_series_convergence()
    plot_series_error_heatmap(df)
    plot_why_series(df)
    plot_accuracy_vs_speed(df)
    plot_error_heatmaps(df)
    plot_error_vs_alpha(df)
    plot_time_vs_alpha(df)
    plot_timing_boxplots(df)
    plot_time_heatmaps(df)
    plot_summary_bars(df)
    write_benchmark_summary_table(df)
    print("\nDone - all plots saved.")


if __name__ == "__main__":
    main()
