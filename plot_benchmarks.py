"""Generate comparative visualizations from benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

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
    "Series": "Series (NumPy)",
    "QAGS": "QAGS",
    "Gauss-Legendre": "Gauss-Legendre",
}
IMPLEMENTATIONS = ("Series", "QAGS", "Gauss-Legendre")
PRECISIONS = ("64-bit", "32-bit")
PLOTS_DIR = Path("plots")


def _as_axes_list(axes: object) -> list[Axes]:
    """Return Matplotlib axes as a flat typed list.

    Parameters
    ----------
    axes : object
        Single axes object or array-like axes result from ``plt.subplots``.

    Returns
    -------
    list[Axes]
        Flattened axes list.
    """
    return [cast("Axes", ax) for ax in np.asarray(axes, dtype=object).ravel()]


def _save(fig: Figure, name: str) -> None:
    """Save and close a figure.

    Parameters
    ----------
    fig : Figure
        Figure instance to save.
    name : str
        Output filename inside the plots directory.
    """
    output_path = PLOTS_DIR / name
    fig.savefig(output_path, bbox_inches="tight")
    print(f"  -> saved {output_path}")
    plt.close(fig)


def _time_median_col(df: pd.DataFrame) -> str:
    """Return the preferred median-time column name.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.

    Returns
    -------
    str
        ``Time_Median_us`` if present, else ``Time_us``.
    """
    return "Time_Median_us" if "Time_Median_us" in df.columns else "Time_us"


def _time_candle_available(df: pd.DataFrame) -> bool:
    """Check whether candle-style timing spread columns are available.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.

    Returns
    -------
    bool
        ``True`` when all candle columns exist.
    """
    needed = ["Time_Min_us", "Time_Q1_us", "Time_Median_us", "Time_Q3_us", "Time_Max_us"]
    return all(col in df.columns for col in needed)


def plot_accuracy_vs_speed(df: pd.DataFrame) -> None:
    """Plot median relative error vs median time for each precision level.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.
    """
    time_col = _time_median_col(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    axes_list = _as_axes_list(axes)

    for ax, prec in zip(axes_list, PRECISIONS, strict=True):
        sub = df[df["Precision"] == prec]
        agg = sub.groupby("Implementation").agg(
            med_err=("Rel_Error", "median"),
            med_time=(time_col, "median"),
        )
        for impl in agg.index:
            err = max(float(agg.loc[impl, "med_err"]), 1e-17)
            ax.scatter(
                float(agg.loc[impl, "med_time"]),
                err,
                color=IMPL_COLORS[impl],
                marker=IMPL_MARKERS[impl],
                s=120,
                zorder=5,
                edgecolors="k",
                linewidths=0.5,
                label=IMPL_LABELS[impl],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Median wall-clock time (us)")
        ax.set_title(prec)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    axes_list[0].set_ylabel("Median relative error")
    fig.suptitle("Accuracy vs Speed Trade-off (grid-aggregated)", fontweight="bold", y=1.02)
    _save(fig, "plot_accuracy_vs_speed.pdf")


def plot_timing_boxplots(df: pd.DataFrame) -> None:
    """Plot time distributions per implementation and precision.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.
    """
    time_col = _time_median_col(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    axes_list = _as_axes_list(axes)

    for ax, prec in zip(axes_list, PRECISIONS, strict=True):
        sub = df[df["Precision"] == prec]
        data: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []
        for impl in IMPLEMENTATIONS:
            vals = sub[sub["Implementation"] == impl][time_col].to_numpy()
            data.append(vals)
            labels.append(IMPL_LABELS[impl])
            colors.append(IMPL_COLORS[impl])

        box_props = dict(marker=".", ms=3, alpha=0.3)
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True, flierprops=box_props)
        for patch, color in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_yscale("log")
        ax.set_ylabel("Wall-clock time (us)" if prec == "64-bit" else "")
        ax.set_title(prec)
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle("Execution Time Distribution Across Parameter Grid", fontweight="bold", y=1.02)
    _save(fig, "plot_timing_boxplots.pdf")


def plot_error_heatmaps(df: pd.DataFrame, precision: str = "64-bit") -> None:
    """Plot log-scaled relative-error heatmaps over ``(alpha, beta)``.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.
    precision : str, optional
        Precision label to filter on, by default ``"64-bit"``.
    """
    sub = df[df["Precision"] == precision]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes_list = _as_axes_list(axes)
    alphas = np.sort(sub["Alpha"].unique())
    betas = np.sort(sub["Beta"].unique())

    floor = -17
    vmin, vmax = floor, 0
    images: list[AxesImage] = []

    for ax, impl in zip(axes_list, IMPLEMENTATIONS, strict=True):
        sample = sub[sub["Implementation"] == impl]
        grid = np.full((len(betas), len(alphas)), np.nan)
        for alpha, beta, rel_error in zip(sample["Alpha"], sample["Beta"], sample["Rel_Error"], strict=True):
            i = int(np.searchsorted(betas, beta))
            j = int(np.searchsorted(alphas, alpha))
            val = rel_error if rel_error > 0 else 10**floor
            grid[i, j] = np.log10(val)

        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
            extent=[np.log10(alphas[0]), np.log10(alphas[-1]), betas[0], betas[-1]],
        )
        images.append(image)
        ax.set_title(IMPL_LABELS[impl], fontweight="bold")
        ax.set_xlabel(r"log$_{10}$(alpha)")
        if impl == "Series":
            ax.set_ylabel("beta")

    cbar = fig.colorbar(images[-1], ax=axes_list, shrink=0.85, pad=0.02)
    cbar.set_label(r"log$_{10}$(relative error)")
    fig.suptitle(f"Relative Error Across Parameter Space ({precision})", fontweight="bold", y=1.03)
    _save(fig, f"plot_error_heatmaps_{precision.replace('-', '')}.pdf")


def plot_error_vs_alpha(df: pd.DataFrame, precision: str = "64-bit") -> None:
    """Plot relative error vs alpha for selected beta slices.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.
    precision : str, optional
        Precision label to filter on, by default ``"64-bit"``.
    """
    sub = df[df["Precision"] == precision]
    betas_sel = [1.0, 4.0, 7.0, 10.0]
    machine_eps = np.finfo(np.float64).eps if precision == "64-bit" else np.finfo(np.float32).eps

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes_list = _as_axes_list(axes)

    for ax, beta in zip(axes_list, betas_sel, strict=True):
        for impl in IMPLEMENTATIONS:
            sample = sub[(sub["Implementation"] == impl) & (np.isclose(sub["Beta"], beta))]
            sample = sample.sort_values("Alpha")
            err = sample["Rel_Error"].to_numpy().copy()
            err[err == 0] = 1e-17
            ax.plot(
                sample["Alpha"],
                err,
                color=IMPL_COLORS[impl],
                marker=IMPL_MARKERS[impl],
                ms=4,
                lw=1.2,
                label=IMPL_LABELS[impl],
            )

        ax.axhline(
            machine_eps,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label="Machine precision",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 10)
        ax.set_title(rf"beta = {beta:.1f}")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.85)

    fig.supxlabel("alpha", fontsize=12)
    fig.supylabel("Relative error", fontsize=12)
    fig.suptitle(f"Relative Error vs alpha at Selected beta Slices ({precision})", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, f"plot_error_vs_alpha_{precision.replace('-', '')}.pdf")


def plot_time_vs_alpha(df: pd.DataFrame, precision: str = "64-bit") -> None:
    """Plot time vs alpha for selected beta slices.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.
    precision : str, optional
        Precision label to filter on, by default ``"64-bit"``.
    """
    sub = df[df["Precision"] == precision]
    betas_sel = [1.0, 4.0, 7.0, 10.0]
    candle_ok = _time_candle_available(df)
    median_col = _time_median_col(df)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes_list = _as_axes_list(axes)

    for ax, beta in zip(axes_list, betas_sel, strict=True):
        for impl in IMPLEMENTATIONS:
            sample = sub[(sub["Implementation"] == impl) & (np.isclose(sub["Beta"], beta))]
            sample = sample.sort_values("Alpha")

            if candle_ok:
                alphas = sample["Alpha"].to_numpy()
                widths = alphas * 0.08
                for j, (alpha, min_time, q1_time, _median_time, q3_time, max_time) in enumerate(
                    zip(
                        sample["Alpha"],
                        sample["Time_Min_us"],
                        sample["Time_Q1_us"],
                        sample["Time_Median_us"],
                        sample["Time_Q3_us"],
                        sample["Time_Max_us"],
                        strict=True,
                    )
                ):
                    ax.vlines(
                        alpha,
                        min_time,
                        max_time,
                        color=IMPL_COLORS[impl],
                        alpha=0.6,
                        linewidth=1.0,
                    )
                    ax.bar(
                        alpha,
                        q3_time - q1_time,
                        bottom=q1_time,
                        width=widths[j],
                        color=IMPL_COLORS[impl],
                        edgecolor="k",
                        linewidth=0.35,
                        alpha=0.35,
                    )
                ax.plot(
                    alphas,
                    sample["Time_Median_us"],
                    color=IMPL_COLORS[impl],
                    marker=IMPL_MARKERS[impl],
                    ms=3.8,
                    lw=1.15,
                    label=IMPL_LABELS[impl],
                )
            else:
                ax.plot(
                    sample["Alpha"],
                    sample[median_col],
                    color=IMPL_COLORS[impl],
                    marker=IMPL_MARKERS[impl],
                    ms=4,
                    lw=1.2,
                    label=IMPL_LABELS[impl],
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"beta = {beta:.1f}")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.85)

    fig.supxlabel("alpha", fontsize=12)
    fig.supylabel("Wall-clock time (us)", fontsize=12)
    fig.suptitle(f"Execution Time vs alpha at Selected beta Slices ({precision})", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, f"plot_time_vs_alpha_{precision.replace('-', '')}.pdf")


def plot_time_heatmaps(df: pd.DataFrame, precision: str = "64-bit") -> None:
    """Plot log-scaled timing heatmaps over ``(alpha, beta)``.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.
    precision : str, optional
        Precision label to filter on, by default ``"64-bit"``.
    """
    time_col = _time_median_col(df)
    sub = df[df["Precision"] == precision]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes_list = _as_axes_list(axes)
    alphas = np.sort(sub["Alpha"].unique())
    betas = np.sort(sub["Beta"].unique())

    all_times = np.log10(sub[time_col].to_numpy())
    vmin, vmax = float(all_times.min()), float(all_times.max())
    images: list[AxesImage] = []

    for ax, impl in zip(axes_list, IMPLEMENTATIONS, strict=True):
        sample = sub[sub["Implementation"] == impl]
        grid = np.full((len(betas), len(alphas)), np.nan)
        for alpha, beta, time_value in zip(sample["Alpha"], sample["Beta"], sample[time_col], strict=True):
            i = int(np.searchsorted(betas, beta))
            j = int(np.searchsorted(alphas, alpha))
            grid[i, j] = np.log10(time_value)

        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            extent=[np.log10(alphas[0]), np.log10(alphas[-1]), betas[0], betas[-1]],
        )
        images.append(image)
        ax.set_title(IMPL_LABELS[impl], fontweight="bold")
        ax.set_xlabel(r"log$_{10}$(alpha)")
        if impl == "Series":
            ax.set_ylabel("beta")

    cbar = fig.colorbar(images[-1], ax=axes_list, shrink=0.85, pad=0.02)
    cbar.set_label(r"log$_{10}$(time / us)")
    fig.suptitle(f"Execution Time Across Parameter Space ({precision})", fontweight="bold", y=1.03)
    _save(fig, f"plot_time_heatmaps_{precision.replace('-', '')}.pdf")


def plot_summary_bars(df: pd.DataFrame) -> None:
    """Plot aggregate bars for error and speed comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data frame.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes_list = _as_axes_list(axes)
    time_col = _time_median_col(df)

    metrics: list[tuple[str, str, str, bool]] = [
        ("Median Relative Error", "Rel_Error", "median", True),
        ("Median Time (us)", time_col, "median", True),
        ("Max Relative Error", "Rel_Error", "max", True),
    ]

    for ax, (title, col, agg_fn, log) in zip(axes_list, metrics, strict=True):
        x_pos = np.arange(3)
        width = 0.35
        for k, prec in enumerate(PRECISIONS):
            vals: list[float] = []
            for impl in IMPLEMENTATIONS:
                sample = df[(df["Precision"] == prec) & (df["Implementation"] == impl)]
                val = float(sample[col].agg(agg_fn))
                vals.append(1e-17 if val == 0 else val)
            ax.bar(
                x_pos + k * width,
                vals,
                width,
                label=prec,
                color=["#60a5fa", "#f87171"][k],
                edgecolor="k",
                linewidth=0.4,
                alpha=0.8,
            )
        ax.set_xticks(x_pos + width / 2)
        ax.set_xticklabels(["Series", "QAGS", "G-L"], fontsize=9)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle("Aggregate Performance Comparison", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "plot_summary_bars.pdf")


def main() -> None:
    """Load benchmark data and generate all plots."""
    PLOTS_DIR.mkdir(exist_ok=True)
    csv_path = "benchmark_results.csv"
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(
        f"  {len(df)} rows  |  {df['Implementation'].nunique()} implementations  "
        f"|  {df['Precision'].nunique()} precision levels\n"
    )

    print("Generating plots ...")
    plot_accuracy_vs_speed(df)
    plot_timing_boxplots(df)
    plot_error_heatmaps(df, "64-bit")
    plot_error_heatmaps(df, "32-bit")
    plot_error_vs_alpha(df, "64-bit")
    plot_error_vs_alpha(df, "32-bit")
    plot_time_vs_alpha(df, "64-bit")
    plot_time_vs_alpha(df, "32-bit")
    plot_time_heatmaps(df, "64-bit")
    plot_time_heatmaps(df, "32-bit")
    plot_summary_bars(df)
    print("\nDone - all plots saved.")


if __name__ == "__main__":
    main()
