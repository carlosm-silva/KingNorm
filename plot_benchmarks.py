"""
Comparative visualizations of King-function normalization benchmark results.

Reads benchmark_results.csv (produced by speed_tests.py) and generates
a set of publication-ready comparison plots saved as PDFs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# APS / publication styling
plt.rcParams.update({
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
})

IMPL_COLORS = {
    "Series": "#2563eb",       # rich blue
    "QAGS": "#dc2626",         # red
    "Gauss-Legendre": "#16a34a" # green
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

PLOTS_DIR = Path("plots")


def _save(fig, name):
    output_path = PLOTS_DIR / name
    fig.savefig(output_path, bbox_inches="tight")
    print(f"  -> saved {output_path}")
    plt.close(fig)


def _time_median_col(df):
    return "Time_Median_us" if "Time_Median_us" in df.columns else "Time_us"


def _time_candle_available(df):
    needed = ["Time_Min_us", "Time_Q1_us", "Time_Median_us", "Time_Q3_us", "Time_Max_us"]
    return all(col in df.columns for col in needed)


def plot_accuracy_vs_speed(df):
    """Scatter of median relative error vs median wall-clock time, one panel per precision."""
    time_col = _time_median_col(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, prec in zip(axes, ["64-bit", "32-bit"]):
        sub = df[df.Precision == prec]
        agg = sub.groupby("Implementation").agg(
            med_err=("Rel_Error", "median"),
            med_time=(time_col, "median"),
        )
        for impl, row in agg.iterrows():
            err = max(row.med_err, 1e-17)
            ax.scatter(
                row.med_time,
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

    axes[0].set_ylabel("Median relative error")
    fig.suptitle("Accuracy vs Speed Trade-off (grid-aggregated)", fontweight="bold", y=1.02)
    _save(fig, "plot_accuracy_vs_speed.pdf")


def plot_timing_boxplots(df):
    """Box plots of wall-clock time per implementation and precision."""
    time_col = _time_median_col(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, prec in zip(axes, ["64-bit", "32-bit"]):
        sub = df[df.Precision == prec]
        data, labels, colors = [], [], []
        for impl in ["Series", "QAGS", "Gauss-Legendre"]:
            vals = sub[sub.Implementation == impl][time_col].values
            data.append(vals)
            labels.append(IMPL_LABELS[impl])
            colors.append(IMPL_COLORS[impl])

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showfliers=True, flierprops=dict(marker=".", ms=3, alpha=0.3))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
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


def plot_error_heatmaps(df, precision="64-bit"):
    """2-D heatmaps of log10(relative error) over the (alpha, beta) grid."""
    sub = df[df.Precision == precision]
    impls = ["Series", "QAGS", "Gauss-Legendre"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    alphas = np.sort(sub.Alpha.unique())
    betas = np.sort(sub.Beta.unique())

    floor = -17
    vmin, vmax = floor, 0

    for ax, impl in zip(axes, impls):
        s = sub[sub.Implementation == impl]
        grid = np.full((len(betas), len(alphas)), np.nan)
        for _, row in s.iterrows():
            i = np.searchsorted(betas, row.Beta)
            j = np.searchsorted(alphas, row.Alpha)
            val = row.Rel_Error if row.Rel_Error > 0 else 10**floor
            grid[i, j] = np.log10(val)

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
            extent=[np.log10(alphas[0]), np.log10(alphas[-1]), betas[0], betas[-1]],
        )
        ax.set_title(IMPL_LABELS[impl], fontweight="bold")
        ax.set_xlabel(r"log$_{10}$(alpha)")
        if impl == "Series":
            ax.set_ylabel("beta")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"log$_{10}$(relative error)")
    fig.suptitle(f"Relative Error Across Parameter Space ({precision})", fontweight="bold", y=1.03)
    _save(fig, f"plot_error_heatmaps_{precision.replace('-', '')}.pdf")


def plot_error_vs_alpha(df, precision="64-bit"):
    """Line plots of relative error vs alpha for selected beta values."""
    sub = df[df.Precision == precision]
    betas_sel = [1.0, 4.0, 7.0, 10.0]
    machine_eps = np.finfo(np.float64).eps if precision == "64-bit" else np.finfo(np.float32).eps

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    for ax, beta in zip(axes.ravel(), betas_sel):
        for impl in ["Series", "QAGS", "Gauss-Legendre"]:
            s = sub[(sub.Implementation == impl) & (np.isclose(sub.Beta, beta))]
            s = s.sort_values("Alpha")
            err = s.Rel_Error.values.copy()
            err[err == 0] = 1e-17
            ax.plot(
                s.Alpha,
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


def plot_time_vs_alpha(df, precision="64-bit"):
    """Time vs alpha for selected beta values, with candle spread when available."""
    sub = df[df.Precision == precision]
    betas_sel = [1.0, 4.0, 7.0, 10.0]
    candle_ok = _time_candle_available(df)
    median_col = _time_median_col(df)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    for ax, beta in zip(axes.ravel(), betas_sel):
        for impl in ["Series", "QAGS", "Gauss-Legendre"]:
            s = sub[(sub.Implementation == impl) & (np.isclose(sub.Beta, beta))]
            s = s.sort_values("Alpha")

            if candle_ok:
                widths = s.Alpha.values * 0.08
                for j, row in enumerate(s.itertuples(index=False)):
                    ax.vlines(
                        row.Alpha,
                        row.Time_Min_us,
                        row.Time_Max_us,
                        color=IMPL_COLORS[impl],
                        alpha=0.6,
                        linewidth=1.0,
                    )
                    ax.bar(
                        row.Alpha,
                        row.Time_Q3_us - row.Time_Q1_us,
                        bottom=row.Time_Q1_us,
                        width=widths[j],
                        color=IMPL_COLORS[impl],
                        edgecolor="k",
                        linewidth=0.35,
                        alpha=0.35,
                    )
                ax.plot(
                    s.Alpha,
                    s.Time_Median_us,
                    color=IMPL_COLORS[impl],
                    marker=IMPL_MARKERS[impl],
                    ms=3.8,
                    lw=1.15,
                    label=IMPL_LABELS[impl],
                )
            else:
                ax.plot(
                    s.Alpha,
                    s[median_col],
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


def plot_time_heatmaps(df, precision="64-bit"):
    """2-D heatmaps of log10(time in us) over the (alpha, beta) grid."""
    time_col = _time_median_col(df)
    sub = df[df.Precision == precision]
    impls = ["Series", "QAGS", "Gauss-Legendre"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    alphas = np.sort(sub.Alpha.unique())
    betas = np.sort(sub.Beta.unique())

    all_times = np.log10(sub[time_col].values)
    vmin, vmax = all_times.min(), all_times.max()

    for ax, impl in zip(axes, impls):
        s = sub[sub.Implementation == impl]
        grid = np.full((len(betas), len(alphas)), np.nan)
        for _, row in s.iterrows():
            i = np.searchsorted(betas, row.Beta)
            j = np.searchsorted(alphas, row.Alpha)
            grid[i, j] = np.log10(row[time_col])

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            extent=[np.log10(alphas[0]), np.log10(alphas[-1]), betas[0], betas[-1]],
        )
        ax.set_title(IMPL_LABELS[impl], fontweight="bold")
        ax.set_xlabel(r"log$_{10}$(alpha)")
        if impl == "Series":
            ax.set_ylabel("beta")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"log$_{10}$(time / us)")
    fig.suptitle(f"Execution Time Across Parameter Space ({precision})", fontweight="bold", y=1.03)
    _save(fig, f"plot_time_heatmaps_{precision.replace('-', '')}.pdf")


def plot_summary_bars(df):
    """Side-by-side bar chart of aggregate statistics per implementation."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    time_col = _time_median_col(df)

    metrics = [
        ("Median Relative Error", "Rel_Error", "median", True),
        ("Median Time (us)", time_col, "median", True),
        ("Max Relative Error", "Rel_Error", "max", True),
    ]

    for ax, (title, col, agg_fn, log) in zip(axes, metrics):
        x_pos = np.arange(3)
        width = 0.35
        for k, prec in enumerate(["64-bit", "32-bit"]):
            vals = []
            for impl in ["Series", "QAGS", "Gauss-Legendre"]:
                s = df[(df.Precision == prec) & (df.Implementation == impl)]
                val = s[col].agg(agg_fn)
                if val == 0:
                    val = 1e-17
                vals.append(val)
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


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    csv_path = "benchmark_results.csv"
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(
        f"  {len(df)} rows  |  {df.Implementation.nunique()} implementations  "
        f"|  {df.Precision.nunique()} precision levels\n"
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
