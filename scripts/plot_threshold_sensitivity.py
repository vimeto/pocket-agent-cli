#!/usr/bin/env python3
"""Generate fig_threshold_sensitivity.pdf — a two-panel heatmap showing
robustness of the cost-aware placement policy's (energy_threshold,
latency_threshold) choice.

Reads  data/results/placement_evaluation/threshold_sensitivity.csv
Writes research/figs/new/fig_threshold_sensitivity.pdf (and .png)

Usage:
    uv run python scripts/plot_threshold_sensitivity.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "results" / "placement_evaluation" / \
    "threshold_sensitivity.csv"
OUT_DIR = BASE_DIR / "research" / "figs" / "new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ACM sigconf single-column: 8.5 cm x 6 cm  (~3.35 in x 2.36 in)
FIG_W_IN = 8.5 / 2.54
FIG_H_IN = 6.0 / 2.54

DEFAULT_ENERGY_THR = 2.0
DEFAULT_LATENCY_THR = 1.0


def _apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _pivot(df: pd.DataFrame, value_col: str) -> tuple:
    """Return (grid matrix, energy_vals, latency_vals).

    Rows = energy threshold (y), columns = latency threshold (x).
    """
    pivot = df.pivot(
        index="energy_threshold",
        columns="latency_threshold",
        values=value_col,
    ).sort_index().sort_index(axis=1)
    return pivot.values, pivot.index.values, pivot.columns.values


def _draw_heatmap(ax, mat, e_vals, l_vals, *, cmap, cbar_label,
                  contour_level, vmin=None, vmax=None):
    """Render a heatmap with a single contour line and the default-threshold
    cross overlaid."""
    # pcolormesh expects edges; derive from cell centres.
    def _edges(v):
        v = np.asarray(v)
        d = np.diff(v)
        left = v[0] - d[0] / 2.0
        right = v[-1] + d[-1] / 2.0
        mids = (v[:-1] + v[1:]) / 2.0
        return np.concatenate([[left], mids, [right]])

    le = _edges(l_vals)
    ee = _edges(e_vals)
    im = ax.pcolormesh(le, ee, mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="flat")

    # Contour (use cell centres, not edges).
    L, E = np.meshgrid(l_vals, e_vals)
    cs = ax.contour(L, E, mat, levels=[contour_level],
                    colors="white", linewidths=1.0, linestyles="--")
    try:
        ax.clabel(cs, inline=True, fontsize=6,
                  fmt=f"{contour_level:.2f}")
    except Exception:
        pass

    # Default threshold marker (paper's cited point).
    ax.plot(DEFAULT_LATENCY_THR, DEFAULT_ENERGY_THR,
            marker="x", color="white", mew=1.8, ms=8, zorder=5)
    ax.plot(DEFAULT_LATENCY_THR, DEFAULT_ENERGY_THR,
            marker="x", color="black", mew=0.8, ms=8, zorder=6)

    ax.set_xlabel(r"Latency threshold $\tau_\ell$ (s)")
    ax.set_ylabel(r"Energy threshold $\tau_e$ (J)")
    ax.set_xlim(le[0], le[-1])
    ax.set_ylim(ee[0], ee[-1])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    return im


def main():
    _apply_style()
    df = pd.read_csv(CSV_PATH)

    pct_mat, e_vals, l_vals = _pivot(df, "pct_of_oracle_pass")
    red_mat, _, _ = _pivot(df, "energy_reduction_vs_local")

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(FIG_W_IN * 2.05, FIG_H_IN),
    )

    _draw_heatmap(
        ax_l, pct_mat, e_vals, l_vals,
        cmap="viridis",
        cbar_label="% of oracle pass rate",
        contour_level=0.95,
        vmin=float(pct_mat.min()),
        vmax=float(pct_mat.max()),
    )
    ax_l.set_title("(a) Pass-rate vs oracle")

    _draw_heatmap(
        ax_r, red_mat, e_vals, l_vals,
        cmap="inferno",
        cbar_label="Energy reduction vs always-local",
        contour_level=0.50,
        vmin=float(red_mat.min()),
        vmax=float(red_mat.max()),
    )
    ax_r.set_title("(b) Energy savings vs local")

    fig.tight_layout(pad=0.4)

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig_threshold_sensitivity.{ext}"
        fig.savefig(path)
        print(f"  -> saved {path}")
    plt.close(fig)

    # Diagnostics
    default_row = df[
        (np.isclose(df["energy_threshold"], DEFAULT_ENERGY_THR))
        & (np.isclose(df["latency_threshold"], DEFAULT_LATENCY_THR))
    ]
    if not default_row.empty:
        r = default_row.iloc[0]
        print(f"  Default (e={DEFAULT_ENERGY_THR}, l={DEFAULT_LATENCY_THR}): "
              f"pct_of_oracle_pass={r['pct_of_oracle_pass']:.4f}, "
              f"energy_reduction_vs_local={r['energy_reduction_vs_local']:.4f}")

    # Robustness stats: how many cells within 2pp of 97.5% / 62%?
    n_total = len(df)
    n_pass_close = int(((df["pct_of_oracle_pass"] - 0.975).abs() <= 0.02).sum())
    n_energy_close = int(
        ((df["energy_reduction_vs_local"] - 0.62).abs() <= 0.02).sum()
    )
    n_both_close = int(
        (((df["pct_of_oracle_pass"] - 0.975).abs() <= 0.02)
         & ((df["energy_reduction_vs_local"] - 0.62).abs() <= 0.02)).sum()
    )
    print(f"  Grid size: {n_total}")
    print(f"  Cells within 2pp of 97.5% oracle-pass: {n_pass_close} "
          f"({100 * n_pass_close / n_total:.1f}%)")
    print(f"  Cells within 2pp of 62% energy-reduction: {n_energy_close} "
          f"({100 * n_energy_close / n_total:.1f}%)")
    print(f"  Cells satisfying both: {n_both_close} "
          f"({100 * n_both_close / n_total:.1f}%)")


if __name__ == "__main__":
    main()
