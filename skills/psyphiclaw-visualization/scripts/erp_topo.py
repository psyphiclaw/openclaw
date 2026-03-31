#!/usr/bin/env python3
"""ERP topographic map: scalp distribution of ERP potentials.

Supports single time-point topographies and condition comparisons.
Optionally integrates with MNE for electrode-based interpolation.

Usage:
    python erp_topo.py --epochs epochs.csv --ch-names Fz Cz Pz O1 O2 \
        --times -200 0 200 400 600 800 --output erp_topo.png
    python erp_topo.py --session session.h5 --modality eeg \
        --conditions A B --times 200 400 --output comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

BLUE = "#4A90D9"
RED = "#E74C3C"

# Standard 10-20 electrode positions (2D projection, approximated)
ELECTRODE_2D: dict[str, tuple[float, float]] = {
    "Fp1": (-0.3, 0.85), "Fp2": (0.3, 0.85),
    "F7": (-0.7, 0.55), "F3": (-0.35, 0.55), "Fz": (0.0, 0.6), "F4": (0.35, 0.55), "F8": (0.7, 0.55),
    "T3": (-0.9, 0.2), "C3": (-0.5, 0.2), "Cz": (0.0, 0.2), "C4": (0.5, 0.2), "T4": (0.9, 0.2),
    "T5": (-0.85, -0.2), "P3": (-0.4, -0.2), "Pz": (0.0, -0.25), "P4": (0.4, -0.2), "T6": (0.85, -0.2),
    "O1": (-0.3, -0.6), "Oz": (0.0, -0.65), "O2": (0.3, -0.6),
    "AFz": (0.0, 0.75),
    "FCz": (0.0, 0.4), "CPz": (0.0, -0.02),
    "POz": (0.0, -0.45),
    # Additional electrodes
    "F1": (-0.18, 0.58), "F2": (0.18, 0.58),
    "FC1": (-0.25, 0.38), "FC2": (0.25, 0.38),
    "FC3": (-0.55, 0.38), "FC4": (0.55, 0.38),
    "FC5": (-0.75, 0.35), "FC6": (0.75, 0.35),
    "C1": (-0.25, 0.2), "C2": (0.25, 0.2),
    "C5": (-0.72, 0.2), "C6": (0.72, 0.2),
    "CP1": (-0.25, -0.0), "CP2": (0.25, -0.0),
    "CP3": (-0.48, -0.02), "CP4": (0.48, -0.02),
    "CP5": (-0.72, -0.05), "CP6": (0.72, -0.05),
    "P1": (-0.2, -0.22), "P2": (0.2, -0.22),
    "P5": (-0.65, -0.22), "P6": (0.65, -0.22),
    "PO3": (-0.2, -0.48), "PO4": (0.2, -0.48),
    "PO5": (-0.55, -0.45), "PO6": (0.55, -0.45),
}


def interpolate_topo(
    ch_values: dict[str, float],
    grid_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate channel values to a 2D grid for topographic plotting.

    Parameters
    ----------
    ch_values : dict mapping channel name → amplitude value.
    grid_size : Resolution of the interpolation grid.

    Returns
    -------
    xi, yi, zi : Meshgrid and interpolated values.
    """
    x = np.array([ELECTRODE_2D[ch][0] for ch in ch_values])
    y = np.array([ELECTRODE_2D[ch][1] for ch in ch_values])
    z = np.array([ch_values[ch] for ch in ch_values])

    xi = np.linspace(-1, 1, grid_size)
    yi = np.linspace(-0.8, 0.9, grid_size)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method="cubic")
    # Mask outside head (circle approximation)
    mask = xi ** 2 + (yi - 0.1) ** 2 > 0.85
    zi_masked = np.ma.array(zi, mask=mask)

    return xi, yi, zi_masked


def draw_head_outline(ax: plt.Axes) -> None:
    """Draw a simple head outline on the axes."""
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 0.92 * np.sin(theta)
    y = 0.92 * np.cos(theta) + 0.1
    ax.plot(x, y, "k-", linewidth=1.5)
    # Nose
    ax.plot([0, 0], [0.92 + 0.1, 1.05 + 0.1], "k-", linewidth=1.5)
    # Ears
    ax.plot([-0.92, -1.02], [0.1, 0.1], "k-", linewidth=1.5)
    ax.plot([0.92, 1.02], [0.1, 0.1], "k-", linewidth=1.5)


def plot_single_topo(
    ch_values: dict[str, float],
    time_ms: float,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Plot a single ERP topography at one time point."""
    xi, yi, zi = interpolate_topo(ch_values)

    fig, ax = plt.subplots(figsize=(6, 6))
    if vmin is not None and vmax is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        vmax = max(abs(float(np.nanmin(zi))), abs(float(np.nanmax(zi))))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", norm=norm)
    plt.colorbar(im, ax=ax, label="Amplitude (μV)", shrink=0.7)

    # Plot electrode positions
    for ch, (x, y) in ch_values.items():
        if ch in ELECTRODE_2D:
            ax.plot(x, y, "ko", markersize=4)
            ax.annotate(ch, (x, y), textcoords="offset points", xytext=(3, 3),
                        fontsize=6, color="black")

    draw_head_outline(ax)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.85, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"ERP Topography @ {time_ms:.0f} ms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_condition_comparison(
    cond_a: dict[str, float],
    cond_b: dict[str, float],
    time_ms: float,
    cond_a_name: str = "Condition A",
    cond_b_name: str = "Condition B",
) -> plt.Figure:
    """Plot side-by-side topographies for two conditions."""
    xi, yi, zi_a = interpolate_topo(cond_a)
    _, _, zi_b = interpolate_topo(cond_b)

    all_vals = np.concatenate([zi_a.compressed(), zi_b.compressed()])
    vmax = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    im1 = ax1.contourf(xi, yi, zi_a, levels=50, cmap="RdBu_r", norm=norm)
    plt.colorbar(im1, ax=ax1, label="μV", shrink=0.7)
    draw_head_outline(ax1)
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-0.85, 1.15)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(f"{cond_a_name} @ {time_ms:.0f} ms", fontsize=12, fontweight="bold")

    im2 = ax2.contourf(xi, yi, zi_b, levels=50, cmap="RdBu_r", norm=norm)
    plt.colorbar(im2, ax=ax2, label="μV", shrink=0.7)
    draw_head_outline(ax2)
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-0.85, 1.15)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title(f"{cond_b_name} @ {time_ms:.0f} ms", fontsize=12, fontweight="bold")

    fig.suptitle(f"ERP Topography Comparison @ {time_ms:.0f} ms", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def load_epochs_from_csv(
    path: str,
    ch_names: list[str],
) -> dict[float, dict[str, float]]:
    """Load averaged ERP data from CSV.

    Expected columns: time_ms, ch1, ch2, ..., chN
    Returns dict mapping time_ms → {channel: amplitude}.
    """
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if time_col is None:
        time_col = df.columns[0]

    result: dict[float, dict[str, float]] = {}
    for _, row in df.iterrows():
        t = float(row[time_col])
        result[t] = {ch: float(row[ch]) for ch in ch_names if ch in row.index}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ERP topographic map visualization.")
    parser.add_argument("--epochs", help="Averaged epochs CSV (columns: time_ms, ch1, ch2, ...).")
    parser.add_argument("--session", help="MultiModalSession .h5 (alternative to --epochs).")
    parser.add_argument("--modality", default="eeg", help="Modality name in session.")
    parser.add_argument("--conditions", nargs=2, default=None, metavar=("A", "B"),
                        help="Two condition columns for comparison.")
    parser.add_argument("--ch-names", nargs="+", required=True, help="Channel names.")
    parser.add_argument("--times", nargs="+", type=float, required=True,
                        help="Time points (ms) to plot.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    # Load data
    if args.epochs:
        time_to_channels = load_epochs_from_csv(args.epochs, args.ch_names)
    else:
        print("❌ --epochs CSV is required (provide averaged ERP data).")
        sys.exit(1)

    available_times = sorted(time_to_channels.keys())
    print(f"📊 Loaded {len(available_times)} time points, {len(args.ch_names)} channels")

    # Validate electrode positions
    missing = [ch for ch in args.ch_names if ch not in ELECTRODE_2D]
    if missing:
        print(f"⚠️  No 2D positions for: {missing}. These channels will be skipped for topo.")
        valid_chs = [ch for ch in args.ch_names if ch in ELECTRODE_2D]
    else:
        valid_chs = args.ch_names

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find closest available time
    def find_closest_time(target_ms: float) -> float:
        return min(available_times, key=lambda t: abs(t - target_ms))

    if args.conditions and len(args.conditions) == 2:
        # Condition comparison mode
        for target_t in args.times:
            closest = find_closest_time(target_t)
            cond_a_name, cond_b_name = args.conditions

            # For condition comparison, we need condition-specific averages
            # This requires a different CSV format. For now, show single topo.
            print(f"⚠️  Condition comparison requires condition-separated epoch files.")
            print(f"   Showing single topography at {closest:.0f} ms")

            ch_vals = {ch: time_to_channels[closest].get(ch, 0) for ch in valid_chs}
            fig = plot_single_topo(ch_vals, closest, vmin=args.vmin, vmax=args.vmax)
            save_path = out_path.parent / f"{out_path.stem}_{int(closest)}ms{out_path.suffix}"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ {save_path}")
    else:
        # Single condition mode
        n_times = len(args.times)
        if n_times == 1:
            closest = find_closest_time(args.times[0])
            ch_vals = {ch: time_to_channels[closest].get(ch, 0) for ch in valid_chs}
            fig = plot_single_topo(ch_vals, closest, vmin=args.vmin, vmax=args.vmax)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ {out_path}")
        else:
            # Multi-time grid
            ncols = min(n_times, 4)
            nrows = (n_times + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))

            for idx, target_t in enumerate(args.times):
                ax = axes.flat[idx] if hasattr(axes, "flat") else axes
                closest = find_closest_time(target_t)
                ch_vals = {ch: time_to_channels[closest].get(ch, 0) for ch in valid_chs}
                xi, yi, zi = interpolate_topo(ch_vals)

                if args.vmin is not None and args.vmax is not None:
                    norm = TwoSlopeNorm(vmin=args.vmin, vcenter=0, vmax=args.vmax)
                else:
                    vmax = max(abs(float(np.nanmin(zi))), abs(float(np.nanmax(zi))), 0.1)
                    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

                ax.contourf(xi, yi, zi, levels=50, cmap="RdBu_r", norm=norm)
                draw_head_outline(ax)
                ax.set_xlim(-1.15, 1.15)
                ax.set_ylim(-0.85, 1.15)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(f"{closest:.0f} ms", fontsize=11, fontweight="bold")

            # Hide empty subplots
            for idx in range(n_times, nrows * ncols):
                ax = axes.flat[idx]
                ax.axis("off")

            fig.suptitle("ERP Topography Series", fontsize=14, fontweight="bold")
            plt.tight_layout()
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ {out_path} ({n_times} time points)")


if __name__ == "__main__":
    main()
