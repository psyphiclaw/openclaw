#!/usr/bin/env python3
"""AOI (Area of Interest) analysis for eye-tracking data.

Defines AOIs (rectangular, circular, polygonal), computes fixation
proportions per AOI, transition matrices, and heatmaps.

Usage:
    python aoi_analysis.py gaze.parquet --aoi-config aoi.json --heatmap heatmap.png
    python aoi_analysis.py gaze.parquet --rect "100,200,300,400:face" --rect "400,200,300,400:text"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"


# --- AOI Geometry ---

def point_in_rect(
    x: float, y: float, x0: float, y0: float, w: float, h: float
) -> bool:
    """Check if point (x, y) is inside rectangle (x0, y0, w, h)."""
    return x0 <= x <= x0 + w and y0 <= y <= y0 + h


def point_in_circle(
    x: float, y: float, cx: float, cy: float, r: float
) -> bool:
    """Check if point (x, y) is inside circle (cx, cy, r)."""
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def point_in_polygon(x: float, y: float, vertices: list[tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def classify_aoi(
    x: float, y: float, aois: list[dict],
) -> Optional[str]:
    """Classify a point into AOIs (first match wins). Returns AOI id or None."""
    for aoi in aois:
        shape = aoi.get("shape", "rect")
        if shape == "rect":
            if point_in_rect(x, y, aoi["x"], aoi["y"], aoi["w"], aoi["h"]):
                return aoi["id"]
        elif shape == "circle":
            if point_in_circle(x, y, aoi["cx"], aoi["cy"], aoi["r"]):
                return aoi["id"]
        elif shape == "polygon":
            verts = [(v[0], v[1]) for v in aoi["vertices"]]
            if point_in_polygon(x, y, verts):
                return aoi["id"]
    return None


# --- AOI Config Loading ---

def load_aoi_config(config_path: Path) -> list[dict]:
    """Load AOI definitions from a JSON file.

    Args:
        config_path: Path to JSON config.

    Returns:
        List of AOI definitions.
    """
    with open(config_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "aois" in data:
        return data["aois"]
    else:
        raise ValueError(f"Unexpected AOI config structure in {config_path}")


def parse_rect_args(rect_args: list[str]) -> list[dict]:
    """Parse --rect arguments like 'x,y,w,h:label'.

    Args:
        rect_args: List of 'x,y,w,h:label' strings.

    Returns:
        List of AOI dicts.
    """
    aois: list[dict] = []
    for arg in rect_args:
        parts = arg.split(":")
        coords = parts[0].split(",")
        if len(coords) != 4:
            print(f"Warning: skipping invalid rect: {arg}")
            continue
        label = parts[1] if len(parts) > 1 else f"aoi_{len(aois)}"
        aois.append({
            "id": label,
            "shape": "rect",
            "x": float(coords[0]),
            "y": float(coords[1]),
            "w": float(coords[2]),
            "h": float(coords[3]),
        })
    return aois


# --- Analysis ---

def assign_aois(
    df: pd.DataFrame,
    aois: list[dict],
    x_col: str = "gaze_x",
    y_col: str = "gaze_y",
) -> pd.DataFrame:
    """Assign AOI labels to each gaze sample.

    Args:
        df: Gaze DataFrame.
        aois: List of AOI definitions.
        x_col, y_col: Gaze coordinate columns.

    Returns:
        DataFrame with added 'aoi_id' column.
    """
    result = df.copy()
    gx = pd.to_numeric(result[x_col], errors="coerce")
    gy = pd.to_numeric(result[y_col], errors="coerce")

    aoi_ids = [classify_aoi(xi, yi, aois) for xi, yi in zip(gx, gy)]
    result["aoi_id"] = aoi_ids

    return result


def compute_aoi_stats(
    df: pd.DataFrame,
    aoi_col: str = "aoi_id",
) -> pd.DataFrame:
    """Compute per-AOI statistics: fixation count, proportion, dwell time.

    Args:
        df: DataFrame with aoi_id column.

    Returns:
        DataFrame with AOI statistics.
    """
    total = len(df)
    results: list[dict] = []

    for aoi_id, group in df.groupby(aoi_col):
        if aoi_id is None or pd.isna(aoi_id):
            continue
        results.append({
            "aoi_id": aoi_id,
            "n_samples": len(group),
            "proportion": len(group) / total if total > 0 else 0,
        })

    return pd.DataFrame(results)


def compute_aoi_transition_matrix(
    df: pd.DataFrame,
    aoi_col: str = "aoi_id",
    fixation_col: str = "fixation_index",
) -> tuple[np.ndarray, list[str]]:
    """Compute AOI-to-AOI transition matrix from fixation sequence.

    Args:
        df: DataFrame with aoi_id and fixation_index columns.

    Returns:
        Tuple of (matrix, labels).
    """
    if fixation_col not in df.columns:
        # Use raw AOI sequence
        seq = df[aoi_col].dropna().values
    else:
        # One AOI per fixation (mode)
        seq = df.groupby(fixation_col)[aoi_col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        ).dropna().values

    labels = sorted(set(seq))
    label_map = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for i in range(1, len(seq)):
        if seq[i - 1] in label_map and seq[i] in label_map:
            matrix[label_map[seq[i - 1]]][label_map[seq[i]]] += 1

    return matrix, labels


def generate_heatmap(
    df: pd.DataFrame,
    x_col: str = "gaze_x",
    y_col: str = "gaze_y",
    width: Optional[int] = None,
    height: Optional[int] = None,
    sigma: float = 30.0,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Generate gaze density heatmap.

    Args:
        df: Gaze DataFrame.
        x_col, y_col: Coordinate columns.
        width, height: Image dimensions. Auto-detected from data if None.
        sigma: Gaussian kernel standard deviation (pixels).
        output_path: Save heatmap image to this path.

    Returns:
        2D numpy array (heatmap).
    """
    from scipy.ndimage import gaussian_filter

    gx = pd.to_numeric(df[x_col], errors="coerce").dropna().values
    gy = pd.to_numeric(df[y_col], errors="coerce").dropna().values

    if len(gx) == 0:
        return np.array([])

    # Determine bounds
    if width is None:
        width = int(np.max(gx)) + 1 if np.max(gx) > 0 else 1920
    if height is None:
        height = int(np.max(gy)) + 1 if np.max(gy) > 0 else 1080

    width = max(int(width), 1)
    height = max(int(height), 1)

    # Create histogram
    heatmap, _, _ = np.histogram2d(
        gx, gy, bins=[width, height],
        range=[[0, width], [0, height]],
    )

    # Smooth
    heatmap = gaussian_filter(heatmap.T, sigma=sigma)

    # Save
    if output_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, height / width * 10))
        im = ax.imshow(heatmap, cmap="hot", origin="upper", aspect="auto")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_title("Gaze Density Heatmap")
        plt.colorbar(im, ax=ax, label="Fixation Density")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {output_path}")
        plt.close(fig)

    return heatmap


def plot_aoi_overlay(
    df: pd.DataFrame,
    aois: list[dict],
    x_col: str = "gaze_x",
    y_col: str = "gaze_y",
    output_path: Optional[str] = None,
) -> None:
    """Plot gaze data with AOI boundaries overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(12, 8))

    gx = pd.to_numeric(df[x_col], errors="coerce").dropna()
    gy = pd.to_numeric(df[y_col], errors="coerce").dropna()

    step = max(1, len(gx) // 5000)
    ax.scatter(gx.iloc[::step], gy.iloc[::step], s=1, alpha=0.3, c=COLOR_PRIMARY)
    ax.invert_yaxis()

    colors = ["#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    for i, aoi in enumerate(aois):
        color = colors[i % len(colors)]
        shape = aoi.get("shape", "rect")
        if shape == "rect":
            rect = Rectangle(
                (aoi["x"], aoi["y"]), aoi["w"], aoi["h"],
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.15,
            )
            ax.add_patch(rect)
            ax.text(
                aoi["x"] + aoi["w"] / 2, aoi["y"] - 10,
                aoi["id"], ha="center", fontsize=10, color=color, fontweight="bold",
            )
        elif shape == "circle":
            circle = Circle(
                (aoi["cx"], aoi["cy"]), aoi["r"],
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.15,
            )
            ax.add_patch(circle)
            ax.text(
                aoi["cx"], aoi["cy"] - aoi["r"] - 10,
                aoi["id"], ha="center", fontsize=10, color=color, fontweight="bold",
            )

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title("Gaze Data with AOI Overlay")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"AOI overlay saved to: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="AOI analysis for eye-tracking data.")
    parser.add_argument("input", type=Path, help="Input gaze data (.parquet or .csv).")
    parser.add_argument("--aoi-config", type=Path, default=None, help="AOI JSON config.")
    parser.add_argument("--rect", action="append", default=[], help="Rect AOI: x,y,w,h:label")
    parser.add_argument("--heatmap", type=Path, default=None, help="Output heatmap path.")
    parser.add_argument("--aoi-plot", type=Path, default=None, help="AOI overlay plot path.")
    parser.add_argument("--x-col", type=str, default="gaze_x")
    parser.add_argument("--y-col", type=str, default="gaze_y")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output stats CSV.")
    args = parser.parse_args()

    # Read
    suffix = args.input.suffix.lower()
    df = pd.read_parquet(args.input) if suffix == ".parquet" else pd.read_csv(args.input)

    # Load AOIs
    aois: list[dict] = []
    if args.aoi_config:
        aois = load_aoi_config(args.aoi_config)
    if args.rect:
        aois.extend(parse_rect_args(args.rect))

    if not aois:
        print("Error: No AOIs defined. Use --aoi-config or --rect.")
        sys.exit(1)

    # Assign AOIs
    df = assign_aois(df, aois, x_col=args.x_col, y_col=args.y_col)

    # Stats
    stats = compute_aoi_stats(df)
    print("AOI Statistics:")
    print(stats.to_string(index=False))

    # Transition matrix
    if "fixation_index" in df.columns:
        matrix, labels = compute_aoi_transition_matrix(df)
        if len(labels) > 0:
            print("\nAOI Transition Matrix:")
            print(pd.DataFrame(matrix, index=labels, columns=labels))

    # Plots
    if args.heatmap:
        generate_heatmap(df, x_col=args.x_col, y_col=args.y_col, output_path=str(args.heatmap))

    if args.aoi_plot:
        plot_aoi_overlay(df, aois, x_col=args.x_col, y_col=args.y_col, output_path=str(args.aoi_plot))

    # Save
    if args.output:
        stats.to_csv(args.output, index=False)
        print(f"\nAOI stats saved to: {args.output}")


if __name__ == "__main__":
    main()
