#!/usr/bin/env python3
"""扫描指定文件夹中的图表文件，按模态分类，生成多面板组合图。

自动检测 PNG/PDF/SVG 文件，根据文件名中的模态关键词分类，
然后按 grid 或 list 布局排列为一张组合图。

Usage:
    python layout_results_foldered.py --input-dir results/ --output-dir figures/ --layout grid
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 色方案
ACCENT_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"

# 模态关键词映射
MODALITY_PATTERNS: dict[str, list[str]] = {
    "eeg": ["eeg", "erp", "eeg_", "_eeg"],
    "emg": ["emg", "sEMG", "_emg", "emg_"],
    "eye_tracking": ["eye", "gaze", "fixation", "saccade", "pupil"],
    "behavioral": ["behavior", "rt_", "reaction_time", "accuracy", "response"],
    "physio": ["ecg", "eda", "gsr", "hrv", "scr", "ppg", "resp"],
}

SUPPORTED_EXTENSIONS = {".png", ".pdf", ".svg"}


def detect_modality(filename: str) -> str:
    """根据文件名检测数据模态。"""
    lower = filename.lower()
    for modality, keywords in MODALITY_PATTERNS.items():
        for kw in keywords:
            if kw.lower() in lower:
                return modality
    return "other"


def collect_images(input_dir: Path) -> dict[str, list[Path]]:
    """收集并按模态分类图片文件。

    Args:
        input_dir: 输入目录路径。

    Returns:
        按模态分组的文件路径字典。
    """
    grouped: dict[str, list[Path]] = {}
    for ext in SUPPORTED_EXTENSIONS:
        for f in sorted(input_dir.glob(f"*{ext}")):
            mod = detect_modality(f.name)
            grouped.setdefault(mod, []).append(f)
    return grouped


def load_image(path: Path) -> np.ndarray:
    """加载图片为 numpy 数组。PDF/SVG 通过 matplotlib 渲染。"""
    ext = path.suffix.lower()
    if ext == ".pdf" or ext == ".svg":
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.image import imread as mpl_imread

        fig_temp, ax_temp = plt.subplots(figsize=(5, 4))
        img_data = mpl_imread(str(path))
        plt.close(fig_temp)
        return img_data
    else:
        return np.array(Image.open(path))


def render_grid(
    images: list[tuple[str, np.ndarray]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """以网格布局渲染多面板组合图。

    Args:
        images: (文件名, 图像数组) 列表。
        output_path: 输出文件路径。
        fig_width: 图幅宽度（英寸）。
        fig_height: 图幅高度（英寸）。
        dpi: 分辨率。
    """
    n = len(images)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (name, img) in enumerate(images):
        ax = axes_flat[idx]
        ax.imshow(img)
        ax.set_title(name, fontsize=8, color=ACCENT_BLUE)
        ax.axis("off")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grid saved: {output_path}")


def render_list(
    images: list[tuple[str, np.ndarray]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    """以纵向列表布局渲染多面板组合图。

    Args:
        images: (文件名, 图像数组) 列表。
        output_path: 输出文件路径。
        fig_width: 图幅宽度（英寸）。
        fig_height: 图幅高度（英寸）。
        dpi: 分辨率。
    """
    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height), dpi=dpi)

    for idx, (name, img) in enumerate(images):
        ax = axes[idx] if n > 1 else axes
        ax.imshow(img)
        ax.set_title(name, fontsize=8, color=ACCENT_BLUE)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] List saved: {output_path}")


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="扫描文件夹中的图表文件，按模态分类并生成多面板组合图。"
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="输入图片目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument(
        "--layout",
        choices=["grid", "list"],
        default="grid",
        help="布局方式 (默认: grid)",
    )
    parser.add_argument("--fig-width", type=float, default=10.0, help="图幅宽度 (英寸)")
    parser.add_argument("--fig-height", type=float, default=8.0, help="图幅高度 (英寸)")
    parser.add_argument("--dpi", type=int, default=300, help="输出 DPI")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = collect_images(args.input_dir)

    if not grouped:
        print("[WARN] 未找到任何支持的图片文件。", file=sys.stderr)
        return 0

    for modality, files in grouped.items():
        images: list[tuple[str, np.ndarray]] = []
        for f in files:
            try:
                img = load_image(f)
                images.append((f.stem, img))
            except Exception as exc:
                print(f"[WARN] 无法加载 {f}: {exc}", file=sys.stderr)

        if not images:
            continue

        per_height = args.fig_height / max(1, len(images) ** 0.5)
        out_path = args.output_dir / f"{args.layout}_{modality}.png"

        if args.layout == "grid":
            render_grid(images, out_path, args.fig_width, args.fig_height, args.dpi)
        else:
            render_list(images, out_path, args.fig_width, per_height * len(images), args.dpi)

    print(f"[DONE] 共处理 {sum(len(v) for v in grouped.values())} 个文件，生成 {len(grouped)} 张组合图。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
