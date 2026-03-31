#!/usr/bin/env python3
"""用 matplotlib GridSpec 创建多面板图。

支持指定子图文件列表、布局行列数、面板标签(A,B,C...)、统一色彩方案。

Usage:
    python create_multi_panel.py --images a.png b.png c.png d.png --layout 2x2 --labels A B C D --output figure.pdf
"""

from __future__ import annotations

import argparse
import re
import string
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# 色方案
ACCENT_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"


def parse_layout(layout_str: str) -> tuple[int, int]:
    """解析布局字符串，如 '2x3' -> (2, 3)。

    Args:
        layout_str: 布局描述，格式为 'RxC'。

    Returns:
        (rows, cols) 元组。
    """
    m = re.match(r"(\d+)\s*[xX×]\s*(\d+)", layout_str)
    if not m:
        raise ValueError(f"布局格式无效: {layout_str!r}，应为 'RxC'，如 '2x3'")
    return int(m.group(1)), int(m.group(2))


def load_image(path: Path) -> np.ndarray:
    """加载图片为 numpy 数组。

    Args:
        path: 图片文件路径。

    Returns:
        图像的 numpy 数组。
    """
    return np.array(Image.open(path))


def create_multi_panel(
    images: list[Path],
    layout: tuple[int, int],
    labels: Optional[list[str]],
    output: Path,
    fmt: str,
    figsize: tuple[float, float],
    dpi: int,
    title: Optional[str] = None,
    color_scheme: str = "default",
) -> None:
    """创建多面板组合图。

    Args:
        images: 子图文件路径列表。
        layout: (rows, cols) 布局。
        labels: 面板标签列表（如 ['A', 'B', 'C']）。
        output: 输出文件路径。
        fmt: 输出格式 (pdf/png/svg)。
        figsize: (width, height) 英寸。
        dpi: 分辨率。
        title: 整图标题。
        color_scheme: 色方案 ('default' 使用 ACCENT_BLUE/RED)。
    """
    n = len(images)
    rows, cols = layout

    if labels is None:
        labels = list(string.ascii_uppercase[:n])
    while len(labels) < n:
        labels.append("")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.15, wspace=0.15)

    for idx, (img_path, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        if r >= rows:
            break
        ax = fig.add_subplot(gs[r, c])
        img = load_image(img_path)
        ax.imshow(img)
        ax.axis("off")

        if label:
            ax.set_title(label, fontsize=12, fontweight="bold", color=ACCENT_BLUE, loc="left")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", color=ACCENT_BLUE, y=0.98)

    fig.savefig(output, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    print(f"[OK] 多面板图已保存: {output} ({fmt}, {dpi} DPI, {figsize[0]:.1f}×{figsize[1]:.1f} in)")


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="用 GridSpec 创建多面板图。")
    parser.add_argument("--images", nargs="+", type=Path, required=True, help="子图文件路径列表")
    parser.add_argument(
        "--layout", default="2x2", help="布局 (默认: 2x2)，格式为 RxC"
    )
    parser.add_argument("--labels", nargs="*", help="面板标签 (默认: A,B,C,...)")
    parser.add_argument("--output", type=Path, required=True, help="输出文件路径")
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf", help="输出格式")
    parser.add_argument("--figsize", nargs=2, type=float, default=[10.0, 8.0], metavar=("W", "H"), help="图幅尺寸 (英寸)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI")
    parser.add_argument("--title", type=str, default=None, help="整图标题")
    args = parser.parse_args()

    for p in args.images:
        if not p.is_file():
            print(f"[ERROR] 文件不存在: {p}", file=sys.stderr)
            return 1

    layout = parse_layout(args.layout)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    create_multi_panel(
        images=args.images,
        layout=layout,
        labels=args.labels or None,
        output=output,
        fmt=args.format,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
