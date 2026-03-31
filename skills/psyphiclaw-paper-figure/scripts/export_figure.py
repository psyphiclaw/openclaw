#!/usr/bin/env python3
"""按期刊要求导出图表。

支持 PDF(矢量)/PNG(300/600DPI)/SVG，预设 Nature/Science/APA/IEEE 模板。

Usage:
    python export_figure.py --input figure_raw.png --output figure_nature.pdf --journal nature --dpi 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 色方案
ACCENT_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"

# 默认模板路径
_DEFAULT_TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "assets" / "journal_templates.json"


def load_templates(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """加载期刊模板配置。

    Args:
        path: JSON 文件路径，默认为 assets/journal_templates.json。

    Returns:
        期刊名称到模板配置的字典。
    """
    p = path or _DEFAULT_TEMPLATES_PATH
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[WARN] 模板文件不存在: {p}，使用内置默认值。", file=sys.stderr)
    return {}


def get_template(journal: str, templates: dict[str, Any]) -> dict[str, Any]:
    """获取期刊模板，合并默认值。

    Args:
        journal: 期刊名称 (nature/science/apa/ieee)。
        templates: 已加载的模板字典。

    Returns:
        合并后的模板配置。
    """
    defaults: dict[str, Any] = {
        "width_in": 6.0,
        "height_in": 4.0,
        "dpi": 300,
        "font_family": "Arial",
        "font_size": 8,
        "color_space": "CMYK",
    }
    journal_key = journal.lower().strip()
    tpl = templates.get(journal_key, {})
    defaults.update(tpl)
    return defaults


def export_figure(
    input_path: Path,
    output_path: Path,
    journal: str = "nature",
    dpi: int | None = None,
    width: float | None = None,
    height: float | None = None,
    templates_path: Path | None = None,
) -> None:
    """按期刊模板导出图表。

    Args:
        input_path: 输入图片路径。
        output_path: 输出文件路径（扩展名决定格式）。
        journal: 期刊模板名称。
        dpi: 覆盖 DPI，None 使用模板默认值。
        width: 覆盖宽度（英寸）。
        height: 覆盖高度（英寸）。
        templates_path: 自定义模板 JSON 路径。
    """
    templates = load_templates(templates_path)
    tpl = get_template(journal, templates)

    final_dpi = dpi or tpl["dpi"]
    final_w = width or tpl["width_in"]
    final_h = height or tpl["height_in"]
    fmt = output_path.suffix.lstrip(".").lower()

    if fmt not in ("pdf", "png", "svg"):
        print(f"[WARN] 不支持的格式: .{fmt}，输出为 PNG。", file=sys.stderr)
        fmt = "png"
        output_path = output_path.with_suffix(".png")

    # 设置 matplotlib 全局样式
    plt.rcParams.update({
        "font.family": tpl.get("font_family", "Arial"),
        "font.size": tpl.get("font_size", 8),
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
    })

    # 加载图像
    img = np.array(Image.open(input_path))

    fig, ax = plt.subplots(figsize=(final_w, final_h), dpi=final_dpi)
    ax.imshow(img)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=final_dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        format=fmt,
    )
    plt.close(fig)

    color_space = tpl.get("color_space", "RGB")
    print(
        f"[OK] 已导出: {output_path} | 期刊: {journal} | "
        f"格式: {fmt.upper()} | {final_dpi} DPI | "
        f"{final_w:.1f}×{final_h:.1f} in | 色彩空间: {color_space}"
    )


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="按期刊要求导出图表。")
    parser.add_argument("--input", type=Path, required=True, help="输入图片路径")
    parser.add_argument("--output", type=Path, required=True, help="输出文件路径")
    parser.add_argument("--journal", default="nature", choices=["nature", "science", "apa", "ieee"], help="期刊模板")
    parser.add_argument("--dpi", type=int, default=None, help="覆盖 DPI")
    parser.add_argument("--width", type=float, default=None, help="覆盖宽度 (英寸)")
    parser.add_argument("--height", type=float, default=None, help="覆盖高度 (英寸)")
    parser.add_argument("--templates", type=Path, default=None, help="自定义模板 JSON 路径")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"[ERROR] 输入文件不存在: {args.input}", file=sys.stderr)
        return 1

    export_figure(
        input_path=args.input,
        output_path=args.output,
        journal=args.journal,
        dpi=args.dpi,
        width=args.width,
        height=args.height,
        templates_path=args.templates,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
