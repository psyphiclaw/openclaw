#!/usr/bin/env python3
"""生成汇总报告。组级别统计、批量导出图表、HTML 仪表盘索引页。

Usage:
    python batch_report.py --project-dir data/my_exp/ --output-dir reports/ --title "My Experiment"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 色方案
ACCENT_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"


def collect_results(results_dir: Path) -> dict[str, list[Path]]:
    """收集所有分析结果文件。

    Args:
        results_dir: 分析结果目录。

    Returns:
        按被试分组的结果文件字典。
    """
    grouped: dict[str, list[Path]] = {}
    for subject_dir in sorted(results_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subj = subject_dir.name
        files = []
        for f in subject_dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                files.append(f)
        if files:
            grouped[subj] = files
    return grouped


def compute_group_stats(results_dir: Path, subjects: list[str]) -> pd.DataFrame:
    """计算组级别描述统计。

    尝试读取 parquet 文件并合并统计。

    Args:
        results_dir: 结果目录。
        subjects: 被试 ID 列表。

    Returns:
        组统计 DataFrame。
    """
    all_data: list[pd.DataFrame] = []
    for subj in subjects:
        subj_dir = results_dir / subj
        for parquet_file in subj_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                df["_subject_id"] = subj
                df["_source_file"] = parquet_file.stem
                all_data.append(df)
            except Exception:
                continue

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=np.number).columns.tolist()
    exclude = {"_subject_id"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    if not numeric_cols:
        return pd.DataFrame()

    stats = combined.groupby("_subject_id")[numeric_cols].agg(["mean", "std", "count"])
    return stats


def generate_html_report(
    project_dir: Path,
    output_dir: Path,
    title: str,
    stats_df: pd.DataFrame | None,
    results: dict[str, list[Path]],
) -> None:
    """生成 HTML 仪表盘索引页。

    Args:
        project_dir: 项目目录。
        output_dir: 输出目录。
        title: 报告标题。
        stats_df: 组统计 DataFrame。
        results: 按被试分组的结果文件。
    """
    html_parts: list[str] = [
        "<!DOCTYPE html>",
        "<html lang='zh-CN'>",
        "<head>",
        f"<title>{title} - 汇总报告</title>",
        "<meta charset='utf-8'>",
        "<style>",
        f"body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}",
        f"h1 {{ color: {ACCENT_BLUE}; }}",
        f"h2 {{ color: {ACCENT_BLUE}; border-bottom: 2px solid {ACCENT_BLUE}; padding-bottom: 5px; }}",
        f".subject-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; "
        f"background: #f9f9f9; }}",
        f".subject-card h3 {{ color: {ACCENT_BLUE}; margin-top: 0; }}",
        f".error {{ color: {ACCENT_RED}; }}",
        f"table {{ border-collapse: collapse; width: 100%; }}",
        f"th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}",
        f"th {{ background-color: {ACCENT_BLUE}; color: white; }}",
        f"tr:nth-child(even) {{ background-color: #f2f2f2; }}",
        f".summary-box {{ background: #e8f4f8; border-left: 4px solid {ACCENT_BLUE}; padding: 10px 15px; margin: 15px 0; }}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>📊 {title}</h1>",
        f"<p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # 摘要
    n_subjects = len(results)
    n_files = sum(len(v) for v in results.values())
    html_parts.append(
        f'<div class="summary-box">'
        f"<strong>被试数:</strong> {n_subjects} | "
        f"<strong>结果文件:</strong> {n_files} | "
        f"<strong>项目目录:</strong> {project_dir}"
        f"</div>"
    )

    # 组统计表
    if stats_df is not None and not stats_df.empty:
        html_parts.append("<h2>组级别统计</h2>")
        html_parts.append(stats_df.to_html(classes="stats-table", float_format="%.3f"))

    # 被试详情
    html_parts.append("<h2>被试详情</h2>")
    for subj_id, files in sorted(results.items()):
        html_parts.append(f'<div class="subject-card">')
        html_parts.append(f"<h3>被试 {subj_id} ({len(files)} 个文件)</h3>")
        html_parts.append("<ul>")
        for f in files:
            size_kb = f.stat().st_size / 1024
            html_parts.append(f"<li>{f.name} ({size_kb:.1f} KB)</li>")
        html_parts.append("</ul></div>")

    # 复制图表到输出目录
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied_figures: list[Path] = []
    for subj_id, files in results.items():
        for f in files:
            if f.suffix.lower() in (".png", ".svg", ".pdf", ".jpg", ".jpeg"):
                dest = figures_dir / f"{subj_id}_{f.name}"
                shutil.copy2(f, dest)
                copied_figures.append(dest)

    if copied_figures:
        html_parts.append("<h2>图表</h2>")
        html_parts.append("<div style='display:flex;flex-wrap:wrap;gap:10px;'>")
        for fig in copied_figures:
            if fig.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rel = f"figures/{fig.name}"
                html_parts.append(
                    f"<div style='border:1px solid #ddd;padding:5px;border-radius:4px;'>"
                    f"<img src='{rel}' style='max-width:300px;max-height:200px;' />"
                    f"<p style='font-size:12px;text-align:center;'>{fig.stem}</p>"
                    f"</div>"
                )
        html_parts.append("</div>")

    html_parts.extend(["</body>", "</html>"])

    report_path = output_dir / "index.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"[OK] HTML 报告: {report_path}")

    # 导出统计 CSV
    if stats_df is not None and not stats_df.empty:
        csv_path = output_dir / "group_stats.csv"
        stats_df.to_csv(csv_path)
        print(f"[OK] 组统计: {csv_path}")


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="生成汇总报告。")
    parser.add_argument("--project-dir", type=Path, required=True, help="项目根目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--title", type=str, default="实验报告", help="报告标题")
    args = parser.parse_args()

    results_dir = args.project_dir / "results"
    if not results_dir.is_dir():
        print(f"[ERROR] 结果目录不存在: {results_dir}", file=sys.stderr)
        print("  请先运行 batch_analyze.py", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(results_dir)

    if not results:
        print("[WARN] 结果目录中无数据。")
        return 0

    subjects = list(results.keys())
    stats_df = compute_group_stats(results_dir, subjects)
    generate_html_report(args.project_dir, args.output_dir, args.title, stats_df, results)

    print(f"[DONE] 报告生成完成: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
