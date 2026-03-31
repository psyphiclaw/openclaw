#!/usr/bin/env python3
"""扫描实验项目目录，自动检测数据文件，生成项目 manifest JSON。

按文件扩展名和命名规则识别数据文件，按模态和被试分类。

Usage:
    python scan_project.py --project-dir data/my_exp/ --output manifest.json --modalities eeg,behavioral
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# 色方案常量（仅用于打印输出）
ACCENT_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"

# 扩展名 → 模态映射
EXTENSION_MODALITY: dict[str, str] = {
    ".csv": "behavioral",
    ".tsv": "behavioral",
    ".xlsx": "behavioral",
    ".json": "behavioral",
    ".edf": "eeg",
    ".bdf": "eeg",
    ".vhdr": "eeg",
    ".set": "eeg",
    ".fif": "eeg",
    ".mat": "eeg",
    ".avi": "video",
    ".mp4": "video",
    ".mov": "video",
    ".ts": "eye_tracking",
    ".asc": "eye_tracking",
}

# 被试 ID 匹配模式
SUBJ_PATTERNS = [
    re.compile(r"(?:sub|subj|subject|participant|S|P)[_-]*(\d+)", re.IGNORECASE),
    re.compile(r"^(\d{2,4})[_-]"),
]


def extract_subject_id(filename: str) -> str | None:
    """从文件名中提取被试 ID。

    Args:
        filename: 文件名（不含路径）。

    Returns:
        被试 ID 字符串，或 None。
    """
    for pat in SUBJ_PATTERNS:
        m = pat.search(filename)
        if m:
            return m.group(1)
    return None


def detect_modality(path: Path) -> str:
    """根据扩展名检测数据模态。

    Args:
        path: 文件路径。

    Returns:
        模态名称字符串。
    """
    ext = path.suffix.lower()
    return EXTENSION_MODALITY.get(ext, "unknown")


def scan_directory(
    project_dir: Path,
    modalities: list[str] | None = None,
) -> dict[str, Any]:
    """扫描项目目录，生成 manifest。

    Args:
        project_dir: 项目根目录。
        modalities: 可选模态过滤列表，None 表示全部。

    Returns:
        Manifest 字典。
    """
    files_info: list[dict[str, str]] = []
    subjects: set[str] = set()
    modality_counts: dict[str, int] = {}

    data_dir = project_dir
    if (project_dir / "data").is_dir():
        data_dir = project_dir / "data"

    for f in sorted(data_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.name.startswith("."):
            continue

        modality = detect_modality(f)
        if modalities and modality not in modalities:
            continue

        subj_id = extract_subject_id(f.name)

        files_info.append({
            "path": str(f.relative_to(project_dir)),
            "filename": f.name,
            "modality": modality,
            "subject_id": subj_id or "unknown",
            "size_bytes": f.stat().st_size,
        })

        if subj_id:
            subjects.add(subj_id)
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    return {
        "project_dir": str(project_dir),
        "total_files": len(files_info),
        "subjects": sorted(subjects),
        "num_subjects": len(subjects),
        "modality_counts": modality_counts,
        "files": files_info,
    }


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="扫描实验项目目录，生成 manifest JSON。")
    parser.add_argument("--project-dir", type=Path, required=True, help="项目根目录")
    parser.add_argument("--output", type=Path, default=None, help="输出 manifest 路径 (默认: project_dir/manifest.json)")
    parser.add_argument("--modalities", type=str, default=None, help="过滤模态（逗号分隔，如 eeg,behavioral）")
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        print(f"[ERROR] 项目目录不存在: {args.project_dir}", file=sys.stderr)
        return 1

    mod_filter = args.modalities.split(",") if args.modalities else None
    manifest = scan_directory(args.project_dir, mod_filter)

    output_path = args.output or (args.project_dir / "manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Manifest 已保存: {output_path}")
    print(f"  文件总数: {manifest['total_files']}")
    print(f"  被试数: {manifest['num_subjects']} ({', '.join(manifest['subjects'][:10])}{'...' if len(manifest['subjects']) > 10 else ''})")
    print(f"  模态分布: {manifest['modality_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
