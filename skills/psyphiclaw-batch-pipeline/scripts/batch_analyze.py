#!/usr/bin/env python3
"""批量执行分析。读取 YAML pipeline 配置，按步骤执行，支持断点续跑。

Usage:
    python batch_analyze.py --project-dir data/my_exp/ --pipeline pipeline.yaml --output-dir results/ --skip-completed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# 色方案
ACCENT_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"


def load_pipeline(path: Path) -> dict[str, Any]:
    """加载 YAML pipeline 配置。

    Args:
        path: YAML 文件路径。

    Returns:
        Pipeline 配置字典。
    """
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("[ERROR] 需要 PyYAML: pip install pyyaml", file=sys.stderr)
        sys.exit(1)


def is_step_completed(output_dir: Path, step_name: str, subject_id: str) -> bool:
    """检查某个步骤是否已完成（断点续跑）。

    Args:
        output_dir: 输出目录。
        step_name: 步骤名称。
        subject_id: 被试 ID。

    Returns:
        是否已完成。
    """
    marker = output_dir / subject_id / step_name / ".completed"
    return marker.is_file()


def mark_step_completed(output_dir: Path, step_name: str, subject_id: str) -> None:
    """标记步骤已完成。

    Args:
        output_dir: 输出目录。
        step_name: 步骤名称。
        subject_id: 被试 ID。
    """
    marker_dir = output_dir / subject_id / step_name
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / ".completed").touch()


def run_step(
    step: dict[str, Any],
    subject_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """执行单个分析步骤。

    支持两种步骤类型：
    - 'command': 执行 shell 命令
    - 'function': 调用 Python 函数（通过 subprocess）

    Args:
        step: 步骤配置字典。
        subject_id: 被试 ID。
        input_dir: 输入数据目录。
        output_dir: 输出目录。

    Returns:
        执行结果字典。
    """
    step_name = step["name"]
    step_output = output_dir / subject_id / step_name
    step_output.mkdir(parents=True, exist_ok=True)

    try:
        if step["type"] == "command":
            cmd_template = step["command"]
            cmd = cmd_template.format(
                subject_id=subject_id,
                input_dir=input_dir / subject_id,
                output_dir=step_output,
                project_dir=str(input_dir.parent),
            )
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                return {"status": "error", "step": step_name, "error": result.stderr[:500]}

        elif step["type"] == "function":
            script = step["script"]
            result = subprocess.run(
                [sys.executable, script, "--subject", subject_id, "--output", str(step_output)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                return {"status": "error", "step": step_name, "error": result.stderr[:500]}

        else:
            return {"status": "error", "step": step_name, "error": f"未知步骤类型: {step['type']}"}

        mark_step_completed(output_dir, step_name, subject_id)
        return {"status": "ok", "step": step_name, "subject": subject_id}

    except subprocess.TimeoutExpired:
        return {"status": "error", "step": step_name, "error": "超时 (600s)"}
    except Exception as exc:
        return {"status": "error", "step": step_name, "error": str(exc)}


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="按 pipeline 配置批量执行分析。")
    parser.add_argument("--project-dir", type=Path, required=True, help="项目根目录")
    parser.add_argument("--pipeline", type=Path, required=True, help="Pipeline YAML 配置路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--skip-completed", action="store_true", help="跳过已完成步骤（断点续跑）")
    args = parser.parse_args()

    if not args.pipeline.is_file():
        print(f"[ERROR] Pipeline 文件不存在: {args.pipeline}", file=sys.stderr)
        return 1

    pipeline = load_pipeline(args.pipeline)
    steps = pipeline.get("steps", [])
    subjects = pipeline.get("subjects", [])

    # 如果没有指定被试，尝试从 manifest 或目录中检测
    if not subjects:
        manifest_path = args.project_dir / "manifest.json"
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            subjects = manifest.get("subjects", [])

    if not subjects:
        print("[ERROR] 未找到被试列表。请在 pipeline.yaml 中指定 subjects 或提供 manifest。", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.project_dir / "imported" if (args.project_dir / "imported").is_dir() else args.project_dir

    total = len(subjects) * len(steps)
    done = 0
    errors: list[dict[str, Any]] = []
    t0 = time.time()

    print(f"[INFO] Pipeline: {pipeline.get('name', 'unnamed')} | 被试: {len(subjects)} | 步骤: {len(steps)} | 总任务: {total}")

    for subject_id in subjects:
        for step in steps:
            step_name = step["name"]

            if args.skip_completed and is_step_completed(args.output_dir, step_name, subject_id):
                done += 1
                continue

            result = run_step(step, subject_id, input_dir, args.output_dir)
            done += 1

            pct = done / total
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            status_icon = "✓" if result["status"] == "ok" else "✗"
            print(f"\r  [{bar}] {done}/{total} {status_icon} {subject_id}/{step_name}", end="", flush=True)

            if result["status"] == "error":
                errors.append(result)

    elapsed = time.time() - t0
    print(f"\n[DONE] Pipeline 执行完成 ({elapsed:.1f}s)")

    if errors:
        err_log = args.output_dir / "analysis_errors.json"
        with open(err_log, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"[WARN] {len(errors)} 个任务失败，详见: {err_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
