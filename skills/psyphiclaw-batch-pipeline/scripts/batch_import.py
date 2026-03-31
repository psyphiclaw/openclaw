#!/usr/bin/env python3
"""批量导入所有被试数据。支持并行处理、进度条和错误日志。

Usage:
    python batch_import.py --project-dir data/my_exp/ --manifest manifest.json --output-dir imported/ --workers 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

# 色方案
ACCENT_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"


def import_single_file(task: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """导入单个数据文件。

    Args:
        task: 任务字典，包含 source_path 和 subject_id。
        output_dir: 输出根目录。

    Returns:
        结果字典。
    """
    src = Path(task["source_path"])
    subject_dir = output_dir / task["subject_id"]
    subject_dir.mkdir(parents=True, exist_ok=True)

    dest = subject_dir / src.name

    try:
        ext = src.suffix.lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(src, sep=sep)
            dest_out = dest.with_suffix(".parquet")
            df.to_parquet(dest_out, index=False)
            return {"status": "ok", "source": str(src), "output": str(dest_out), "rows": len(df)}
        else:
            shutil.copy2(src, dest)
            return {"status": "ok", "source": str(src), "output": str(dest)}
    except Exception as exc:
        return {"status": "error", "source": str(src), "error": str(exc)}


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="批量导入被试数据。")
    parser.add_argument("--project-dir", type=Path, required=True, help="项目根目录")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest JSON 路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--workers", type=int, default=2, help="并行工作数 (默认: 2)")
    parser.add_argument("--continue-on-error", action="store_true", help="遇到错误继续处理")
    args = parser.parse_args()

    # 加载 manifest
    manifest_path = args.manifest or (args.project_dir / "manifest.json")
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest 不存在: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", [])
    if not files:
        print("[WARN] Manifest 中无文件。")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 构建任务列表
    tasks: list[dict[str, Any]] = []
    for file_info in files:
        abs_src = args.project_dir / file_info["path"]
        if abs_src.is_file():
            tasks.append({
                "source_path": str(abs_src),
                "subject_id": file_info["subject_id"],
            })

    total = len(tasks)
    print(f"[INFO] 开始导入 {total} 个文件，工作线程: {args.workers}")

    results_ok: list[dict[str, Any]] = []
    results_err: list[dict[str, Any]] = []
    t0 = time.time()

    # 并行导入
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(import_single_file, t, args.output_dir): t for t in tasks}
        done_count = 0
        for future in as_completed(future_map):
            done_count += 1
            result = future.result()
            # 进度条
            pct = done_count / total
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {done_count}/{total} ({pct:.0%})", end="", flush=True)

            if result["status"] == "ok":
                results_ok.append(result)
            else:
                results_err.append(result)
                if not args.continue_on_error:
                    print(f"\n[ERROR] 导入失败: {result['source']}: {result['error']}", file=sys.stderr)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    elapsed = time.time() - t0
    print(f"\n[DONE] 导入完成 ({elapsed:.1f}s)")

    # 写入错误日志
    if results_err:
        err_log = args.output_dir / "import_errors.json"
        with open(err_log, "w", encoding="utf-8") as f:
            json.dump(results_err, f, indent=2, ensure_ascii=False)
        print(f"[WARN] {len(results_err)} 个文件导入失败，详见: {err_log}")

    print(f"  成功: {len(results_ok)} | 失败: {len(results_err)} | 输出: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
