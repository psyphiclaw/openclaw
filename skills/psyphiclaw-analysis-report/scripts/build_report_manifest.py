#!/usr/bin/env python3
"""Build report manifest by scanning a PsyPhiClaw project directory.

Scans data files, figures, and statistics to generate a manifest.json
that drives report rendering.

Usage:
    python build_report_manifest.py --project-dir ./my_project -o manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# File type classification
# ---------------------------------------------------------------------------
MODALITY_MAP: dict[str, list[str]] = {
    "face": ["face", "facial", "emotion", "facereader", "action_unit", "au"],
    "eeg": ["eeg", "erp", "eeg_", "_eeg"],
    "eye": ["eye", "fixation", "saccade", "pupil", "gaze", "tobii", "eyelink", "smi"],
    "physio": ["ecg", "eda", "gsr", "emg", "resp", "ppg", "biopac", "heart", "skin"],
    "fnirs": ["fnirs", "nirs", "near_infrared", "hemoglobin", "oxy"],
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}
DATA_EXTS = {".csv", ".tsv", ".xlsx", ".hdf", ".h5", ".hdf5", ".parquet", ".pkl", ".npz"}
STAT_EXTS = {".json", ".yaml", ".yml", ".toml"}


def classify_modality(filename: str) -> str | None:
    """Classify a file's modality based on its name."""
    lower = filename.lower()
    for modality, keywords in MODALITY_MAP.items():
        for kw in keywords:
            if kw in lower:
                return modality
    return None


def scan_directory(project_dir: Path) -> dict[str, Any]:
    """Scan project directory and return structured scan results."""
    data_files: list[str] = []
    figure_files: list[str] = []
    stat_files: list[str] = []
    modality_files: dict[str, list[str]] = {m: [] for m in MODALITY_MAP}
    unclassified: list[str] = []
    subjects: set[str] = set()

    for root, _dirs, files in os.walk(project_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            ext = Path(f).suffix.lower()
            rel = os.path.relpath(fp, project_dir)

            if ext in IMAGE_EXTS:
                figure_files.append(rel)
            elif ext in DATA_EXTS:
                data_files.append(rel)
                mod = classify_modality(f)
                if mod:
                    modality_files[mod].append(rel)
                else:
                    unclassified.append(rel)
                # Try to extract subject IDs from path
                parts = Path(rel).parts
                for p in parts:
                    if p.lower().startswith(("sub", "subj", "subject", "p")) and p != "subjects":
                        subjects.add(p)
            elif ext in STAT_EXTS:
                stat_files.append(rel)

    return {
        "data_files": data_files,
        "figure_files": figure_files,
        "stat_files": stat_files,
        "modality_files": modality_files,
        "unclassified_data": unclassified,
        "subjects": sorted(subjects),
    }


def build_manifest(scan: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    """Build the full manifest dict from scan results."""
    modalities_detected = [m for m, files in scan["modality_files"].items() if files]

    # Build figures gallery with auto-classification
    figures = []
    for fig_path in scan["figure_files"]:
        mod = classify_modality(fig_path) or "general"
        section_map = {
            "face": "results_face",
            "eeg": "results_eeg",
            "eye": "results_eye",
            "physio": "results_physio",
            "fnirs": "results_fnirs",
            "general": "results_fusion",
        }
        figures.append({
            "path": fig_path,
            "caption": Path(fig_path).stem.replace("_", " ").replace("-", " "),
            "section": section_map.get(mod, "results_fusion"),
        })

    # Unconfirmed items (data files not classified to a modality)
    unconfirmed = scan.get("unclassified_data", [])[:20]

    manifest: dict[str, Any] = {
        "meta": {
            "project_name": project_dir.name,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "subjects_count": len(scan["subjects"]),
            "subjects": scan["subjects"][:100],
            "modalities": modalities_detected,
            "data_files_count": len(scan["data_files"]),
            "figure_files_count": len(scan["figure_files"]),
        },
        "facts": {
            "data_summary": f"Found {len(scan['data_files'])} data files across {len(modalities_detected)} modalities.",
            "modalities_info": {
                m: f"{len(scan['modality_files'][m])} files"
                for m in modalities_detected
            },
            "sample_check": f"Total subjects detected: {len(scan['subjects'])}",
            "unconfirmed_items": unconfirmed,
        },
        "galleries": {
            "figures": figures,
        },
        "section_bodies": {
            "overview": {"title": "实验概述", "body": ""},
            "methods": {"title": "方法", "body": ""},
            "results_face": {"title": "表情分析结果", "body": ""},
            "results_eeg": {"title": "EEG 分析结果", "body": ""},
            "results_eye": {"title": "眼动分析结果", "body": ""},
            "results_physio": {"title": "生理信号分析", "body": ""},
            "results_fnirs": {"title": "fNIRS 分析结果", "body": ""},
            "results_fusion": {"title": "多模态融合分析", "body": ""},
            "results_insight": {"title": "AI 洞察", "body": ""},
            "discussion": {"title": "讨论", "body": ""},
            "conclusion": {"title": "结论", "body": ""},
        },
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a PsyPhiClaw project directory and generate manifest.json"
    )
    parser.add_argument("--project-dir", type=Path, required=True, help="Project root directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output manifest path (default: <project_dir>/manifest.json)")
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        print(f"Error: {args.project_dir} is not a directory", file=sys.stderr)
        return 1

    output = args.output or args.project_dir / "manifest.json"
    scan = scan_directory(args.project_dir)
    manifest = build_manifest(scan, args.project_dir)

    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written to {output}")
    print(f"  Project: {manifest['meta']['project_name']}")
    print(f"  Subjects: {manifest['meta']['subjects_count']}")
    print(f"  Modalities: {', '.join(manifest['meta']['modalities']) or 'None detected'}")
    print(f"  Figures: {manifest['meta']['figure_files_count']}")
    print(f"  Data files: {manifest['meta']['data_files_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
