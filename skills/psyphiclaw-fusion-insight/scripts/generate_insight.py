#!/usr/bin/env python3
"""Generate natural language insights from multimodal data using LLM.

Summarizes multimodal statistics into structured JSON, calls LLM to produce
natural language insights, and identifies key findings automatically.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI insights from multimodal data.")
    parser.add_argument("--data", "-d", required=True, help="Session data JSON file")
    parser.add_argument("--anomalies", "-a", default=None, help="Anomalies JSON file")
    parser.add_argument("--stats", "-s", default=None, help="Pre-computed stats JSON")
    parser.add_argument("--output", "-o", default=None, help="Output file (.md or .json)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or OPENAI_API_KEY env)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Output language")
    parser.add_argument("--no-llm", action="store_true", help="Generate structured summary only, no LLM call")
    return parser.parse_args()


# ── Build Structured Summary ─────────────────────────────────────────────────

def build_structured_summary(
    data: dict,
    anomalies: Optional[dict] = None,
    stats: Optional[dict] = None,
) -> dict[str, Any]:
    """Build a structured summary of multimodal session data.

    Args:
        data: Session data dict with modality arrays.
        anomalies: Anomaly detection results.
        stats: Pre-computed statistics.

    Returns:
        Structured summary dict ready for LLM prompting.
    """
    summary: dict[str, Any] = {
        "session_info": {
            "modalities": list(data.keys()),
            "n_modalities": len(data),
            "timestamp": datetime.now().isoformat(),
        },
        "per_modality": {},
        "key_findings": [],
        "anomaly_summary": None,
    }

    for modality, info in data.items():
        if isinstance(info, dict) and "data" in info:
            arr = np.array(info["data"], dtype=float)
            sfreq = info.get("sfreq", None)
        elif isinstance(info, list):
            arr = np.array(info, dtype=float)
            sfreq = None
        else:
            continue

        mod_stats: dict[str, Any] = {
            "n_samples": len(arr),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

        if sfreq:
            mod_stats["duration_s"] = round(len(arr) / sfreq, 2)
            mod_stats["sfreq"] = sfreq

        # Detect trends
        if len(arr) > 10:
            x = np.arange(len(arr))
            slope = np.polyfit(x, arr, 1)[0]
            mod_stats["trend_slope"] = round(float(slope), 6)
            mod_stats["trend_direction"] = "increasing" if slope > 0.001 else ("decreasing" if slope < -0.001 else "stable")

        summary["per_modality"][modality] = mod_stats

        # Auto-detect key findings
        if mod_stats.get("trend_direction") != "stable":
            summary["key_findings"].append(
                f"{modality} shows {mod_stats['trend_direction']} trend (slope={mod_stats.get('trend_slope', 0):.4f})"
            )

        # Variability check
        cv = mod_stats["std"] / (abs(mod_stats["mean"]) + 1e-12)
        if cv > 2.0:
            summary["key_findings"].append(
                f"{modality} has high variability (CV={cv:.2f})"
            )

    # Anomaly summary
    if anomalies:
        total = anomalies.get("total_anomalies", 0)
        sync = anomalies.get("cross_modal_sync", {})
        summary["anomaly_summary"] = {
            "total_anomalies": total,
            "cross_modal_events": sync.get("n_events", 0),
            "per_modality": {
                k: v["n_anomalies"] for k, v in anomalies.get("modalities", {}).items()
            },
        }

        if sync.get("n_events", 0) > 0:
            top_sync = sync.get("events", [])[:3]
            for event in top_sync:
                mods = event["modalities_involved"]
                summary["key_findings"].append(
                    f"Cross-modal anomaly at sample {event['index']}: {', '.join(mods)} "
                    f"(severity={event['max_severity']:.2f})"
                )

    return summary


# ── LLM Insight Generation ────────────────────────────────────────────────────

SYSTEM_PROMPT_ZH = """你是一位多模态行为分析专家。根据提供的结构化数据摘要，生成自然语言分析报告。

要求：
1. 用简洁的中文撰写
2. 重点突出跨模态关联发现
3. 每个洞察需引用具体数据
4. 如果发现异常事件，分析可能的心理/生理原因
5. 在报告末尾给出建议性分析方向
6. 使用 Markdown 格式"""

SYSTEM_PROMPT_EN = """You are a multimodal behavioral analysis expert. Generate a natural language analysis report from the structured data summary.

Requirements:
1. Write in concise English
2. Highlight cross-modal correlation findings
3. Cite specific data for each insight
4. If anomalies are found, analyze potential psychological/physiological causes
5. Suggest further analysis directions at the end
6. Use Markdown format"""


def call_llm(
    structured_summary: dict,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    language: str = "zh",
) -> str:
    """Call LLM to generate natural language insights.

    Args:
        structured_summary: Structured data summary dict.
        api_key: OpenAI API key.
        model: Model name.
        language: Output language.

    Returns:
        LLM-generated insight text.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return f"[#E74C3C] ERROR: openai package not installed. pip install openai"

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return "[#E74C3C] ERROR: No API key provided. Set --api-key or OPENAI_API_KEY env."

    client = OpenAI(api_key=key)
    system_prompt = SYSTEM_PROMPT_ZH if language == "zh" else SYSTEM_PROMPT_EN

    user_content = json.dumps(structured_summary, indent=2, ensure_ascii=False, default=str)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请基于以下数据摘要生成分析报告：\n\n{user_content}"},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content or "No response generated."
    except Exception as e:
        return f"[#E74C3C] LLM API error: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    if not Path(args.data).exists():
        print(f"[{COLOR_ALERT}] ERROR: File not found: {args.data}", file=sys.stderr)
        return 1

    with open(args.data) as f:
        data = json.load(f)

    anomalies = None
    if args.anomalies and Path(args.anomalies).exists():
        with open(args.anomalies) as f:
            anomalies = json.load(f)

    stats = None
    if args.stats and Path(args.stats).exists():
        with open(args.stats) as f:
            stats = json.load(f)

    print(f"[{COLOR_PRIMARY}] Building structured summary...")
    summary = build_structured_summary(data, anomalies, stats)

    print(f"[{COLOR_PRIMARY}] Key findings: {len(summary['key_findings'])}")
    for finding in summary["key_findings"]:
        print(f"  • {finding}")

    # LLM generation
    llm_text: str = ""
    if not args.no_llm:
        print(f"[{COLOR_PRIMARY}] Calling LLM ({args.model})...")
        llm_text = call_llm(summary, args.api_key, args.model, args.language)
        print(f"[{COLOR_PRIMARY}] LLM insight generated ({len(llm_text)} chars)")
    else:
        print(f"[{COLOR_PRIMARY}] Skipping LLM (--no-llm)")

    # Build final output
    result: dict[str, Any] = {
        "structured_summary": summary,
        "llm_insight": llm_text,
        "generated_at": datetime.now().isoformat(),
        "language": args.language,
    }

    out_path = args.output or "insight.json"
    if out_path.endswith(".md"):
        # Markdown output
        md_lines = [
            f"# Multimodal Insight Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"**Modalities:** {', '.join(summary['session_info']['modalities'])}\n",
            f"## Key Findings\n",
        ]
        for f_text in summary["key_findings"]:
            md_lines.append(f"- {f_text}")
        md_lines.append("\n## LLM Analysis\n")
        md_lines.append(llm_text if llm_text else "*LLM analysis skipped*")
        md_lines.append("")

        with open(out_path, "w") as f:
            f.write("\n".join(md_lines))
        print(f"[{COLOR_PRIMARY}] ✓ Markdown saved: {out_path}")
    else:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"[{COLOR_PRIMARY}] ✓ JSON saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
