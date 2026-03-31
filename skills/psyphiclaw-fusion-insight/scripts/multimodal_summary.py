#!/usr/bin/env python3
"""Generate multimodal session summary reports.

Combines key statistics, anomaly events, and LLM-generated conclusions
into a comprehensive Markdown + JSON summary.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multimodal session summary.")
    parser.add_argument("--data", "-d", required=True, help="Session data JSON")
    parser.add_argument("--anomalies", "-a", default=None, help="Anomalies JSON")
    parser.add_argument("--insight", "-i", default=None, help="Guarded insight JSON")
    parser.add_argument("--output", "-o", default="summary", help="Output directory or file prefix")
    parser.add_argument("--title", default=None, help="Report title")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    return parser.parse_args()


# ── Generate Markdown Report ──────────────────────────────────────────────────

def generate_markdown(
    data: dict,
    anomalies: Optional[dict] = None,
    insight: Optional[dict] = None,
    title: Optional[str] = None,
    language: str = "zh",
) -> str:
    """Generate a comprehensive Markdown summary report.

    Args:
        data: Session data dict.
        anomalies: Anomaly detection results.
        insight: Guarded insight results.
        title: Report title.
        language: Output language.

    Returns:
        Markdown string.
    """
    if language == "zh":
        lines = [
            f"# {title or '多模态行为分析摘要报告'}",
            "",
            f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
    else:
        lines = [
            f"# {title or 'Multimodal Behavioral Analysis Summary'}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

    # ── Per-Modality Statistics ──
    section_title = "## 📊 各模态统计指标" if language == "zh" else "## 📊 Per-Modality Statistics"
    lines.extend([section_title, ""])

    table_header = "| 模态 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |" if language == "zh" else \
                  "| Modality | Samples | Mean | Std | Min | Max |"
    lines.extend([table_header, "|---|---|---|---|---|---|"])

    for modality, info in data.items():
        if isinstance(info, dict) and "data" in info:
            arr = np.array(info["data"], dtype=float).flatten()
            sfreq = info.get("sfreq", None)
        elif isinstance(info, list):
            arr = np.array(info, dtype=float).flatten()
            sfreq = None
        else:
            continue

        duration = f"{len(arr)/sfreq:.1f}s" if sfreq else "N/A"
        lines.append(
            f"| {modality} | {len(arr)} | {np.mean(arr):.4f} | {np.std(arr):.4f} "
            f"| {np.min(arr):.4f} | {np.max(arr):.4f} |"
        )

    lines.append("")

    # ── Anomaly Events ──
    if anomalies:
        anom_title = "## ⚠️ 异常事件" if language == "zh" else "## ⚠️ Anomaly Events"
        lines.extend([anom_title, ""])

        total = anomalies.get("total_anomalies", 0)
        sync_n = anomalies.get("cross_modal_sync", {}).get("n_events", 0)

        if language == "zh":
            lines.append(f"- 总异常数: **{total}**")
            lines.append(f"- 跨模态同步事件: **{sync_n}**")
        else:
            lines.append(f"- Total anomalies: **{total}**")
            lines.append(f"- Cross-modal sync events: **{sync_n}**")

        per_mod = anomalies.get("modalities", {})
        if per_mod:
            lines.append("")
            mod_title = "各模态异常分布:" if language == "zh" else "Per-modality breakdown:"
            lines.append(f"**{mod_title}**")
            for mod, info in per_mod.items():
                lines.append(f"  - {mod}: {info['n_anomalies']} anomalies (avg severity: {info['mean_severity']:.3f})")

        # Top cross-modal events
        sync_events = anomalies.get("cross_modal_sync", {}).get("events", [])
        if sync_events:
            sync_title = "### 主要跨模态事件" if language == "zh" else "### Key Cross-Modal Events"
            lines.extend(["", sync_title, ""])
            for evt in sync_events[:5]:
                mods = ", ".join(evt["modalities_involved"])
                lines.append(
                    f"- **Sample {evt['index']}**: {mods} — severity={evt['max_severity']:.3f}"
                )

        lines.append("")

    # ── AI Analysis ──
    if insight:
        ai_title = "## 🤖 AI 分析结论" if language == "zh" else "## 🤖 AI Analysis"
        lines.extend([ai_title, ""])

        # Guardrails status
        guardrails = insight.get("guardrails", {})
        if guardrails:
            status = guardrails.get("status_message", "Unknown")
            score = guardrails.get("overall_score", 0)
            color = "#4A90D9" if score >= 0.8 else "#E74C3C"
            lines.append(f"> **置信度评分:** {score:.3f} — {status}")
            lines.append("")

        # LLM insight
        llm_text = insight.get("llm_insight", "")
        if llm_text:
            lines.append(llm_text)

    lines.append("")
    lines.extend(["---", f"*Generated by PsyPhiClaw Fusion Insight · {datetime.now().strftime('%Y-%m-%d')}*"])

    return "\n".join(lines)


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

    anomalies: Optional[dict] = None
    if args.anomalies and Path(args.anomalies).exists():
        with open(args.anomalies) as f:
            anomalies = json.load(f)

    insight: Optional[dict] = None
    if args.insight and Path(args.insight).exists():
        with open(args.insight) as f:
            insight = json.load(f)

    print(f"[{COLOR_PRIMARY}] Generating session summary...")

    # Markdown
    md = generate_markdown(data, anomalies, insight, args.title, args.language)

    out_dir = Path(args.output)
    if not out_dir.suffix:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "session_summary.md"
        json_path = out_dir / "session_summary.json"
    else:
        md_path = Path(str(out_dir).replace(out_dir.suffix, ".md"))
        json_path = Path(str(out_dir).replace(out_dir.suffix, ".json"))
        md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "w") as f:
        f.write(md)
    print(f"[{COLOR_PRIMARY}] ✓ Markdown: {md_path}")

    # JSON summary
    json_summary: dict[str, Any] = {
        "title": args.title,
        "generated_at": datetime.now().isoformat(),
        "n_modalities": len(data),
        "anomalies_total": anomalies.get("total_anomalies", 0) if anomalies else 0,
        "cross_modal_events": anomalies.get("cross_modal_sync", {}).get("n_events", 0) if anomalies else 0,
        "guardrails_score": insight.get("guardrails", {}).get("overall_score") if insight else None,
        "guardrails_status": insight.get("guardrails", {}).get("status") if insight else None,
    }

    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    print(f"[{COLOR_PRIMARY}] ✓ JSON: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
