# 🤖 PsyPhiClaw Fusion Insight

## Overview

AI 驱动的多模态洞察引擎——**PsyPhiClaw 的核心差异化能力**。自动检测多模态数据中的异常模式，生成跨模态关联洞察，使用 LLM 生成自然语言分析报告。

> ⚠️ EthoClaw 没有任何 AI 洞察能力。这是 PsyPhiClaw 独有的功能。

## Pipeline

```
多模态数据 → detect_anomaly.py → 异常列表
                                    ↓
           multimodal_summary.py → 结构化汇总 → generate_insight.py → LLM 报告
                                    ↓
                            insight_guardrails.py → 质量评分 → 最终洞察
```

## 6-Layer Guardrail System

1. **数据完整性** — 缺失值、采样率异常
2. **统计显著性** — p-value 阈值验证
3. **效应量评估** — Cohen's d > 0.2 才报告
4. **因果推断警告** — 相关≠因果标注
5. **多重比较校正** — Bonferroni/FDR 提醒
6. **LLM 幻觉检测** — 与原始数据交叉验证

## Scripts

| Script | Purpose |
|--------|---------|
| `detect_anomaly.py` | 统计异常 + 趋势突变 + 跨模态同步检测 |
| `generate_insight.py` | LLM 驱动自然语言洞察生成 |
| `insight_guardrails.py` | 6 层护栏质量评分 |
| `multimodal_summary.py` | Session 摘要报告生成 |

## Quick Start

```bash
# 异常检测
python scripts/detect_anomaly.py --data session_data.json --output anomalies.json

# 生成洞察
python scripts/generate_insight.py --data session_data.json --anomalies anomalies.json --output insight.md

# 质量护栏
python scripts/insight_guardrails.py --insight insight.json --data session_data.json --output guarded.json

# 摘要报告
python scripts/multimodal_summary.py --data session_data.json --output summary/
```

## Color Scheme

- Primary: `#4A90D9`
- Alert: `#E74C3C`
