---
name: psyphiclaw-analysis-report
description: 多模态分析报告生成。基于 manifest.json 架构，扫描项目数据，生成结构化的 HTML/PDF 分析报告，包含实验概述、方法描述、统计结果、图表嵌入和 AI 洞察摘要。
metadata:
  openclaw:
    emoji: "📝"
    requires:
      python: ["pandas", "numpy", "Pillow", "jinja2"]
---

# PsyPhiClaw Analysis Report

## 概述

基于 manifest → fill → render 三步架构生成多模态分析报告。

## 工作流

1. **构建 Manifest**: `build_report_manifest.py --project-dir ./data`
2. **编辑内容**: 手动填充 `manifest.json` 中的 `section_bodies`
3. **渲染报告**: `render_report.py --manifest manifest.json --lang cn`

## 使用

```bash
# Step 1: 扫描项目生成 manifest
python scripts/build_report_manifest.py --project-dir /path/to/project -o manifest.json

# Step 2: 编辑 manifest.json 填充各 section 的 body 字段

# Step 3: 渲染 HTML 报告
python scripts/render_report.py --manifest manifest.json --lang cn -o report.html
```

## 参考文档

- `references/report_sections.md` — 报告各部分内容要求
- `references/interpretation_guardrails.md` — 结果解释注意事项
- `assets/section_templates/` — 各 section 写作指引
