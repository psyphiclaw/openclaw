---
name: psyphiclaw-web-dashboard
description: 基于 Dash + Plotly 的交互式 Web 仪表盘。提供多模态数据浏览、分析结果可视化、被试管理等功能，支持本地部署。
metadata:
  openclaw:
    emoji: "🖥️"
    requires:
      python: ["dash", "plotly", "pandas", "numpy"]
    install:
      - id: pip
        kind: pip
        package: dash
        label: Install Dash for web dashboard
---

# PsyPhiClaw Web Dashboard

## 概述

基于 Dash + Plotly 的交互式 Web 仪表盘，用于多模态行为数据浏览与分析结果可视化。

## 启动

```bash
# 安装依赖
pip install dash plotly pandas numpy

# 启动仪表盘
python scripts/app.py --project-dir /path/to/project --port 8050

# 浏览器访问
# http://127.0.0.1:8050
```

## 功能页面

1. **概览页** — 项目统计卡片、模态覆盖、数据完整性
2. **数据浏览** — 多模态时间线交互图（Plotly）
3. **分析结果** — 图表画廊、统计表格、AI 洞察
4. **被试管理** — 被试列表、数据状态、处理进度
5. **设置** — 显示配置、主题切换

## 参考文档

- `references/dashboard_guide.md` — 部署与自定义指南
