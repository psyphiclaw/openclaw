---
name: psyphiclaw-visualization
description: 多模态数据综合可视化工具。生成交互式多模态时间线、跨模态叠加图、ERP 拓扑图、统计图表等，支持论文级排版和 HTML 交互式输出。
metadata:
  openclaw:
    emoji: "📈"
    requires:
      python: ["plotly", "matplotlib", "seaborn", "numpy", "pandas"]
---

# PsyPhiClaw Visualization — 多模态综合可视化

## 概述

在同一时间线上展示多个模态的数据，支持交互式 HTML 和论文级静态输出。

## 可视化工具

| 工具 | 功能 | 输出格式 |
|------|------|----------|
| multimodal_timeline | 交互式多模态时间线 (子图布局) | HTML + PNG |
| heatmap_generator | 特征热力图、相关性热力图、聚类热力图 | HTML + PNG |
| statistical_charts | 小提琴图、箱线图、雷达图、条形图 | PNG + HTML |
| erp_topo | ERP 头皮拓扑图 | PNG |

## 使用方法

```bash
# 多模态时间线
python scripts/multimodal_timeline.py \
  --session session.h5 \
  --modalities eeg face physio \
  --events events.csv \
  --output timeline.html

# 热力图
python scripts/heatmap_generator.py \
  --data features.csv \
  --type correlation \
  --method spearman \
  --output heatmap.html

# 统计图表
python scripts/statistical_charts.py \
  --data comparison.csv \
  --chart violin \
  --group-col condition \
  --value-col valence \
  --output violin_plot.png

# ERP 拓扑
python scripts/erp_topo.py \
  --epochs epochs.csv \
  --ch-names Fz Cz Pz O1 O2 \
  --times -200 0 200 400 600 800 \
  --output erp_topo.png
```

## 颜色方案

| 用途 | 颜色 |
|------|------|
| 主色 (EEG/脑电) | #4A90D9 (蓝) |
| 强调色 (表情/情绪) | #E74C3C (红) |
| 生理信号 | #2ECC71 (绿) |
| 事件标记 | #F39C12 (橙) |
| 背景色 | #FAFAFA |
