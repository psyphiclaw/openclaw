---
name: psyphiclaw-fusion-correlation
description: 跨模态关联分析工具。在时间对齐的多模态数据上执行相关性分析、时间锁相分析、回归分析等，发现表情、脑电、生理信号之间的统计关联。
metadata:
  openclaw:
    emoji: "📊"
    requires:
      python: ["pandas", "numpy", "scipy", "statsmodels", "plotly"]
---

# PsyPhiClaw Fusion Correlation — 跨模态关联分析

## 概述

在时间对齐的多模态数据上执行统计关联分析，发现表情、脑电、生理信号之间的定量关系。

## 分析工具

| 工具 | 功能 | 输出 |
|------|------|------|
| cross_modal_corr | Pearson/Spearman 相关矩阵 + 滑动窗口 + 置换检验 | HTML + CSV |
| time_locked_analysis | 事件锁相片段提取 + ERP/表情/生理对比 | HTML + CSV |
| multimodal_stats | 多元回归 + 混合效应 + 多重比较校正 | 文本报告 + CSV |

## 使用方法

```bash
# 跨模态相关分析
python scripts/cross_modal_corr.py \
  --session session.h5 \
  --modalities eeg_alpha face_valence physio_eda \
  --method spearman \
  --window-size 5.0 \
  --step-size 1.0 \
  --n-permutations 1000 \
  --output results/

# 时间锁相分析
python scripts/time_locked_analysis.py \
  --session session.h5 \
  --event-times event_times.csv \
  --pre-stim 1.0 \
  --post-stim 3.0 \
  --output time_locked/

# 多模态统计
python scripts/multimodal_stats.py \
  --session session.h5 \
  --predictor eeg_alpha eeg_beta \
  --outcome face_valence \
  --group-by subject \
  --correction fdr \
  --output stats/
```

## 数据要求

输入为 `psyphiclaw-fusion-align` 产出的 MultiModalSession (.h5)，确保数据已对齐到统一时间轴。
