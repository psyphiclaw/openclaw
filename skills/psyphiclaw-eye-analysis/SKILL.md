---
name: psyphiclaw-eye-analysis
description: >
  眼动数据高级分析。包括注视分析（注视时长、注视次数）、扫视检测与分析、
  AOI（兴趣区）分析、瞳孔测量分析。
metadata:
  openclaw:
    emoji: "🔍"
    requires:
      python: ["pandas", "numpy", "scipy", "plotly", "matplotlib"]
---

# PsyPhiClaw Eye Analysis 🔍

## 分析模块

| 脚本 | 功能 |
|------|------|
| `fixation_analysis.py` | 注视时长分布、计数、首次注视、转移矩阵 |
| `saccade_detection.py` | 速度阈值扫视检测、幅度/方向分析 |
| `aoi_analysis.py` | AOI 定义、注视比例、转移矩阵、热力图 |
| `pupil_analysis.py` | 瞳孔预处理、事件锁相响应 |

## 使用示例

```bash
# 注视分析
python scripts/fixation_analysis.py data.gaze.parquet --output fixation_stats.csv

# 扫视检测
python scripts/saccade_detection.py data.gaze.parquet --threshold 100 --plot saccades.png

# AOI 分析
python scripts/aoi_analysis.py data.gaze.parquet --aoi-config aoi.json --heatmap heatmap.png

# 瞳孔分析
python scripts/pupil_analysis.py data.gaze.parquet --events events.csv --output pupil_results.csv
```

## 参考

- [眼动分析方法推荐](references/eye_methods.md)
