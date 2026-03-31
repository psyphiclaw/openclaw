---
name: psyphiclaw-eye-import
description: >
  导入眼动追踪数据，支持 Tobii TSV、EyeLink ASC、Pupil Labs JSON/CSV 格式。
  提供数据解析、标准化和基础可视化摘要。
metadata:
  openclaw:
    emoji: "👁️"
    requires:
      python: ["pandas", "numpy", "matplotlib", "plotly", "scipy"]
---

# PsyPhiClaw Eye Import 👁️

导入多格式眼动追踪数据，转换为统一 DataFrame 并生成摘要可视化。

## 支持格式

| 格式 | 脚本 | 扩展名 |
|------|------|--------|
| Tobii Pro Lab | `import_tobii.py` | `.tsv` |
| EyeLink | `import_eyelink.py` | `.asc` |
| Pupil Labs | `import_pupil.py` | `.csv`, `.json` |

## 统一输出列

所有导入脚本输出标准 DataFrame，包含以下列（按可用性填充）：

- `timestamp` — 时间戳（ms 或 s）
- `gaze_x`, `gaze_y` — 注视点坐标（像素）
- `pupil_left`, `pupil_right` — 瞳孔直径
- `fixation_index` — 注视事件 ID
- `event` — 事件类型（FixationStart, FixationEnd, SaccadeStart, SaccadeEnd, Blink）

## 使用示例

```bash
# 导入 Tobii 数据
python scripts/import_tobii.py data/recording_001.tsv --output results/tobii.parquet --summary

# 导入 EyeLink 数据
python scripts/import_eyelink.py data/subject01.asc --output results/eyelink.parquet

# 导入 Pupil Labs 数据
python scripts/import_pupil.py data/pupil_exports/ --output results/pupil.parquet --surface
```

## 参考

- [眼动数据格式说明](references/eye_formats.md)
