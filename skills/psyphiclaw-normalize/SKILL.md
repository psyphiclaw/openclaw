---
name: psyphiclaw-normalize
description: 多模态数据标准化与格式转换工具。将不同厂商、不同格式的数据转换为 PsyPhiClaw 统一内部格式，支持数据清洗、缺失值处理、Z-score 标准化等。
metadata:
  openclaw:
    emoji: "🔧"
    requires:
      python: ["pandas", "numpy", "scipy"]
---

# PsyPhiClaw Normalize

## 概述

多模态数据标准化与格式转换工具集，包含三个核心脚本：
- `normalize_data.py` — 标准化（Z-score、Min-Max、基线校正、百分位秩）
- `clean_data.py` — 数据清洗（缺失值、异常值、伪迹、质量评分）
- `convert_format.py` — 格式转换（CSV→HDF5、FaceReader、EEG 格式）

## 使用

```bash
# Z-score 标准化
python scripts/normalize_data.py --method zscore -i data.csv -o normalized.csv

# Min-Max 归一化
python scripts/normalize_data.py --method minmax -i data.csv -o normalized.csv

# 基线校正
python scripts/normalize_data.py --method baseline --baseline-start 0 --baseline-end 5 -i data.csv -o corrected.csv

# 数据清洗
python scripts/clean_data.py --missing drop --outliers zscore -i data.csv -o cleaned.csv

# 质量评分
python scripts/clean_data.py --quality-only -i data.csv

# CSV → HDF5 转换
python scripts/convert_format.py --to hdf5 -i data.csv -o data.h5

# FaceReader CSV 转换
python scripts/convert_format.py --from facereader -i facereader_export.csv -o output.h5
```

## 参考文档

- `references/cleaning_rules.md` — 各模态数据清洗规则与质量控制
