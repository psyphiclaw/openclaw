# 🔴 PsyPhiClaw fNIRS Import

## Overview

导入 fNIRS（近红外光谱脑成像）数据，支持 SNIRF 通用格式、NIRSport、Artinis OxySoft 等主流设备。基于 MNE-Python 生态处理血红蛋白浓度信号。

## Pipeline

```
Raw Data → import_fnirs.py → MNE Raw → process_fnirs.py → HbO/HbR → analyze_fnirs.py → 统计结果
```

## Scripts

| Script | Purpose |
|--------|---------|
| `import_fnirs.py` | 读取 SNIRF/NIRSport/Artinis，输出 MNE Raw |
| `process_fnirs.py` | Beer-Lambert 转换、质量检查、滤波、HbO/HbR 分离 |
| `analyze_fnirs.py` | GLM 分析、统计参数映射、时间序列分析 |

## Quick Start

```bash
# 导入 SNIRF 数据
python scripts/import_fnirs.py --input session.snirf --format snirf --output raw.fif

# 预处理
python scripts/process_fnirs.py --input raw.fif --output processed.fif

# GLM 分析
python scripts/analyze_fnirs.py --input processed.fif --events events.tsv --output results/
```

## References

- [references/fnirs_formats.md](references/fnirs_formats.md) — 格式说明与预处理参数

## Dependencies

```bash
pip install mne numpy pandas matplotlib nilearn h5py
```

## Color Scheme

- Primary: `#4A90D9`
- Alert: `#E74C3C`
