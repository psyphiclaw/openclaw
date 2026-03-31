---
name: psyphiclaw-physio-import
description: >
  导入生理信号数据，支持 Biopac ACQ 格式、ADInstruments LabChart 导出、通用 CSV。
  解析 ECG、EDA、EMG、呼吸、皮温等信号通道。
metadata:
  openclaw:
    emoji: "💓"
    requires:
      python: ["pandas", "numpy", "scipy", "wfdb", "plotly"]
---

# PsyPhiClaw Physio Import 💓

导入多格式生理信号数据，自动检测通道类型并转换为统一时间序列。

## 支持格式

| 格式 | 脚本 | 扩展名 |
|------|------|--------|
| Biopac AcqKnowledge | `import_biopac.py` | `.acq` |
| ADInstruments LabChart | `import_adinstruments.py` | `.csv`, `.txt` |
| 通用表格 | `import_physio_csv.py` | `.csv`, `.tsv`, `.xlsx` |

## 通用输出格式

- `timestamp_s` — 时间戳（秒）
- `ecg` — ECG 信号（mV）
- `eda` — 皮肤电导（μS）
- `emg` — 肌电（mV）
- `resp` — 呼吸信号
- `temp` — 皮温（°C）
- `sampling_rate_hz` — 采样率（元数据）

## 使用示例

```bash
# Biopac ACQ
python scripts/import_biopac.py recording.acq --output result.parquet --summary

# LabChart
python scripts/import_adinstruments.py labchart_export.csv --output result.parquet

# 通用 CSV
python scripts/import_physio_csv.py signals.csv --output result.parquet --time-col Time --signal-cols ECG,EDA
```

## 参考

- [生理信号格式说明](references/physio_formats.md)
