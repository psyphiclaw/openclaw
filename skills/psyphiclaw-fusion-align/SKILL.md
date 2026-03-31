---
name: psyphiclaw-fusion-align
description: 多模态数据时间对齐工具。支持 TTL 触发同步、事件标记对齐、手动时间偏移三种模式，将不同设备、不同采样率的数据统一到共享时间轴上。输出标准化的对齐数据集。
metadata:
  openclaw:
    emoji: "🔗"
    requires:
      python: ["pandas", "numpy", "scipy"]
---

# PsyPhiClaw Fusion Align — 多模态时间对齐

## 概述

将来自不同设备（FaceReader、EEG、生理信号采集器等）、不同采样率的多模态数据统一到同一个时间轴上，为后续跨模态分析奠定基础。

## 三种对齐模式

| 模式 | 适用场景 | 精度 |
|------|----------|------|
| TTL 触发同步 | 实验室采集，硬件 TTL 信号连接各设备 | 亚毫秒 |
| 事件标记对齐 | 共享刺激事件（如屏幕闪烁、声音提示） | 毫秒级 |
| 手动时间偏移 | 无硬件同步，人工校准 | 秒级 |

## 数据流

```
FaceReader CSV ──┐
EEG Raw/Epochs ──┼──► align_trigger / align_marker ──► resample_sync ──► MultiModalSession
Physio CSV ──────┘                                       │
                                                         ▼
                                                    session_manager (.h5)
```

## 兼容上游 Skill

- **psyphiclaw-face-import** → 输出 pandas DataFrame（含 Timestamp 列）
- **psyphiclaw-eeg-import** → 输出 MNE Raw/Epochs 对象

## 使用方法

```bash
# 1. TTL 触发对齐
python scripts/align_trigger.py \
  --eeg-events eeg_events.csv \
  --modality-timestamps facereader_ts.csv \
  --modality-timestamps physio_ts.csv \
  --output alignment_params.json

# 2. 事件标记对齐
python scripts/align_marker.py \
  --files eeg_data.csv facereader_data.csv physio_data.csv \
  --timestamp-cols timestamp timestamp Timestamp \
  --event-marker stimulus_onset \
  --method nearest \
  --output aligned_params.json

# 3. 重采样同步
python scripts/resample_sync.py \
  --inputs aligned_eeg.csv aligned_face.csv aligned_physio.csv \
  --target-freq 250 \
  --method linear \
  --output synced_session.h5

# 4. Session 管理
python scripts/session_manager.py create \
  --name "subject_001_session_1" \
  --output session.h5

python scripts/session_manager.py add-modality \
  --session session.h5 \
  --modality eeg \
  --file eeg_data.csv \
  --sampling-rate 250

python scripts/session_manager.py export \
  --session session.h5 \
  --output export_dir/
```

## 输出格式

### MultiModalSession (.h5)

```python
session.h5
├── metadata/           # Session 元信息
│   ├── name
│   ├── created_at
│   └── aligned
├── eeg/                # EEG 模态数据
│   ├── data            # numpy array [channels × time]
│   ├── ch_names
│   ├── sampling_rate
│   └── events
├── face/               # FaceReader 模态数据
│   ├── data            # DataFrame
│   ├── sampling_rate
│   └── columns
├── physio/             # 生理信号模态
│   ├── data
│   ├── sampling_rate
│   └── columns
└── alignment/          # 对齐参数
    ├── offsets
    ├── target_freq
    └── method
```
