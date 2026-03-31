---
name: psyphiclaw-physio-analysis
description: >
  生理信号高级分析。基于 NeuroKit2 处理 ECG（心率变异性 HRV）、EDA（皮电反应）、
  EMG（肌电）、呼吸信号。提取时域/频域/非线性特征。
metadata:
  openclaw:
    emoji: "心律️"
    requires:
      python: ["neurokit2", "pandas", "numpy", "scipy", "plotly", "antropy"]
---

# PsyPhiClaw Physio Analysis 💓

## 分析模块

| 脚本 | 功能 |
|------|------|
| `hrv_analysis.py` | ECG R峰检测、HRV 时域/频域/非线性特征 |
| `eda_analysis.py` | EDA 分解 (tonic+phasic)、SCR 峰值检测 |
| `emg_analysis.py` | EMG 滤波、RMS/MAV 特征、onset 检测、疲劳分析 |
| `respiration_analysis.py` | 呼吸频率、幅度变异性、RSA |

## 使用示例

```bash
# HRV 分析
python scripts/hrv_analysis.py data.parquet --ecg-col ecg --fs 1000 --output hrv.csv --plot hrv.png

# EDA 分析
python scripts/eda_analysis.py data.parquet --eda-col eda --fs 200 --output eda.csv --plot eda.png

# EMG 分析
python scripts/emg_analysis.py data.parquet --emg-col emg --fs 2000 --output emg.csv

# 呼吸分析
python scripts/respiration_analysis.py data.parquet --resp-col resp --fs 100 --ecg-col ecg --output resp.csv
```

## 依赖

核心依赖为 NeuroKit2。如未安装，各脚本提供纯 scipy/numpy 的 fallback 实现。

## 参考

- [生理信号分析方法](references/physio_methods.md)
