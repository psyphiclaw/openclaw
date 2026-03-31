# PsyPhiClaw Skill 索引

> **版本**: v1.0-draft | **日期**: 2026-03-31

本文档列出 PsyPhiClaw 全部 14 个核心 Skill 的接口信息、依赖和对标关系。

---

## Skill 总览

| # | Skill | 类别 | 对标 EthoClaw |
|---|-------|------|---------------|
| 1 | psyphiclaw-face-import | 数据接入 | — |
| 2 | psyphiclaw-eye-import | 数据接入 | — |
| 3 | psyphiclaw-eeg-import | 数据接入 | — |
| 4 | psyphiclaw-physio-import | 数据接入 | — |
| 5 | psyphiclaw-fnirs-import | 数据接入 | — |
| 6 | psyphiclaw-preprocess-face | 预处理 | ethoclaw-normalize-tabular |
| 7 | psyphiclaw-preprocess-eye | 预处理 | ethoclaw-normalize-tabular |
| 8 | psyphiclaw-preprocess-eeg | 预处理 | ethoclaw-normalize-tabular |
| 9 | psyphiclaw-preprocess-physio | 预处理 | ethoclaw-normalize-tabular |
| 10 | psyphiclaw-eeg-analysis | 分析 | ethoclaw-kinematic-parameter-generator |
| 11 | psyphiclaw-eye-analysis | 分析 | ethoclaw-trajectory-velocity-heatmap-generate |
| 12 | psyphiclaw-physio-analysis | 分析 | ethoclaw-kinematic-parameter-generator |
| 13 | psyphiclaw-fusion-insight | AI 洞察 | **无对标（首创）** |
| 14 | psyphiclaw-visualization | 可视化 | ethoclaw-paper-figure-layout |

---

## 数据接入类 (5个)

### 1. psyphiclaw-face-import

| 属性 | 说明 |
|------|------|
| **名称** | 面部表情数据导入 |
| **描述** | 导入 Noldus FaceReader 导出的 CSV 数据，解析 Action Unit、基本情绪维度、头部朝向等特征 |
| **功能** | 自动检测编码格式(UTF-8/GBK)；解析时间戳、AU 系列列、Valence/Arousal 维度、Head 方向；输出标准化 DataFrame |
| **输入格式** | FaceReader CSV 文件 (`.csv`) |
| **输出格式** | `StandardizedSession.face_df` — Pandas DataFrame: `[timestamp, au_1..au_45, valence, arousal, head_yaw/pitch/roll, ...]` |
| **核心依赖** | `pandas >= 2.0`, `chardet` |
| **对标 EthoClaw** | 无直接对标（人类专属模态） |

### 2. psyphiclaw-eye-import

| 属性 | 说明 |
|------|------|
| **名称** | 眼动数据导入 |
| **描述** | 导入 Tobii Pro 眼动仪 (TSV) 或 EyeLink (ASC) 导出的原始数据 |
| **功能** | 解析注视点坐标、瞳孔直径、扫视/注视事件分类；支持 AOI 定义导入；自动检测采样率 |
| **输入格式** | Tobii TSV (`.tsv`, UTF-16) 或 EyeLink ASC (`.asc`, ASCII) |
| **输出格式** | `StandardizedSession.eye_df` — Pandas DataFrame: `[timestamp, left_x, left_y, right_x, right_y, left_pupil, right_pupil, event_type, ...]` |
| **核心依赖** | `pandas >= 2.0` |
| **对标 EthoClaw** | 无直接对标（人类专属模态） |

### 3. psyphiclaw-eeg-import

| 属性 | 说明 |
|------|------|
| **名称** | 脑电数据导入 |
| **描述** | 导入 BrainVision、EGI、EDF 等标准 EEG 格式 |
| **功能** | 读取通道数据、采样率、事件标记；自动检测通道名称；支持预加载和惰性加载 |
| **输入格式** | BrainVision (`.vhdr`) / EGI (`.raw`, `.mff`) / EDF/BDF (`.edf`, `.bdf`) |
| **输出格式** | `StandardizedSession.eeg_raw` — `mne.io.Raw` 对象 + `events` ndarray + `event_id` dict |
| **核心依赖** | `mne >= 1.6`, `numpy >= 1.24` |
| **对标 EthoClaw** | 无直接对标 |

### 4. psyphiclaw-physio-import

| 属性 | 说明 |
|------|------|
| **名称** | 生理信号数据导入 |
| **描述** | 导入 Biopac AcqKnowledge 导出的生理信号数据（ECG、EDA、EMG、呼吸等） |
| **功能** | 解析多通道 CSV；自动检测通道类型（基于列名匹配）；支持自定义通道映射 |
| **输入格式** | Biopac CSV (`.csv`) |
| **输出格式** | `StandardizedSession.physio_df` — Pandas DataFrame: `[timestamp, ecg, eda, emg_*, respiration, skin_temp, ...]` |
| **核心依赖** | `pandas >= 2.0`, `scipy >= 1.11` |
| **对标 EthoClaw** | 无直接对标 |

### 5. psyphiclaw-fnirs-import

| 属性 | 说明 |
|------|------|
| **名称** | 近红外光谱数据导入 |
| **描述** | 导入 Homer3 或 NIRX fNIRS 系统采集的光学密度数据 |
| **功能** | 读取光极配置、原始光强数据；自动解析发射-接收对；输出 MNE Raw 对象 |
| **输入格式** | Homer3 数据目录 / NIRX (`.hdr` + `.dat`) |
| **输出格式** | `StandardizedSession.fnirs_raw` — `mne.io.Raw` 对象 (光学密度数据) |
| **核心依赖** | `mne >= 1.6`, `mne-nirs >= 0.6`, `numpy >= 1.24` |
| **对标 EthoClaw** | 无直接对标 |

---

## 预处理类 (4个)

### 6. psyphiclaw-preprocess-face

| 属性 | 说明 |
|------|------|
| **名称** | 面部表情数据预处理 |
| **描述** | 清洗 FaceReader 导入数据，处理缺失值和异常值 |
| **功能** | 线性插值缺失帧（<500ms 间隙）；Savitzky-Golay 平滑滤波；Z-score 异常值裁剪（±3σ）；按时间范围截取 |
| **输入格式** | `StandardizedSession.face_df` (DataFrame) |
| **输出格式** | `StandardizedSession.face_clean` (DataFrame) |
| **核心依赖** | `pandas`, `scipy` |
| **对标 EthoClaw** | `ethoclaw-normalize-tabular` — 同为表格数据标准化清洗 |

### 7. psyphiclaw-preprocess-eye

| 属性 | 说明 |
|------|------|
| **名称** | 眼动数据预处理 |
| **描述** | 清洗眼动数据，修复眨眼和数据缺失 |
| **功能** | 眨眼检测（瞳孔丢失段）+ 线性插值；基于速度阈值（30-100°/s）的注视/扫视分类；低质量采样点过滤（基于置信度/速度标准差）；AOI 自动匹配 |
| **输入格式** | `StandardizedSession.eye_df` (DataFrame) |
| **输出格式** | `StandardizedSession.eye_clean` (DataFrame) |
| **核心依赖** | `pandas`, `scipy`, `numpy` |
| **对标 EthoClaw** | `ethoclaw-normalize-tabular` |

### 8. psyphiclaw-preprocess-eeg

| 属性 | 说明 |
|------|------|
| **名称** | 脑电数据预处理 |
| **描述** | EEG 数据的完整预处理流水线 |
| **功能** | 通道选择（标准 10-20 系统）；带通滤波（1-40Hz，Butterworth）；陷波滤波（50/60Hz 工频）；ICA 去眼电/肌电伪迹；重参考（平均/乳突）；基线校正；事件锁定分段 |
| **输入格式** | `StandardizedSession.eeg_raw` (mne.io.Raw) |
| **输出格式** | `StandardizedSession.eeg_epochs` (mne.Epochs) |
| **核心依赖** | `mne`, `scikit-learn` (ICA) |
| **对标 EthoClaw** | `ethoclaw-normalize-tabular` — 数据清洗理念一致，但 EEG 需要专门的信号处理 |

### 9. psyphiclaw-preprocess-physio

| 属性 | 说明 |
|------|------|
| **名称** | 生理信号预处理 |
| **描述** | ECG/EDA/EMG/呼吸等多通道生理信号的预处理 |
| **功能** | ECG → R 峰检测(Pan-Tompkins) + IBI 序列提取；EDA → 低通滤波(4Hz) + 基线校正 + 峰值检测；EMG → 整流 + RMS 平滑(200ms 窗口)；呼吸 → 低通滤波 + 呼吸率提取 |
| **输入格式** | `StandardizedSession.physio_df` (DataFrame) |
| **输出格式** | `StandardizedSession.physio_features` (DataFrame: 特征级数据) |
| **核心依赖** | `neurokit2 >= 0.2`, `scipy` |
| **对标 EthoClaw** | `ethoclaw-normalize-tabular` |

---

## 分析类 (3个)

### 10. psyphiclaw-eeg-analysis

| 属性 | 说明 |
|------|------|
| **名称** | 脑电数据分析 |
| **描述** | EEG/ERP 的核心分析能力 |
| **功能** | ERP 成分分析（N100, P200, P300, N400, LRP）；时频分析（Morlet 小波变换）；频谱功率（delta/theta/alpha/beta/gamma 功率谱密度）；功能连接（相干性、相位锁定值）；溯源估计（可选，需 MNE 标准头模型） |
| **输入格式** | `StandardizedSession.eeg_epochs` (mne.Epochs) |
| **输出格式** | `analysis_results.eeg`: `{erp_data, power_data, connectivity, source_estimates, figures}` |
| **核心依赖** | `mne`, `numpy`, `scipy`, `matplotlib` |
| **对标 EthoClaw** | `ethoclaw-kinematic-parameter-generator` — 同为从原始数据中提取有意义的行为参数 |

### 11. psyphiclaw-eye-analysis

| 属性 | 说明 |
|------|------|
| **名称** | 眼动数据分析 |
| **描述** | 眼动追踪数据的可视化与统计分析 |
| **功能** | 注视热力图（2D 高斯核密度）；扫视路径图（连线+注视圆）；AOI 统计（注视时长/次数/首次注视时间/回视次数）；瞳孔响应分析（任务锁定平均）；阅读模式分析（跳读/回视指标） |
| **输入格式** | `StandardizedSession.eye_clean` (DataFrame) |
| **输出格式** | `analysis_results.eye`: `{heatmap_path, scanpath_path, aoi_stats, pupil_data}` |
| **核心依赖** | `matplotlib`, `seaborn`, `numpy` |
| **对标 EthoClaw** | `ethoclaw-trajectory-velocity-heatmap-generate` — 同为轨迹热力图生成，但针对注视点而非动物位置 |

### 12. psyphiclaw-physio-analysis

| 属性 | 说明 |
|------|------|
| **名称** | 生理信号分析 |
| **描述** | ECG/EDA/EMG 的特征提取与量化分析 |
| **功能** | HRV 分析（时域: SDNN/RMSSD; 频域: HF/LF/HF功率; 非线性: 熵/SD1/SD2）；EDA 响应分析（SCR 数量/幅度/恢复时间）；EMG 活动量分析（均方根值/积分值/爆发次数）；综合压力指数计算（加权融合多项指标） |
| **输入格式** | `StandardizedSession.physio_features` (DataFrame) |
| **输出格式** | `analysis_results.physio`: `{hrv_data, eda_data, emg_data, stress_index}` |
| **核心依赖** | `neurokit2`, `scipy`, `numpy` |
| **对标 EthoClaw** | `ethoclaw-kinematic-parameter-generator` — 同为行为参数的量化提取 |

---

## AI 洞察 + 可视化类 (2个)

### 13. psyphiclaw-fusion-insight ⭐ 核心差异化 Skill

| 属性 | 说明 |
|------|------|
| **名称** | 多模态融合洞察 |
| **描述** | **PsyPhiClaw 的核心差异化能力**。跨模态时间对齐 + 关联分析 + 异常检测 + LLM 洞察生成 |
| **功能** | ① 多模态时间对齐（统一时间基线 + 滑动窗口分割）；② 跨模态关联分析（Pearson/互信息/时滞交叉相关/Granger 因果）；③ 异常模式检测（Z-score 偏离 + 规则引擎 + 孤立森林）；④ LLM 自然语言洞察生成（结构化 Prompt + 护栏机制）；⑤ 洞察置信度评估 |
| **输入格式** | `StandardizedSession`（含全部已处理的模态数据） |
| **输出格式** | `{aligned_data, correlations, anomalies: [{time_range, type, severity, evidence}], insights: [{text, confidence, evidence}], confidence_scores, figures}` |
| **核心依赖** | `pandas`, `scipy`, `numpy`, `scikit-learn` |
| **对标 EthoClaw** | **无对标 — PsyPhiClaw 首创**。EthoClaw 不支持多模态融合分析 |

### 14. psyphiclaw-visualization

| 属性 | 说明 |
|------|------|
| **名称** | 论文级多模态可视化 |
| **描述** | 生成高质量的分析图表和论文 Figure |
| **功能** | 多模态时间轴对齐图（多面板堆叠）；小提琴图/箱线图（组间比较）；雷达图（多维度综合评估）；聚类热力图（被试/条件聚类）；论文 Figure 自动布局（子图排列、标注、配色方案）；Plotly 交互式 HTML（探索用）；Matplotlib PDF/SVG（论文用） |
| **输入格式** | `analysis_results`（所有 Skill 的分析结果） |
| **输出格式** | `{figures: [路径列表], html_path: str (交互式), layout_config: dict}` |
| **核心依赖** | `matplotlib`, `seaborn`, `plotly`, `numpy` |
| **对标 EthoClaw** | `ethoclaw-paper-figure-layout` — 同为论文级图表布局生成 |

---

## 辅助类 Skill（Phase 3，沿用 EthoClaw）

### psyphiclaw-literature-search

| 属性 | 说明 |
|------|------|
| **描述** | 每日 arXiv/PubMed 关键词论文推送，支持自定义检索词 |
| **对标 EthoClaw** | `ethoclaw-daily-paper`（直接复用） |

### psyphiclaw-pdf-research

| 属性 | 说明 |
|------|------|
| **描述** | 本地 PDF 论文阅读、摘要生成、关键信息提取 |
| **对标 EthoClaw** | `ethoclaw-pdf-research`（直接复用） |

### psyphiclaw-analysis-report

| 属性 | 说明 |
|------|------|
| **描述** | 自动生成完整分析报告（实验背景、方法、结果、结论） |
| **对标 EthoClaw** | `ethoclaw-analysis-report`（扩展为多模态） |

---

## 依赖总览

```
核心依赖:
  pandas >= 2.0        # 数据结构
  numpy >= 1.24        # 数值计算
  scipy >= 1.11        # 信号处理
  mne >= 1.6           # EEG/fNIRS 分析
  mne-nirs >= 0.6      # fNIRS 支持
  neurokit2 >= 0.2     # 生理信号分析
  scikit-learn >= 1.3  # ICA、机器学习
  matplotlib >= 3.8    # 静态图表
  seaborn >= 0.13      # 统计可视化
  plotly >= 5.18       # 交互式可视化
  jinja2 >= 3.1        # 报告模板
```

---

*PsyPhiClaw Skill Index v1.0 — 14 Skills, 1 Vision.*
