# PsyPhiClaw 技术架构设计文档

> **版本**: v1.0-draft | **日期**: 2026-03-31

---

## 目录

1. [系统分层架构](#1-系统分层架构)
2. [数据流图](#2-数据流图)
3. [Skill 接口设计](#3-skill-接口设计)
4. [时间同步协议](#4-时间同步协议)
5. [数据标准化方案](#5-数据标准化方案)
6. [AI 洞察引擎设计](#6-ai-洞察引擎设计)

---

## 1. 系统分层架构

### 1.1 四层架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Layer 4: 可视化与报告层                            │
│    psyphiclaw-visualization │ psyphiclaw-report                         │
│    Plotly / Matplotlib / Markdown / PDF                                │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│                        Layer 3: AI 洞察层                                │
│    psyphiclaw-fusion-insight                                             │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │
│    │ 时间对齐  │ │ 关联分析  │ │ 异常检测  │ │ LLM 生成 │                 │
│    └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│                        Layer 2: 分析处理层                                │
│    psyphiclaw-eeg-analysis │ psyphiclaw-eye-analysis                     │
│    psyphiclaw-physio-analysis │ psyphiclaw-preprocess-*                  │
│    MNE-Python │ NeuroKit2 │ 自研算法                                     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│                     Layer 1: 数据接入 + 标准化层                          │
│    psyphiclaw-face-import │ psyphiclaw-eye-import                        │
│    psyphiclaw-eeg-import │ psyphiclaw-physio-import                      │
│    psyphiclaw-fnirs-import                                              │
│    ──→ StandardizedSession (统一数据模型)                                 │
└──────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│                     Layer 0: 基础设施层                                   │
│    OpenClaw Agent Framework │ Conda 环境 │ LSL │ 文件系统                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 各层职责

| 层次 | 职责 | 关键组件 |
|------|------|----------|
| Layer 0 | 运行环境、Agent 框架、文件 I/O、时间同步基础设施 | OpenClaw, Conda, LSL |
| Layer 1 | 原始数据解析、格式转换、标准化为统一数据模型 | Import Skills, StandardizedSession |
| Layer 2 | 单模态预处理与分析计算 | Preprocess Skills, Analysis Skills |
| Layer 3 | 跨模态融合、异常检测、AI 洞察生成 | Fusion Insight Skill |
| Layer 4 | 结果可视化、报告生成、输出 | Visualization Skill, Report Skill |

---

## 2. 数据流图

### 2.1 完整数据流

```
                        ┌─────────────────┐
                        │   用户自然语言    │
                        │   "分析受试者01" │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    OpenClaw LLM 路由     │
                    │  解析意图 → 选择 Skill    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
    │ psyphiclaw-     │ │ psyphiclaw-  │ │ psyphiclaw-  │
    │ face-import     │ │ eeg-import   │ │ physio-import│
    └────────┬────────┘ └──────┬───────┘ └──────┬───────┘
             │                 │                 │
             ▼                 ▼                 ▼
    ┌──────────────────────────────────────────────────┐
    │          StandardizedSession (统一数据模型)         │
    │  ┌─────────────────────────────────────────────┐ │
    │  │ face_df:   [timestamp, AU*, valence, ...]   │ │
    │  │ eye_df:    [timestamp, x, y, pupil, aoi, ...]│ │
    │  │ eeg_raw:   MNE Raw (channels × samples)     │ │
    │  │ physio_df: [timestamp, ecg, eda, emg, ...]   │ │
    │  │ fnirs_raw: MNE Raw (chroma × samples)        │ │
    │  │ events:    [onset, duration, label]           │ │
    │  └─────────────────────────────────────────────┘ │
    └─────────────────────┬────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │  psyphiclaw-preprocess-* │
              │  各模态独立预处理       │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  psyphiclaw-*-analysis │
              │  各模态独立分析         │
              └───────────┬───────────┘
                          │
              ┌───────────▼──────────────────┐
              │  psyphiclaw-fusion-insight    │
              │  ① 时间对齐                   │
              │  ② 跨模态关联分析              │
              │  ③ 异常模式检测                │
              │  ④ LLM 洞察生成               │
              └───────────┬──────────────────┘
                          │
              ┌───────────▼───────────┐
              │  psyphiclaw-report     │
              │  生成分析报告           │
              └───────────┬───────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  报告 + 图表 + 数据  │
              │  Markdown / PDF     │
              └─────────────────────┘
```

### 2.2 实时数据流（未来扩展）

```
实验设备 ──→ LSL Streams ──→ psyphiclaw-lsl-receiver ──→ StandardizedSession
     │                                           │
     ├── FaceReader LSL outlet          实时缓冲队列
     ├── Tobii LSL outlet                       │
     ├── EEG Amplifier LSL outlet               ▼
     ├── Biopac LSL outlet            psyphiclaw-live-monitor
     └── fNIRS LSL outlet             (实时可视化 + 预警)
```

> 实时数据流作为 v2.0 规划，v1.0 聚焦离线分析。

---

## 3. Skill 接口设计

### 3.1 通用接口约定

所有 Skill 遵循统一的输入输出约定：

```python
# 通用输入
{
    "file_path": str,           # 原始数据文件路径
    "session_id": str,          # 实验会话标识
    "config": dict,             # 模态特定配置参数
    "output_dir": str           # 输出目录
}

# 通用输出
{
    "status": "success" | "error",
    "session_id": str,
    "result_path": str,         # 结果文件路径
    "summary": dict,            # 结果摘要（供 LLM 读取）
    "messages": list[str]       # 给用户的自然语言说明
}
```

### 3.2 数据接入 Skill

#### psyphiclaw-face-import

```
输入:
  file_path: FaceReader CSV 导出文件
  config:
    encoding: "utf-8" | "gbk"
    timestamp_column: "TimeStamp"          # 时间戳列名
    au_columns: ["AU1", "AU4", "AU6", ...] # 要导入的 AU 列
    emotion_columns: ["Valence", "Arousal", ...]
    
输出:
  StandardizedSession.face_df: DataFrame
    columns: [timestamp, au_1, au_4, au_6, valence, arousal, 
              head_orientation_x/y/z, ...]
  StandardizedSession.meta.face_source: "FaceReader CSV"
  
依赖: pandas
```

#### psyphiclaw-eye-import

```
输入:
  file_path: Tobii TSV / EyeLink ASC 文件
  config:
    format: "tobii" | "eyelink"
    sample_rate: float | "auto"
    aoi_definitions: [{name, type, coordinates}, ...]  # 可选
    
输出:
  StandardizedSession.eye_df: DataFrame
    columns: [timestamp, left_x, left_y, right_x, right_y, 
              left_pupil, right_pupil, gaze_x, gaze_y, 
              event_type("fixation"/"saccade"/"blink"), ...]
  StandardizedSession.meta.eye_source: str
  
依赖: pandas
```

#### psyphiclaw-eeg-import

```
输入:
  file_path: BrainVision (.vhdr/.eeg/.vmrk) / EGI / EDF 文件
  config:
    format: "brainvision" | "egi" | "edf"
    preload: bool = True
    reference: str = "average" | "linked_mastoid"
    
输出:
  StandardizedSession.eeg_raw: mne.io.Raw
  StandardizedSession.events: ndarray (onset, duration, label)
  StandardizedSession.meta.eeg_source: str
  StandardizedSession.meta.sfreq: float
  StandardizedSession.meta.eeg_channels: list[str]
  
依赖: mne, numpy
```

#### psyphiclaw-physio-import

```
输入:
  file_path: Biopac AcqKnowledge 导出 CSV
  config:
    channels: ["ECG", "EDA", "EMG_zygomatic", "EMG_corrugator", "Resp", ...]
    sample_rate: float | "auto"
    
输出:
  StandardizedSession.physio_df: DataFrame
    columns: [timestamp, ecg, eda, emg_zygomatic, emg_corrugator, respiration, ...]
  StandardizedSession.meta.physio_source: str
  StandardizedSession.meta.physio_channels: list[str]
  
依赖: pandas, scipy
```

#### psyphiclaw-fnirs-import

```
输入:
  file_path: Homer3 / NIRX 数据目录
  config:
    format: "homer3" | "nirx"
    
输出:
  StandardizedSession.fnirs_raw: mne.io.Raw
  StandardizedSession.meta.fnirs_source: str
  StandardizedSession.meta.fnirs_channels: list[str]
  
依赖: mne, mne_nirs, numpy
```

### 3.3 预处理 Skill

#### psyphiclaw-preprocess-face

```
输入: StandardizedSession.face_df
输出: StandardizedSession.face_clean (DataFrame)
  处理: 线性插值缺失帧 → Savitzky-Golay 平滑 → Z-score 异常值裁剪 → 时长截取
依赖: pandas, scipy
```

#### psyphiclaw-preprocess-eye

```
输入: StandardizedSession.eye_df
输出: StandardizedSession.eye_clean (DataFrame)
  处理: 眨眼检测+线性插值 → 基于速度的注视/扫视分类 → 低质量采样过滤 → AOI 匹配
依赖: pandas, scipy, numpy
```

#### psyphiclaw-preprocess-eeg

```
输入: StandardizedSession.eeg_raw
输出: StandardizedSession.eeg_epochs (mne.Epochs)
  处理: 通道选择 → 带通滤波(1-40Hz) → 陷波(50/60Hz) → ICA 去眼电 → 重参考 → 基线校正 → 分段
依赖: mne, sklearn
```

#### psyphiclaw-preprocess-physio

```
输入: StandardizedSession.physio_df
输出: StandardizedSession.physio_features (DataFrame)
  处理: ECG → R峰检测 + IBI 序列; EDA → 低通滤波 + 峰值检测; EMG → 整流 + RMS 平滑
依赖: neurokit2, scipy
```

#### psyphiclaw-preprocess-fnirs

```
输入: StandardizedSession.fnirs_raw
输出: StandardizedSession.fnirs_od (mne.io.Raw)  # 光学密度 → 血氧浓度
  处理: 运动伪迹校正(TDDR/Spline) → Beer-Lambert 转换 → 频率滤波(0.01-0.2Hz) → 基线校正
依赖: mne_nirs, scipy
```

### 3.4 分析 Skill

#### psyphiclaw-eeg-analysis

```
输入: StandardizedSession.eeg_epochs
输出:
  erp_data: dict  {condition: {channel: array}}
  power_data: dict {band: {channel: value}}  (delta/theta/alpha/beta/gamma)
  connectivity: dict {method: matrix}
  source_estimates: mne.SourceEstimate (可选)
  figures: list[str]  # 图表路径
依赖: mne, numpy, scipy
```

#### psyphiclaw-eye-analysis

```
输入: StandardizedSession.eye_clean
输出:
  heatmap_path: str  # 注视热力图
  scanpath_path: str  # 扫视路径图
  aoi_stats: DataFrame  # 各 AOI 的注视时长、次数、首次注视时间
  pupil_data: DataFrame  # 瞳孔直径时序（任务锁定）
依赖: matplotlib, seaborn, numpy
```

#### psyphiclaw-physio-analysis

```
输入: StandardizedSession.physio_features
输出:
  hrv_data: dict  {sdnn, rmssd, hf_power, lf_hf_ratio, ...}
  eda_data: dict  {peak_count, peak_amplitude_mean, SCR_onset, ...}
  emg_data: dict  {channel: {mean_activation, peak_activation, ...}}
  stress_index: float  # 综合压力指数 (0-100)
依赖: neurokit2, scipy, numpy
```

### 3.5 AI 洞察 Skill

#### psyphiclaw-fusion-insight ⭐ 核心

```
输入: StandardizedSession (全部模态数据)
输出:
  aligned_data: dict  # 对齐后的多模态时间序列
  correlations: DataFrame  # 跨模态相关性矩阵
  anomalies: list[dict]  # [{time_range, type, modalities, description, confidence}]
  insights: list[str]    # LLM 生成的自然语言洞察
  confidence_scores: list[float]  # 每条洞察的置信度
  figures: list[str]  # 多模态对齐可视化图

处理流程:
  ① 时间对齐 → 统一到最高公共采样率的 DataFrame
  ② 滑动窗口分割 → 按事件/固定窗口切分
  ③ 关联分析 → 皮尔逊/互信息/时滞交叉相关
  ④ 异常检测 → Z-score + 规则引擎
  ⑤ LLM 洞察 → 结构化 Prompt → 护栏过滤 → 输出

依赖: pandas, scipy, numpy, scikit-learn
```

#### psyphiclaw-report

```
输入: psyphiclaw-fusion-insight 的全部输出
输出:
  report_path: str  # Markdown 格式分析报告
  pdf_path: str     # PDF 格式（可选）
  
报告结构:
  1. 实验概述（被试信息、任务描述、模态覆盖）
  2. 数据质量报告（各模态数据完整性、预处理摘要）
  3. 单模态分析结果
  4. 多模态融合洞察（关联分析 + 异常检测 + AI 洞察）
  5. 结论与建议
  6. 附录（方法说明、参数配置）

依赖: jinja2, markdown, weasyprint
```

---

## 4. 时间同步协议

### 4.1 三种同步策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    时间同步协议选择树                              │
│                                                                 │
│  实验设备是否支持 LSL？                                          │
│    ├── 是 → 使用 LSL 时间戳（精度 <1ms）                         │
│    │        所有流自动对齐到 LSL 时钟                             │
│    │                                                            │
│    └── 否 → 是否有 TTL 触发线？                                  │
│              ├── 是 → 使用 TTL 事件标记（精度 ~1ms）              │
│              │        各设备记录 TTL 事件对应的时间点              │
│              │        以 EEG TTL 为主时钟，其他设备对齐            │
│              │                                                  │
│              └── 否 → 使用手动标记（精度 ~100ms-1s）              │
│                       实验开始/结束手动按键标记                    │
│                       各设备数据按标记点对齐                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 TTL 同步流程

```
                    ┌──────────┐
                    │  刺激软件  │
                    │(E-Prime/  │
                    │ PsychoPy) │
                    └─────┬────┘
                          │ TTL 脉冲
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ EEG 放大器│ │ Biopac   │ │ EyeLink  │
        │ (标记通道)│ │ (数字输入)│ │ (端口输入)│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            ▼            ▼
        TTL 事件日志   TTL 事件日志   TTL 事件日志
             │            │            │
             └────────────┼────────────┘
                          ▼
                  ┌──────────────┐
                  │ 对齐引擎      │
                  │ 以 EEG 为基准 │
                  │ 时间差补偿    │
                  └──────────────┘
```

### 4.3 时间戳精度评估

同步完成后，系统自动计算并报告时间同步精度：

```python
def evaluate_sync_quality(session: StandardizedSession) -> dict:
    """
    评估时间同步质量
    返回:
      sync_method: 使用的同步方法
      estimated_precision_ms: 估计精度（毫秒）
      drift_detected: bool  # 是否检测到时钟漂移
      drift_rate_hz: float  # 漂移速率（Hz）
      recommendation: str   # 对精度的评估和建议
    """
```

| 同步方法 | 典型精度 | 适用场景 |
|----------|----------|----------|
| LSL | <1ms | 实时采集，所有设备支持 LSL |
| TTL | ~1ms | 大部分实验室标准配置 |
| 手动标记 | 100ms-1s | 缺乏硬件同步的简易实验 |

---

## 5. 数据标准化方案

### 5.1 StandardizedSession 数据模型

```python
@dataclass
class StandardizedSession:
    """统一的实验会话数据模型"""
    
    session_id: str                    # 唯一会话标识
    subject_id: str = ""               # 被试编号
    
    # 原始数据
    face_df: Optional[pd.DataFrame] = None      # 面部表情数据
    eye_df: Optional[pd.DataFrame] = None       # 眼动数据
    eeg_raw: Optional[mne.io.Raw] = None        # EEG 原始数据
    physio_df: Optional[pd.DataFrame] = None    # 生理信号数据
    fnirs_raw: Optional[mne.io.Raw] = None      # fNIRS 原始数据
    
    # 事件标记
    events: Optional[np.ndarray] = None         # [onset, duration, label]
    event_id: Optional[dict] = None             # {事件名: 编号}
    
    # 处理后数据
    face_clean: Optional[pd.DataFrame] = None
    eye_clean: Optional[pd.DataFrame] = None
    eeg_epochs: Optional[mne.Epochs] = None
    physio_features: Optional[pd.DataFrame] = None
    fnirs_od: Optional[mne.io.Raw] = None
    
    # 分析结果
    analysis_results: dict = field(default_factory=dict)
    
    # 元数据
    meta: SessionMeta = field(default_factory=SessionMeta)
    
    # 时间同步信息
    sync_info: SyncInfo = field(default_factory=SyncInfo)
    
    def to_disk(self, path: str):
        """序列化保存到磁盘（使用 HDF5 + pickle）"""
        
    @classmethod
    def from_disk(cls, path: str) -> 'StandardizedSession':
        """从磁盘反序列化加载"""
```

### 5.2 时间戳统一方案

所有模态的时间戳统一为 **实验相对时间（秒）**，以实验开始时刻为 t=0：

| 模态 | 原始时间格式 | 转换规则 |
|------|-------------|----------|
| FaceReader | 绝对时间戳 (ms) | t = (timestamp - start_timestamp) / 1000 |
| Tobii | 绝对时间戳 (μs) | t = (timestamp - start_timestamp) / 1e6 |
| EEG | 采样点索引 | t = sample_index / sfreq |
| Biopac | 采样点索引 | t = sample_index / sfreq |
| fNIRS | 采样点索引 | t = sample_index / sfreq |

### 5.3 采样率对齐

融合分析时，将所有模态重采样到统一采样率：

```python
def align_sample_rates(
    session: StandardizedSession,
    target_rate: float = 10.0,  # 融合分析默认 10Hz
    method: str = "resample"    # "resample" | "interpolate"
) -> pd.DataFrame:
    """
    将各模态数据对齐到目标采样率
    - 表情/眼动: 下采样（均值聚合）
    - EEG: 低通滤波后下采样
    - 生理: 低通滤波后下采样
    - fNIRS: 低通滤波后下采样
    
    返回: 统一采样率的 DataFrame，每列为一个特征通道
    """
```

### 5.4 事件标记映射

```python
@dataclass
class SyncInfo:
    """时间同步信息"""
    sync_method: str           # "lsl" | "ttl" | "manual"
    master_clock: str          # 主时钟来源（默认 EEG）
    sync_events: list          # 同步事件列表 [{master_time, slave_time, ...}]
    estimated_precision_ms: float
    calibration_notes: str     # 校准说明
```

---

## 6. AI 洞察引擎设计

### 6.1 引擎架构

```
┌───────────────────────────────────────────────────────────────┐
│                    psyphiclaw-fusion-insight                    │
│                                                               │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐  │
│  │ Step 1  │───▶│ Step 2   │───▶│ Step 3   │───▶│ Step 4  │  │
│  │ 时间对齐 │    │ 关联分析  │    │ 异常检测  │    │ LLM生成 │  │
│  └─────────┘    └──────────┘    └──────────┘    └────┬────┘  │
│                                                      │       │
│                                               ┌──────▼─────┐ │
│                                               │ Step 5     │ │
│                                               │ 护栏校验    │ │
│                                               └──────┬─────┘ │
│                                                      │       │
└──────────────────────────────────────────────────────┼───────┘
                                                       ▼
                                              结构化洞察输出
```

### 6.2 Step 1: 时间对齐引擎

```python
class TimeAligner:
    """
    将多模态数据对齐到统一时间轴
    """
    def align(self, session, window_size=1.0, step_size=0.5) -> AlignedData:
        """
        滑动窗口对齐:
        1. 以事件标记为锚点，提取事件前后的时间窗口
        2. 对窗口内各模态数据聚合为特征向量
        3. 生成 AlignedData 对象
        
        AlignedData:
          windows: list[TimeWindow]
            TimeWindow:
              start: float
              end: float
              label: str
              features: {
                "face": {"valence": float, "arousal": float, "au_4": float, ...},
                "eye": {"fixation_duration": float, "pupil_diameter": float, ...},
                "eeg": {"alpha_power_Fz": float, "theta_power_Cz": float, ...},
                "physio": {"hr": float, "eda_level": float, "emg_corr": float, ...},
                "fnirs": {"hbo_F3": float, "hbr_F4": float, ...}
              }
        """
```

### 6.3 Step 2: 跨模态关联分析

```python
class CrossModalAnalyzer:
    """
    计算模态间的统计关联
    """
    def compute_correlations(self, aligned_data) -> CorrelationMatrix:
        """
        计算方法:
        1. Pearson 相关性（线性关系）
        2. Spearman 秩相关（单调关系）
        3. 互信息（非线性关系）
        4. 时滞交叉相关（因果关系方向）
        5. Granger 因果检验（预测关系）
        
        输出: 跨模态特征的相关性矩阵 + 统计显著性
        """
    
    def find_significant_pairs(self, alpha=0.05) -> list[SignificantPair]:
        """
        返回统计显著的跨模态关联对:
        [{feature_1, feature_2, method, statistic, p_value, effect_size, lag_ms}]
        """
```

### 6.4 Step 3: 异常模式检测

```python
class AnomalyDetector:
    """
    检测多模态数据中的异常模式
    """
    def detect(self, aligned_data, baseline_windows) -> list[Anomaly]:
        """
        检测方法:
        1. Z-score 基线偏离: 各特征相对基线的偏离程度
        2. 规则引擎: 预定义的异常模式（如 EDA 峰 + 负性情绪 > 阈值）
        3. 孤立森林: 多维特征空间中的异常点
        
        异常模式示例:
        - "认知过载": theta 功率 ↑ + EDA 峰 + 瞳孔 ↑ + 负性 AU ↑
        - "注意力分散": P300 振幅 ↓ + 注视偏离 AOI + 反应时 ↑
        - "情绪冲突": AU12(微笑)↑ + AU4(皱眉)↑ + HR ↑ + EDA ↑
        - "疲劳迹象": 眨眼频率 ↑ + 瞳孔 ↓ + alpha 功率 ↑ + 反应时 ↑
        
        输出: [{time_range, anomaly_type, modalities, severity, evidence}]
        """
```

### 6.5 Step 4: LLM 洞察生成

```python
class InsightGenerator:
    """
    基于分析结果生成自然语言洞察
    """
    INSIGHT_PROMPT = """
你是一个多模态行为分析专家。基于以下分析数据，生成清晰、准确的洞察报告。

## 实验信息
被试: {subject_id}
任务: {task_description}
时长: {duration}

## 统计关联（已通过显著性检验，p < 0.05）
{correlations}

## 检测到的异常模式
{anomalies}

## 单模态分析摘要
{modality_summaries}

## 输出要求
1. 每条洞察必须基于上述数据，不得编造
2. 标注支撑数据来源（如"基于 EDA 和面部表情数据"）
3. 置信度分级：高（多模态一致）、中（单模态统计显著）、低（趋势性发现）
4. 按重要性排序
5. 语言简洁专业，适合研究报告中使用
"""
    
    def generate(self, analysis_context) -> list[Insight]:
        """
        生成洞察，返回:
        [{text, confidence("high"|"medium"|"low"), 
          evidence: [{modality, metric, value}], 
          time_range}]
        """
```

### 6.6 Step 5: 护栏机制

```python
class InsightGuardrail:
    """
    防止 LLM 生成不可靠的洞察
    """
    def validate(self, insight: Insight, analysis_context) -> ValidationResult:
        """
        校验规则:
        1. 数据支撑检查: 每条洞察必须有对应的统计结果支撑
        2. 效应量检查: 仅报告达到最小效应量（Cohen's d > 0.3 或 r > 0.2）的发现
        3. 统计显著性: 相关性必须 p < 0.05（已校正）
        4. 因果措辞过滤: 不允许使用"导致"、"引起"等强因果措辞，
           替换为"与...相关"、"在...时同时观察到"
        5. 置信度校准: 高置信度至少需要 2 种模态的一致证据
        6. 幻觉检测: 洞察中引用的数据点必须在原始数据中可追溯
        """
    
    def filter_insights(self, insights: list[Insight]) -> list[Insight]:
        """
        过滤并标注:
        - 通过校验: 保留
        - 部分通过: 降级置信度 + 添加注意事项
        - 未通过: 丢弃 + 记录原因
        """
```

---

## 附录

### A. 数据格式速查表

| 模态 | 主要格式 | 解析库 |
|------|----------|--------|
| FaceReader | CSV (UTF-8/GBK) | pandas |
| Tobii | TSV (UTF-16) | pandas |
| EyeLink | ASC (ASCII) | 自研解析器 |
| BrainVision | .vhdr + .eeg + .vmrk | mne.io.read_raw_brainvision |
| EGI | .raw / .mff | mne.io.read_raw_egi |
| EDF | .edf / .bdf | mne.io.read_raw_edf |
| Biopac | CSV | pandas |
| fNIRS (Homer3) | .nirs / 目录 | mne_nirs |
| fNIRS (NIRX) | .hdr + .dat | mne_nirs |

### B. 推荐硬件配置

| 场景 | CPU | RAM | GPU | 存储 |
|------|-----|-----|-----|------|
| 轻量（单模态） | 4核 | 16GB | 不需要 | 100GB SSD |
| 标准（多模态） | 8核 | 32GB | 不需要 | 500GB SSD |
| 完整（含 ICA/溯源） | 16核 | 64GB | 可选(CUDA) | 1TB SSD |

---

*PsyPhiClaw Architecture v1.0 — 让多模态融合触手可及。*
