# 同步协议参考 (Sync Protocols Reference)

## 1. TTL 触发同步

### 原理
TTL (Transistor-Transistor Logic) 脉冲是一种常见的硬件同步信号。实验时，刺激呈现软件通过并行端口或 USB 接口向所有采集设备同时发送一个短脉冲（通常 1-5ms），各设备记录下脉冲到达的精确时间。

### 布线方式
```
刺激电脑 ──并行端口/USB──► TTL 分配器 ──► EEG 放大器 (标记 trigger)
                                   │
                                   ├─► FaceReader (视频帧标记)
                                   └─► 生理信号采集器 (事件输入)
```

### 最佳实践
1. **脉冲宽度**: 1-5ms，太短可能被漏检，太长会干扰信号
2. **脉冲编号**: 每个实验条件使用不同 trigger code（如 1=刺激A, 2=刺激B）
3. **基线 trigger**: 实验开始前发送一次 baseline trigger 用于初始对齐
4. **验证**: 实验后检查各设备的 trigger 计数是否一致
5. **精度**: 典型 TTL 延迟 < 1ms

### 常见问题
| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Trigger 数量不匹配 | 信号线接触不良/衰减 | 检查 BNC 连接，降低 cable 长度 |
| 偏移不稳定 (>5ms) | USB 延迟 | 使用并行端口或专用 trigger 接口 |
| 编号错乱 | 并发 trigger 发送过快 | 每个 trigger 间隔 ≥ 100ms |

---

## 2. LSL (Lab Streaming Layer)

### 概述
Lab Streaming Layer 是一个跨平台的实时数据流同步框架，通过网络时间协议实现亚毫秒级同步。

### 核心概念
- **Outlet**: 数据源（如 EEG 放大器驱动）
- **Inlet**: 数据接收端（如记录软件）
- **Stream**: 带类型标签的数据通道
- **Clock Synchronization**: 基于 NTP 的时钟校正

### 典型 Stream 类型
```python
# EEG stream
info = StreamInfo('EEG', 'EEG', 64, 250, 'float32', 'eeg1234')

# Marker stream (trigger events)
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'markers1234')
```

### 同步精度
- 局域网: 1-5ms
- 同一台机器: < 1ms
- 跨网段: 5-20ms

### 何时使用
- 无法物理连接 TTL 线的场景（如远程采集）
- 需要实时同步显示的场景
- 多台电脑协作采集

---

## 3. 事件标记对齐

### 适用场景
当无法使用硬件 TTL 同步时，利用共享的实验事件作为时间锚点。

### 常用事件类型
| 事件 | 来源 | 精度 |
|------|------|------|
| 刺激呈现 | E-Prime/Psychopy 日志 | ±16ms (60Hz 显示) |
| 音频 onset | 音频文件时间戳 | ±1ms |
| 按键响应 | 行为软件记录 | ±1ms |
| 视频帧标记 | FaceReader 帧号 | ±33ms (30fps) |

### 对齐方法
1. **最近邻匹配**: 找到最接近事件时间的数据点
2. **插值匹配**: 在事件时间点做线性插值
3. **中值偏移**: 跨多个事件计算稳定偏移量

---

## 4. 手动时间偏移校准

### 步骤
1. 在数据中找到同步点（如屏幕闪烁的视觉标记）
2. 记录各模态中该点的时间
3. 计算相对偏移: `offset = modality_time - reference_time`
4. 验证: 检查多个同步点的偏移是否一致

### 精度限制
- 依赖人工识别 → 通常 ±1 个采样周期
- FaceReader 视频帧 → ±33ms
- 行为日志 → 取决于软件精度

---

## 5. 采样率转换

### 上采样 (Upsampling)
- 线性插值: 适用于平滑信号（表情强度、生理指标）
- 样条插值: 更平滑但可能引入伪影

### 下采样 (Downsampling)
- ⚠️ 必须先低通滤波（抗混叠滤波器）
- 滤波截止频率 = 目标奈奎斯特频率 (target_freq / 2)
- EEG 推荐使用 MNE 的 `raw.resample()` (自动包含滤波)

---

## 6. PsyPhiClaw 推荐

| 场景 | 推荐方法 |
|------|----------|
| 实验室 EEG + FaceReader + BIOPAC | TTL 触发 (⭐ 首选) |
| 移动 EEG + 手机 APP | LSL 或事件标记 |
| 已有数据回溯分析 | 事件标记 + 手动校准 |
| 实时多设备 | LSL |
