# Visualization Templates & Guidelines

## Color Palette

### Primary (色盲友好)
- Blue: `#4A90D9`
- Red: `#E74C3C`
- Green: `#27AE60`
- Orange: `#F39C12`
- Purple: `#8E44AD`
- Gray: `#7F8C8D`

### Sequential (热力图)
- `YlOrRd` (黄-橙-红): 适合正相关
- `YlGnBu` (黄-绿-蓝): 适合负相关
- `viridis`: 通用，色盲友好

### Diverging (相关性矩阵)
- `RdBu_r`: 红-蓝反转，0 为白色
- `coolwarm`: 蓝-白-红

### Categorical (多组对比)
- `Set2`, `tab10`: 区分度高

## Chart Type Selection Guide

| 场景 | 推荐图表 | 工具 |
|------|----------|------|
| 多模态时间线概览 | 多子图时间序列 | Plotly (交互) / Matplotlib (论文) |
| 单模态信号预览 | 线图 | Matplotlib |
| ERP 波形对比 | 线图 + SEM 阴影 | Plotly / Matplotlib |
| 头皮拓扑分布 | 拓扑图 (topomap) | MNE + Matplotlib |
| 两组均值对比 | 小提琴图 / 箱线图 | Seaborn / Plotly |
| 多组多维度对比 | 雷达图 | Matplotlib / Plotly |
| 特征-时间矩阵 | 热力图 | Seaborn / Plotly |
| 相关性矩阵 | 热力图 + 注释 | Seaborn |
| 动态相关性变化 | 滑动窗口折线图 | Plotly |
| 条件间差异 | 条形图 + 误差棒 | Matplotlib / Seaborn |

## Journal Format Recommendations

### Nature / Science
- Font: Arial/Helvetica, 6-8pt
- Figure width: 89mm (single) / 183mm (double)
- DPI: 300+ (raster) or vector (PDF/SVG)
- Color: 色盲友好，避免红绿对比

### APA (心理学期刊)
- Font: Times New Roman, 12pt
- Figure width: ~6 inches
- 标注: 无标题，图注放在图下方
- 统计标注: *p < .05, **p < .01, ***p < .001

### IEEE (工程类)
- Font: Times New Roman, 10pt
- Figure width: 3.5 inches (single column)
- 线宽: 1.5pt minimum

## Plotly vs Matplotlib

### Use Plotly when:
- 需要交互式探索（缩放、悬停、选区）
- 数据量大（>10k 数据点）
- 输出 HTML 报告
- 需要分享给非技术用户

### Use Matplotlib when:
- 论文投稿（需要矢量 PDF/EPS）
- 精确控制每个元素
- 与 MNE/Seaborn 深度集成
- 需要一致的学术风格

### Dual Output Pattern
所有 visualization 脚本默认同时输出：
1. `*.png` — 静态截图 (300 DPI)
2. `*.html` — 交互式 (Plotly)

## Multi-Modal Timeline Layout

标准子图布局（从上到下）：
1. 事件标记 (Event markers) — 垂直虚线
2. EEG 信号 (选择代表性通道，如 Fz, Cz, Pz)
3. 表情维度 (Valence / Arousal / Dominance)
4. 生理信号 (HR / EDA / Respiration)
5. 眼动数据 (Pupil diameter / Fixation)

时间轴对齐，共享 X 轴，不同模态用不同 Y 轴范围。
