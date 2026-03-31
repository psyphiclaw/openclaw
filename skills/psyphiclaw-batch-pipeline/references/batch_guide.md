# 批量处理最佳实践

## 推荐目录结构

```
project/
├── manifest.json              # 项目清单（scan_project 生成）
├── pipeline.yaml              # 分析流程配置
├── data/                      # 原始数据
│   ├── sub-001/
│   │   ├── sub-001_eeg.bdf
│   │   └── sub-001_behavioral.csv
│   ├── sub-002/
│   └── sub-003/
├── imported/                  # 导入后数据（batch_import 输出）
│   ├── 001/
│   │   ├── sub-001_eeg.parquet
│   │   └── sub-001_behavioral.parquet
│   └── ...
├── results/                   # 分析结果（batch_analyze 输出）
│   ├── 001/
│   │   ├── preprocess/.completed
│   │   ├── features/.completed
│   │   └── visualize/
│   │       └── sub-001_erps.png
│   └── ...
├── reports/                   # 汇总报告（batch_report 输出）
│   ├── index.html
│   ├── group_stats.csv
│   └── figures/
└── logs/                      # 日志
    ├── import_errors.json
    └── analysis_errors.json
```

## 命名规范

### 被试 ID
- 格式: `001`, `002`, ... 或 `sub-001`, `sub-002`
- 统一使用三位数字编号
- 文件名模板: `{subject_id}_{modality}_{description}.{ext}`

### 文件命名示例
```
sub-001_eeg_raw.bdf
sub-001_eeg_preprocessed.parquet
sub-001_behavioral_trials.csv
sub-001_features_erp.csv
sub-001_figure_erps.png
```

### 模态缩写
| 模态 | 缩写 | 扩展名 |
|------|------|--------|
| 脑电 | eeg | .bdf, .edf, .vhdr, .set |
| 眼动 | eye | .ts, .asc |
| 肌电 | emg | .csv |
| 行为 | beh | .csv, .xlsx |
| 视频 | vid | .avi, .mp4 |

## 批处理最佳实践

### 1. 先扫描再处理
```bash
# 始终先扫描，确认数据完整性
python scan_project.py --project-dir project/ --output manifest.json
# 检查输出，确认文件数和被试数符合预期
```

### 2. 使用断点续跑
```bash
# 大规模分析务必加 --skip-completed
python batch_analyze.py --project-dir project/ --pipeline pipeline.yaml \
  --output-dir results/ --skip-completed
```

### 3. 并行度控制
- CPU 密集型（特征提取、统计）: workers = CPU 核数 - 1
- I/O 密集型（导入、导出）: workers = CPU 核数 × 2
- 内存受限: workers = 1-2，避免 OOM

### 4. 错误处理
- 始终加 `--continue-on-error`（batch_import）或 `--skip-completed`（batch_analyze）
- 检查 `*_errors.json` 日志文件
- 单个被试失败不应阻塞整个流程

### 5. 数据备份
- 导入前备份原始 `data/` 目录
- 导入后数据使用 parquet 格式（高效、自描述）
- 分析结果保留中间文件，便于调试

### 6. 质量控制
- 导入后验证记录数是否一致
- 每个被试检查数据完整性
- 组统计检查异常值
