---
name: psyphiclaw-batch-pipeline
description: >
  批量处理管道。自动化处理多个被试的数据（导入→对齐→分析→可视化→报告），
  支持并行处理和断点续跑。
metadata:
  openclaw:
    emoji: "⚡"
    requires:
      python: ["pandas", "numpy", "pathlib"]
---

# psyphiclaw-batch-pipeline

批量处理管道技能。自动化处理多被试实验数据的完整流程。

## 流程

```
扫描项目 → 批量导入 → 批量分析 → 批量报告
scan_project → batch_import → batch_analyze → batch_report
```

## 功能

| 脚本 | 用途 |
|------|------|
| `scripts/scan_project.py` | 扫描实验目录，自动检测数据文件，生成 manifest |
| `scripts/batch_import.py` | 批量导入被试数据，支持并行处理 |
| `scripts/batch_analyze.py` | 按 YAML pipeline 配置批量执行分析，支持断点续跑 |
| `scripts/batch_report.py` | 生成汇总报告和 HTML 仪表盘索引页 |

## 快速开始

```bash
# 1. 扫描项目目录
python scripts/scan_project.py --project-dir data/my_exp/ --output manifest.json

# 2. 批量导入
python scripts/batch_import.py --project-dir data/my_exp/ --manifest manifest.json --output-dir imported/ --workers 4

# 3. 批量分析
python scripts/batch_analyze.py --project-dir data/my_exp/ --pipeline pipeline.yaml --output-dir results/ --skip-completed

# 4. 生成报告
python scripts/batch_report.py --project-dir data/my_exp/ --output-dir reports/ --title "My Experiment"
```

## 参考

详见 `references/batch_guide.md`：推荐目录结构、命名规范、批处理最佳实践。
