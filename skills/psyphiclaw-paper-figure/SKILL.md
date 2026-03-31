---
name: psyphiclaw-paper-figure
description: >
  论文级配图自动排版。将多个分析结果图表自动排版为符合期刊要求的多面板图，
  支持 Nature/Science/APA/IEEE 等格式模板。
metadata:
  openclaw:
    emoji: "📐"
    requires:
      python: ["matplotlib", "numpy", "pandas", "Pillow"]
---

# psyphiclaw-paper-figure

论文级配图自动排版技能。将分析结果图表组合为符合期刊要求的多面板图。

## 功能

| 脚本 | 用途 |
|------|------|
| `scripts/layout_results_foldered.py` | 扫描文件夹，按模态分类，自动生成多面板组合图 |
| `scripts/create_multi_panel.py` | 用 GridSpec 创建自定义多面板图，支持标签和配色 |
| `scripts/export_figure.py` | 按期刊要求导出图表（PDF/PNG/SVG），预设模板 |

## 快速开始

```bash
# 扫描结果文件夹，生成 2×3 grid 多面板图
python scripts/layout_results_foldered.py --input-dir results/ --output-dir figures/ --layout grid

# 手动指定子图
python scripts/create_multi_panel.py --images a.png b.png c.png d.png --layout 2x2 --labels A B C D --output figure.pdf

# 按 Nature 格式导出
python scripts/export_figure.py --input figure_raw.png --output figure_nature.pdf --journal nature --dpi 600
```

## 期刊模板

预置 Nature、Science、APA、IEEE 四种模板，定义在 `assets/journal_templates.json`。

## 参考

详见 `references/layout_guide.md`：各期刊图表要求汇总、多面板图最佳实践、色盲友好方案。
