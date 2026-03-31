# 📄 PsyPhiClaw PDF Research

## Overview

本地 PDF 文献批量阅读和摘要。提取论文标题、作者、摘要、方法、结果、结论，生成结构化摘要报告和阅读日志。

## Pipeline

```
PDF files → extract_pdf_bundle.py → structured JSON → build_summary_md.py → Markdown 摘要
                                                         ↓
                                              build_research_log.py → 阅读日志
```

## Scripts

| Script | Purpose |
|--------|---------|
| `extract_pdf_bundle.py` | 批量提取 PDF 文本，识别论文结构，输出 JSON |
| `build_summary_md.py` | JSON → Markdown 结构化摘要 |
| `build_research_log.py` | 维护文献阅读日志 |

## Quick Start

```bash
# 批量提取
python scripts/extract_pdf_bundle.py --input papers/ --output extracted.json

# 生成摘要
python scripts/build_summary_md.py --input extracted.json --output summaries/

# 更新阅读日志
python scripts/build_research_log.py --input extracted.json --log research_log.json --output log.md
```

## Dependencies

```bash
pip install PyPDF2 pdfplumber pandas
```
