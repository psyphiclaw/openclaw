# 📚 PsyPhiClaw Daily Paper

## Overview

自动检索多模态行为分析相关文献。支持 PubMed、arXiv、Semantic Scholar 搜索，合并去重，生成每日 Top-5 论文摘要。

## Pipeline

```
PubMed ──┐
arXiv   ──┼──→ merge_results.py → dedup → sort → build_top5_digest.py → Markdown
S2      ──┘
```

## Scripts

| Script | Purpose |
|--------|---------|
| `search_pubmed.py` | PubMed E-utilities API |
| `search_arxiv.py` | arXiv Atom API |
| `search_semantic_scholar.py` | Semantic Scholar API |
| `merge_results.py` | 合并去重排序 |
| `build_top5_digest.py` | 生成 Top-5 Markdown 摘要 |

## Quick Start

```bash
# 搜索所有来源
python scripts/search_pubmed.py --query "multimodal AND (EEG OR eye-tracking)" --max 20 -o pubmed.json
python scripts/search_arxiv.py --query "multimodal emotion recognition" --max 20 -o arxiv.json
python scripts/search_semantic_scholar.py --query "EEG facial expression fusion" --max 20 -o s2.json

# 合并去重
python scripts/merge_results.py pubmed.json arxiv.json s2.json -o merged.json

# 生成摘要
python scripts/build_top5_digest.py merged.json -o digest.md
```

## Configuration

See `assets/config.template.yaml` for keyword lists, cron settings, and push channels.

## Dependencies

```bash
pip install requests biopython arxiv feedparser
```
