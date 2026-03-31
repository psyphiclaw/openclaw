# Output Formats & Patterns Reference

## Structured JSON Output Format

Each extracted paper follows this schema:

```json
{
  "file": "path/to/paper.pdf",
  "title": "Paper Title",
  "authors": "Author A, Author B, Author C",
  "journal": "Journal Name",
  "year": "2024",
  "n_pages": 12,
  "n_lines": 4500,
  "abstract": "Full abstract text...",
  "methods": "Methods section text...",
  "results": "Results section text...",
  "discussion": "Discussion section text...",
  "sections": {
    "abstract": "...",
    "introduction": "...",
    "methods": "...",
    "results": "...",
    "discussion": "...",
    "conclusion": "...",
    "references": "..."
  },
  "tables": [["cell1", "cell2", ...], ...]
}
```

## Paper Classification Tags

### By Methodology
| Tag | Description |
|-----|-------------|
| `eeg` | EEG-based study |
| `fnirs` | fNIRS study |
| `eye-tracking` | Eye-tracking study |
| `facial` | Facial expression analysis |
| `physiological` | General physiological measures |
| `self-report` | Questionnaire/scale based |
| `multimodal` | Uses 2+ modalities |
| `deep-learning` | Uses neural networks |
| `bc` | Brain-computer interface |

### By Domain
| Tag | Description |
|-----|-------------|
| `emotion` | Emotion/affect |
| `cognitive` | Cognitive processes |
| `clinical` | Clinical application |
| `ergonomics` | UX/ergonomics |
| `neuro` | Neuroscience |
| `developmental` | Developmental |

### By Quality
| Tag | Description |
|-----|-------------|
| `⭐5` | Essential reading |
| `⭐4` | Highly relevant |
| `⭐3` | Useful reference |
| `⭐2` | Background reading |
| `⭐1` | Tangentially related |

## Reading Log Status

| Status | Emoji | Description |
|--------|-------|-------------|
| `unread` | 📖 未读 | In queue |
| `reading` | 📖 在读 | Currently reading |
| `done` | ✅ 已读 | Finished |
| `skipped` | ⏭️ 跳过 | Not relevant |

## Common PDF Parsing Issues

### Multi-Column Layouts
- **Problem**: pdfplumber reads columns left-to-right, mixing text
- **Solution**: Use `layout=True` parameter or consider `pdftotext -layout`

### Scanned PDFs
- **Problem**: No selectable text
- **Solution**: OCR with `pytesseract` + `pdf2image`

### Math Formulas
- **Problem**: LaTeX formulas not extracted
- **Solution**: May appear as garbled text; consider `marker` or `pymupdf`

### Superscripts/Subscripts
- **Problem**: Numbers in citations lose formatting
- **Solution**: Post-process with regex patterns

### Header/Footer Repeats
- **Problem**: Page numbers and headers repeated in text
- **Solution**: Filter lines matching page number patterns `\d{1,3}` alone

### Encrypted PDFs
- **Problem**: Extraction fails
- **Solution**: `qpdf --decrypt input.pdf output.pdf` then retry
