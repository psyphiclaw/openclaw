# 🎭 PsyPhiClaw Face Import

Import and analyze FaceReader CSV output data — Action Units, VAD dimensions, basic emotions, head orientation, and gaze direction.

## Overview

This skill handles the complete pipeline for FaceReader facial expression data:
1. **Import** — Parse FaceReader CSV with automatic encoding detection
2. **Summarize** — Sampling rate, duration, valid frames, missing data
3. **Visualize** — Interactive Plotly charts for VAD, emotions, AUs, face presence

## File Structure

```
psyphiclaw-face-import/
├── SKILL.md
├── scripts/
│   ├── import_facereader.py   # CSV import, parsing, summary
│   └── visualize_face.py      # VAD, emotion, AU, face-presence plots
└── references/
    └── facereader_spec.md     # FaceReader CSV format documentation
```

## Usage

### Import & Summarize
```bash
python3 scripts/import_facereader.py /path/to/output.csv --summary
python3 scripts/import_facereader.py /path/to/output.csv --time-range 5000 30000 --summary
```

### Visualize
```bash
python3 scripts/visualize_face.py /path/to/output.csv --output-dir ./plots
python3 scripts/visualize_face.py /path/to/output.csv --vad --emotions --au-heatmap --face-presence
```

## Dependencies

- Python 3.10+
- pandas, numpy, matplotlib, seaborn, plotly

---
metadata:
  openclaw:
    emoji: "🎭"
    requires:
      python: ["pandas", "numpy", "matplotlib", "seaborn", "plotly"]
    install:
      - id: pip-pandas
        kind: pip
        package: pandas
        label: Install pandas for data handling
      - id: pip-numpy
        kind: pip
        package: numpy
        label: Install numpy
      - id: pip-matplotlib
        kind: pip
        package: matplotlib
        label: Install matplotlib
      - id: pip-seaborn
        kind: pip
        package: seaborn
        label: Install seaborn
      - id: pip-plotly
        kind: pip
        package: plotly
        label: Install Plotly for interactive visualization
