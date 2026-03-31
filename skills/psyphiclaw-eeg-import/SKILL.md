# 🧠 PsyPhiClaw EEG Import

Import, preprocess, and analyze EEG data using MNE-Python. Supports Brain Vision, EGI, and EEGLAB formats.

## Overview

This skill handles the complete EEG analysis pipeline:
1. **Import** — Brain Vision (.vhdr), EGI (.raw/.mff), EEGLAB (.set/.fdt)
2. **Preprocess** — Filtering, re-referencing, artifact detection, ICA, epoching
3. **Analyze** — ERP component extraction, time-window averaging, condition comparison
4. **Visualize** — Raw signal, PSD, ERP waveforms, topomaps (PNG + HTML)

## File Structure

```
psyphiclaw-eeg-import/
├── SKILL.md
├── scripts/
│   ├── import_eeg.py        # File loading & metadata summary
│   ├── preprocess_eeg.py    # Filtering, re-ref, artifact removal, epoching
│   ├── analyze_erp.py       # ERP extraction & comparison
│   └── visualize_eeg.py     # All visualization functions
└── references/
    └── eeg_formats.md       # EEG format documentation & preprocessing recommendations
```

## Usage

### Import & Summary
```bash
python3 scripts/import_eeg.py data.vhdr --summary
python3 scripts/import_eeg.py data.mff --summary
python3 scripts/import_eeg.py data.set --summary
```

### Preprocess
```bash
python3 scripts/preprocess_eeg.py data.vhdr --filter 0.1 40 --reref average --epochs --tmin -0.2 --tmax 1.0
```

### ERP Analysis
```bash
python3 scripts/analyze_erp.py data.vhdr --conditions target standard --erp-components
```

### Visualize
```bash
python3 scripts/visualize_eeg.py data.vhdr --output-dir ./eeg_plots
```

## Dependencies

- Python 3.10+
- mne, numpy, matplotlib, plotly

---
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      python: ["mne", "numpy", "matplotlib", "plotly"]
    install:
      - id: pip-mne
        kind: pip
        package: mne
        label: Install MNE-Python for EEG analysis
      - id: pip-numpy
        kind: pip
        package: numpy
        label: Install numpy
      - id: pip-matplotlib
        kind: pip
        package: matplotlib
        label: Install matplotlib
      - id: pip-plotly
        kind: pip
        package: plotly
        label: Install Plotly for interactive visualization
