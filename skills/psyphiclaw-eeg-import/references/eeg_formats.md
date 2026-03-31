# EEG Data Formats Reference

## Brain Vision Format

### Files
Brain Vision export produces **three files** that must be kept together:

| File | Extension | Description |
|------|-----------|-------------|
| Header | `.vhdr` | Text file with metadata: channel names, sampling rate, format info |
| Data | `.eeg` | Binary data file (actual voltage values) |
| Marker | `.vmrk` | Event markers / triggers with timestamps |

### vhdr Structure
```
[Common Info]
DataFile=filename.eeg
DataFormat=BINARY
NumberOfChannels=64
SamplingFrequency=500

[Binary Info]
BinaryFormat=IEEE_FLOAT_32
DataOrientation=MULTIPLEXED

[Channel Info]
; Each line: Ch1=Name,,,,Resolution
1=Fp1,,,,0.1
2=Fp2,,,,0.1
...
```

### vmrk Structure
```
[Common Info]
DataFile=filename.eeg

[Marker Infos]
; Type, Description, Position (samples), Length, Channel
1,Stimulus,S  1,1,0
2,Stimulus,S  2,1,0
3,Response,R  1,1,0
```

### Key Notes
- **Import reference**: Use the `.vhdr` file — MNE reads `.eeg` and `.vmrk` automatically
- **Data orientation**: MULTIPLEXED (channels × samples) is standard
- **Binary formats**: INT_16, INT_32, IEEE_FLOAT_32, IEEE_FLOAT_64

---

## EGI Format

### .raw (Simple Binary)
- Header + binary data
- 16-bit integer format
- Channel order follows EGI net layout (not 10-20)

### .mff (MFF Archive)
- Modern EGI format
- ZIP-based archive containing XML metadata and binary segments
- Stores events, channel info, and device calibration
- **Recommended** over .raw when available

### Key Notes
- EGI uses its own channel naming (E1, E2, ... or A1, A2, ...)
- May need montage mapping to 10-20 system
- MNE provides `mne.channels.make_standard_montage('GSN-HydroCel-129')` for EGI nets

---

## EEGLAB Format

### .set / .fdt
- MATLAB-based format
- `.set` — MATLAB .mat file with header and metadata
- `.fdt` — Binary data file (separate, for large datasets)
- Smaller datasets (<2GB) may have all data in the `.set` file

### Key Notes
- Channel locations stored as EEGLAB EEG.data structure
- Events stored in `EEG.event` struct
- MNE reads `.set` and auto-loads `.fdt` if present
- EEGLAB uses µV by default

---

## Preprocessing Recommendations

### Filter Settings
| Application | Bandpass | Notes |
|-------------|----------|-------|
| Standard ERP | 0.1–30 Hz | Good for P3, N400 |
| Gamma analysis | 0.1–80 Hz | Preserve high frequencies |
| ICA preparation | 1.0–40 Hz | Remove slow drifts and line noise |
| Time-frequency | 0.1–100+ Hz | Depends on frequency band of interest |

### Notch Filter
- Apply **before** bandpass: 50 Hz or 60 Hz (region-dependent)
- MNE: `raw.notch_filter([50, 100])` for harmonics

### Re-referencing
| Reference | When to use |
|-----------|-------------|
| Average | Default for most analyses |
| Linked mastoids (TP9+TP10) | Common in clinical EEG |
| Cz | Some language/ERP paradigms |
| REST (Reference Electrode Standardization Technique) | When accurate amplitude is critical |

### ICA Settings
- **n_components**: 15–20 (or sqrt of n_channels)
- **Method**: `fastica` (fast) or `picard` (more stable)
- **Components to exclude**: Typically 1–4 EOG-related
- Run ICA **after** bandpass filtering but **before** epoching for best results

### Epoching
| Paradigm | Baseline | Epoch window |
|----------|----------|--------------|
| Visual oddball | -200 to 0 ms | -200 to 1000 ms |
| Language (N400) | -200 to 0 ms | -200 to 1000 ms |
| Face perception (N170) | -100 to 0 ms | -100 to 800 ms |
| Error-related (ERN) | -200 to 0 ms | -200 to 600 ms |

### Artifact Rejection Thresholds
| Artifact | Typical threshold |
|----------|-------------------|
| Peak-to-peak amplitude | ±100 µV |
| EOG blink | ±100 µV |
| EMG | ±50 µV |
| Step (drift) | ±50 µV |

### Common Sampling Rates
| System | Typical rate |
|--------|-------------|
| EEG research | 500–1000 Hz |
| Clinical EEG | 250–500 Hz |
| EGI HydroCel | 500 Hz |
| Brain Products | 500–1000 Hz |
| Emotiv / consumer | 128–256 Hz |
