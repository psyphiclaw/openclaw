# Physiological Signal Formats Reference

## Biopac ACQ Format

### Overview
Biopac AcqKnowledge uses a proprietary binary format (.acq) for storing
multi-channel physiological data. Newer versions also support WFDB-compatible
exports (.hea + .dat).

### File Structure
- **Header**: ~900 bytes of metadata (sampling rate, channel count, channel labels)
- **Data**: Binary samples (typically int16 or float32), interleaved by channel
- **Events**: Event markers embedded in data stream

### Channel Numbering Convention
| Channel # | Typical Signal |
|-----------|---------------|
| 100 (CH1) | ECG |
| 101 (CH2) | EDA/GSR |
| 102 (CH3) | EMG |
| 103 (CH4) | Respiration |
| 104 (CH5) | Skin Temperature |
| 105 (CH6) | PPG |

### Sampling Rate Recommendations
| Signal | Min Rate | Recommended |
|--------|----------|-------------|
| ECG | 250 Hz | 1000 Hz |
| EDA | 50 Hz | 200 Hz |
| EMG | 1000 Hz | 2000 Hz |
| Respiration | 50 Hz | 100 Hz |
| Skin Temp | 10 Hz | 50 Hz |

---

## ADInstruments LabChart Export

### CSV Export Format
- **Header rows**: May contain metadata (date, subject, channel settings)
- **Column separator**: Tab, comma, or semicolon (user-configurable)
- **First column**: Time (seconds, configurable precision)
- **Subsequent columns**: One per channel

### Common Export Settings
- **File → Export → Text/CSV**
- Check "Include channel titles" for header names
- Time format: seconds with configurable decimal places

---

## General CSV Format

### Recommended Structure
```
Time(s)    ECG(mV)    EDA(uS)    EMG(mV)    RESP(V)    TEMP(C)
0.000      0.052      3.21       0.011      0.500      32.1
0.001      0.048      3.22       0.012      0.501      32.1
0.002      0.045      3.21       0.010      0.502      32.1
...
```

### Column Naming Conventions
- Use descriptive names with signal type: `ECG`, `EDA`, `EMG`, `RESP`, `TEMP`
- Include units in parentheses: `ECG (mV)`, `EDA (μS)`
- Time column: `Time (s)`, `Timestamp (ms)`, or `Time`

### Multi-rate Data
If channels have different sampling rates:
- Use separate files per sampling rate
- Or use the highest rate and interpolate lower-rate channels
