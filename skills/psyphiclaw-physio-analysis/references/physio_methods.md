# Psychophysiological Signal Analysis — Methods Reference

## Overview

This document provides recommended analysis methods, sampling requirements, common artifacts, and default parameter values for the four physiological signal types supported by the `psyphiclaw-physio-analysis` toolkit: ECG/HRV, EDA, EMG, and respiration.

---

## 1. ECG / Heart Rate Variability (HRV)

### Recommended Sampling Rate
| Minimum | Recommended | Notes |
|---------|-------------|-------|
| ≥ 200 Hz | 500–1000 Hz | Higher rates improve R-peak timing precision. Clinical-grade devices often use 1000 Hz. |

### Analysis Pipeline
1. **Band-pass filtering** (5–15 Hz, 4th order Butterworth) to isolate QRS morphology
2. **Differentiation → Squaring → Moving average** (Pan-Tompkins-inspired R-peak detection)
3. **RR interval extraction** with outlier removal (±3 SD)
4. **Time-domain metrics**: SDNN, RMSSD, pNN50, mean RR, mean HR
5. **Frequency-domain** (Welch PSD on 4 Hz interpolated tachogram):
   - VLF: 0.0033–0.04 Hz
   - LF: 0.04–0.15 Hz
   - HF: 0.15–0.4 Hz
   - LF/HF ratio as sympathovagal balance index
6. **Non-linear**: Poincaré SD1/SD2, Approximate Entropy (ApEn, m=2, r=0.2·SD)

### Interpretation Notes
- **RMSSD** reflects parasympathetic (vagal) modulation — higher = more relaxed
- **LF/HF ratio** > 2 suggests sympathetic dominance (stress/arousal)
- **SDNN** reflects overall autonomic variability; sensitive to recording length
- **Minimum recording**: 5 min (short-term), 24 h preferred for clinical use

### Common Artifacts
| Artifact | Cause | Mitigation |
|----------|-------|------------|
| Baseline wander | Respiration, movement | High-pass filter (≥ 0.5 Hz) before analysis |
| Power-line noise | 50/60 Hz mains | Notch filter or adaptive subtraction |
| Motion artifacts | Physical movement | Amplitude-based rejection, visual inspection |
| Ectopic beats | Premature ventricular contractions | ±3 SD outlier removal on RR intervals |

---

## 2. Electrodermal Activity (EDA)

### Recommended Sampling Rate
| Minimum | Recommended | Notes |
|---------|-------------|-------|
| ≥ 100 Hz | 200–1000 Hz | Higher rates improve SCR onset/peak timing. |

### Analysis Pipeline
1. **Signal decomposition** using continuous deconvolution or filtering:
   - **Tonic (SCL)**: Low-pass filter at 0.05 Hz (4th order Butterworth)
   - **Phasic (SCR)**: High-pass filter at 0.05 Hz, then non-negative rectification
2. **SCR peak detection**:
   - Onset threshold: > 0.02 μS
   - Peak threshold: > 0.05 μS
   - Minimum rise time: 0.5 s
   - Maximum duration: 5 s
3. **Event-locked analysis**: Epoch ±1 s / +4 s around event markers; extract peak amplitude, latency, AUC

### Interpretation Notes
- **SCR frequency** increases with arousal (emotional, cognitive, physical)
- **SCL** reflects general arousal level; slow changes over minutes
- **SCR amplitude** is phasic event-related; useful for stimulus-response studies

### Common Artifacts
| Artifact | Cause | Mitigation |
|----------|-------|------------|
| Sudden DC offset | Electrode detachment | Amplitude/threshold rejection |
| Motion artifacts | Movement, cable tug | Low-pass filter; visual inspection |
| Temperature drift | Environmental changes | Continuous rather than discrete recording |
| Skin condition | Dry skin, calluses | Proper site preparation, conductive gel |

---

## 3. Electromyography (EMG)

### Recommended Sampling Rate
| Minimum | Recommended | Notes |
|---------|-------------|-------|
| ≥ 1000 Hz | 2000–4000 Hz | Surface EMG contains energy up to ~500 Hz; oversampling recommended for anti-aliasing. |

### Analysis Pipeline
1. **Band-pass filtering** (20–450 Hz, 4th order Butterworth) — removes movement artifact and high-frequency noise
2. **Full-wave rectification** — converts bipolar AC signal to unipolar
3. **Envelope extraction**:
   - Low-pass smoothing at ~10 Hz for general envelope
   - Running RMS with ~50 ms window for onset detection
4. **Onset detection**: Z-score thresholding (baseline from first 1 s, z > 3)
5. **Fatigue analysis**: Median frequency (MDF) via sliding-window Welch PSD (0.5 s windows, 50% overlap)
   - MDF decreases with sustained muscle contraction (muscle fatigue indicator)

### Interpretation Notes
- **RMS amplitude** correlates with muscle force (non-linear, varies by individual)
- **Median frequency shift** (decrease of ≥5 Hz over time) indicates neuromuscular fatigue
- **Onset latency** is critical for reaction-time and motor-control studies

### Common Artifacts
| Artifact | Cause | Mitigation |
|----------|-------|------------|
| ECG contamination | Heart signal on trunk muscles | Band-pass (20–450 Hz) strongly attenuates ECG |
| Power-line noise | 50/60 Hz mains | Notch filter at 50/60 Hz |
| Motion artifact | Cable movement, skin stretch | Proper electrode placement, adhesive anchors |
| Crosstalk | Adjacent muscle activity | Differential electrodes, narrow inter-electrode distance |

---

## 4. Respiration

### Recommended Sampling Rate
| Minimum | Recommended | Notes |
|---------|-------------|-------|
| ≥ 100 Hz | 200–500 Hz | Respiratory belt or thermistor; higher rates useful for thoraco-abdominal coordination studies. |

### Analysis Pipeline
1. **Low-pass filtering** at 1 Hz to remove high-frequency noise
2. **Breath detection**: Peak/trough detection on filtered signal
3. **Key metrics**: Breathing rate (breaths/min), I:E ratio, inspiration/expiration duration, tidal volume (calibrated)
4. **Respiratory sinus arrhythmia (RSA)**: Cross-correlation with ECG RR intervals to quantify vagal tone

### Common Artifacts
| Artifact | Cause | Mitigation |
|----------|-------|------------|
| Motion artifact | Postural changes, speaking | Artifact detection thresholds |
| Signal clipping | Over-tight belt or saturated sensor | Gain adjustment |
| Swallowing | Voluntary swallowing events | Epoch exclusion |

---

## 5. Default Parameters Summary

| Parameter | ECG/HRV | EDA | EMG | Respiration |
|-----------|---------|-----|-----|-------------|
| **Sampling rate (min)** | 200 Hz | 100 Hz | 1000 Hz | 100 Hz |
| **Filter band** | 5–15 Hz (R-peak) | LP 0.05 Hz / HP 0.05 Hz | 20–450 Hz | LP 1 Hz |
| **Filter order** | 4 | 4 | 4 | 4 |
| **Filter type** | Butterworth BP | Butterworth LP/HP | Butterworth BP | Butterworth LP |
| **Peak threshold** | Adaptive (μ + 0.5σ) | 0.05 μS | z > 3.0 | Adaptive |
| **Window size** | — | — | RMS: 50 ms; MDF: 500 ms | — |
| **Minimum events** | 10 R-peaks | 1 SCR peak | — | — |
| **Outlier removal** | RR ±3 SD | Amplitude-based | Baseline z-score | — |
| **PSD method** | Welch (nperseg=256) | — | Welch (nperseg=256) | — |
| **Frequency bands** | VLF/LF/HF | — | MDF: 20–450 Hz | — |

---

## 6. References

1. Task Force of the European Society of Cardiology (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043–1065.
2. Boucsein, W. et al. (2012). Publication recommendations for electrodermal measurements. *Psychophysiology*, 49(8), 1017–1034.
3. De Luca, C. J. (1997). The use of surface electromyography in biomechanics. *Journal of Applied Biomechanics*, 13(2), 135–163.
4. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, (3), 230–236.
5. Benedek, M., & Kaernbach, C. (2010). Decomposition of skin conductance data by means of nonnegative deconvolution. *Psychophysiology*, 47(4), 647–658.
