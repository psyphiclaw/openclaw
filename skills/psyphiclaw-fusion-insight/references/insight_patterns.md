# Multi-Modal Correlation Patterns Reference

## Common Patterns by Emotional State

### Fear / Anxiety Response
| Modality | Direction | Typical Magnitude |
|----------|-----------|-------------------|
| EEG (LPP) | ↑ (400-700ms) | +2-8 μV |
| EDA (Skin Conductance) | ↑ | +1-5 μS |
| Pupil Diameter | ↑ | +0.5-2 mm |
| Heart Rate | ↑ | +5-15 bpm |
| Heart Rate Variability | ↓ | -10-30% |
| fNIRS (dlPFC) | ↑ HbO | +0.3-1.0 μM |
| Facial EMG (Corrugator) | ↑ | +10-50 μV |
| Valence (Self-report) | ↓ (negative) | 1-3/9 |

### Disgust
| Modality | Direction | Notes |
|----------|-----------|-------|
| EEG (LPP) | ↑ (later, 500-800ms) | Sustained |
| EDA | ↑ | Moderate |
| Pupil | ↑ | Brief |
| Facial EMG (Levator Labii) | ↑↑ | Characteristic marker |

### Surprise
| Modality | Direction | Notes |
|----------|-----------|-------|
| EEG (P300) | ↑ (300-500ms) | Large amplitude |
| Pupil | ↑↑ | Rapid dilation |
| Heart Rate | Brief ↓ then ↑ | Startle response |
| EEG (Frontal Theta) | ↑ | Novelty detection |

### Positive Arousal (Excitement)
| Modality | Direction | Notes |
|----------|-----------|-------|
| EEG (Frontal Alpha) | ↓ (desynchronization) | Approach motivation |
| EDA | ↑ | Moderate |
| zygomaticus EMG | ↑ | Smile-related |
| Heart Rate | ↑ | Variable |
| fNIRS (NAcc/mPFC) | ↑ HbO | Reward processing |

---

## Cross-Modal Coupling Patterns

### Strong Coupling (Expected)
- **EEG LPP ↔ EDA**: r = 0.3-0.6 during emotional stimuli
- **Pupil ↔ EDA**: r = 0.2-0.5 (sympathetic arousal)
- **HR ↔ EDA**: r = 0.2-0.4 (general arousal)

### Modality-Specific Patterns
- **fNIRS ↔ EEG**: Weak direct coupling; both reflect cortical activity
  with different temporal resolution (seconds vs milliseconds)
- **Facial EMG ↔ Self-report**: r = 0.2-0.4 (moderate validity)

---

## Experimental Paradigm Templates

### Emotion Elicitation (IAPS-based)
```
Fixation (500ms) → Image (3000ms) → Rating (4000ms) → ISI (1000ms)
Expected: LPP peak at 400-600ms, EDA onset 1000-2000ms
```

### Go/No-Go (Inhibitory Control)
```
Fixation (500ms) → Cue (200ms) → Target (1500ms) → Feedback (1000ms)
Expected: N2/P3 at 200-350ms, commission errors → EDA peak
```

### Oddball (P300)
```
Standard (80%) / Deviant (20%) at 1000ms SOA
Expected: P3b at 300-400ms for deviants, Pupil dilation 200-400ms
```

### Resting State (rs-fNIRS + EEG)
```
5 min eyes-open → 5 min eyes-closed
Expected: Alpha power ↑ in eyes-closed, HbO ↑ in visual cortex
```

---

## Insight Report Template

```markdown
# Multimodal Insight Report

## Session Overview
- Duration: X min
- Modalities: EEG, EDA, Pupil, fNIRS
- N trials: X

## Key Findings
1. [Finding with data citation]
2. [Cross-modal correlation]
3. [Temporal pattern]

## Anomaly Events
| Time | Modalities | Severity | Possible Cause |
|------|-----------|----------|---------------|

## Statistical Summary
| Finding | Effect | p-value | Cohen's d |
|---------|--------|---------|-----------|

## Confidence Assessment
- Overall: X.XX (PASS/REVIEW/REJECT)
- Data Quality: ✓
- Effect Sizes: ✓
- Causal Claims: ⚠️

## Recommended Next Steps
1. ...
2. ...

---
*PsyPhiClaw Fusion Insight · Auto-generated*
```
