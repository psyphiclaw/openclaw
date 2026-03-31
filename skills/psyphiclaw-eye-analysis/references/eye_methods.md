# Eye-Tracking Analysis Methods Reference

## Fixation Analysis

### Recommended Methods

| Metric | Description | Typical Values |
|--------|-------------|----------------|
| Mean fixation duration | Average of all fixation durations | 200–350 ms (reading), 300–500 ms (scene viewing) |
| Median fixation duration | Robust central tendency | Similar to mean but less affected by outliers |
| Fixation count | Number of fixations per trial/condition | Varies widely by task |
| Time to first fixation | Latency to first fixation on target | 200–800 ms |
| Fixation dispersion | Spatial spread within a fixation | < 1° for stable fixations |
| Saccade-to-fixation ratio | Number of saccades / fixations | ~0.9–1.1 |

### Best Practices
- **Outlier removal**: Exclude fixations < 50 ms (likely measurement noise) and > 800–1000 ms (likely blinks or tracking loss)
- **Distribution**: Fixation durations are typically right-skewed; consider log transformation for parametric tests
- **Perceptual span**: Fixations < 100 ms may not allow full processing

---

## Saccade Detection

### Methods Comparison

| Method | Pros | Cons |
|--------|------|------|
| Velocity threshold | Simple, fast | Sensitive to threshold |
| Dispersion threshold | Robust | More complex |
| Hidden Markov Model | Adaptive | Requires training |
| I-DT (Identification by Dispersion-Threshold) | Standard | Window size critical |

### Recommended Parameters
- **Velocity threshold**: 30–100 deg/s (adjust for stimulus type)
- **Minimum duration**: 20–30 ms
- **Minimum amplitude**: 1–2 degrees
- **Peak velocity**: Main sequence relationship: amplitude ≈ 30 × log(peak velocity)

### Main Sequence
The relationship between saccade amplitude and peak velocity:
- Low amplitude (< 5°): ~200–400 deg/s
- Medium (5–15°): ~300–600 deg/s
- Large (> 15°): ~400–800 deg/s

---

## AOI Analysis

### Definition Best Practices

1. **Size**: AOIs should be large enough to capture natural fixation dispersion (≥ 2° visual angle)
2. **Overlap**: Avoid overlapping AOIs or establish clear priority rules
3. **Number**: Limit to 3–8 AOIs to maintain statistical power
4. **Validation**: Test AOI definitions on pilot data before main experiment

### Common AOI Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Dwell time | Σ fixation durations in AOI | Total attention to region |
| Fixation count | N fixations in AOI | Number of visits |
| Proportion | dwell time / total time | Relative attention |
| Time to first fixation | first fixation start - stimulus onset | Attention capture |
| Revisits | Times leaving and returning | Processing difficulty |
| Transition probability | N(A→B) / N(A→any) | Scan path pattern |

### Statistical Testing
- Repeated-measures ANOVA for within-subjects AOI comparisons
- Consider using empirical logit transformation for proportion data
- Cluster-based permutation tests for temporal dynamics

---

## Pupillometry

### Methodological Considerations

| Issue | Recommendation |
|-------|----------------|
| Sampling rate | ≥ 60 Hz for task-evoked responses; ≥ 120 Hz preferred |
| Blink handling | Interpolate gaps ≤ 150 ms; exclude longer gaps |
| Smoothing | Savitzky-Golay (window ~30–50 ms, polynomial order 2–3) |
| Baseline | Use pre-stimulus window (−200 to 0 ms) for baseline correction |
| Lighting | Control or record luminance changes |
| Head position | Correct for distance changes if possible |

### Event-Locked Pupil Response
- **Latency**: Task-evoked dilation peaks at 300–500 ms post-stimulus
- **Duration**: Response typically lasts 1–3 seconds
- **Amplitude**: 0.1–0.5 mm typical for cognitive load changes
- **Baseline**: Subtract mean of pre-event window (−200 to 0 ms)

### Common Confounds
- **Light reflex**: Pupil constricts to light increases (3–5 ms latency)
- **Accommodation**: Near focus causes constriction
- **Eye movements**: Saccades cause transient pupil changes
- **Fatigue**: Pupillary unrest increases over time
