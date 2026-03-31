# fNIRS Data Formats Reference

## SNIRF Format (Shared NIRS Data Format)

SNIRF is the standardized format for NIRS data, defined by the
Society for functional Near-Infrared Spectroscopy.

### File Structure (HDF5-based)

```
/
├── formatVersion        # "1.0" or "1.1"
├── nirs/
│   ├── source1/
│   │   ├── source wavelengths     # [760, 850]
│   │   ├── measurementList1/
│   │   │   ├── sourceIndex        # 0
│   │   │   ├── detectorIndex      # 0
│   │   │   ├── wavelengthIndex    # 0
│   │   │   └── dataType           # 1 = OD
│   │   ├── dataTimes              # (N,) timestamps
│   │   └── data1                  # (N,) signal
│   ├── ...
├── probe/
│   ├── sourcePositions             # (nSources, 3)
│   ├── detectorPositions           # (nDetectors, 3)
│   └── wavelengths                 # [760, 850]
├── stim/
│   ├── data1/
│   │   ├── name                    # "ConditionA"
│   │   ├── data                    # onset times
│   │   └── data                    # durations
│   └── ...
└── metaDataTags/
    ├── SubjectID
    ├── MeasurementDate
    └── LengthUnit                   # "mm"
```

### Key Parameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| Wavelengths | 760 nm, 850 nm | Two wavelengths for chromophore separation |
| Sampling Rate | 10–100 Hz | Varies by device |
| Source-Detector Distance | 20–40 mm | Standard for adults |

---

## NIRSport (NIRx) Data

### Directory Structure

```
session/
├── 2024-01-15_001.snirf    # SNIRF format (newer)
├── .hdr                    # Header file (older format)
├── .dat                    # Data file
├── .evt                    # Event file
└── .wl                     # Wavelength info
```

### Header Format (.hdr)

```
#NIRSport
#NIRS data file
#Channels: S1-D1 S1-D2 S2-D1 ...
#Wavelengths: 760 850
#SamplingRate: 7.81
#SourceDetectorDistance: 30
```

---

## Beer-Lambert Law Conversion

### Modified Beer-Lambert Law (MBLL)

```
ΔOD(λ) = -log(I(t) / I₀) = ε(λ) · d · BPF(λ) · ΔC + G(λ)
```

Where:
- `ΔOD` = Change in optical density
- `ε(λ)` = Extinction coefficient at wavelength λ
- `d` = Source-detector distance
- `BPF(λ)` = Differential pathlength factor (≈ 6.0 for adults)
- `ΔC` = Concentration change
- `G(λ)` = Geometry-dependent scattering term

### Extinction Coefficients (μM⁻¹·mm⁻¹)

| Chromophore | 760 nm | 850 nm |
|------------|--------|--------|
| HbO | 0.1234 | 0.2114 |
| HbR | 0.9083 | 0.3042 |

### HbO / HbR Calculation

```
[HbO] = (ε_HbR(λ2)·ΔOD(λ1) - ε_HbR(λ1)·ΔOD(λ2)) / (d·BPF·det)
[HbR] = (ε_HbO(λ1)·ΔOD(λ2) - ε_HbO(λ2)·ΔOD(λ1)) / (d·BPF·det)

where det = ε_HbO(λ1)·ε_HbR(λ2) - ε_HbO(λ2)·ε_HbR(λ1)
```

---

## Typical Preprocessing Pipeline

```
1. Raw OD Import
   ↓
2. Quality Check (SCI < 0.75 → flag)
   ↓
3. Beer-Lambert → HbO, HbR concentration
   ↓
4. Bandpass Filter (0.01–0.5 Hz)
   ↓
5. Artifact Removal (Motion → TDDR / CBSI)
   ↓
6. Baseline Correction (per epoch)
   ↓
7. Trial Averaging or GLM
```

### Partial Pathlength Factor (DPF) by Age

| Age Group | DPF |
|-----------|-----|
| Adult | 6.0 |
| Child (7-17) | 5.0 |
| Infant (0-1) | 4.0 |

### Bandpass Filter Rationale

| Frequency | Removed | Reason |
|-----------|---------|--------|
| < 0.01 Hz | Slow drift | Instrument + physiological |
| 0.01–0.5 Hz | **Kept** | Hemodynamic response |
| 0.5–1.0 Hz | Respiratory | ~0.3 Hz alias |
| > 1.0 Hz | Cardiac | ~1 Hz heart rate |
