# Eye-Tracking Data Formats Reference

## Tobii TSV Format (Pro Lab Export)

### Structure
- **Header rows**: Variable number of preamble rows before column headers
- **Column separator**: Tab (`\t`) or comma
- **Encoding**: UTF-8 (occasionally Latin-1 for older exports)

### Common Columns
| Column | Description |
|--------|-------------|
| RecordingTimestamp | Timestamp in ms (from recording start) |
| GazePointX / GazePointY | Screen gaze coordinates (pixels) |
| GazePointLeftX / GazePointRightX | Per-eye gaze X |
| PupilLeft / PupilRight | Pupil diameter (mm) |
| FixationIndex | Fixation event ID |
| Event | Event type (FixationStart, FixationEnd, SaccadeStart, SaccadeEnd) |
| ValidityLeft / ValidityRight | Data validity (0–4, 0 = valid) |

### Event Types
- `FixationStart` / `FixationEnd` — Gaze stabilized on a point
- `SaccadeStart` / `SaccadeEnd` — Rapid eye movement
- No explicit blink events (detected via validity flags)

---

## EyeLink ASC Format

### Structure
- **Header**: Lines starting with `**` containing metadata
- **Samples**: Numeric lines (timestamp, x, y, pupil, ...)
- **Events**: Lines starting with `EFIX`, `ESACC`, `EBLINK`

### Sample Format
```
timestamp  x_left  y_left  pupil_left  x_right  y_right  pupil_right  ...
1234       512.3   384.1   4.5         510.8    383.9    4.3          ...
```

### Event Formats
```
EFIX L  start end duration avg_x avg_y avg_pupil
ESACC L start end duration start_x start_y end_x end_y amplitude peak_vel
EBLINK L start end duration
```

### Eye Codes
- `L` = Left eye
- `R` = Right eye
- `B` = Both eyes (binocular)

### Timestamps
- In ms from recording start
- Can be 0-padded or unpadded

---

## Pupil Labs Export Format

### Directory Structure
```
exports/
├── gaze_positions.csv
├── pupil_positions.csv
├── fixations.csv
├── blinks.csv
└── surfaces/
    └── surface_name_gaze_on_surface.csv
```

### gaze_positions.csv
| Column | Description |
|--------|-------------|
| world_timestamp | Timestamp (seconds) |
| norm_pos_x / norm_pos_y | Normalized gaze position (0–1) |
| confidence | Detection confidence (0–1) |
| base_data | JSON reference to eye data |

### pupil_positions.csv
| Column | Description |
|--------|-------------|
| world_timestamp | Timestamp (seconds) |
| diameter | 2D pupil diameter (pixels) |
| diameter_3d | 3D pupil diameter (mm) |
| confidence | Detection confidence (0–1) |

### Surface Tracking
- `surface_gaze_on_surface.csv`: Gaze positions mapped to a defined surface
- `on_surf`: Boolean indicating if gaze falls on the surface
- `norm_pos_x / norm_pos_y`: Position on surface (0–1)

---

## AOI (Area of Interest) Definition

### Methods

1. **Static AOI** (image/stimulus-based):
   - Defined relative to stimulus image coordinates
   - Rectangular: (x, y, width, height)
   - Circular: (cx, cy, radius)
   - Polygon: list of (x, y) vertices

2. **Dynamic AOI** (screen/viewport-based):
   - Defined in screen coordinates
   - Useful for web/mobile stimuli

3. **Semantic AOI**:
   - Defined by content (e.g., "face area", "text region")
   - Requires image processing or manual tagging

### Implementation Notes
- AOI definitions should be stored in a separate JSON/YAML config
- Coordinates must match the gaze data coordinate system (pixels vs normalized)
- Overlapping AOIs require priority rules or hierarchical handling
