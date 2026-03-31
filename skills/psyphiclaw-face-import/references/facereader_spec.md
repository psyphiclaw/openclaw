# FaceReader CSV Format Specification

## Overview

FaceReader (Noldus Information Technology) outputs facial expression analysis results as CSV files. This document describes the column format, encoding issues, and common pitfalls.

## CSV Structure

### Header
- First row contains column names
- Separator: typically comma `,`, sometimes semicolon `;` (locale-dependent)
- Decimal separator: period `.` (English) or comma `,` (some European locales)

### Encoding
- **UTF-8**: Default in recent FaceReader versions
- **GBK/GB2312**: Common in Chinese Windows systems
- **UTF-8 BOM (UTF-8-sig)**: Possible with Excel export

The `import_facereader.py` script auto-detects encoding.

## Column Groups

### 1. Timestamp & Frame
| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | float | Time in milliseconds from start of recording |
| `Frame` | int | Frame number (sequential) |

### 2. Face Detection
| Column | Type | Description |
|--------|------|-------------|
| `FaceReader_FacePresence` | float | Face detected: 0 = no, 1 = yes |
| `FaceReader_Quality` | float | Face quality score (0-1) |

### 3. Action Units (FACS)
Columns named `ActionUnit_N` where N is the FACS code:
- **AU1**: Inner Brow Raiser
- **AU2**: Outer Brow Raiser
- **AU4**: Brow Lowerer
- **AU5**: Upper Lid Raiser
- **AU6**: Cheek Raiser
- **AU7**: Lid Tightener
- **AU9**: Nose Wrinkler
- **AU10**: Upper Lip Raiser
- **AU12**: Lip Corner Puller
- **AU14**: Dimpler
- **AU15**: Lip Corner Depressor
- **AU17**: Chin Raiser
- **AU20**: Lip Stretcher
- **AU23**: Lip Tightener
- **AU24**: Lip Pressor
- **AU25**: Lips Part
- **AU26**: Jaw Drop
- **AU27**: Mouth Stretch
- **AU43**: Eyes Closed

Values: 0.0 (inactive) to 1.0 (maximum activation)

### 4. VAD Emotional Dimensions
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Valence` | float | 0.0 - 1.0 | Negative (0) to Positive (1) |
| `Arousal` | float | 0.0 - 1.0 | Calm (0) to Excited (1) |
| `Dominance` | float | 0.0 - 1.0 | Submissive (0) to Dominant (1) |

Midpoint (0.5) = neutral.

### 5. Basic Emotions
Probabilities summing to ~1.0:
| Column | Description |
|--------|-------------|
| `Neutral` | No clear emotion |
| `Happy` | Joy, amusement |
| `Sad` | Sadness, sorrow |
| `Angry` | Anger, frustration |
| `Surprised` | Surprise |
| `Scared` | Fear |
| `Disgusted` | Disgust, revulsion |
| `Contempt` | Contempt, disdain |

### 6. Head Orientation (radians)
| Column | Description |
|--------|-------------|
| `HeadOrientation_Roll` | Head tilt (left/right rotation around forward axis) |
| `HeadOrientation_Pitch` | Head nod (up/down rotation) |
| `HeadOrientation_Yaw` | Head turn (left/right rotation around vertical axis) |

### 7. Gaze Direction
| Column | Description |
|--------|-------------|
| `GazeDirection_X` | Horizontal gaze (left/negative, right/positive) |
| `GazeDirection_Y` | Vertical gaze (down/negative, up/positive) |
| `GazeDirection_Z` | Depth gaze direction |

## Common Issues

### Missing Values
- FaceReader outputs NaN when no face is detected
- AU values, VAD, and emotions are all NaN during no-face frames
- **Recommendation**: Filter by `FaceReader_FacePresence == 1` before analysis

### Sampling Rate
- Default: 30 Hz (common for webcam recordings)
- Can vary: 20 Hz, 60 Hz, or camera-native rate
- Always compute from data: `(frames - 1) / duration_seconds`

### Large Files
- 1-hour recording at 30 Hz = ~108,000 rows
- Use `--time-range` to filter before visualization

### Locale Issues
- European locales may use semicolons and comma decimals
- The import script tries both separators automatically

### Multiple Faces
- When multiple faces are tracked, FaceReader may use numbered suffixes
- e.g., `FaceReader_FacePresence_2`, `Valence_2`
