#!/usr/bin/env python3
"""Preprocess EEG data: filtering, re-referencing, artifact detection, ICA, epoching."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger("psyphiclaw-eeg-preprocess")

PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"

# Standard 10-20 mastoid channels
MASTOID_PAIRS = [("TP9", "TP10"), ("M1", "M2"), ("A1", "A2")]


def _load_raw(file_path: str, preload: bool = True) -> mne.io.Raw:
    """Load raw data with auto-detection."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=preload, verbose="INFO")
    elif ext in (".mff", ".raw"):
        return mne.io.read_raw_egi(str(path), preload=preload, verbose="INFO")
    elif ext == ".set":
        return mne.io.read_raw_eeglab(str(path), preload=preload, verbose="INFO")
    else:
        raise ValueError(f"Unknown EEG format: {ext}")


def apply_filter(raw: mne.io.Raw, l_freq: float, h_freq: float) -> mne.io.Raw:
    """Apply bandpass filter."""
    logger.info("Applying bandpass filter: %.1f - %.1f Hz", l_freq, h_freq)
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method="fir",
                                      fir_design="firwin", verbose="INFO")
    return raw_filtered


def set_reference(raw: mne.io.Raw, ref: str = "average") -> mne.io.Raw:
    """Set EEG reference."""
    if ref == "average":
        logger.info("Setting average reference")
        return raw.copy().set_eeg_reference("average", verbose="INFO")
    elif ref == "mastoid":
        # Try common mastoid channel names
        for left, right in MASTOID_PAIRS:
            if left in raw.ch_names and right in raw.ch_names:
                logger.info("Setting linked-mastoid reference: %s + %s", left, right)
                return raw.copy().set_eeg_reference([left, right], verbose="INFO")
        raise ValueError("Mastoid channels not found in data. Available: " + ", ".join(raw.ch_names[:10]) + "...")
    else:
        # Try as channel name
        if ref in raw.ch_names:
            logger.info("Setting reference to %s", ref)
            return raw.copy().set_eeg_reference(ref, verbose="INFO")
        raise ValueError(f"Unknown reference: {ref}")


def detect_artifacts(raw: mne.io.Raw, eog_threshold: float = 100e-6,
                     emg_threshold: float = 50e-6) -> dict:
    """Detect EOG/EMG artifacts using simple threshold method."""
    logger.info("Detecting artifacts (EOG > %.0fµV, EMG > %.0fµV)",
                eog_threshold * 1e6, emg_threshold * 1e6)

    # Find EOG and EMG channels
    eog_chs = [c for c in raw.ch_names if "eog" in c.lower() or c in ("HEOG", "VEOG")]
    emg_chs = [c for c in raw.ch_names if "emg" in c.lower()]

    # Auto-detect: use channels near eyes for EOG, temporal for EMG
    if not eog_chs:
        eog_chs = [c for c in raw.ch_names if c.upper().startswith(("Fp", "EOG"))]
    if not emg_chs:
        emg_chs = [c for c in raw.ch_names if any(x in c.upper() for x in ("TEMP", "MASS", "EMG"))]

    artifact_info = {"eog_channels": eog_chs, "emg_channels": emg_chs}

    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    if eog_chs:
        for ch in eog_chs:
            if ch in raw.ch_names:
                idx = raw.ch_names.index(ch)
                amp = np.abs(data[idx])
                bad_samples = amp > eog_threshold
                n_bad = int(np.sum(bad_samples))
                pct = n_bad / len(bad_samples) * 100
                artifact_info[f"eog_{ch}_bad_pct"] = round(pct, 2)
                logger.info("  %s: %.1f%% bad samples (>%dµV)", ch, pct, eog_threshold * 1e6)

    if emg_chs:
        for ch in emg_chs:
            if ch in raw.ch_names:
                idx = raw.ch_names.index(ch)
                amp = np.abs(data[idx])
                bad_samples = amp > emg_threshold
                n_bad = int(np.sum(bad_samples))
                pct = n_bad / len(bad_samples) * 100
                artifact_info[f"emg_{ch}_bad_pct"] = round(pct, 2)
                logger.info("  %s: %.1f%% bad samples (>%dµV)", ch, pct, emg_threshold * 1e6)

    return artifact_info


def apply_ica(raw: mne.io.Raw, n_components: int = 15, method: str = "fastica",
              eog_channels: Optional[list[str]] = None) -> tuple[mne.io.Raw, mne.preprocessing.ICA]:
    """Apply ICA for artifact removal."""
    logger.info("Running ICA (n_components=%d, method=%s)", n_components, method)

    # Pick only EEG channels for ICA fitting
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, exclude="bads")
    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found for ICA")
        return raw, None

    ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=42,
                                 max_iter=200, verbose="INFO")
    ica.fit(raw, picks=eeg_picks)

    # Auto-detect EOG components
    if eog_channels:
        eog_found = [c for c in eog_channels if c in raw.ch_names]
        if eog_found:
            eog_indices, _ = ica.find_bads_eog(raw, eog_found)
            if eog_indices:
                logger.info("Found EOG artifact components: %s", eog_indices)
                ica.exclude = eog_indices

    raw_cleaned = ica.apply(raw.copy(), verbose="INFO")
    logger.info("ICA applied, %d components excluded", len(ica.exclude))
    return raw_cleaned, ica


def create_epochs(raw: mne.io.Raw, tmin: float = -0.2, tmax: float = 1.0,
                  baseline: Optional[tuple] = None, reject_peak: float = 100e-6) -> mne.Epochs:
    """Create epochs from events."""
    if baseline is None:
        baseline = (tmin, 0)

    events, event_id = mne.events_from_annotations(raw, verbose="INFO")
    if len(events) == 0:
        raise ValueError("No events found in data")

    logger.info("Creating epochs: tmin=%.2f, tmax=%.2f, baseline=%s", tmin, tmax, baseline)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, reject={"eeg": reject_peak},
                        preload=True, verbose="INFO")
    logger.info("Created %d epochs (%d dropped)", len(epochs), len(epochs.drop_log))
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EEG data")
    parser.add_argument("file_path", type=str, help="Path to EEG file")
    parser.add_argument("--filter", nargs=2, type=float, metavar=("LOW", "HIGH"), default=[0.1, 40.0],
                        help="Bandpass filter range in Hz (default: 0.1 40)")
    parser.add_argument("--no-filter", action="store_true", help="Skip filtering")
    parser.add_argument("--reref", type=str, choices=["average", "mastoid"], default=None,
                        help="Re-referencing method")
    parser.add_argument("--ica", action="store_true", help="Apply ICA artifact removal")
    parser.add_argument("--ica-components", type=int, default=15, help="Number of ICA components")
    parser.add_argument("--epochs", action="store_true", help="Create epochs from events")
    parser.add_argument("--tmin", type=float, default=-0.2, help="Epoch start (s)")
    parser.add_argument("--tmax", type=float, default=1.0, help="Epoch end (s)")
    parser.add_argument("--detect-artifacts", action="store_true", help="Detect artifacts (no removal)")
    parser.add_argument("--eog-threshold", type=float, default=100e-6, help="EOG threshold (V)")
    parser.add_argument("--emg-threshold", type=float, default=50e-6, help="EMG threshold (V)")
    parser.add_argument("--output", type=str, default=None, help="Save preprocessed data (.fif)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    raw = _load_raw(args.file_path, preload=True)

    if not args.no_filter:
        raw = apply_filter(raw, args.filter[0], args.filter[1])

    if args.reref:
        raw = set_reference(raw, args.reref)

    if args.detect_artifacts:
        art_info = detect_artifacts(raw, args.eog_threshold, args.emg_threshold)
        for k, v in art_info.items():
            if k.endswith("_bad_pct"):
                print(f"  {k}: {v}%")

    if args.ica:
        eog_chs = [c for c in raw.ch_names if "eog" in c.lower() or c in ("HEOG", "VEOG")]
        raw, ica = apply_ica(raw, args.ica_components, eog_channels=eog_chs if eog_chs else None)

    if args.epochs:
        epochs = create_epochs(raw, tmin=args.tmin, tmax=args.tmax)
        print(f"✅ Created {len(epochs)} epochs")

    if args.output:
        if args.epochs:
            epochs.save(args.output, overwrite=True, verbose="INFO")
        else:
            raw.save(args.output, overwrite=True, verbose="INFO")
        logger.info("Saved to %s", args.output)

    print("✅ Preprocessing complete")


if __name__ == "__main__":
    main()
