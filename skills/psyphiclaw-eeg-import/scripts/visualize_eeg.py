#!/usr/bin/env python3
"""Visualize EEG data: raw signal preview, PSD, ERP waveforms, topomaps (PNG + HTML)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt

logger = logging.getLogger("psyphiclaw-eeg-viz")

PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"


def _load_raw(file_path: str) -> mne.io.Raw:
    """Load raw EEG data."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=True, verbose="INFO")
    elif ext in (".mff", ".raw"):
        return mne.io.read_raw_egi(str(path), preload=True, verbose="INFO")
    elif ext == ".set":
        return mne.io.read_raw_eeglab(str(path), preload=True, verbose="INFO")
    elif ext == ".fif":
        return mne.io.read_raw(str(path), preload=True, verbose="INFO")
    raise ValueError(f"Unknown format: {ext}")


def _load_epochs(file_path: str) -> mne.Epochs:
    """Load epochs."""
    path = Path(file_path)
    if path.suffix.lower() == ".fif":
        return mne.read_epochs(str(path), verbose="INFO")
    # Create from raw
    raw = _load_raw(file_path)
    raw.filter(0.1, 40.0, verbose="INFO")
    events, event_id = mne.events_from_annotations(raw, verbose="INFO")
    return mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=1.0,
                      baseline=(-0.2, 0), preload=True, verbose="INFO")


def plot_raw_preview(raw: mne.io.Raw, output_dir: Path, n_seconds: float = 10) -> None:
    """Raw signal preview (first N seconds, EEG channels only)."""
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found")
        return

    # Show at most 20 channels
    if len(eeg_picks) > 20:
        # Prefer standard 10-20
        std_chs = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                    "Fz", "Cz", "Pz", "Oz", "F7", "F8", "T7", "T8", "P7", "P8"]
        picks = [raw.ch_names.index(c) for c in std_chs if c in raw.ch_names][:20]
        if not picks:
            picks = eeg_picks[:20]
    else:
        picks = eeg_picks

    data = raw.get_data(picks, start=0, end=int(n_seconds * raw.info["sfreq"])) * 1e6
    times = np.arange(data.shape[1]) / raw.info["sfreq"]

    # Offset each channel for visibility
    offsets = np.arange(len(picks)) * (np.max(np.abs(data)) * 2.5 + 10)
    ch_names = [raw.ch_names[i] for i in picks]

    fig = go.Figure()
    for i, (name, offset) in enumerate(zip(ch_names, offsets)):
        fig.add_trace(go.Scatter(x=times, y=data[i] + offset, mode="lines",
                                 name=name, line=dict(color=PRIMARY, width=0.8),
                                 hovertemplate=f"{name}: %{{y:.1f}}µV<extra></extra>"))

    fig.update_layout(title=f"Raw EEG Preview (first {n_seconds}s)", xaxis_title="Time (s)",
                      yaxis_title="Amplitude (µV, offset)", template="plotly_white",
                      height=max(400, len(picks) * 30), showlegend=False)
    _save(fig, "raw_preview", output_dir)


def plot_psd(raw: mne.io.Raw, output_dir: Path, fmin: float = 0.5, fmax: float = 50) -> None:
    """Power spectral density plot."""
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        return

    # Use MNE's PSD
    fig_mpl, ax = plt.subplots(figsize=(12, 5))
    raw.plot_psd(fmin=fmin, fmax=fmax, picks=eeg_picks, ax=ax, verbose="INFO")
    ax.set_title("Power Spectral Density", fontsize=14)
    plt.tight_layout()
    png_path = output_dir / "psd.png"
    fig_mpl.savefig(str(png_path), dpi=150)
    plt.close(fig_mpl)

    # Also create interactive plotly version
    from scipy.signal import welch
    fig = go.Figure()
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(eeg_picks))))
    for i, pick in enumerate(eeg_picks[:20]):
        data = raw.get_data(pick)
        fs = raw.info["sfreq"]
        freqs, psd = welch(data[0], fs=fs, nperseg=min(1024, len(data[0]) // 4))
        mask = (freqs >= fmin) & (freqs <= fmax)
        fig.add_trace(go.Scatter(x=freqs[mask], y=10 * np.log10(psd[mask]),
                                 name=raw.ch_names[pick], mode="lines",
                                 line=dict(color=f"rgb({colors[i][0]*255:.0f},{colors[i][1]*255:.0f},{colors[i][2]*255:.0f})", width=1)))
    fig.update_layout(title="Power Spectral Density", xaxis_title="Frequency (Hz)",
                      yaxis_title="Power (dB/Hz)", template="plotly_white", height=400,
                      legend=dict(font=dict(size=8)))
    _save_html(fig, "psd", output_dir)
    logger.info("Saved PSD plots")


def plot_erp(epochs: mne.Epochs, output_dir: Path, ch: str = "Pz") -> None:
    """ERP waveform plot comparing conditions."""
    if ch not in epochs.ch_names:
        # Find closest EEG channel
        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        if len(eeg_picks) == 0:
            logger.warning("No EEG channels")
            return
        ch = epochs.ch_names[eeg_picks[len(eeg_picks) // 2]]

    colors = [PRIMARY, SECONDARY, "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#34495E", "#E67E22"]
    fig = go.Figure()
    for i, (cond_name, cond_id) in enumerate(epochs.event_id.items()):
        evoked = epochs[cond_name].average()
        pick = epochs.ch_names.index(ch)
        data = evoked.get_data(pick) * 1e6
        times = evoked.times * 1000  # ms
        sem = np.std(epochs[cond_name].get_data()[:, pick, :], axis=0) * 1e6 / np.sqrt(len(epochs[cond_name]))

        fig.add_trace(go.Scatter(x=times, y=data, mode="lines", name=cond_name,
                                 line=dict(color=colors[i % len(colors)], width=2)))
        fig.add_trace(go.Scatter(x=times, y=data + sem, mode="lines",
                                 showlegend=False, line=dict(width=0, color=colors[i % len(colors)]),
                                 fillcolor=colors[i % len(colors)] + "20"))
        fig.add_trace(go.Scatter(x=times, y=data - sem, mode="lines", fill="tonexty",
                                 showlegend=False, line=dict(width=0, color=colors[i % len(colors)]),
                                 fillcolor=colors[i % len(colors)] + "20"))

    fig.update_layout(title=f"ERP Waveforms at {ch}", xaxis_title="Time (ms)",
                      yaxis_title="Amplitude (µV)", template="plotly_white", height=450)
    _save(fig, f"erp_{ch}", output_dir)


def plot_topomap(epochs: mne.Epochs, output_dir: Path, times_ms: Optional[list[float]] = None) -> None:
    """ERP topomap at specific time points."""
    evoked = epochs.average()
    times_s = np.linspace(evoked.tmin, evoked.tmax, 8)

    if times_ms:
        times_s = [t / 1000.0 for t in times_ms if evoked.tmin <= t / 1000.0 <= evoked.tmax]
        if not times_s:
            times_s = np.linspace(evoked.tmin, evoked.tmax, 8)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    evoked.plot_topomap(times=times_s, axes=axes, show=False, verbose="INFO")
    fig.suptitle("ERP Topomaps", fontsize=14, y=1.02)
    plt.tight_layout()
    png_path = output_dir / "topomap.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved topomap: %s", png_path)

    # HTML version: create individual topomaps
    try:
        from mne.viz.backends.renderer import _get_renderer
        for t in times_s[:8]:
            fig_mpl, ax = plt.subplots(figsize=(4, 4))
            evoked.plot_topomap(times=[t], axes=ax, show=False, verbose=False)
            ax.set_title(f"{t*1000:.0f} ms")
            plt.tight_layout()
            fname = f"topomap_{int(t*1000)}ms"
            fig_mpl.savefig(str(output_dir / f"{fname}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig_mpl)
    except Exception as e:
        logger.debug("Individual topomap export skipped: %s", e)


def _save(fig: go.Figure, name: str, output_dir: Path) -> None:
    """Save Plotly figure as PNG + HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_dir / f"{name}.png"), width=1200, height=fig.layout.height or 400)
    fig.write_html(str(output_dir / f"{name}.html"))
    logger.info("Saved %s.png and %s.html", name, name)


def _save_html(fig: go.Figure, name: str, output_dir: Path) -> None:
    """Save Plotly figure as HTML only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_dir / f"{name}.html"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize EEG data")
    parser.add_argument("file_path", type=str, help="Path to EEG file (.vhdr, .fif, etc.)")
    parser.add_argument("--output-dir", type=str, default="./eeg_plots", help="Output directory")
    parser.add_argument("--raw", action="store_true", help="Plot raw signal preview")
    parser.add_argument("--psd", action="store_true", help="Plot power spectral density")
    parser.add_argument("--erp", action="store_true", help="Plot ERP waveforms")
    parser.add_argument("--topomap", action="store_true", help="Plot ERP topomaps")
    parser.add_argument("--channel", type=str, default="Pz", help="Channel for ERP plot")
    parser.add_argument("--topo-times", nargs="+", type=float, help="Topomap times (ms)")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.all:
        args.raw = args.psd = args.erp = args.topomap = True

    if args.raw or args.psd:
        raw = _load_raw(args.file_path)
        raw.filter(0.1, 40.0, verbose="INFO")
        if args.raw:
            plot_raw_preview(raw, out)
        if args.psd:
            plot_psd(raw, out)

    if args.erp or args.topomap:
        epochs = _load_epochs(args.file_path)
        if args.erp:
            plot_erp(epochs, out, args.channel)
        if args.topomap:
            plot_topomap(epochs, out, args.topo_times)

    print(f"✅ EEG plots saved to {out.resolve()}")


if __name__ == "__main__":
    main()
