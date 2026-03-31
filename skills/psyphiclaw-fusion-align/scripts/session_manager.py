#!/usr/bin/env python3
"""MultiModalSession: core data structure for managing aligned multimodal data.

Provides CRUD operations for modalities, HDF5 persistence, and metadata
management.  Designed to be compatible with psyphiclaw-face-import (pandas
DataFrame) and psyphiclaw-eeg-import (MNE Raw/Epochs) outputs.

Usage:
    # CLI
    python session_manager.py create --name "subject_001" --output session.h5
    python session_manager.py add-modality --session session.h5 \
        --modality eeg --file eeg.csv --sampling-rate 250
    python session_manager.py list --session session.h5
    python session_manager.py export --session session.h5 --output export_dir/
    python session_manager.py info --session session.h5

    # Python API
    from session_manager import MultiModalSession
    session = MultiModalSession(name="subject_001")
    session.add_modality("eeg", df, timestamps_ms, 250.0)
    session.save("session.h5")
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# Optional MNE support
try:
    import mne  # type: ignore[import-untyped]
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

BLUE = "#4A90D9"
RED = "#E74C3C"


class ModalityData:
    """Container for a single modality's aligned data."""

    def __init__(
        self,
        name: str,
        data: Union[pd.DataFrame, np.ndarray, "mne.io.Raw", "mne.Epochs"],
        timestamps_ms: Optional[np.ndarray] = None,
        sampling_rate: Optional[float] = None,
        source: str = "",
        columns: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self.sampling_rate = sampling_rate
        self.source = source
        self._mne_object: Optional[Any] = None

        # Handle MNE objects
        if HAS_MNE and isinstance(data, (mne.io.Raw, mne.Epochs)):
            self._mne_object = data
            self.sampling_rate = data.info["sfreq"]
            if isinstance(data, mne.io.Raw):
                self.data = data.get_data()  # (n_channels, n_times)
                self.columns = list(data.ch_names)
                n_times = data.n_times
                self.timestamps_ms = (
                    np.arange(n_times) / self.sampling_rate * 1000.0
                    + data.first_time * 1000.0
                )
                self.events = np.array([])  # placeholder
            else:  # Epochs
                self.data = data.get_data()  # (n_epochs, n_channels, n_times)
                self.columns = list(data.info["ch_names"])
                self.timestamps_ms = timestamps_ms if timestamps_ms is not None else np.array([])
                self.events = data.events if hasattr(data, "events") else np.array([])
            return

        # pandas DataFrame
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.columns = columns if columns else list(data.columns)
        elif isinstance(data, np.ndarray):
            self.data = data
            self.columns = columns if columns else [
                f"ch_{i}" for i in range(data.shape[1] if data.ndim > 1 else 1)
            ]
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Ensure 2-D
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        self.timestamps_ms = timestamps_ms
        if self.timestamps_ms is not None and sampling_rate is None and len(self.timestamps_ms) > 1:
            dt = np.median(np.diff(self.timestamps_ms)) / 1000.0
            self.sampling_rate = 1.0 / dt if dt > 0 else None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with timestamps."""
        if self.timestamps_ms is not None and len(self.timestamps_ms) == len(self.data):
            df = pd.DataFrame(self.data, columns=self.columns)
            df.insert(0, "timestamp_ms", self.timestamps_ms)
            return df
        return pd.DataFrame(self.data, columns=self.columns)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict."""
        return {
            "name": self.name,
            "shape": list(self.data.shape) if isinstance(self.data, np.ndarray) else "mne_object",
            "sampling_rate": self.sampling_rate,
            "columns": self.columns,
            "source": self.source,
            "duration_ms": (
                float(self.timestamps_ms[-1] - self.timestamps_ms[0])
                if self.timestamps_ms is not None and len(self.timestamps_ms) > 1
                else None
            ),
        }


class MultiModalSession:
    """Manage multiple aligned modalities and persist to HDF5."""

    def __init__(self, name: str = "unnamed_session") -> None:
        self.name = name
        self.modalities: dict[str, ModalityData] = {}
        self.metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "aligned": False,
        }

    def add_modality(
        self,
        name: str,
        data: Union[pd.DataFrame, np.ndarray, Any],
        timestamps_ms: Optional[np.ndarray] = None,
        sampling_rate: Optional[float] = None,
        source: str = "",
        columns: Optional[list[str]] = None,
    ) -> None:
        """Register a new modality."""
        self.modalities[name] = ModalityData(
            name=name,
            data=data,
            timestamps_ms=timestamps_ms,
            sampling_rate=sampling_rate,
            source=source,
            columns=columns,
        )
        print(f"  [+{name}] shape={self.modalities[name].data.shape if isinstance(self.modalities[name].data, np.ndarray) else 'mne'}, "
              f"sfreq={self.modalities[name].sampling_rate}")

    def remove_modality(self, name: str) -> None:
        """Remove a modality by name."""
        if name in self.modalities:
            del self.modalities[name]
            print(f"  [-{name}] removed")

    def get_modality(self, name: str) -> Optional[ModalityData]:
        """Retrieve a modality by name."""
        return self.modalities.get(name)

    def list_modalities(self) -> list[dict[str, Any]]:
        """Summarize all registered modalities."""
        return [m.summary() for m in self.modalities.values()]

    def save(self, path: Union[str, Path]) -> None:
        """Save session to HDF5."""
        if not HAS_H5PY:
            # Fallback: save as directory of CSVs + metadata JSON
            self._save_csv_fallback(path)
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as hf:
            # Metadata
            meta_grp = hf.create_group("metadata")
            for k, v in self.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta_grp.attrs[k] = v
                elif isinstance(v, (list, dict)):
                    meta_grp.attrs[k] = json.dumps(v)

            # Modalities
            for name, mod in self.modalities.items():
                grp = hf.create_group(name)
                if isinstance(mod.data, np.ndarray):
                    grp.create_dataset("data", data=mod.data, compression="gzip")
                if mod.timestamps_ms is not None:
                    grp.create_dataset("timestamps_ms", data=mod.timestamps_ms)
                if mod.columns:
                    grp.attrs["columns"] = json.dumps(mod.columns)
                if mod.sampling_rate:
                    grp.attrs["sampling_rate"] = mod.sampling_rate
                if mod.source:
                    grp.attrs["source"] = mod.source
                # Events (from MNE)
                if hasattr(mod, "events") and mod.events is not None and len(mod.events) > 0:
                    grp.create_dataset("events", data=mod.events)

        print(f"✅ Session saved to {path}")

    def _save_csv_fallback(self, path: Union[str, Path]) -> None:
        """Fallback: save each modality as CSV + metadata as JSON."""
        out_dir = Path(path)
        if out_dir.suffix == ".h5":
            out_dir = out_dir.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, mod in self.modalities.items():
            df = mod.to_dataframe()
            df.to_csv(out_dir / f"{name}.csv", index=False)

        meta_path = out_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ Session saved (CSV fallback) to {out_dir}/")

    @classmethod
    def load(cls, path: Union[str, Path]) -> MultiModalSession:
        """Load session from HDF5 or CSV directory."""
        path = Path(path)
        session = cls(name=path.stem)

        if HAS_H5PY and path.suffix == ".h5" and path.exists():
            with h5py.File(path, "r") as hf:
                # Metadata
                if "metadata" in hf:
                    for k, v in hf["metadata"].attrs.items():
                        if isinstance(v, bytes):
                            v = v.decode()
                        try:
                            session.metadata[k] = json.loads(v)
                        except (json.JSONDecodeError, TypeError):
                            session.metadata[k] = v

                # Modalities
                for name in hf:
                    if name == "metadata":
                        continue
                    grp = hf[name]
                    data = grp["data"][:] if "data" in grp else None
                    ts = grp["timestamps_ms"][:] if "timestamps_ms" in grp else None
                    cols = json.loads(grp.attrs["columns"]) if "columns" in grp.attrs else None
                    sfreq = float(grp.attrs["sampling_rate"]) if "sampling_rate" in grp.attrs else None
                    src = str(grp.attrs["source"]) if "source" in grp.attrs else ""
                    events = grp["events"][:] if "events" in grp else None

                    if data is not None:
                        session.add_modality(name, data, ts, sfreq, src, cols)
                        if events is not None:
                            session.modalities[name].events = events
            print(f"📂 Loaded session from {path}")
            return session

        # CSV fallback
        if path.is_dir():
            meta_path = path / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    session.metadata.update(json.load(f))
            for csv_file in sorted(path.glob("*.csv")):
                if csv_file.stem == "metadata":
                    continue
                name = csv_file.stem
                df = pd.read_csv(csv_file)
                ts_col = None
                for c in ("timestamp_ms", "timestamp", "Timestamp"):
                    if c in df.columns:
                        ts_col = c
                        break
                ts = df[ts_col].values if ts_col else None
                session.add_modality(name, df, ts)
            print(f"📂 Loaded session (CSV) from {path}")
            return session

        raise FileNotFoundError(f"No session found at {path}")

    def export_csv(self, output_dir: Union[str, Path]) -> None:
        """Export all modalities as CSV files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, mod in self.modalities.items():
            df = mod.to_dataframe()
            df.to_csv(out / f"{name}.csv", index=False)
        meta_path = out / "session_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"📁 Exported {len(self.modalities)} modalities to {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage MultiModalSession objects."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new session.")
    p_create.add_argument("--name", required=True, help="Session name.")
    p_create.add_argument("--output", required=True, help="Output .h5 path.")

    # add-modality
    p_add = sub.add_parser("add-modality", help="Add a modality to an existing session.")
    p_add.add_argument("--session", required=True, help="Session .h5 path.")
    p_add.add_argument("--modality", required=True, help="Modality name.")
    p_add.add_argument("--file", required=True, help="CSV file path.")
    p_add.add_argument("--sampling-rate", type=float, default=None, help="Sampling rate Hz.")
    p_add.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")

    # list
    p_list = sub.add_parser("list", help="List modalities in a session.")
    p_list.add_argument("--session", required=True, help="Session .h5 path.")

    # info
    p_info = sub.add_parser("info", help="Show session info.")
    p_info.add_argument("--session", required=True, help="Session .h5 path.")

    # export
    p_export = sub.add_parser("export", help="Export session to CSV directory.")
    p_export.add_argument("--session", required=True, help="Session .h5 path.")
    p_export.add_argument("--output", required=True, help="Output directory.")

    args = parser.parse_args()

    if args.command == "create":
        session = MultiModalSession(name=args.name)
        session.save(args.output)

    elif args.command == "add-modality":
        session = MultiModalSession.load(args.session)
        df = pd.read_csv(args.file)
        ts_col = args.timestamp_col
        ts = df[ts_col].values if ts_col in df.columns else None
        session.add_modality(
            name=args.modality,
            data=df,
            timestamps_ms=ts,
            sampling_rate=args.sampling_rate,
            source=args.file,
        )
        session.save(args.session)

    elif args.command == "list":
        session = MultiModalSession.load(args.session)
        mods = session.list_modalities()
        print(f"\n📋 Session: {session.name} ({len(mods)} modalities)")
        for m in mods:
            print(f"  {m['name']}: shape={m['shape']}, sfreq={m['sampling_rate']}")

    elif args.command == "info":
        session = MultiModalSession.load(args.session)
        print(f"\n📋 Session: {session.name}")
        print(f"   Created: {session.metadata.get('created_at', 'N/A')}")
        print(f"   Aligned: {session.metadata.get('aligned', False)}")
        print(f"   Modalities: {list(session.modalities.keys())}")

    elif args.command == "export":
        session = MultiModalSession.load(args.session)
        session.export_csv(args.output)


if __name__ == "__main__":
    main()
