"""格式转换工具 — 在不同数据格式之间转换。"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def csv_to_hdf5(
    csv_path: str,
    output_path: str,
    key: str = "data",
    encoding: str = "utf-8",
) -> None:
    """将 CSV 文件转换为 HDF5 格式。"""
    df = pd.read_csv(csv_path, encoding=encoding)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_hdf(output_path, key=key, mode="w")
    print(f"[OK] {csv_path} -> {output_path} (rows={len(df)}, cols={len(df.columns)})")


def batch_csv_to_hdf5(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.csv",
    key: str = "data",
) -> list[str]:
    """批量转换 CSV 为 HDF5。"""
    csv_files = sorted(Path(input_dir).glob(pattern))
    converted = []
    for csv_file in csv_files:
        out = Path(output_dir) / csv_file.with_suffix(".h5").name
        try:
            csv_to_hdf5(str(csv_file), str(out), key=key)
            converted.append(str(out))
        except Exception as e:
            print(f"[ERR] {csv_file}: {e}", file=sys.stderr)
    print(f"\nConverted {len(converted)}/{len(csv_files)} files")
    return converted


def export_session_json(
    session_data: dict,
    output_path: str,
) -> None:
    """将 session 数据导出为 JSON。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Session saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PsyPhiClaw format converter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_csv = subparsers.add_parser("csv2hdf5", help="Convert CSV to HDF5")
    p_csv.add_argument("csv_path", help="Input CSV file or directory")
    p_csv.add_argument("-o", "--output", required=True, help="Output HDF5 file or directory")
    p_csv.add_argument("--pattern", default="*.csv", help="Glob pattern for batch (default: *.csv)")
    p_csv.add_argument("--key", default="data", help="HDF5 key (default: data)")

    p_json = subparsers.add_parser("to-json", help="Export session to JSON")
    p_json.add_argument("json_path", help="Output JSON file")

    args = parser.parse_args()

    if args.command == "csv2hdf5":
        if os.path.isdir(args.csv_path):
            batch_csv_to_hdf5(args.csv_path, args.output, args.pattern, args.key)
        else:
            csv_to_hdf5(args.csv_path, args.output, key=args.key)

    elif args.command == "to-json":
        print("[INFO] to-json requires session data from stdin or file")
        print("[INFO] Use with: python convert_format.py to-json output.json < session.json")


if __name__ == "__main__":
    main()
