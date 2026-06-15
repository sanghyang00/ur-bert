#!/usr/bin/env python3
"""
Create small dummy CSVs from large source CSVs by streaming rows.

This script avoids loading full files into memory and keeps the same
column format as the originals.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def sample_csv_streaming(src_path: Path, dst_path: Path, rows: int, stride: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with src_path.open("r", encoding="utf-8", newline="") as src_f, dst_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst_f:
        reader = csv.reader(src_f)
        writer = csv.writer(dst_f)

        header = next(reader, None)
        if header is None:
            writer.writerow([])
            return
        writer.writerow(header)

        written = 0
        for idx, row in enumerate(reader):
            if idx % stride != 0:
                continue
            writer.writerow(row)
            written += 1
            if written >= rows:
                break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create dummy CSVs by streaming from source CSV files."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("csvs"),
        help="Source directory containing CSV files (default: csvs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("csvs_dummy"),
        help="Output directory for dummy CSVs (default: csvs_dummy).",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=200,
        help="Number of data rows per output CSV (default: 200).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Write one row every N rows while streaming (default: 50).",
    )
    args = parser.parse_args()

    csv_files = sorted(args.src_dir.glob("*/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {args.src_dir}")

    for src_path in csv_files:
        rel = src_path.relative_to(args.src_dir)
        dst_path = args.out_dir / rel
        sample_csv_streaming(
            src_path=src_path,
            dst_path=dst_path,
            rows=args.rows_per_file,
            stride=args.stride,
        )
        print(f"Created: {dst_path}")


if __name__ == "__main__":
    main()
