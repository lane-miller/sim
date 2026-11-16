#!/usr/bin/env python3
"""List or delete kept validation output trees under proj/boxed_piston/outputs/."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# ``validation`` package import (install boxrad separately).
sys.path.insert(0, str(ROOT))

import validation._common as vc  # noqa: E402


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List or delete proj/boxed_piston/outputs/ (kept --keep-h5 runs)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete listed trees (default is dry run)",
    )
    args = parser.parse_args(argv)

    out_root = ROOT / "outputs"
    if not out_root.is_dir():
        print(f"No directory {out_root}")
        return 0

    entries = sorted(p for p in out_root.iterdir() if p.is_dir())
    if not entries:
        print(f"{out_root} is empty")
        return 0

    total = 0
    print(f"Contents of {out_root}:")
    for p in entries:
        nbytes = _dir_size(p)
        total += nbytes
        print(f"  {p.name}/  {vc.fmt_bytes(nbytes)}")
    print(f"  total: {vc.fmt_bytes(total)}")

    if not args.yes:
        print("\nDry run — pass --yes to delete the directories above.")
        return 0

    for p in entries:
        shutil.rmtree(p)
    print(f"\nDeleted {len(entries)} director{'y' if len(entries) == 1 else 'ies'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())