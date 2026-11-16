"""CLI summary for boxrad HDF5 results."""

from __future__ import annotations

import argparse
from pathlib import Path

from boxrad.result import Result


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    return f"{nbytes / 1e3:.1f} kB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a boxrad HDF5 result file")
    parser.add_argument("h5", type=Path, help="HDF5 result path")
    args = parser.parse_args(argv)
    path = args.h5
    res = Result(path)
    print(res.summary())
    print("\n--- accuracy ---")
    for key, val in sorted(res.accuracy.items()):
        print(f"  {key}: {val}")
    print("\n--- datasets ---")
    import h5py

    with h5py.File(path, "r") as f:
        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")

        f.visititems(visit)
    print(f"\nfile size: {_fmt_size(path.stat().st_size)}")
    res.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())