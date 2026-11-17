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
    """``boxrad-info H5``: print summary, provenance (``boxrad_version``,
    ``zrad_source``, ``/results/zrad`` range and point count), ``/meta``
    conventions, ``/meta/accuracy``, every dataset, file size.

    Args:
        argv: Argument list without the program name (default ``sys.argv[1:]``).

    Returns:
        Exit code 0.

    Raises:
        FileNotFoundError: ``H5`` does not exist.
    """
    parser = argparse.ArgumentParser(description="Summarize a boxrad HDF5 result file")
    parser.add_argument("h5", type=Path, help="HDF5 result path")
    args = parser.parse_args(argv)
    path = args.h5
    res = Result(path)
    print(res.summary())
    meta = res.meta
    print("\n--- provenance ---")
    print(f"  boxrad_version: {meta.get('boxrad_version', '?')}")
    print(f"  zrad_source: {meta.get('zrad_source', '?')}")
    f_zr, _ = res.z()
    if f_zr.size:
        print(
            f"  /results/zrad: {f_zr.size} points, "
            f"{f_zr.min().item():.6g}–{f_zr.max().item():.6g} Hz"
        )
    else:
        print("  /results/zrad: empty")
    conventions = str(meta.get("conventions", "")).strip()
    print("\n--- conventions ---")
    for line in conventions.splitlines() or ["?"]:
        print(f"  {line}")
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