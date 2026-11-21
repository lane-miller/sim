#!/usr/bin/env python3
"""Print license metadata for boxrad-related distributions (no network).

Run from a venv with base and optional extras installed. Missing packages print
NOT INSTALLED. Flags metadata that mentions GPL without LGPL, or non-commercial.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

DIST_NAMES = (
    "numpy",
    "scipy",
    "numba",
    "llvmlite",
    "h5py",
    "matplotlib",
    "manifold3d",
    "cadquery-ocp-novtk",
    "bempp-cl",
)

LICENSE_NAME_RE = re.compile(r"LICENSE|COPYING|NOTICE|OCCT|LGPL", re.IGNORECASE)
SHARED_LIB_SUFFIXES = (".dylib", ".so", ".dll")

_NON_COMMERCIAL = re.compile(r"non[- ]commercial", re.IGNORECASE)


def _metadata_text(meta: importlib.metadata.PackageMetadata) -> str:
    parts: list[str] = []
    for key in meta:
        parts.extend(meta.get_all(key) or [])
    return "\n".join(parts)


def _flag_metadata(text: str, dist_name: str) -> list[str]:
    flags: list[str] = []
    if re.search(r"GPL", text, re.IGNORECASE) and not re.search(
        r"LGPL", text, re.IGNORECASE
    ):
        flags.append(f"{dist_name}: metadata mentions GPL without LGPL")
    if _NON_COMMERCIAL.search(text):
        flags.append(f"{dist_name}: metadata mentions non-commercial terms")
    return flags


def _print_dist(dist_name: str) -> list[str]:
    print(f"=== {dist_name} ===")
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        print("NOT INSTALLED")
        print()
        return []

    meta = dist.metadata
    version = dist.version
    print(f"name:    {meta['Name']}")
    print(f"version: {version}")
    print(f"License: {meta.get('License', '')}")
    print(f"License-Expression: {meta.get('License-Expression', '')}")
    classifiers = meta.get_all("Classifier") or []
    license_classifiers = [c for c in classifiers if c.startswith("License ::")]
    if license_classifiers:
        print("License classifiers:")
        for c in license_classifiers:
            print(f"  {c}")
    else:
        print("License classifiers: (none)")

    flags = _flag_metadata(_metadata_text(meta), dist_name)

    files = dist.files
    if not files:
        print("dist.files: (empty)")
        print()
        return flags

    license_files: list[Path] = []
    for file in files:
        base = Path(file.name).name
        if LICENSE_NAME_RE.search(base):
            license_files.append(file.locate())
            print(f"license-related file: {file.name}")

    if dist_name == "cadquery-ocp-novtk":
        shared: list[str] = []
        for file in files:
            if file.name.lower().endswith(SHARED_LIB_SUFFIXES):
                shared.append(file.name)
        if shared:
            print("bundled shared libraries:")
            for name in sorted(shared):
                print(f"  {name}")
        else:
            print("bundled shared libraries: (none listed in dist.files)")

        for path in license_files:
            base = path.name.upper()
            if "OCCT" in base or "LGPL" in base or LICENSE_NAME_RE.search(path.name):
                print(f"--- first 40 lines of {path.name} ---")
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError as exc:
                    print(f"(could not read: {exc})")
                    continue
                for line in lines[:40]:
                    print(line)
                print("--- end excerpt ---")

    print()
    return flags


def main() -> int:
    all_flags: list[str] = []
    for name in DIST_NAMES:
        all_flags.extend(_print_dist(name))

    if all_flags:
        print("FLAGS:")
        for flag in all_flags:
            print(f"  {flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())