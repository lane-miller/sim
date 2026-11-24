#!/usr/bin/env python3
"""Step 12: dry-run reports and s6 geometry plot (no stepping)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from boxrad.config import Config
from boxrad.geom.geometry import build_geometry, radiator_clearance
from boxrad.report import format_geom_report, geom_report
from boxrad.plot.geometry import plot_geometry

_PKG = Path(__file__).resolve().parents[1]
_SHOWCASE = _PKG / "configs" / "showcase"
_OUT = _PKG / "outputs" / "step12"


def _load(name: str) -> Config:
    return Config.from_json((_SHOWCASE / name).read_text(encoding="utf-8"))


def _json_numpy(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def main() -> None:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    configs = {
        "s1": "s1_cylinder.json",
        "s2": "s2_bookshelf.json",
        "s3": "s3p_c09_parametric.json",
        "s4": "s4_tilted.json",
        "s5": "s5_platform.json",
        "s6": "s6_downfire.json",
        "c13_sphere_ps": str(_PKG / "configs" / "c13_sphere_ps.json"),
    }
    all_reports: dict[str, object] = {}
    for key, path in configs.items():
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = _SHOWCASE / path
        cfg = Config.from_json(cfg_path.read_text(encoding="utf-8"))
        rep = geom_report(cfg, cfg_path=cfg_path)
        all_reports[key] = rep
        print(format_geom_report(rep), end="")
        print(f"--- {key} ---")

    json_path = _OUT / "reports.json"
    json_path.write_text(
        json.dumps(all_reports, indent=2, default=_json_numpy), encoding="utf-8"
    )
    print(f"wrote {json_path}")

    s6 = _load("s6_downfire.json")
    geom = build_geometry(s6)
    assert geom is not None
    rd = geom.radiators[0]
    c = rd.pose.to_world(
        np.array([rd.outline.centroid[0], rd.outline.centroid[1], 0.0], dtype=np.float64)
    )
    png_path = _OUT / "s6_geometry.png"
    plot_geometry(
        s6,
        out=png_path,
        slice_xyz=(float(c[0]), float(c[1]), float(c[2])),
        name="s6_downfire",
    )
    print(f"wrote {png_path}")

    rep_s6 = all_reports["s6"]
    h = rep_s6["grid"]["h_m"]
    cmin, _ = radiator_clearance(geom, rd, n=64)
    print(
        f"s6 woofer clearance min={cmin:g} m at h={h:g} m "
        f"({cmin / h:.2f} cells); warnings: {rep_s6.get('warnings')}"
    )

    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()