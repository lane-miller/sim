#!/usr/bin/env python3
"""Step 2 showcase: radiator poses and wireframe geometry for S1/S2 fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from boxrad.config import Config
from boxrad.frames import Pose
from boxrad.geomconfig import GeometrySpec, PrimitiveSpec, parse_geometry

_PKG = Path(__file__).resolve().parents[1]
_SHOWCASE_CFG = _PKG / "configs" / "showcase"
_OUT = Path(__file__).resolve().parent / "out" / "step02_frames.png"

RTOL = 1e-12


def _circle_xy(radius: float, n: int = 64) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)))


def _wire_box(pose: Pose, size: tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = size
    corners = np.array(
        [
            [-0.5 * sx, -0.5 * sy, -0.5 * sz],
            [0.5 * sx, -0.5 * sy, -0.5 * sz],
            [0.5 * sx, 0.5 * sy, -0.5 * sz],
            [-0.5 * sx, 0.5 * sy, -0.5 * sz],
            [-0.5 * sx, -0.5 * sy, 0.5 * sz],
            [0.5 * sx, -0.5 * sy, 0.5 * sz],
            [0.5 * sx, 0.5 * sy, 0.5 * sz],
            [-0.5 * sx, 0.5 * sy, 0.5 * sz],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    pts = pose.to_world(corners)
    return pts, edges


def _wire_cylinder(pose: Pose, diameter: float, length: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    r = 0.5 * diameter
    n = 48
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    bot = np.column_stack((r * np.cos(t), r * np.sin(t), -0.5 * length * np.ones(n)))
    top = np.column_stack((r * np.cos(t), r * np.sin(t), 0.5 * length * np.ones(n)))
    local = np.vstack((bot, top))
    world = pose.to_world(local)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        j = n + i
        edges.append((j, n + (i + 1) % n))
    for k in (0, n // 4, n // 2, 3 * n // 4):
        edges.append((k, n + k))
    return world, edges


def _draw_edges(ax, pts: np.ndarray, edges: list[tuple[int, int]], **kwargs) -> None:
    for i, j in edges:
        seg = pts[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], **kwargs)


def _draw_triad(ax, pose: Pose, length: float = 0.05) -> None:
    o = pose.position
    R = pose.R
    colors = ("red", "green", "blue")
    for k, color in enumerate(colors):
        tip = o + length * R[:, k]
        ax.plot([o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]], color=color, lw=1.5)


def _draw_world_triad(ax, length: float = 0.05) -> None:
    _draw_triad(ax, Pose.identity(), length=length)


def _radiator_metrics(name: str, pose: Pose) -> tuple[str, bool]:
    R = pose.R
    det_r = float(np.linalg.det(R))
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    center = pose.position
    rt = float(np.linalg.norm(pose.to_world(pose.to_local(center)) - center))
    n_hat = pose.normal / np.linalg.norm(pose.normal)
    mapped = pose.to_world(np.array([0.0, 0.0, 1.0]))
    expected = center + n_hat
    map_err = float(np.linalg.norm(mapped - expected))
    ok = max(rt, map_err, ortho_err) <= RTOL
    flag = "PASS" if ok else "CHECK"
    lines = [
        f"{name}: pos={center}, n={pose.normal}, x={pose.x_axis}",
        f"  det R={det_r:.6g}  ||RᵀR-I||_max={ortho_err:.3e}  center RT={rt:.3e}  z→world err={map_err:.3e}  {flag}",
    ]
    return "\n".join(lines), ok


def _plot_spec(ax, spec: GeometrySpec, title: str) -> list[bool]:
    checks: list[bool] = []
    for part in spec.parts:
        if isinstance(part, PrimitiveSpec):
            if part.kind == "box":
                sz = (part.params["sx"], part.params["sy"], part.params["sz"])
                pts, edges = _wire_box(part.pose, sz)
            elif part.kind == "cylinder":
                pts, edges = _wire_cylinder(part.pose, part.params["d"], part.params["length"])
            else:
                continue
            _draw_edges(ax, pts, edges, color="0.35", lw=0.8)
    for rd in spec.radiators:
        assert rd.pose is not None and rd.shape is not None
        d = rd.shape.params.get("d", 0.1)
        circ = _circle_xy(0.5 * d)
        wcirc = rd.pose.to_world(circ)
        ax.plot(wcirc[:, 0], wcirc[:, 1], wcirc[:, 2], "k-", lw=1.2)
        _draw_triad(ax, rd.pose)
        text, ok = _radiator_metrics(rd.name, rd.pose)
        print(text)
        checks.append(ok)
    _draw_world_triad(ax)
    for pl in spec.environment.planes:
        if pl.axis == "z" and pl.offset == 0.0:
            s = 0.3
            sq = np.array(
                [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [-s, -s, 0]],
                dtype=np.float64,
            )
            ax.plot(sq[:, 0], sq[:, 1], sq[:, 2], color="cyan", alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return checks


def _set_equal_aspect(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    span = (limits[:, 1] - limits[:, 0]).max() * 0.5
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the plot window")
    args = parser.parse_args()
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"step02_frames start {t0}")

    specs: list[tuple[str, GeometrySpec]] = []
    for name in ("s1_cylinder.json", "s2_bookshelf.json"):
        path = _SHOWCASE_CFG / name
        raw = json.loads(path.read_text(encoding="utf-8"))
        Config.from_json(path.read_text(encoding="utf-8"))
        specs.append((name, parse_geometry(raw)))

    if specs[1][0] == "s2_bookshelf.json":
        s2 = specs[1][1]
        woofer = next(r for r in s2.radiators if r.name == "woofer")
        tweeter = next(r for r in s2.radiators if r.name == "tweeter")
        dist = float(np.linalg.norm(woofer.pose.position - tweeter.pose.position))
        print(f"S2 woofer–tweeter distance: {dist:.4f} m (expected 0.15)")
        for r in (woofer, tweeter):
            y_err = float(r.pose.position[1] - 0.125)
            print(f"  {r.name} y={r.pose.position[1]:.4f}  (y-0.125)={y_err:.3e}")

    fig = plt.figure(figsize=(12, 5))
    all_ok: list[bool] = []
    for i, (name, spec) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        all_ok.extend(_plot_spec(ax, spec, name))
        _set_equal_aspect(ax)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(_OUT, dpi=120)
    print(f"summary: {'PASS' if all(all_ok) else 'CHECK'}  → {_OUT}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()