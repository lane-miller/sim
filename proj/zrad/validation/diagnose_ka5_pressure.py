"""Diagnose ka=5 baffle pressure field: radial samples, ripple stats, mesh vs λ."""

import json
import os

import bempp_cl.api as bempp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, SymLogNorm
import matplotlib.cm as cm

from mesh import BAFFLE_TAG, C, PISTON_TAG, RHO, U0
from plot_ka5_slide import (
    A_M,
    CMAP,
    DPI,
    KA,
    MESH_FILE,
    OUT_DIR,
    PRESSURE_CACHE,
    P_REF,
    _axis_limits,
    _style_axes,
    _top_surface_mask,
    _top_surface_triangles,
)

RESULT_JSON = os.path.join(
    os.path.dirname(__file__), "results", "offcenter_sweep", "result_dy0_ka5.json"
)


def load_field():
    data = np.load(PRESSURE_CACHE)
    grid = bempp.import_grid(MESH_FILE)
    p_gf = bempp.GridFunction(
        bempp.function_space(grid, "P", 1),
        coefficients=data["coefficients"],
    )
    return grid, p_gf


def top_surface_data(grid, p_gf):
    mask = _top_surface_mask(grid)
    elem_idx = np.where(mask)[0]
    domain = grid.domain_indices.astype(int)
    centers = np.asarray(grid.centroids)[elem_idx].T  # (3, n)
    areas = grid.volumes.astype(float)[elem_idx]
    p_centers = p_gf.evaluate_on_element_centers().ravel()
    p_mag = np.abs(p_centers[elem_idx]) / P_REF
    p_cplx = p_centers[elem_idx] / P_REF
    is_piston = domain[elem_idx] == PISTON_TAG
    is_baffle = domain[elem_idx] == BAFFLE_TAG
    xy = centers[:2, :].T
    r = np.hypot(centers[0], centers[1])
    return {
        "elem_idx": elem_idx,
        "xy": xy,
        "r": r,
        "p_mag": p_mag,
        "p_cplx": p_cplx,
        "areas": areas,
        "is_piston": is_piston,
        "is_baffle": is_baffle,
    }


def element_size_at_radius(grid, elem_idx, r_query, band_frac=0.08):
    """Mean triangle edge length for top elements near radius r_query."""
    verts = np.asarray(grid.vertices)
    elems = np.asarray(grid.elements)
    centers = np.asarray(grid.centroids)[elem_idx]
    r = np.hypot(centers[:, 0], centers[:, 1])
    band = band_frac * r_query
    sel = np.abs(r - r_query) < band
    if not np.any(sel):
        return float("nan"), 0
    local_idx = elem_idx[sel]
    edge_lens = []
    for e in local_idx:
        vids = elems[:, e]
        pts = verts[:2, vids].T
        for i in range(3):
            edge_lens.append(np.linalg.norm(pts[(i + 1) % 3] - pts[i]))
    return float(np.mean(edge_lens)), int(np.sum(sel))


def sample_radial_ring(data, r_from_edge, a=A_M, n_bins=36):
    """Stats on baffle-only elements in an annulus at radius a + r_from_edge."""
    r_center = a + r_from_edge
    r = data["r"]
    band = 0.06 * r_from_edge if r_from_edge > 0 else 0.002
    sel = (
        data["is_baffle"]
        & (np.abs(r - r_center) < max(band, 0.0015))
    )
    if np.sum(sel) < 5:
        return None

    p_mag = data["p_mag"][sel]
    p_cplx = data["p_cplx"][sel]
    xy = data["xy"][sel]
    theta = np.arctan2(xy[:, 1], xy[:, 0])

    # Sort by azimuth for a 1-D profile along the ring.
    order = np.argsort(theta)
    p_ring = p_mag[order]
    theta_ring = theta[order]

    # Ripple metric: peak-to-peak relative to mean on the ring.
    p_mean = np.mean(p_mag)
    p_ptp = np.max(p_mag) - np.min(p_mag)
    p_std = np.std(p_mag)
    p_cv = p_std / p_mean if p_mean > 0 else float("nan")

    # Also check phase variation (interference signature).
    phase = np.angle(p_cplx[order])
    phase_ptp = np.max(phase) - np.min(phase)

    return {
        "r_from_edge": r_from_edge,
        "r_center_m": r_center,
        "n_elem": int(np.sum(sel)),
        "p_mean": p_mean,
        "p_min": np.min(p_mag),
        "p_max": np.max(p_mag),
        "p_ptp": p_ptp,
        "p_std": p_std,
        "p_cv": p_cv,
        "phase_ptp_rad": phase_ptp,
        "theta_ring": theta_ring,
        "p_ring": p_ring,
    }


def print_diagnostics(grid, data):
    k = KA / A_M
    wavelength = 2.0 * np.pi / k
    h_outer = wavelength / 6.0

    print("=" * 72)
    print("ka=5 centered pressure diagnostics (top surface, |p|/(ρcU₀))")
    print("=" * 72)
    print(f"  k={k:.1f} rad/m   λ={wavelength*1000:.3f} mm   h_outer=λ/6={h_outer*1000:.3f} mm")
    print(f"  a={A_M*1000:.3f} mm")

    baffle = data["is_baffle"]
    piston = data["is_piston"]
    p = data["p_mag"]

    print("\nGlobal |p| on top surface:")
    print(f"  all:    min={p.min():.6f}  max={p.max():.6f}  mean={p.mean():.6f}")
    print(f"  piston: min={p[piston].min():.6f}  max={p[piston].max():.6f}  mean={p[piston].mean():.6f}")
    print(f"  baffle: min={p[baffle].min():.6f}  max={p[baffle].max():.6f}  mean={p[baffle].mean():.6f}")

    print("\nRadial ring samples (baffle only, from piston edge):")
    print(
        f"  {'r/a':>6}  {'r [mm]':>8}  {'n':>4}  {'mean':>8}  {'min':>8}  {'max':>8}"
        f"  {'ptp':>8}  {'CV':>7}  {'Δφ [deg]':>9}  {'h_elem [mm]':>11}  {'h/λ':>6}"
    )
    radii_a = [1.5, 2.0, 3.0, 5.0]
    ring_results = []
    for ra in radii_a:
        r_edge = ra * A_M
        ring = sample_radial_ring(data, r_edge)
        if ring is None:
            print(f"  {ra:6.1f}  — insufficient elements")
            continue
        h_elem, _ = element_size_at_radius(grid, data["elem_idx"], ring["r_center_m"])
        h_over_lambda = h_elem / wavelength
        print(
            f"  {ra:6.1f}  {ring['r_center_m']*1000:8.3f}  {ring['n_elem']:4d}"
            f"  {ring['p_mean']:8.5f}  {ring['p_min']:8.5f}  {ring['p_max']:8.5f}"
            f"  {ring['p_ptp']:8.5f}  {ring['p_cv']:7.3f}"
            f"  {np.degrees(ring['phase_ptp_rad']):9.1f}"
            f"  {h_elem*1000:11.3f}  {h_over_lambda:6.3f}"
        )
        ring_results.append(ring)

    # Sample a few explicit points on +x axis.
    print("\nExplicit +x axis samples (nearest baffle element center):")
    for ra in radii_a:
        r_target = A_M + ra * A_M
        baffle_mask = data["is_baffle"]
        idx = np.where(baffle_mask)[0]
        i = idx[np.argmin(np.abs(data["r"][baffle_mask] - r_target))]
        print(
            f"  r={ra:.1f}a from edge (R={data['r'][i]*1000:.3f} mm):"
            f"  |p|={data['p_mag'][i]:.6f}"
        )

    # Azimuthal ripple check: coefficient of variation vs radius
    print("\nRipple strength summary (baffle rings, CV = std/mean of |p|):")
    for ring in ring_results:
        ripple = "visible structure" if ring["p_cv"] > 0.05 else "weak/monotonic"
        print(
            f"  r={ring['r_from_edge']/A_M:.1f}a: CV={ring['p_cv']:.3f}, ptp/mean={ring['p_ptp']/ring['p_mean']:.3f} → {ripple}"
        )

    with open(RESULT_JSON, encoding="utf-8") as f:
        mesh_info = json.load(f)["mesh_info"]
    print(f"\nMesh grading (from result JSON): h_min={mesh_info['h_min_m']*1000:.3f} mm,"
          f" h_max={mesh_info['h_max_m']*1000:.3f} mm")
    print(f"  Expected ripple wavelength scale ≈ λ = {wavelength*1000:.3f} mm")
    print(f"  At far baffle (r~L={0.1*1000:.0f} mm): h_max/λ = {mesh_info['h_max_m']/wavelength:.3f}")

    return ring_results, wavelength


def plot_pressure_variant(triangles, p_mag_norm, xlim, ylim, out_path, norm, note=None):
    fig, ax = plt.subplots(figsize=(4.6, 8.0), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cmap = cm.get_cmap(CMAP)
    facecolors = cmap(norm(p_mag_norm))
    pc = PolyCollection(triangles, facecolors=facecolors, edgecolors="none", antialiaseds=True)
    ax.add_collection(pc)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _style_axes(ax)
    cax = fig.add_axes([0.88, 0.22, 0.035, 0.56])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8, length=2, width=0.6, pad=1)
    cb.set_label(r"$|p|/(\rho c U_0)$", fontsize=9, labelpad=4)
    if note:
        fig.text(0.04, 0.02, note, fontsize=7, color="#444444", ha="left", va="bottom")
    fig.subplots_adjust(left=0, right=0.84, bottom=0.04 if note else 0, top=1)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    grid, p_gf = load_field()
    data = top_surface_data(grid, p_gf)
    ring_results, wavelength = print_diagnostics(grid, data)

    triangles, elem_idx = _top_surface_triangles(grid)
    xlim, ylim = _axis_limits(triangles)
    p_mag_norm = data["p_mag"]

    pmax = float(np.max(p_mag_norm))
    baffle_p = p_mag_norm[data["is_baffle"]]

    # Option A: SymLogNorm
    linthresh = max(0.005, float(np.percentile(baffle_p, 5)))
    norm_a = SymLogNorm(linthresh=linthresh, linscale=0.4, vmin=0.0, vmax=pmax)
    out_a = os.path.join(OUT_DIR, "pressure_heatmap_ka5_symlog.png")
    plot_pressure_variant(
        triangles,
        p_mag_norm,
        xlim,
        ylim,
        out_a,
        norm_a,
        note=f"SymLogNorm (linthresh={linthresh:.3f})",
    )

    # Option B: capped vmax
    vmax_b = 0.15
    norm_b = Normalize(vmin=0.0, vmax=vmax_b)
    out_b = os.path.join(OUT_DIR, "pressure_heatmap_ka5_vmax015.png")
    plot_pressure_variant(
        triangles,
        p_mag_norm,
        xlim,
        ylim,
        out_b,
        norm_b,
        note=f"vmax={vmax_b:.2f} (piston saturates above scale)",
    )

    # Bonus: radial profile plot for report
    if ring_results:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
        for ring in ring_results:
            ax.plot(
                np.degrees(ring["theta_ring"]),
                ring["p_ring"],
                ".-",
                ms=3,
                label=f"r={ring['r_from_edge']/A_M:.1f}a",
            )
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel(r"$|p|/(\rho c U_0)$")
        ax.set_title("Baffle |p| vs azimuth at fixed radius (ka=5)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        prof_out = os.path.join(OUT_DIR, "pressure_ka5_azimuth_profiles.png")
        fig.savefig(prof_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {prof_out}")


if __name__ == "__main__":
    main()
