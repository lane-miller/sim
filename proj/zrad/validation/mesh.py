"""Generate graded Gmsh surface meshes for the baffled-piston BEM validation."""

import argparse
import os

import gmsh
import numpy as np

# ── Physical parameters ───────────────────────────────────────────────────────
A = 0.01            # piston radius [m]
C = 343.0           # speed of sound [m/s]
RHO = 1.21          # air density [kg/m^3]
U0 = 1.0            # piston normal velocity [m/s]
THICKNESS = A / 30  # baffle thickness (a/30 — thin relative to a)

KA_VALUES = [0.05, 0.1, 0.5, 1.0, 5.0]
KA_VALUES_DENSE = list(np.geomspace(0.05, 5.0, 12))
KD_MARGIN_DEFAULT = 10.0

PISTON_TAG = 1
BAFFLE_TAG = 2

# Piston-disk sizing: independent of wavelength / outer baffle grading (Change 2).
PISTON_CIRC_ELEMENTS = 32  # rim circumference target (~24–32)
H_PISTON_LOCAL = A / 15.0  # interior cap — finer than legacy min(h_outer, a/12)

MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes")
DEFAULT_RESULT_SUFFIX = "_ff_pistref"


def baffle_half_width(ka, kd=None):
    """Half-width from baffle center to outer edge [m].

    Margin from piston rim to baffle edge is d = kd/k:
        L_baffle = a + d
    Square baffle side = 2 * L_baffle.
    """
    if kd is None:
        kd = KD_MARGIN_DEFAULT
    k = ka / A
    d = kd / k
    return A + d


def frequency_hz(ka):
    """Frequency [Hz] for a given ka."""
    k = ka / A
    return k * C / (2.0 * np.pi)


def mesh_path(ka, kd=None):
    tag = f"{ka:g}".replace(".", "p")
    name = f"mesh_ka{tag}"
    if kd is not None and kd != KD_MARGIN_DEFAULT:
        kd_tag = f"{kd:g}".replace(".", "p")
        name = f"{name}_kd{kd_tag}"
    return os.path.join(MESH_DIR, f"{name}.msh")


def _closed_baffle_surface_area(L):
    """Total exterior surface area of the thin square baffle solid [m^2]."""
    top_annulus = np.pi * (L ** 2 - A ** 2)
    bottom = (2.0 * L) ** 2
    sides = 4.0 * (2.0 * L) * THICKNESS
    return top_annulus + bottom + sides


# Empirical calibration: kd=10 mesh at ka=0.05 (2172 total, 94 piston).
_ESTIMATE_REF_KA = 0.05
_ESTIMATE_REF_KD = KD_MARGIN_DEFAULT
_ESTIMATE_REF_TOTAL = 2172
_ESTIMATE_REF_PISTON = 94


def estimate_mesh_stats(ka, kd=KD_MARGIN_DEFAULT):
    """Estimate element/DOF counts and runtime before mesh generation."""
    L = baffle_half_width(ka, kd=kd)
    L_ref = baffle_half_width(_ESTIMATE_REF_KA, kd=_ESTIMATE_REF_KD)
    area_ref = _closed_baffle_surface_area(L_ref)
    area = _closed_baffle_surface_area(L)
    n_piston = _ESTIMATE_REF_PISTON
    n_baffle_ref = _ESTIMATE_REF_TOTAL - _ESTIMATE_REF_PISTON
    n_baffle = int(round(n_baffle_ref * area / area_ref))
    n_total = n_piston + n_baffle
    n_dofs = int(round(n_total * 0.5))

    # kd=10 ka=0.05 direct-LU baseline from result_ka0p05_ff_pistref.json.
    ref_dofs = 1088
    ref_asm_s = 19.0
    ref_lu_s = 0.19
    ref_ff_s = 2.4
    dof_ratio = n_dofs / ref_dofs
    est_asm_s = ref_asm_s * dof_ratio ** 2
    est_lu_s = ref_lu_s * dof_ratio ** 3
    est_ff_s = ref_ff_s * dof_ratio ** 1.5
    est_total_s = est_asm_s + est_lu_s + est_ff_s

    return {
        "ka": ka,
        "kd": kd,
        "L_baffle_m": L,
        "baffle_side_m": 2.0 * L,
        "n_elements_est": n_total,
        "n_piston_elements_est": n_piston,
        "n_baffle_elements_est": n_baffle,
        "n_dofs_est": n_dofs,
        "area_ratio_vs_kd10": area / area_ref,
        "runtime_est_direct_s": est_total_s,
        "runtime_est_breakdown_s": {
            "assembly": est_asm_s,
            "lu": est_lu_s,
            "far_field": est_ff_s,
        },
    }


def print_mesh_estimate(stats):
    """Print a pre-generation sanity-check summary."""
    print(
        f"ka={stats['ka']:g}  kd={stats['kd']:g}  "
        f"L_baffle={stats['L_baffle_m']:.3f} m  side={stats['baffle_side_m']:.2f} m"
    )
    print(
        f"  Elements (est): {stats['n_elements_est']:,}"
        f"  (piston ~{stats['n_piston_elements_est']},"
        f" baffle ~{stats['n_baffle_elements_est']:,})"
    )
    print(f"  DOFs (est)    : {stats['n_dofs_est']:,}")
    print(f"  Area vs kd=10 : {stats['area_ratio_vs_kd10']:.1f}x")
    br = stats["runtime_est_breakdown_s"]
    print(
        f"  Runtime (est, direct LU): ~{stats['runtime_est_direct_s'] / 60:.0f} min"
        f"  (asm ~{br['assembly'] / 60:.0f} min,"
        f" LU ~{br['lu'] / 60:.0f} min,"
        f" ff ~{br['far_field']:.0f} s)"
    )


def _surface_max_radius(surf_tag, n_curve_samples=32):
    """Maximum cylindrical radius r=sqrt(x^2+y^2) over a surface's boundary."""
    max_r = 0.0

    def _update(x, y):
        nonlocal max_r
        max_r = max(max_r, np.hypot(x, y))

    boundaries = gmsh.model.getBoundary(
        [(2, surf_tag)], combined=False, oriented=False, recursive=False
    )
    for bdim, btag in boundaries:
        if bdim == 0:
            xyz = gmsh.model.getValue(0, btag, [])
            _update(xyz[0], xyz[1])
        elif bdim == 1:
            umin, umax = gmsh.model.getParametrizationBounds(1, btag)
            umin = float(np.asarray(umin).flat[0])
            umax = float(np.asarray(umax).flat[0])
            for i in range(n_curve_samples + 1):
                u = umin + (umax - umin) * i / n_curve_samples
                xyz = gmsh.model.getValue(1, btag, [u])
                _update(xyz[0], xyz[1])

    return max_r


def _classify_surfaces(L, t, tol=None):
    """Assign each boundary surface to piston (top disk) or baffle.

    Uses max radial extent of boundary points (not center-of-mass or area):
      piston: max(r) <= a + tol
      baffle: max(r) >  a + tol
    """
    if tol is None:
        tol = max(A * 1e-6, L * 1e-6, 1e-9)

    piston = []
    baffle = []

    for dim, tag in gmsh.model.getEntities(2):
        r_max = _surface_max_radius(tag)
        if r_max <= A + tol:
            piston.append(tag)
        else:
            baffle.append(tag)

    return piston, baffle


def _compute_h_piston():
    """Piston target element size — decoupled from h_outer / wavelength grading."""
    h_circ = 2.0 * np.pi * A / PISTON_CIRC_ELEMENTS
    return min(H_PISTON_LOCAL, h_circ)


def _piston_rim_curves(piston_surfs):
    """1D boundary curves of the piston disk (shared with the top annular baffle)."""
    curves = []
    for surf_tag in piston_surfs:
        for bdim, btag in gmsh.model.getBoundary(
            [(2, surf_tag)], combined=False, oriented=False, recursive=False
        ):
            if bdim == 1:
                curves.append(btag)
    return curves


def _set_piston_boundary_mesh(piston_surfs, h_piston):
    """Force rim 1D resolution: transfinite seed + setSize backup.

    Background mesh fields override setSize alone; transfiniteCurve pins the
    shared piston/baffle-rim edge before 2D meshing.
    """
    n_circ = max(PISTON_CIRC_ELEMENTS, int(np.ceil(2.0 * np.pi * A / h_piston)))
    curves = _piston_rim_curves(piston_surfs)
    for ctag in curves:
        gmsh.model.mesh.setSize([(1, ctag)], h_piston)
        gmsh.model.mesh.setTransfiniteCurve(ctag, n_circ)
    return curves, n_circ


def _setup_mesh_fields(piston_surfs, baffle_surfs, h_piston, h_outer, piston_curves):
    """Background sizing: graded baffle only + constant piston disk and rim."""
    t = THICKNESS

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Baffle grading only — exclude piston disk from Threshold Restrict so the
    # shared rim curve is not graded with h_outer from the annulus interior.
    gmsh.model.mesh.field.add("Ball", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_piston)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_outer)
    gmsh.model.mesh.field.setNumber(1, "Radius", A)
    gmsh.model.mesh.field.setNumber(1, "XCenter", 0.0)
    gmsh.model.mesh.field.setNumber(1, "YCenter", 0.0)
    gmsh.model.mesh.field.setNumber(1, "ZCenter", t / 2)

    gmsh.model.mesh.field.add("Restrict", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumbers(2, "FacesList", baffle_surfs)

    gmsh.model.mesh.field.add("Threshold", 3)
    gmsh.model.mesh.field.setNumber(3, "InField", 2)
    gmsh.model.mesh.field.setNumber(3, "SizeMin", h_piston)
    gmsh.model.mesh.field.setNumber(3, "SizeMax", h_outer)
    gmsh.model.mesh.field.setNumber(3, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(3, "DistMax", A)

    # Constant sizing on piston disk faces.
    gmsh.model.mesh.field.add("Constant", 4)
    gmsh.model.mesh.field.setNumber(4, "VIn", h_piston)
    gmsh.model.mesh.field.add("Restrict", 5)
    gmsh.model.mesh.field.setNumber(5, "InField", 4)
    gmsh.model.mesh.field.setNumbers(5, "FacesList", piston_surfs)

    # Constant sizing on piston rim 1D edges (background field overrides setSize).
    gmsh.model.mesh.field.add("Constant", 7)
    gmsh.model.mesh.field.setNumber(7, "VIn", h_piston)
    gmsh.model.mesh.field.add("Restrict", 8)
    gmsh.model.mesh.field.setNumber(8, "InField", 7)
    gmsh.model.mesh.field.setNumbers(8, "EdgesList", piston_curves)

    gmsh.model.mesh.field.add("Min", 6)
    gmsh.model.mesh.field.setNumbers(6, "FieldsList", [3, 5, 8])
    gmsh.model.mesh.field.setAsBackgroundMesh(6)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_piston)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_outer)


def generate_mesh(ka, kd=None, gui=False):
    """Build one closed-surface mesh for the given ka."""
    k = ka / A
    f = frequency_hz(ka)
    L = baffle_half_width(ka, kd=kd)
    wavelength = 2.0 * np.pi / k
    h_outer = wavelength / 6.0
    h_piston = _compute_h_piston()

    gmsh.initialize()
    gmsh.model.add(f"baffle_ka{ka:g}")

    occ = gmsh.model.occ
    t = THICKNESS

    # Thin square plate (closed surface for exterior BEM).
    box = occ.addBox(-L, -L, -t / 2, 2 * L, 2 * L, t)
    occ.synchronize()

    # Imprint a 2D disk on the top face only — splits that face into a piston
    # patch (r <= a) and the surrounding baffle.  Does NOT cut a hole through
    # the plate; bottom face and side walls stay intact.
    disk = occ.addDisk(0, 0, t / 2, A, A, zAxis=[0, 0, 1])
    occ.fragment([(3, box)], [(2, disk)])
    occ.synchronize()

    piston_surfs, baffle_surfs = _classify_surfaces(L, t)

    if not piston_surfs:
        raise RuntimeError(f"No piston surfaces found for ka={ka}")

    gmsh.model.addPhysicalGroup(2, piston_surfs, tag=PISTON_TAG, name="piston")
    gmsh.model.addPhysicalGroup(2, baffle_surfs, tag=BAFFLE_TAG, name="baffle")

    piston_curves, n_circ = _set_piston_boundary_mesh(piston_surfs, h_piston)
    _setup_mesh_fields(piston_surfs, baffle_surfs, h_piston, h_outer, piston_curves)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)

    n_piston_tri = 0
    for st in piston_surfs:
        _, tags, _ = gmsh.model.mesh.getElements(2, st)
        n_piston_tri += sum(len(t) for t in tags)

    out_path = mesh_path(ka, kd=kd)
    os.makedirs(MESH_DIR, exist_ok=True)
    gmsh.write(out_path)

    # Element count and characteristic length sanity check.
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    n_tri = sum(len(tags) for tags in elem_tags)

    node_tags, coord_flat, _ = gmsh.model.mesh.getNodes()
    node_coords = {
        tag: np.array(coord_flat[3 * i : 3 * i + 3])
        for i, tag in enumerate(node_tags)
    }

    sizes = []
    for tags in elem_tags:
        for et in tags:
            _, elem_node_tags, _, _ = gmsh.model.mesh.getElement(et)
            pts = [node_coords[n] for n in elem_node_tags]
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                sizes.append(np.linalg.norm(pts[i] - pts[j]))

    h_min = min(sizes) if sizes else float("nan")
    h_max = max(sizes) if sizes else float("nan")

    kd_label = KD_MARGIN_DEFAULT if kd is None else kd
    print(f"ka={ka:g}  kd={kd_label:g}  f={f:.1f} Hz  L_baffle={L:.3f} m")
    print(
        f"  Target h: piston={h_piston:.6f} m  outer={h_outer:.6f} m  (λ/6)"
        f"  [circ={n_circ}, local≤{H_PISTON_LOCAL:.6f}]"
    )
    print(f"  Elements : {n_tri}  (piston: {n_piston_tri})")
    print(f"  h_min    : {h_min:.6f} m")
    print(f"  h_max    : {h_max:.6f} m")
    print(f"  Piston surfaces: {len(piston_surfs)}  Baffle surfaces: {len(baffle_surfs)}")
    print(f"  Written  : {out_path}")

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate baffled-piston BEM validation meshes.")
    parser.add_argument(
        "--ka",
        type=float,
        nargs="*",
        default=None,
        help="ka values to mesh (default: all five validation points)",
    )
    parser.add_argument("--gui", action="store_true", help="Open Gmsh GUI after meshing.")
    parser.add_argument(
        "--kd-override",
        type=float,
        default=None,
        metavar="KD",
        help=(
            f"Override rim-to-edge margin kd (default {KD_MARGIN_DEFAULT:g}: "
            "L_baffle = a + kd/k). Non-default kd writes mesh_ka*_kd*.msh."
        ),
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print element/DOF/runtime estimate and exit (no Gmsh run).",
    )
    args = parser.parse_args()

    ka_list = args.ka if args.ka is not None else KA_VALUES
    kd = args.kd_override

    if args.estimate_only:
        for ka in ka_list:
            print_mesh_estimate(estimate_mesh_stats(ka, kd=kd))
        return

    for ka in ka_list:
        generate_mesh(ka, kd=kd, gui=args.gui)


if __name__ == "__main__":
    main()
