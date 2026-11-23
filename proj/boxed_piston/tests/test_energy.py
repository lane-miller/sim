"""Step 10a: discrete energy functional (fast)."""

from __future__ import annotations

import math

import numpy as np

from boxrad.config import Config
from boxrad.cutcell import Fractions, Grid, compute_fractions
from boxrad.cutkernel import step_cut
from boxrad.geom import mesh
from boxrad.geom import geometry as geom_mod
from boxrad.refstep import _div, closed_alpha, init_velocity
from boxrad.smallcell import apply_floor
from boxrad.validation.energy import discrete_energy

_F32 = np.float32
_LAM_REF = 0.99 / math.sqrt(3.0)


def _rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _ref_energy_geom():
    R = _rot_z(37.0) @ _rot_x(20.0)
    solid = mesh.box_mesh([6.0, 6.0, 6.0]).transformed(R, np.zeros(3))
    grid = Grid(
        x0=np.array([-7.0, -7.0, -7.0], dtype=np.float64),
        h=1.0,
        shape=(14, 14, 14),
        rigid_sides={},
    )
    g = geom_mod.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    fr = compute_fractions(g, grid)
    ax, ay, az = (a.copy() for a in fr.alpha)
    ax[0], ax[-1] = 0.0, 0.0
    ay[:, 0], ay[:, -1] = 0.0, 0.0
    az[:, :, 0], az[:, :, -1] = 0.0, 0.0
    fr2 = Fractions(
        alpha=(ax, ay, az),
        beta=fr.beta,
        state=fr.state,
        patch_cell=fr.patch_cell,
        patch_A=fr.patch_A,
        patch_centroid=fr.patch_centroid,
        patch_normal=fr.patch_normal,
        radiator_cell=fr.radiator_cell,
        radiator_a=fr.radiator_a,
        radiator_S=fr.radiator_S,
        stats=fr.stats,
        radiator_centroid=fr.radiator_centroid,
    )
    return fr2, closed_alpha(fr2.alpha)


def test_discrete_energy_refstep_conserved() -> None:
    fr, alpha_c = _ref_energy_geom()
    beta_tilde, _ = apply_floor(fr, _LAM_REF, h=1.0)
    lam = _LAM_REF
    ax, ay, az = alpha_c
    fluid = beta_tilde > 0
    inv_b = np.where(fluid, 1.0 / np.where(fluid, beta_tilde, 1.0), 0.0)
    rng = np.random.default_rng(0)
    p = np.zeros_like(fr.beta)
    p[fluid] = rng.standard_normal(int(np.sum(fluid)))
    ux, uy, uz = init_velocity(alpha_c)
    energies: list[float] = []
    for _ in range(2000):
        v_prev = (ux.copy(), uy.copy(), uz.copy())
        p_mid = p.copy()
        gx = np.zeros_like(ux)
        gx[1:-1] = p_mid[1:] - p_mid[:-1]
        gy = np.zeros_like(uy)
        gy[:, 1:-1] = p_mid[:, 1:] - p_mid[:, :-1]
        gz = np.zeros_like(uz)
        gz[:, :, 1:-1] = p_mid[:, :, 1:] - p_mid[:, :, :-1]
        nx_ = np.where(ax > 0, ux - lam * gx, 0.0)
        ny_ = np.where(ay > 0, uy - lam * gy, 0.0)
        nz_ = np.where(az > 0, uz - lam * gz, 0.0)
        energies.append(
            discrete_energy(
                p_mid,
                (nx_, ny_, nz_),
                v_prev,
                beta_tilde,
                alpha_c,
                1.0,
                1.0,
                1.0,
            )
        )
        ux, uy, uz = nx_, ny_, nz_
        p = p - lam * inv_b * _div((ux, uy, uz), alpha_c)
    e0 = energies[0]
    drift = float(np.max(np.abs(np.asarray(energies) - e0))) / max(abs(e0), 1e-30)
    assert drift <= 1e-12


def _closed_sphere_config() -> Config:
    sphere_d = 0.05
    sphere_r = sphere_d * 0.5
    gap_m = 0.02
    box_size = [0.02, 0.02, 0.01]
    half_x, _, half_z = (s * 0.5 for s in box_size)
    box_center = [
        sphere_r + gap_m + half_x,
        0.0,
        0.0,
    ]
    rad_position = [
        box_center[0],
        box_center[1],
        box_center[2] + half_z,
    ]
    raw = {
        "geometry": "parametric",
        "air": {"c": 343.0, "rho": 1.204},
        "box": None,
        "band": {"f_min": 200.0, "f_max": 2000.0},
        "numerics": {
            "h_override": 0.005,
            "cells_across_radiator": 6,
            "cfl_safety": 0.99,
            "gap_cells": None,
            "cpml_cells": 8,
            "cpml_alpha_factor": 1.0,
            "cpml_m": 3.0,
            "cpml_R": 1e-6,
            "cpml_kappa_max": 1.0,
            "cell_budget": 60_000_000,
            "pulse_db": -40.0,
            "pulse_f_factor": 1.2,
            "n_transits": 0.05,
            "min_edge_cells": 3,
        },
        "output": {"path": ""},
        "parts": [
            {
                "sphere": {"d": sphere_d},
                "position": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
            },
            {
                "box": {"size": box_size},
                "position": box_center,
                "normal": [0.0, 0.0, 1.0],
                "x_axis": [1.0, 0.0, 0.0],
            },
        ],
        "radiators": [
            {
                "name": "tiny",
                "rect": {"w": 0.006, "h": 0.006},
                "position": rad_position,
                "normal": [0.0, 0.0, 1.0],
                "x_axis": [1.0, 0.0, 0.0],
            }
        ],
    }
    return Config.from_dict(raw)


def _prepare_closed_domain_run(cfg: Config, *, cache=None):
    import dataclasses

    from boxrad.cpml import make_cpml_state
    from boxrad.geom.geometry import build_geometry
    from boxrad.geomderive import derive_geom, validate_geom
    from boxrad.geomop import build_run_operator
    from boxrad.geomrun import GeomRun
    from boxrad.grid import allocate_fields
    from boxrad.kernels import make_update_coeffs

    geometry = build_geometry(cfg)
    assert geometry is not None
    derived, gg = derive_geom(cfg, geometry)
    errors, warnings = validate_geom(cfg, derived, geometry)
    if errors:
        raise ValueError("; ".join(errors))
    derived = dataclasses.replace(derived, cpml_sides=(False,) * 6)
    fields = allocate_fields(gg)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, derived.h, derived.dt)
    cpml = make_cpml_state(gg, cfg, derived, derived.dt)
    op, info = build_run_operator(cfg, derived, gg, geometry, cache=cache)
    return GeomRun(
        cfg=cfg,
        derived=derived,
        gg=gg,
        geometry=geometry,
        fields=fields,
        coeffs=coeffs,
        cpml=cpml,
        op=op,
        info=info,
        sigma=derived.pulse_sigma,
        t0=derived.pulse_t0,
        dt=derived.dt,
        warnings=warnings,
    )


def _record_energy_production(run) -> float:
    """Leapfrog energy at step *n* via production ``step_cut`` (zero drive)."""
    fields = run.fields
    p_n = fields.p.copy()
    v_prev = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
    nf = len(run.op.radiators_fast) + len(run.op.radiators_patch)
    step_cut(fields, run.coeffs, run.op, run.cpml, [_F32(0.0)] * nf)
    rho = float(run.cfg.air.rho)
    c = float(run.cfg.air.c)
    h = float(run.derived.h)
    fr = run.info["fractions"]
    bt = run.info["beta_tilde"]
    v_now = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
    return discrete_energy(p_n, v_now, v_prev, bt, fr.alpha, rho, c, h)


def test_production_closed_domain_energy() -> None:
    cfg = _closed_sphere_config()
    run = _prepare_closed_domain_run(cfg)
    fluid = run.op.inv_beta.reshape(-1) > 0
    rng = np.random.default_rng(0)
    p = run.fields.p.reshape(-1)
    p[fluid] = rng.uniform(-1.0, 1.0, size=int(np.sum(fluid))).astype(_F32)
    run.fields.vx.fill(0.0)
    run.fields.vy.fill(0.0)
    run.fields.vz.fill(0.0)
    energies: list[float] = []
    for _ in range(2000):
        energies.append(_record_energy_production(run))
    e0 = energies[0]
    drift = float(np.max(np.abs(np.asarray(energies) - e0))) / max(abs(e0), 1e-30)
    assert drift <= 1e-5