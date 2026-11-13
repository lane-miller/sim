"""FDTD run configuration: schema, derived parameters, validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

F_MAX_CAP_HZ = 20_000.0
DEFAULT_CPML_CELLS = 16
DEFAULT_GAP_CELLS = 2
DEFAULT_RAM_LIMIT_BYTES = 5 * 1024**3
SQRT3 = math.sqrt(3.0)

RadiatorShape = Literal["circle", "rect", "racetrack"]


@dataclass
class Air:
    c: float = 343.0
    rho: float = 1.204


@dataclass
class Box:
    W: float
    H: float
    D: float


@dataclass
class Radiator:
    shape: RadiatorShape
    dims: list[float]
    x_c: float = 0.0
    y_c: float = 0.0


@dataclass
class Band:
    f_min: float = 20.0
    f_max: float | None = None


@dataclass
class Numerics:
    ppw: float = 20.0
    cells_across_radiator: int = 10
    cfl_safety: float = 0.99
    gap_cells: int | None = None
    cpml_cells: int = DEFAULT_CPML_CELLS
    cpml_alpha_factor: float = 1.0
    cell_budget: int = 60_000_000
    pulse_db: float = -40.0
    pulse_f_factor: float = 1.2
    n_transits: float = 3.0
    min_edge_cells: int = 3


@dataclass
class Output:
    path: str = ""


@dataclass
class Config:
    air: Air = field(default_factory=Air)
    box: Box | None = None
    radiator: Radiator = field(
        default_factory=lambda: Radiator("circle", [0.05])
    )
    band: Band = field(default_factory=Band)
    numerics: Numerics = field(default_factory=Numerics)
    output: Output = field(default_factory=Output)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        box = None if data.get("box") is None else Box(**data["box"])
        return cls(
            air=Air(**data.get("air", {})),
            box=box,
            radiator=Radiator(**data["radiator"]),
            band=Band(**data.get("band", {})),
            numerics=Numerics(**data.get("numerics", {})),
            output=Output(**data.get("output", {})),
        )

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> Config:
        return cls.from_dict(json.loads(text))


@dataclass
class Derived:
    f_max: float
    lambda_min: float
    h: float
    dt: float
    zrad_fallback: bool
    realised_W: float
    realised_H: float
    realised_D: float
    radiator_ic: int
    radiator_jc: int
    box_nx: int
    box_ny: int
    box_nz: int
    gap_cells: int
    gap_m: float
    cpml_cells: int
    cpml_alpha: float
    nx: int
    ny: int
    nz: int
    total_cells: int
    memory_bytes: float
    pulse_sigma: float
    pulse_t0: float
    run_time_s: float
    n_steps: int
    ppw_realised: float


def _radiator_D(radiator: Radiator) -> float:
    """Smallest characteristic dimension (m), per brief §2."""
    if not radiator.dims:
        raise ValueError("radiator.dims must be non-empty")
    if radiator.shape == "circle":
        return radiator.dims[0]
    if radiator.shape == "rect":
        return min(radiator.dims[0], radiator.dims[1])
    if radiator.shape == "racetrack":
        return min(radiator.dims[0], radiator.dims[1])
    raise ValueError(f"unknown radiator shape: {radiator.shape!r}")


def _f_at_ka10(c: float, radiator: Radiator) -> float:
    """Frequency where ka = 10 using D and a = D/2 for the wave number."""
    D = _radiator_D(radiator)
    return 10.0 * c / (math.pi * D)


def _radiator_half_extents(radiator: Radiator) -> tuple[float, float]:
    if radiator.shape == "circle":
        r = radiator.dims[0] / 2.0
        return r, r
    if radiator.shape == "rect":
        return radiator.dims[0] / 2.0, radiator.dims[1] / 2.0
    if radiator.shape == "racetrack":
        return radiator.dims[0] / 2.0, radiator.dims[1] / 2.0
    raise ValueError(f"unknown radiator shape: {radiator.shape!r}")


def _domain_counts(
    box: Box | None, h: float, gap_cells: int, cpml_cells: int
) -> tuple[int, int, int, int, int, int, float, float, float]:
    if box is None:
        nbx = nby = nbz = 0
        rw = rh = rd = 0.0
    else:
        nbx = max(1, round(box.W / h))
        nby = max(1, round(box.H / h))
        nbz = max(1, round(box.D / h))
        rw, rh, rd = nbx * h, nby * h, nbz * h
    nx = nbx + 2 * gap_cells + 2 * cpml_cells
    ny = nby + 2 * gap_cells + 2 * cpml_cells
    nz = nbz + 2 * gap_cells + 2 * cpml_cells
    if box is None:
        nx = ny = nz = max(nx, 1)
    return nbx, nby, nbz, nx, ny, nz, rw, rh, rd


def _memory_bytes(nx: int, ny: int, nz: int, cpml_cells: int) -> float:
    """p + three velocity components (float32); ψ only in CPML slabs (6× float32)."""
    n_total = nx * ny * nz
    main = 4.0 * 4.0 * n_total
    ix0, iy0, iz0 = cpml_cells, cpml_cells, cpml_cells
    ix1, iy1, iz1 = nx - cpml_cells, ny - cpml_cells, nz - cpml_cells
    n_inner = max(0, ix1 - ix0) * max(0, iy1 - iy0) * max(0, iz1 - iz0)
    n_cpml = n_total - n_inner
    psi = 6.0 * 4.0 * n_cpml
    return main + psi


def derive(cfg: Config) -> Derived:
    """Compute derived FDTD parameters from *cfg* (SI units throughout).

    Rules (see brief §2–§3):

    * **f_max** — ``band.f_max`` if set, else ``min(20 kHz, f at ka=10)`` with
      ``f_ka10 = 10·c/(π·D)`` and ``D`` the smallest radiator dimension.
    * **λ_min** — ``c / f_max``.
    * **h** — ``min(λ_min/ppw, D/cells_across_radiator)``. If the resulting
      domain cell count exceeds ``cell_budget``, keep ``h = λ_min/ppw`` and set
      ``zrad_fallback`` (analytic Z_rad path in later stages).
    * **Δt** — ``cfl_safety · h / (c·√3)`` (3-D CFL).
    * **Realised box** — each side rounded to an integer number of cells at ``h``.
    * **Radiator centre in cells** — ``round(x_c/h)``, ``round(y_c/h)`` on the front face.
    * **Domain** — along each axis: ``cpml + gap + box_cells + gap + cpml`` (free-field:
      zero box cells; at least one cell per axis).
    * **CPML α** — ``cpml_alpha_factor · π · f_min``.
    * **Gap** — ``gap_cells`` (default 2) times ``h``.
    * **Memory** — four float32 field arrays over the full grid, plus six float32 ψ
      arrays counted only in CPML cells (slab union approximation).
    * **Gaussian pulse** — ``σ`` such that ``|V(f)|`` is ``pulse_db`` at
      ``pulse_f_factor·f_max``:
      ``σ = sqrt(2·ln(10^(-pulse_db/20))) / (2π·pulse_f_factor·f_max)``;
      delay ``t0 = 6σ``.
    * **Run length** — ``t0 + 6σ + n_transits·(max box dimension + 2·gap)/c``;
      ``n_steps = ceil(run_time / Δt)``.
    """
    c = cfg.air.c
    num = cfg.numerics
    gap_cells = DEFAULT_GAP_CELLS if num.gap_cells is None else num.gap_cells

    f_ka10 = _f_at_ka10(c, cfg.radiator)
    f_max = cfg.band.f_max if cfg.band.f_max is not None else min(F_MAX_CAP_HZ, f_ka10)
    lambda_min = c / f_max

    D = _radiator_D(cfg.radiator)
    h_from_ppw = lambda_min / num.ppw
    h_from_radiator = D / num.cells_across_radiator
    h_try = min(h_from_ppw, h_from_radiator)

    nbx, nby, nbz, nx, ny, nz, _, _, _ = _domain_counts(
        cfg.box, h_try, gap_cells, num.cpml_cells
    )
    zrad_fallback = False
    if nx * ny * nz > num.cell_budget:
        h = h_from_ppw
        zrad_fallback = True
    else:
        h = h_try

    nbx, nby, nbz, nx, ny, nz, rw, rh, rd = _domain_counts(
        cfg.box, h, gap_cells, num.cpml_cells
    )
    total_cells = nx * ny * nz
    dt = num.cfl_safety * h / (c * SQRT3)

    radiator_ic = round(cfg.radiator.x_c / h)
    radiator_jc = round(cfg.radiator.y_c / h)
    gap_m = gap_cells * h
    cpml_alpha = num.cpml_alpha_factor * math.pi * cfg.band.f_min
    memory_bytes = _memory_bytes(nx, ny, nz, num.cpml_cells)

    att = math.pow(10.0, -num.pulse_db / 20.0)
    sigma = math.sqrt(2.0 * math.log(att)) / (2.0 * math.pi * num.pulse_f_factor * f_max)
    t0 = 6.0 * sigma

    if cfg.box is None:
        max_dim = 0.0
    else:
        max_dim = max(rw, rh, rd)
    run_time_s = t0 + 6.0 * sigma + num.n_transits * (max_dim + 2.0 * gap_m) / c
    n_steps = max(1, math.ceil(run_time_s / dt))

    ppw_realised = lambda_min / h

    return Derived(
        f_max=f_max,
        lambda_min=lambda_min,
        h=h,
        dt=dt,
        zrad_fallback=zrad_fallback,
        realised_W=rw,
        realised_H=rh,
        realised_D=rd,
        radiator_ic=radiator_ic,
        radiator_jc=radiator_jc,
        box_nx=nbx,
        box_ny=nby,
        box_nz=nbz,
        gap_cells=gap_cells,
        gap_m=gap_m,
        cpml_cells=num.cpml_cells,
        cpml_alpha=cpml_alpha,
        nx=nx,
        ny=ny,
        nz=nz,
        total_cells=total_cells,
        memory_bytes=memory_bytes,
        pulse_sigma=sigma,
        pulse_t0=t0,
        run_time_s=run_time_s,
        n_steps=n_steps,
        ppw_realised=ppw_realised,
    )


def validate(
    cfg: Config,
    derived: Derived,
    *,
    ram_limit_bytes: float = DEFAULT_RAM_LIMIT_BYTES,
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` lists; no printing."""
    errors: list[str] = []
    warnings: list[str] = []

    c = cfg.air.c
    num = cfg.numerics

    if cfg.box is not None:
        hx, hy = _radiator_half_extents(cfg.radiator)
        xc, yc = cfg.radiator.x_c, cfg.radiator.y_c
        half_w = cfg.box.W / 2.0
        half_h = cfg.box.H / 2.0
        clear_x_neg = (xc - hx) + half_w
        clear_x_pos = half_w - (xc + hx)
        clear_y_neg = (yc - hy) + half_h
        clear_y_pos = half_h - (yc + hy)
        min_clear = min(clear_x_neg, clear_x_pos, clear_y_neg, clear_y_pos)
        if min_clear < 0.0:
            errors.append("radiator does not fit on the box front face")
        elif min_clear < num.min_edge_cells * derived.h:
            errors.append(
                f"radiator edge clearance {min_clear / derived.h:.2f} cells "
                f"< min_edge_cells={num.min_edge_cells}"
            )

    if derived.memory_bytes > ram_limit_bytes:
        errors.append(
            f"estimated memory {derived.memory_bytes / 1e9:.2f} GB exceeds "
            f"limit {ram_limit_bytes / 1e9:.2f} GB"
        )

    D = _radiator_D(cfg.radiator)
    ka = math.pi * derived.f_max * D / c
    if ka > 10.0:
        warnings.append(f"ka={ka:.2f} > 10 at f_max={derived.f_max:.1f} Hz")

    if derived.zrad_fallback:
        warnings.append("cell-budget fallback active (zrad_source will be analytic)")

    if derived.ppw_realised < num.ppw - 1e-9:
        warnings.append(
            f"realised {derived.ppw_realised:.1f} points per wavelength < "
            f"target ppw={num.ppw}"
        )

    if cfg.box is not None:
        dims_mm = (cfg.box.W * 1e3, cfg.box.H * 1e3, cfg.box.D * 1e3)
        for name, val in zip(("W", "H", "D"), dims_mm):
            if val < 10.0 or val > 600.0:
                warnings.append(f"box {name}={val:.1f} mm outside 10–600 mm range")
        smin, smax = min(dims_mm), max(dims_mm)
        if smin > 0 and smax / smin > 4.0:
            warnings.append(f"box aspect ratio {smax / smin:.2f} > 4")

    gap_cells = derived.gap_cells
    if gap_cells < DEFAULT_GAP_CELLS:
        warnings.append(
            f"gap_cells={gap_cells} below default {DEFAULT_GAP_CELLS}"
        )
    if num.cpml_cells < DEFAULT_CPML_CELLS:
        warnings.append(
            f"cpml_cells={num.cpml_cells} below default {DEFAULT_CPML_CELLS}"
        )

    return errors, warnings
