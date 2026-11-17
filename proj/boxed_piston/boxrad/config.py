"""FDTD run configuration: schema, derived parameters, validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

F_MAX_CAP_HZ = 20_000.0
# C3-cpml: cpml_cells=16, R=1e-6, m=3, κ_max=1, α_factor=1 (α=π·f_min in derive); gap≥5 cells LF Z.
DEFAULT_CPML_CELLS = 16
DEFAULT_GAP_CELLS = 10  # default margin over C3 gap pass at 5 cells (until box checks C9)
DEFAULT_RAM_LIMIT_BYTES = 5 * 1024**3
SQRT3 = math.sqrt(3.0)

RadiatorShape = Literal["circle", "rect", "racetrack"]
GeometryMode = Literal["box", "baffle"]


@dataclass
class Air:
    """Lossless air: sound speed ``c`` [m/s] and density ``rho`` [kg/m³]."""

    c: float = 343.0
    rho: float = 1.204


@dataclass
class Box:
    """Rigid box [m]: front face ``W`` (x) × ``H`` (y), depth ``D`` (−z). Each side is rounded to h."""

    W: float
    H: float
    D: float


@dataclass
class Radiator:
    """Rigid flush radiator on the +z face.

    ``dims`` [m]: circle ``[diameter]``; rect ``[width, height]``; racetrack
    ``[width, height]`` with semicircular ends of diameter ``min(width, height)``
    on the longer axis. ``(x_c, y_c)`` [m] is the radiator center relative to the
    front-face center (not snapped to h).
    """

    shape: RadiatorShape
    dims: list[float]
    x_c: float = 0.0
    y_c: float = 0.0


@dataclass
class Band:
    """Frequency band [Hz]; ``f_max=None`` means ``min(20 kHz, f at ka = 10)``."""

    f_min: float = 20.0
    f_max: float | None = None


@dataclass
class Numerics:
    """Discretization, absorber, pulse, and run-length controls (see :func:`derive`)."""

    ppw: float = 20.0
    h_override: float | None = None  # fixed *h* (validation sweeps); skips ppw/radiator cap
    cells_across_radiator: int = 10
    cfl_safety: float = 0.99
    gap_cells: int | None = None
    cpml_cells: int = DEFAULT_CPML_CELLS
    cpml_alpha_factor: float = 1.0
    cpml_m: float = 3.0
    cpml_R: float = 1e-6
    cpml_kappa_max: float = 1.0
    cell_budget: int = 60_000_000
    pulse_db: float = -40.0
    pulse_f_factor: float = 1.2
    n_transits: float = 3.0
    min_edge_cells: int = 3


@dataclass
class Output:
    """Run output path plus optional pressure probe targets."""

    path: str = ""
    probes: list[list[float]] | None = None  # each [x, y, z] in meters
    probes_baffle_z_cells: list[int] | None = None  # baffle: z = (k + ½) h on radiator axis


@dataclass
class Config:
    """Complete run configuration; JSON sections map one-to-one to the fields.

    Unknown keys inside a section raise ``TypeError`` in :meth:`from_dict`;
    unknown top-level keys (e.g. ``"_doc"``) are ignored.
    """

    geometry: GeometryMode = "box"
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
            geometry=data.get("geometry", "box"),
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
    """Per-run parameters computed by :func:`derive` (SI units; stored in ``/meta/derived_json``)."""

    f_max: float
    lambda_min: float
    h: float
    dt: float
    realized_W: float
    realized_H: float
    realized_D: float
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
    ppw_realized: float
    cpml_sides: tuple[bool, bool, bool, bool, bool, bool]
    geometry: GeometryMode
    cell_budget_exceeded: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Derived:
        """Build from a stored ``derived_json`` blob, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        blob = {k: v for k, v in data.items() if k in known}
        if "cpml_sides" in blob:
            blob["cpml_sides"] = tuple(blob["cpml_sides"])
        return cls(**blob)


CPML_SIDES_ALL = (True, True, True, True, True, True)
CPML_SIDES_BAFFLE = (True, True, True, True, False, True)


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


def baffle_radiator_clip(radiator: Radiator, derived: Derived) -> float:
    """Largest distance (m) by which the radiator extends past the baffle-mode mask.

    In baffle mode the radiator mask covers ``[-x_c − W/2, -x_c + W/2]`` (and likewise
    in y), with ``W`` the radiator bounding box rounded to whole cells, while the
    radiator itself spans ``[−w/2, w/2]`` around the origin. Returns 0 if nothing is
    clipped.
    """
    hx, hy = _radiator_half_extents(radiator)
    clip_x = hx + abs(radiator.x_c) - 0.5 * derived.realized_W
    clip_y = hy + abs(radiator.y_c) - 0.5 * derived.realized_H
    return max(0.0, clip_x, clip_y)


def format_realized_box_dims(
    cfg: Config,
    derived: Derived,
    *,
    abs_tol: float = 1e-9,
) -> str:
    """Human-readable realized box size, noting config dims when snapped to *h*."""
    w, h, d = derived.realized_W, derived.realized_H, derived.realized_D
    if cfg.geometry != "box" or cfg.box is None:
        return f"box {w:.4g}×{h:.4g}×{d:.4g} m"
    req = cfg.box
    snapped = (
        abs(w - req.W) > abs_tol
        or abs(h - req.H) > abs_tol
        or abs(d - req.D) > abs_tol
    )
    if not snapped:
        return f"box {w:.4g}×{h:.4g}×{d:.4g} m"
    return (
        f"box {w:.4g}×{h:.4g}×{d:.4g} m "
        f"(requested {req.W:.4g}×{req.H:.4g}×{req.D:.4g}, snapped to h)"
    )


def _baffle_domain_counts(
    radiator: Radiator, h: float, gap_cells: int, cpml_cells: int
) -> tuple[int, int, int, int, int, int, float, float, float]:
    """Radiator-centered baffle: lateral = bbox + gap + CPML; +z = gap + CPML (no z-min CPML)."""
    hx, hy = _radiator_half_extents(radiator)
    nbx = max(1, round(2.0 * hx / h))
    nby = max(1, round(2.0 * hy / h))
    nbz = 0
    rw, rh = nbx * h, nby * h
    rd = 0.0
    nx = nbx + 2 * gap_cells + 2 * cpml_cells
    ny = nby + 2 * gap_cells + 2 * cpml_cells
    nz = gap_cells + cpml_cells
    return nbx, nby, nbz, nx, ny, nz, rw, rh, rd


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


def estimate_memory_bytes(nx: int, ny: int, nz: int, cpml_cells: int) -> int:
    """Analytic byte count matching :func:`boxrad.grid.allocate_fields` (for pre-flight).

    Main Yee fields use staggered shapes ``(nx, ny, nz)``, ``(nx+1, ny, nz)``, etc.,
    not four equal cubes. CPML ``ψ`` arrays are sized to face slabs only; corner cells
    appear in multiple slab arrays (intentional double-count vs a single union volume).
    """
    n_main = (
        nx * ny * nz
        + (nx + 1) * ny * nz
        + nx * (ny + 1) * nz
        + nx * ny * (nz + 1)
    )
    c = cpml_cells
    n_psi = (
        2 * c * ny * nz
        + 2 * nx * c * nz
        + 2 * nx * ny * c
        + 2 * (c + 1) * ny * nz
        + 2 * nx * (c + 1) * nz
        + 2 * nx * ny * (c + 1)
    )
    return 4 * (n_main + n_psi)


def _memory_bytes(nx: int, ny: int, nz: int, cpml_cells: int) -> float:
    return float(estimate_memory_bytes(nx, ny, nz, cpml_cells))


def derive(cfg: Config) -> Derived:
    """Compute derived FDTD parameters from *cfg* (SI units throughout).

    Rules (see brief §2–§3):

    * **f_max** — ``band.f_max`` if set, else ``min(20 kHz, f at ka=10)`` with
      ``f_ka10 = 10·c/(π·D)`` and ``D`` the smallest radiator dimension.
    * **λ_min** — ``c / f_max``.
    * **h** — ``numerics.h_override`` if set; else ``min(λ_min/ppw, D/cells_across_radiator)``.
      If the resulting domain exceeds ``cell_budget``, ``cell_budget_exceeded`` is set
      and :func:`validate` reports an error (there is no coarser fallback).
    * **Δt** — ``cfl_safety · h / (c·√3)`` (3-D CFL).
    * **Realized box** — each side rounded to an integer number of cells at ``h``.
    * **Radiator center in cells** — ``round(x_c/h)``, ``round(y_c/h)`` on the front face.
    * **Domain** — along each axis: ``cpml + gap + box_cells + gap + cpml`` (free-field:
      zero box cells; at least one cell per axis).
    * **CPML α** — ``cpml_alpha_factor · π · f_min``.
    * **Gap** — ``gap_cells`` (default 10) times ``h``.
    * **Memory** — staggered float32 ``p``/``v`` arrays plus CPML ``ψ`` face slabs
      (see :func:`estimate_memory_bytes`).
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
    if cfg.geometry == "baffle":
        count_fn = lambda hh: _baffle_domain_counts(  # noqa: E731
            cfg.radiator, hh, gap_cells, num.cpml_cells
        )
    else:
        count_fn = lambda hh: _domain_counts(cfg.box, hh, gap_cells, num.cpml_cells)  # noqa: E731

    if num.h_override is not None:
        h = float(num.h_override)
    else:
        h = min(lambda_min / num.ppw, D / num.cells_across_radiator)

    nbx, nby, nbz, nx, ny, nz, rw, rh, rd = count_fn(h)
    total_cells = nx * ny * nz
    cell_budget_exceeded = total_cells > num.cell_budget
    dt = num.cfl_safety * h / (c * SQRT3)

    radiator_ic = round(cfg.radiator.x_c / h)
    radiator_jc = round(cfg.radiator.y_c / h)
    gap_m = gap_cells * h
    cpml_alpha = num.cpml_alpha_factor * math.pi * cfg.band.f_min
    memory_bytes = _memory_bytes(nx, ny, nz, num.cpml_cells)

    att = math.pow(10.0, -num.pulse_db / 20.0)
    sigma = math.sqrt(2.0 * math.log(att)) / (2.0 * math.pi * num.pulse_f_factor * f_max)
    t0 = 6.0 * sigma

    if cfg.geometry == "baffle":
        z_extent = float(gap_cells + num.cpml_cells) * h
        max_dim = max(rw, rh, z_extent)
    elif cfg.box is None:
        max_dim = 0.0
    else:
        max_dim = max(rw, rh, rd)
    run_time_s = t0 + 6.0 * sigma + num.n_transits * (max_dim + 2.0 * gap_m) / c
    n_steps = max(1, math.ceil(run_time_s / dt))

    ppw_realized = lambda_min / h
    if cfg.geometry == "baffle":
        cpml_sides = CPML_SIDES_BAFFLE
    else:
        cpml_sides = CPML_SIDES_ALL

    return Derived(
        f_max=f_max,
        lambda_min=lambda_min,
        h=h,
        dt=dt,
        realized_W=rw,
        realized_H=rh,
        realized_D=rd,
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
        ppw_realized=ppw_realized,
        cpml_sides=cpml_sides,
        geometry=cfg.geometry,
        cell_budget_exceeded=cell_budget_exceeded,
    )


def validate(
    cfg: Config,
    derived: Derived,
    *,
    ram_limit_bytes: float = DEFAULT_RAM_LIMIT_BYTES,
) -> tuple[list[str], list[str]]:
    """Pre-flight checks of *cfg* against *derived*; no printing.

    Args:
        cfg: Run configuration.
        derived: Output of :func:`derive` for *cfg*.
        ram_limit_bytes: Field-memory estimate above this is an error.

    Returns:
        ``(errors, warnings)``: lists of human-readable messages. Any error
        stops :func:`boxrad.run`; warnings are printed and the run continues.
    """
    errors: list[str] = []
    warnings: list[str] = []

    c = cfg.air.c
    num = cfg.numerics

    if cfg.geometry == "box" and cfg.box is None:
        errors.append("geometry=box requires box dimensions in config")

    if cfg.geometry == "box" and cfg.box is not None:
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

    if derived.cell_budget_exceeded:
        errors.append(
            f"domain {derived.nx}×{derived.ny}×{derived.nz} = {derived.total_cells:,} cells "
            f"at h={derived.h * 1e3:.4g} mm exceeds numerics.cell_budget="
            f"{num.cell_budget:,}; raise h (lower numerics.cells_across_radiator or "
            f"numerics.ppw, or set numerics.h_override), lower band.f_max, or raise "
            f"numerics.cell_budget if RAM allows"
        )

    if cfg.geometry == "baffle":
        clip = baffle_radiator_clip(cfg.radiator, derived)
        offset = (cfg.radiator.x_c, cfg.radiator.y_c) != (0.0, 0.0)
        cause = (
            f"radiator offset (x_c={cfg.radiator.x_c:g}, y_c={cfg.radiator.y_c:g}) m"
            if offset
            else "rounding of the radiator bounding box to whole cells"
        )
        if clip > derived.h * (1.0 + 1e-9):
            errors.append(
                f"baffle mode: {cause} clips the radiator outline by "
                f"{clip / derived.h:.2f} cells (> 1 cell); use x_c = y_c = 0 in baffle mode"
            )
        elif clip > 1e-9 * derived.h:
            warnings.append(
                f"baffle mode: {cause} clips the radiator outline by "
                f"{clip / derived.h:.2f} cells (S_grid slightly below S)"
            )

    _ppw_dispersion_ref = 20.0  # brief §2 default; warning is vs realized spacing, not config ppw
    if derived.ppw_realized < _ppw_dispersion_ref - 1e-9:
        warnings.append(
            f"realized {derived.ppw_realized:.1f} points per wavelength < "
            f"default {_ppw_dispersion_ref:g}"
        )

    if cfg.geometry == "baffle" and cfg.box is not None:
        warnings.append("box dimensions ignored in baffle geometry mode")

    if cfg.geometry == "box" and cfg.box is not None:
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

    if num.h_override is not None:
        warnings.append(
            f"h_override={num.h_override:.6g} m active (intended for validation sweeps)"
        )

    return errors, warnings
