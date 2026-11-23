"""Geometry schema parsing (v0.3 step 2): frames, primitives, radiators; no mesh build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from boxrad.config import Radiator
from boxrad.frames import Pose

GeometryTag = Literal["box", "baffle", "parametric", "step"]
PrimitiveKind = Literal["box", "cylinder", "sphere"]
ShapeKind = Literal["rect", "circle", "racetrack", "ellipse"]
PlaneAxis = Literal["x", "y", "z"]
PlaneSide = Literal["-", "+"]
MotionKind = Literal["axial"]

_ALLOWED_GEOMETRY_TAGS = ("box", "baffle", "parametric", "step")
_BAFFLE_EXTRAS_MSG = (
    "baffle mode already models an infinite baffle at z=0; extras/planes not supported"
)
_PRIMITIVE_KEYS = frozenset({"box", "cylinder", "sphere"})
_SHAPE_KEYS = frozenset({"rect", "circle", "racetrack", "ellipse"})
_PLANE_AXES = frozenset({"x", "y", "z"})
_PLANE_SIDES = frozenset({"-", "+"})


@dataclass(frozen=True)
class PlaneSpec:
    axis: PlaneAxis
    side: PlaneSide
    offset: float


@dataclass(frozen=True)
class EnvironmentSpec:
    planes: tuple[PlaneSpec, ...]


@dataclass(frozen=True)
class PrimitiveSpec:
    kind: PrimitiveKind
    params: dict[str, float]
    pose: Pose


@dataclass(frozen=True)
class StepPartSpec:
    file: str
    enclosure_part: str = "enclosure"
    placement: Pose | None = None


@dataclass(frozen=True)
class ShapeSpec:
    kind: ShapeKind
    params: dict[str, float]
    hole: ShapeSpec | None = None


@dataclass(frozen=True)
class RadiatorSpec:
    name: str
    shape: ShapeSpec | None
    part: str | None
    axis: tuple[float, float, float] | None
    pose: Pose | None
    motion: MotionKind = "axial"


@dataclass(frozen=True)
class GeometrySpec:
    tag: GeometryTag
    parts: tuple[PrimitiveSpec | StepPartSpec, ...]
    radiators: tuple[RadiatorSpec, ...]
    environment: EnvironmentSpec


def check_geometry_v03_guards(cfg: Any) -> None:
    """Reject configs whose geometry features land in later v0.3 steps."""
    spec = cfg.geometry_spec
    if spec.tag == "parametric":
        return
    if spec.tag == "step":
        raise NotImplementedError("parametric/STEP geometry lands in v0.3 step 3+")
    if spec.tag in ("box", "baffle"):
        if len(spec.radiators) > 1:
            raise NotImplementedError("multi-radiator lands in v0.3 step 8")
        return
    raise NotImplementedError("parametric/STEP geometry lands in v0.3 step 3+")


def _require_positive_finite(d: dict[str, float], *, ctx: str) -> None:
    for key, val in d.items():
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{ctx}: {key} must be finite and > 0")


def _parse_pose_block(
    entry: dict[str, Any],
    *,
    ctx: str,
    require_position: bool = True,
) -> Pose:
    if require_position and "position" not in entry:
        raise ValueError(f"{ctx}: missing position")
    position = entry.get("position", (0.0, 0.0, 0.0))
    normal = entry.get("normal", (0.0, 0.0, 1.0))
    x_axis = entry.get("x_axis")
    return Pose.make(position, normal, x_axis)


def _parse_shape(entry: dict[str, Any], *, ctx: str, allow_hole: bool = True) -> ShapeSpec:
    if not allow_hole and "hole" in entry and entry["hole"]:
        raise ValueError(f"{ctx}: nested holes are not supported")
    keys = _SHAPE_KEYS.intersection(entry.keys())
    if len(keys) != 1:
        raise ValueError(f"{ctx}: exactly one shape key required among {_SHAPE_KEYS}")
    kind = next(iter(keys))
    raw = entry[kind]
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx}: {kind} params must be a dict")
    params = {k: float(v) for k, v in raw.items()}
    if kind == "rect":
        if set(params) != {"w", "h"}:
            raise ValueError(f"{ctx}: rect requires w and h")
    elif kind == "circle":
        if set(params) != {"d"}:
            raise ValueError(f"{ctx}: circle requires d")
    elif kind == "racetrack":
        if set(params) != {"w", "h"}:
            raise ValueError(f"{ctx}: racetrack requires w and h")
        if params["w"] < params["h"]:
            raise ValueError(f"{ctx}: racetrack requires w >= h")
    elif kind == "ellipse":
        if set(params) != {"a", "b"}:
            raise ValueError(f"{ctx}: ellipse requires a and b")
    _require_positive_finite(params, ctx=ctx)
    hole = None
    if allow_hole and "hole" in entry:
        if isinstance(entry["hole"], dict) and entry["hole"]:
            hole = _parse_shape(entry["hole"], ctx=f"{ctx} hole", allow_hole=False)
            if hole.hole is not None:
                raise ValueError(f"{ctx}: nested holes are not supported")
        else:
            hole = None
    return ShapeSpec(kind=kind, params=params, hole=hole)


def _legacy_shape(radiator: Radiator) -> ShapeSpec | None:
    shape = radiator.shape
    dims = radiator.dims
    if not dims:
        return None
    try:
        if shape == "circle":
            return ShapeSpec("circle", {"d": float(dims[0])}, None)
        if shape == "rect":
            return ShapeSpec("rect", {"w": float(dims[0]), "h": float(dims[1])}, None)
        if shape == "racetrack":
            return ShapeSpec(
                "racetrack",
                {"w": float(dims[0]), "h": float(dims[1])},
                None,
            )
    except (IndexError, TypeError, ValueError):
        return None
    return None


def _parse_primitive(entry: dict[str, Any], *, ctx: str) -> PrimitiveSpec:
    kinds = _PRIMITIVE_KEYS.intersection(entry.keys())
    if len(kinds) != 1:
        raise ValueError(f"{ctx}: exactly one primitive key among {_PRIMITIVE_KEYS}")
    kind = next(iter(kinds))
    raw = entry[kind]
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx}: {kind} params must be a dict")
    if kind == "box":
        if "size" not in raw:
            raise ValueError(f"{ctx}: box requires size [sx, sy, sz]")
        size = [float(x) for x in raw["size"]]
        if len(size) != 3:
            raise ValueError(f"{ctx}: box size must have length 3")
        _require_positive_finite(
            {"sx": size[0], "sy": size[1], "sz": size[2]},
            ctx=ctx,
        )
        pose = _parse_pose_block(entry, ctx=ctx)
        return PrimitiveSpec(
            kind="box",
            params={"sx": size[0], "sy": size[1], "sz": size[2]},
            pose=pose,
        )
    if kind == "cylinder":
        d = float(raw["d"])
        length = float(raw["length"])
        _require_positive_finite({"d": d, "length": length}, ctx=ctx)
        pose = _parse_pose_block(entry, ctx=ctx)
        return PrimitiveSpec(kind="cylinder", params={"d": d, "length": length}, pose=pose)
    if kind == "sphere":
        d = float(raw["d"])
        _require_positive_finite({"d": d}, ctx=ctx)
        pose = _parse_pose_block(entry, ctx=ctx)
        return PrimitiveSpec(kind="sphere", params={"d": d}, pose=pose)
    raise ValueError(f"{ctx}: unknown primitive {kind!r}")


def _parse_step_part(entry: dict[str, Any], *, ctx: str) -> StepPartSpec:
    if "step" not in entry:
        raise ValueError(f"{ctx}: expected step entry")
    raw = entry["step"]
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx}: step must be a dict")
    file_name = raw.get("file")
    if not file_name:
        raise ValueError(f"{ctx}: step.file is required")
    enclosure = raw.get("enclosure_part", "enclosure")
    placement = None
    if "placement" in raw:
        placement = Pose.from_dict(raw["placement"])
    return StepPartSpec(file=str(file_name), enclosure_part=str(enclosure), placement=placement)


def _parse_radiator_entry(entry: dict[str, Any], *, ctx: str) -> RadiatorSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"{ctx}: radiator entry must be a dict")
    name = entry.get("name")
    if not name:
        raise ValueError(f"{ctx}: radiator name is required")
    motion = entry.get("motion", "axial")
    if motion != "axial":
        raise ValueError(f"{ctx}: motion must be 'axial'")
    shape_keys = _SHAPE_KEYS.intersection(entry.keys())
    part = entry.get("part")
    axis = entry.get("axis")
    axis_tuple: tuple[float, float, float] | None = None
    if axis is not None:
        axis_tuple = tuple(float(x) for x in axis)
    if shape_keys and part is not None:
        raise ValueError(f"{ctx}: radiator cannot mix shape and part")
    if len(shape_keys) > 1:
        raise ValueError(f"{ctx}: exactly one shape key required among {_SHAPE_KEYS}")
    shape = None
    pose: Pose | None = None
    if shape_keys:
        shape = _parse_shape(entry, ctx=ctx)
        pose = _parse_pose_block(entry, ctx=ctx)
    elif part is not None:
        if "position" in entry or "normal" in entry:
            pose = _parse_pose_block(entry, ctx=ctx)
    else:
        raise ValueError(f"{ctx}: radiator needs a shape or part")
    return RadiatorSpec(
        name=str(name),
        shape=shape,
        part=str(part) if part is not None else None,
        axis=axis_tuple,
        pose=pose,
        motion=motion,
    )


def _parse_environment(data: dict[str, Any]) -> EnvironmentSpec:
    raw = data.get("environment")
    if raw is None:
        return EnvironmentSpec(())
    if not isinstance(raw, dict):
        raise ValueError("environment must be a dict")
    planes_in = raw.get("planes", raw.get("plane", None))
    if planes_in is None:
        return EnvironmentSpec(())
    if isinstance(planes_in, dict):
        planes_in = [planes_in]
    if not isinstance(planes_in, list):
        raise ValueError("environment.planes must be a list")
    axes_seen: set[str] = set()
    planes: list[PlaneSpec] = []
    for i, pl in enumerate(planes_in):
        if not isinstance(pl, dict):
            raise ValueError(f"environment.planes[{i}] must be a dict")
        axis = pl.get("axis")
        if axis not in _PLANE_AXES:
            raise ValueError(
                f'environment.planes[{i}].axis must be one of {sorted(_PLANE_AXES)}'
            )
        side = pl.get("side")
        if side not in _PLANE_SIDES:
            raise ValueError(
                f'environment.planes[{i}].side must be one of {sorted(_PLANE_SIDES)}'
            )
        offset = float(pl.get("offset", 0.0))
        if axis in axes_seen:
            raise ValueError(f"environment: duplicate plane axis {axis!r}")
        axes_seen.add(axis)
        planes.append(PlaneSpec(axis=axis, side=side, offset=offset))
    return EnvironmentSpec(tuple(planes))


def _geometry_tag(cfg_dict: dict[str, Any]) -> GeometryTag:
    raw = cfg_dict.get("geometry", "box")
    if isinstance(raw, dict):
        raise ValueError(
            f'geometry tag must be one of {", ".join(_ALLOWED_GEOMETRY_TAGS)}; got a dict'
        )
    if raw in _ALLOWED_GEOMETRY_TAGS:
        return raw  # type: ignore[return-value]
    raise ValueError(
        f'geometry tag must be one of {", ".join(_ALLOWED_GEOMETRY_TAGS)}; got {raw!r}'
    )


def _legacy_radiator_spec(radiator_dict: dict[str, Any]) -> RadiatorSpec:
    legacy_rad = Radiator(**radiator_dict)
    return RadiatorSpec(
        name="radiator",
        shape=_legacy_shape(legacy_rad),
        part=None,
        axis=None,
        pose=Pose.identity(),
        motion="axial",
    )


def _baffle_forbids_extras(cfg_dict: dict[str, Any]) -> None:
    if cfg_dict.get("radiators") is not None:
        raise ValueError(_BAFFLE_EXTRAS_MSG)
    env = _parse_environment(cfg_dict)
    if env.planes:
        raise ValueError(_BAFFLE_EXTRAS_MSG)


def parse_geometry(cfg_dict: dict[str, Any]) -> GeometrySpec:
    """Parse the geometry section of a config dict into a :class:`GeometrySpec`."""
    tag = _geometry_tag(cfg_dict)
    radiator_dict = cfg_dict.get("radiator")
    if tag == "baffle":
        if radiator_dict is None:
            raise ValueError("baffle geometry requires radiator section")
        _baffle_forbids_extras(cfg_dict)
        radiators_b = [_legacy_radiator_spec(radiator_dict)]
        return GeometrySpec(
            tag="baffle",
            parts=(),
            radiators=tuple(radiators_b),
            environment=EnvironmentSpec(()),
        )

    if tag == "box":
        if radiator_dict is None:
            raise ValueError("box geometry requires radiator section")
        environment = _parse_environment(cfg_dict)
        radiators: list[RadiatorSpec] = [_legacy_radiator_spec(radiator_dict)]
        extra = cfg_dict.get("radiators")
        if extra is not None:
            if not isinstance(extra, list):
                raise ValueError("radiators must be a list")
            for i, entry in enumerate(extra):
                radiators.append(_parse_radiator_entry(entry, ctx=f"radiators[{i}]"))
        _check_unique_radiator_names(radiators)
        return GeometrySpec(tag="box", parts=(), radiators=tuple(radiators), environment=environment)

    parts_in = cfg_dict.get("parts")
    if not isinstance(parts_in, list):
        raise ValueError(f'{tag} geometry requires non-empty "parts" list')

    environment = _parse_environment(cfg_dict)

    if tag == "parametric":
        parts: list[PrimitiveSpec | StepPartSpec] = []
        for i, entry in enumerate(parts_in):
            if "step" in entry:
                raise ValueError(
                    "mixing STEP and parametric parts is not supported in v0.3"
                )
            parts.append(_parse_primitive(entry, ctx=f"parts[{i}]"))
        if not parts:
            raise ValueError("parametric geometry requires at least one part")
        rad_in = cfg_dict.get("radiators")
        if not isinstance(rad_in, list) or not rad_in:
            raise ValueError("parametric geometry requires non-empty radiators")
        radiators_p: list[RadiatorSpec] = []
        for i, entry in enumerate(rad_in):
            rd = _parse_radiator_entry(entry, ctx=f"radiators[{i}]")
            if rd.shape is None or rd.pose is None:
                raise ValueError(f"radiators[{i}]: shape and pose are required")
            radiators_p.append(rd)
        _check_unique_radiator_names(radiators_p)
        return GeometrySpec(
            tag="parametric",
            parts=tuple(parts),
            radiators=tuple(radiators_p),
            environment=environment,
        )

    if tag == "step":
        if len(parts_in) != 1:
            raise ValueError("step geometry requires exactly one step part")
        entry = parts_in[0]
        if "step" not in entry:
            raise ValueError("step geometry part must be a step entry")
        part = _parse_step_part(entry, ctx="parts[0]")
        rad_in = cfg_dict.get("radiators")
        if not isinstance(rad_in, list) or not rad_in:
            raise ValueError("step geometry requires non-empty radiators")
        radiators_s: list[RadiatorSpec] = []
        for i, entry_r in enumerate(rad_in):
            rd = _parse_radiator_entry(entry_r, ctx=f"radiators[{i}]")
            if rd.part is None:
                raise ValueError(f"radiators[{i}]: part is required for step geometry")
            radiators_s.append(rd)
        _check_unique_radiator_names(radiators_s)
        return GeometrySpec(
            tag="step",
            parts=(part,),
            radiators=tuple(radiators_s),
            environment=environment,
        )

    raise ValueError(f"unsupported geometry tag {tag!r}")


def _check_unique_radiator_names(radiators: list[RadiatorSpec]) -> None:
    seen: set[str] = set()
    for rd in radiators:
        if rd.name in seen:
            raise ValueError(f"duplicate radiator name {rd.name!r}")
        seen.add(rd.name)
