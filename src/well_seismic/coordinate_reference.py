"""Deterministic horizontal CRS validation and coordinate transformation.

The platform keeps well coordinates in the seismic survey's projected,
metre-based CRS before any spatial matching.  ``always_xy=True`` is deliberate:
user-facing X/Y always means easting/longitude followed by northing/latitude,
independent of the authority axis order stored in the CRS definition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import CRS, Transformer, __proj_version__
from pyproj import __version__ as pyproj_version
from pyproj.exceptions import CRSError, ProjError
from pyproj.transformer import TransformerGroup


class CoordinateReferenceError(ValueError):
    """Raised when a CRS contract cannot be applied without guessing."""


@dataclass(frozen=True)
class CoordinateTransformResult:
    x: np.ndarray
    y: np.ndarray
    source_crs: str
    target_crs: str
    transformed: bool
    operation: str
    accuracy_m: float | None
    pyproj_version: str = pyproj_version
    proj_version: str = __proj_version__

    def provenance(self) -> dict[str, Any]:
        return {
            "source_crs": self.source_crs,
            "target_crs": self.target_crs,
            "transformed": self.transformed,
            "operation": self.operation,
            "accuracy_m": self.accuracy_m,
            "axis_contract": "always_xy",
            "pyproj_version": self.pyproj_version,
            "proj_version": self.proj_version,
        }


_AMBIGUOUS_BARE_NAMES = {
    "WGS84",
    "WGS1984",
    "BEIJING54",
    "BEIJING1954",
    "BJ54",
    "XIAN80",
    "XIAN1980",
    "CGCS2000",
}

_LOCAL_PROJECTED_METRE_CRS_DEFINITIONS = {
    "LOCAL_NAD27_WYOMING_EAST_CENTRAL_METRE": (
        "+proj=tmerc +lat_0=40.6666666666667 +lon_0=-107.333333333333 "
        "+k=0.999941177 +x_0=152400.30480061 +y_0=0 +datum=NAD27 "
        "+units=m +no_defs +type=crs"
    ),
}


def _canonical_authority(crs: CRS) -> str:
    for identifier, definition in _LOCAL_PROJECTED_METRE_CRS_DEFINITIONS.items():
        if crs.equals(CRS.from_user_input(definition)):
            return identifier
    authority = crs.to_authority()
    if authority is not None:
        return f"{authority[0].upper()}:{authority[1]}"
    # WKT/PROJJSON remains a valid explicit definition even when it has no
    # authority code.  Use stable WKT2 rather than a display name.
    return crs.to_wkt(version="WKT2_2019", pretty=False)


def parse_crs(value: str | CRS, *, field: str = "CRS") -> CRS:
    if isinstance(value, CRS):
        return value
    text = str(value or "").strip()
    if not text:
        raise CoordinateReferenceError(f"{field}未声明")
    normalized_name = re.sub(r"[^A-Z0-9]", "", text.upper())
    if normalized_name in _AMBIGUOUS_BARE_NAMES:
        raise CoordinateReferenceError(
            f"{field}={text!r}不唯一；请提供EPSG代码或完整WKT/PROJJSON，"
            "并明确投影分带/中央经线"
        )
    local_definition = _LOCAL_PROJECTED_METRE_CRS_DEFINITIONS.get(text.upper())
    if local_definition is not None:
        text = local_definition
    epsg = re.fullmatch(r"\s*EPSG\s*[,=: ]\s*(\d{4,6})\s*", text, re.I)
    if epsg:
        text = f"EPSG:{epsg.group(1)}"
    try:
        return CRS.from_user_input(text)
    except CRSError as exc:
        raise CoordinateReferenceError(f"无法解析{field}={value!r}: {exc}") from exc


def canonical_crs_id(value: str | CRS, *, field: str = "CRS") -> str:
    return _canonical_authority(parse_crs(value, field=field))


def require_projected_metre_crs(value: str | CRS, *, field: str = "目标CRS") -> CRS:
    crs = parse_crs(value, field=field)
    if not crs.is_projected:
        raise CoordinateReferenceError(
            f"{field}必须是投影坐标系且单位为米；{_canonical_authority(crs)}"
            "是地理/非投影坐标系"
        )
    horizontal_axes = tuple(crs.axis_info[:2])
    if len(horizontal_axes) != 2 or any(
        axis.unit_conversion_factor is None
        or not np.isclose(float(axis.unit_conversion_factor), 1.0, rtol=0.0, atol=1e-12)
        for axis in horizontal_axes
    ):
        units = ", ".join(str(axis.unit_name or "unknown") for axis in horizontal_axes)
        raise CoordinateReferenceError(
            f"{field}必须使用米作为水平单位；{_canonical_authority(crs)}轴单位为{units or 'unknown'}"
        )
    return crs


def _best_transformer(source: CRS, target: CRS) -> Transformer:
    try:
        group = TransformerGroup(
            source,
            target,
            always_xy=True,
            allow_ballpark=False,
        )
    except (CRSError, ProjError) as exc:
        raise CoordinateReferenceError(f"无法建立坐标转换：{exc}") from exc
    if not group.transformers:
        raise CoordinateReferenceError(
            f"没有可用的非ballpark坐标转换：{_canonical_authority(source)} -> "
            f"{_canonical_authority(target)}"
        )
    if not group.best_available:
        missing = sorted(
            {
                grid.short_name
                for operation in group.unavailable_operations
                for grid in operation.grids
                if grid.short_name
            }
        )
        suffix = f"；缺少网格：{', '.join(missing)}" if missing else ""
        raise CoordinateReferenceError(
            "最高精度坐标转换当前不可用，拒绝静默降级"
            f"：{_canonical_authority(source)} -> {_canonical_authority(target)}{suffix}"
        )
    if len(group.transformers) == 1:
        return group.transformers[0]

    # PROJ orders candidate operations, but selecting the first item without
    # an explicit area of interest can silently choose between equally valid
    # datum operations.  A unique, declared accuracy is deterministic; tied or
    # unknown candidates need an operation/AOI decision instead of a guess.
    known_accuracy = [
        transformer
        for transformer in group.transformers
        if float(transformer.accuracy) >= 0.0
    ]
    if known_accuracy:
        best_accuracy = min(float(item.accuracy) for item in known_accuracy)
        best = [
            item
            for item in known_accuracy
            if np.isclose(
                float(item.accuracy),
                best_accuracy,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if len(best) == 1:
            return best[0]
    raise CoordinateReferenceError(
        "存在多个同等级坐标转换操作，无法在没有测区范围或指定操作的情况下"
        "确定性选择："
        f"{_canonical_authority(source)} -> {_canonical_authority(target)}"
    )


def transform_xy(
    x: Any,
    y: Any,
    *,
    source_crs: str | CRS,
    target_crs: str | CRS,
) -> CoordinateTransformResult:
    """Transform finite X/Y pairs into a projected metre target CRS.

    NaN pairs are preserved as NaN.  Partially finite pairs fail closed because
    transforming only one coordinate would silently corrupt a trajectory.
    """

    source = parse_crs(source_crs, field="源CRS")
    target = require_projected_metre_crs(target_crs, field="目标CRS")
    source_id = _canonical_authority(source)
    target_id = _canonical_authority(target)
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )
    x_out = np.full(x_array.shape, np.nan, dtype=float)
    y_out = np.full(y_array.shape, np.nan, dtype=float)
    finite_x = np.isfinite(x_array)
    finite_y = np.isfinite(y_array)
    if np.any(finite_x != finite_y):
        raise CoordinateReferenceError("X/Y必须成对有效，不能只转换单个坐标分量")
    valid = finite_x & finite_y
    if source == target:
        x_out[valid] = x_array[valid]
        y_out[valid] = y_array[valid]
        return CoordinateTransformResult(
            x=x_out,
            y=y_out,
            source_crs=source_id,
            target_crs=target_id,
            transformed=False,
            operation="identity",
            accuracy_m=0.0,
        )
    transformer = _best_transformer(source, target)
    try:
        transformed_x, transformed_y = transformer.transform(
            x_array[valid],
            y_array[valid],
            errcheck=True,
        )
    except (ProjError, ValueError) as exc:
        raise CoordinateReferenceError(
            f"坐标转换失败：{source_id} -> {target_id}: {exc}"
        ) from exc
    x_out[valid] = np.asarray(transformed_x, dtype=float)
    y_out[valid] = np.asarray(transformed_y, dtype=float)
    if not (np.isfinite(x_out[valid]).all() and np.isfinite(y_out[valid]).all()):
        raise CoordinateReferenceError(
            f"坐标转换产生非有限值：{source_id} -> {target_id}"
        )
    accuracy = float(transformer.accuracy)
    return CoordinateTransformResult(
        x=x_out,
        y=y_out,
        source_crs=source_id,
        target_crs=target_id,
        transformed=True,
        operation=str(transformer.description or transformer.definition),
        accuracy_m=None if accuracy < 0 else accuracy,
    )


def transform_offset_path(
    head_x: float,
    head_y: float,
    east_offset_m: Any,
    north_offset_m: Any,
    *,
    source_crs: str | CRS,
    target_crs: str | CRS,
) -> CoordinateTransformResult:
    """Compose metre offsets with a source-CRS well head, then reproject.

    Projected source axes use their declared unit conversion.  Geographic
    sources use the CRS ellipsoid geodesic, so metre offsets are never added
    directly to longitude/latitude degrees.
    """

    source = parse_crs(source_crs, field="井轨迹源CRS")
    east, north = np.broadcast_arrays(
        np.asarray(east_offset_m, dtype=float),
        np.asarray(north_offset_m, dtype=float),
    )
    if np.any(np.isfinite(east) != np.isfinite(north)):
        raise CoordinateReferenceError("轨迹东/北偏移量必须成对有效")
    valid = np.isfinite(east) & np.isfinite(north)
    source_x = np.full(east.shape, np.nan, dtype=float)
    source_y = np.full(north.shape, np.nan, dtype=float)
    if source.is_projected:
        axes = tuple(source.axis_info[:2])
        if len(axes) != 2 or any(axis.unit_conversion_factor is None for axis in axes):
            raise CoordinateReferenceError("投影源CRS缺少可用的水平轴单位")
        x_factor = float(axes[0].unit_conversion_factor)
        y_factor = float(axes[1].unit_conversion_factor)
        if x_factor <= 0 or y_factor <= 0:
            raise CoordinateReferenceError("投影源CRS水平轴单位换算系数无效")
        source_x[valid] = float(head_x) + east[valid] / x_factor
        source_y[valid] = float(head_y) + north[valid] / y_factor
    elif source.is_geographic:
        distance = np.hypot(east[valid], north[valid])
        azimuth = np.degrees(np.arctan2(east[valid], north[valid]))
        geod = source.get_geod()
        longitude, latitude, _ = geod.fwd(
            np.full(distance.shape, float(head_x), dtype=float),
            np.full(distance.shape, float(head_y), dtype=float),
            azimuth,
            distance,
        )
        source_x[valid] = longitude
        source_y[valid] = latitude
    else:
        raise CoordinateReferenceError("井轨迹源CRS必须是地理或投影坐标系")
    return transform_xy(
        source_x,
        source_y,
        source_crs=source,
        target_crs=target_crs,
    )
