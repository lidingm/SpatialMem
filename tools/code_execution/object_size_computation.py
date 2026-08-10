"""Object size computation from Checker-validated 3D instance geometry."""

from __future__ import annotations

import numpy as np

from tools.code_execution.distance_computation import (
    category_from_id,
    checker_direct_answer,
    checker_locate_or_fallback,
    find_first_instance_id,
    instance_geometry,
)
from tools.visual_generation.sam3_segmentation import missing_tracked_categories


def _dimension_value(size_xyz: np.ndarray, dimension: str) -> float:
    dim = str(dimension or "longest").strip().lower()
    size_xyz = np.asarray(size_xyz, dtype=np.float64)
    if dim in ("x", "width", "w"):
        return float(size_xyz[0])
    if dim in ("y", "height", "h", "vertical"):
        return float(size_xyz[1])
    if dim in ("z", "depth", "length", "d", "l"):
        return float(size_xyz[2])
    if dim in ("shortest", "min"):
        return float(np.min(size_xyz))
    return float(np.max(size_xyz))


def _dimension_label(dimension: str) -> str:
    dim = str(dimension or "longest").strip().lower()
    if dim in ("x", "width", "w"):
        return "width"
    if dim in ("y", "height", "h", "vertical"):
        return "height"
    if dim in ("z", "depth", "length", "d", "l"):
        return "depth"
    if dim in ("shortest", "min"):
        return "shortest dimension"
    return "longest dimension"


def run_object_size_task(
    params: dict,
    ctx: dict,
    checker,
    frame_paths: list[str],
) -> dict:
    """Formal object-size workflow with Checker-based target validation.

    Parameters accepted:
      - object_name / obj / target / object_category: target category name
      - obj_id: optional pre-resolved 3D instance id
      - dimension: width / height / depth / longest(default) / shortest
      - unit: m(default) or cm
    """
    r3d = ctx.get("results_3d")
    if r3d is None:
        raise RuntimeError("instance_3d_localization must be run first")
    if ctx.get("da3_result") is None:
        raise RuntimeError("depth_estimation must be run first")

    obj_id = params.get("obj_id")
    object_name = (
        params.get("object_name")
        or params.get("obj")
        or params.get("target")
        or params.get("object_category")
        or category_from_id(r3d, obj_id)
    )
    if object_name:
        missing = missing_tracked_categories(ctx, [object_name])
        if missing:
            return checker_direct_answer(
                checker, ctx, frame_paths,
                reason=f"SAM3 produced zero tracks for required target category/categories: {missing}",
            )

    if obj_id is None and object_name:
        obj_id = find_first_instance_id(r3d, object_name)
    if obj_id is None:
        ids = r3d.get("obj_id_list", [])
        obj_id = int(ids[0]) if ids else None
    if obj_id is None:
        raise RuntimeError("no object instance for object_size_computation")

    loc = checker_locate_or_fallback(checker, ctx, object_name or str(obj_id), int(obj_id))
    geom = instance_geometry(ctx, int(loc["instance_id"]), loc.get("keep_frames"), dejitter=False)
    from tools.code_execution.distance_computation import _render_final_localizations
    _render_final_localizations(ctx)
    size_m = np.asarray(geom["bbox_size"], dtype=np.float64)
    dimension = params.get("dimension", "longest")
    value_m = _dimension_value(size_m, dimension)
    volume_m3 = float(np.prod(size_m))
    unit = str(params.get("unit", "m")).strip().lower()
    value = value_m * 100.0 if unit in ("cm", "centimeter", "centimeters") else value_m
    unit = "cm" if unit in ("cm", "centimeter", "centimeters") else "m"

    result = {
        "object": object_name or loc.get("instance_label") or loc.get("instance_id"),
        "instance_id": int(loc["instance_id"]),
        "dimension": dimension,
        "dimension_label": _dimension_label(dimension),
        "value": float(value),
        "unit": unit,
        "size_xyz_m": [float(v) for v in size_m],
        "bbox_volume_m3": volume_m3,
        "frames_used": geom.get("frames_used"),
        "checker": loc,
    }
    ctx["object_size"] = result
    return {
        "direct_answer": False,
        "object_size": result,
        "summary": (
            f"Object size for {result['object']}: "
            f"bbox_size_xyz=({size_m[0]:.3f}, {size_m[1]:.3f}, {size_m[2]:.3f})m, "
            f"bbox_volume={volume_m3:.4f}m3; "
            f"queried {result['dimension_label']}={value:.3f}{unit}"
        ),
    }
