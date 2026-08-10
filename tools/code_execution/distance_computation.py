"""Distance computation between objects and/or camera positions."""

from __future__ import annotations

import numpy as np
from typing import Any

from tools.visual_generation.sam3_segmentation import missing_tracked_categories


def _as_homogeneous(ext: np.ndarray) -> np.ndarray:
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        out = np.eye(4, dtype=ext.dtype)
        out[:3, :4] = ext
        return out
    raise ValueError(f"Expected (3,4) or (4,4), got {ext.shape}")


def get_camera_position(extrinsics: np.ndarray, frame_index: int) -> np.ndarray:
    """从 DA3 外参中提取某帧相机的世界坐标位置。"""
    w2c = _as_homogeneous(extrinsics[frame_index])
    c2w = np.linalg.inv(w2c)
    return c2w[:3, 3]


def distance_object_to_object(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """质心到质心距离（保留用于内部调用）。"""
    return float(np.linalg.norm(np.asarray(pos_a) - np.asarray(pos_b)))


def distance_bbox_to_bbox(
    bmin_a: np.ndarray, bmax_a: np.ndarray,
    bmin_b: np.ndarray, bmax_b: np.ndarray,
) -> float:
    """两个 AABB 包围盒之间的最近点距离（closest-point distance）。

    当两个盒子重叠时返回 0。这是 VSI 题目中
    'Measuring from the closest point of each object' 所要求的度量。
    """
    gap = np.maximum(0.0, np.maximum(
        np.asarray(bmin_a) - np.asarray(bmax_b),
        np.asarray(bmin_b) - np.asarray(bmax_a),
    ))
    return float(np.linalg.norm(gap))


def distance_object_to_camera(
    obj_pos: np.ndarray,
    extrinsics: np.ndarray,
    frame_index: int,
) -> float:
    """计算物体到某帧相机的距离。

    Args:
        obj_pos:     物体的 3D 位置 (3,)
        extrinsics:  DA3 外参数组 (N, 3, 4) 或 (N, 4, 4)
        frame_index: 帧序号

    Returns:
        欧氏距离
    """
    cam_pos = get_camera_position(extrinsics, frame_index)
    return distance_object_to_object(obj_pos, cam_pos)


# ─── Task workflow helpers ─────────────────────────────────────────────────

def _norm_name(name: Any) -> str:
    return str(name or "").strip().lower()


def category_from_id(r3d: dict, obj_id: int | None) -> str | None:
    labels = r3d.get("merged_labels") or []
    if obj_id is None:
        return None
    try:
        obj_id = int(obj_id)
    except Exception:
        return None
    if 0 <= obj_id < len(labels):
        return str(labels[obj_id])
    return None


def find_first_instance_id(r3d: dict, object_name: str) -> int | None:
    target = _norm_name(object_name)
    labels = r3d.get("merged_labels") or []
    for oid in r3d.get("obj_id_list", []):
        if 0 <= int(oid) < len(labels) and _norm_name(labels[int(oid)]) == target:
            return int(oid)
    return None


def checker_direct_answer(
    checker,
    ctx: dict,
    frame_paths: list[str],
    reason: str,
) -> dict:
    if checker is None:
        ans = {"answer": "", "reasoning": "", "confidence": 0.0,
               "reason": reason, "error": "no checker/llm available"}
        ans["summary"] = f"Checker raw-sample direct answer failed: {ans['error']}"
    else:
        ans = checker.answer_from_raw_sample(
            question=ctx.get("question", ""),
            frame_paths=frame_paths,
            answer_format=ctx.get("answer_format", ""),
            reason=reason,
        )
    ctx["checker_direct_answer"] = ans
    return {
        "direct_answer": True,
        "answer": ans.get("answer", ""),
        "checker_result": ans,
        "summary": (
            f"skipped geometry because {reason}; "
            f"Checker direct answer={ans.get('answer')!r}, "
            f"confidence={float(ans.get('confidence', 0.0) or 0.0):.2f}"
        ),
    }


def instance_geometry(
    ctx: dict,
    obj_id: int,
    frames: list[int] | None = None,
    dejitter: bool = True,
) -> dict:
    """Recompute one object's 3D geometry, optionally using Checker-kept frames."""
    frame_key = None if frames is None else tuple(sorted(int(f) for f in frames))
    cache_key = (int(obj_id), frame_key, bool(dejitter))
    geom_cache = ctx.setdefault("instance_geometry_cache", {})
    if cache_key in geom_cache:
        return geom_cache[cache_key]

    r3d = ctx["results_3d"]
    da3 = ctx["da3_result"]
    from tools.code_execution.instance_3d_localization import back_project_instance

    seg = ctx.get("seg_results")
    f2g = r3d.get("frame_to_global")
    if seg is not None and f2g is not None:
        g = back_project_instance(
            seg, da3["depth"], da3["intrinsics"], da3["extrinsics"],
            int(obj_id), f2g, frames=frames, dejitter=dejitter,
        )
        if g is not None:
            geom_cache[cache_key] = g
            return g

    idx = int(obj_id)
    g = {
        "avg_pos": r3d["merged_positions"][idx],
        "bbox_min": r3d["merged_bbox_min"][idx],
        "bbox_max": r3d["merged_bbox_max"][idx],
        "bbox_size": r3d["merged_bbox_size"][idx],
        "n_points": 0,
        "frames_used": None,
    }
    geom_cache[cache_key] = g
    return g


def checker_locate_or_fallback(
    checker,
    ctx: dict,
    object_name: str,
    fallback_id: int | None = None,
) -> dict:
    r3d = ctx["results_3d"]
    cache_key = _norm_name(object_name)
    cached = (ctx.get("final_localizations") or {}).get(cache_key)
    if cached is None:
        cached = (ctx.get("checker_location") or {}).get(cache_key)
    if cached is not None and cached.get("instance_id") is not None:
        return cached

    if checker is not None and object_name:
        loc = checker.locate_single_object(object_name, r3d)
    else:
        loc = {"object": object_name, "instance_id": None, "instance_label": None,
               "keep_frames": [], "dropped_frames": [], "reason": "",
               "error": "no checker/llm available"}

    if loc.get("instance_id") is None:
        oid = fallback_id if fallback_id is not None else find_first_instance_id(r3d, object_name)
        if oid is not None:
            loc.update({
                "instance_id": int(oid),
                "instance_label": f"{object_name} {int(oid)}",
                "keep_frames": None,
                "dropped_frames": [],
                "reason": loc.get("reason") or "fallback to pre-resolved instance id",
            })
    loc["object"] = cache_key or object_name
    ctx.setdefault("checker_location", {})[cache_key] = loc
    ctx.setdefault("checker_summaries", []).append(loc.get("summary") or (
        f"Checker location for {object_name}: chose instance {loc.get('instance_id')}"
    ))
    if loc.get("instance_id") is not None:
        ctx.setdefault("final_localizations", {})[cache_key] = loc
    return loc


def _render_final_localizations(ctx: dict) -> None:
    """Best-effort final 2D visualization of Checker-approved target tracks."""
    r3d = ctx.get("results_3d")
    frame_paths = ctx.get("frame_paths") or []
    final_locs = ctx.get("final_localizations") or {}
    output_root = ctx.get("output_root")
    if r3d is None or not frame_paths or not final_locs or not output_root:
        return
    try:
        from pathlib import Path
        from tools.code_execution.instance_3d_localization import (
            draw_final_localization_annotations,
        )
        geometry_by_id = {}
        for loc in final_locs.values():
            if loc is None or loc.get("instance_id") is None:
                continue
            oid = int(loc["instance_id"])
            geom = instance_geometry(ctx, oid, loc.get("keep_frames"), dejitter=False)
            if geom is not None:
                geometry_by_id[oid] = geom
        meta = draw_final_localization_annotations(
            r3d,
            frame_paths,
            final_locs,
            Path(output_root) / "spatial" / "final_localization",
            geometry_by_id=geometry_by_id,
        )
        ctx["final_localization_annotations"] = meta
    except Exception as e:  # noqa: BLE001 - visualization must not break geometry
        ctx["final_localization_annotation_error"] = str(e)


def _object_names_for_distance(params: dict, r3d: dict) -> tuple[str | None, str | None]:
    a_id = params.get("obj_a_id")
    b_id = params.get("obj_b_id")
    if "obj_id" in params and a_id is None:
        a_id = params.get("obj_id")
    return (
        params.get("obj_a") or params.get("obj") or params.get("object_name") or category_from_id(r3d, a_id),
        params.get("obj_b") or category_from_id(r3d, b_id),
    )


def run_distance_task(
    params: dict,
    ctx: dict,
    checker,
    frame_paths: list[str],
) -> dict:
    """Formal distance workflow with Checker-based target validation."""
    ctx.pop("distance", None)
    r3d = ctx.get("results_3d")
    da3 = ctx.get("da3_result")
    if r3d is None:
        raise RuntimeError("instance_3d_localization must be run first")
    if da3 is None:
        raise RuntimeError("depth_estimation must be run first")

    mode = params.get("mode", "object_to_object")
    id_list = [int(x) for x in r3d.get("obj_id_list", [])]
    a_name, b_name = _object_names_for_distance(params, r3d)
    required = [n for n in [a_name, b_name if mode == "object_to_object" else None] if n]
    missing = missing_tracked_categories(ctx, required)
    if missing:
        return checker_direct_answer(
            checker, ctx, frame_paths,
            reason=f"SAM3 produced zero tracks for required target category/categories: {missing}",
        )

    if mode == "object_to_object":
        a_id = params.get("obj_a_id")
        b_id = params.get("obj_b_id")
        if a_id is None:
            a_id = find_first_instance_id(r3d, a_name or "")
        if b_id is None:
            b_id = find_first_instance_id(r3d, b_name or "")
        if a_id is None and len(id_list) > 0:
            a_id = id_list[0]
        if b_id is None and len(id_list) > 1:
            b_id = id_list[1]
        if a_id is None or b_id is None:
            raise RuntimeError("not enough object instances for object_to_object distance")

        loc_a = checker_locate_or_fallback(checker, ctx, a_name or str(a_id), int(a_id))
        loc_b = checker_locate_or_fallback(checker, ctx, b_name or str(b_id), int(b_id))
        a_id = int(loc_a["instance_id"])
        b_id = int(loc_b["instance_id"])
        ga = instance_geometry(ctx, a_id, loc_a.get("keep_frames"))
        gb = instance_geometry(ctx, b_id, loc_b.get("keep_frames"))
        d = distance_bbox_to_bbox(ga["bbox_min"], ga["bbox_max"], gb["bbox_min"], gb["bbox_max"])
        _render_final_localizations(ctx)
        result = {
            "mode": "object_to_object",
            "meters": float(d),
            "a": a_name or a_id,
            "b": b_name or b_id,
            "a_id": a_id,
            "b_id": b_id,
            "a_frames": ga.get("frames_used"),
            "b_frames": gb.get("frames_used"),
            "checker": {str(a_name or a_id): loc_a, str(b_name or b_id): loc_b},
        }
        ctx["distance"] = result
        return {
            "direct_answer": False,
            "distance": result,
            "summary": (
                f"Closest-point distance between {a_name or a_id} and {b_name or b_id}: "
                f"{d:.3f} meters"
            ),
        }

    obj_id = params.get("obj_id")
    if obj_id is None:
        obj_id = find_first_instance_id(r3d, a_name or "")
    if obj_id is None and id_list:
        obj_id = id_list[0]
    if obj_id is None:
        raise RuntimeError("no object instance for object_to_camera distance")

    loc = checker_locate_or_fallback(checker, ctx, a_name or str(obj_id), int(obj_id))
    obj_id = int(loc["instance_id"])
    geom = instance_geometry(ctx, obj_id, loc.get("keep_frames"))
    _render_final_localizations(ctx)
    fi = int(params.get("frame_index", 0))
    d = distance_object_to_camera(geom["avg_pos"], da3["extrinsics"], fi)
    result = {
        "mode": "object_to_camera",
        "meters": float(d),
        "a": a_name or obj_id,
        "b": f"camera@{fi}",
        "obj_id": obj_id,
        "a_frames": geom.get("frames_used"),
        "checker": {str(a_name or obj_id): loc},
    }
    ctx["distance"] = result
    return {
        "direct_answer": False,
        "distance": result,
        "summary": (
            f"Distance from {a_name or obj_id} to camera at frame {fi}: {d:.3f} meters"
        ),
    }
