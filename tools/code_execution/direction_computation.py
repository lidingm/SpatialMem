"""Direction/bearing computation: given viewpoint X, reference Y, and target Z,
determine the relative direction of Z with respect to Y as seen from X.

All three inputs can be either a 3D object position or a camera position
(extracted via frame index + extrinsics).
"""

from __future__ import annotations

import numpy as np

from tools.code_execution.distance_computation import (
    category_from_id,
    checker_direct_answer,
    checker_locate_or_fallback,
    find_first_instance_id,
    get_camera_position,
    instance_geometry,
)
from tools.visual_generation.sam3_segmentation import missing_tracked_categories


def _resolve_position(
    pos_or_frame: np.ndarray | int,
    extrinsics: np.ndarray | None = None,
) -> np.ndarray:
    """将输入统一解析为 3D 位置向量。

    Args:
        pos_or_frame: (3,) ndarray 表示 3D 位置，或 int 表示帧序号。
        extrinsics:   当 pos_or_frame 为 int 时需要提供 DA3 外参。
    """
    if isinstance(pos_or_frame, (int, np.integer)):
        if extrinsics is None:
            raise ValueError("需要提供 extrinsics 才能从帧序号获取相机位置")
        return get_camera_position(extrinsics, int(pos_or_frame))
    return np.asarray(pos_or_frame, dtype=np.float64)


def compute_relative_direction(
    viewpoint,
    reference,
    target,
    facing=None,
    extrinsics: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> dict[str, str | float]:
    """计算从 viewpoint 看，target 在 reference 的什么方位。

    在以 viewpoint 为原点的水平面上：
    - 前方 = viewpoint → facing 的方向（若 facing 提供）
             否则 = viewpoint → reference 的方向
    - 左右 = 前方向量叉乘 up 得到

    Args:
        viewpoint:  观察点，(3,) ndarray 或 int（帧序号）
        reference:  基准物体，(3,) ndarray 或 int
        target:     方位物体，(3,) ndarray 或 int
        facing:     定义正面朝向的物体/位置，(3,) ndarray 或 int（可选）。
                    当 viewpoint == reference 时必须提供，否则 forward 为零向量。
                    例：人站在 bed（viewpoint=reference=bed），朝向 tv（facing=tv）。
        extrinsics: DA3 外参，当任一输入为帧序号时需要
        up:         世界坐标系的"上"方向，默认 (0, -1, 0)（DA3/OpenCV 的 y 轴朝下）

    Returns:
        dict with keys:
          direction    - str, 如 "left-front", "right-back", "left", "front" 等
          lr_label     - str, "left" / "right" / "center"
          fb_label     - str, "front" / "back" / "center"
          lr_angle_deg - float, 左右偏角（左为正，右为负）
          fb_angle_deg - float, 前后偏角（前为正，后为负）
          angle_from_forward_deg - float, 观察正前方到 reference→target 方向的夹角，0-180
          distance_ref_to_target - float, reference 与 target 之间的距离
    """
    if up is None:
        up = np.array([0.0, -1.0, 0.0])
    up = np.asarray(up, dtype=np.float64)

    vp = _resolve_position(viewpoint, extrinsics)
    ref = _resolve_position(reference, extrinsics)
    tgt = _resolve_position(target, extrinsics)

    if facing is not None:
        fac = _resolve_position(facing, extrinsics)
        forward = fac - vp   # 朝向 facing 物体的方向
    else:
        forward = ref - vp   # 默认：朝向 reference
    forward_h = forward - np.dot(forward, up) * up
    norm_fh = np.linalg.norm(forward_h)
    if norm_fh < 1e-8:
        return {
            "direction": "undefined",
            "lr_label": "undefined", "fb_label": "undefined",
            "lr_angle_deg": 0.0, "fb_angle_deg": 0.0,
            "angle_from_forward_deg": 0.0,
            "distance_ref_to_target": float(np.linalg.norm(tgt - ref)),
        }
    forward_h = forward_h / norm_fh

    right_h = np.cross(forward_h, up)
    right_norm = np.linalg.norm(right_h)
    if right_norm < 1e-8:
        right_h = np.array([1.0, 0.0, 0.0])
    else:
        right_h = right_h / right_norm

    delta = tgt - ref
    delta_h = delta - np.dot(delta, up) * up

    lr_proj = float(np.dot(delta_h, right_h))
    fb_proj = float(np.dot(delta_h, forward_h))

    ref_tgt_dist = float(np.linalg.norm(delta))
    horiz_dist = float(np.linalg.norm(delta_h))

    if horiz_dist < 1e-6:
        lr_angle = 0.0
        fb_angle = 0.0
        angle_from_forward = 0.0
    else:
        lr_angle = float(np.degrees(np.arctan2(-lr_proj, fb_proj)))
        fb_angle = float(np.degrees(np.arctan2(fb_proj, abs(lr_proj))))
        cosang = float(np.clip(fb_proj / max(horiz_dist, 1e-8), -1.0, 1.0))
        angle_from_forward = float(np.degrees(np.arccos(cosang)))

    angle_thresh = 20.0
    abs_lr = abs(lr_angle)

    if abs_lr < angle_thresh:
        lr_label = "center"
    elif lr_angle > 0:
        lr_label = "left"
    else:
        lr_label = "right"

    if abs(fb_proj) < horiz_dist * np.sin(np.radians(angle_thresh)):
        fb_label = "center"
    elif fb_proj > 0:
        fb_label = "front"
    else:
        fb_label = "back"

    parts = []
    if lr_label != "center":
        parts.append(lr_label)
    if fb_label != "center":
        parts.append(fb_label)
    direction = "-".join(parts) if parts else "same position"

    return {
        "direction": direction,
        "lr_label": lr_label,
        "fb_label": fb_label,
        "lr_angle_deg": lr_angle,
        "fb_angle_deg": fb_angle,
        "angle_from_forward_deg": angle_from_forward,
        "distance_ref_to_target": ref_tgt_dist,
    }


# ─── Formal task workflow ──────────────────────────────────────────────────

def _name_from_params(params: dict, name_key: str, id_key: str, r3d: dict) -> str | None:
    return params.get(name_key) or category_from_id(r3d, params.get(id_key))


def _resolve_checked_object(
    params_id,
    object_name: str | None,
    ctx: dict,
    checker,
    loc_cache: dict[str, dict],
) -> tuple[np.ndarray, dict]:
    r3d = ctx["results_3d"]
    fallback_id = params_id
    if fallback_id is None and object_name:
        fallback_id = find_first_instance_id(r3d, object_name)
    if fallback_id is None:
        ids = r3d.get("obj_id_list", [])
        fallback_id = int(ids[0]) if ids else None
    if fallback_id is None:
        raise RuntimeError(f"no instance found for object {object_name!r}")

    key = object_name or str(fallback_id)
    if key not in loc_cache:
        loc_cache[key] = checker_locate_or_fallback(checker, ctx, key, int(fallback_id))
    loc = loc_cache[key]
    geom = instance_geometry(ctx, int(loc["instance_id"]), loc.get("keep_frames"))
    return geom["avg_pos"], loc


def _resolve_entity(
    entity_id,
    entity_type: str,
    object_name: str | None,
    ctx: dict,
    checker,
    loc_cache: dict[str, dict],
) -> tuple[np.ndarray, dict | None, str]:
    if entity_type == "camera":
        fi = int(entity_id or 0)
        return get_camera_position(ctx["da3_result"]["extrinsics"], fi), None, f"camera(frame {fi})"
    pos, loc = _resolve_checked_object(entity_id, object_name, ctx, checker, loc_cache)
    return pos, loc, object_name or f"instance {loc.get('instance_id')}"


def run_direction_task(
    params: dict,
    ctx: dict,
    checker,
    frame_paths: list[str],
) -> dict:
    """Formal direction workflow with Checker-based target validation."""
    r3d = ctx.get("results_3d")
    da3 = ctx.get("da3_result")
    if r3d is None:
        raise RuntimeError("instance_3d_localization must be run first")
    if da3 is None:
        raise RuntimeError("depth_estimation must be run first")

    vp_type = params.get("viewpoint_type", "camera")
    ref_type = params.get("reference_type", "object")
    tgt_type = params.get("target_type", "object")
    fac_type = params.get("facing_type", "object")

    if vp_type == "camera":
        vp_id = params.get("viewpoint", params.get("viewpoint_frame", 0))
    else:
        vp_id = params.get("viewpoint")
    ref_id = params.get("reference")
    tgt_id = params.get("target")
    fac_id = params.get("facing")

    vp_name = params.get("viewpoint_object") or params.get("viewpoint_target") \
        or _name_from_params(params, "viewpoint_name", "viewpoint", r3d)
    ref_name = params.get("reference_object") or params.get("reference_target") \
        or _name_from_params(params, "reference_name", "reference", r3d)
    tgt_name = params.get("target_object") or params.get("target_target") \
        or _name_from_params(params, "target_name", "target", r3d)
    fac_name = params.get("facing_object") or params.get("facing_target") \
        or _name_from_params(params, "facing_name", "facing", r3d)

    required: list[str] = []
    for typ, name in ((vp_type, vp_name), (ref_type, ref_name),
                      (tgt_type, tgt_name), (fac_type, fac_name if fac_id is not None or fac_name else None)):
        if typ == "object" and name and name not in required:
            required.append(name)
    missing = missing_tracked_categories(ctx, required)
    if missing:
        return checker_direct_answer(
            checker, ctx, frame_paths,
            reason=f"SAM3 produced zero tracks for required target category/categories: {missing}",
        )

    loc_cache: dict[str, dict] = {}
    viewpoint_pos, vp_loc, vp_label = _resolve_entity(
        vp_id, vp_type, vp_name, ctx, checker, loc_cache)
    reference_pos, ref_loc, ref_label = _resolve_entity(
        ref_id, ref_type, ref_name, ctx, checker, loc_cache)
    target_pos, tgt_loc, tgt_label = _resolve_entity(
        tgt_id, tgt_type, tgt_name, ctx, checker, loc_cache)
    facing_pos = None
    fac_label = ""
    fac_loc = None
    if fac_id is not None or fac_name:
        facing_pos, fac_loc, fac_label = _resolve_entity(
            fac_id, fac_type, fac_name, ctx, checker, loc_cache)

    from tools.code_execution.distance_computation import _render_final_localizations
    _render_final_localizations(ctx)

    dr = compute_relative_direction(
        viewpoint=viewpoint_pos,
        reference=reference_pos,
        target=target_pos,
        facing=facing_pos,
        extrinsics=da3["extrinsics"],
    )
    result = {
        **dr,
        "viewpoint": vp_label,
        "reference": ref_label,
        "target": tgt_label,
        "facing": fac_label or None,
        "checker": {
            k: v for k, v in {
                "viewpoint": vp_loc,
                "reference": ref_loc,
                "target": tgt_loc,
                "facing": fac_loc,
            }.items() if v is not None
        },
    }
    ctx["direction"] = result
    summary = (
        f"Direction from {vp_label}{' facing ' + fac_label if fac_label else ''}: "
        f"{tgt_label} is {dr['direction']} of {ref_label} "
        f"(angle_from_forward={dr['angle_from_forward_deg']:.1f}deg, "
        f"lr_angle={dr['lr_angle_deg']:.1f}deg, fb_angle={dr['fb_angle_deg']:.1f}deg)"
    )
    return {"direct_answer": False, "direction": result, "summary": summary}
