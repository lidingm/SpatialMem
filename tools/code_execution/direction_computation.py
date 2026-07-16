"""Direction/bearing computation: given viewpoint X, reference Y, and target Z,
determine the relative direction of Z with respect to Y as seen from X.

All three inputs can be either a 3D object position or a camera position
(extracted via frame index + extrinsics).
"""

from __future__ import annotations

import numpy as np

from tools.code_execution.distance_computation import get_camera_position


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
    else:
        lr_angle = float(np.degrees(np.arctan2(-lr_proj, fb_proj)))
        fb_angle = float(np.degrees(np.arctan2(fb_proj, abs(lr_proj))))

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
        "distance_ref_to_target": ref_tgt_dist,
    }
