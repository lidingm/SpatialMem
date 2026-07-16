"""Distance computation between objects and/or camera positions."""

from __future__ import annotations

import numpy as np


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
