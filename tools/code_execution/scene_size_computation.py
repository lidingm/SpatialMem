"""Scene size computation from metric point cloud.

Filters out sparse/low-confidence outliers, then computes the scene bounding box
dimensions from the cleaned point cloud.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_scene_size(
    points: np.ndarray,
    conf: np.ndarray | None = None,
    conf_percentile: float = 10.0,
    spatial_percentile: float = 2.0,
    output_dir: str | Path | None = None,
) -> dict[str, np.ndarray | float]:
    """基于 metric 点云计算场景尺寸。

    过滤流程：
      1. 去除非有限值点
      2. 如果有置信度，去掉低于 conf_percentile 分位数的低置信点
      3. 按 spatial_percentile 裁剪空间离群点（每轴去掉两端极值）

    Args:
        points:              (N, 3) 世界坐标点云
        conf:                (N,) 置信度，None 则跳过置信度过滤
        conf_percentile:     置信度过滤分位数，低于此百分位的点被去除
        spatial_percentile:  空间裁剪分位数，每轴两端各去掉此百分比

    Returns:
        dict with keys:
          extent_xyz    - (3,) 场景 X/Y/Z 方向尺寸
          bbox_min      - (3,) 裁剪后包围盒最小值
          bbox_max      - (3,) 裁剪后包围盒最大值
          floor_area    - float, 水平面积估计 (X * Z)
          height        - float, 竖直方向高度 (Y)
          n_points_raw  - int, 原始点数
          n_points_clean- int, 过滤后点数
    """
    n_raw = len(points)

    # 1. 去除非有限值
    finite = np.isfinite(points).all(axis=1)
    pts = points[finite]
    c = conf[finite] if conf is not None else None

    # 2. 置信度过滤
    if c is not None and len(c) > 0:
        thr = float(np.percentile(c[np.isfinite(c)], conf_percentile))
        mask = c >= thr
        pts = pts[mask]

    # 3. 空间离群点裁剪
    if len(pts) > 0 and spatial_percentile > 0:
        lo = np.percentile(pts, spatial_percentile, axis=0)
        hi = np.percentile(pts, 100.0 - spatial_percentile, axis=0)
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts = pts[mask]
        bbox_min = lo
        bbox_max = hi
    elif len(pts) > 0:
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
    else:
        bbox_min = np.zeros(3)
        bbox_max = np.zeros(3)

    extent = bbox_max - bbox_min
    # DA3 坐标系: Y 轴朝下，X 水平，Z 深度方向
    floor_area = float(extent[0] * extent[2])
    height = float(extent[1])

    result = {
        "extent_xyz": extent,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "floor_area": floor_area,
        "height": height,
        "n_points_raw": n_raw,
        "n_points_clean": len(pts),
    }

    if output_dir is not None:
        _save_scene_size(result, output_dir)

    return result


def _save_scene_size(result: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "scene_size.npz",
        extent_xyz=result["extent_xyz"],
        bbox_min=result["bbox_min"],
        bbox_max=result["bbox_max"],
    )

    metadata = {
        "extent_xyz": result["extent_xyz"].tolist(),
        "bbox_min": result["bbox_min"].tolist(),
        "bbox_max": result["bbox_max"].tolist(),
        "floor_area_m2": result["floor_area"],
        "height_m": result["height"],
        "n_points_raw": result["n_points_raw"],
        "n_points_clean": result["n_points_clean"],
    }
    (output_dir / "scene_size_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_scene_size(result: dict) -> None:
    """打印场景尺寸信息。"""
    ext = result["extent_xyz"]
    print(f'场景尺寸 (过滤后 {result["n_points_clean"]}/{result["n_points_raw"]} 点):')
    print(f'  X (宽):  {ext[0]:.2f} m')
    print(f'  Y (高):  {ext[1]:.2f} m')
    print(f'  Z (深):  {ext[2]:.2f} m')
    print(f'  水平面积: {result["floor_area"]:.2f} m²')
    print(f'  层高:     {result["height"]:.2f} m')
    print(f'  包围盒:   ({result["bbox_min"][0]:.2f}, {result["bbox_min"][1]:.2f}, {result["bbox_min"][2]:.2f})')
    print(f'          → ({result["bbox_max"][0]:.2f}, {result["bbox_max"][1]:.2f}, {result["bbox_max"][2]:.2f})')
