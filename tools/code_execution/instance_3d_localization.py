"""3D instance localization: back-project 2D regions to 3D, then cluster by 3D proximity.

Accepts two kinds of per-frame detections, in the same unified format:
  - SAM-style: precise boolean masks (from object_segmentation)
  - Annotation-style: bounding boxes / points given directly by the sample
    (no SAM needed — when the sample already provides exact 2D locations,
    skip segmentation entirely and back-project the annotated region directly)

Per-frame detection (no cross-frame obj_ids), so we cluster 3D centroids
across frames to identify unique instances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _as_homogeneous(ext: np.ndarray) -> np.ndarray:
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        out = np.eye(4, dtype=ext.dtype)
        out[:3, :4] = ext
        return out
    raise ValueError(f"Expected (3,4) or (4,4), got {ext.shape}")


def _box_to_mask(box, h, w):
    """Fill a boolean mask for the given xyxy pixel box (already at image resolution)."""
    mask = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, int(round(x0))))
    x1 = max(0, min(w, int(round(x1)) + 1))
    y0 = max(0, min(h - 1, int(round(y0))))
    y1 = max(0, min(h, int(round(y1)) + 1))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _back_project_region(mask, box, d_map, K, c2w, d_h, d_w):
    """Back-project a region (mask if available, else box) into 3D world points.

    Args:
        mask: (H, W) bool array at the ORIGINAL image resolution, or None.
        box: (4,) xyxy pixel box at the ORIGINAL image resolution, used when
             mask is None, or to know the original resolution for rescaling.
    """
    if mask is not None:
        m_h, m_w = mask.shape
        if (m_h, m_w) != (d_h, d_w):
            from PIL import Image as _Img
            mask = np.array(_Img.fromarray(mask).resize((d_w, d_h), _Img.NEAREST))
    else:
        # Rescale box from original resolution to depth map resolution.
        # box and an implicit original (H, W) must already match the frame
        # the depth map corresponds to; callers pass boxes already in the
        # original image's pixel space, and depth maps may be a different
        # resolution, so rescale proportionally using d_h/d_w vs caller-known
        # original size is handled by the caller (box is pre-scaled there).
        mask = _box_to_mask(box, d_h, d_w)

    valid = mask & np.isfinite(d_map) & (d_map > 0)
    if not valid.any():
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    vs, us = np.where(valid)
    ds = d_map[vs, us]
    pts_cam = np.stack([(us - cx) / fx * ds, (vs - cy) / fy * ds, ds, np.ones_like(ds)], axis=1)
    return (c2w @ pts_cam.T).T[:, :3]


def _compute_3d_bbox(pts, percentile=2.0):
    lo = np.percentile(pts, percentile, axis=0)
    hi = np.percentile(pts, 100.0 - percentile, axis=0)
    return lo, hi, hi - lo


def _aabb_iou_ios(min_a, max_a, min_b, max_b):
    """3D axis-aligned box overlap, returned as (IoU, IoS).

    IoU = intersection / union                 (symmetric volumetric overlap)
    IoS = intersection / min(vol_a, vol_b)     (how much of the SMALLER box is
                                                inside the other — catches a
                                                partial view contained in a
                                                fuller one)
    """
    lo = np.maximum(min_a, min_b)
    hi = np.minimum(max_a, max_b)
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    if inter <= 0.0:
        return 0.0, 0.0
    va = float(np.prod(np.clip(max_a - min_a, 0.0, None)))
    vb = float(np.prod(np.clip(max_b - min_b, 0.0, None)))
    union = va + vb - inter
    iou = inter / union if union > 0.0 else 0.0
    smaller = min(va, vb)
    ios = inter / smaller if smaller > 0.0 else 0.0
    return iou, ios


def annotations_to_detections(
    annotations: dict,
    n_frames: int,
    image_size: tuple[int, int],
    frame_index: int = 0,
    point_radius: int = 15,
) -> list[dict[str, np.ndarray]]:
    """Convert SPAR-style annotations (red_point, blue_bbox, ...) into the
    same per-frame detection format used by compute_instance_3d_positions,
    so annotated objects can be localized WITHOUT running SAM.

    Each annotation key becomes one detection, placed at `frame_index`
    (all other frames get zero detections). Points are expanded into a
    small square box of side `2 * point_radius` so the same back-projection
    code path can be reused for both points and boxes.

    Args:
        annotations: dict like {"red_point": [[x, y]], "blue_bbox": [[x0,y0,x1,y1]]}
        n_frames: total number of frames (to build a full per-frame list)
        image_size: (width, height) of the image the annotation coordinates are in
        frame_index: which frame the annotations belong to (usually 0)
        point_radius: half-width in pixels of the box synthesized around a point

    Returns:
        list[dict] in the same format as object_segmentation output, but
        WITHOUT a "masks" key (boxes-only) and with "labels" giving the
        original annotation key (e.g. "red_point") for each detection,
        in a stable order (sorted by key) so instance IDs are predictable.
    """
    img_w, img_h = image_size
    keys = sorted(annotations.keys())  # stable order: blue_bbox, blue_point, red_bbox, red_point, ...

    boxes = []
    labels = []
    for key in keys:
        val = annotations[key]
        if not val:
            continue
        v = val[0] if (isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple))) else val

        if key.endswith("_point") and len(v) >= 2:
            x, y = float(v[0]), float(v[1])
            boxes.append([x - point_radius, y - point_radius, x + point_radius, y + point_radius])
            labels.append(key)
        elif key.endswith("_bbox") and len(v) == 4:
            boxes.append([float(c) for c in v])
            labels.append(key)

    n_det = len(boxes)
    boxes_arr = np.clip(np.array(boxes, dtype=np.float32), [0, 0, 0, 0], [img_w, img_h, img_w, img_h]) \
        if n_det > 0 else np.empty((0, 4), dtype=np.float32)
    scores_arr = np.ones(n_det, dtype=np.float32)

    detections = []
    for fi in range(n_frames):
        if fi == frame_index and n_det > 0:
            detections.append({"boxes": boxes_arr, "scores": scores_arr, "labels": labels,
                               "image_size": image_size})
        else:
            detections.append({"boxes": np.empty((0, 4), dtype=np.float32),
                               "scores": np.empty((0,), dtype=np.float32), "labels": [],
                               "image_size": image_size})
    return detections


def compute_instance_3d_positions(
    seg_results: list[dict[str, np.ndarray]],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    distance_threshold: float = 0.5,
    bbox_percentile: float = 2.0,
    iou_threshold: float = 0.30,
    ios_threshold: float = 0.60,
    ios_threshold_small: float = 0.30,
    ios_small_diag: float = 0.5,
    ios_large_diag: float = 1.2,
    centroid_frac: float = 0.30,
    output_dir: str | Path | None = None,
) -> dict:
    """Back-project per-frame detections to 3D, then merge into unique instances.

    Works with two kinds of input detections (per frame, in `seg_results`):
      - precise masks: dict has a "masks" key, (N, H_img, W_img) bool array
      - boxes only (e.g. from annotations_to_detections, no SAM needed):
        dict has only "boxes" (N, 4) xyxy pixel coords, no "masks" key —
        the box region itself is back-projected.
    Both can be mixed across different object_segmentation/annotation calls
    after merging (see tool_registry._merge_all_seg_results).

    Cross-frame association (two-stage merge):
      Each frame dict may carry an optional "obj_ids" list (one id per
      detection) produced by SAM3.1 *video* tracking. Detections that share an
      obj_id are already known to be the same physical instance, so they are
      merged directly (stage A) — this does most of the work and removes noise.
      Only the remaining, separately-tracked groups are then merged by a strict
      3D-geometry test (stage B), which fixes SAM3's track fragmentation
      (one object split across several obj_ids) without over-merging distinct
      neighbours. Detections with no obj_id (box-only annotations) each start as
      their own group and rely on stage B alone.

    Flow:
      1. For each frame, back-project each detection (mask or box) -> 3D centroid + bbox
      2a. Group detections by video obj_id (trust the tracker).
      2b. Merge groups whose 3D boxes strictly coincide (IoU / IoS / near-coincident
          centroids), same label only.
      3. For each final cluster (= unique instance), take median centroid and union bbox.

    Merge thresholds (stage B), tune for precision/recall of the merge:
      iou_threshold  - min 3D IoU to merge (default 0.30)
      ios_threshold  - min 3D intersection-over-smaller to merge, for LARGE
                       objects (smaller box diagonal >= ios_large_diag). (0.60)
      ios_threshold_small - same, for SMALL objects (diagonal <= ios_small_diag).
                       (0.30) Rationale: a fixed depth-noise displacement eats a
                       larger FRACTION of a small object, dropping its IoS with
                       the parent even when it's the same object — so require
                       less overlap when the smaller object is small. The
                       effective IoS threshold ramps linearly with the smaller
                       box's diagonal between ios_small_diag and ios_large_diag.
      ios_small_diag / ios_large_diag - diagonal (m) endpoints of the ramp
                       (defaults 0.5 and 1.2).
      centroid_frac  - merge if centroid distance <= this * smaller box diagonal
                       (default 0.30; higher merges depth-noise fragments but
                       risks chain-merging distinct nearby objects)
    `distance_threshold` is kept for backward compatibility and is unused.

    Returns:
        dict with keys:
          per_frame     - list[dict] per-frame raw 3D results
          obj_id_list   - list[int] unique instance IDs (0, 1, 2, ...)
          merged_positions - (K, 3) final 3D centroid per instance
          merged_bbox_size - (K, 3) final bbox size per instance
          merged_bbox_min  - (K, 3)
          merged_bbox_max  - (K, 3)
          merged_scores    - (K,) best score per instance
          merged_labels    - (K,) list of source labels (e.g. annotation keys), best-effort
    """
    nan3 = np.full(3, np.nan, dtype=np.float64)
    n_frames = len(seg_results)

    # Step 1: per-frame back-projection
    per_frame = []
    all_centroids = []
    all_bsizes = []
    all_bmins = []
    all_bmaxs = []
    all_scores_flat = []
    all_labels_flat = []
    all_indices = []  # (frame_idx, det_idx)
    all_track_ids = []  # video obj_id per detection (or None)

    for fi in range(n_frames):
        seg = seg_results[fi]
        masks = seg.get("masks")
        boxes = seg.get("boxes")
        labels = seg.get("labels")
        obj_ids = seg.get("obj_ids")
        n_det = len(seg["scores"])

        if n_det == 0:
            per_frame.append({
                "positions_3d": np.empty((0, 3), dtype=np.float64),
                "bbox_min": np.empty((0, 3), dtype=np.float64),
                "bbox_max": np.empty((0, 3), dtype=np.float64),
                "bbox_size": np.empty((0, 3), dtype=np.float64),
                "frame_index": fi,
            })
            continue

        d_map = depth[fi]
        K = intrinsics[fi]
        w2c = _as_homogeneous(extrinsics[fi])
        c2w = np.linalg.inv(w2c)
        d_h, d_w = d_map.shape

        positions = np.zeros((n_det, 3), dtype=np.float64)
        bmins = np.zeros((n_det, 3), dtype=np.float64)
        bmaxs = np.zeros((n_det, 3), dtype=np.float64)
        bsizes = np.zeros((n_det, 3), dtype=np.float64)

        for i in range(n_det):
            mask_i = masks[i] if masks is not None else None
            box_i = _rescale_box(boxes[i], d_w, d_h, seg.get("image_size")) if (mask_i is None and boxes is not None) else None
            pts = _back_project_region(mask_i, box_i, d_map, K, c2w, d_h, d_w)
            if pts is None:
                positions[i] = bmins[i] = bmaxs[i] = bsizes[i] = nan3
                continue
            positions[i] = np.median(pts, axis=0)
            bmins[i], bmaxs[i], bsizes[i] = _compute_3d_bbox(pts, bbox_percentile)

            if not np.any(np.isnan(positions[i])):
                all_centroids.append(positions[i])
                all_bsizes.append(bsizes[i])
                all_bmins.append(bmins[i])
                all_bmaxs.append(bmaxs[i])
                all_scores_flat.append(seg["scores"][i])
                all_labels_flat.append(labels[i] if labels and i < len(labels) else "")
                all_indices.append((fi, i))
                all_track_ids.append(
                    obj_ids[i] if (obj_ids is not None and i < len(obj_ids)) else None
                )

        per_frame.append({
            "positions_3d": positions,
            "bbox_min": bmins, "bbox_max": bmaxs, "bbox_size": bsizes,
            "frame_index": fi,
        })

    # Step 2: 3D bbox-overlap clustering via union-find.
    # Two detections are merged iff their 3D bounding boxes overlap in all three
    # axes. This handles the case where the same large object (e.g. a bed) is
    # only partially visible in each frame: as long as any two partial views
    # overlap in 3D space they are correctly identified as the same instance,
    # even if their centroids are far apart.
    if len(all_centroids) == 0:
        result = {
            "per_frame": per_frame,
            "obj_id_list": [],
            "merged_positions": np.empty((0, 3), dtype=np.float64),
            "merged_bbox_size": np.empty((0, 3), dtype=np.float64),
            "merged_bbox_min": np.empty((0, 3), dtype=np.float64),
            "merged_bbox_max": np.empty((0, 3), dtype=np.float64),
            "merged_scores": np.empty((0,), dtype=np.float64),
            "merged_labels": [],
            "frame_to_global": [np.full(len(s["scores"]), -1, dtype=int) for s in seg_results],
        }
        if output_dir:
            _save_results(result, output_dir)
        return result

    N = len(all_centroids)
    scores = np.array(all_scores_flat)
    bmins_arr = np.array(all_bmins)   # (N, 3)
    bmaxs_arr = np.array(all_bmaxs)   # (N, 3)

    # ── Stage A: trust the video tracker ──────────────────────────────────
    # Detections that share a SAM3 video obj_id are, by construction, the same
    # physical instance across frames — put them in one group up front (their
    # 3D points get merged directly). Detections with no obj_id (box-only
    # annotations, or per-frame detections) each start as their own group.
    group_of = np.empty(N, dtype=int)
    track_to_group: dict = {}
    n_groups = 0
    for i in range(N):
        tid = all_track_ids[i]
        if tid is None:
            group_of[i] = n_groups
            n_groups += 1
        else:
            if tid not in track_to_group:
                track_to_group[tid] = n_groups
                n_groups += 1
            group_of[i] = track_to_group[tid]

    # Aggregate each group's 3D AABB (union of member boxes) and label.
    g_members: list[list[int]] = [[] for _ in range(n_groups)]
    for i in range(N):
        g_members[group_of[i]].append(i)
    g_min = np.stack([bmins_arr[m].min(axis=0) for m in g_members])   # (G, 3)
    g_max = np.stack([bmaxs_arr[m].max(axis=0) for m in g_members])   # (G, 3)
    g_label = [all_labels_flat[max(m, key=lambda x: scores[x])] for m in g_members]
    # ── Stage B: agglomerative merge by a STRICT 3D criterion ─────────────
    # Two clusters (same label only) are the same instance iff their 3D boxes
    # substantially coincide — NOT merely touch. Merge if ANY of:
    #   • 3D IoU  ≥ iou_threshold   (strong symmetric volumetric overlap)
    #   • 3D IoS  ≥ ios_threshold   (smaller box mostly inside the other:
    #                                a partial view contained in a fuller one)
    #   • centroids near-coincident (‖cA−cB‖ ≤ centroid_frac · smaller diag:
    #                                noisy fragments at the same 3D location)
    # Unlike a static pairwise pass, this is AGGLOMERATIVE: after two clusters
    # merge, their boxes are unioned and the combined cluster is compared, as a
    # whole, against the rest. So an object close to a cluster's overall extent
    # (but not to any single original member) still gets absorbed. Repeats until
    # no pair merges.
    clusters = [
        {"members": list(g_members[g]),
         "min": g_min[g].copy(),
         "max": g_max[g].copy(),
         "label": g_label[g]}
        for g in range(n_groups)
    ]

    def _clusters_merge(a: dict, b: dict) -> bool:
        if a["label"] != b["label"]:
            return False
        iou, ios = _aabb_iou_ios(a["min"], a["max"], b["min"], b["max"])
        ca = 0.5 * (a["min"] + a["max"])
        cb = 0.5 * (b["min"] + b["max"])
        diag_a = float(np.linalg.norm(a["max"] - a["min"]))
        diag_b = float(np.linalg.norm(b["max"] - b["min"]))
        smaller = min(diag_a, diag_b)
        # Size-adaptive IoS threshold: a fixed depth-noise displacement is a
        # bigger fraction of a small object, so its true-same-object IoS is lower.
        # Ramp the required IoS with the smaller box's diagonal.
        t = np.clip(
            (smaller - ios_small_diag) / max(ios_large_diag - ios_small_diag, 1e-6),
            0.0, 1.0,
        )
        ios_thr = ios_threshold_small + t * (ios_threshold - ios_threshold_small)
        cdist = float(np.linalg.norm(ca - cb))
        coincident = cdist <= centroid_frac * smaller
        return iou >= iou_threshold or ios >= ios_thr or coincident

    merged_any = True
    while merged_any:
        merged_any = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if _clusters_merge(clusters[a], clusters[b]):
                    clusters[a]["members"].extend(clusters[b]["members"])
                    clusters[a]["min"] = np.minimum(clusters[a]["min"], clusters[b]["min"])
                    clusters[a]["max"] = np.maximum(clusters[a]["max"], clusters[b]["max"])
                    del clusters[b]
                    merged_any = True
                    break
            if merged_any:
                break

    cluster_members: list[list[int]] = [c["members"] for c in clusters]
    det_to_cluster = {}
    for ci, members in enumerate(cluster_members):
        for m in members:
            det_to_cluster[m] = ci

    # Step 3: merge per cluster — union bbox, median centroid, best score/label
    centroids = np.array(all_centroids)
    k = len(cluster_members)
    merged_positions = np.zeros((k, 3), dtype=np.float64)
    merged_bbox_min  = np.zeros((k, 3), dtype=np.float64)
    merged_bbox_max  = np.zeros((k, 3), dtype=np.float64)
    merged_bbox_size = np.zeros((k, 3), dtype=np.float64)
    merged_scores    = np.zeros(k, dtype=np.float64)
    merged_labels    = []

    for ci, members in enumerate(cluster_members):
        merged_positions[ci] = np.median(centroids[members], axis=0)
        # Union of all individual bboxes → tightest box covering every detection
        merged_bbox_min[ci]  = np.min(bmins_arr[members], axis=0)
        merged_bbox_max[ci]  = np.max(bmaxs_arr[members], axis=0)
        merged_bbox_size[ci] = merged_bbox_max[ci] - merged_bbox_min[ci]
        merged_scores[ci]    = max(scores[m] for m in members)
        best_m = max(members, key=lambda m: scores[m])
        merged_labels.append(all_labels_flat[best_m])

    # Build frame_to_global mapping
    frame_to_global = [np.full(len(s["scores"]), -1, dtype=int) for s in seg_results]
    for det_idx, (fi, local_i) in enumerate(all_indices):
        frame_to_global[fi][local_i] = det_to_cluster[det_idx]
    cluster_id = np.array([det_to_cluster[i] for i in range(N)], dtype=int)

    result = {
        "per_frame": per_frame,
        "obj_id_list": list(range(k)),
        "merged_positions": merged_positions,
        "merged_bbox_size": merged_bbox_size,
        "merged_bbox_min": merged_bbox_min,
        "merged_bbox_max": merged_bbox_max,
        "merged_scores": merged_scores,
        "merged_labels": merged_labels,
        "frame_to_global": frame_to_global,
    }

    if output_dir:
        _save_results(result, output_dir)
    return result


def _rescale_box(box, d_w, d_h, image_size):
    """Rescale a box from its original image resolution to the depth map resolution."""
    if image_size is None:
        return box
    img_w, img_h = image_size
    sx, sy = d_w / img_w, d_h / img_h
    x0, y0, x1, y1 = box
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]


def _save_results(result: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "instance_3d_positions.npz",
        merged_positions=result["merged_positions"],
        merged_bbox_size=result["merged_bbox_size"],
        merged_bbox_min=result["merged_bbox_min"],
        merged_bbox_max=result["merged_bbox_max"],
        merged_scores=result["merged_scores"],
    )
    metadata = {
        "total_instances": len(result["obj_id_list"]),
        "per_instance": [
            {"id": i,
             "center": result["merged_positions"][i].tolist(),
             "bbox_size": result["merged_bbox_size"][i].tolist(),
             "best_score": float(result["merged_scores"][i]),
             "label": result["merged_labels"][i] if i < len(result.get("merged_labels", [])) else ""}
            for i in result["obj_id_list"]
        ],
    }
    (output_dir / "instance_3d_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_instance_3d_summary(result: dict, text_prompt: str) -> None:
    k = len(result["obj_id_list"])
    print(f'"{text_prompt}": {k} unique instances (merged by 3D proximity)\n')
    for i in range(k):
        pos = result["merged_positions"][i]
        size = result["merged_bbox_size"][i]
        sc = result["merged_scores"][i]
        label = result.get("merged_labels", [""] * k)[i]
        label_str = f"  label={label}" if label else ""
        print(f'  Instance {i}: center=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  '
              f'bbox=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})  score={sc:.2f}{label_str}')
