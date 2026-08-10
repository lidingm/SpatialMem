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


def _maybe_annotate(result, seg_results, frame_paths, object_name, output_dir) -> None:
    """Render the merged instances onto the frames (best-effort).

    Called after every 3D clustering run so the annotated frames are always
    available downstream (e.g. for the VLM count calibration). Never fatal: a
    rendering problem must not take down the localization result.
    """
    if frame_paths is None or not object_name or not output_dir:
        return
    try:
        ann = draw_instance_annotations(seg_results, result, frame_paths,
                                        object_name, output_dir)
        result["annotated_frames"] = ann["annotated_frames"]
        result["instances"] = ann["instances"]
        result["annotated_dir"] = ann["dir"]
        result["annotated_frames_by_category"] = ann.get("annotated_frames_by_category", {})
        result["instances_by_category"] = ann.get("instances_by_category", {})
        result["annotated_dir_by_category"] = ann.get("annotated_dir_by_category", {})
    except Exception as e:  # noqa: BLE001 - annotation is a nice-to-have
        print(f"[instance_3d_localization] annotation skipped: {e}")


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
    frame_paths: Sequence[str] | None = None,
    object_name: str | None = None,
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
        # still render the (box-free) frames so downstream can see "nothing found"
        _maybe_annotate(result, seg_results, frame_paths, object_name, output_dir)
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
    # (but not to any single original member) still gets absorbed.
    #
    # Two hard rules on top of the geometry:
    #   (1) Clusters that CO-OCCUR in the same frame are NEVER merged — if two
    #       detections show up in one image they are, by definition, distinct
    #       physical objects, no matter how close in 3D.
    #   (2) Each round merges the globally CLOSEST eligible pair (min centroid
    #       distance), not the first found. So when a track is eligible with two
    #       clusters, it joins the NEARER one; its frames then fold in, and if the
    #       other cluster co-occurs with it, rule (1) blocks that second merge.
    clusters = [
        {"members": list(g_members[g]),
         "min": g_min[g].copy(),
         "max": g_max[g].copy(),
         "label": g_label[g],
         "frames": {all_indices[m][0] for m in g_members[g]}}
        for g in range(n_groups)
    ]

    def _merge_dist(a: dict, b: dict):
        """Return centroid distance if a,b MAY merge, else None."""
        if a["label"] != b["label"]:
            return None
        if a["frames"] & b["frames"]:          # rule (1): co-occur in a frame
            return None
        iou, ios = _aabb_iou_ios(a["min"], a["max"], b["min"], b["max"])
        ca = 0.5 * (a["min"] + a["max"])
        cb = 0.5 * (b["min"] + b["max"])
        diag_a = float(np.linalg.norm(a["max"] - a["min"]))
        diag_b = float(np.linalg.norm(b["max"] - b["min"]))
        smaller = min(diag_a, diag_b)
        # Size-adaptive IoS threshold: a fixed depth-noise displacement is a
        # bigger fraction of a small object, so its true-same-object IoS is lower.
        t = np.clip(
            (smaller - ios_small_diag) / max(ios_large_diag - ios_small_diag, 1e-6),
            0.0, 1.0,
        )
        ios_thr = ios_threshold_small + t * (ios_threshold - ios_threshold_small)
        cdist = float(np.linalg.norm(ca - cb))
        if iou >= iou_threshold or ios >= ios_thr or cdist <= centroid_frac * smaller:
            return cdist
        return None

    while True:
        best = None  # (dist, a_idx, b_idx)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = _merge_dist(clusters[a], clusters[b])
                if d is not None and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best  # merge the closest eligible pair (rule (2))
        clusters[a]["members"].extend(clusters[b]["members"])
        clusters[a]["min"] = np.minimum(clusters[a]["min"], clusters[b]["min"])
        clusters[a]["max"] = np.maximum(clusters[a]["max"], clusters[b]["max"])
        clusters[a]["frames"] |= clusters[b]["frames"]
        del clusters[b]

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
    _maybe_annotate(result, seg_results, frame_paths, object_name, output_dir)
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


# ─── Instance annotation rendering ──────────────────────────────────────────

# Distinct, human-nameable colours, so a downstream VLM can be told
# "instance 1 is the red box".
_PALETTE = [
    ("red", (230, 25, 75)), ("blue", (0, 130, 200)), ("green", (60, 180, 75)),
    ("orange", (245, 130, 48)), ("purple", (145, 30, 180)), ("cyan", (70, 240, 240)),
    ("magenta", (240, 50, 230)), ("yellow", (255, 215, 0)), ("lime", (170, 255, 0)),
    ("teal", (0, 128, 128)), ("pink", (255, 150, 200)), ("brown", (154, 99, 36)),
]


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Largest connected component of a boolean mask.

    Stray speckles far from the object would otherwise blow the min/max box up
    to many times the visible region. Returns the mask unchanged if scipy is
    unavailable or there is only one component.
    """
    try:
        from scipy import ndimage
    except Exception:
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def draw_instance_annotations(
    seg_results: list[dict],
    result: dict,
    frame_paths: Sequence[str],
    object_name: str,
    output_dir: str | Path,
) -> dict:
    """Draw merged instances onto category-specific annotated frame sets.

    Frames are written to `output_dir/<category>/frame-XX.png`. When a 3D
    localization run contains multiple object prompts, each category gets its
    own clean 32-frame overlay instead of one crowded all-object overlay.

    Returns:
        dict with compatibility fields (`annotated_frames`, `instances`, `dir`)
        plus category-indexed fields:
          annotated_frames_by_category: {category: [frame paths]}
          instances_by_category: {category: [instance dicts]}
          annotated_dir_by_category: {category: dir}
    """
    from PIL import Image as _Img, ImageDraw, ImageFont

    def _safe_name(name: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                       for ch in str(name).strip().lower()) or "object"

    ids = list(result.get("obj_id_list", []))
    id_color = {oid: _PALETTE[i % len(_PALETTE)][1] for i, oid in enumerate(ids)}
    id_cname = {oid: _PALETTE[i % len(_PALETTE)][0] for i, oid in enumerate(ids)}

    # Label each instance by ITS OWN category (seg_results may merge several
    # prompts, e.g. "sofa" + "stove"), numbering restarts per category so the
    # labels read "sofa 1", "sofa 2", "stove 1" rather than one shared counter.
    merged_labels = result.get("merged_labels") or []

    def _cat_of(oid: int) -> str:
        lab = merged_labels[oid] if oid < len(merged_labels) else ""
        return str(lab) if lab else str(object_name)

    _seen: dict[str, int] = {}
    id_label, id_cat = {}, {}
    for oid in ids:
        cat = _cat_of(oid)
        _seen[cat] = _seen.get(cat, 0) + 1
        id_cat[oid] = cat
        id_label[oid] = f"{cat} {_seen[cat]}"

    categories = list(dict.fromkeys(id_cat[oid] for oid in ids))
    if not categories:
        categories = [str(object_name)]

    try:
        import matplotlib
        font = ImageFont.truetype(
            str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 15)
    except Exception:
        font = ImageFont.load_default()

    f2g = result.get("frame_to_global") or []
    annotated_by_cat: dict[str, list[str]] = {cat: [] for cat in categories}
    dir_by_cat: dict[str, str] = {}
    id_frames: dict[int, list[int]] = {oid: [] for oid in ids}
    # per-instance 2D box set — kept so downstream (VLM calibration, reflector,
    # skill distillation) can reason about WHERE each instance was seen, not just
    # how many there were.
    id_boxes: dict[int, list[dict]] = {oid: [] for oid in ids}

    output_dir = Path(output_dir)
    for cat in categories:
        out_dir = output_dir / _safe_name(cat)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("frame-*.png"):
            stale.unlink()
        dir_by_cat[cat] = str(out_dir)

        for fi, path in enumerate(frame_paths):
            img = _Img.open(path).convert("RGB")
            dr = ImageDraw.Draw(img)
            if fi < len(seg_results) and fi < len(f2g):
                seg = seg_results[fi]
                masks = seg.get("masks")
                for li in range(len(seg.get("scores", []))):
                    gid = int(f2g[fi][li])
                    if gid < 0 or masks is None or id_cat.get(gid) != cat:
                        continue
                    mask = np.asarray(masks[li], bool)
                    if not mask.any():
                        continue
                    ys, xs = np.where(_largest_cc(mask))
                    x0, y0 = max(int(xs.min()), 0), max(int(ys.min()), 0)
                    x1 = min(int(xs.max()), img.width - 1)
                    y1 = min(int(ys.max()), img.height - 1)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    c = id_color[gid]
                    dr.rectangle([x0, y0, x1, y1], outline=c, width=1)
                    dr.text((x0 + 2, max(y0 - 17, 0)), id_label[gid], fill=c, font=font)
                    id_frames[gid].append(fi)
                    id_boxes[gid].append({"frame": int(fi), "box": [x0, y0, x1, y1]})
            fp = out_dir / f"frame-{fi:02d}.png"
            img.save(fp)
            annotated_by_cat[cat].append(str(fp))

    instances = [{
        "label": id_label[oid],
        "category": id_cat[oid],
        "id": int(oid),
        "color": id_cname[oid],
        "frames": sorted(set(id_frames[oid])),
        "boxes_2d": id_boxes[oid],          # [{"frame": i, "box": [x0,y0,x1,y1]}, ...]
        "bbox_size_m": [round(float(v), 3) for v in result["merged_bbox_size"][oid]],
        "center_3d_m": [round(float(v), 3) for v in result["merged_positions"][oid]],
        "score": round(float(result["merged_scores"][oid]), 3),
    } for oid in ids]

    instances_by_cat = {
        cat: [it for it in instances if it["category"] == cat]
        for cat in categories
    }
    for cat in categories:
        cat_dir = Path(dir_by_cat[cat])
        cat_instances = instances_by_cat[cat]
        (cat_dir / "instances.json").write_text(
            json.dumps({"object_name": cat, "n_instances": len(cat_instances),
                        "instances": cat_instances}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    compat_cat = str(object_name) if str(object_name) in annotated_by_cat else categories[0]
    return {
        "annotated_frames": annotated_by_cat.get(compat_cat, []),
        "instances": instances,
        "dir": dir_by_cat.get(compat_cat, ""),
        "annotated_frames_by_category": annotated_by_cat,
        "instances_by_category": instances_by_cat,
        "annotated_dir_by_category": dir_by_cat,
    }


def draw_final_localization_annotations(
    results_3d: dict,
    frame_paths: Sequence[str],
    final_locs: dict[str, dict],
    output_dir: str | Path,
    geometry_by_id: dict[int, dict] | None = None,
) -> dict:
    """Draw the Checker-validated final target localization on one 32-frame set.

    Unlike ``draw_instance_annotations`` this is NOT category-separated and does
    not show every candidate. It overlays only the final selected instance for
    each target object, and only on the Checker-approved ``keep_frames``.
    """
    from PIL import Image as _Img, ImageDraw, ImageFont

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("frame-*.png"):
        stale.unlink()

    try:
        import matplotlib
        font = ImageFont.truetype(
            str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 15)
    except Exception:
        font = ImageFont.load_default()

    instances = results_3d.get("instances") or []
    by_id = {int(it["id"]): it for it in instances if "id" in it}
    geometry_by_id = geometry_by_id or {}
    selected: list[dict] = []
    for idx, (name, loc) in enumerate(final_locs.items()):
        if loc is None or loc.get("instance_id") is None:
            continue
        oid = int(loc["instance_id"])
        inst = loc if loc.get("boxes_2d") is not None else by_id.get(oid)
        if inst is None:
            continue
        keep = {int(f) for f in (loc.get("keep_frames") or inst.get("frames") or [])}
        geom = geometry_by_id.get(oid) or {}
        center = geom.get("avg_pos") if geom else inst.get("center_3d_m")
        bbox_size = geom.get("bbox_size") if geom else inst.get("bbox_size_m")
        color_name, color = _PALETTE[idx % len(_PALETTE)]
        label = str(loc.get("instance_label") or inst.get("label") or name)
        selected.append({
            "object": str(name),
            "instance_id": oid,
            "instance_label": label,
            "color": color_name,
            "keep_frames": sorted(keep),
            "dropped_frames": list(loc.get("dropped_frames") or []),
            "reason": str(loc.get("reason") or ""),
            "center_3d_m": [round(float(v), 3) for v in center] if center is not None else None,
            "bbox_size_m": [round(float(v), 3) for v in bbox_size] if bbox_size is not None else None,
            "boxes_2d": [b for b in inst.get("boxes_2d", [])
                         if int(b.get("frame", -1)) in keep],
            "_rgb": color,
        })

    frames_out: list[str] = []
    for fi, path in enumerate(frame_paths):
        img = _Img.open(path).convert("RGB")
        dr = ImageDraw.Draw(img)
        for item in selected:
            for box_item in item["boxes_2d"]:
                if int(box_item.get("frame", -1)) != fi:
                    continue
                box = [int(v) for v in box_item.get("box", [])]
                if len(box) != 4:
                    continue
                x0, y0, x1, y1 = box
                c = item["_rgb"]
                text = (
                    item["instance_label"]
                    if str(item["object"]) == str(item["instance_label"])
                    else f"{item['object']}: {item['instance_label']}"
                )
                dr.rectangle([x0, y0, x1, y1], outline=c, width=3)
                dr.text((x0 + 2, max(y0 - 17, 0)), text, fill=c, font=font)
        fp = output_dir / f"frame-{fi:02d}.png"
        img.save(fp)
        frames_out.append(str(fp))

    clean_selected = [{k: v for k, v in item.items() if k != "_rgb"} for item in selected]
    meta = {
        "dir": str(output_dir),
        "annotated_frames": frames_out,
        "n_targets": len(clean_selected),
        "targets": clean_selected,
    }
    (output_dir / "final_localizations.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def back_project_instance(
    seg_results: list[dict],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    cluster_id: int,
    frame_to_global,
    frames: Sequence[int] | None = None,
    bbox_percentile: float = 2.0,
    dejitter: bool = True,
) -> dict | None:
    """Back-project ONE clustered instance to 3D, over all or a subset of frames.

    Reusable by any task (distance / direction / size): given the per-frame
    detections and the cluster id, recompute the instance's 3D position and
    bounding box from scratch — optionally restricting to `frames` (e.g. the
    frames a VLM deemed reliable). `frames=None` uses every frame the cluster
    appears in (the default, so callers that don't filter get the natural result).

    dejitter=True: each frame's points are recentered to their own centroid
    before pooling, so cross-frame position jitter does not inflate the box
    (pooling raw points would make the box ≈ true_size + jitter_range, which
    pulls closest-point distances too small). The box size then reflects the
    object's true extent; it is placed at the average of the per-frame centroids.
    dejitter=False: plain pooled AABB over the raw points.

    Returns dict(avg_pos, bbox_min, bbox_max, bbox_size, n_points, frames_used)
    or None if no valid 3D points were produced.
    """
    want = None if frames is None else {int(f) for f in frames}
    per_frame_cent, pooled_pts, used = [], [], []
    n_frames = len(seg_results)

    for fi in range(n_frames):
        if want is not None and fi not in want:
            continue
        if fi >= len(frame_to_global):
            continue
        seg = seg_results[fi]
        d_map, K = depth[fi], intrinsics[fi]
        d_h, d_w = d_map.shape
        c2w = np.linalg.inv(_as_homogeneous(extrinsics[fi]))
        masks, boxes = seg.get("masks"), seg.get("boxes")
        f2g_fi = frame_to_global[fi]
        for li in range(len(seg.get("scores", []))):
            if int(f2g_fi[li]) != int(cluster_id):
                continue
            mask_i = masks[li] if masks is not None else None
            box_i = (_rescale_box(boxes[li], d_w, d_h, seg.get("image_size"))
                     if (mask_i is None and boxes is not None) else None)
            pts = _back_project_region(mask_i, box_i, d_map, K, c2w, d_h, d_w)
            if pts is None or not len(pts):
                continue
            c = np.median(pts, axis=0)
            per_frame_cent.append(c)
            pooled_pts.append(pts - c if dejitter else pts)
            used.append(fi)

    if not pooled_pts:
        return None

    avg_pos = np.mean(per_frame_cent, axis=0)
    pooled = np.concatenate(pooled_pts, axis=0)
    lo_s, hi_s, _ = _compute_3d_bbox(pooled, bbox_percentile)
    if dejitter:
        bmin, bmax = avg_pos + lo_s, avg_pos + hi_s   # shape box re-placed at centroid
    else:
        bmin, bmax = lo_s, hi_s                        # already absolute
    return {
        "avg_pos": avg_pos,
        "bbox_min": bmin,
        "bbox_max": bmax,
        "bbox_size": bmax - bmin,
        "n_points": int(len(pooled)),
        "frames_used": sorted(set(used)),
    }
