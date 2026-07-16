"""Tool registry: register, describe, and dispatch all spatial reasoning tools.

Migrated from the old `tool_registry.py`. All hardcoded paths now come from
`agent.config`. Semantics of the ten tools are unchanged.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import OUTPUTS_DIR


# When SPATIALMEM_DEBUG_TRACES=1, tool failures print full tracebacks to stderr.
# Off by default: failures are always captured in the returned dict, and
# noisy tracebacks pollute stderr during normal reflection loops (tools
# routinely fail on missing dependencies, that's how the agent probes state).
_DEBUG_TRACES = os.getenv("SPATIALMEM_DEBUG_TRACES", "0") == "1"


TOOL_DESCRIPTIONS = """=== Available Tools ===

You have 10 tools in two categories. Choose ONLY the tools needed for the question.
Dependencies between tools are handled automatically at runtime — you only need to plan the logical sequence.

────────────────────────────────────────────
[A] Visual Generation Tools — produce visual/geometric evidence
────────────────────────────────────────────

1. depth_estimation
   What it does: Runs DA3 model on ALL input video frames at once. Produces metric depth maps (in meters), camera intrinsics (3x3), camera extrinsics (world-to-camera 3x4 poses), per-pixel confidence maps, and a dense colored 3D point cloud by back-projecting all depth pixels.
   When to use: Any task requiring 3D geometry — distances, sizes, directions, scene layout, BEV, NVS. If ANY downstream tool needs depth/poses/point-cloud, include this.
   Parameters: none (auto-processes all sample frames)
   Output in context:
     - depth: (N, H, W) float array, metric depth per pixel
     - intrinsics: (N, 3, 3) camera intrinsic matrices
     - extrinsics: (N, 3, 4) world-to-camera pose matrices
     - is_metric: 1 if output is in real-world meters
     - scale_factor: metric scaling factor applied
     - points: (M, 3) world-space 3D point cloud
   NOT needed for: pure object detection/counting where 3D is unnecessary

2. bev_generation
   What it does: Renders a bird's-eye-view (top-down) color image from the 3D point cloud. Shows the scene layout as viewed from directly above.
   When to use: Questions about spatial layout, relative positions of objects in the floor plan, or understanding the overall room structure.
   Parameters: none
   Output in context: BEV color image with spatial extent in meters
   Depends on: depth_estimation (needs point cloud)

3. novel_view_synthesis
   What it does: Renders the scene from a new camera viewpoint using 3D Gaussian Splatting. You specify which source frame to start from and how to move the virtual camera.
   When to use: Exploring occluded regions, verifying what's behind an object, spatial imagination tasks ("what would you see if you moved to position X").
   Parameters:
     - frame_index (int): which frame's camera to start from (default: 0)
     - movement (str): one of yaw_left, yaw_right, pitch_up, pitch_down, move_left, move_right, move_up, move_down, move_forward, move_backward
     - angle_deg (float): rotation amount in degrees for yaw/pitch (default: 30)
     - distance (float): translation amount in meters for move_* (default: 0.2)
   Output in context: rendered RGB image from the new viewpoint
   Depends on: depth_estimation (needs 3DGS model)

4. object_segmentation
   What it does: Detects all instances of a specified object in each frame independently using SAM3 with a text prompt. Returns per-frame binary masks, bounding boxes (xyxy pixel coords), and confidence scores.
   When to use: Any task that involves specific objects — counting, measuring, locating, comparing objects.
   Parameters:
     - text_prompt (str): a descriptive text prompt to identify the target object. Be as specific as needed to find the RIGHT object.
       * For counting all of a category: use the bare category name (e.g. "chair", "shelf", "door").
       * For a specific object among similar ones: add visual descriptors (e.g. "black office chair", "wooden dining table").
     - object_category (str): the simplified canonical class name for memory bookkeeping.
   Output in context (per frame):
     - masks: (N_det, H, W) boolean masks
     - boxes: (N_det, 4) bounding boxes in xyxy pixel coordinates
     - scores: (N_det,) detection confidence scores
   Does NOT need depth_estimation to run. Can be called independently.

5. annotation_localization
   What it does: Back-projects sample-supplied 2D annotation markers (red_point, blue_bbox, ...) into 3D using the depth map, skipping SAM entirely.
   When to use: The sample has annotation markers AND the question is about those specific marked objects.
   Note: VSIBench samples have no annotations, so this tool is generally not used for VSI.
   Parameters: none
   Depends on: depth_estimation

────────────────────────────────────────────
[B] Code Execution Tools — compute precise numerical results
────────────────────────────────────────────

6. instance_3d_localization
   What it does: For each detected 2D mask, back-projects it into 3D using depth + camera pose. Computes 3D centroid and axis-aligned 3D bounding box per detection, then clusters detections across frames by 3D bounding-box overlap (same object label only) to merge the same physical object into one instance.
   When to use: After detecting objects, to get their 3D positions, sizes, and cross-frame identity.
   Parameters: none
   Output in context:
     - obj_id_list: [0, 1, 2, ...] unique instance IDs after merging
     - merged_positions: (K, 3) final 3D centroid per unique instance
     - merged_bbox_size: (K, 3) width/height/depth of each instance in meters
     - merged_scores: (K,) best detection confidence per instance
     - merged_labels: list of object category labels, indexed by instance ID
     - frame_to_global: mapping from per-frame detection index to global instance ID
   Depends on: depth_estimation AND (object_segmentation OR annotation_localization)

7. distance_computation
   What it does: Metric distance in meters between two objects or an object and the camera.
   Two modes:
     - object_to_object: closest-point distance between two objects' 3D bounding boxes (matches "Measuring from the closest point of each object")
     - object_to_camera: distance between an object centroid and a camera position at a specific frame
   Parameters:
     - mode (str): "object_to_object" or "object_to_camera"
     - obj_a_id (int), obj_b_id (int): instance IDs for object_to_object
     - obj_id (int), frame_index (int): for object_to_camera
   Depends on: instance_3d_localization

8. direction_computation
   What it does: Computes the relative direction of a target object w.r.t. a reference object, as seen from a viewpoint. Projects onto the horizontal plane; returns direction labels (front/back/left/right combinations) and angular offsets.
   How forward direction is determined:
     - If `facing` is given: forward = viewpoint → facing object (use when the person faces a specific object)
     - If `facing` is absent: forward = viewpoint → reference object (default: person implicitly faces the reference)
     - If viewpoint_type="camera": forward = camera's own optical axis toward reference
   Parameters:
     - viewpoint (int): instance ID (if viewpoint_type="object") or frame index (if viewpoint_type="camera") — WHERE the observer stands
     - viewpoint_type (str): "camera" or "object"
     - reference (int): instance ID of the anchor object (direction is relative to this)
     - reference_type (str): "object" (almost always)
     - target (int): instance ID of the object whose direction is asked
     - target_type (str): "object" (almost always)
     - facing (int, optional): instance ID of the object the person is LOOKING AT — defines the forward direction. Required when viewpoint == reference (otherwise forward is undefined). Omit if the person implicitly faces the reference.
     - facing_type (str): "object" (default)
   Depends on: instance_3d_localization, depth_estimation

9. instance_counting
   What it does: Total number of unique object instances after 3D clustering.
   Parameters: none
   Depends on: instance_3d_localization

10. scene_size_computation
   What it does: Overall scene bounding box + floor area from the metric point cloud. Filters low-confidence and outlier points before measuring.
   Parameters: none
   Depends on: depth_estimation
"""


class ToolRegistry:
    def __init__(self, da3_tool=None, sam3_tool=None,
                 output_root: str | Path | None = None):
        self.da3_tool = da3_tool
        self.sam3_tool = sam3_tool
        self.output_root = Path(output_root or OUTPUTS_DIR)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._context: dict[str, Any] = {}
        self._tool_log: list[dict] = []

    def reset_context(self):
        self._context.clear()
        self._tool_log.clear()

    @property
    def context(self) -> dict:
        return self._context

    @property
    def tool_descriptions(self) -> str:
        return TOOL_DESCRIPTIONS

    @property
    def tool_log(self) -> list[dict]:
        return self._tool_log

    def get_context_summary(self) -> str:
        parts: list[str] = []
        if "da3_result" in self._context:
            r = self._context["da3_result"]
            n_frames = r["depth"].shape[0]
            parts.append(f"[depth_estimation] {n_frames} frames, depth shape={r['depth'].shape}, "
                         f"is_metric={r.get('is_metric')}, scale_factor={r.get('scale_factor')}, "
                         f"point_cloud={len(self._context.get('points', []))} points")

        if "bev" in self._context:
            bev = self._context["bev"]
            parts.append(f"[bev_generation] size={bev['raw_size_hw']}, "
                         f"extent=({bev['xy_extent'][0]:.2f}m, {bev['xy_extent'][1]:.2f}m)")

        if "nvs_results" in self._context:
            for nvs in self._context["nvs_results"]:
                parts.append(f"[novel_view_synthesis] {nvs['movement']} {nvs['angle_deg']}deg from frame {nvs['frame_index']}")

        if "all_seg_results" in self._context:
            all_seg = self._context["all_seg_results"]
            for prompt, segs in all_seg.items():
                per_frame = [f"frame{i}:{len(s['scores'])}" for i, s in enumerate(segs) if len(s["scores"]) > 0]
                total_det = sum(len(s["scores"]) for s in segs)
                parts.append(f"[object_segmentation] prompt='{prompt}', {total_det} detections "
                             f"({', '.join(per_frame) if per_frame else 'none'})")

        if "results_3d" in self._context:
            r3d = self._context["results_3d"]
            k = len(r3d.get("obj_id_list", []))
            parts.append(f"[instance_3d_localization] {k} unique instances after 3D clustering:")
            for i in r3d.get("obj_id_list", []):
                pos = r3d["merged_positions"][i]
                size = r3d["merged_bbox_size"][i]
                sc = r3d["merged_scores"][i]
                label = r3d.get("merged_labels", [""] * k)[i]
                label_str = f", source={label}" if label else ""
                parts.append(f"  instance {i}: center=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                             f"bbox_size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})m, "
                             f"score={sc:.2f}{label_str}")

        if "counting" in self._context:
            c = self._context["counting"]
            parts.append(f"[instance_counting] {c['total_unique']} unique instances, ids={c['obj_id_list']}")

        if "scene_size" in self._context:
            ss = self._context["scene_size"]
            ext = ss["extent_xyz"]
            parts.append(f"[scene_size_computation] width={ext[0]:.2f}m, height={ext[1]:.2f}m, "
                         f"depth={ext[2]:.2f}m, floor_area={ss['floor_area']:.2f}m²")

        for entry in self._tool_log:
            if entry.get("tool_name") in ("distance_computation", "direction_computation") \
               and entry.get("success"):
                parts.append(f"[{entry['tool_name']}] {entry.get('result_summary', '')}")

        return "\n".join(parts) if parts else "No tool results yet."

    def execute_tool(self, tool_name: str, params: dict,
                     frame_paths: list[str]) -> dict:
        try:
            result = self._dispatch(tool_name, params, frame_paths)
            entry = {"tool_name": tool_name, "params": params,
                     "success": True, "result_summary": result["summary"],
                     "error": None}
            self._tool_log.append(entry)
            return entry
        except Exception as e:
            entry = {"tool_name": tool_name, "params": params,
                     "success": False, "result_summary": "",
                     "error": str(e)}
            self._tool_log.append(entry)
            if _DEBUG_TRACES:
                traceback.print_exc()
            return entry

    def _dispatch(self, name: str, params: dict, frame_paths: list[str]) -> dict:
        dispatch_map = {
            "depth_estimation":          lambda: self._run_depth(frame_paths),
            "bev_generation":            lambda: self._run_bev(),
            "novel_view_synthesis":      lambda: self._run_nvs(params),
            "object_segmentation":       lambda: self._run_sam3(params, frame_paths),
            "annotation_localization":   lambda: self._run_annotation_loc(frame_paths),
            "instance_3d_localization":  lambda: self._run_3d_loc(),
            "instance_counting":         lambda: self._run_counting(),
            "distance_computation":      lambda: self._run_distance(params),
            "direction_computation":     lambda: self._run_direction(params),
            "scene_size_computation":    lambda: self._run_scene_size(),
        }
        fn = dispatch_map.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        return fn()

    # ─── Tool implementations ─────────────────────────────────────────

    def _run_depth(self, frame_paths):
        out_dir = self.output_root / "da3"
        result = self.da3_tool.run(
            frame_paths=frame_paths, output_dir=out_dir,
            process_res=504, save_depth_vis=True,
        )
        self._context["da3_result"] = result
        self._context["frame_paths"] = frame_paths

        from tools.visual_generation.da3_geometry import prediction_to_point_cloud
        pts, cols = prediction_to_point_cloud(result["prediction"], use_conf=False)
        self._context["points"] = pts
        self._context["colors"] = cols

        n = result["depth"].shape[0]
        return {"summary": f"Depth estimation done: {n} frames, depth shape {result['depth'].shape}, "
                           f"is_metric={result.get('is_metric')}, scale_factor={result.get('scale_factor')}, "
                           f"point cloud: {len(pts)} points"}

    def _run_bev(self):
        pred = self._context.get("da3_result", {}).get("prediction")
        if pred is None:
            raise RuntimeError("depth_estimation must be run first")
        bev = self.da3_tool.make_bev(pred, output_dir=self.output_root / "da3")
        self._context["bev"] = bev
        return {"summary": f"BEV generated: size {bev['raw_size_hw']}, "
                           f"extent x={bev['xy_extent'][0]:.2f}m z={bev['xy_extent'][1]:.2f}m"}

    def _run_nvs(self, params):
        pred = self._context.get("da3_result", {}).get("prediction")
        if pred is None:
            raise RuntimeError("depth_estimation must be run first")
        if pred.gaussians is None:
            pred_gs = self.da3_tool.infer(self._context["frame_paths"], infer_gs=True)
            self._context["da3_result"]["prediction"] = pred_gs
            pred = pred_gs

        fi = params.get("frame_index", 0)
        mov = params.get("movement", "yaw_left")
        ang = params.get("angle_deg", 30.0)
        dist = params.get("distance", 0.2)

        self.da3_tool.render_nvs(
            pred, frame_index=fi, movement=mov,
            angle_deg=ang, distance=dist,
            output_dir=self.output_root / "da3",
        )
        self._context.setdefault("nvs_results", []).append({
            "frame_index": fi, "movement": mov, "angle_deg": ang, "distance": dist,
        })
        return {"summary": f"Novel view rendered: {mov} {ang}deg from frame {fi}"}

    def _run_sam3(self, params, frame_paths):
        text_prompt = params.get("text_prompt", "object")
        object_category = params.get("object_category", "").strip().lower()
        # If annotated images (rendered by orchestrator) exist in ctx, use them;
        # otherwise fall back to raw frame_paths (VSI has no annotations).
        input_paths = self._context.get("annotated_paths", frame_paths)

        # Video-mode segmentation: SAM3.1 tracks instances across all frames, so
        # each per-frame detection carries a cross-frame obj_id (same physical
        # instance shares an id). Downstream 3D localization trusts these ids and
        # only uses 3D geometry to merge the few fragmented tracks — far less
        # work / error than clustering every raw per-frame detection.
        track = self.sam3_tool.segment_video_track(input_paths, text_prompt=text_prompt)
        seg_results = self._video_track_to_seg_results(track, input_paths[0])

        self._context.setdefault("all_seg_results", {})[text_prompt] = seg_results
        if object_category:
            self._context.setdefault("seg_categories", {})[text_prompt] = object_category

        self._context["seg_results"] = self._merge_all_seg_results()
        self._context["text_prompt"] = text_prompt

        total = sum(len(s["scores"]) for s in seg_results)
        n_tracks = len(track.get("unique_obj_ids", []))
        per_frame = [f"frame{i}:{len(s['scores'])}" for i, s in enumerate(seg_results) if len(s["scores"]) > 0]
        return {"summary": f"Object segmentation done: prompt='{text_prompt}', "
                           f"{n_tracks} tracked instance(s), {total} detections across "
                           f"{len(seg_results)} frames ({', '.join(per_frame)})"}

    @staticmethod
    def _video_track_to_seg_results(track, first_frame_path):
        """Convert segment_video_track output into the per-frame seg_results
        format used downstream (masks/boxes/scores/image_size), plus an
        `obj_ids` list per frame carrying the cross-frame track association.

        Masks from the video predictor are at model resolution; resize them
        back to the original image resolution (what _back_project_region and
        the box coords assume).
        """
        from PIL import Image as _Img
        with _Img.open(first_frame_path) as _im:
            orig_w, orig_h = _im.size

        seg_results = []
        for fr in track["per_frame"]:
            n = len(fr["scores"])
            if n == 0:
                seg_results.append({
                    "masks": np.empty((0, orig_h, orig_w), dtype=bool),
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "scores": np.empty((0,), dtype=np.float32),
                    "obj_ids": [],
                    "image_size": (orig_w, orig_h),
                })
                continue
            masks = np.asarray(fr["masks"], dtype=bool)
            if masks.shape[1:] != (orig_h, orig_w):
                resized = np.empty((n, orig_h, orig_w), dtype=bool)
                for i in range(n):
                    resized[i] = np.array(
                        _Img.fromarray(masks[i]).resize((orig_w, orig_h), _Img.NEAREST)
                    )
                masks = resized
            seg_results.append({
                "masks": masks,
                "boxes": np.asarray(fr["boxes"], dtype=np.float32),
                "scores": np.asarray(fr["scores"], dtype=np.float32),
                "obj_ids": [int(o) for o in fr["obj_ids"]],
                "image_size": (orig_w, orig_h),
            })
        return seg_results

    def _merge_all_seg_results(self) -> list[dict[str, np.ndarray]]:
        all_seg = self._context.get("all_seg_results", {})
        if not all_seg:
            return []

        n_frames = max(len(v) for v in all_seg.values())
        merged = []
        for fi in range(n_frames):
            frame_masks, frame_boxes, frame_scores, frame_labels = [], [], [], []
            frame_obj_ids = []
            frame_image_size = None
            for source, seg_results in all_seg.items():
                if fi >= len(seg_results):
                    continue
                res = seg_results[fi]
                n = len(res["scores"])
                if n == 0:
                    continue
                src_masks = res.get("masks")
                if src_masks is not None:
                    frame_masks.extend(list(src_masks))
                else:
                    frame_masks.extend([None] * n)
                frame_boxes.append(res["boxes"])
                frame_scores.append(res["scores"])
                src_labels = res.get("labels") or [source] * n
                frame_labels.extend(src_labels)
                # Namespace obj_ids by source so track ids from different prompts
                # never collide; sources without obj_ids contribute None (each
                # such detection becomes its own instance downstream).
                src_obj_ids = res.get("obj_ids")
                if src_obj_ids is not None and len(src_obj_ids) == n:
                    frame_obj_ids.extend([f"{source}#{int(o)}" for o in src_obj_ids])
                else:
                    frame_obj_ids.extend([None] * n)
                if res.get("image_size") is not None:
                    frame_image_size = res["image_size"]

            if frame_boxes:
                merged.append({
                    "masks": frame_masks,
                    "boxes": np.concatenate(frame_boxes, axis=0),
                    "scores": np.concatenate(frame_scores, axis=0),
                    "labels": frame_labels,
                    "obj_ids": frame_obj_ids,
                    "image_size": frame_image_size,
                })
            else:
                merged.append({
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "scores": np.empty((0,), dtype=np.float32),
                    "labels": [],
                    "obj_ids": [],
                })
        return merged

    def _run_annotation_loc(self, frame_paths):
        annotations = self._context.get("sample_annotations", {})
        if not annotations:
            raise RuntimeError("No annotations available for this sample — use object_segmentation instead")

        from PIL import Image as _Img
        img = _Img.open(frame_paths[0])
        image_size = img.size

        from tools.code_execution.instance_3d_localization import annotations_to_detections
        dets = annotations_to_detections(
            annotations, n_frames=len(frame_paths),
            image_size=image_size, frame_index=0,
        )
        self._context.setdefault("all_seg_results", {})["__annotations__"] = dets
        self._context["seg_results"] = self._merge_all_seg_results()

        n_det = len(dets[0]["scores"]) if dets else 0
        labels = dets[0].get("labels", []) if dets else []
        return {"summary": f"Annotation-based localization: {n_det} annotated object(s) "
                           f"({', '.join(labels)}), no SAM needed"}

    def _run_3d_loc(self):
        seg = self._context.get("seg_results")
        da3 = self._context.get("da3_result")
        if seg is None or da3 is None:
            raise RuntimeError("depth_estimation and object_segmentation must be run first")

        from tools.code_execution.instance_3d_localization import compute_instance_3d_positions
        r3d = compute_instance_3d_positions(
            seg_results=seg, depth=da3["depth"],
            intrinsics=da3["intrinsics"], extrinsics=da3["extrinsics"],
            output_dir=self.output_root / "spatial",
        )
        self._context["results_3d"] = r3d
        k = len(r3d.get("obj_id_list", []))
        lines = [f"3D localization done: {k} unique instances (merged by 3D proximity)"]
        for i in r3d.get("obj_id_list", []):
            pos = r3d["merged_positions"][i]
            size = r3d["merged_bbox_size"][i]
            sc = r3d["merged_scores"][i]
            lines.append(f"  instance {i}: center=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                         f"bbox_size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})m, score={sc:.2f}")
        return {"summary": "\n".join(lines)}

    def _run_counting(self):
        r3d = self._context.get("results_3d")
        if r3d is None:
            raise RuntimeError("instance_3d_localization must be run first")
        from tools.code_execution.instance_counting import count_unique_instances
        c = count_unique_instances(r3d)
        self._context["counting"] = c
        return {"summary": f"Instance counting done: {c['total_unique']} unique instances, "
                           f"ids={c['obj_id_list']}"}

    def _run_distance(self, params):
        r3d = self._context.get("results_3d")
        da3 = self._context.get("da3_result")
        if r3d is None:
            raise RuntimeError("instance_3d_localization must be run first")

        from tools.code_execution.distance_computation import (
            distance_bbox_to_bbox, distance_object_to_camera,
        )
        mode = params.get("mode", "object_to_object")
        id_list = r3d["obj_id_list"]

        if mode == "object_to_object":
            a_id = params.get("obj_a_id", id_list[0] if len(id_list) > 0 else 0)
            b_id = params.get("obj_b_id", id_list[1] if len(id_list) > 1 else 0)
            a_idx = id_list.index(a_id) if a_id in id_list else 0
            b_idx = id_list.index(b_id) if b_id in id_list else min(1, len(id_list) - 1)
            # Use closest-point (bbox-to-bbox) distance, matching the VSI question phrasing
            # "Measuring from the closest point of each object"
            d = distance_bbox_to_bbox(
                r3d["merged_bbox_min"][a_idx], r3d["merged_bbox_max"][a_idx],
                r3d["merged_bbox_min"][b_idx], r3d["merged_bbox_max"][b_idx],
            )
            return {"summary": f"Closest-point distance between instance {a_id} and instance {b_id}: {d:.3f} meters"}
        else:
            obj_id = params.get("obj_id", id_list[0] if id_list else 0)
            fi = params.get("frame_index", 0)
            idx = id_list.index(obj_id) if obj_id in id_list else 0
            d = distance_object_to_camera(r3d["merged_positions"][idx], da3["extrinsics"], fi)
            return {"summary": f"Distance from instance {obj_id} to camera at frame {fi}: {d:.3f} meters"}

    def _resolve_position(self, pos_id, pos_type, r3d, da3):
        from tools.code_execution.distance_computation import get_camera_position
        if pos_type == "camera":
            return get_camera_position(da3["extrinsics"], int(pos_id))
        else:
            id_list = r3d["obj_id_list"]
            idx = id_list.index(pos_id) if pos_id in id_list else 0
            return r3d["merged_positions"][idx]

    def _run_direction(self, params):
        r3d = self._context.get("results_3d")
        da3 = self._context.get("da3_result")
        if r3d is None:
            raise RuntimeError("instance_3d_localization must be run first")

        from tools.code_execution.direction_computation import compute_relative_direction

        vp_type = params.get("viewpoint_type", "camera")
        ref_type = params.get("reference_type", "object")
        tgt_type = params.get("target_type", "object")

        vp_id  = params.get("viewpoint", 0)
        ref_id = params.get("reference", 0)
        tgt_id = params.get("target", 1)
        fac_id = params.get("facing")          # optional: object that defines forward direction
        fac_type = params.get("facing_type", "object")

        viewpoint_pos = self._resolve_position(vp_id, vp_type, r3d, da3)
        reference_pos = self._resolve_position(ref_id, ref_type, r3d, da3)
        target_pos    = self._resolve_position(tgt_id, tgt_type, r3d, da3)
        facing_pos    = self._resolve_position(fac_id, fac_type, r3d, da3) if fac_id is not None else None

        dr = compute_relative_direction(
            viewpoint=viewpoint_pos, reference=reference_pos, target=target_pos,
            facing=facing_pos,
            extrinsics=da3["extrinsics"],
        )

        vp_label  = f"camera(frame {vp_id})"  if vp_type  == "camera" else f"instance {vp_id}"
        ref_label = f"camera(frame {ref_id})"  if ref_type == "camera" else f"instance {ref_id}"
        tgt_label = f"camera(frame {tgt_id})"  if tgt_type == "camera" else f"instance {tgt_id}"
        fac_label = f" facing instance {fac_id}" if fac_id is not None else ""

        return {"summary": f"Direction (from {vp_label}{fac_label}): {tgt_label} is {dr['direction']} of "
                           f"{ref_label} (lr_angle={dr['lr_angle_deg']:.1f}deg, fb_angle={dr['fb_angle_deg']:.1f}deg)"}

    def _run_scene_size(self):
        pts = self._context.get("points")
        da3 = self._context.get("da3_result")
        if pts is None:
            raise RuntimeError("depth_estimation must be run first")

        from tools.code_execution.scene_size_computation import compute_scene_size
        conf = da3["conf"].reshape(-1) if da3.get("conf") is not None else None
        ss = compute_scene_size(pts, conf=conf, output_dir=self.output_root / "spatial")
        self._context["scene_size"] = ss
        ext = ss["extent_xyz"]
        return {"summary": f"Scene size: width={ext[0]:.2f}m, height={ext[1]:.2f}m, "
                           f"depth={ext[2]:.2f}m, floor_area={ss['floor_area']:.2f}m2, "
                           f"points used: {ss['n_points_clean']}/{ss['n_points_raw']}"}
