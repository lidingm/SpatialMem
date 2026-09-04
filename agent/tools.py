"""Tool registry: register, describe, and dispatch all spatial reasoning tools.

Migrated from the old `tool_registry.py`. All hardcoded paths now come from
`agent.config`. Semantics of the core tools are unchanged.
"""

from __future__ import annotations

import os
import json
import traceback
import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from agent.checker import Checker
from agent.config import OUTPUTS_DIR


# When SPATIALMEM_DEBUG_TRACES=1, tool failures print full tracebacks to stderr.
# Off by default: failures are always captured in the returned dict, and
# noisy tracebacks pollute stderr during normal reflection loops (tools
# routinely fail on missing dependencies, that's how the agent probes state).
_DEBUG_TRACES = os.getenv("SPATIALMEM_DEBUG_TRACES", "0") == "1"


TOOL_DESCRIPTIONS = """=== Available Tools ===

You have 11 tools in two categories. Choose ONLY the tools needed for the question.
Plan tools in executable dependency order. In the interactive raw-tool loop, missing dependencies may be recovered by follow-up actions, but any directly re-executed plan (reconstruct/corrected_plan, run_with_plan, or generated SKILL code) must explicitly include all prerequisite tools.
Distance, direction, counting, and object-size tools may internally call the passive Checker role to verify target objects from annotated frames. If SAM3 finds zero tracks for a required target, Checker answers directly from the raw frames and geometry is skipped.

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
   What it does: Renders a bird's-eye-view (top-down) color image from the DA3 depth back-projected 3D point cloud. It uses DA3 confidence filtering (keeps the top 75% confidence points by default), so the map is useful spatial evidence but can still be sparse or noisy when depth/pose confidence is poor.
   When to use: Questions about spatial layout, relative positions of objects in the floor plan, or understanding the overall room structure.
   Parameters: none
   Output in context: BEV color image with spatial extent in meters
   Depends on: depth_estimation (needs point cloud)

3. novel_view_synthesis
   What it does: Renders the scene from a new camera viewpoint using DA3 3D Gaussian Splatting. If the current DA3 prediction has no gaussians, the tool may re-run DA3 with infer_gs=True before rendering. You specify which source frame to start from and how to move the virtual camera.
   When to use: Exploring occluded regions, verifying what's behind an object, spatial imagination tasks ("what would you see if you moved to position X").
   Parameters:
     - frame_index (int): which frame's camera to start from (default: 0)
     - movement (str): one of yaw_left, yaw_right, pitch_up, pitch_down, move_left, move_right, move_up, move_down, move_forward, move_backward
     - angle_deg (float): rotation amount in degrees for yaw/pitch (default: 30)
     - distance (float): translation amount in meters for move_* (default: 0.2)
   Output in context: rendered RGB image from the new viewpoint
   Depends on: depth_estimation (needs DA3-Giant/3DGS-capable model and gsplat)

4. object_segmentation
   What it does: Uses SAM3 video segmentation/tracking to detect and track all instances of a specified object across the input frames. Returns per-frame binary masks, bounding boxes (xyxy pixel coords), confidence scores, and cross-frame obj_ids for each tracked physical instance.
   When to use: Any task that involves specific objects — counting, measuring, locating, comparing objects.
   Parameters:
     - text_prompt (str): the short plain object noun from the question
       (e.g. "chair", "door", "sofa"). SAM3 only segments short terms well, so
       use the question's own word — keep it to 1-2 words.
     - object_category (str): same short noun, for memory bookkeeping.
   Output in context (per frame):
     - masks: (N_det, H, W) boolean masks
     - boxes: (N_det, 4) bounding boxes in xyxy pixel coordinates
     - scores: (N_det,) detection confidence scores
     - obj_ids: stable cross-frame SAM3 track IDs
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
     - instances: human-readable labels consistent with saved annotations, e.g. "chair 1", with frames and scores
   Notes: The summary intentionally reports candidate labels/frames/scores before Checker review. Final object choices are reported later by Checker/final_localization.
   Depends on: depth_estimation AND (object_segmentation OR annotation_localization)

7. distance_computation
   What it does: Metric distance in meters between two objects or an object and the camera.
   Two modes:
     - object_to_object: closest-point distance between two objects' 3D bounding boxes (matches "Measuring from the closest point of each object")
     - object_to_camera: distance between an object centroid and a camera position at a specific frame
   Parameters:
     - mode (str): "object_to_object" or "object_to_camera"
     - obj_a, obj_b (str): object category names for object_to_object; pass these whenever known
     - obj (str), frame_index (int): object category name and camera frame for object_to_camera
     - obj_a_id, obj_b_id, obj_id (int, optional): pre-resolved instance IDs
   Depends on: instance_3d_localization

8. direction_computation
   What it does: Computes the direction of `target_target` from an anchor point, under an observer viewpoint/facing direction. Projects onto the horizontal plane; returns direction labels (front/back/left/right combinations) and angular offsets.
   Fill params from one of these templates:
     - Egocentric / "my" direction: "standing by X facing Y, is Z to my left/right/front-left/..." -> `reference_target=X`, `target_target=Z`, `viewpoint_type="object"`, `viewpoint_target=X`, `facing_target=Y`. Do NOT use Y as `reference_target`.
     - Object-relative direction: "standing by A facing B, is Z to the left/right/front/back of Y" -> `reference_target=Y`, `target_target=Z`, `viewpoint_type="object"`, `viewpoint_target=A`, `facing_target=B`. If there is no facing phrase, omit `facing_target`; if there is no standing/from object, use `viewpoint_type="camera"`.
   How forward direction is determined:
     - If `facing_target`/`facing` is given: forward = viewpoint -> facing object (use when the person faces a specific object)
     - If facing is absent: forward = viewpoint -> reference object (default: person implicitly faces the reference)
     - If viewpoint_type="camera": forward = camera's own optical axis toward reference
   Parameters:
     - viewpoint_target / reference_target / target_target / facing_target (str, optional): object category names; prefer these whenever known so Checker can validate tracks and resolve the final instance internally
     - viewpoint_type (str): "camera" or "object"
     - viewpoint (int, optional): concrete instance ID if viewpoint_type="object", or frame index if viewpoint_type="camera"; do NOT pass placeholder strings such as "<id_of_chair>"
     - reference (int, optional): concrete instance ID of the anchor object; for "to my ..." questions this anchor is the observer position, not the facing object
     - reference_type (str): "object" (almost always)
     - target (int, optional): concrete instance ID of the object whose direction is asked; do NOT pass placeholder strings
     - target_type (str): "object" (almost always)
     - facing (int, optional): concrete instance ID of the object the person is LOOKING AT; defines the forward direction. Required when viewpoint == reference. Omit only if the person implicitly faces the reference.
     - facing_type (str): "object" (default)
   Output includes: direction label, lr_angle_deg, fb_angle_deg, and angle_from_forward_deg (unsigned 0-180 degrees from observer forward to reference->target).
   Depends on: instance_3d_localization, depth_estimation

9. instance_counting
   What it does: Total number of unique object instances after 3D clustering.
   Parameters: none
   Depends on: instance_3d_localization

10. object_size_computation
   What it does: Computes a target object's 3D bounding-box size from Checker-validated instance geometry. Context summary reports bbox_size_xyz in meters, bbox volume, then the queried width/height/depth/longest/shortest value requested by `dimension`.
   Parameters:
     - object_name / obj / target / object_category (str): target object category
     - obj_id (int, optional): pre-resolved instance id
     - dimension (str): width, height, depth, longest(default), or shortest
     - unit (str): m(default) or cm
   Depends on: instance_3d_localization, depth_estimation

11. scene_size_computation
   What it does: Overall scene bounding box + floor area from the metric point cloud. Filters low-confidence and outlier points before measuring.
   Parameters: none
   Depends on: depth_estimation
"""


class ToolRegistry:
    def __init__(self, da3_tool=None, sam3_tool=None,
                 output_root: str | Path | None = None,
                 llm=None, memory=None):
        self.da3_tool = da3_tool
        self.sam3_tool = sam3_tool
        # Optional: when provided, instance_counting has a VLM review the annotated
        # frames and correct the count; memory supplies the object size priors.
        self.llm = llm
        self.memory = memory
        self.checker = Checker(llm, memory=memory) if llm is not None else None
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

    def set_checker_notes(self, notes: str | None) -> None:
        notes = str(notes or "").strip()
        self._context["active_skill_checker_notes"] = notes
        if self.checker is not None:
            self.checker.set_checker_notes(notes)

    def append_context_text(self, text: str) -> None:
        text = str(text or "").strip()
        if text:
            self._context.setdefault("ordered_context", []).append(text)

    def append_context_note(self, label: str, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        lines = text.splitlines()
        block = f"[{label}] {lines[0]}"
        if len(lines) > 1:
            block += "\n" + "\n".join(lines[1:])
        self.append_context_text(block)

    def _format_final_localization_summary(self) -> str:
        ann = self._context.get("final_localization_annotations") or {}
        r3d = self._context.get("results_3d") or {}
        instances = r3d.get("instances") or []
        by_id = {int(it["id"]): it for it in instances if "id" in it}
        targets = ann.get("targets", [])
        lines = [f"{len(targets)} final target localization(s) after Checker review:"]
        for t in targets:
            it = by_id.get(int(t.get("instance_id", -999999)))
            center = t.get("center_3d_m") or (it or {}).get("center_3d_m")
            size = t.get("bbox_size_m") or (it or {}).get("bbox_size_m")
            geom = ""
            if center is not None and size is not None:
                geom = (
                    f", center=({float(center[0]):.3f}, {float(center[1]):.3f}, {float(center[2]):.3f})m"
                    f", bbox_size=({float(size[0]):.3f}, {float(size[1]):.3f}, {float(size[2]):.3f})m"
                )
            lines.append(f"  {t.get('instance_label')}: keep={t.get('keep_frames')}{geom}")
        return "\n".join(lines)

    def _maybe_append_new_checker_context(self, prev_checker_len: int) -> None:
        for s in self._context.get("checker_summaries", [])[prev_checker_len:]:
            if s:
                self.append_context_note("checker", str(s))

        ann = self._context.get("final_localization_annotations") or {}
        targets = ann.get("targets") or []
        if not targets:
            return
        sig = tuple(
            (str(t.get("instance_label", "")), tuple(t.get("keep_frames") or []))
            for t in targets
        )
        if sig and sig != self._context.get("_ordered_context_final_loc_sig"):
            self._context["_ordered_context_final_loc_sig"] = sig
            self.append_context_note("final_localization", self._format_final_localization_summary())

    def get_context_summary(self) -> str:
        ordered = self._context.get("ordered_context") or []
        if ordered:
            return "\n".join(str(x) for x in ordered if str(x).strip())

        parts: list[str] = []
        if "da3_result" in self._context:
            r = self._context["da3_result"]
            n_frames = r["depth"].shape[0]
            parts.append(f"[depth_estimation] {n_frames} frames, depth shape={r['depth'].shape}, "
                         f"is_metric={r.get('is_metric')}, scale_factor={r.get('scale_factor')}, "
                         f"point_cloud={len(self._context.get('points', []))} points")

        if "bev" in self._context:
            bev = self._context["bev"]
            conf_text = ""
            if bev.get("use_conf"):
                pct = float(bev.get("conf_percentile", 25.0))
                conf_text = f", conf_filter=top {100.0 - pct:.0f}%"
            parts.append(f"[bev_generation] size={bev['raw_size_hw']}, "
                         f"extent=({bev['xy_extent'][0]:.2f}m, {bev['xy_extent'][1]:.2f}m)"
                         f"{conf_text}")

        if "nvs_results" in self._context:
            for nvs in self._context["nvs_results"]:
                img_path = f", image={nvs.get('image_path')}" if nvs.get("image_path") else ""
                parts.append(f"[novel_view_synthesis] {nvs['movement']} {nvs['angle_deg']}deg from frame {nvs['frame_index']}{img_path}")

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
            # Flag this as a pre-calibration candidate list: when instance_counting
            # later runs a VLM review, ITS count supersedes this one.
            parts.append(f"[instance_3d_localization] {k} candidate instance(s) after 3D "
                         f"clustering (before any later review):")
            instances = r3d.get("instances") or []
            by_id = {int(it["id"]): it for it in instances if "id" in it}
            for i in r3d.get("obj_id_list", []):
                it = by_id.get(int(i))
                if it is not None:
                    parts.append(
                        f"  {it.get('label', f'instance {i}')}: "
                        f"frames={it.get('frames', [])}, score={float(it.get('score', 0.0)):.2f}"
                    )
                else:
                    label = r3d.get("merged_labels", [""] * k)[i]
                    name = f"{label} {i}" if label else f"instance {i}"
                    parts.append(f"  {name}: frames=unknown, score={float(r3d['merged_scores'][i]):.2f}")

        if "counting" in self._context:
            c = self._context["counting"]
            # FINAL COUNT is the authoritative answer and must be unmistakable.
            # Never print an id list next to it: after a VLM split/missed correction
            # the count legitimately exceeds the number of surviving ids, and a
            # reader who counts the ids instead of reading the number gets it wrong.
            parts.append(f"[instance_counting] FINAL COUNT = {c['total_unique']} "
                         f"— this is the authoritative number of unique instances; "
                         f"use it directly as the count answer.")
            cal = c.get("calibration")
            if cal and cal.get("error") is None:
                parts.append(f"  (3D clustering proposed {cal['base_count']}; a VLM reviewed "
                             f"the annotated frames and corrected it to {cal['adjusted_count']})")

        if "checker_direct_answer" in self._context:
            ans = self._context["checker_direct_answer"]
            parts.append(f"[checker] raw-sample direct answer = {ans.get('answer')!r}; "
                         f"reason={ans.get('reason', '')}")

        for s in self._context.get("checker_summaries", []):
            if s:
                parts.append(f"[checker] {s}")

        if "final_localization_annotations" in self._context:
            ann = self._context["final_localization_annotations"]
            r3d = self._context.get("results_3d") or {}
            instances = r3d.get("instances") or []
            by_id = {int(it["id"]): it for it in instances if "id" in it}
            targets = ann.get("targets", [])
            parts.append(f"[final_localization] {len(targets)} final target localization(s) after Checker review:")
            for t in ann.get("targets", []):
                it = by_id.get(int(t.get("instance_id", -999999)))
                center = t.get("center_3d_m") or (it or {}).get("center_3d_m")
                size = t.get("bbox_size_m") or (it or {}).get("bbox_size_m")
                geom = ""
                if center is not None and size is not None:
                    geom = (
                        f", center=({float(center[0]):.3f}, {float(center[1]):.3f}, {float(center[2]):.3f})m"
                        f", bbox_size=({float(size[0]):.3f}, {float(size[1]):.3f}, {float(size[2]):.3f})m"
                    )
                parts.append(
                    f"  {t.get('instance_label')}: "
                    f"keep={t.get('keep_frames')}{geom}"
                )

        if "object_size" in self._context:
            osz = self._context["object_size"]
            sx, sy, sz = osz.get("size_xyz_m", [0.0, 0.0, 0.0])
            volume = float(osz.get("bbox_volume_m3", float(sx) * float(sy) * float(sz)))
            label = osz.get("dimension_label") or osz.get("dimension") or "queried dimension"
            value = osz.get("value")
            unit = osz.get("unit", "m")
            queried = f"; queried {label}={float(value):.3f}{unit}" if value is not None else ""
            parts.append(f"[object_size_computation] {osz.get('object')}: "
                         f"bbox_size_xyz=({sx:.3f}, {sy:.3f}, {sz:.3f})m, "
                         f"bbox_volume={volume:.3f}m3{queried}")

        if "scene_size" in self._context:
            ss = self._context["scene_size"]
            ext = ss["extent_xyz"]
            parts.append(f"[scene_size_computation] width={ext[0]:.2f}m, height={ext[1]:.2f}m, "
                         f"depth={ext[2]:.2f}m, floor_area={ss['floor_area']:.2f}m²")

        for entry in self._tool_log:
            if entry.get("tool_name") in (
                "distance_computation", "direction_computation",
            ) \
               and entry.get("success"):
                parts.append(f"[{entry['tool_name']}] {entry.get('result_summary', '')}")


        for s in self._context.get("skill_summaries", []):
            if s:
                parts.append(str(s))

        return "\n".join(parts) if parts else "No tool results yet."

    def get_visual_evidence_content(self, max_side: int = 640) -> list[dict]:
        """Build multimodal evidence blocks for Reflector/Reconstructor/Finalizer.

        Preference order:
          1. Checker-final localization frames, if available.
          2. Otherwise the prepared raw 32 input frames.
        Then append BEV and every NVS render that exists.
        """
        content: list[dict] = []

        final_ann = self._context.get("final_localization_annotations") or {}
        frame_paths = final_ann.get("annotated_frames") or self._context.get("frame_paths") or []
        if final_ann.get("annotated_frames"):
            desc = (
                "Visual evidence group A: Checker-final localization frames. "
                "These are the 32 sample frames with only the final selected/kept target boxes drawn. "
                "Use these to verify which tracks and frames were actually used by downstream geometry."
            )
            label = "final localization frame"
        else:
            desc = (
                "Visual evidence group A: original input frames. "
                "No final localization overlay has been produced yet, so these are the raw prepared 32 frames."
            )
            label = "raw input frame"
        if frame_paths:
            content.append({"type": "text", "text": desc})
            for i, fp in enumerate(frame_paths):
                self._append_image_block(content, fp, f"{label} {i}", max_side=max_side)

        bev_path = self._context.get("bev", {}).get("image_path")
        if bev_path:
            content.append({
                "type": "text",
                "text": (
                    "Visual evidence group B: BEV image. "
                    "This is a top-down color projection from DA3 depth/pose back-projection; "
                    "use it for overall room layout and object spatial arrangement, while remembering it may be sparse/noisy."
                ),
            })
            self._append_image_block(content, bev_path, "BEV top-down image", max_side=max_side)

        nvs_items = [n for n in self._context.get("nvs_results", []) if n.get("image_path")]
        if nvs_items:
            content.append({
                "type": "text",
                "text": (
                    "Visual evidence group C: NVS render(s). "
                    "These are DA3 3DGS novel-view renders from requested virtual camera motions; "
                    "use them to inspect occlusion, alternate viewpoints, and spatial imagination evidence."
                ),
            })
            for nvs in nvs_items:
                label = (f"NVS image: frame {nvs.get('frame_index')} "
                         f"{nvs.get('movement')} angle={nvs.get('angle_deg')} "
                         f"distance={nvs.get('distance')}")
                self._append_image_block(content, nvs["image_path"], label, max_side=max_side)

        return content

    @staticmethod
    def _append_image_block(content: list[dict], path: str | Path, label: str,
                            max_side: int = 640) -> None:
        try:
            from PIL import Image

            p = Path(path)
            im = Image.open(p).convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "text", "text": f"{label}: {p}"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        except Exception:
            return

    def execute_tool(self, tool_name: str, params: dict,
                     frame_paths: list[str]) -> dict:
        prev_checker_len = len(self._context.get("checker_summaries", []))
        if self.checker is not None:
            self.checker.set_context_summary(self.get_context_summary())
        try:
            result = self._dispatch(tool_name, params, frame_paths)
            entry = {"tool_name": tool_name, "params": params,
                     "success": True, "result_summary": result["summary"],
                     "error": None}
            self._tool_log.append(entry)
            # Checker/final localization are internal evidence for derived tools;
            # show them before the derived computation summary in ordered context.
            self._maybe_append_new_checker_context(prev_checker_len)
            self.append_context_note(tool_name, result["summary"])
            return entry
        except Exception as e:
            entry = {"tool_name": tool_name, "params": params,
                     "success": False, "result_summary": "",
                     "error": str(e)}
            self._tool_log.append(entry)
            self.append_context_note(tool_name, f"FAILED: {e}")
            self._maybe_append_new_checker_context(prev_checker_len)
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
            "distance_computation":      lambda: self._run_distance(params, frame_paths),
            "direction_computation":     lambda: self._run_direction(params, frame_paths),
            "object_size_computation":   lambda: self._run_object_size(params, frame_paths),
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
        bev = self.da3_tool.make_bev(
            pred,
            output_dir=self.output_root / "da3",
            use_conf=True,
            conf_percentile=25.0,
        )
        bev["image_path"] = str(self.output_root / "da3" / "bev_from_depth.png")
        self._context["bev"] = bev
        return {"summary": f"BEV generated: size {bev['raw_size_hw']}, "
                           f"extent x={bev['xy_extent'][0]:.2f}m z={bev['xy_extent'][1]:.2f}m, "
                           f"conf_filter=top 75%"}

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

        stem = f"nvs_f{int(fi):04d}_{mov}"
        self.da3_tool.render_nvs(
            pred, frame_index=fi, movement=mov,
            angle_deg=ang, distance=dist,
            output_dir=self.output_root / "da3",
            output_name=stem,
        )
        self._context.setdefault("nvs_results", []).append({
            "frame_index": fi, "movement": mov, "angle_deg": ang, "distance": dist,
            "image_path": str(self.output_root / "da3" / f"{stem}.png"),
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
        sam3_dir = self.output_root / "sam3" / self._safe_output_name(object_category or text_prompt)
        self._save_sam3_track_outputs(seg_results, input_paths, text_prompt, sam3_dir)

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
    def _safe_output_name(name: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                       for ch in str(name).strip().lower()) or "object"

    @staticmethod
    def _save_sam3_track_outputs(seg_results, frame_paths, text_prompt: str, output_dir: Path) -> None:
        """Persist raw SAM3 video tracking outputs for debugging and audit."""
        output_dir.mkdir(parents=True, exist_ok=True)

        all_masks, all_boxes, all_scores, all_obj_ids, counts = [], [], [], [], []
        for res in seg_results:
            n = len(res.get("scores", []))
            counts.append(n)
            if n == 0:
                continue
            all_masks.append(np.asarray(res["masks"], dtype=bool))
            all_boxes.append(np.asarray(res["boxes"], dtype=np.float32))
            all_scores.append(np.asarray(res["scores"], dtype=np.float32))
            all_obj_ids.extend([int(o) for o in res.get("obj_ids", [])])

        total = int(sum(counts))
        np.savez_compressed(
            output_dir / "sam3_track.npz",
            masks=np.concatenate(all_masks, axis=0) if total else np.empty((0,), dtype=bool),
            boxes=np.concatenate(all_boxes, axis=0) if total else np.empty((0, 4), dtype=np.float32),
            scores=np.concatenate(all_scores, axis=0) if total else np.empty((0,), dtype=np.float32),
            obj_ids=np.asarray(all_obj_ids, dtype=np.int32),
            counts=np.asarray(counts, dtype=np.int32),
        )

        metadata = {
            "text_prompt": text_prompt,
            "frames": [str(p) for p in frame_paths],
            "per_frame_counts": counts,
            "total_detections": total,
            "unique_obj_ids": sorted({int(o) for o in all_obj_ids}),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        # frame_paths + object_name let it render the annotated frames into
        # output_root/spatial/<object_name>/ (used by the counting calibration).
        r3d = compute_instance_3d_positions(
            seg_results=seg, depth=da3["depth"],
            intrinsics=da3["intrinsics"], extrinsics=da3["extrinsics"],
            frame_paths=self._context.get("frame_paths"),
            object_name=self._context.get("text_prompt"),
            output_dir=self.output_root / "spatial",
        )
        self._context["results_3d"] = r3d
        k = len(r3d.get("obj_id_list", []))
        lines = [f"3D localization done: {k} candidate instance(s) after 3D clustering"]
        instances = r3d.get("instances") or []
        by_id = {int(it["id"]): it for it in instances if "id" in it}
        for i in r3d.get("obj_id_list", []):
            it = by_id.get(int(i))
            if it is not None:
                lines.append(
                    f"  {it.get('label', f'instance {i}')}: "
                    f"frames={it.get('frames', [])}, score={float(it.get('score', 0.0)):.2f}"
                )
            else:
                label = r3d.get("merged_labels", [""] * k)[i]
                name = f"{label} {i}" if label else f"instance {i}"
                lines.append(f"  {name}: frames=unknown, score={float(r3d['merged_scores'][i]):.2f}")
        return {"summary": "\n".join(lines)}

    def _run_counting(self):
        r3d = self._context.get("results_3d")
        if r3d is None:
            raise RuntimeError("instance_3d_localization must be run first")
        from tools.code_execution.instance_counting import count_unique_instances
        # When an llm is wired in, a VLM reviews the annotated frames and the count
        # is recomputed from its structured corrections (missed/spurious/merge/split).
        c = count_unique_instances(
            r3d, llm=self.llm,
            object_name=self._context.get("text_prompt"),
            memory=self.memory,
            checker_notes=self._context.get("active_skill_checker_notes", ""),
            context_summary=self.get_context_summary(),
        )
        self._context["counting"] = c
        cal = c.get("calibration")
        if cal and cal.get("summary"):
            self._context.setdefault("checker_summaries", []).append(cal["summary"])
        final_instances = c.get("final_instances") or []
        if final_instances:
            try:
                from tools.code_execution.instance_3d_localization import (
                    draw_final_localization_annotations,
                )
                final_locs = {}
                for idx, it in enumerate(final_instances, start=1):
                    label = str(it.get("label") or f"{self._context.get('text_prompt', 'object')} {idx}")
                    key = label
                    final_locs[key] = {
                        "instance_id": int(it.get("id", idx)),
                        "instance_label": label,
                        "keep_frames": list(it.get("frames") or []),
                        "dropped_frames": [],
                        "reason": "final instance after count calibration",
                        "boxes_2d": list(it.get("boxes_2d") or []),
                        "center_3d_m": it.get("center_3d_m"),
                        "bbox_size_m": it.get("bbox_size_m"),
                        "original_label": it.get("original_label"),
                    }
                ann = draw_final_localization_annotations(
                    r3d,
                    self._context.get("frame_paths") or [],
                    final_locs,
                    self.output_root / "spatial" / "final_localization",
                )
                self._context["final_localization_annotations"] = ann
            except Exception as e:  # noqa: BLE001 - visualization must not break counting
                self._context["final_localization_annotation_error"] = str(e)
        # FINAL COUNT must be unmistakable: after calibration the surviving id list
        # can be shorter than the count (split/missed add without adding ids), so
        # never let the reader infer the answer by counting ids.
        parts = [f"Instance counting done: FINAL COUNT = {c['total_unique']}"]
        if cal and cal.get("error") is None:
            parts.append(f"  (3D clustering produced {cal['base_count']} candidate instance(s); "
                         f"a VLM reviewed the annotated frames and corrected it to "
                         f"{cal['adjusted_count']})")
            parts.append(f"  Answer with FINAL COUNT = {c['total_unique']}.")
        else:
            parts.append(f"  instance ids={c['obj_id_list']}")
        return {"summary": "\n".join(parts)}

    def _run_distance(self, params, frame_paths):
        from tools.code_execution.distance_computation import run_distance_task
        result = run_distance_task(params, self._context, self.checker, frame_paths)
        return {"summary": result["summary"]}

    def _run_direction(self, params, frame_paths):
        from tools.code_execution.direction_computation import run_direction_task
        result = run_direction_task(params, self._context, self.checker, frame_paths)
        return {"summary": result["summary"]}

    @staticmethod
    def _format_stat_prior(stat: dict, unit: str) -> str | None:
        if not isinstance(stat, dict) or "mean" not in stat:
            return None
        mean = float(stat.get("mean", 0.0))
        std = float(stat.get("std", 0.0))
        lo = max(0.0, mean - std)
        hi = mean + std
        return f"{mean:.2f}+/-{std:.2f}{unit} (approx range {lo:.2f}-{hi:.2f}{unit})"

    def _object_size_prior_summary(self, object_name: str) -> str:
        name = str(object_name or "").strip().lower()
        if not name or self.memory is None:
            return f"[prior] object size prior for {name or 'target object'}: no prior available"
        try:
            prior = self.memory.get_object_size_prior(name)
        except Exception:
            prior = None
        if not prior:
            return f"[prior] object size prior for {name}: no prior available"
        parts = []
        for key in ("width", "height", "depth"):
            item = self._format_stat_prior(prior.get(key), "m")
            if item:
                parts.append(f"{key}={item}")
        return f"[prior] object size prior for {name}: " + (", ".join(parts) if parts else "no prior available")

    def _scene_size_prior_summary(self) -> str:
        scene_type = str(self._context.get("scene_type") or "").strip().lower()
        if not scene_type or self.memory is None:
            return f"[prior] scene size prior for {scene_type or 'current room'}: no prior available"
        try:
            prior = self.memory.get_scene_scale_prior(scene_type)
        except Exception:
            prior = None
        if not prior:
            return f"[prior] scene size prior for {scene_type}: no prior available"
        parts = []
        for key in ("floor_area", "width", "height", "depth"):
            unit = "m2" if key == "floor_area" else "m"
            item = self._format_stat_prior(prior.get(key), unit)
            if item:
                parts.append(f"{key}={item}")
        return f"[prior] scene size prior for {scene_type}: " + (", ".join(parts) if parts else "no prior available")

    def append_task_prior_context(self, task_type: str = "", task_category: str = "",
                                  object_names: list[str] | None = None) -> None:
        """Append task-relevant priors before finalization, even if size tools failed."""
        task = f"{task_type} {task_category}".lower()
        existing = "\n".join(str(x) for x in self._context.get("ordered_context", []))

        if "object_size" in task or ("object" in task and "size" in task):
            names = []
            for name in list((self._context.get("seg_categories") or {}).values()) + list(object_names or []):
                name = str(name or "").strip().lower()
                if name and name not in names:
                    names.append(name)
            for name in names:
                line = self._object_size_prior_summary(name)
                if line not in existing:
                    self.append_context_text(line)
                    existing += "\n" + line

        if "room_size" in task or "scene_size" in task:
            line = self._scene_size_prior_summary()
            if line not in existing:
                self.append_context_text(line)

    def _run_object_size(self, params, frame_paths):
        from tools.code_execution.object_size_computation import run_object_size_task
        result = run_object_size_task(params, self._context, self.checker, frame_paths)
        object_name = (
            (result.get("object_size") or {}).get("object")
            or params.get("object_name")
            or params.get("target")
            or params.get("obj")
            or params.get("object_category")
        )
        summary = result["summary"]
        return {"summary": summary}

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
        summary = (
            f"Scene size: width={ext[0]:.2f}m, height={ext[1]:.2f}m, "
            f"depth={ext[2]:.2f}m, floor_area={ss['floor_area']:.2f}m2, "
            f"points used: {ss['n_points_clean']}/{ss['n_points_raw']}"
        )
        return {"summary": summary}
