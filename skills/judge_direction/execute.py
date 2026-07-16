"""SKILL: judge_direction.

Determine the relative direction of `target_target` w.r.t. `reference_target`
from the given viewpoint.

Parameters
----------
reference_target : str  (REQUIRED)
    Anchor object for the directional judgment.
    "is X to the left of Y" → reference_target = Y.

target_target : str  (REQUIRED)
    The object whose direction is asked.
    "is X to the left of Y" → target_target = X.

viewpoint_type : str
    "camera"  – observer is the camera at frame `viewpoint_frame`.
    "object"  – observer is at object `viewpoint_target`.
    Default: "camera".

viewpoint_target : str  (required when viewpoint_type="object")
    Object name of WHERE the person stands.
    If equal to reference_target: forward direction is undefined unless
    `facing_target` is also provided.

facing_target : str  (optional)
    Object name of WHAT the person faces; defines the forward direction.
    Required when viewpoint_target == reference_target (otherwise forward=0).
    Optional otherwise (default forward = viewpoint → reference).

viewpoint_frame : int
    Camera frame index used when viewpoint_type="camera". Default 0.

Forward direction resolution
----------------------------
  facing_target given  →  forward = viewpoint_pos → facing_pos
  facing_target absent →  forward = viewpoint_pos → reference_pos
"""

from __future__ import annotations


def execute(sample, tools, ctx, params=None):
    params = params or {}
    reference_target: str = params.get("reference_target", "")
    target_target:    str = params.get("target_target",    "")
    viewpoint_type:   str = params.get("viewpoint_type",   "camera")
    viewpoint_target: str = params.get("viewpoint_target", "")
    facing_target:    str = params.get("facing_target",    "")
    viewpoint_frame:  int = int(params.get("viewpoint_frame", 0))

    if not reference_target or not target_target:
        return {"success": False, "tool_calls": [],
                "summary": f"missing required params; reference_target={reference_target!r}, "
                           f"target_target={target_target!r}"}
    if viewpoint_type == "object" and not viewpoint_target:
        return {"success": False, "tool_calls": [],
                "summary": "viewpoint_type='object' requires 'viewpoint_target'"}

    # Deduplicated list of objects to segment
    seg_targets: list[str] = []
    for t in (reference_target, target_target, viewpoint_target, facing_target):
        if t and t not in seg_targets:
            seg_targets.append(t)

    frame_paths = ctx["frame_paths"]
    calls: list[dict] = []

    def _call(name, p):
        r = tools.execute_tool(name, p, frame_paths)
        calls.append(r)
        return r

    # ── Step 1: depth estimation ──────────────────────────────────────
    r = _call("depth_estimation", {})
    if not r["success"]:
        return {"success": False, "tool_calls": calls,
                "summary": f"depth_estimation failed: {r['error']}"}

    # ── Step 2: segmentation for each object ─────────────────────────
    for t in seg_targets:
        r = _call("object_segmentation", {"text_prompt": t, "object_category": t})
        if not r["success"]:
            return {"success": False, "tool_calls": calls,
                    "summary": f"object_segmentation failed for '{t}': {r['error']}"}

    # ── Step 3: 3D localization ───────────────────────────────────────
    r = _call("instance_3d_localization", {})
    if not r["success"]:
        return {"success": False, "tool_calls": calls,
                "summary": f"instance_3d_localization failed: {r['error']}"}

    # ── Step 4: resolve instance IDs by label ────────────────────────
    r3d = ctx.get("results_3d", {})
    merged_labels = r3d.get("merged_labels", [])
    obj_ids = r3d.get("obj_id_list", [])

    def _find_id(label: str) -> int | None:
        for i in obj_ids:
            if i < len(merged_labels) and merged_labels[i] == label:
                return i
        return None

    reference_id = _find_id(reference_target)
    target_id    = _find_id(target_target)

    if reference_id is None:
        return {"success": False, "tool_calls": calls,
                "summary": f"no instance found for reference_target={reference_target!r}; "
                           f"detected labels={merged_labels}"}
    if target_id is None:
        return {"success": False, "tool_calls": calls,
                "summary": f"no instance found for target_target={target_target!r}; "
                           f"detected labels={merged_labels}"}

    # ── Step 5: build direction_computation params ────────────────────
    if viewpoint_type == "camera":
        dir_params = {
            "viewpoint":      viewpoint_frame,
            "viewpoint_type": "camera",
            "reference":      reference_id,
            "reference_type": "object",
            "target":         target_id,
            "target_type":    "object",
        }

    else:  # viewpoint_type == "object"
        vp_id     = _find_id(viewpoint_target)
        facing_id = _find_id(facing_target) if facing_target else None

        if vp_id is None:
            # viewpoint_target not detected → fall back to camera
            dir_params = {
                "viewpoint":      viewpoint_frame,
                "viewpoint_type": "camera",
                "reference":      reference_id,
                "reference_type": "object",
                "target":         target_id,
                "target_type":    "object",
            }
        else:
            dir_params = {
                "viewpoint":      vp_id,
                "viewpoint_type": "object",
                "reference":      reference_id,
                "reference_type": "object",
                "target":         target_id,
                "target_type":    "object",
            }
            # Add facing if provided (or if viewpoint == reference, facing is required)
            if facing_id is not None:
                dir_params["facing"]      = facing_id
                dir_params["facing_type"] = "object"
            elif vp_id == reference_id:
                # viewpoint = reference but no facing → forward = 0, direction undefined
                return {"success": False, "tool_calls": calls,
                        "summary": f"viewpoint_target={viewpoint_target!r} and "
                                   f"reference_target={reference_target!r} resolved to the same "
                                   f"instance (id={vp_id}), so forward direction is undefined. "
                                   f"Please provide 'facing_target' to specify the facing direction."}

    r = _call("direction_computation", dir_params)
    return {
        "success": r["success"],
        "tool_calls": calls,
        "summary": r.get("result_summary", "") if r["success"]
                   else f"direction_computation failed: {r.get('error', '')}",
    }
