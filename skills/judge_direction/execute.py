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
    Default: "object" when `viewpoint_target` is provided, otherwise "camera".

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
    viewpoint_target: str = params.get("viewpoint_target", "")
    facing_target:    str = params.get("facing_target",    "")
    viewpoint_type:   str = params.get("viewpoint_type") or (
        "object" if viewpoint_target else "camera"
    )
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

    # ── Step 4: build direction_computation params ────────────────────
    # Pass names, not guessed IDs. The tool layer runs Checker and resolves
    # the final instance/frames for each target.
    if viewpoint_type == "camera":
        dir_params = {
            "viewpoint":      viewpoint_frame,
            "viewpoint_type": "camera",
            "reference_target": reference_target,
            "reference_type": "object",
            "target_target":  target_target,
            "target_type":    "object",
        }

    else:  # viewpoint_type == "object"
        dir_params = {
            "viewpoint_type": "object",
            "viewpoint_target": viewpoint_target,
            "reference_type": "object",
            "reference_target": reference_target,
            "target_type":    "object",
            "target_target":  target_target,
        }
        # Add facing if provided (or if viewpoint == reference, facing is required)
        if facing_target:
            dir_params["facing_type"] = "object"
            dir_params["facing_target"] = facing_target
        elif viewpoint_target == reference_target:
            return {"success": False, "tool_calls": calls,
                    "summary": f"viewpoint_target={viewpoint_target!r} and "
                               f"reference_target={reference_target!r} are the same target, "
                               f"so forward direction is undefined without facing_target."}

    r = _call("direction_computation", dir_params)
    return {
        "success": r["success"],
        "tool_calls": calls,
        "summary": r.get("result_summary", "") if r["success"]
                   else f"direction_computation failed: {r.get('error', '')}",
    }
