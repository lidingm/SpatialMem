"""SKILL: measure_distance.

Compute the metric distance between two objects (or an object and the camera).
Instance IDs are resolved by label from merged_labels, not hardcoded indices.
"""

from __future__ import annotations


def execute(sample, tools, ctx, params=None):
    params = params or {}
    targets: list[str] = params.get("targets", [])
    mode: str = params.get("mode", "object_to_object")
    frame_paths = ctx["frame_paths"]
    calls: list[dict] = []

    if mode == "object_to_object" and (not targets or len(targets) < 2):
        return {"success": False, "tool_calls": calls,
                "summary": f"missing required param 'targets' (list of >=2 object descriptions) "
                           f"for mode='object_to_object'; got {targets!r}"}
    if mode == "object_to_camera" and not targets:
        return {"success": False, "tool_calls": calls,
                "summary": f"missing required param 'targets' (list of >=1 object) "
                           f"for mode='object_to_camera'; got {targets!r}"}

    def _call(name, p):
        r = tools.execute_tool(name, p, frame_paths)
        calls.append(r)
        return r

    r = _call("depth_estimation", {})
    if not r["success"]:
        return {"success": False, "tool_calls": calls,
                "summary": f"depth_estimation failed: {r['error']}"}

    for t in targets:
        r = _call("object_segmentation", {"text_prompt": t, "object_category": t})
        if not r["success"]:
            return {"success": False, "tool_calls": calls,
                    "summary": f"object_segmentation failed for '{t}': {r['error']}"}

    r = _call("instance_3d_localization", {})
    if not r["success"]:
        return {"success": False, "tool_calls": calls,
                "summary": f"instance_3d_localization failed: {r['error']}"}

    # Resolve instance IDs by label
    r3d = ctx.get("results_3d", {})
    merged_labels = r3d.get("merged_labels", [])
    obj_ids = r3d.get("obj_id_list", [])

    def _find_id(label: str) -> int | None:
        for i in obj_ids:
            if i < len(merged_labels) and merged_labels[i] == label:
                return i
        return None

    if mode == "object_to_object":
        a_id = _find_id(targets[0])
        b_id = _find_id(targets[1])
        if a_id is None:
            return {"success": False, "tool_calls": calls,
                    "summary": f"no instance found for '{targets[0]}'; detected labels={merged_labels}"}
        if b_id is None:
            return {"success": False, "tool_calls": calls,
                    "summary": f"no instance found for '{targets[1]}'; detected labels={merged_labels}"}
        r = _call("distance_computation", {"mode": "object_to_object",
                                           "obj_a_id": a_id, "obj_b_id": b_id})
    else:
        obj_id = _find_id(targets[0])
        if obj_id is None:
            return {"success": False, "tool_calls": calls,
                    "summary": f"no instance found for '{targets[0]}'; detected labels={merged_labels}"}
        r = _call("distance_computation", {"mode": "object_to_camera",
                                           "obj_id": obj_id,
                                           "frame_index": params.get("frame_index", 0)})

    return {
        "success": r["success"],
        "tool_calls": calls,
        "summary": r.get("result_summary", "") if r["success"]
                   else f"distance_computation failed: {r.get('error', '')}",
    }
