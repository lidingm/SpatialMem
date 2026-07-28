"""SKILL: measure_distance.

Compute the metric distance between two objects (or an object and the camera).
Target names are passed to the tool layer; Checker resolves the final instances.
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

    if mode == "object_to_object":
        p = {"mode": "object_to_object", "obj_a": targets[0], "obj_b": targets[1]}
        r = _call("distance_computation", p)
    else:
        p = {"mode": "object_to_camera", "obj": targets[0],
             "frame_index": params.get("frame_index", 0)}
        r = _call("distance_computation", p)

    return {
        "success": r["success"],
        "tool_calls": calls,
        "summary": r.get("result_summary", "") if r["success"]
                   else f"distance_computation failed: {r.get('error', '')}",
    }
