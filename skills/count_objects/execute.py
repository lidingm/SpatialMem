"""SKILL: count_objects.

Count unique instances of a target object category across the video frames.
"""

from __future__ import annotations


def execute(sample, tools, ctx, params=None):
    params = params or {}
    target: str = params.get("target", "")
    if not target:
        return {"success": False, "tool_calls": [],
                "summary": "missing required param 'target'"}

    frame_paths = ctx["frame_paths"]
    calls: list[dict] = []

    def _call(name, p):
        r = tools.execute_tool(name, p, frame_paths)
        calls.append(r)
        return r

    r = _call("depth_estimation", {})
    if not r["success"]:
        return {"success": False, "tool_calls": calls,
                "summary": f"depth_estimation failed: {r['error']}"}

    r = _call("object_segmentation", {"text_prompt": target, "object_category": target})
    if not r["success"]:
        return {"success": False, "tool_calls": calls,
                "summary": f"object_segmentation failed: {r['error']}"}

    r = _call("instance_3d_localization", {})
    if not r["success"]:
        return {"success": False, "tool_calls": calls,
                "summary": f"instance_3d_localization failed: {r['error']}"}

    r = _call("instance_counting", {})
    return {
        "success": r["success"],
        "tool_calls": calls,
        "summary": r.get("result_summary", "") if r["success"]
                   else f"instance_counting failed: {r.get('error', '')}",
    }
