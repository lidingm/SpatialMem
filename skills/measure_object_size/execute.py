"""SKILL: measure_object_size.

Compute the metric size of one target object using SAM3, DA3 geometry,
3D localization, and Checker-validated target selection.
"""

from __future__ import annotations


def execute(sample, tools, ctx, params=None):
    params = params or {}
    object_name = (
        params.get("object_name")
        or params.get("target")
        or params.get("obj")
        or params.get("object_category")
        or ""
    )
    if not object_name:
        return {
            "success": False,
            "tool_calls": [],
            "summary": "missing required param 'object_name'",
        }

    dimension = params.get("dimension", "longest")
    unit = params.get("unit", "m")
    frame_paths = ctx["frame_paths"]
    calls: list[dict] = []

    def _call(name, p):
        r = tools.execute_tool(name, p, frame_paths)
        calls.append(r)
        return r

    r = _call("depth_estimation", {})
    if not r["success"]:
        return {
            "success": False,
            "tool_calls": calls,
            "summary": f"depth_estimation failed: {r['error']}",
        }

    r = _call("object_segmentation", {
        "text_prompt": object_name,
        "object_category": object_name,
    })
    if not r["success"]:
        return {
            "success": False,
            "tool_calls": calls,
            "summary": f"object_segmentation failed for '{object_name}': {r['error']}",
        }

    r = _call("instance_3d_localization", {})
    if not r["success"]:
        return {
            "success": False,
            "tool_calls": calls,
            "summary": f"instance_3d_localization failed: {r['error']}",
        }

    r = _call("object_size_computation", {
        "object_name": object_name,
        "dimension": dimension,
        "unit": unit,
    })
    return {
        "success": r["success"],
        "tool_calls": calls,
        "summary": r.get("result_summary", "") if r["success"]
        else f"object_size_computation failed: {r.get('error', '')}",
    }
