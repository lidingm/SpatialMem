"""SKILL: measure_scene_size.

Compute the overall scene extent and floor area from the reconstructed point cloud.
"""

from __future__ import annotations


def execute(sample, tools, ctx, params=None):
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

    r = _call("scene_size_computation", {})
    return {
        "success": r["success"],
        "tool_calls": calls,
        "summary": r.get("result_summary", "") if r["success"]
                   else f"scene_size_computation failed: {r.get('error', '')}",
    }
