"""Template SKILL executor.

Copy this file into a new SKILL folder and fill in the body of `execute`.
"""

from __future__ import annotations


def execute(sample, tools, ctx, params=None):
    """Run the SKILL end-to-end.

    Args:
        sample: agent.data_loader.SpatialSample.
        tools: agent.tools.ToolRegistry. Call `tools.execute_tool(name, params, frame_paths)`.
        ctx: shared mutable dict. Populated by tools; read `ctx["frame_paths"]` for prepared frames.
        params: caller-supplied parameters (from Planner or Reasoner).

    Returns:
        dict with:
          - success (bool): whether the SKILL produced usable evidence
          - tool_calls (list[dict]): per-step {tool_name, params, success, result_summary, error}
          - summary (str): human-readable summary
    """
    params = params or {}
    return {"success": False, "tool_calls": [], "summary": "Template not implemented."}
