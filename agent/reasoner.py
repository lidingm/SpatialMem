"""Reasoner role: per-step tool call refinement in the raw-tool fallback path.

Only invoked when the Planner did NOT choose a SKILL. Takes the current step
of the plan and refines the exact tool + params against the accumulated
context.
"""

from __future__ import annotations

from agent.llm_client import LLMClient
from agent.planner import FRAMEWORK_OVERVIEW, format_plan


def reasoner_system_prompt(tool_descriptions: str) -> str:
    return f"""{FRAMEWORK_OVERVIEW}

You are the REASONER (per-step). The PLANNER produced a raw-tool plan. For each step you receive, confirm or refine the tool name and parameters based on the question and the accumulated evidence.

{tool_descriptions}

Output JSON:
{{
    "tool": "tool_name",
    "params": {{...}},
    "reasoning": "1 sentence: why this tool + these parameters given the accumulated evidence"
}}

CRITICAL: `tool` MUST be exactly one of these eleven names — do NOT invent new tools:
  depth_estimation, bev_generation, novel_view_synthesis, object_segmentation,
  annotation_localization, instance_3d_localization, instance_counting,
  distance_computation, direction_computation, object_size_computation,
  scene_size_computation

If none of the eleven tools can produce the needed evidence, pick the closest one that CAN run and let the Reflector handle the gap.

Guidelines:
- Follow the planned tool unless the accumulated evidence CLEARLY indicates a different tool is needed (e.g., a dependency failed; a different object was detected).
- Populate params by inspecting the accumulated evidence and the original question.
- For `distance_computation`, `direction_computation`, and `object_size_computation`, prefer object-name params (`obj_a`, `obj_b`, `object_name`, `viewpoint_target`, `reference_target`, `target_target`, `facing_target`). The tool layer will call Checker and resolve the intended track/instance.
- Use concrete integer instance IDs only if the accumulated evidence makes that ID unambiguous. Never pass placeholder strings such as `<id_of_chair>`, `chair_instance_id`, or `"id_of_chair"`.
- For camera-relative questions, pick the right `frame_index`; for "standing by X" / "facing Y" questions, use `viewpoint_type="object"` with `viewpoint_target="X"` and `facing_target="Y"`.
- For `object_segmentation`, the `text_prompt` should be as specific as needed to disambiguate but as terse as possible when counting all instances of a category.
- For `distance_computation`, `direction_computation`, and `object_size_computation`, only invoke after `instance_3d_localization` has produced usable `obj_id_list`.
"""


def reasoner_user_message(plan: dict, current_step: int,
                          context_summary: str, question: str) -> str:
    steps = plan.get("plan", [])
    step = steps[current_step] if current_step < len(steps) else None
    text = f"Original question: {question}\n\n"
    text += f"Execution plan:\n{format_plan(plan)}\n\n"
    text += f">>> Now executing step {current_step + 1}\n"
    if step:
        text += f"    Planned tool: {step['tool']}\n"
        text += f"    Planned params: {step.get('params', {})}\n"
        text += f"    Expected evidence: {step.get('expected_evidence', '')}\n\n"
    text += f"Accumulated evidence so far:\n{context_summary}\n\n"
    text += "Confirm or refine the tool call. Output JSON."
    return text


class Reasoner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def next_action(self, plan: dict, current_step: int,
                    context_summary: str, question: str,
                    tool_descriptions: str) -> dict:
        messages = [
            {"role": "system", "content": reasoner_system_prompt(tool_descriptions)},
            {"role": "user", "content": reasoner_user_message(
                plan, current_step, context_summary, question)},
        ]
        return self.llm.chat_json(messages)
