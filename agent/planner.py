"""Planner role for SpatialMem.

The Planner sees:
  - the spatial question and all prepared input frames
  - retrieved SKILL briefs for the sample's task category
  - the memory context (declarative priors)
  - the tool descriptions (fallback path)

It outputs a JSON plan that either picks a SKILL and its parameters, OR lays
out a sequence of raw tool calls.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from agent.llm_client import LLMClient


OBJECT_CATEGORY_LIST = """\
Common examples (pick the closest, or write a concise 1-3 word label if none fit):
  chair | sofa | bed | desk | dining table | coffee table | nightstand
  cabinet | shelf | wardrobe | bookcase | drawer | trash can
  door | window | curtain | mirror | lamp | television | monitor
  refrigerator | microwave | sink | toilet | bathtub
  plant | box | bag | backpack | bottle"""

ROOM_TYPE_LIST = """\
Common examples (pick the closest, or write a concise label if none fit):
  bedroom | living room | kitchen | bathroom | dining room | home office |
  hallway | storage room | conference room | classroom | gym | lobby"""


FRAMEWORK_OVERVIEW = """SpatialMem is a spatial reasoning agent with:
  * a VLM planner that decides the strategy
  * a library of SKILLs (parameterized tool sequences that were curated or evolved from past trajectories)
  * a memory bank of declarative priors (typical object sizes, scene scales)

The agent flow is: PLANNER → (invoke a SKILL) OR (REASONER-driven raw-tool loop) → REFLECTOR → answer.
When a matching SKILL exists, prefer invoking it — it encodes lessons already validated on past training samples."""


def planner_system_prompt(tool_descriptions: str,
                          skills_brief: str,
                          memory_context: str) -> str:
    return f"""{FRAMEWORK_OVERVIEW}

You are the PLANNER. Analyze the spatial question, review the retrieved SKILLs, and produce a plan.

=== Retrieved SKILLs for this task category ===
{skills_brief or "(none)"}

=== Available Raw Tools (fallback path) ===
{tool_descriptions}

=== Memory Context ===
{memory_context}

Output JSON:
{{
    "task_category": "distance" | "counting" | "spatial_relation" | "room_size" | "object_size" | "appearance_order" | "route_planning" | ...,
    "scene_type": "<room type — pick from Room Types list below, or null if unclear>",
    "analysis": "1-3 sentences: what the question asks; what evidence is needed to answer it.",
    "object_targets": ["chair", "table"],
    "chosen_skill": "<name of one SKILL from the retrieved list>" or null,
    "skill_params": {{ ... }},
    "plan": [
        {{
            "step": 1,
            "tool": "object_segmentation",
            "params": {{
                "text_prompt": "<the short plain object noun from the question — SAM3 only segments short terms well, so use the question's own word (1-2 words)>",
                "object_category": "<same short noun — 1-3 words>"
            }},
            "expected_evidence": "..."
        }},
        ...
    ]
}}

CRITICAL RULE for `chosen_skill`:
- `chosen_skill` MUST be exactly one of the SKILL names from the "Retrieved SKILLs" list above, OR null.
- `chosen_skill` MUST NOT be a tool name.
- If the Retrieved SKILLs list is empty or none match, set `chosen_skill: null` and lay out `plan` as raw tool calls.

Choosing between SKILL and raw-tool plan:
- If a retrieved SKILL's `When to Use` clearly matches this question AND its `success_rate` is reasonable (>= 0.3, or seeded SKILL), pick it and fill `skill_params`.
- If no SKILL fits, set `chosen_skill: null`, `skill_params: {{}}`, and lay out `plan` as raw tool calls.
- Only resort to a raw-tool plan when you need a genuinely different strategy that no retrieved SKILL covers. If a suitable SKILL already exists, always prefer it — do NOT re-plan the same strategy from scratch.

Category assignment rules:
- `scene_type`: infer from the images — look for furniture arrangement, lighting, fixtures. Pick from the Room Types list; if no match, add a short label (e.g. "studio apartment"). Use null only if truly unrecognizable. This is used to look up and update scene-scale priors.
- `object_category` (inside each object_segmentation step):
  - This is the CANONICAL CLASS name — NOT a visual description.
  - text_prompt="small wooden chair next to window" → object_category="chair"
  - text_prompt="white IKEA bookcase on the left wall" → object_category="bookcase"
  - text_prompt="trash can near the door" → object_category="trash can"
  - Pick from the Object Categories list. If nothing fits, use the simplest 1-3 word class name.
  - This key is used to store and retrieve size priors — consistency across samples matters.
- For non-segmentation tools (depth_estimation, distance_computation, etc.), no object_category is needed.

Raw-tool parameter rules:
- For distance_computation, direction_computation, and object_size_computation, prefer object-name parameters from the question, such as `obj_a`, `obj_b`, `object_name`, `viewpoint_target`, `reference_target`, `target_target`, and `facing_target`. The tool layer will call Checker and resolve the final instance.
- Never invent placeholder IDs like "<id_of_chair>" or "chair_instance_id". Use integer IDs only when a previous context explicitly contains an unambiguous concrete ID.
- For direction questions, if the question says "standing by X" or "from X", set `viewpoint_type="object"` and `viewpoint_target="X"`. If it says "facing Y", set `facing_target="Y"`.
- For object_segmentation, SAM3 tracks instances across video frames; use short plain object nouns so those tracks are stable.

=== Object Categories ===
{OBJECT_CATEGORY_LIST}

=== Room Types ===
{ROOM_TYPE_LIST}
"""


def planner_user_message(question: str, image_paths: list[str],
                         max_preview_images: int = 32) -> list[dict]:
    content: list[dict] = []
    header = f"Question: {question}\n\nInput: {len(image_paths)} frame(s)"
    if len(image_paths) > 1:
        header += " — multi-frame/video, the scene is observed from several viewpoints."
    else:
        header += " — single image."
    content.append({"type": "text", "text": header})

    for idx in _select_preview_indices(len(image_paths), max_preview_images):
        try:
            content.append({"type": "text", "text": f"Input frame {idx}:"})
            b64 = _encode_image_base64(image_paths[idx])
            suffix = Path(image_paths[idx]).suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        except Exception:
            pass
    return content


class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, sample, memory_context: str,
             retrieved_skills, tool_descriptions: str,
             max_preview_images: int = 32,
             image_paths: list[str] | None = None) -> dict:
        skills_brief = "\n\n".join(s.brief() for s in retrieved_skills)
        sys_prompt = planner_system_prompt(tool_descriptions, skills_brief, memory_context)
        user_content = planner_user_message(sample.question, image_paths or sample.image_paths,
                                            max_preview_images=max_preview_images)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.llm.chat_json(messages)


# ─── helpers ──────────────────────────────────────────────────────────

def _encode_image_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _select_preview_indices(n: int, max_images: int) -> list[int]:
    if n == 0:
        return []
    if n <= max_images:
        return list(range(n))
    return list(np.linspace(0, n - 1, max_images).astype(int))


def format_plan(plan: dict) -> str:
    """Compact one-liner-per-step rendering; used by other roles' prompts."""
    lines = []
    for step in plan.get("plan", []):
        status = step.get("status", "pending")
        lines.append(f"  Step {step['step']}: {step['tool']}({step.get('params', {})}) "
                     f"-> {step.get('expected_evidence', '')} [{status}]")
    return "\n".join(lines) if lines else "  (empty plan)"
