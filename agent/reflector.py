"""Reflector role for SpatialMem.

Three responsibilities:

  1. reflect(question, plan, context_summary, failure_patterns) — after a
     round of tool calls, decide continue vs terminate and (if continue)
     propose follow-up actions.

  2. reconstruct(sample, wrong_result, tool_descriptions) — training-only.
     Given a failed trajectory, diagnose the root cause and propose a
     corrected plan for verification re-run. Trust is granted only if
     re-execution actually produces the correct answer.

  3. finalize(question, answer_format, context_summary, memory_context) —
     extract the final answer in the required format from the accumulated
     context. Replaces the old Summarizer.
"""

from __future__ import annotations

from agent.llm_client import LLMClient
from agent.planner import FRAMEWORK_OVERVIEW, format_plan


def _with_visual_evidence(text: str, visual_evidence: list[dict] | None = None):
    if not visual_evidence:
        return text
    return [{"type": "text", "text": text}] + list(visual_evidence)


# ─── reflect ──────────────────────────────────────────────────────────

def reflector_system_prompt() -> str:
    return f"""{FRAMEWORK_OVERVIEW}

You are the REFLECTOR. After a round of tool calls, decide whether the accumulated evidence is SUFFICIENT to answer the question.

Output JSON:
{{
    "confidence": 0.0-1.0,
    "confidence_reasoning": "what evidence do we have, what's missing, is it enough?",
    "evidence_gaps": ["specific missing piece 1", ...],
    "decision": "continue" | "terminate",
    "next_actions": [
        {{"tool": "tool_name", "params": {{...}}, "reason": "why this specific additional step is needed"}}
    ]
}}

TERMINATE (confidence >= 0.8) when:
- Distance question: you have a computed distance value from distance_computation.
- Counting question: you have the instance count from instance_counting.
- Spatial-relation question: you have direction labels from direction_computation.
- Object-size question: you have a computed size value from object_size_computation.
- Room-size question: you have extents from scene_size_computation.
- Checker raw-sample direct answer exists because a required SAM3 target had zero tracks.

CONTINUE when:
- A tool in the plan failed and needs retry with different parameters.
- Required evidence for the question hasn't been produced yet.
- Ambiguity remains that a different tool (BEV, NVS) could resolve.

Do NOT continue when you already have the answer — do not add unnecessary tools.
Set next_actions = [] when decision is "terminate".
"""


def reflector_user_message(question: str, plan: dict,
                           context_summary: str, failure_patterns: str = "") -> str:
    text = f"Question: {question}\n\n"
    text += f"Executed plan:\n{format_plan(plan)}\n\n"
    text += f"Accumulated evidence:\n{context_summary}\n"
    if failure_patterns:
        text += f"\nKnown pitfalls to watch for:\n{failure_patterns}\n"
    text += "\nIs the evidence sufficient? Output JSON."
    return text


# ─── reconstruct ──────────────────────────────────────────────────────

def reconstructor_system_prompt(tool_descriptions: str) -> str:
    return f"""{FRAMEWORK_OVERVIEW}

You are the REFLECTOR in TRAINING mode. The agent produced an answer that disagrees with ground truth. Your job:

1. Analyze the failed trajectory (plan + tool outputs + predicted answer + GT).
2. Diagnose the most likely root cause.
3. Propose a CORRECTED plan (ordered raw tool calls with concrete params) that you believe will produce the correct answer.

Your proposed plan will be RE-EXECUTED on the same sample. Only if the re-run yields the correct answer is your diagnosis trusted and distilled into a SKILL update. Speculative diagnoses that don't verify are discarded.

{tool_descriptions}

Output JSON:
{{
    "diagnosis": "1-3 sentences: why the original attempt likely produced the wrong answer",
    "corrected_plan": [
        {{"tool": "tool_name", "params": {{...}}, "reason": "1 sentence: why this fixes the failure"}}
    ]
}}

If you cannot form a concrete corrected plan (e.g. the failure is beyond the tools), output `"corrected_plan": []` and let the system record the sample as unsolved.
"""


def reconstructor_user_message(question: str, gt_answer: str,
                               predicted: str, plan: dict,
                               context_summary: str) -> str:
    text = f"Question: {question}\n"
    text += f"Ground truth answer: {gt_answer}\n"
    text += f"Agent's (wrong) answer: {predicted}\n\n"
    text += f"Executed plan:\n{format_plan(plan)}\n\n"
    text += f"Accumulated evidence:\n{context_summary}\n\n"
    text += "Diagnose the failure and propose a corrected plan. Output JSON."
    return text


# ─── finalize ─────────────────────────────────────────────────────────

def finalizer_system_prompt() -> str:
    return f"""{FRAMEWORK_OVERVIEW}

You are the FINALIZER. Given all accumulated evidence, produce the FINAL answer.

Output JSON:
{{
    "answer": "the answer value ONLY — see format rules",
    "reasoning_chain": "brief reasoning: keep it concise, focusing only on the decisive evidence and final option mapping when relevant",
    "confidence": 0.0-1.0,
    "key_evidence": ["the 1-3 most important pieces of evidence that determined the answer"]
}}

Keep the JSON reasonably concise. Prefer short reasoning_chain text and avoid unnecessary long explanations or full option-list restatements.

Answer format rules (STRICT — wrong format = wrong answer):
- Numeric fill: ONLY the number. "0.6", "3", "16". No units, no words.
- Multiple choice: ONLY one letter (A/B/C/D). Nothing else.
- Yes/No: ONLY "Yes" or "No".

Multiple-choice selection rule:
- First derive the semantic answer from evidence.
- Then compare that derived answer against EVERY option text in the question.
- Return the letter whose option text exactly matches the derived answer.
- If your reasoning says one option text but your answer letter points to another option, the answer is wrong. Re-check the option mapping before finalizing.
- For ordering questions, preserve the full comma-separated order and match it exactly to the listed option text.

How to derive answers from evidence:
- [distance_computation] → distance in meters, rounded to 1 decimal
- [instance_counting] → total_unique count
- [scene_size_computation] → floor_area (rounded to integer) for room-size questions
- [object_size_computation] → bbox_size_xyz and bbox volume for object-size questions; if the question asks a specific width/height/depth/longest/shortest dimension, derive it from bbox_size_xyz and then match the requested answer format
- [direction_computation] → match the direction label to the multiple-choice options; use angle_from_forward_deg when the question defines front/back thresholds or asks about angular relation

Size sanity check via memory priors (if present in the memory context):
- If your computed object size is >3× or <0.3× the prior mean, treat it as suspicious and lower confidence — likely a depth or segmentation error.
- Only intervene when the discrepancy is extreme; do NOT blindly override plausible measurements.
"""


def finalizer_user_message(question: str, answer_format: str,
                           context_summary: str,
                           memory_context: str = "") -> str:
    format_instructions = {
        "fill":     "Answer with a SINGLE NUMBER only.",
        "select":   "Answer with EXACTLY ONE LETTER: A, B, C, or D. Nothing else.",
        "judge":    "Answer with exactly 'Yes' or 'No'.",
        "sentence": "Answer in one complete sentence.",
    }
    text = f"Question: {question}\n\n"
    text += f"REQUIRED answer format: {format_instructions.get(answer_format, answer_format)}\n\n"
    text += f"All accumulated evidence:\n{context_summary}\n"
    if memory_context and memory_context != "No relevant memory found.":
        text += f"\nMemory context (for sanity checks):\n{memory_context}\n"
    text += "\nProduce the final answer. Output JSON."
    return text


# ─── class ────────────────────────────────────────────────────────────

class Reflector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def reflect(self, question: str, plan: dict,
                context_summary: str, failure_patterns: str = "",
                visual_evidence: list[dict] | None = None) -> dict:
        text = reflector_user_message(
            question, plan, context_summary, failure_patterns)
        messages = [
            {"role": "system", "content": reflector_system_prompt()},
            {"role": "user", "content": _with_visual_evidence(text, visual_evidence)},
        ]
        return self.llm.chat_json(messages)

    def reconstruct(self, sample, plan: dict, predicted: str,
                    context_summary: str, tool_descriptions: str,
                    visual_evidence: list[dict] | None = None) -> dict:
        text = reconstructor_user_message(
            sample.question, sample.gt_answer, predicted, plan, context_summary)
        messages = [
            {"role": "system", "content": reconstructor_system_prompt(tool_descriptions)},
            {"role": "user", "content": _with_visual_evidence(text, visual_evidence)},
        ]
        # reconstruct produces a full corrected plan; give it headroom.
        return self.llm.chat_json(messages, max_tokens=6144)

    def finalize(self, question: str, answer_format: str,
                 context_summary: str, memory_context: str = "",
                 visual_evidence: list[dict] | None = None) -> dict:
        text = finalizer_user_message(
            question, answer_format, context_summary, memory_context)
        messages = [
            {"role": "system", "content": finalizer_system_prompt()},
            {"role": "user", "content": _with_visual_evidence(text, visual_evidence)},
        ]
        return self.llm.chat_json(messages)
