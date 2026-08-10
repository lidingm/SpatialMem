"""Orchestrator: per-sample reasoning loop for SpatialMem.

Ties Planner → SKILL invocation OR raw-tool loop with Reasoner → Reflector →
Finalizer. Also exposes `run_with_plan()` used by evolve.py to verify
reconstructed trajectories after a wrong answer.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import MAX_FRAMES_PER_SAMPLE, MAX_ROUNDS, CONFIDENCE_THRESHOLD
from agent.data_loader import SpatialSample
from agent.memory import Memory
from agent.planner import Planner
from agent.reasoner import Reasoner
from agent.reflector import Reflector
from agent.skill_lib import SkillLib
from agent.tools import ToolRegistry


# Hard whitelist of tool names. Reasoner / Reflector next_actions must
# resolve to one of these; anything else is silently rejected so we don't
# waste rounds on hallucinated tools like "first_appearance_analysis".
KNOWN_TOOLS = frozenset([
    "depth_estimation", "bev_generation", "novel_view_synthesis",
    "object_segmentation", "annotation_localization",
    "instance_3d_localization", "instance_counting",
    "distance_computation", "direction_computation",
    "object_size_computation",
    "scene_size_computation",
])


_DIRECTION_SKILL_PARAM_KEYS = {
    "viewpoint_type", "viewpoint_target", "viewpoint_frame",
    "reference_target", "target_target", "facing_target",
}


def _fill_skill_params_from_plan(chosen_skill: str | None,
                                 skill_params: dict,
                                 plan_steps: list[dict]) -> dict:
    """Use the concrete planned tool call to repair incomplete skill params.

    Planner sometimes fills the human-readable raw-tool plan correctly but leaves
    `skill_params` incomplete. The skill invocation only sees `skill_params`, so
    copy direction parameters from the planned `direction_computation` step before
    executing judge_direction.
    """
    params = dict(skill_params or {})
    if chosen_skill != "judge_direction":
        return params
    for step in plan_steps or []:
        if step.get("tool") != "direction_computation":
            continue
        step_params = step.get("params") or {}
        for key in _DIRECTION_SKILL_PARAM_KEYS:
            value = step_params.get(key)
            if value not in (None, "", "null"):
                params[key] = value
        break
    return params


def _object_names_from_plan(plan: dict) -> list[str]:
    """Extract explicit Planner/skill target names, avoiding broad question regexes."""
    names: list[str] = []

    def add(value: Any) -> None:
        if value in (None, "", "null"):
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        if not isinstance(value, str):
            return
        name = value.strip().lower()
        if name and name not in names:
            names.append(name)

    add((plan or {}).get("object_targets") or [])
    param_sources = [((plan or {}).get("skill_params") or {})]
    param_sources.extend((step.get("params") or {}) for step in ((plan or {}).get("plan") or []))
    for params in param_sources:
        for key in (
            "object_name", "object_category", "text_prompt", "target", "obj",
            "obj_a", "obj_b", "reference_target", "target_target",
            "viewpoint_target", "facing_target",
        ):
            add(params.get(key))
    return names


@dataclass
class AgentResult:
    sample_id: str
    question: str
    gt_answer: str
    predicted_answer: str
    reasoning_chain: str
    confidence: float
    tool_calls: list[dict] = field(default_factory=list)
    num_rounds: int = 0
    plan: dict = field(default_factory=dict)
    success: bool | None = None
    task_type: str = ""
    task_category: str = ""
    answer_format: str = ""
    duration_seconds: float = 0.0
    skills_used: list[str] = field(default_factory=list)
    chosen_skill: str | None = None
    memory_context: str = ""
    final_context: str = ""


class Orchestrator:
    def __init__(
        self,
        tools: ToolRegistry,
        memory: Memory,
        skill_lib: SkillLib,
        planner: Planner,
        reasoner: Reasoner,
        reflector: Reflector,
        max_rounds: int = MAX_ROUNDS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        verbose: bool = True,
        allow_post_skill_reflection: bool = True,
    ):
        self.tools = tools
        self.memory = memory
        self.skill_lib = skill_lib
        self.planner = planner
        self.reasoner = reasoner
        self.reflector = reflector
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.allow_post_skill_reflection = allow_post_skill_reflection

    # ─── Public entry points ──────────────────────────────────────────

    def run(self, sample: SpatialSample) -> AgentResult:
        """Full pipeline: planner → SKILL or raw-tool loop → finalize."""
        t0 = time.time()
        self._prepare_run(sample)

        # Planner 阶段：用正则解析的 object_names（此时还没跑分割，没有 seg_categories）
        memory_ctx = self.memory.format_context(
            sample.task_category, object_names=sample.object_names, scene_type=None)
        retrieved = self.skill_lib.retrieve(sample.task_category, sample.question, top_k=3)
        if self.verbose:
            print(f"\n{'='*60}\nSample {sample.id} | type={sample.task_type} "
                  f"(category={sample.task_category})\nQ: {sample.question}\n"
                  f"GT: {sample.gt_answer}\nRetrieved SKILLs: "
                  f"{[s.name for s in retrieved]}\n{'='*60}")

        plan = self.planner.plan(
            sample, memory_ctx, retrieved, self.tools.tool_descriptions,
            image_paths=self.tools.context.get("frame_paths"))
        chosen_skill_name = plan.get("chosen_skill") or None
        plan["skill_params"] = _fill_skill_params_from_plan(
            chosen_skill_name, plan.get("skill_params", {}) or {}, plan.get("plan", []))
        # Cache Planner-inferred scene_type in tool context so evolve.py can
        # bucket scene_scale_prior updates under the right key (bedroom, kitchen, ...)
        # instead of everything collapsing into "indoor_room".
        scene_type = (plan.get("scene_type") or "").strip().lower()
        if scene_type and scene_type != "null":
            self.tools.context["scene_type"] = scene_type
        skills_used: list[str] = []

        if self.verbose:
            print(f"[PLANNER] analysis: {plan.get('analysis', '')}")
            print(f"[PLANNER] scene_type: {scene_type or '(none)'}")
            print(f"[PLANNER] chosen_skill: {chosen_skill_name or '(none — raw-tool loop)'}")
            if chosen_skill_name:
                print(f"[PLANNER] skill_params: {plan.get('skill_params', {})}")
            for step in plan.get("plan", []):
                print(f"[PLANNER] step {step.get('step')}: {step.get('tool')} "
                      f"params={step.get('params', {})} | {step.get('expected_evidence', '')}")

        if chosen_skill_name:
            if self.verbose:
                print(f"[PLANNER] chose SKILL: {chosen_skill_name}")
            skill = self.skill_lib.get(chosen_skill_name)
            num_rounds = 0
            if skill is None:
                if self.verbose:
                    print(f"[PLANNER] SKILL '{chosen_skill_name}' not found; "
                          f"falling back to raw-tool loop.")
                num_rounds = self._raw_tool_loop(sample, plan)
            else:
                params = plan.get("skill_params", {}) or {}
                checker_notes = skill.sections.get("Checker", "").strip()
                self.tools.set_checker_notes(checker_notes)
                self.tools.append_context_text(
                    f"Round {self._ordered_round_count() + 1}: selected skill[{skill.name}]")
                num_rounds = 1
                prev_skill_summary_len = len(self.tools.context.get("skill_summaries", []))
                try:
                    inv = self.skill_lib.invoke(skill, sample, self.tools,
                                                self.tools.context, params)
                    self._append_new_skill_summaries(prev_skill_summary_len)
                    prev_skill_summary_len = len(self.tools.context.get("skill_summaries", []))
                    if not inv.get("success", True):
                        summary = str(inv.get("summary", "")).strip() or "skill returned success=False"
                        self.tools.append_context_note(f"skill:{skill.name}", f"FAILED: {summary}")
                        if self.allow_post_skill_reflection:
                            # Evaluation/notebook mode: recover with raw tools after a failed skill.
                            if self.verbose:
                                print(f"[SKILL:{skill.name}] returned failure: "
                                      f"{summary}; falling back to raw-tool loop")
                            self.tools.set_checker_notes("")
                            num_rounds += self._raw_tool_loop(
                                sample, plan, max_rounds=max(0, self.max_rounds - num_rounds))
                        else:
                            # Training mode: keep the failed skill trajectory pure for Path C.
                            skills_used.append(skill.name)
                            if self.verbose:
                                print(f"[SKILL:{skill.name}] returned failure: {summary}; "
                                      "post-skill recovery disabled")
                    else:
                        skills_used.append(skill.name)
                        if self.verbose:
                            print(f"[SKILL:{skill.name}] {inv.get('summary', '')}")
                        if self.allow_post_skill_reflection:
                            reflection = self._reflect_after_round(sample, plan)
                            if not self._should_terminate(reflection):
                                followup_plan = self._plan_from_next_actions(reflection)
                                if followup_plan.get("plan"):
                                    plan["post_skill_plan"] = followup_plan["plan"]
                                    self.tools.set_checker_notes("")
                                    num_rounds += self._raw_tool_loop(
                                        sample, followup_plan,
                                        max_rounds=max(0, self.max_rounds - num_rounds))
                except Exception as e:
                    self._append_new_skill_summaries(prev_skill_summary_len)
                    self.tools.append_context_note(f"skill:{skill.name}", f"FAILED: {e}")
                    if self.verbose:
                        suffix = ("falling back to raw-tool loop" if self.allow_post_skill_reflection
                                  else "post-skill recovery disabled")
                        print(f"[SKILL:{skill.name}] raised: {e!r}; {suffix}")
                    self.tools._tool_log.append({
                        "tool_name": f"skill:{skill.name}",
                        "params": params, "success": False,
                        "result_summary": "", "error": str(e),
                    })
                    if self.allow_post_skill_reflection:
                        self.tools.set_checker_notes("")
                        num_rounds += self._raw_tool_loop(
                            sample, plan, max_rounds=max(0, self.max_rounds - num_rounds))
                    else:
                        # Training mode: mark this as an attempted SKILL so evolve uses Path C.
                        skills_used.append(skill.name)
        else:
            self.tools.set_checker_notes("")
            num_rounds = self._raw_tool_loop(sample, plan)

        plan_object_names = _object_names_from_plan(plan) or sample.object_names
        self.tools.append_task_prior_context(
            sample.task_type, sample.task_category, object_names=plan_object_names)
        ctx_summary = self.tools.get_context_summary()

        # Finalizer 阶段：重建 memory_ctx，用工具执行期间真实存储的 seg_categories
        # 而非 Planner 阶段正则解析的名字，保证存读 key 一致
        final_scene_type = self.tools.context.get("scene_type", "") or scene_type or ""
        seg_cats = list(dict.fromkeys(          # 去重同时保持顺序
            v for v in self.tools.context.get("seg_categories", {}).values() if v
        ))
        final_memory_ctx = self.memory.format_context(
            sample.task_category,
            object_categories=seg_cats or None,
            object_names=plan_object_names if not seg_cats else None,
            scene_type=final_scene_type or None,
        )
        final = self.reflector.finalize(
            sample.question, sample.answer_format, ctx_summary, final_memory_ctx,
            visual_evidence=self.tools.get_visual_evidence_content())

        return AgentResult(
            sample_id=sample.id, question=sample.question,
            gt_answer=sample.gt_answer,
            predicted_answer=str(final.get("answer", "")),
            reasoning_chain=final.get("reasoning_chain", ""),
            confidence=float(final.get("confidence", 0.0)),
            tool_calls=list(self.tools.tool_log),
            num_rounds=num_rounds, plan=plan,
            task_type=sample.task_type,
            task_category=sample.task_category,
            answer_format=sample.answer_format,
            duration_seconds=time.time() - t0,
            skills_used=skills_used,
            chosen_skill=chosen_skill_name,
            memory_context=final_memory_ctx,
            final_context=ctx_summary,
        )

    def run_with_plan(self, sample: SpatialSample,
                      corrected_plan: list[dict]) -> AgentResult:
        """Execute a plan directly (no planner, no SKILL). Used by evolve.py
        to verify a Reflector-reconstructed trajectory."""
        t0 = time.time()
        self._prepare_run(sample)

        memory_ctx = self.memory.format_context(
            sample.task_category, object_names=sample.object_names)
        plan = {"plan": [{"step": i + 1,
                          "tool": step["tool"],
                          "params": step.get("params", {}),
                          "expected_evidence": step.get("reason", "")}
                         for i, step in enumerate(corrected_plan)]}

        frame_paths = self.tools.context.get("frame_paths_prepared", sample.image_paths)
        for step in plan["plan"]:
            self.tools.execute_tool(step["tool"], step.get("params", {}), frame_paths)

        plan_object_names = _object_names_from_plan(plan) or sample.object_names
        self.tools.append_task_prior_context(
            sample.task_type, sample.task_category, object_names=plan_object_names)
        ctx_summary = self.tools.get_context_summary()
        # run_with_plan 也用 seg_categories 查先验（和 run() 保持一致）
        seg_cats = list(dict.fromkeys(
            v for v in self.tools.context.get("seg_categories", {}).values() if v
        ))
        final_scene_type = self.tools.context.get("scene_type", "")
        final_memory_ctx = self.memory.format_context(
            sample.task_category,
            object_categories=seg_cats or None,
            object_names=plan_object_names if not seg_cats else None,
            scene_type=final_scene_type or None,
        )
        final = self.reflector.finalize(
            sample.question, sample.answer_format, ctx_summary, final_memory_ctx,
            visual_evidence=self.tools.get_visual_evidence_content())

        return AgentResult(
            sample_id=sample.id, question=sample.question,
            gt_answer=sample.gt_answer,
            predicted_answer=str(final.get("answer", "")),
            reasoning_chain=final.get("reasoning_chain", ""),
            confidence=float(final.get("confidence", 0.0)),
            tool_calls=list(self.tools.tool_log),
            num_rounds=1, plan=plan,
            task_type=sample.task_type,
            task_category=sample.task_category,
            answer_format=sample.answer_format,
            duration_seconds=time.time() - t0,
            skills_used=[], chosen_skill=None,
            memory_context=final_memory_ctx, final_context=ctx_summary,
        )

    # ─── Internals ────────────────────────────────────────────────────

    def _prepare_run(self, sample: SpatialSample) -> None:
        self.tools.reset_context()
        self.tools.set_checker_notes("")
        # Use a per-process output directory so concurrent processes don't overwrite each other.
        from agent.config import OUTPUTS_DIR
        import os
        self.tools.output_root = OUTPUTS_DIR / f"proc_{os.getpid()}"
        shutil.rmtree(self.tools.output_root, ignore_errors=True)
        for subdir in ("da3", "sam3", "spatial"):
            (self.tools.output_root / subdir).mkdir(parents=True, exist_ok=True)
        self.tools.context["output_root"] = str(self.tools.output_root)
        frame_paths = _downsample_frames(sample.image_paths, MAX_FRAMES_PER_SAMPLE)
        # Store in ctx for SKILLs and Reasoner to read
        self.tools.context["frame_paths"] = frame_paths
        self.tools.context["frame_paths_prepared"] = frame_paths
        # VSI has no annotations; keep empty dict for compatibility
        self.tools.context["sample_annotations"] = sample.annotations
        self.tools.context["annotated_paths"] = frame_paths
        self.tools.context["question"] = sample.question
        self.tools.context["answer_format"] = sample.answer_format
        self.tools.context["task_type"] = sample.task_type
        self.tools.context["task_category"] = sample.task_category

    def _ordered_round_count(self) -> int:
        return sum(
            1 for item in self.tools.context.get("ordered_context", [])
            if str(item).strip().startswith("Round ")
        )

    def _append_new_skill_summaries(self, prev_len: int) -> None:
        for s in self.tools.context.get("skill_summaries", [])[prev_len:]:
            if s:
                self.tools.append_context_text(str(s))

    def _append_reflection_context(self, reflection: dict) -> None:
        refl_lines = [
            f"confidence={float(reflection.get('confidence', 0.0)):.2f}, "
            f"decision={reflection.get('decision', '')}"
        ]
        reason = str(reflection.get("confidence_reasoning", "")).strip()
        if reason:
            refl_lines.append(f"  reason={reason}")
        gaps = reflection.get("evidence_gaps") or []
        if gaps:
            refl_lines.append(f"  evidence_gaps={gaps}")
        next_actions_raw = reflection.get("next_actions", [])
        if next_actions_raw:
            refl_lines.append(f"  next_actions={next_actions_raw}")
        self.tools.append_context_note("reflector", "\n".join(refl_lines))

    def _reflect_after_round(self, sample: SpatialSample, plan: dict) -> dict:
        reflection = self.reflector.reflect(
            sample.question, plan, self.tools.get_context_summary(),
            visual_evidence=self.tools.get_visual_evidence_content())
        self._append_reflection_context(reflection)
        if self.verbose:
            print(f"[REFLECTOR] confidence={reflection.get('confidence', 0):.2f} "
                  f"decision={reflection.get('decision')}")
        return reflection

    def _should_terminate(self, reflection: dict) -> bool:
        return (reflection.get("decision") == "terminate" or
                float(reflection.get("confidence", 0.0)) >= self.confidence_threshold)

    def _valid_next_actions(self, reflection: dict) -> list[dict]:
        return [
            a for a in (reflection.get("next_actions") or [])
            if a.get("tool") in KNOWN_TOOLS
        ]

    def _plan_from_next_actions(self, reflection: dict) -> dict:
        actions = self._valid_next_actions(reflection)
        return {"plan": [
            {
                "step": i + 1,
                "tool": action["tool"],
                "params": action.get("params", {}),
                "expected_evidence": action.get("reason", ""),
            }
            for i, action in enumerate(actions)
        ]}

    def _raw_tool_loop(self, sample: SpatialSample, plan: dict,
                       max_rounds: int | None = None) -> int:
        """Run raw-tool rounds; return number of rounds executed."""
        num_rounds = 0
        frame_paths = self.tools.context["frame_paths"]
        base_round = self._ordered_round_count()
        round_limit = self.max_rounds if max_rounds is None else max_rounds
        for round_idx in range(round_limit):
            num_rounds += 1
            self.tools.append_context_text(f"Round {base_round + round_idx + 1}: raw-tool loop")
            if self.verbose:
                print(f"\n--- Round {round_idx + 1} ---")

            for step_idx, step in enumerate(plan.get("plan", [])):
                if step.get("status") == "done":
                    continue
                if self.verbose:
                    print(f"[REASONER] step {step_idx + 1}: planned {step['tool']}")
                action = self.reasoner.next_action(
                    plan, step_idx, self.tools.get_context_summary(),
                    sample.question, self.tools.tool_descriptions)
                planned_tool = step["tool"]
                tool_name = action.get("tool", planned_tool)
                # Guard against Reasoner hallucinating tool names: fall back to the planned tool.
                if tool_name not in KNOWN_TOOLS:
                    if self.verbose:
                        print(f"  ! Reasoner picked unknown tool {tool_name!r}; "
                              f"falling back to planned {planned_tool!r}")
                    tool_name = planned_tool
                    if tool_name not in KNOWN_TOOLS:
                        step["status"] = "failed"
                        self.tools.append_context_note(planned_tool, "FAILED: unknown planned tool")
                        continue
                params = action.get("params", step.get("params", {}))
                r = self.tools.execute_tool(tool_name, params, frame_paths)
                step["status"] = "done" if r["success"] else "failed"
                if self.verbose:
                    status = "OK" if r["success"] else f"FAILED: {r['error']}"
                    print(f"  -> {tool_name}: {status}")
                    if r["success"]:
                        print(f"     {r['result_summary']}")

            reflection = self._reflect_after_round(sample, plan)

            if self._should_terminate(reflection):
                break
            next_actions = self._valid_next_actions(reflection)
            if not next_actions:
                break
            base = len(plan.get("plan", []))
            for i, action in enumerate(next_actions):
                plan.setdefault("plan", []).append({
                    "step": base + i + 1,
                    "tool": action["tool"],
                    "params": action.get("params", {}),
                    "expected_evidence": action.get("reason", ""),
                })
        return num_rounds

def _downsample_frames(paths: list[str], k: int) -> list[str]:
    if len(paths) <= k:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, k).astype(int)
    return [paths[i] for i in idx]
