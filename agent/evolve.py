"""Self-evolution engine for SpatialMem.

After each training sample, `evolve_after_sample` dispatches to one of four
paths depending on (correct/wrong) × (SKILL used / no SKILL used). Every path
that produces a durable update goes through an LLM-driven **distillation**
step — we never append raw trajectories or verbatim GT diffs.

  Path A  ✅ + SKILL:    reinforce + periodic distill of the SKILL
  Path B  ✅ + no SKILL:  distill a new SKILL draft into skills/pending/
  Path C  ❌ + SKILL:    reconstruct → re-run to verify → distill a pitfall
  Path D  ❌ + no SKILL:  reconstruct → re-run to verify → distill a new SKILL

If the reconstructed plan does NOT verify, the sample is logged to
memory/unsolved_cases.jsonl and no SKILL is touched.
"""

from __future__ import annotations

import ast
import json
import threading
from pathlib import Path

import numpy as np

from agent.config import (
    DISTILL_TRIGGER_N, DISTILL_WINDOW, REFLECT_MAX_ATTEMPTS,
    SKILLS_DIR, BOOTSTRAP_TRIGGER_N,
)
from agent.llm_client import LLMClient
from agent.memory import Memory
from agent.orchestrator import Orchestrator, AgentResult
from agent.reflector import Reflector
from agent.skill_lib import SkillLib, Skill


# ─── Entry point ──────────────────────────────────────────────────────

def _record_distilled_skill(updates: dict, skill_lib: SkillLib, name: str | None) -> None:
    """Classify distill_new_skill result as new pending skill or existing-skill update."""
    if not name:
        return
    if (skill_lib.pending_dir / name).exists():
        updates["pending_saved"] = name
    else:
        updates["skill_updates"].append(f"{name}:when_to_use")

def evolve_after_sample(
    result: AgentResult,
    sample,
    memory: Memory,
    skill_lib: SkillLib,
    reflector: Reflector,
    orchestrator: Orchestrator,
    llm: LLMClient,
    evaluate_answer,
    verbose: bool = False,
) -> dict:
    """Dispatch a completed sample result to one of the 4 evolution paths.

    Also proactively logs uncovered-category trajectories and triggers a
    category-level SKILL bootstrap when the accumulated count crosses
    BOOTSTRAP_TRIGGER_N. This is what turns "we accidentally succeeded once"
    into "we noticed we keep seeing this category, let's design a SKILL".

    Returns a summary dict describing what was updated (useful for logging).
    """
    trajectory = _summarize_trajectory(result)
    updates: dict = {"path": None, "skill_updates": [], "memory_updates": [],
                     "pending_saved": None, "bootstrapped": None}

    # Always log samples where no SKILL was chosen; this feeds category bootstrap.
    if not result.chosen_skill:
        _log_uncovered_category(sample, result, memory.root / "category_trajectories")

    if result.success and result.skills_used:
        updates["path"] = "A"
        for name in result.skills_used:
            skill_lib.record_call(name, success=True, trajectory_summary=trajectory)
            s = skill_lib.get(name)
            if s is None:
                continue
            if s.success_count > 0 and s.success_count % DISTILL_TRIGGER_N == 0:
                updated = distill_success(s, skill_lib, llm, verbose=verbose)
                if updated:
                    updates["skill_updates"].append(f"{name}:when_to_use")
        _update_memory_from_tools(orchestrator.tools.context, memory, sample, updates)

    elif result.success and not result.skills_used:
        updates["path"] = "B"
        # Only distill a new SKILL if the trajectory involved multiple tools
        # (single-tool "trajectories" are not worth encoding as a SKILL).
        if len(trajectory["tool_calls"]) >= 2:
            distilled = distill_new_skill(sample, trajectory, llm, skill_lib, verbose=verbose)
            _record_distilled_skill(updates, skill_lib, distilled)
        _update_memory_from_tools(orchestrator.tools.context, memory, sample, updates)

    elif not result.success and result.skills_used:
        updates["path"] = "C"
        for name in result.skills_used:
            skill_lib.record_call(name, success=False, trajectory_summary=trajectory)

        corrected = _reflect_and_verify(
            sample, result, reflector, orchestrator, evaluate_answer,
            tool_descriptions=orchestrator.tools.tool_descriptions,
            verbose=verbose,
        )
        if corrected is not None:
            _, corrected_result, _ = corrected
            for name in result.skills_used:
                skill = skill_lib.get(name)
                if skill is None:
                    continue
                lessons = distill_failure_lessons(
                    sample, result, corrected_result, llm,
                    skill=skill, verbose=verbose)
                pitfall_text = lessons.get("pitfall", "")
                example_text = lessons.get("example", "")
                checker_text = lessons.get("checker", "")
                if pitfall_text:
                    skill_lib.append_pitfall(name, pitfall_text)
                    skill_lib.bump_version(name)
                    updates["skill_updates"].append(f"{name}:pitfall")
                if example_text:
                    if skill_lib.append_example(name, example_text, sample_id=str(sample.id)):
                        skill_lib.bump_version(name)
                        updates["skill_updates"].append(f"{name}:example")
                if checker_text:
                    if skill_lib.append_checker_note(name, checker_text):
                        skill_lib.bump_version(name)
                        updates["skill_updates"].append(f"{name}:checker")

            corrected_traj = _summarize_trajectory(corrected_result)
            corrected_traj["reconstructed_from_wrong"] = True
            corrected_traj["failed_chosen_skill"] = result.chosen_skill
            distilled = distill_new_skill(sample, corrected_traj, llm, skill_lib, verbose=verbose)
            _record_distilled_skill(updates, skill_lib, distilled)
        else:
            memory.record_unsolved(
                sample.id, sample.task_type,
                result.predicted_answer, sample.gt_answer,
                reason="reconstruct verification failed",
                trajectory_summary=trajectory,
            )
            updates["memory_updates"].append("unsolved")

    else:
        updates["path"] = "D"
        corrected = _reflect_and_verify(
            sample, result, reflector, orchestrator, evaluate_answer,
            tool_descriptions=orchestrator.tools.tool_descriptions,
            verbose=verbose,
        )
        if corrected is not None:
            corrected_plan, corrected_result, _ = corrected
            corrected_traj = _summarize_trajectory(corrected_result)
            corrected_traj["reconstructed_from_wrong"] = True
            distilled = distill_new_skill(sample, corrected_traj, llm, skill_lib, verbose=verbose)
            _record_distilled_skill(updates, skill_lib, distilled)
        else:
            memory.record_unsolved(
                sample.id, sample.task_type,
                result.predicted_answer, sample.gt_answer,
                reason="no SKILL used; reconstruct verification failed",
                trajectory_summary=trajectory,
            )
            updates["memory_updates"].append("unsolved")

    # After the per-sample paths, check whether this category deserves a
    # proactive bootstrap. This runs at most once per category.
    if not result.chosen_skill:
        bootstrapped = _maybe_bootstrap_category(
            sample.task_category, skill_lib, memory, llm, verbose=verbose,
        )
        if bootstrapped:
            updates["bootstrapped"] = bootstrapped

    return updates


# ─── Reflect-and-verify helper ────────────────────────────────────────

def _print_reconstruct_tool_details(verify: AgentResult) -> None:
    """Print the concrete tool execution trace from a reconstructed rerun."""
    calls = list(verify.tool_calls or [])
    if not calls:
        print("[RECONSTRUCT TOOL CALLS] none")
        return

    for idx, tc in enumerate(calls, start=1):
        status = "OK" if tc.get("success") else "FAIL"
        tool = tc.get("tool_name", "?")
        params = tc.get("params", {})
        print(f"[RECONSTRUCT TOOL {idx:02d}] {status} {tool} params={params}")
        detail = tc.get("result_summary") if tc.get("success") else tc.get("error")
        detail = str(detail or "").strip()
        if detail:
            for line in detail.splitlines():
                print(f"     {line}")


def _reflect_and_verify(sample, result, reflector, orchestrator,
                        evaluate_answer, tool_descriptions,
                        verbose: bool):
    """Run reflector.reconstruct → orchestrator.run_with_plan; return the
    tuple (corrected_plan, corrected_result, verify_ok) iff verified, else None."""
    for attempt in range(REFLECT_MAX_ATTEMPTS):
        rc = reflector.reconstruct(
            sample, result.plan, result.predicted_answer,
            result.final_context, tool_descriptions,
            visual_evidence=orchestrator.tools.get_visual_evidence_content())
        corrected_plan = rc.get("corrected_plan", [])
        if not corrected_plan:
            return None
        plan_str = " -> ".join(s.get("tool", "?") for s in corrected_plan)
        print(f"[RECONSTRUCT] attempt {attempt+1}  diagnosis: {rc.get('diagnosis', '(none)')}")
        print(f"[RECONSTRUCT] new plan: {plan_str}")
        verify = orchestrator.run_with_plan(sample, corrected_plan)
        _print_reconstruct_tool_details(verify)
        verify.success = evaluate_answer(
            verify.predicted_answer, sample.gt_answer,
            sample.answer_format, sample.task_type)
        status = "✓ CORRECT" if verify.success else "✗ STILL WRONG"
        print(f"[RECONSTRUCT] {status} | pred={verify.predicted_answer!r}")
        if verify.success:
            return corrected_plan, verify, True
    return None


# ─── LLM-driven distillation ──────────────────────────────────────────

def _structured_ctx_reference() -> str:
    """Runtime ctx fields that generated SKILL code may read after tools run."""
    return """Structured runtime ctx reference for execute_py_body:
- Do NOT parse human-readable result_summary text when a structured ctx field exists.
  Tool calls mutate the shared `ctx` dict; SKILL code may read these fields after the corresponding tool succeeds.
- `distance_computation` overwrites `ctx["distance"]` on each successful call:
  - object-object: `{mode: "object_to_object", meters, a, b, a_id, b_id, a_frames, b_frames, checker}`.
  - object-camera: `{mode: "object_to_camera", meters, a, b: "camera@<frame>", obj_id, a_frames, checker}`.
  - If looping over candidates, call `ctx.pop("distance", None)` before each distance call so a failed/no-distance case cannot reuse the previous candidate's value.
- `direction_computation` overwrites `ctx["direction"]` on each successful call:
  - `{direction, lr_label, fb_label, lr_angle_deg, fb_angle_deg, angle_from_forward_deg, distance_ref_to_target, viewpoint, reference, target, facing, checker}`.
  - Use `direction` for option text such as `front-left`; use angles only when the task asks for thresholds or tie-breaking.
- `object_size_computation` writes `ctx["object_size"]`:
  - `{object, instance_id, dimension, dimension_label, value, unit, size_xyz_m, bbox_volume_m3, frames_used, checker}`.
  - `size_xyz_m` is `[width_x, height_y, depth_z]` in meters; use `value/unit` for the requested dimension.
- `scene_size_computation` writes `ctx["scene_size"]`:
  - `{extent_xyz, bbox_min, bbox_max, floor_area, height, n_points_raw, n_points_clean}`.
  - Use `floor_area` for room-area answers; `extent_xyz` gives width/height/depth-like extents.
- Checker-backed distance/direction/object-size tools may write `ctx["final_localizations"]`:
  - keys are normalized object names; values include `object`, `instance_id`, `instance_label`, `keep_frames`, `dropped_frames`, and `reason`.
  - Use this only when a derived task needs to know which checked instance/frames were used.
- When a SKILL computes a derived task result, store it in a clear ctx key such as `ctx["closest_candidate_result"]` or `ctx["route_planning_result"]`, and append a concise human-readable line to `ctx.setdefault("skill_summaries", []).append(evidence_text)` so Reflector/Finalizer can see it.
"""


def distill_success(skill: Skill, skill_lib: SkillLib, llm: LLMClient,
                    window: int = DISTILL_WINDOW, verbose: bool = False) -> bool:
    """After N successes, rewrite the SKILL's `When to Use` to reflect the
    accumulated evidence of when it works. Returns True if a rewrite occurred."""
    trajectories = _read_recent_trajectories(skill.dir, window)
    if len(trajectories) < 3:
        return False
    prompt = f"""You are refining a SKILL's documentation based on recent invocations.

SKILL name: {skill.name}
Current `When to Use`:
---
{skill.sections.get("When to Use", "").strip()}
---

Recent invocation summaries (JSON, oldest first):
---
{json.dumps(trajectories, indent=2, ensure_ascii=False)}
---

Rewrite the `When to Use` section to be more precise: emphasize the question patterns / inputs where this SKILL succeeded; note any question patterns where it fails; keep it 2-5 sentences. Do NOT include a heading or markdown fence. Output JSON only.

Output JSON:
{{
    "when_to_use": "<the new plain-text body of the When to Use section>",
    "changed": true | false,   // false if the current text already captures the pattern
    "rationale": "1 sentence: what changed and why"
}}
"""
    resp = llm.chat_json([{"role": "system", "content": "You update SKILL documentation."},
                          {"role": "user", "content": prompt}])
    if not resp.get("changed"):
        return False
    new_text = resp.get("when_to_use", "").strip()
    if not new_text:
        return False
    skill_lib.rewrite_section(skill.name, "When to Use", new_text)
    skill_lib.bump_version(skill.name)
    if verbose:
        print(f"[EVOLVE] distill_success: rewrote {skill.name}/When to Use")
    return True


def distill_new_skill(sample, trajectory: dict, llm: LLMClient,
                      skill_lib: SkillLib, verbose: bool = False) -> str | None:
    """Ask the LLM to distill a trajectory into a new SKILL. Returns the
    pending SKILL name if saved, else None."""
    tool_seq = [tc["tool_name"] for tc in trajectory.get("tool_calls", []) if tc.get("success")]
    if len(tool_seq) < 2:
        return None


    # Show all existing skills (main + pending) for this category so LLM can decide
    # whether to update an existing skill or create a genuinely new one.
    all_existing = skill_lib.retrieve(sample.task_category, include_pending=True, top_k=20)
    existing_brief = "\n".join(
        f"- [{('PENDING' if 'pending' in s.dir.parts else 'MAIN')}] {s.name}: "
        f"{s.sections.get('When to Use', '').strip()[:250]}"
        for s in all_existing
    ) if all_existing else "(none)"

    prompt = f"""You are reviewing a successful trajectory to decide whether to update an existing SKILL or create a new one.

This Path-B trajectory is a successful raw-tool trajectory with no SKILL selected. Use it to decide whether the strategy should become a new SKILL or update an existing one.

Sample question type: {sample.task_type}
Task category: {sample.task_category}
Question: {sample.question[:400]}
Ground truth: {sample.gt_answer}

Successful tool sequence: {tool_seq}
Trajectory details:
{json.dumps(trajectory, indent=2, ensure_ascii=False)[:3000]}

All existing SKILLs for this category (MAIN = production-ready, PENDING = auto-distilled):
{existing_brief}

Decision rules:
1. If an existing SKILL already covers this strategy well → output {{"action": "skip"}}
2. If an existing SKILL is highly similar but missing some nuance this trajectory reveals →
   output {{"action": "update", "name": "<existing_skill_name>", "append_when_to_use": "1-2 sentences to append to that skill's When to Use section."}}
3. If this trajectory represents a genuinely different strategy not covered by any existing SKILL →
   output {{"action": "create", ...}} with a full new SKILL definition. The name should reflect what makes this strategy distinct (e.g. "evolved_distance_two_objects_closest_point" not just "evolved_distance").

For action="create", output JSON:
{{
    "action": "create",
    "name": "snake_case_name_describing_the_distinct_strategy",
    "task_categories": ["{sample.task_category}"],
    "when_to_use": "2-4 sentences on what class of questions this SKILL handles",
    "parameters": [
        {{"name": "...", "type": "str|int|list[str]", "description": "..."}}
    ],
    "tool_sequence": ["tool1", "tool2", ...],
    "execute_py_body": "<see contract below>"
}}

{_structured_ctx_reference()}

Contract for `execute_py_body` — READ CAREFULLY:

1. Output **only the function body**, NOT the `def execute(sample, tools, ctx, params=None):` signature line. The signature is added by the caller.

2. The tool API is: `r = tools.execute_tool(name, params_dict, frame_paths)` returns a SINGLE DICT
   with keys `{{ "tool_name": str, "params": dict, "success": bool, "result_summary": str, "error": str|None }}`.
   It does NOT return a tuple. Access success via `r["success"]`.

3. Get frames from context: `frame_paths = ctx["frame_paths"]`.

4. Read caller params from `params` (already normalized to a dict at entry): `params.get("target", "default")`.

5. Return a dict: `{{"success": bool, "tool_calls": list_of_r_dicts, "summary": str}}`.

Reference example (measure_distance-style):
```
    params = params or {{}}
    targets = params.get("targets", [])
    frame_paths = ctx["frame_paths"]
    calls = []
    r = tools.execute_tool("depth_estimation", {{}}, frame_paths); calls.append(r)
    if not r["success"]:
        return {{"success": False, "tool_calls": calls, "summary": f"depth failed: {{r['error']}}"}}
    for t in targets:
        r = tools.execute_tool("object_segmentation", {{"text_prompt": t, "object_category": t}}, frame_paths); calls.append(r)
        if not r["success"]:
            return {{"success": False, "tool_calls": calls, "summary": f"seg failed for {{t}}"}}
    r = tools.execute_tool("instance_3d_localization", {{}}, frame_paths); calls.append(r)
    return {{"success": r["success"], "tool_calls": calls, "summary": r.get("result_summary", "")}}
```

Constraints:
- Only use tools from: depth_estimation, bev_generation, novel_view_synthesis, object_segmentation, annotation_localization, instance_3d_localization, instance_counting, distance_computation, direction_computation, object_size_computation, scene_size_computation.
- Keep execute_py_body concise and defensive: check `r["success"]` after every tool call and return early on failure.
- For distance_computation, direction_computation, and object_size_computation, prefer object-name parameters from `params` (e.g. `obj_a`, `obj_b`, `object_name`, `viewpoint_target`, `reference_target`, `target_target`, `facing_target`). The tool layer will run Checker and resolve the final instance.
- Never emit placeholder IDs such as `<id_of_chair>` or `chair_instance_id`. Use integer IDs only when the execute body obtained a concrete ID from prior tool context.
- Object-size evidence should be treated as bbox_size_xyz plus bbox volume; derive a requested width/height/depth/longest/shortest value only when the final task needs it.
"""
    resp = llm.chat_json([{"role": "system", "content": "You design reusable SKILLs."},
                          {"role": "user", "content": prompt}],
                         max_tokens=8192)

    action = resp.get("action", "create")

    if action == "skip":
        if verbose:
            print(f"[EVOLVE] distill_new_skill: LLM decided existing skill covers this — skipping")
        return None

    if action == "update":
        target_name = resp.get("name", "").strip()
        append_text = resp.get("append_when_to_use", "").strip()
        if target_name and append_text:
            skill = skill_lib.get(target_name)
            if skill is not None:
                existing_wtu = skill.sections.get("When to Use", "").strip()
                skill_lib.rewrite_section(target_name, "When to Use", existing_wtu + "\n" + append_text)
                skill_lib.bump_version(target_name)
                if verbose:
                    print(f"[EVOLVE] distill_new_skill: appended to '{target_name}' When to Use")
                return target_name
        return None

    # action == "create"
    name = resp.get("name", "").strip()
    if not name:
        return None
    # Avoid name clashes
    if (skill_lib.root / name).exists() or (skill_lib.pending_dir / name).exists():
        name = f"{name}_v{int(np.random.randint(1000, 9999))}"

    md = _render_new_skill_md(
        name=name,
        task_categories=resp.get("task_categories", [sample.task_category]),
        when_to_use=resp.get("when_to_use", "").strip(),
        parameters=resp.get("parameters", []),
        tool_sequence=resp.get("tool_sequence", tool_seq),
    )
    py = _render_new_execute_py(name=name, body=resp.get("execute_py_body", "").strip())

    # Reject pending SKILL if the generated execute.py fails to parse as Python
    # or doesn't define execute(). We do not save broken drafts.
    try:
        tree = ast.parse(py)
    except SyntaxError as e:
        if verbose:
            print(f"[EVOLVE] distill_new_skill: rejected '{name}' — SyntaxError: {e}")
        return None
    defines_execute = any(
        isinstance(n, ast.FunctionDef) and n.name == "execute"
        for n in ast.walk(tree)
    )
    if not defines_execute:
        if verbose:
            print(f"[EVOLVE] distill_new_skill: rejected '{name}' — no execute() defined")
        return None

    skill_lib.save_pending(name, md, py)

    # Runtime import validation: syntactic ok isn't enough — the module might
    # reference undefined names or fail to import. Load it once and reject if
    # the import raises.
    try:
        pending_skill = skill_lib.get(name)
        if pending_skill is None:
            raise RuntimeError("pending skill not loadable via SkillLib.get")
        skill_lib._import_execute(pending_skill.dir)  # would raise on runtime error
    except Exception as e:
        if verbose:
            print(f"[EVOLVE] distill_new_skill: rejected '{name}' — import error: {e}")
        # Remove the broken pending directory
        import shutil
        broken = skill_lib.pending_dir / name
        if broken.exists():
            shutil.rmtree(broken, ignore_errors=True)
        return None

    if verbose:
        print(f"[EVOLVE] distill_new_skill: saved skills/pending/{name}")
    return name


def distill_failure_lessons(sample, fail_result: AgentResult, corrected_result: AgentResult,
                            llm: LLMClient, skill: Skill | None = None,
                            verbose: bool = False) -> dict[str, str]:
    """Distill one corrected failure into optional Pitfall and Example entries.

    This is a single LLM request. Pitfall is category/tool-strategy oriented;
    Example is sample-question oriented and should only be emitted when the
    question/sample itself teaches a useful task-specific lesson.
    """
    skill_name = skill.name if skill is not None else str(fail_result.chosen_skill or "")
    existing_pitfalls = (skill.sections.get("Known Pitfalls", "").strip()
                         if skill is not None else "")
    existing_examples = (skill.sections.get("Examples", "").strip()
                         if skill is not None else "")
    existing_checker = (skill.sections.get("Checker", "").strip()
                        if skill is not None else "")
    prompt = f"""You are updating one SpatialMem SKILL after a failed run was corrected and verified.

SKILL being updated: {skill_name or "(unknown)"}

Existing `Known Pitfalls`:
---
{existing_pitfalls or "None yet."}
---

Existing `Examples`:
---
{existing_examples or "None yet."}
---

Existing `Checker`:
---
{existing_checker or "None yet."}
---

Produce zero or more updates:

1. Known Pitfall:
   A compact general warning about what can go wrong for this SKILL. This should
   focus on planning/tool-use traps that may recur across similar samples.

2. Example:
   A compact sample-specific insight for the SKILL's `Examples` section. Add this
   ONLY if this particular question/sample is educational. The insight can be
   about question interpretation, category boundary decisions (e.g. whether a
   loveseat counts as a chair), viewpoint/reference/facing semantics, exact
   option matching, special definitions/thresholds, scene-specific ambiguity, or
   any other detail tightly tied to THIS question text. Do not force it into a
   fixed parameter-extraction template, and do not write generic tool advice.

3. Checker:
   A compact Checker-specific lesson. Add this ONLY if this sample used Checker
   and the failure/correction was mainly caused by Checker behavior, such as
   choosing the wrong target track, keeping/dropping the wrong frames, accepting
   a spurious track, missing a real target in count calibration, or direct-answer
   fallback from raw frames. If the final context contains no `[checker]` entry,
   set `checker_is_new=false`.

You may output any subset of pitfall/example/checker, or none.
Avoid low-information repetition:
- If the proposed pitfall is already covered by Existing `Known Pitfalls`, set `pitfall_is_new=false`.
- If the proposed example insight is already covered by Existing `Examples`, set `example_is_useful=false`.
- If the proposed checker lesson is already covered by Existing `Checker`, set `checker_is_new=false`.
- A new example should add a distinct sample-specific nuance, not just restate the same lesson on another question.

Question type: {fail_result.task_category}
Sample id: {sample.id}
Original question:
{sample.question}
Ground truth answer: {sample.gt_answer}

Failed trajectory (wrong answer '{fail_result.predicted_answer}', GT '{fail_result.gt_answer}'):
  tool calls: {[tc["tool_name"] for tc in fail_result.tool_calls]}
  plan: {json.dumps(fail_result.plan, ensure_ascii=False)[:600]}
  final evidence:
{fail_result.final_context[:1200]}

Corrected re-run (answer '{corrected_result.predicted_answer}' matched GT):
  plan: {json.dumps(corrected_result.plan, ensure_ascii=False)[:600]}
  final evidence:
{corrected_result.final_context[:1200]}

Output JSON:
{{
    "pitfall_is_new": true | false,
    "pitfall": "<one-line general lesson for Known Pitfalls, or empty string>",
    "example_is_useful": true | false,
    "example": "<1-3 concise sentences capturing the sample-specific insight, or empty string>",
    "checker_is_new": true | false,
    "checker": "<one-line Checker-specific lesson, or empty string>",
    "rationale": "1 sentence"
}}
"""
    resp = llm.chat_json([{"role": "system", "content": "You distill SKILL updates."},
                          {"role": "user", "content": prompt}])
    out = {"pitfall": "", "example": "", "checker": ""}

    if resp.get("pitfall_is_new"):
        out["pitfall"] = str(resp.get("pitfall", "")).strip()
        if verbose and out["pitfall"]:
            print(f"[EVOLVE] distill_failure_lessons pitfall: {out['pitfall']}")

    if resp.get("example_is_useful"):
        insight = str(resp.get("example", "")).strip()
        if insight:
            question = str(sample.question).strip().replace("\n", " ")
            entry = (
                f"- **Sample {sample.id}**\n"
                f"  - Question: {question}\n"
                f"  - Insight: {insight}"
            )
            out["example"] = entry
            if verbose:
                print(f"[EVOLVE] distill_failure_lessons example: sample {sample.id}")

    if resp.get("checker_is_new"):
        checker_note = str(resp.get("checker", "")).strip()
        if checker_note:
            out["checker"] = checker_note
            if verbose:
                print(f"[EVOLVE] distill_failure_lessons checker: {checker_note}")

    return out


# ─── Proactive category-level bootstrap ───────────────────────────────

def _log_uncovered_category(sample, result: AgentResult, log_dir: Path) -> None:
    """Persist a compact trajectory to memory/category_trajectories/{cat}.jsonl
    whenever Planner didn't pick a SKILL. Feeds `_maybe_bootstrap_category`.
    Includes both correct and wrong outcomes so the bootstrap LLM can compare.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "sample_id": result.sample_id,
        "task_type": result.task_type,
        "question": sample.question[:400],
        "gt": result.gt_answer,
        "predicted": result.predicted_answer,
        "success": bool(result.success),
        "tool_calls": [
            {"tool_name": tc["tool_name"], "success": tc["success"],
             "params": tc.get("params", {})}
            for tc in result.tool_calls
        ],
    }
    path = log_dir / f"{result.task_category}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Guards bootstrap decisions from race conditions when multiple threads
# concurrently push a category over the trigger threshold. Without this,
# two workers could both pass the marker check, both invoke the LLM to
# design a SKILL, and both save (with a version suffix). This lock
# serializes the check-and-fire block per process.
_BOOTSTRAP_LOCK = threading.Lock()


def _maybe_bootstrap_category(
    category: str, skill_lib: SkillLib, memory: Memory,
    llm: LLMClient, threshold: int = BOOTSTRAP_TRIGGER_N,
    verbose: bool = False,
) -> str | None:
    """Trigger `bootstrap_skill_for_category` at most once per category, when:
      - no SKILL (main or pending) covers this category, AND
      - >= threshold uncovered trajectories have accumulated, AND
      - we haven't already tried bootstrapping this category (marker file).

    Thread-safe: acquires _BOOTSTRAP_LOCK before the marker check so two
    concurrent threads can't both fire bootstrap for the same category.
    """
    log_dir = memory.root / "category_trajectories"
    log_path = log_dir / f"{category}.jsonl"
    marker = log_dir / f"{category}.bootstrapped"

    with _BOOTSTRAP_LOCK:
        if marker.exists():
            return None
        if not log_path.exists():
            return None

        # Skip if a SKILL for this category already exists (main or pending).
        # Path B/D may have distilled a pending SKILL between when trajectories
        # started accumulating and now.
        if skill_lib.retrieve(category, top_k=1, include_pending=True):
            return None

        with log_path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if len(lines) < threshold:
            return None

        # Mark BEFORE the (slow) LLM call so any concurrent thread that
        # acquires the lock next sees the marker and bails. If the LLM call
        # later fails, we leave the marker in place — one attempt per
        # category, on purpose (retrying tends to reproduce the same LLM
        # failure mode).
        marker.touch()

    # LLM call happens OUTSIDE the lock so other categories can still fire
    # bootstrap in parallel. Only same-category re-entries are blocked.
    trajs = [json.loads(ln) for ln in lines]
    name = bootstrap_skill_for_category(
        category, trajs, llm, skill_lib, verbose=verbose,
    )
    return name


def bootstrap_skill_for_category(
    category: str, trajectories: list[dict], llm: LLMClient,
    skill_lib: SkillLib, verbose: bool = False,
) -> str | None:
    """Synthesize a SKILL from a batch of category-level trajectories.

    Unlike `distill_new_skill` (which learns from ONE successful trajectory),
    this reads MANY trajectories — including failed ones — and asks the LLM
    to identify a working pattern. This is what turns "the framework has been
    reactively failing at this category" into "the framework decides to make
    a SKILL for this category".
    """
    correct = [t for t in trajectories if t.get("success")]
    wrong = [t for t in trajectories if not t.get("success")]

    def _compact(t: dict) -> dict:
        return {
            "question": t.get("question", "")[:250],
            "gt": t.get("gt"),
            "predicted": t.get("predicted"),
            "success": t.get("success"),
            "tool_calls": [
                {"tool_name": tc.get("tool_name"), "success": tc.get("success"),
                 "params": tc.get("params", {})}
                for tc in t.get("tool_calls", [])
            ],
        }

    correct_view = [_compact(t) for t in correct[-5:]]
    wrong_view = [_compact(t) for t in wrong[-5:]]

    prompt = f"""The SpatialMem framework has been repeatedly encountering task_category='{category}' samples without any matching SKILL to invoke. It has fallen back to raw-tool reasoning {len(trajectories)} times ({len(correct)} correct, {len(wrong)} wrong). Your job: design a SKILL that would have covered these samples.

Correct trajectories (learn the working pattern):
{json.dumps(correct_view, indent=2, ensure_ascii=False)[:3500]}

Wrong trajectories (avoid these anti-patterns; note where they diverged):
{json.dumps(wrong_view, indent=2, ensure_ascii=False)[:3500]}

Output JSON:
{{
    "name": "snake_case_skill_name",
    "task_categories": ["{category}"],
    "when_to_use": "3-5 sentences: the class of questions, expected inputs, and what evidence pattern indicates this SKILL applies",
    "parameters": [ {{"name": "...", "type": "str|int|list[str]", "description": "..."}} ],
    "tool_sequence": ["tool1", "tool2", ...],
    "execute_py_body": "<see contract>"
}}

{_structured_ctx_reference()}

Contract for `execute_py_body`:
1. Output ONLY the function body — no `def execute(...):` line.
2. Tool API returns a DICT with keys `success/tool_name/params/result_summary/error`. Access via `r["success"]`. It is NOT a tuple.
3. Read frame paths from `frame_paths = ctx["frame_paths"]`.
4. Read caller params via `params.get("target", "default")`.
5. Return `{{"success": bool, "tool_calls": list, "summary": str}}`.
6. Only use tools from: depth_estimation, bev_generation, novel_view_synthesis, object_segmentation, annotation_localization, instance_3d_localization, instance_counting, distance_computation, direction_computation, object_size_computation, scene_size_computation.
7. Be defensive: check `r["success"]` after every tool call.
8. Prefer object-name parameters for distance/direction/object-size tools and let the tool layer resolve final instances with Checker. Never generate placeholder IDs like `<id_of_chair>`.
9. For object-size tasks, preserve bbox_size_xyz/volume as the general evidence; derive a specific dimension only if the question requires it.

If you cannot design a coherent SKILL for this category (e.g., tools do not support the required semantics), output `{{"name": "", "unsupported_reason": "..."}}` and we will skip bootstrap.
"""
    resp = llm.chat_json(
        [{"role": "system", "content": "You bootstrap new SKILLs from category-level trajectories."},
         {"role": "user", "content": prompt}],
        max_tokens=12288,
    )

    name = (resp.get("name") or "").strip()
    if not name:
        if verbose:
            reason = resp.get("unsupported_reason", "no name returned")
            print(f"[EVOLVE] bootstrap({category}): skipped — {reason}")
        return None

    # Namespace clashes
    if (skill_lib.root / name).exists() or (skill_lib.pending_dir / name).exists():
        name = f"{name}_v{int(np.random.randint(1000, 9999))}"

    md = _render_new_skill_md(
        name=name,
        task_categories=resp.get("task_categories", [category]),
        when_to_use=resp.get("when_to_use", "").strip(),
        parameters=resp.get("parameters", []),
        tool_sequence=resp.get("tool_sequence", []),
    )
    py = _render_new_execute_py(name=name, body=resp.get("execute_py_body", "").strip())

    # Same AST + runtime validation as distill_new_skill
    try:
        tree = ast.parse(py)
    except SyntaxError as e:
        if verbose:
            print(f"[EVOLVE] bootstrap({category}): rejected '{name}' — SyntaxError: {e}")
        return None
    if not any(isinstance(n, ast.FunctionDef) and n.name == "execute" for n in ast.walk(tree)):
        if verbose:
            print(f"[EVOLVE] bootstrap({category}): rejected '{name}' — no execute() defined")
        return None

    skill_lib.save_pending(name, md, py)
    try:
        pending_skill = skill_lib.get(name)
        if pending_skill is None:
            raise RuntimeError("pending skill not loadable")
        skill_lib._import_execute(pending_skill.dir)
    except Exception as e:
        import shutil
        broken = skill_lib.pending_dir / name
        if broken.exists():
            shutil.rmtree(broken, ignore_errors=True)
        if verbose:
            print(f"[EVOLVE] bootstrap({category}): rejected '{name}' — import error: {e}")
        return None

    if verbose:
        print(f"[EVOLVE] bootstrap({category}): saved skills/pending/{name} "
              f"(from {len(trajectories)} trajectories)")
    return name


# ─── Bookkeeping helpers ──────────────────────────────────────────────

def _summarize_trajectory(result: AgentResult) -> dict:
    # Best-effort read of skill_params from the plan.
    skill_params = None
    if isinstance(result.plan, dict):
        skill_params = result.plan.get("skill_params")
    return {
        "sample_id": result.sample_id,
        "task_type": result.task_type,
        "task_category": result.task_category,
        "chosen_skill": result.chosen_skill,
        "skill_params": skill_params,
        "trajectory_source": "raw_tool" if not result.skills_used else "skill_only",
        "success": bool(result.success),
        "predicted": result.predicted_answer,
        "gt": result.gt_answer,
        "num_rounds": result.num_rounds,
        "confidence": result.confidence,
        "tool_calls": [
            {"tool_name": tc["tool_name"], "success": tc["success"],
             "params": tc.get("params", {})}
            for tc in result.tool_calls
        ],
    }


def _read_recent_trajectories(skill_dir: Path, n: int) -> list[dict]:
    p = skill_dir / "trajectories.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    out: list[dict] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _update_memory_from_tools(ctx: dict, memory: Memory, sample, updates: dict) -> None:
    """Feed 3D localization and scene-size results into Memory priors."""
    r3d = ctx.get("results_3d")
    if r3d is not None:
        seg_categories = ctx.get("seg_categories", {})
        labels = r3d.get("merged_labels", [])
        for i in r3d.get("obj_id_list", []):
            label = labels[i] if i < len(labels) else ""
            if not label or label in ("__annotations__",) or label.endswith("_point") \
               or label.endswith("_bbox"):
                continue
            size = r3d["merged_bbox_size"][i]
            if not (np.all(np.isfinite(size)) and np.all(size > 0.05)):
                continue
            obj_name = seg_categories.get(label, label).strip().lower()
            # Filter out labels that don't look like canonical object names:
            #   - too long (probably a descriptive viewpoint phrase like
            #     "standing by the chair and facing the microwave")
            #   - too many words (canonical object categories are 1-3 words)
            #   - empty
            if not obj_name or len(obj_name) > 40 or len(obj_name.split()) > 3:
                continue
            memory.update_object_size_prior(
                obj_name,
                width=float(size[0]),
                height=float(size[1]),
                depth=float(size[2]),
            )
            updates["memory_updates"].append(f"size_prior:{obj_name}")

    ss = ctx.get("scene_size")
    if ss is not None:
        ext = ss["extent_xyz"]
        if np.all(np.isfinite(ext)) and np.all(ext > 0.1):
            scene_type = ctx.get("scene_type") or "indoor_room"
            memory.update_scene_scale_prior(
                scene_type,
                width=float(ext[0]),
                height=float(ext[1]),
                depth=float(ext[2]),
                floor_area=float(ss["floor_area"]),
            )
            updates["memory_updates"].append(f"scale_prior:{scene_type}")


# ─── SKILL file rendering (for pending drafts) ────────────────────────

def _render_new_skill_md(name: str, task_categories: list[str],
                         when_to_use: str, parameters: list[dict],
                         tool_sequence: list[str]) -> str:
    param_lines = []
    for p in parameters:
        pname = p.get("name", "?")
        ptype = p.get("type", "?")
        pdesc = p.get("description", "")
        param_lines.append(f"- `{pname}` ({ptype}): {pdesc}")
    tool_lines = [f"{i+1}. `{t}`" for i, t in enumerate(tool_sequence)]

    return f"""---
name: {name}
task_categories: {list(task_categories)}
total_calls: 0
success_count: 0
failure_count: 0
success_rate: 0.0
version: 1
seeded: false
---

# When to Use

{when_to_use or "Draft — evolved from a single trajectory."}

# Parameters

{chr(10).join(param_lines) if param_lines else "None."}

# Tool Sequence

{chr(10).join(tool_lines) if tool_lines else "None."}

# Known Pitfalls

None yet.

# Examples

None yet.

# Checker

None yet.
"""


def _render_new_execute_py(name: str, body: str) -> str:
    if not body:
        body = ('return {"success": False, "tool_calls": [], '
                '"summary": "empty body — evolve.py fallback"}')

    # Defensive: if the LLM included the function signature or a code fence,
    # strip them. Also strip any wrapping `def execute(...)` block if present.
    body = body.strip()
    if body.startswith("```"):
        # remove opening fence line
        body = "\n".join(body.split("\n")[1:])
        if body.endswith("```"):
            body = body[: body.rfind("```")].rstrip()
    # If body starts with `def execute(...)`, unwrap to its own body
    stripped_lines = body.splitlines()
    if stripped_lines and stripped_lines[0].lstrip().startswith("def execute"):
        # find the indentation of the inner block (first non-empty line after def)
        inner = []
        for ln in stripped_lines[1:]:
            if not ln.strip():
                inner.append(ln)
                continue
            # dedent by one level (assume 4 spaces)
            if ln.startswith("    "):
                inner.append(ln[4:])
            else:
                inner.append(ln)
        body = "\n".join(inner).rstrip()

    # Indent every non-empty line by 4 spaces (function body indentation).
    lines = body.splitlines()
    indented = []
    for ln in lines:
        if not ln.strip():
            indented.append("")
        elif ln.startswith("    "):
            indented.append(ln)
        else:
            indented.append("    " + ln)
    indented_body = "\n".join(indented)

    return f'''"""SKILL: {name} (evolved).

Generated by agent/evolve.py.distill_new_skill from a successful trajectory.
"""

from __future__ import annotations


def execute(sample, tools, ctx, params=None):
    params = params or {{}}
{indented_body}
'''
