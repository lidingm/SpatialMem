# Skills

Procedural knowledge — reusable, parameterized tool-call sequences that the agent can invoke. Each SKILL is a folder.

## Layout

```
skills/
├── _template/           # copy-paste base for evolution-generated SKILLs
│   ├── SKILL.md
│   └── execute.py
├── pending/             # freshly distilled SKILL drafts; promoted after PENDING_PROMOTE_K successful reuses
│   └── ...
├── measure_distance/    # seed skill (covers VSI object_abs_distance + object_rel_distance)
├── count_objects/       # seed skill (covers VSI object_counting)
├── judge_direction/     # seed skill (covers VSI object_rel_direction_easy/medium/hard)
└── measure_scene_size/  # seed skill (covers VSI room_size_estimation)
```

## SKILL folder structure

Every SKILL folder must contain:

- **`SKILL.md`** — declarative description. YAML frontmatter with statistics, then Markdown sections. See `_template/SKILL.md`.
- **`execute.py`** — a function `execute(sample, tools, ctx, params=None) -> dict` that runs the SKILL. See `_template/execute.py`.
- **`trajectories.jsonl`** — append-only log of recent invocations (one summary per line). Older entries can be pruned.

## Update semantics

SKILL is **evolved** (过程性变化，涉及新旧替换、生成、晋升). See `agent/skill_lib.py` and `agent/evolve.py`.

Four evolution paths after each training sample:

| Sample outcome | Skill used? | Action |
|---|---|---|
| ✅ correct | yes | reinforce; every N successes, LLM-distill `When to Use` and default params |
| ✅ correct | no | LLM distills a new SKILL draft into `pending/` |
| ❌ wrong | yes | reflect → reconstruct plan → **re-run to verify** → if success, LLM appends/rewrites `Known Pitfalls` on the used SKILL |
| ❌ wrong | no | reflect → reconstruct → verify → if success, LLM distills a new SKILL draft into `pending/` |

If verification also fails, the sample is recorded to `memory/unsolved_cases.jsonl` (no SKILL update).

## Retrieval

The Planner picks a SKILL by matching `task_categories` in the frontmatter to the current sample's category, then ranking by `success_rate`. Top-k is offered to the Planner LLM, which chooses one (or falls back to raw tools).
