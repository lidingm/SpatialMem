# SpatialMem Design

## Motivation

Spatial reasoning is a hard modality for VLMs. Two agent-based approaches have appeared recently:

- **S-Agent** (arxiv 2606.20515) treats spatial reasoning as spatio-temporal evidence accumulation over multi-view / video inputs. It has hierarchical tools (2D grounding → 3D lifting → spatial experts) and two per-episode memories: Scene Memory (evolving scene state) and Agent Memory (reasoning history). Nothing transfers between samples.
- **SpatialClaw** (arxiv 2606.13673) uses code-as-action: a persistent Jupyter kernel per sample, with SAM3 / Depth-Anything-3 primitives. A planner outlines a strategy without seeing images; a coder writes cells iteratively. No memory.

SpatialMem's contribution is orthogonal: **cross-sample, training-time SKILL + Memory self-evolution** for spatial reasoning. The agent runs on a training set with ground truth, and every sample either strengthens or extends the SKILL library and Memory priors. At test time the evolved artifacts are used.

## Two persistent artifacts

The design keeps these two ideas strictly separate — different data structure, different update semantics, different code modules:

### Memory (declarative)

Facts about the physical world that calibrate downstream tools. Currently:

- `object_size_priors.json` — Welford online mean/std of width/height/depth per object category. Kept in `candidates` until `count >= WARMUP_N`, then promoted to main pool.
- `scene_scale_priors.json` — same for room extents and floor area per scene type.
- `unsolved_cases.jsonl` — append-only log of samples the agent got wrong AND reflection failed to fix.

Operations on Memory are called **updates** in the code (`update_object_size_prior`, `update_scene_scale_prior`, `record_unsolved`). They are monotonic quantity changes.

Layer3-style failure lessons from the old memory bank are NOT stored in Memory — they belong to a specific procedure, and live inside the corresponding SKILL's `Known Pitfalls`.

### SKILL (procedural)

Reusable, parameterized tool sequences. Each SKILL is a folder:

```
skills/measure_distance/
├── SKILL.md              # YAML frontmatter + Markdown sections
├── execute.py            # def execute(sample, tools, ctx, params=None) -> dict
└── trajectories.jsonl    # compact per-invocation summaries
```

Frontmatter fields: `name`, `task_categories`, `total_calls`, `success_count`, `failure_count`, `success_rate`, `version`, `seeded`.

Markdown sections: `When to Use`, `Parameters`, `Tool Sequence`, `Known Pitfalls`, `Examples`.

Operations on SKILLs are called **evolutions** in the code (`evolve_after_sample`, `distill_success`, `distill_new_skill`, `distill_pitfall`, `promote_from_pending`). They involve structural change: new drafts, promotion, versioning, appending or rewriting sections.

## Four seed SKILLs

Rather than cold-start, four seeds cover the highest-frequency VSIBench question types:

| SKILL | Covers VSI type(s) | Tool sequence |
| --- | --- | --- |
| `measure_distance` | `object_abs_distance`, `object_rel_distance` | depth → seg × N → 3d_loc → dist |
| `count_objects` | `object_counting` | depth → seg → 3d_loc → count |
| `judge_direction` | `object_rel_direction_easy/medium/hard` | depth → seg × N → 3d_loc → dir |
| `measure_scene_size` | `room_size_estimation` | depth → scene_size |

Question types NOT seeded in VSI-Train-10k (`object_size`, `appearance_order`) rely on Path B/D evolution to discover their own SKILLs from successful trajectories. (VSIBench additionally has `route_planning` and `object_size_estimation` which are also unseeded.)

## Per-sample flow

1. Orchestrator resets the tool context and uses all `MAX_FRAMES_PER_SAMPLE = 32` available frames per scene.
2. Memory context (relevant object/scene priors) is formatted.
3. SkillLib retrieves the top-k SKILLs whose `task_categories` matches the sample. Both main-library and pending SKILLs are retrieved (`include_pending=True`); pending ones are labelled `[AUTO-DISTILLED — self-summarized by the system; feel free to use if it fits]`.
4. Planner sees images + question + memory context + SKILL briefs + tool descriptions, and outputs a JSON plan with `chosen_skill` (or `null`).
5. If a SKILL was chosen: `skill_lib.invoke(skill, sample, tools, ctx, skill_params)` runs its `execute()`.
6. Otherwise: raw-tool loop of Reasoner (refines each step's params against accumulated evidence) + Reflector.reflect (continue vs terminate).
7. Reflector.finalize extracts the answer in the required format.

## Four evolution paths

After each training sample, `evolve_after_sample` dispatches by (correct?, SKILL used?):

**Path A — correct, SKILL used.** Reinforce (bump `total_calls` and `success_count`). Every `DISTILL_TRIGGER_N` successes, call `distill_success`: LLM reads the most recent `DISTILL_WINDOW` trajectory summaries and rewrites the SKILL's `When to Use` if a pattern justifies it. Also feed 3D localization outputs into Memory priors.

**Path B — correct, no SKILL.** `distill_new_skill` first shows the LLM **all existing SKILLs (main + pending) for this category** and asks it to decide: (a) `"action": "skip"` if an existing SKILL already covers this strategy; (b) `"action": "update"` + `append_when_to_use` text if an existing SKILL is close but missing nuance; (c) `"action": "create"` with a full new SKILL definition (name should reflect the distinct strategy). New SKILLs are saved to `skills/pending/` after AST + runtime validation. Also update Memory.

**Path C — wrong, SKILL used.** Log the failure on the SKILL. Reflector.reconstruct proposes a corrected plan; orchestrator re-runs the sample under that plan; if the re-run matches GT, LLM distills a `Known Pitfalls` bullet and appends it. If verification fails, record to `unsolved_cases.jsonl`.

**Path D — wrong, no SKILL.** Same verify loop as C. If the corrected plan verifies, distill it as a new pending SKILL. Otherwise unsolved.

Key property: **updates that involve LLM interpretation always go through verification.** We never persist an LLM's diagnosis without first checking that it actually produces the correct answer on the failing sample.

## Distillation semantics

- **When to Use** is rewritten (overwrite the whole section) because it's a running summary.
- **Known Pitfalls** is appended (new bullets) OR rewritten (when the new pitfall is a refinement of an existing one). The LLM makes the append-vs-rewrite decision based on similarity to existing entries.
- **execute.py** is only modified for evolved (`pending`) SKILLs via the new-skill distillation pipeline. Seed SKILLs' `execute.py` is not automatically edited — changes are limited to the SKILL.md sections.
- **trajectories.jsonl** is always append-only; each entry is a compact summary (no full images or raw arrays).

## Promotion from pending

A pending SKILL is promoted once its `success_count >= PENDING_PROMOTE_K` (default 3). This is checked every `PROMOTE_EVERY = 50` samples inside `train_vsitrain10k.py`, and again at the end of each question-type loop.

## Why the naming diverges from SpatialClaw

SpatialClaw's node names are `plan_node.py`, `llm_step_node.py`, `execute_node.py`, `feedback_node.py`, `reflection_node.py`. SpatialMem uses `planner.py`, `reasoner.py`, `reflector.py`, `orchestrator.py`. Structurally similar (planner + step-writer + executor + reflector all appear), but the names are deliberately distinct to avoid the appearance of copying.

## What is NOT preserved from the old system

- `memory_bank/Layer2/strategy_memory.json` and `Layer3/strategy_failure_memory.json` are NOT migrated. They were polluted with 15-step degenerate tool sequences and generic failure texts like `"Predicted 1.9 but GT 1.3" → "Review evidence and try alternative tools"`. The new system starts fresh; seed SKILLs replace Layer2, and Path C distillation of `Known Pitfalls` replaces Layer3.
- The 4-role framework (Planner / Executor / Reflector / Summarizer) is collapsed into 3 (Planner / Reasoner / Reflector), with the Summarizer folded into `Reflector.finalize()`. This matches the level of granularity used in most recent agent frameworks.

## Tunable hyper-parameters

All live in `agent/config.py`:

- `MAX_FRAMES_PER_SAMPLE` (32) — all available VSI frames are used per scene.
- `MAX_ROUNDS` (3) — max reflection loops in the raw-tool fallback path.
- `CONFIDENCE_THRESHOLD` (0.8) — early-terminate threshold used by Reflector.reflect.
- `WARMUP_N` (5) — how many observations before a candidate prior graduates to the main pool.
- `DISTILL_TRIGGER_N` (5) — trigger `distill_success` every N successes.
- `DISTILL_WINDOW` (20) — how many recent trajectory summaries to feed the distillation LLM.
- `PENDING_PROMOTE_K` (3) — successful reuses required to promote a pending SKILL.
- `REFLECT_MAX_ATTEMPTS` (1) — attempts to reconstruct a corrected plan per wrong answer.

## Checkpoint system

Training writes to a **checkpoint** (`ckpts/<name>/skills/` + `ckpts/<name>/memory/`) rather than the project's base `skills/` and `memory/` directories. The base directories are the pristine seeded state (read-only during training). Use `agent/checkpoint.py::clone_from_base(name)` to create a new checkpoint, then pass `--ckpt <name>` to `train_vsitrain10k.py`. Multiple training runs can each have their own checkpoint; eval can load any specific checkpoint.

## Output isolation

Each process writes tool intermediates (DA3 depth maps, SAM3 masks) to `outputs/proc_{os.getpid()}/` so concurrent processes do not overwrite each other's files. This directory is ephemeral and can be deleted after training.
