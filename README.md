# SpatialMem

A **training-time self-evolving spatial reasoning agent** for VSIBench. Combines a VLM planner, a folder-based **SKILL library** (procedural, executable), and a **Memory bank** of physical-world priors (declarative). During training, four evolution paths distill lessons from every sample into the persistent library; at test time the evolved artifacts drive the reasoning.

**Positioning vs. related work**

| System | Cross-sample knowledge | Where it lives |
| --- | --- | --- |
| S-Agent (arxiv 2606.20515) | ✗ | Scene / Agent memory is per-episode |
| SpatialClaw (arxiv 2606.13673) | ✗ | Persistent Jupyter kernel per question, no memory |
| **SpatialMem** | ✅ | Folder-based SKILL library + declarative Memory, accumulated across the whole training set |

## Architecture

```
                                       ┌──────────────────────────┐
                                       │        Sample            │
                                       └────────────┬─────────────┘
                                                    ▼
                          ┌────────────────── Planner (VLM) ──────────────────┐
                          │                                                   │
                          │   Retrieve SKILL briefs (skill_lib)               │
                          │   Retrieve Memory context (declarative priors)    │
                          │   Output plan JSON with chosen_skill              │
                          └───┬─────────────────────────────────┬─────────────┘
                              │ chosen_skill                    │ chosen_skill: null
                              ▼                                 ▼
                   ┌────────────────────┐        ┌─────────────────────────────┐
                   │  SkillLib.invoke   │        │  Raw-tool ReAct loop:       │
                   │  execute.py runs   │        │    Reasoner (VLM) → tools   │
                   │  the tool sequence │        │    → Reflector.reflect      │
                   └──────────┬─────────┘        └──────────────┬──────────────┘
                              │                                 │
                              └─────────────┬───────────────────┘
                                            ▼
                                Reflector.finalize (VLM)
                                            │
                                            ▼
                                      final answer
                                            │
                       ┌─── (training only) ┴─────────────────────────────┐
                       ▼                                                  │
              compare with GT                                             │
                       │                                                  │
                       ▼                                                  │
              evolve.evolve_after_sample dispatches to 4 paths            │
              (see below), each writing back to SKILL / Memory ───────────┘
```

## Four evolution paths + proactive bootstrap

**Reactive (per-sample) evolution** — dispatched by `evolve.evolve_after_sample` after each training sample:

| Correct? | Used a SKILL? | Path | What it does |
| --- | --- | --- | --- |
| ✅ | yes | **A** | Reinforce (bump `success_count`); every N successes, LLM-distill `When to Use` (`distill_success`) |
| ✅ | no | **B** | LLM distills the trajectory into a new SKILL draft → `skills/pending/` (`distill_new_skill`) |
| ❌ | yes | **C** | Reflector reconstructs a corrected plan → **re-run to verify** → if OK, LLM appends a `Known Pitfalls` bullet (`distill_pitfall`) |
| ❌ | no | **D** | Reconstruct → verify → if OK, distill a new SKILL draft → `skills/pending/` |

**Proactive (per-category) bootstrap** — the framework does not have to wait for a lucky single-sample success:

- Every sample where Planner picked **no** SKILL logs its trajectory to `memory/category_trajectories/{cat}.jsonl` (both correct and wrong).
- When a category accumulates `BOOTSTRAP_TRIGGER_N` uncovered trajectories **and** no pending/main SKILL matches, `bootstrap_skill_for_category` triggers.
- The bootstrap LLM reads correct + wrong trajectories side by side and synthesizes a SKILL draft targeting that category's common working pattern.
- Each category is bootstrapped at most once (`.bootstrapped` marker file).
- Bootstrap yields to Path B: if a single-sample distillation already produced a pending SKILL for the category, the bootstrap step is skipped.

Unverifiable reconstructions land in `memory/unsolved_cases.jsonl`. SKILLs in `pending/` are promoted to the main library once they succeed `PENDING_PROMOTE_K` times (retrieved via `SkillLib.retrieve(..., include_pending=True)` — pending SKILLs ARE offered to the Planner so they can accrue the successes needed for promotion).

**Five evolution mechanisms combined** — reactive Paths A/B/C/D + proactive bootstrap — cover both easy task types (where a single success spawns a SKILL) and hard task types (where evolution has to see many failures before finding a pattern).

**Terminology (used consistently in code + docs)**

- Memory is **updated** — declarative, monotonic quantity change (Welford online mean/std for numeric priors).
- SKILL is **evolved** — structural change: new draft, versioned rewrite, promoted from pending.

## Repository layout

```
SpatialMem/
├── agent/
│   ├── config.py          # unified config (paths, LLM, evolution hyperparams)
│   ├── llm_client.py      # OpenAI-compatible HTTP client (retries, JSON mode)
│   ├── data_loader.py     # VSIBenchDataLoader + SpatialSample
│   ├── tools.py           # ToolRegistry — dispatch for the 10 spatial tools
│   ├── planner.py         # Planner role (picks SKILL or raw-tool plan)
│   ├── reasoner.py        # Reasoner role (per-step tool refinement)
│   ├── reflector.py       # reflect() + reconstruct() + finalize()
│   ├── orchestrator.py    # per-sample main loop (run + run_with_plan)
│   ├── memory.py          # declarative Memory (priors + unsolved cases)
│   ├── skill_lib.py       # SKILL library — retrieve / invoke / append / evolve
│   ├── evolve.py          # 4-path evolution engine + LLM distillation
│   └── train_loop.py      # multi-sample driver + evaluation + evolve trigger
│
├── skills/                # SKILL library (folder-based)
│   ├── _template/         # seed for evolution-generated SKILLs
│   ├── measure_distance/  # seed (distance)
│   ├── count_objects/     # seed (counting)
│   ├── judge_direction/   # seed (spatial_relation)
│   ├── measure_scene_size/# seed (room_size)
│   └── pending/           # evolved drafts awaiting promotion
│
├── memory/                # declarative Memory
│   ├── object_size_priors.json
│   ├── scene_scale_priors.json
│   ├── unsolved_cases.jsonl
│   └── category_trajectories/    # per-category logs of no-SKILL samples (feeds bootstrap)
│       ├── {task_category}.jsonl
│       └── {task_category}.bootstrapped   # marker (one-shot per category)
│
├── tools/                 # low-level implementations (unchanged)
│   ├── visual_generation/ # DA3, SAM3 wrappers
│   └── code_execution/    # 3D localization, distances, directions, counting, scene size
│
├── scripts/
│   └── debug_llm.py       # LLM-side debug harness (mocks DA3/SAM3)
│
├── docs/
│   ├── design.md          # full design rationale (English)
│   └── 项目介绍.md         # Chinese project overview
│
├── outputs/               # per-run tool outputs (depth maps, seg masks, ...)
├── results/               # per-run evaluation JSONL
├── SpatialMem.ipynb       # end-to-end demo notebook
├── test.ipynb             # low-level tool tests (unchanged)
└── README.md              # this file
```

## SKILL folder contract

Every SKILL folder must contain:

- `SKILL.md` — YAML frontmatter (`name`, `task_categories`, `total_calls`, `success_count`, `failure_count`, `success_rate`, `version`, `seeded`) followed by five Markdown sections: `When to Use`, `Parameters`, `Tool Sequence`, `Known Pitfalls`, `Examples`.
- `execute.py` — defines `execute(sample, tools, ctx, params=None) -> dict` with keys `{success, tool_calls, summary}`.
- `trajectories.jsonl` — append-only, one compact summary per invocation.

Distillation reads the top of `trajectories.jsonl` and rewrites (`When to Use`, defaults) or appends (`Known Pitfalls`).

## Getting started

```bash
# 1. LLM credentials (any OpenAI-compatible endpoint)
export LLM_BASE_URL="https://your-llm-endpoint/v1"
export LLM_API_KEY="..."
export LLM_MODEL="gpt-4o-mini"     # or qwen3-vl-30b-a3b-instruct, etc.

# 2. Optional: override VSIBench location
export VSI_PARQUET=/mnt/stepeval/datasets/VL_datasets/VSI-Bench/test-00000-of-00001.parquet
export VSI_IMAGES_ROOT=/mnt/stepeval/VSI-Bench/images

# 3. Wire in your DA3 + SAM3 wrappers, then run the notebook.
jupyter lab SpatialMem.ipynb
```

**LLM-side debug harness** (no DA3/SAM3 needed — DA3/SAM3 are mocked):

```bash
python scripts/debug_llm.py --model gpt-4o-mini --n 2 \
    --types object_abs_distance object_counting room_size_estimation
```

This exercises Planner → SKILL → Reflector → 4-path evolution end-to-end. Useful for verifying prompt changes without waiting on GPU tool inference.

## Checkpoints — SKILL/Memory are the trained state, not the framework

The framework treats `skills/` and `memory/` at project root as the **base**: a read-only, seeded starting state. Every training run produces a **checkpoint** under `ckpts/{name}/` which snapshots the evolved SKILL library + Memory. This mirrors how model weights work in ML — the untrained model is fixed, ckpts store learned state.

**Rule enforced by the training script**: `--save-ckpt DST` is **required** for `--mode train`. Training never mutates the base. The base only changes when a human edits a seed SKILL.

**Basic ckpt workflow**:

```bash
# 1. Fresh training from the seeded base state
python -m scripts.train_and_eval --mode train --split train \
    --model qwen3-vl-30b-a3b-instruct --max-samples 1000 \
    --save-ckpt run_v1 --notes "first attempt"

# 2. Continue training from a previous ckpt
python -m scripts.train_and_eval --mode train --split train \
    --ckpt run_v1 --save-ckpt run_v2 --max-samples 1000

# 3. Eval a specific ckpt
python -m scripts.train_and_eval --mode eval --split test \
    --ckpt run_v1 --max-samples 4130 --workers 20

# 4. Eval the seeded base (untrained)
python -m scripts.train_and_eval --mode eval --split test \
    --ckpt base --max-samples 4130 --workers 20

# 5. Ckpt housekeeping
python -m scripts.manage_ckpts list
python -m scripts.manage_ckpts inspect run_v1
python -m scripts.manage_ckpts clone --src run_v1 --dst run_v1_backup
python -m scripts.manage_ckpts delete run_v1_old
```

Each ckpt directory contains a full snapshot:

```
ckpts/run_v1/
├── skills/                 # full SKILL library at checkpoint time
│   ├── measure_distance/       (seeded)
│   ├── count_objects/          (seeded)
│   ├── judge_direction/        (seeded, with accumulated Known Pitfalls)
│   ├── measure_scene_size/     (seeded)
│   ├── evolved_...             (evolved-and-promoted SKILLs)
│   └── pending/                (unpromoted drafts)
├── memory/                 # full Memory state
│   ├── object_size_priors.json
│   ├── scene_scale_priors.json
│   ├── unsolved_cases.jsonl
│   └── category_trajectories/
└── ckpt_meta.json          # created_at, base_ckpt, samples_seen, training_accuracy, ...
```

`ckpts/` is gitignored — checkpoints are per-user runtime state, not code. Share them via other channels (rsync, cloud storage) or by committing a specific artifact if needed.

## Configuration knobs (`agent/config.py`)

| Constant | Default | Meaning |
| --- | --- | --- |
| `MAX_FRAMES_PER_SAMPLE` | 10 | Uniform downsample from VSI's 32 per-scene frames |
| `MAX_ROUNDS` | 3 | Reflection loops in the raw-tool fallback path |
| `CONFIDENCE_THRESHOLD` | 0.8 | Reflector early-terminate |
| `WARMUP_N` | 5 | Numeric-prior candidate → main promotion threshold |
| `DISTILL_TRIGGER_N` | 5 | Fire `distill_success` every N SKILL successes |
| `DISTILL_WINDOW` | 20 | Recent trajectory summaries fed to the distillation LLM |
| `PENDING_PROMOTE_K` | 3 | Successful reuses before promotion out of `pending/` |
| `REFLECT_MAX_ATTEMPTS` | 1 | Reconstruct attempts per wrong answer |
| `BOOTSTRAP_TRIGGER_N` | 8 | Uncovered-category samples before proactive bootstrap fires |

All are env-var overridable (`BOOTSTRAP_TRIGGER_N=3` for faster testing, etc.).

## Disk usage: tool output artifacts

**Heads-up before running at scale.** The `tools/` layer writes intermediate artifacts to `outputs/` on every sample. When running a full 4K test set, this can grow to **tens of GB**. Contents:

| Path | Written by | Approximate size per sample |
| --- | --- | --- |
| `outputs/da3/prediction.npz` | `depth_estimation` — depth maps + intrinsics + extrinsics + processed images | ~10–50 MB (depends on frame count × resolution) |
| `outputs/da3/depth_vis/*.png` | `depth_estimation` — one colored PNG per frame | ~50 KB × N frames |
| `outputs/da3/metadata.json` | `depth_estimation` — small metadata | ~1 KB |
| `outputs/da3/bev_from_depth.png` | `bev_generation` (only when called) | ~200 KB |
| `outputs/da3/*.ply`, NVS frames | `novel_view_synthesis` / 3DGS export (only when called) | 10–100 MB when triggered |
| `outputs/sam3/{prompt_slug}/*.png` | `object_segmentation` — one PNG per prompt per frame | ~30 KB × N × #prompts |
| `outputs/spatial/*.json` | `instance_3d_localization`, `scene_size_computation` | ~10 KB |

For a full VSIBench eval (4130 samples × 10 frames × 2–3 object prompts), rough total: **~100 GB** of intermediate artifacts.

**Mitigation options** (pick per your needs):

1. **Point `outputs/` at scratch/tmp storage**. `outputs/` is already in `.gitignore` — it's disposable. Symlink or `export OUTPUTS_DIR=/scratch/spatialmem_outputs` (needs a small config patch, `output_root` currently reads `agent.config.OUTPUTS_DIR` at construction time).

2. **Periodically wipe between phases**. E.g., after a training run finishes, `rm -rf outputs/`. The framework re-creates the subdirs lazily via `_prepare_run`.

3. **Disable visualizations at the tool level** if you don't need them for debugging. In `agent/tools.py::_run_depth`, change `save_depth_vis=True` to `save_depth_vis=False` — this alone eliminates the depth_vis PNGs (the biggest volume). `sam3_segmentation.py::save_segmentation_results` is called unconditionally; disabling would require a small patch.

4. **Skip the numeric NPZ dump** by wrapping `save_prediction_outputs` differently. This is more invasive and only worth doing if disk is the bottleneck.

None of these artifacts are needed by the agent framework itself (SKILL/Memory state persists separately in `skills/` and `memory/` / `ckpts/`). They exist for **manual debugging** of individual tool calls (looking at depth maps, seg masks, etc.). If you're just training/evaluating and don't plan to inspect intermediate steps, feel free to point `OUTPUTS_DIR` at a scratch disk and forget about it.

## What was verified during development

- **All 4 reactive evolution paths trigger correctly** on real VSI samples. Verified via `scripts/debug_llm.py` with GPT-4o-mini and qwen3-vl-30b-a3b-instruct as the driver LLM.
- **Proactive bootstrap fires** for uncovered task types (`obj_appearance_order`, `route_planning`) that lack a seed SKILL — synthesizes a SKILL draft from accumulated trajectories once `BOOTSTRAP_TRIGGER_N` is crossed.
- **Pending SKILLs are actually promotable**: `SkillLib.retrieve(include_pending=True)` (now the default) offers pending SKILLs to the Planner, so they can accrue the successes needed for promotion (previously a catch-22 dead code path).
- **Small VLM can drive the framework**: `qwen3-vl-30b-a3b-instruct` (MoE, 3B active — ≈8B-equivalent) matches GPT-4o-mini on SKILL hit rate (76% vs 73%) and produces higher-quality distilled pitfalls in some cases. Supports the paper's plan of distilling into an 8B-class open-weight VLM.
- **Fifteen prompt- and integration-layer bugs found and fixed** during harness runs (Planner SKILL/tool confusion, missing Parameters visibility in `Skill.brief()`, syntax issues in evolved `execute.py`, non-object labels leaking into Memory, hallucinated tool names, output truncation, dead pending pool, scene_type never threaded, etc.). All logged in `docs/design.md`.
- **Structured-output JSON mode** (`response_format={"type": "json_object"}`) is used by default in `LLMClient.chat_json`, with automatic fallback if the endpoint doesn't support it. Cuts JSON parse failures on complex distillation prompts to near zero.

## Migration from the previous 4-role framework

If you have code that used the old modules, apply this map:

| Old import | New import |
| --- | --- |
| `from agent.agent import SpatialMemAgent, AgentResult` | `from agent.orchestrator import Orchestrator, AgentResult` |
| `from agent.memory_manager import MemoryManager` | `from agent.memory import Memory` |
| `from agent.tool_registry import ToolRegistry` | `from agent.tools import ToolRegistry` |
| `from agent.runner import AgentRunner, evaluate_answer` | `from agent.train_loop import TrainLoop, evaluate_answer` |
| `from agent.data_loader import SPARDataLoader` | `from agent.data_loader import VSIBenchDataLoader` |
| `from agent import roles` | Prompts now live inside `planner.py` / `reasoner.py` / `reflector.py` |

The old `memory_bank/` (Layer 1/2/3) is deprecated. Layer 1 numeric priors migrate to `memory/*.json`; Layer 2/3 do **not** migrate — those files were polluted with degenerate 15-step tool sequences and generic failure texts. The new system starts fresh from the four seed SKILLs and grows via the four evolution paths.

See `docs/design.md` for the full design rationale (English) or `docs/项目介绍.md` for the Chinese overview.
