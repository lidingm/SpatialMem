# Memory

Declarative knowledge accumulated across training samples. Consumed by the agent for calibration and sanity-checking, not for procedural decisions.

## Contents

| File | Purpose |
| --- | --- |
| `object_size_priors.json` | Physical dimensions of common objects (width/height/depth in meters). Online Welford mean/std. Entries with `count < WARMUP_N` live under `candidates`; graduate to `objects` once warmed up. |
| `scene_scale_priors.json` | Room extent and camera height per scene type. Same warmup semantics. |
| `unsolved_cases.jsonl` | Samples where the agent answered wrong AND reflection could not reconstruct a working trajectory. Kept for later revisit. |

## Update semantics

Memory is **updated** (陈述性变化，累加或修正). See `agent/memory.py`.

- Numeric priors: online mean/std via Welford's algorithm.
- New categories: appear as new keys under `candidates`; promoted to the main dict once `count >= WARMUP_N`.
- `unsolved_cases.jsonl`: append-only.

Layer3-style "failure lessons" from the old memory bank are NOT stored here.
They live inside individual SKILLs' `SKILL.md` under the **Known Pitfalls** section, because failure modes are almost always tied to a specific procedure.
