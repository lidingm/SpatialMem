---
name: judge_direction
task_categories:
- spatial_relation
total_calls: 0
success_count: 0
failure_count: 0
success_rate: 0.0
version: 1
seeded: true
---

# When to Use

Use this SKILL when the task asks for the direction of object X relative to a person who is:
- Standing at/near an object and optionally facing another object, OR
- Described from the camera's perspective.
Works for all difficulty levels.

# Parameters

Extract ONLY bare object names — never full sentences or descriptions.
Do not provide placeholder instance IDs such as `<id_of_chair>` or `chair_instance_id`. Pass object-name parameters and let `direction_computation` resolve Checker-validated instances internally.

- `reference_target` (str, **REQUIRED**): anchor for the direction judgment ("is X to the left of **Y**" → Y). When the question says "to my left/right", this is the same as `viewpoint_target` (the person's own position is the anchor).
- `target_target` (str, **REQUIRED**): object whose direction is asked ("is **X** to the left of Y" → X).
- `viewpoint_type` (str): `"camera"` or `"object"`. If `viewpoint_target` is present, set `"object"`; otherwise use `"camera"`.
  - `"camera"`: observer is the camera at a specific video frame; forward = camera's own optical axis toward `reference_target`. Use when the question says "from the camera / from this viewpoint" or no explicit human position is mentioned.
  - `"object"`: observer is a person standing at `viewpoint_target`. Use when the question explicitly says "standing by X" or "if I am at X".
- `viewpoint_target` (str): **WHERE the person physically stands** (their position). Required when `viewpoint_type="object"`. **NOT the object they are looking at.**
  - "standing by the stool" → `viewpoint_target="stool"`
  - "standing by the sofa" → `viewpoint_target="sofa"`
- `facing_target` (str): **WHAT object the person faces** (defines the forward/front direction). This is the object they are LOOKING AT, not where they stand.
  - "facing the dishwasher" → `facing_target="dishwasher"`
  - "facing the tv" → `facing_target="tv"`
  - **If omitted**: forward direction defaults to viewpoint → reference_target (the person is implicitly assumed to face the reference object).
  - Required when `viewpoint_target == reference_target` (forward otherwise undefined — must provide `facing_target`).
  - **Must be a bare object name, NOT a full phrase.**
- `viewpoint_frame` (int): camera frame index when `viewpoint_type="camera"`. Default `0`.

> **CRITICAL**: `viewpoint_target` = WHERE you STAND. `facing_target` = WHAT you LOOK AT.
> "standing by the stool facing the dishwasher" → `viewpoint_target="stool"`, `facing_target="dishwasher"`.
> Do NOT put the facing object in `viewpoint_target`.

### Example mappings

| Question phrase | reference_target | target_target | viewpoint_target | facing_target |
|---|---|---|---|---|
| "standing by the **stool** facing the **dishwasher**, is the **stove** to my front-left?" | `"stool"` | `"stove"` | `"stool"` | `"dishwasher"` |
| "standing by the **bed** facing the **tv**, is the **radiator** to my left?" | `"bed"` | `"radiator"` | `"bed"` | `"tv"` |
| "from the **sofa**, is the **lamp** to the left of the **desk**?" | `"desk"` | `"lamp"` | `"sofa"` | *(none)* |
| "from the **sofa** facing the **window**, is the **lamp** to the left of the **desk**?" | `"desk"` | `"lamp"` | `"sofa"` | `"window"` |

# Tool Sequence

1. `depth_estimation`.
2. `object_segmentation` × K — one call per unique name in `{reference_target, target_target, viewpoint_target, facing_target}`.
3. `instance_3d_localization`.
4. `direction_computation` — pass the target names, not invented IDs; final instance IDs are resolved by Checker. The tool returns the direction label plus `angle_from_forward_deg`, `lr_angle_deg`, and `fb_angle_deg`. Forward direction is determined by viewpoint mode:
   - camera mode: forward = camera → reference
   - object mode, viewpoint ≠ reference (Case A): forward = viewpoint → reference
   - object mode, viewpoint ≈ reference (Case B "standing at X facing Y"): forward = reference → facing(Y)

# Known Pitfalls

None yet.

# Examples

None yet.

# Checker

None yet.
