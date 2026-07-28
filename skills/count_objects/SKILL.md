---
name: count_objects
task_categories:
- counting
total_calls: 1
success_count: 1
failure_count: 0
success_rate: 1.0
version: 1
seeded: true
---

# When to Use

Use this skill when asked to count distinct, moderately spaced instances of common object categories (e.g., tables, chairs) in a scene. It successfully combines 2D segmentation with 3D localization to deduplicate detections across views. Avoid using it for highly cluttered scenes, heavily occluded objects, or categories with ambiguous boundaries, as it tends to overcount or undercount in those conditions.

# Parameters

- `target` (str, **REQUIRED**): the object category to count (e.g., `"chair"`, `"table"`). Empty string will cause the SKILL to fail.

# Tool Sequence

1. `depth_estimation` — required for 3D clustering downstream.
2. `object_segmentation` — use SAM3 video tracking to detect candidate instances of `target` across frames with stable track IDs.
3. `instance_3d_localization` — cluster detections across frames by 3D proximity.
4. `instance_counting` — let Checker review the target-specific annotated 32 frames when available, then read authoritative `total_unique`.

# Known Pitfalls

None yet.

# Examples

None yet.

# Checker

None yet.
