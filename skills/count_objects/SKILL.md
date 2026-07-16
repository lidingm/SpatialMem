---
name: count_objects
task_categories:
- counting
total_calls: 0
success_count: 0
failure_count: 0
success_rate: 0.0
version: 1
seeded: true
---

# When to Use

Use this skill when asked to count distinct, moderately spaced instances of common object categories (e.g., tables, chairs) in a scene. It successfully combines 2D segmentation with 3D localization to deduplicate detections across views. Avoid using it for highly cluttered scenes, heavily occluded objects, or categories with ambiguous boundaries, as it tends to overcount or undercount in those conditions.

# Parameters

- `target` (str, **REQUIRED**): the object category to count (e.g., `"chair"`, `"table"`). Empty string will cause the SKILL to fail.

# Tool Sequence

1. `depth_estimation` — required for 3D clustering downstream.
2. `object_segmentation` — detect all candidate instances of `target` per frame.
3. `instance_3d_localization` — cluster detections across frames by 3D proximity.
4. `instance_counting` — read `total_unique` from the clustering result.

# Known Pitfalls

None yet.

# Examples

None yet.
