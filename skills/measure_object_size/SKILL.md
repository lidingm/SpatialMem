---
name: measure_object_size
task_categories:
- object_size
total_calls: 0
success_count: 0
failure_count: 0
success_rate: 0.0
version: 1
seeded: true
---

# When to Use

Use this skill when the question asks for the physical size of one target object, such as its height, width, depth, longest side, or shortest side. It is intended for common indoor objects that can be segmented by text and localized in 3D. It uses Checker validation before computing the final size, so it can handle multiple candidate tracks of the same category better than directly taking the first instance.

# Parameters

- `object_name` (str, **REQUIRED**): the bare target object category or short description to segment and measure, e.g. `"chair"`, `"table"`, `"cabinet"`.
- `dimension` (str): one of `"height"`, `"width"`, `"depth"`, `"longest"`, or `"shortest"`. Default `"longest"` if the question does not specify a dimension.
- `unit` (str): `"m"` or `"cm"`. Default `"m"`.

# Tool Sequence

1. `depth_estimation` — recover metric depth and camera poses.
2. `object_segmentation` — detect all candidate tracks for `object_name`.
3. `instance_3d_localization` — back-project masks into 3D and merge object tracks.
4. `object_size_computation` — ask Checker to select the intended instance/frames, then report bbox_size_xyz, bbox volume, and the queried width/height/depth/longest/shortest value requested by `dimension`.

# Known Pitfalls

None yet.

# Examples

None yet.

# Checker

None yet.
