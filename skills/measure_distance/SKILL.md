---
name: measure_distance
task_categories:
- distance
total_calls: 0
success_count: 0
failure_count: 0
success_rate: 0.0
version: 1
seeded: true
---

# When to Use

Use this skill for queries requesting the distance between two distinct, standard household objects (e.g., 'tv and bookshelf', 'bathtub and dishwasher') where segmentation is reliable. It performs best with moderately sized, clearly visible items that are not physically touching or occluded. Avoid using it for small objects (e.g., 'printer') or pairs prone to segmentation ambiguity (e.g., 'trash bin' near 'door'), as these often result in significant underestimation or zero-distance errors.

# Parameters

- `targets` (list[str], **REQUIRED**): object names for segmentation. For `object_to_object` supply at least 2 (e.g. `["backpack", "table"]`); for `object_to_camera` at least 1. Use short, visually clear labels — this is the #1 failure mode.
- `mode` (str): `"object_to_object"` (default) or `"object_to_camera"`.
- `frame_index` (int): only for `object_to_camera` mode — which frame's camera position to measure from. Default `0`.

# Tool Sequence

1. `depth_estimation` — recover metric depth + camera poses for all frames.
2. `object_segmentation` × N — one call per target in `targets`; SAM3 tracks candidate instances across frames and returns masks, boxes, scores, and track IDs.
3. `instance_3d_localization` — back-project masks into 3D, cluster across frames.
4. `distance_computation` — Checker selects the intended track/frames for each target, then the tool computes Euclidean distance in the appropriate mode. Prefer passing object names from `targets`; do not invent placeholder instance IDs.

# Known Pitfalls

None yet.

# Examples

None yet.

# Checker

None yet.
