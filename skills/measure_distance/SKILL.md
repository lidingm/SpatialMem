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
2. `object_segmentation` × N — one call per target in `targets`; produces per-frame 2D masks.
3. `instance_3d_localization` — back-project masks into 3D, cluster across frames.
4. `distance_computation` — Euclidean distance in the appropriate mode.

# Known Pitfalls

- **[Generic/functional object terms]** Object segmentation fails on abstract names (e.g., 'heater') → use common visual synonyms (e.g., 'radiator').
- **[Overly specific segmentation prompts]** Compound nouns like "ceiling light" yield zero detections → simplify to core category labels (e.g., 'light').
- **[Broad semantic prompts]** Generic names capture background clutter → add spatial context (e.g., 'on ceiling', 'on floor') to isolate the target instance.
- **[Large planar targets]** Flat/featureless objects (rugs, floor markings) are dropped during 3D back-projection → validate that the target appears in instance output; if missing, skip this skill.
- **[Multiple instances of same class]** When several instances of the target object exist, the skill picks the first found; this may not be the closest one → check if multiple instances are present and run distance_computation for each, then pick the minimum.
- **[Missing target in 3D clustering]** When 3D localization fails to detect or correctly label one target object (e.g., bucket merged into trash can), the model guesses the distance → explicitly compute pairwise distances between all high-confidence instances and validate against 2D segmentation masks
- **[Multiple instances of target object]** System defaults to the first or highest-score detection instead of the contextually intended one → filter instances by spatial proximity or explicit query cues before metric computation

# Examples

None yet.
