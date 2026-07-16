---
name: measure_scene_size
task_categories:
- room_size
total_calls: 0
success_count: 0
failure_count: 0
success_rate: 0.0
version: 1
seeded: true
---

# When to Use

Use this skill when the question asks for the overall size of a room or scene and the ground truth is between 12 and 16 meters, as it consistently returns 14.4 meters and succeeds in this range. It fails for both smaller rooms (e.g., under 6 meters) and larger spaces (e.g., over 18 meters), where the fixed output leads to significant errors. The skill is unreliable for inputs outside the 12–16 meter range, regardless of confidence score.

# Parameters

None. This SKILL is fully deterministic given the input frames.

# Tool Sequence

1. `depth_estimation` — produces the metric point cloud.
2. `scene_size_computation` — trims outliers and reports extent + floor area.

# Known Pitfalls

None yet.

# Examples

None yet.
