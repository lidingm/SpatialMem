"""End-to-end LLM-side debug script for SpatialMem.

Exercises Planner / Reflector.finalize / evolve_after_sample against real
VSIBench samples via the stepfun models-proxy. DA3 and SAM3 are mocked so
the run doesn't need model weights on this machine.

Run:
    export MODEL_PROXY_TOKEN=...     # already exported in this env
    python -m scripts.debug_llm --model gpt-4o-mini --n 2

What it verifies:
  1. Planner produces a valid JSON plan for real VSI questions
  2. It correctly picks the seeded SKILL for distance / counting / direction / room-size
  3. Reflector.finalize returns a well-formatted answer
  4. evolve_after_sample dispatches to the right path (A/B/C/D)
     and actually writes to skills/*/SKILL.md and trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make repo importable when run as `python scripts/debug_llm.py`
_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agent.config import ensure_dirs
from agent.data_loader import VSIBenchDataLoader
from agent.evolve import evolve_after_sample
from agent.llm_client import LLMClient
from agent.memory import Memory
from agent.orchestrator import AgentResult, Orchestrator
from agent.planner import Planner
from agent.reasoner import Reasoner
from agent.reflector import Reflector
from agent.skill_lib import SkillLib
from agent.train_loop import evaluate_answer
from agent.tools import ToolRegistry


# ─── Mocks for DA3 / SAM3 ─────────────────────────────────────────────

class FakeDA3:
    """Return synthetic depth + intrinsics + extrinsics of plausible shape."""
    def run(self, frame_paths, output_dir, process_res=504, save_depth_vis=True):
        import numpy as np
        n = len(frame_paths)
        H, W = 96, 128
        depth = np.random.uniform(0.5, 5.0, size=(n, H, W)).astype("float32")
        intr = np.tile(np.array([[100., 0., W / 2.],
                                 [0., 100., H / 2.],
                                 [0., 0., 1.]], dtype="float32"), (n, 1, 1))
        extr = np.tile(np.eye(4, dtype="float32")[:3], (n, 1, 1))
        conf = np.ones((n, H, W), dtype="float32")
        pred = type("Pred", (), {"gaussians": None})()
        return {
            "prediction": pred,
            "depth": depth, "intrinsics": intr, "extrinsics": extr,
            "conf": conf, "is_metric": 1, "scale_factor": 1.0,
        }

    def make_bev(self, prediction, output_dir):
        return {"raw_size_hw": (200, 200), "xy_extent": (4.0, 4.0)}

    def render_nvs(self, prediction, frame_index, movement, angle_deg, distance, output_dir):
        return None

    def infer(self, frame_paths, infer_gs=False):
        return self.run(frame_paths, output_dir=None)["prediction"]


class FakeSAM3:
    """Return plausible detections. Detection count varies deterministically by
    text_prompt hash so different prompts produce different instance counts —
    important for counting-question plumbing tests."""
    def segment_frames(self, frame_paths, text_prompt, output_dir):
        import numpy as np
        H, W = 96, 128
        # hash-based per-prompt detection count in [1, 3]
        n_det = 1 + (abs(hash(text_prompt)) % 3)
        results = []
        for fi, _ in enumerate(frame_paths):
            # Vary detection presence per-frame so not every frame has all dets
            visible = n_det if fi % 2 == 0 else max(1, n_det - 1)
            boxes = np.array([[10 + 20 * k, 10 + 15 * k,
                               50 + 20 * k, 50 + 15 * k]
                              for k in range(visible)], dtype="float32")
            results.append({
                "masks": np.ones((visible, H, W), dtype=bool),
                "boxes": boxes,
                "scores": np.full((visible,), 0.9, dtype="float32"),
                "labels": [text_prompt] * visible,
                "image_size": (W, H),
            })
        return results


# ─── Utilities ────────────────────────────────────────────────────────

def _monkey_patch_point_cloud():
    """Bypass tools.visual_generation.da3_geometry.prediction_to_point_cloud
    so tools._run_depth doesn't need the real DA3 postprocessing."""
    import numpy as np
    import types
    mod = types.ModuleType("tools.visual_generation.da3_geometry")
    def prediction_to_point_cloud(prediction, use_conf=False):
        pts = np.random.uniform(-2, 2, size=(500, 3)).astype("float32")
        cols = np.zeros_like(pts)
        return pts, cols
    mod.prediction_to_point_cloud = prediction_to_point_cloud
    sys.modules["tools.visual_generation.da3_geometry"] = mod


def _monkey_patch_code_tools():
    """Patch instance_3d_localization, distance_computation, direction_computation,
    instance_counting, scene_size_computation so that their outputs are shaped
    correctly for the tool_registry.get_context_summary formatter."""
    import numpy as np
    import types

    inst3d = types.ModuleType("tools.code_execution.instance_3d_localization")
    def compute_instance_3d_positions(seg_results, depth, intrinsics, extrinsics, output_dir):
        # Simulate 3D clustering: pick min(#labels, 4) unique instances
        # so counting queries produce varied results.
        all_labels = []
        for s in seg_results:
            all_labels.extend(s.get("labels", []) or [])
        unique_labels = list(dict.fromkeys(all_labels))
        k = max(1, min(len(unique_labels), 4))
        positions = np.array([
            [float(i) * 0.8, 0.0, 1.0 + float(i) * 0.3]
            for i in range(k)
        ])
        sizes = np.array([[0.5, 0.9, 0.5]] * k)
        scores = np.full(k, 0.9)
        return {
            "obj_id_list": list(range(k)),
            "merged_positions": positions,
            "merged_bbox_size": sizes,
            "merged_scores": scores,
            "merged_labels": unique_labels[:k] or ["obj"] * k,
        }
    def annotations_to_detections(annotations, n_frames, image_size, frame_index=0):
        return []
    inst3d.compute_instance_3d_positions = compute_instance_3d_positions
    inst3d.annotations_to_detections = annotations_to_detections
    sys.modules["tools.code_execution.instance_3d_localization"] = inst3d

    dist = types.ModuleType("tools.code_execution.distance_computation")
    def distance_object_to_object(a, b): return float(np.linalg.norm(a - b))
    def distance_object_to_camera(pos, extrinsics, frame_index): return float(np.linalg.norm(pos))
    def get_camera_position(extrinsics, frame_index): return np.array([0.0, 0.0, 0.0])
    dist.distance_object_to_object = distance_object_to_object
    dist.distance_object_to_camera = distance_object_to_camera
    dist.get_camera_position = get_camera_position
    sys.modules["tools.code_execution.distance_computation"] = dist

    direc = types.ModuleType("tools.code_execution.direction_computation")
    def compute_relative_direction(viewpoint, reference, target, extrinsics):
        return {"direction": "right", "lr_angle_deg": 30.0, "fb_angle_deg": 0.0}
    direc.compute_relative_direction = compute_relative_direction
    sys.modules["tools.code_execution.direction_computation"] = direc

    count = types.ModuleType("tools.code_execution.instance_counting")
    def count_unique_instances(r3d):
        return {"total_unique": len(r3d["obj_id_list"]), "obj_id_list": r3d["obj_id_list"]}
    count.count_unique_instances = count_unique_instances
    sys.modules["tools.code_execution.instance_counting"] = count

    scene = types.ModuleType("tools.code_execution.scene_size_computation")
    def compute_scene_size(pts, conf=None, output_dir=None):
        return {
            "extent_xyz": np.array([4.5, 2.8, 3.2]),
            "floor_area": 14.4, "height": 2.8,
            "n_points_clean": 400, "n_points_raw": 500,
        }
    scene.compute_scene_size = compute_scene_size
    sys.modules["tools.code_execution.scene_size_computation"] = scene


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Model name available on the stepfun proxy")
    parser.add_argument("--n", type=int, default=2, help="Samples per question_type")
    parser.add_argument("--types", nargs="+",
                        default=["object_abs_distance", "object_counting",
                                 "room_size_estimation"],
                        help="Question types to smoke-test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _monkey_patch_point_cloud()
    _monkey_patch_code_tools()
    ensure_dirs()

    llm = LLMClient(
        base_url=os.environ.get("LLM_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"),
        api_key=os.environ.get("LLM_API_KEY", os.environ.get("MODEL_PROXY_TOKEN", "")),
        model=args.model,
        temperature=0.3,
    )
    print(f"[LLM] endpoint={llm.base_url}  model={args.model}\n")

    loader = VSIBenchDataLoader()
    memory = Memory()
    skill_lib = SkillLib()
    tools = ToolRegistry(da3_tool=FakeDA3(), sam3_tool=FakeSAM3())
    planner = Planner(llm)
    reasoner = Reasoner(llm)
    reflector = Reflector(llm)
    orch = Orchestrator(tools, memory, skill_lib, planner, reasoner, reflector,
                        verbose=True)

    overall: list[dict] = []
    for qt in args.types:
        samples = loader.load_task_type(qt, max_samples=args.n, shuffle=True, seed=args.seed)
        print(f"\n{'#' * 60}\n# question_type={qt}   samples={len(samples)}\n{'#' * 60}")
        for sample in samples:
            print(f"\n--- sample {sample.id} : {sample.question[:120]}")
            try:
                result = orch.run(sample)
            except Exception as e:
                print(f"[FAIL] orchestrator raised: {e}")
                continue
            result.success = evaluate_answer(
                result.predicted_answer, sample.gt_answer,
                sample.answer_format, sample.task_type)

            try:
                upd = evolve_after_sample(
                    result, sample, memory, skill_lib, reflector, orch, llm,
                    evaluate_answer, verbose=True,
                )
            except Exception as e:
                print(f"[EVOLVE FAIL] {e}")
                upd = {"path": None, "error": str(e)}
            summary = {
                "id": sample.id, "qt": qt,
                "pred": result.predicted_answer, "gt": sample.gt_answer,
                "success": result.success,
                "chosen_skill": result.chosen_skill,
                "evolve_path": upd.get("path"),
                "evolve_updates": {
                    "skill_updates": upd.get("skill_updates", []),
                    "memory_updates": upd.get("memory_updates", []),
                    "pending_saved": upd.get("pending_saved"),
                },
            }
            overall.append(summary)
            print(f"[OK] {json.dumps(summary, ensure_ascii=False)}")

    print(f"\n{'=' * 60}\nSUMMARY  ({len(overall)} samples)")
    print(f"  correct: {sum(1 for s in overall if s['success'])}")
    print(f"  path A/B/C/D: "
          f"{sum(1 for s in overall if s['evolve_path'] == 'A')}/"
          f"{sum(1 for s in overall if s['evolve_path'] == 'B')}/"
          f"{sum(1 for s in overall if s['evolve_path'] == 'C')}/"
          f"{sum(1 for s in overall if s['evolve_path'] == 'D')}")
    print(f"  SKILL chosen: "
          f"{sum(1 for s in overall if s['chosen_skill'])}/{len(overall)}")

    print("\n=== SKILL library after run ===")
    for s in skill_lib.load_all():
        fm = s.frontmatter
        print(f"  {s.name:22s}  v={fm.get('version')}  "
              f"calls={fm.get('total_calls')}  succ={fm.get('success_count')}  "
              f"succ_rate={fm.get('success_rate')}  "
              f"pitfalls_len={len(s.sections.get('Known Pitfalls', ''))}")
    pending = [p.name for p in (skill_lib.pending_dir).iterdir() if p.is_dir()]
    print(f"\n=== Pending SKILLs: {pending}")


if __name__ == "__main__":
    main()
