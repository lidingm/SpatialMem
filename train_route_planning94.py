#!/usr/bin/env python3
"""
Self-evolving training run for a route_planning split carved out of VSI-Bench.

This script creates a deterministic split from the 194 `route_planning`
VSI-Bench samples:
  - first `train_count` samples after seeded shuffle -> train
  - the rest -> holdout

The split is saved to the results directory so later evaluation can exclude
the 94 training samples and use only the holdout subset.

Usage
-----
  python train_route_planning94.py
  python train_route_planning94.py --device cuda:7 --ckpt v2_route
  python train_route_planning94.py --train_count 94 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LLM_BASE_URL", "http://10.130.138.46:8010/v1")
os.environ.setdefault("LLM_API_KEY", "sk-Vd4l7HmLN1NK2MVi0NEKRA")
os.environ.setdefault("LLM_MODEL", "qwen3.6-plus")
os.environ.setdefault("SAM3_REPO", "/home/zhouruofan/Training-Free/tool_model/sam3")
os.environ.setdefault("SAM3_CHECKPOINT", "/home/zhouruofan/Training-Free/tool_model/sam3.1/sam3.1_multiplex.pt")
os.environ.setdefault("DA3_MODEL_DIR", "/home/zhouruofan/Training-Free/tool_model/DA3NESTED-GIANT-LARGE-1.1")
os.environ.setdefault("DA3_REPO", "/home/zhouruofan/Training-Free/tool_model/depth-anything-3")
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

RESULTS_SUBDIR = "train_route_planning94_v1"
TASK_TYPE = "route_planning"
PRINT_EVERY = 10
PROMOTE_EVERY = 50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_count", type=int, default=94,
                   help="Number of route_planning samples used for training (default: 94)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed used to shuffle route_planning samples before splitting")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_evolve", action="store_true",
                   help="Disable evolution (eval-only mode)")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="Print per-step agent output")
    p.add_argument("--ckpt", default="v2",
                   help="Checkpoint name under ckpts/ to read/write skills and memory (default: v2)")
    p.add_argument("--results_subdir", default=RESULTS_SUBDIR,
                   help=f"Subdirectory under results/ for jsonl and split files (default: {RESULTS_SUBDIR})")
    p.add_argument("--log_name", default="run.log",
                   help="Log filename under the results subdirectory (default: run.log)")
    return p.parse_args()


def save_split(results_dir: Path, sample_ids: list[str], train_count: int, seed: int) -> tuple[list[str], list[str]]:
    train_ids = sample_ids[:train_count]
    holdout_ids = sample_ids[train_count:]
    payload = {
        "task_type": TASK_TYPE,
        "seed": seed,
        "total_count": len(sample_ids),
        "train_count": len(train_ids),
        "holdout_count": len(holdout_ids),
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
    }

    split_json = results_dir / "route_planning_split.json"
    split_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (results_dir / "route_planning_train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (results_dir / "route_planning_holdout_ids.txt").write_text("\n".join(holdout_ids) + "\n", encoding="utf-8")
    return train_ids, holdout_ids


def main():
    args = parse_args()

    import torch
    from agent.checkpoint import ckpt_exists, ckpt_paths
    from agent.config import PENDING_PROMOTE_K, ensure_dirs
    from agent.data_loader import VSIBenchJSONLDataLoader
    from agent.evolve import evolve_after_sample
    from agent.llm_client import LLMClient
    from agent.memory import Memory
    from agent.orchestrator import Orchestrator
    from agent.planner import Planner
    from agent.reasoner import Reasoner
    from agent.reflector import Reflector
    from agent.skill_lib import SkillLib
    from agent.tools import ToolRegistry
    from agent.train_loop import evaluate_answer
    from tools.visual_generation.da3_geometry import DA3GeometryTool
    from tools.visual_generation.sam3_segmentation import SAM3SegmentationTool
    from train_vsitrain10k import (
        Tee,
        append_result,
        load_done_entries,
        print_summary,
        score_sample,
    )

    ensure_dirs()

    results_dir = PROJECT_ROOT / "results" / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / args.log_name
    log_f = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    print(f"\n{'='*60}")
    print(f"Run log: {log_path}")
    print(f"LLM model: {os.environ['LLM_MODEL']}")
    print(f"Task type: {TASK_TYPE}")
    print(f"{'='*60}\n")

    print(f"Initialising models on {args.device} ...")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        print(f"GPU: {torch.cuda.get_device_name()}")

    da3_tool = DA3GeometryTool(
        model_dir=os.environ["DA3_MODEL_DIR"],
        da3_repo=os.environ["DA3_REPO"],
        device=args.device,
    )
    sam3_tool = SAM3SegmentationTool(
        sam3_repo=os.environ["SAM3_REPO"],
        checkpoint_path=os.environ["SAM3_CHECKPOINT"],
        device=args.device,
        confidence_threshold=0.3,
    )
    llm = LLMClient(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
        temperature=1.0,
    )

    if not ckpt_exists(args.ckpt):
        print(f"[ERROR] checkpoint '{args.ckpt}' not found. "
              f"Create it first: python -c \"from agent.checkpoint import clone_from_base; clone_from_base('{args.ckpt}')\"")
        sys.exit(1)
    skills_root, memory_root = ckpt_paths(args.ckpt)
    print(f"Checkpoint: {args.ckpt}  (skills={skills_root}, memory={memory_root})")

    loader = VSIBenchJSONLDataLoader()
    all_samples = loader.load_task_type(TASK_TYPE, shuffle=True, seed=args.seed)
    all_ids = [s.id for s in all_samples]
    train_ids, holdout_ids = save_split(results_dir, all_ids, args.train_count, args.seed)
    train_id_set = set(train_ids)
    train_samples = [s for s in all_samples if s.id in train_id_set]

    print(f"Split saved to: {results_dir / 'route_planning_split.json'}")
    print(f"VSI-Bench route_planning total={len(all_samples)}  train={len(train_samples)}  holdout={len(holdout_ids)}")
    if len(all_samples) < args.train_count:
        print(f"[ERROR] requested train_count={args.train_count}, but only found {len(all_samples)} samples")
        sys.exit(1)

    memory = Memory(root=memory_root)
    skill_lib = SkillLib(root=skills_root, pending_dir=skills_root / "pending")
    tools = ToolRegistry(da3_tool=da3_tool, sam3_tool=sam3_tool, llm=llm, memory=memory)
    planner = Planner(llm)
    reasoner = Reasoner(llm)
    reflector = Reflector(llm)
    # Route training logs the ordered final context once per sample below.
    orch = Orchestrator(tools, memory, skill_lib, planner, reasoner, reflector,
                        verbose=False)

    evolve = not args.no_evolve
    results_file = results_dir / f"{TASK_TYPE}.jsonl"
    done_entries = load_done_entries(results_file)
    done_ids = set(done_entries)
    remaining = [s for s in train_samples if s.id not in done_ids]

    print(f"\n{'='*60}")
    print(f"[{TASK_TYPE}] train_total={len(train_samples)}  already done={len(done_ids)}  remaining={len(remaining)}")
    print(f"holdout ids are saved separately and never trained here")
    print(f"{'='*60}")

    qt_total = qt_ok = 0
    qt_score_sum = 0.0
    qt_metric = "score"
    qt_start = time.time()

    for prev in done_entries.values():
        if "score" not in prev:
            continue
        qt_total += 1
        score = float(prev.get("score") or 0.0)
        qt_score_sum += score
        if prev.get("success"):
            qt_ok += 1
        qt_metric = prev.get("metric") or qt_metric

    if not remaining:
        print("All train samples already processed — using saved results for summary.")
        print_summary(TASK_TYPE, qt_score_sum, qt_total, qt_metric, time.time() - qt_start, qt_ok)
        return

    with results_file.open("a", encoding="utf-8") as out_f:
        try:
            for i, sample in enumerate(remaining):
                idx = len(done_ids) + i + 1
                print(f"\n[SAMPLE] {TASK_TYPE} {idx}/{len(train_samples)} id={sample.id}")
                print(f"Q: {sample.question[:220]}")

                try:
                    result = orch.run(sample)
                    result.success = evaluate_answer(
                        result.predicted_answer, result.gt_answer,
                        result.answer_format, sample.task_type,
                    )
                except Exception:
                    traceback.print_exc()
                    print(f"[ERROR] sample {sample.id} failed, skipping.")
                    continue

                if args.verbose:
                    if result.chosen_skill:
                        skill_params = result.plan.get("skill_params") or {}
                        print(f"[PLAN] SKILL={result.chosen_skill}  params={skill_params}")
                    else:
                        steps = result.plan.get("plan", [])
                        plan_str = " -> ".join(s.get("tool", "?") for s in steps)
                        print(f"[PLAN] raw-tool: {plan_str or '(empty)'}")

                print("[SUMMARY]")
                print(result.final_context.strip() or "(empty)")

                score = score_sample(
                    result.predicted_answer, result.gt_answer,
                    result.answer_format, getattr(sample, "raw", {}).get("options"),
                )
                metric = "MRA" if result.answer_format != "select" else "ACC"
                status = "OK" if result.success else "WRONG"
                print(f"[RESULT] {status} {metric}={score:.3f} "
                      f"gt={result.gt_answer!r} pred={result.predicted_answer!r} "
                      f"time={result.duration_seconds:.1f}s skill={result.chosen_skill or 'none'}")

                append_result(out_f, result, sample)
                qt_total += 1
                qt_score_sum += score
                qt_metric = metric
                if result.success:
                    qt_ok += 1

                if evolve:
                    try:
                        update = evolve_after_sample(
                            result, sample, memory, skill_lib,
                            reflector, orch, llm, evaluate_answer,
                            verbose=args.verbose,
                        )
                        if args.verbose:
                            parts = [f"path={update['path']}"]
                            if update.get("skill_updates"):
                                parts.append(f"skill_updates={update['skill_updates']}")
                            if update.get("memory_updates"):
                                parts.append(f"mem={update['memory_updates']}")
                            if update.get("pending_saved"):
                                parts.append(f"NEW_SKILL={update['pending_saved']}")
                            if update.get("bootstrapped"):
                                parts.append(f"BOOTSTRAPPED={update['bootstrapped']}")
                            print(f"[EVOLVE] {'  '.join(parts)}")
                    except Exception:
                        traceback.print_exc()
                        print("[EVOLVE] evolution step failed, continuing.")

                if evolve and (i + 1) % PROMOTE_EVERY == 0:
                    promoted = skill_lib.promote_eligible_pending(PENDING_PROMOTE_K)
                    if promoted:
                        print(f"[PROMOTE] {promoted}")

                if (i + 1) % PRINT_EVERY == 0:
                    elapsed = time.time() - qt_start
                    avg_score = qt_score_sum / qt_total if qt_total else 0
                    print(f"\n  ↳ [{TASK_TYPE}] running {qt_metric}={avg_score:.3f}  "
                          f"OK={qt_ok}/{qt_total}  elapsed={elapsed:.0f}s  "
                          f"~{elapsed/(i+1):.1f}s/sample\n")

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Flushing and exiting cleanly ...")

    if evolve:
        promoted = skill_lib.promote_eligible_pending(PENDING_PROMOTE_K)
        if promoted:
            print(f"[PROMOTE final] {promoted}")

    print_summary(TASK_TYPE, qt_score_sum, qt_total, qt_metric, time.time() - qt_start, qt_ok)
    print(f"Holdout ids: {len(holdout_ids)} saved to {results_dir / 'route_planning_holdout_ids.txt'}")
    print("Suggested holdout eval command:")
    print(
        "  python eval_vsibench.py --types route_planning "
        f"--ckpt {args.ckpt} --route_split_json {results_dir / 'route_planning_split.json'} "
        "--route_partition holdout"
    )


if __name__ == "__main__":
    main()
