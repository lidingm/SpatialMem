#!/usr/bin/env python3
"""
Self-evolving training run on the SPAR ScanNet subset.

Iterates the 17 SPAR-unique task types (not in VSI-Bench / VSI-Train-10k)
running the SpatialMem agent with evolution enabled.

Features
--------
- Annotation rendering: points/bboxes are drawn onto image copies before
  the agent sees them.
- Format preference: select > fill > sentence per task type.
- Resume: skips samples already logged in results/train_spar/<task_type>.jsonl
- Per-type result files + periodic accuracy prints.
- Graceful keyboard-interrupt.

Usage
-----
  python train_spar.py                              # all 15 types, 300/type
  python train_spar.py --types depth_prediction_oc  # single type
  python train_spar.py --max_per_type 50            # quick test
  python train_spar.py --no_evolve                  # eval only
  python train_spar.py --device cuda:1              # pick GPU
  python train_spar.py --ckpt my_ckpt               # checkpoint name
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

os.environ.setdefault("LLM_BASE_URL",    "http://10.130.138.46:8010/v1")
os.environ.setdefault("LLM_API_KEY",     "sk-ZAJy47c5eid1MW_wjx7Fpg")
os.environ.setdefault("LLM_MODEL",       "qwen3.6-plus")
os.environ.setdefault("SAM3_REPO",       "/home/zhouruofan/Training-Free/tool_model/sam3")
os.environ.setdefault("SAM3_CHECKPOINT", "/home/zhouruofan/Training-Free/tool_model/sam3.1/sam3.1_multiplex.pt")
os.environ.setdefault("DA3_MODEL_DIR",   "/home/zhouruofan/Training-Free/tool_model/DA3NESTED-GIANT-LARGE-1.1")
os.environ.setdefault("DA3_REPO",        "/home/zhouruofan/Training-Free/tool_model/depth-anything-3")
os.environ.setdefault("DA3_LOG_LEVEL",   "WARN")

RESULTS_SUBDIR = "train_spar"
PROMOTE_EVERY  = 50
PRINT_EVERY    = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpatialMem on SPAR ScanNet subset")
    p.add_argument("--types", nargs="*", default=None,
                   help="Task types to run (default: all 17 SPAR-unique types)")
    p.add_argument("--max_per_type", type=int, default=300,
                   help="Max samples per task type (default: 200)")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device (default: cuda:0)")
    p.add_argument("--ckpt", default="init_v1",
                   help="Checkpoint name under ckpts/ (default: init_v1)")
    p.add_argument("--no_evolve", action="store_true",
                   help="Disable evolution (eval-only mode)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-step agent output")
    return p.parse_args()


def load_done_ids(results_file: Path) -> set[str]:
    done: set[str] = set()
    if results_file.exists():
        with results_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
    return done


def append_result(f, result, sample) -> None:
    entry = {
        "sample_id":        result.sample_id,
        "task_type":        result.task_type,
        "task_category":    result.task_category,
        "answer_format":    result.answer_format,
        "question":         result.question[:300],
        "gt_answer":        result.gt_answer,
        "predicted_answer": result.predicted_answer,
        "success":          result.success,
        "confidence":       result.confidence,
        "num_rounds":       result.num_rounds,
        "duration_seconds": round(result.duration_seconds, 2),
        "chosen_skill":     result.chosen_skill,
        "skills_used":      result.skills_used,
        "tool_calls": [
            {"tool": t["tool_name"], "success": t["success"]}
            for t in result.tool_calls
        ],
    }
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    f.flush()


def print_summary(task_type: str, correct: int, total: int, elapsed: float) -> None:
    acc = correct / total if total else 0
    print(f"\n{'─'*50}")
    print(f"[{task_type}] {correct}/{total} = {acc:.1%}  |  {elapsed/max(total,1):.1f}s/sample")
    print(f"{'─'*50}")


def main() -> None:
    args = parse_args()

    import torch
    from agent.checkpoint import ckpt_exists, ckpt_paths
    from agent.config import PENDING_PROMOTE_K, ensure_dirs
    from agent.data_loader import SPARDataLoader, SPAR_UNIQUE_TASK_TYPES
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

    ensure_dirs()

    results_dir = PROJECT_ROOT / "results" / RESULTS_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Model init ─────────────────────────────────────────────────────
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
        temperature=0.7,
    )

    # ── Checkpoint ─────────────────────────────────────────────────────
    if not ckpt_exists(args.ckpt):
        print(
            f"[ERROR] Checkpoint '{args.ckpt}' not found.\n"
            f"Create it first:\n"
            f"  python -c \"from agent.checkpoint import clone_from_base; "
            f"clone_from_base('{args.ckpt}')\""
        )
        sys.exit(1)
    skills_root, memory_root = ckpt_paths(args.ckpt)
    print(f"Checkpoint : {args.ckpt}")
    print(f"  skills   : {skills_root}")
    print(f"  memory   : {memory_root}")

    loader    = SPARDataLoader()
    memory    = Memory(root=memory_root)
    skill_lib = SkillLib(root=skills_root, pending_dir=skills_root / "pending")
    tools     = ToolRegistry(da3_tool=da3_tool, sam3_tool=sam3_tool)
    planner   = Planner(llm)
    reasoner  = Reasoner(llm)
    reflector = Reflector(llm)
    orch      = Orchestrator(tools, memory, skill_lib, planner, reasoner, reflector,
                             verbose=args.verbose)

    evolve = not args.no_evolve

    # ── Task types ─────────────────────────────────────────────────────
    run_types = args.types if args.types else SPAR_UNIQUE_TASK_TYPES
    invalid   = set(run_types) - set(SPAR_UNIQUE_TASK_TYPES)
    if invalid:
        print(f"[ERROR] Unknown SPAR task types: {invalid}")
        print(f"Available: {SPAR_UNIQUE_TASK_TYPES}")
        sys.exit(1)

    print(f"\nTask types  : {run_types}")
    print(f"Max/type    : {args.max_per_type}")
    print(f"Evolve      : {evolve}")
    print(f"Results dir : {results_dir}\n")

    g_total = g_correct = 0
    g_start = time.time()

    for task_type in run_types:
        fmt = loader.best_format(task_type)
        if fmt is None:
            print(f"[SKIP] {task_type}: no JSONL data found under {loader.jsonl_root}")
            continue

        try:
            samples = loader.load_task_type(task_type, max_samples=args.max_per_type)
        except Exception:
            traceback.print_exc()
            print(f"[SKIP] {task_type}: failed to load samples.")
            continue

        results_file = results_dir / f"{task_type}.jsonl"
        done_ids     = load_done_ids(results_file)
        remaining    = [s for s in samples if s.id not in done_ids]

        print(f"\n{'='*60}")
        print(f"[{task_type}]  fmt={fmt}  total={len(samples)}  "
              f"done={len(done_ids)}  remaining={len(remaining)}")
        print(f"{'='*60}")

        if not remaining:
            print("  All done — skipping.")
            continue

        qt_correct = qt_total = 0
        qt_start   = time.time()

        with results_file.open("a", encoding="utf-8") as out_f:
            try:
                for i, sample in enumerate(remaining):
                    idx = len(done_ids) + i + 1
                    print(f"\n{'#'*55}")
                    print(f"# [{task_type}] {idx}/{len(samples)}  id={sample.id}")
                    print(f"# Q: {sample.question[:120]}")
                    print(f"{'#'*55}")

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

                    if result.chosen_skill:
                        print(f"[PLAN] SKILL={result.chosen_skill}  "
                              f"params={result.plan.get('skill_params') or {}}")
                    else:
                        steps    = result.plan.get("plan", [])
                        plan_str = " -> ".join(s.get("tool", "?") for s in steps)
                        print(f"[PLAN] raw-tool: {plan_str or '(empty)'}")

                    status = "✓ CORRECT" if result.success else "✗ WRONG"
                    print(f">>> {status} | pred={result.predicted_answer!r}  "
                          f"gt={result.gt_answer!r}  time={result.duration_seconds:.1f}s")

                    append_result(out_f, result, sample)
                    qt_total += 1
                    g_total  += 1
                    if result.success:
                        qt_correct += 1
                        g_correct  += 1

                    if evolve:
                        try:
                            update = evolve_after_sample(
                                result, sample, memory, skill_lib,
                                reflector, orch, llm, evaluate_answer,
                                verbose=args.verbose,
                            )
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
                        acc = qt_correct / qt_total if qt_total else 0
                        print(f"\n  ↳ [{task_type}] acc={qt_correct}/{qt_total}={acc:.1%}  "
                              f"elapsed={elapsed:.0f}s  "
                              f"~{elapsed/(i+1):.1f}s/sample\n")

            except KeyboardInterrupt:
                print("\n[INTERRUPT] Flushing and exiting cleanly ...")
                break

        if evolve:
            promoted = skill_lib.promote_eligible_pending(PENDING_PROMOTE_K)
            if promoted:
                print(f"[PROMOTE final] {promoted}")

        print_summary(task_type, qt_correct, qt_total, time.time() - qt_start)

    elapsed = time.time() - g_start
    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"  Global accuracy : {g_correct}/{g_total} = "
          f"{g_correct/max(g_total, 1):.1%}")
    print(f"  Total time      : {elapsed/3600:.1f}h  "
          f"({elapsed/max(g_total,1):.1f}s/sample)")
    print(f"  Results dir     : {results_dir}")
    print(f"  Checkpoint      : {args.ckpt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
