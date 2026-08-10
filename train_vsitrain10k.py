#!/usr/bin/env python3
"""
Self-evolving training run on the full VSI-Train-10k dataset.

Iterates all 10,010 samples across 7 question types, running the
SpatialMem agent with evolution enabled (memory + SKILL accumulation).

Features
--------
- Resume: skips samples already logged in results/train_vsitrain10k/<qt>.jsonl
- Per-question-type result files + a global progress log
- Periodic SKILL promotion and summary prints
- Graceful keyboard-interrupt: flushes results before exit

Usage
-----
  python train_vsitrain10k.py                         # all types, 300 samples per type
  python train_vsitrain10k.py --types object_count    # single type
  python train_vsitrain10k.py --max_per_type 50       # quick test
  python train_vsitrain10k.py --no_evolve             # eval only, no evolution
  python train_vsitrain10k.py --device cuda:1         # pick GPU
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LLM_BASE_URL",  "http://10.130.138.46:8010/v1")
os.environ.setdefault("LLM_API_KEY",   "sk-ZAJy47c5eid1MW_wjx7Fpg")
os.environ.setdefault("LLM_MODEL",     "qwen3.6-plus")
os.environ.setdefault("SAM3_REPO",     "/home/zhouruofan/Training-Free/tool_model/sam3")
os.environ.setdefault("SAM3_CHECKPOINT", "/home/zhouruofan/Training-Free/tool_model/sam3.1/sam3.1_multiplex.pt")
os.environ.setdefault("DA3_MODEL_DIR", "/home/zhouruofan/Training-Free/tool_model/DA3NESTED-GIANT-LARGE-1.1")
os.environ.setdefault("DA3_REPO",      "/home/zhouruofan/Training-Free/tool_model/depth-anything-3")
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")  # suppress DA3 INFO timing logs

RESULTS_SUBDIR = "train_vsitrain10k_v2"
PROMOTE_EVERY  = 50   # promote eligible pending SKILLs every N samples
PRINT_EVERY    = 10   # print running score every N samples


class Tee:
    """Mirror stdout/stderr to a log file while keeping terminal output."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--types", nargs="*", default=None,
                   help="Question types to run (default: all 7)")
    p.add_argument("--max_per_type", type=int, default=300,
                   help="Max samples per question type (default: 300)")
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no_shuffle", dest="shuffle", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_evolve", action="store_true",
                   help="Disable evolution (eval-only mode)")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="Print per-step agent output")
    p.add_argument("--ckpt", default="v2",
                   help="Checkpoint name under ckpts/ to read/write skills and memory (default: v2)")
    p.add_argument("--results_subdir", default=RESULTS_SUBDIR,
                   help=f"Subdirectory under results/ for jsonl and logs (default: {RESULTS_SUBDIR})")
    p.add_argument("--log_name", default="run.log",
                   help="Log filename under the results subdirectory (default: run.log)")
    return p.parse_args()


def load_done_entries(results_file: Path) -> dict[str, dict]:
    """Read completed sample entries already written to a results file."""
    done = {}
    if results_file.exists():
        with results_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done[entry["sample_id"]] = entry
                except Exception:
                    pass
    return done


def append_result(f, result, sample) -> None:
    score = score_sample(result.predicted_answer, result.gt_answer,
                         result.answer_format, getattr(sample, "raw", {}).get("options"))
    entry = {
        "sample_id":        result.sample_id,
        "task_type":        result.task_type,
        "task_category":    result.task_category,
        "answer_format":    result.answer_format,
        "question":         result.question[:300],
        "gt_answer":        result.gt_answer,
        "predicted_answer": result.predicted_answer,
        "success":          result.success,
        "score":            score,
        "metric":           "MRA" if result.answer_format != "select" else "ACC",
        "confidence":       result.confidence,
        "num_rounds":       result.num_rounds,
        "duration_seconds": round(result.duration_seconds, 2),
        "chosen_skill":     result.chosen_skill,
        "skills_used":      result.skills_used,
        "tool_calls": [{"tool": t["tool_name"], "success": t["success"]}
                       for t in result.tool_calls],
    }
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    f.flush()


def print_summary(qt: str, score_sum: float, total: int, metric: str,
                  t_elapsed: float, ok_count: int):
    avg_score = score_sum / total if total else 0
    print(f"\n{'─'*50}")
    print(f"[{qt}] avg {metric}={avg_score:.3f}  |  OK={ok_count}/{total}  "
          f"|  {t_elapsed/max(total,1):.1f}s/sample")
    print(f"{'─'*50}")


def _extract_number(text: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+", str(text))
    return float(m.group()) if m else None


def _extract_option(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", str(text).upper())
    return m.group(1) if m else None


def mean_relative_accuracy(pred: str, gt: str) -> float:
    """VSI-style MRA over thresholds 0.5, 0.55, ..., 0.95."""
    p = _extract_number(pred)
    g = _extract_number(gt)
    if p is None or g is None:
        return 0.0
    if abs(g) < 1e-9:
        return 1.0 if abs(p - g) < 0.5 else 0.0
    rel_err = abs(p - g) / abs(g)
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    return sum(1.0 for t in thresholds if rel_err < (1.0 - t)) / len(thresholds)


def mc_accuracy(pred: str, gt: str, options: list[str] | None = None) -> float:
    g = _extract_option(gt) or str(gt).strip().upper()
    p = _extract_option(pred)
    if p is None and options:
        pl = str(pred).strip().lower()
        for opt in options:
            head, _, body = str(opt).partition(".")
            body = body.strip().lower()
            if body and (pl == body or body in pl):
                p = head.strip().upper()
                break
    return 1.0 if (p is not None and p == g) else 0.0


def score_sample(pred: str, gt: str, answer_format: str,
                 options: list[str] | None = None) -> float:
    if answer_format == "select":
        return mc_accuracy(pred, gt, options)
    return mean_relative_accuracy(pred, gt)


def main():
    args = parse_args()

    # ── Imports (after env vars are set) ──────────────────────────────
    import torch
    from agent.config import ensure_dirs
    from agent.data_loader import VSITrain10KDataLoader
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

    results_dir = PROJECT_ROOT / "results" / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / args.log_name
    log_f = log_path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    print(f"\n{'='*60}")
    print(f"Run log: {log_path}")
    print(f"LLM model: {os.environ['LLM_MODEL']}")
    print(f"{'='*60}\n")

    # ── Model init ────────────────────────────────────────────────────
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

    # ── Checkpoint ────────────────────────────────────────────────────
    from agent.checkpoint import ckpt_paths, ckpt_exists
    if not ckpt_exists(args.ckpt):
        print(f"[ERROR] checkpoint '{args.ckpt}' not found. "
              f"Create it first: python -c \"from agent.checkpoint import clone_from_base; clone_from_base('{args.ckpt}')\"")
        sys.exit(1)
    skills_root, memory_root = ckpt_paths(args.ckpt)
    print(f"Checkpoint: {args.ckpt}  (skills={skills_root}, memory={memory_root})")

    loader    = VSITrain10KDataLoader()
    memory    = Memory(root=memory_root)
    skill_lib = SkillLib(root=skills_root, pending_dir=skills_root / "pending")
    tools     = ToolRegistry(da3_tool=da3_tool, sam3_tool=sam3_tool, llm=llm, memory=memory)
    planner   = Planner(llm)
    reasoner  = Reasoner(llm)
    reflector = Reflector(llm)
    orch      = Orchestrator(tools, memory, skill_lib, planner, reasoner, reflector,
                             verbose=args.verbose, allow_post_skill_reflection=False)

    evolve = not args.no_evolve
    from agent.config import PENDING_PROMOTE_K

    # ── Question types ────────────────────────────────────────────────
    all_types = loader.question_types()
    run_types = args.types if args.types else all_types
    invalid   = set(run_types) - set(all_types)
    if invalid:
        print(f"Unknown question types: {invalid}. Available: {all_types}")
        sys.exit(1)

    print(f"\nQuestion types to run: {run_types}")
    print(f"Evolve: {evolve}  |  Max per type: {args.max_per_type or 'all'}")
    print(f"Results dir: {results_dir}\n")

    # ── Global counters ───────────────────────────────────────────────
    g_total = g_ok = 0
    g_score_sum = 0.0
    g_start = time.time()

    for qt in run_types:
        samples = loader.load_task_type(
            qt,
            max_samples=args.max_per_type,
            shuffle=args.shuffle,
            seed=args.seed,
        )

        results_file = results_dir / f"{qt}.jsonl"
        done_entries = load_done_entries(results_file)
        done_ids     = set(done_entries)
        remaining    = [s for s in samples if s.id not in done_ids]

        print(f"\n{'='*60}")
        print(f"[{qt}]  total={len(samples)}  already done={len(done_ids)}  "
              f"remaining={len(remaining)}")
        print(f"{'='*60}")

        qt_total = qt_ok = 0
        qt_score_sum = 0.0
        qt_metric = "score"
        qt_start   = time.time()

        # Count previously done results toward resumed summaries.
        for prev in done_entries.values():
            if "score" not in prev:
                continue
            qt_total += 1
            g_total += 1
            score = float(prev.get("score") or 0.0)
            qt_score_sum += score
            g_score_sum += score
            if prev.get("success"):
                qt_ok += 1
                g_ok += 1
            qt_metric = prev.get("metric") or qt_metric

        if not remaining:
            print("  All done — using saved results for summary.")
            print_summary(qt, qt_score_sum, qt_total, qt_metric,
                          time.time() - qt_start, qt_ok)
            continue

        with results_file.open("a", encoding="utf-8") as out_f:
            try:
                for i, sample in enumerate(remaining):
                    idx = len(done_ids) + i + 1
                    print(f"\n[SAMPLE] {qt} {idx}/{len(samples)} id={sample.id}")
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

                    # Log skill selection / raw-tool plan
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
                    qt_total   += 1
                    g_total    += 1
                    qt_score_sum += score
                    g_score_sum  += score
                    qt_metric = metric
                    if result.success:
                        qt_ok += 1
                        g_ok  += 1

                    # Evolution
                    if evolve:
                        try:
                            update = evolve_after_sample(
                                result, sample, memory, skill_lib,
                                reflector, orch, llm, evaluate_answer,
                                verbose=args.verbose,
                            )
                            parts = [
                                f"path={update.get('path')}",
                                f"used_skill={result.chosen_skill or 'none'}",
                                f"skill_updates={update.get('skill_updates') or 'none'}",
                                f"new_skill={update.get('pending_saved') or 'none'}",
                                f"bootstrapped={update.get('bootstrapped') or 'none'}",
                            ]
                            if update.get("memory_updates"):
                                parts.append(f"mem={update['memory_updates']}")
                            print(f"[EVOLVE] {'  '.join(parts)}")
                        except Exception:
                            traceback.print_exc()
                            print("[EVOLVE] evolution step failed, continuing.")

                    # Periodic SKILL promotion
                    if evolve and (i + 1) % PROMOTE_EVERY == 0:
                        promoted = skill_lib.promote_eligible_pending(PENDING_PROMOTE_K)
                        if promoted:
                            print(f"[PROMOTE] {promoted}")

                    # Running accuracy print
                    if (i + 1) % PRINT_EVERY == 0:
                        elapsed = time.time() - qt_start
                        avg_score = qt_score_sum / qt_total if qt_total else 0
                        print(f"\n  ↳ [{qt}] running {qt_metric}={avg_score:.3f}  "
                              f"OK={qt_ok}/{qt_total}  "
                              f"elapsed={elapsed:.0f}s  "
                              f"~{elapsed/(i+1):.1f}s/sample\n")

            except KeyboardInterrupt:
                print("\n[INTERRUPT] Flushing and exiting cleanly ...")

        # Final promotion for this question type
        if evolve:
            promoted = skill_lib.promote_eligible_pending(PENDING_PROMOTE_K)
            if promoted:
                print(f"[PROMOTE final] {promoted}")

        print_summary(qt, qt_score_sum, qt_total, qt_metric,
                      time.time() - qt_start, qt_ok)

    # ── Global summary ────────────────────────────────────────────────
    elapsed = time.time() - g_start
    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"  Global avg score: {g_score_sum/max(g_total,1):.3f}")
    print(f"  Global OK rate  : {g_ok}/{g_total} = "
          f"{g_ok/max(g_total,1):.1%}")
    print(f"  Total time      : {elapsed/3600:.1f}h  "
          f"({elapsed/max(g_total,1):.1f}s/sample)")
    print(f"  Results dir     : {results_dir}")
    print(f"  Checkpoint      : {args.ckpt}")
    print(f"  Memory dir      : {memory_root}")
    print(f"  Skills dir      : {skills_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
