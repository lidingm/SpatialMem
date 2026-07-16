#!/usr/bin/env python3
"""
VSI-Bench evaluation on the official test set (test.jsonl).

Runs the SpatialMem agent on each sample to produce a final answer (via the
FINALIZER / summarizer — NO evolution, NO GT-driven reconstruction, NO skill
extraction), then scores with the OFFICIAL VSI-Bench metrics:

  - Numerical tasks (answer_format == "fill"):
      MRA (Mean Relative Accuracy) — mean over thresholds θ∈{0.5,0.55,…,0.95}
      of 1[ |pred - gt| / |gt| < 1 - θ ].   (paper: "Thinking in Space")
  - Multiple-choice tasks (answer_format == "select"):
      exact option-letter accuracy.

route_planning is skipped by default (per request).

Usage
-----
  python eval_vsibench.py                              # all types except route_planning
  python eval_vsibench.py --types object_counting      # single type
  python eval_vsibench.py --max_per_type 50            # quick smoke test
  python eval_vsibench.py --device cuda:0 --ckpt init_v1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


@contextlib.contextmanager
def _silence():
    """Redirect stdout+stderr to devnull — hides SAM3/DA3 chatter, tqdm bars,
    checkpoint 'Missing keys' dumps, etc. Our own eval prints happen OUTSIDE
    this block, so they still reach the log."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LLM_BASE_URL",  "http://10.130.138.46:8010/v1")
os.environ.setdefault("LLM_API_KEY",   "sk-ZAJy47c5eid1MW_wjx7Fpg")
os.environ.setdefault("LLM_MODEL",     "qwen3.6-plus")
os.environ.setdefault("SAM3_REPO",     "/home/zhouruofan/Training-Free/tool_model/sam3")
os.environ.setdefault("SAM3_CHECKPOINT", "/home/zhouruofan/Training-Free/tool_model/sam3.1/sam3.1_multiplex.pt")
os.environ.setdefault("DA3_MODEL_DIR", "/home/zhouruofan/Training-Free/tool_model/DA3NESTED-GIANT-LARGE-1.1")
os.environ.setdefault("DA3_REPO",      "/home/zhouruofan/Training-Free/tool_model/depth-anything-3")
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

DEFAULT_TEST_JSONL = "/home/zhouruofan/datasets/VSI-Bench/test.jsonl"
RESULTS_SUBDIR = "eval_vsibench"
PRINT_EVERY = 10

# route_planning (and its corrupted variant) are excluded by default.
EXCLUDE_TYPES = {"route_planning", "e_planniroutng"}


# ─── Official VSI-Bench scoring ──────────────────────────────────────────

def _extract_number(text: str) -> float | None:
    import re
    m = re.search(r"[-+]?\d*\.?\d+", str(text))
    return float(m.group()) if m else None


def _extract_option(text: str) -> str | None:
    """Pull an A/B/C/D letter from a predicted answer."""
    import re
    m = re.search(r"\b([A-D])\b", str(text).upper())
    return m.group(1) if m else None


def mean_relative_accuracy(pred: str, gt: str) -> float:
    """MRA over θ∈{0.5,0.55,…,0.95} (10 thresholds)."""
    p = _extract_number(pred)
    g = _extract_number(gt)
    if p is None or g is None:
        return 0.0
    if abs(g) < 1e-9:
        return 1.0 if abs(p - g) < 0.5 else 0.0
    rel_err = abs(p - g) / abs(g)
    thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 .. 0.95
    return sum(1.0 for t in thresholds if rel_err < (1.0 - t)) / len(thresholds)


def mc_accuracy(pred: str, gt: str, options: list[str] | None) -> float:
    """1.0 if the predicted option letter matches gt, else 0.0.

    Robust to the model answering with the option TEXT instead of the letter.
    """
    g = _extract_option(gt) or str(gt).strip().upper()
    p = _extract_option(pred)
    if p is None and options:
        # model may have echoed the option text; map text -> letter
        pl = str(pred).strip().lower()
        for opt in options:
            # opt like "A. left"
            head, _, body = str(opt).partition(".")
            body = body.strip().lower()
            if body and (pl == body or body in pl):
                p = head.strip().upper()
                break
    return 1.0 if (p is not None and p == g) else 0.0


def score_sample(pred: str, gt: str, answer_format: str, options) -> float:
    if answer_format == "select":
        return mc_accuracy(pred, gt, options)
    return mean_relative_accuracy(pred, gt)  # numeric fill -> MRA


# ─── Load VSI-Bench test.jsonl -> SpatialSample ──────────────────────────

def load_test_samples(jsonl_path: str, images_root: Path):
    from agent.data_loader import SpatialSample, _list_scene_frames

    samples: dict[str, list] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qt = str(r["question_type"])
            options = r.get("options")
            has_options = bool(options)
            answer_format = "select" if has_options else "fill"

            question = str(r["question"])
            if has_options:
                question = question + "\n" + "\n".join(str(o) for o in options)

            scene_dir = images_root / str(r["dataset"]) / str(r["scene_name"])
            image_paths = _list_scene_frames(scene_dir)

            s = SpatialSample(
                id=str(r["id"]),
                question=question,
                gt_answer=str(r["ground_truth"]),
                image_paths=image_paths,
                task_type=qt,
                answer_format=answer_format,
                annotations={},
                raw={
                    "id": r["id"],
                    "dataset": str(r["dataset"]),
                    "scene_name": str(r["scene_name"]),
                    "question_type": qt,
                    "options": list(options) if has_options else None,
                },
            )
            samples.setdefault(qt, []).append(s)
    return samples


def load_done(results_file: Path) -> dict[str, float]:
    """Map already-scored sample_id -> score (for resume + aggregation)."""
    done = {}
    if results_file.exists():
        with results_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[str(r["sample_id"])] = float(r["score"])
                except Exception:
                    pass
    return done


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_jsonl", default=DEFAULT_TEST_JSONL)
    p.add_argument("--types", nargs="*", default=None,
                   help="Question types to eval (default: all in the file except route_planning)")
    p.add_argument("--max_per_type", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", default="init_v1",
                   help="Checkpoint under ckpts/ for skills+memory (read-only). "
                        "Pass 'base' to use the project's base skills/memory.")
    p.add_argument("--out", default=RESULTS_SUBDIR,
                   help="Results subdir under results/ (shared across parallel "
                        "workers — safe as long as each worker runs DISJOINT --types)")
    p.add_argument("--out_root", default=None,
                   help="Tool artifact dir (DA3/SAM3 intermediate files). Defaults "
                        "to outputs/eval_<device> so parallel workers don't clobber "
                        "each other's files.")
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from agent.config import ensure_dirs, VSI_IMAGES_ROOT, OUTPUTS_DIR
    from agent.llm_client import LLMClient
    from agent.memory import Memory
    from agent.orchestrator import Orchestrator
    from agent.planner import Planner
    from agent.reasoner import Reasoner
    from agent.reflector import Reflector
    from agent.skill_lib import SkillLib
    from agent.tools import ToolRegistry
    from tools.visual_generation.da3_geometry import DA3GeometryTool
    from tools.visual_generation.sam3_segmentation import SAM3SegmentationTool

    # Drop all INFO/DEBUG logging globally (SAM3/DA3 emit a lot). WARNING+ stays.
    import logging
    logging.disable(logging.INFO)

    ensure_dirs()
    results_dir = PROJECT_ROOT / "results" / args.out
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initialising models on {args.device} (SAM3/DA3 build output silenced) ...")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        print(f"GPU: {torch.cuda.get_device_name()}")

    with _silence():
        da3_tool = DA3GeometryTool(
            model_dir=os.environ["DA3_MODEL_DIR"], da3_repo=os.environ["DA3_REPO"],
            device=args.device)
        sam3_tool = SAM3SegmentationTool(
            sam3_repo=os.environ["SAM3_REPO"], checkpoint_path=os.environ["SAM3_CHECKPOINT"],
            device=args.device, confidence_threshold=0.3)
    print("Models ready.")
    llm = LLMClient(
        base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"], temperature=0.7)

    # Skills + memory (read-only; no evolution).
    if args.ckpt == "base":
        from agent.config import SKILLS_DIR, SKILLS_PENDING_DIR, MEMORY_DIR
        skills_root, pending_root, memory_root = SKILLS_DIR, SKILLS_PENDING_DIR, MEMORY_DIR
    else:
        from agent.checkpoint import ckpt_paths, ckpt_exists
        if not ckpt_exists(args.ckpt):
            print(f"[ERROR] checkpoint '{args.ckpt}' not found.")
            sys.exit(1)
        skills_root, memory_root = ckpt_paths(args.ckpt)
        pending_root = skills_root / "pending"
    print(f"Skills: {skills_root}\nMemory: {memory_root}")

    memory    = Memory(root=memory_root)
    skill_lib = SkillLib(root=skills_root, pending_dir=pending_root)
    # Per-worker tool artifact dir so parallel processes don't clobber each
    # other's DA3/SAM3 intermediate files (these are artifacts only; the actual
    # computation flows through in-memory context, so this is just for cleanliness).
    tool_out_root = Path(args.out_root) if args.out_root else \
        Path(OUTPUTS_DIR) / f"eval_{args.device.replace(':', '')}"
    tools     = ToolRegistry(da3_tool=da3_tool, sam3_tool=sam3_tool,
                             output_root=tool_out_root)
    print(f"Tool artifacts: {tool_out_root}")
    orch      = Orchestrator(tools, memory, skill_lib,
                             Planner(llm), Reasoner(llm), Reflector(llm),
                             verbose=args.verbose)

    # ── Load data ─────────────────────────────────────────────────────
    by_type = load_test_samples(args.test_jsonl, Path(VSI_IMAGES_ROOT))
    available = [qt for qt in by_type if qt not in EXCLUDE_TYPES]
    run_types = args.types if args.types else sorted(available)
    run_types = [qt for qt in run_types if qt not in EXCLUDE_TYPES]

    print(f"\nTest file : {args.test_jsonl}")
    print(f"Types     : {run_types}")
    print(f"Excluded  : {sorted(EXCLUDE_TYPES)}")
    print(f"Results   : {results_dir}\n")

    per_type_scores: dict[str, list[float]] = {}
    g_start = time.time()

    for qt in run_types:
        samples = by_type.get(qt, [])
        if args.max_per_type is not None:
            samples = samples[:args.max_per_type]
        if not samples:
            print(f"[{qt}] no samples, skipping.")
            continue

        results_file = results_dir / f"{qt}.jsonl"
        done = load_done(results_file)
        per_type_scores[qt] = list(done.values())  # carry resumed scores

        remaining = [s for s in samples if s.id not in done]
        metric_name = "MRA" if samples[0].answer_format == "fill" else "ACC"

        print(f"\n{'='*60}\n[{qt}] total={len(samples)} done={len(done)} "
              f"remaining={len(remaining)}  metric={metric_name}\n{'='*60}")

        with results_file.open("a", encoding="utf-8") as out_f:
            try:
                for i, sample in enumerate(remaining):
                    idx = len(done) + i + 1
                    print(f"\n# [{qt}] {idx}/{len(samples)} id={sample.id}  "
                          f"Q: {sample.question.splitlines()[0][:90]}")
                    try:
                        with _silence():                   # hide SAM3/DA3 chatter
                            result = orch.run(sample)      # summarizer -> answer
                        pred = result.predicted_answer
                    except Exception:
                        traceback.print_exc()
                        pred = ""
                    options = sample.raw.get("options")
                    score = score_sample(pred, sample.gt_answer, sample.answer_format, options)
                    per_type_scores[qt].append(score)

                    out_f.write(json.dumps({
                        "sample_id": sample.id, "task_type": qt,
                        "answer_format": sample.answer_format,
                        "gt": sample.gt_answer, "pred": pred,
                        "score": score,
                    }, ensure_ascii=False) + "\n")
                    out_f.flush()

                    print(f">>> {metric_name}={score:.3f}  pred={pred!r} gt={sample.gt_answer!r}")
                    if (i + 1) % PRINT_EVERY == 0:
                        cur = per_type_scores[qt]
                        print(f"  ↳ [{qt}] running {metric_name}={sum(cur)/len(cur):.3f} "
                              f"over {len(cur)}")
            except KeyboardInterrupt:
                print("\n[INTERRUPT] flushing and exiting ...")
                break

        cur = per_type_scores[qt]
        if cur:
            print(f"\n[{qt}] {metric_name} = {sum(cur)/len(cur):.4f}  (n={len(cur)})")

    # ── Final report ──────────────────────────────────────────────────
    print(f"\n{'='*60}\nVSI-Bench RESULTS\n{'='*60}")
    task_means = []
    for qt in sorted(per_type_scores):
        cur = per_type_scores[qt]
        if not cur:
            continue
        m = sum(cur) / len(cur)
        task_means.append(m)
        metric_name = "MRA" if by_type[qt][0].answer_format == "fill" else "ACC"
        print(f"  {qt:32s} {metric_name}={m*100:5.1f}%  (n={len(cur)})")
    if task_means:
        print(f"{'-'*60}")
        print(f"  {'AVG over evaluated tasks':32s}     {sum(task_means)/len(task_means)*100:5.1f}%")
    print(f"  Total time: {(time.time()-g_start)/3600:.2f}h")
    print(f"  Results dir: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
