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
  python eval_vsibench.py --device cuda:0 --ckpt base
  python eval_vsibench.py --ckpt base --types object_counting --llm_base_url http://127.0.0.1:8000/v1 --llm_model Qwen3-VL-8B-Instruct
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

RESULTS_SUBDIR = "eval_vsibench"
PRINT_EVERY = 10

# route_planning (and its corrupted variant) are excluded by default.
EXCLUDE_TYPES = {"route_planning", "e_planniroutng"}
ROUTE_PLANNING_TYPES = {"route_planning", "e_planniroutng"}


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


# ─── Pipeline intermediates ─────────────────────────────────────────────

def collect_stages(tools) -> dict:
    """Pull the pipeline's intermediate numbers out of the tool context.

    The agent's own prints are silenced during the run, so read the state
    directly afterwards. Recording base_count vs adjusted_count is what makes
    the VLM calibration's effect measurable WITHIN a single run — comparing two
    runs conflates it with any other code change.
    """
    st: dict = {}
    try:
        ctx = tools.context
        seg = ctx.get("all_seg_results") or {}
        if seg:
            st["n_tracks"] = {p: len({int(o) for s in segs for o in (s.get("obj_ids") or [])})
                              for p, segs in seg.items()}
        appearance = ctx.get("appearance_order_result") or {}
        if appearance:
            per_object = appearance.get("per_object") or {}
            st["appearance_first_frames"] = {
                k: (None if (not v.get("found") or v.get("first_frame") == float("inf")) else int(v.get("first_frame")))
                for k, v in per_object.items()
            }
            st["appearance_found"] = {k: bool(v.get("found")) for k, v in per_object.items()}
            st["appearance_semantic_order"] = list(appearance.get("semantic_order") or [])
        r3d = ctx.get("results_3d")
        if r3d is not None:
            st["n_clusters_3d"] = len(r3d.get("obj_id_list", []))
            # per-category breakdown (distance/direction cluster 2 objects together)
            insts = r3d.get("instances")
            if insts:
                by_cat = {}
                for it in insts:
                    by_cat[it.get("category", "?")] = by_cat.get(it.get("category", "?"), 0) + 1
                if len(by_cat) > 1:
                    st["n_clusters_by_cat"] = by_cat
        cnt = ctx.get("counting")
        if cnt is not None:
            st["count_final"] = cnt.get("total_unique")
            cal = cnt.get("calibration")
            if cal and cal.get("error") is None:
                st["calib_base"] = cal.get("base_count")
                st["calib_adjusted"] = cal.get("adjusted_count")
                st["calib_applied"] = cal.get("applied", [])
                raw = cal.get("raw") or {}
                st["calib_raw_response"] = raw.get("_raw_response", raw)
            elif cal:
                st["calib_error"] = cal.get("error")
        # distance/direction: what the VLM picked + the measured distance
        dl = ctx.get("distance_localization")
        if dl:
            st["vlm_pick"] = {n: ([l["instance_label"], l["keep_frames"], l.get("reason", "")]
                                  if l else None) for n, l in dl.items()}
        dist = ctx.get("distance")
        if dist is not None:
            st["distance_m"] = round(float(dist["meters"]), 3)
            st["distance_pair"] = f"{dist['a']}<->{dist['b']}"
    except Exception:
        pass
    return st


def print_stages(st: dict) -> None:
    if not st:
        return
    bits = []
    if "n_tracks" in st:
        bits.append("tracks=" + ",".join(f"{k}:{v}" for k, v in st["n_tracks"].items()))
    if "n_clusters_by_cat" in st:
        bits.append("3D_clusters=" + ",".join(f"{k}:{v}" for k, v in st["n_clusters_by_cat"].items()))
    elif "n_clusters_3d" in st:
        bits.append(f"3D_clusters={st['n_clusters_3d']}")
    if "calib_base" in st:
        bits.append(f"calib={st['calib_base']}->{st['calib_adjusted']}")
    elif "calib_error" in st:
        bits.append(f"calib_ERR={st['calib_error']}")
    if "count_final" in st:
        bits.append(f"count={st['count_final']}")
    if "distance_m" in st:
        bits.append(f"dist={st['distance_m']}m ({st.get('distance_pair','')})")
    if bits:
        print(f"    stages: {'  |  '.join(bits)}")
    for a in st.get("calib_applied", []):
        print(f"      - {a}")
    if "calib_raw_response" in st:
        print(f"[CHECKER RAW] count calibration:\n{st['calib_raw_response']}")
    elif "calib_error" in st:
        print(f"[CHECKER RAW] count calibration ERROR: {st['calib_error']}")
    for n, v in (st.get("vlm_pick") or {}).items():
        if v is None:
            print(f"      VLM {n}: no candidate")
        else:
            lbl, keep, reason = v
            print(f"      VLM {n}: {lbl}  frames={keep}" + (f"  — {reason}" if reason else ""))


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


def load_excluded_ids(path_str: str | None) -> set[str]:
    ids: set[str] = set()
    if not path_str:
        return ids
    path = Path(path_str)
    with path.open(encoding="utf-8") as f:
        for line in f:
            sample_id = line.strip()
            if sample_id:
                ids.add(sample_id)
    return ids


def load_route_split(path_str: str | None) -> dict | None:
    if not path_str:
        return None
    path = Path(path_str)
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    from agent.config import VSI_TEST_JSONL
    p = argparse.ArgumentParser()
    p.add_argument("--test_jsonl", default=str(VSI_TEST_JSONL))
    p.add_argument("--types", nargs="*", default=None,
                   help="Question types to eval (default: all in the file except route_planning)")
    p.add_argument("--max_per_type", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", default="base",
                   help="Checkpoint under ckpts/ for skills+memory (read-only). "
                        "Pass 'base' to use the project's base skills/memory.")
    p.add_argument("--out", default=RESULTS_SUBDIR,
                   help="Results subdir under results/ (shared across parallel "
                        "workers — safe as long as each worker runs DISJOINT --types)")
    p.add_argument("--out_root", default=None,
                   help="Tool artifact dir (DA3/SAM3 intermediate files). Defaults "
                        "to outputs/eval_<device> so parallel workers don't clobber "
                        "each other's files.")
    p.add_argument("--llm_base_url", default=os.environ.get("LLM_BASE_URL"),
                   help="OpenAI-compatible LLM/VLM base URL, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--llm_model", default=os.environ.get("LLM_MODEL"),
                   help="Model id served by /v1/models")
    p.add_argument("--llm_api_key", default=os.environ.get("LLM_API_KEY", "EMPTY"),
                   help="API key for OpenAI-compatible endpoint; local servers often accept EMPTY")
    p.add_argument("--include_route_planning", action="store_true", default=False,
                   help="Allow route_planning evaluation. By default it is still skipped.")
    p.add_argument("--exclude_ids_file", default=None,
                   help="Optional newline-separated sample-id file to exclude before evaluation.")
    p.add_argument("--route_split_json", default=None,
                   help="Optional route_planning split json written by train_route_planning94.py.")
    p.add_argument("--route_partition", choices=["holdout", "train", "all"], default="holdout",
                   help="When --route_split_json is provided, evaluate the holdout or train partition. Default: holdout.")
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
    llm_base_url = args.llm_base_url or os.environ["LLM_BASE_URL"]
    llm_model = args.llm_model or os.environ["LLM_MODEL"]
    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY", "EMPTY")
    print(f"LLM endpoint: {llm_base_url} | model: {llm_model}")
    llm = LLMClient(
        base_url=llm_base_url, api_key=llm_api_key,
        model=llm_model, temperature=0.7)

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
                             output_root=tool_out_root, llm=llm, memory=memory)
    print(f"Tool artifacts: {tool_out_root}")
    orch      = Orchestrator(tools, memory, skill_lib,
                             Planner(llm), Reasoner(llm), Reflector(llm),
                             verbose=args.verbose)

    # ── Load data ─────────────────────────────────────────────────────
    by_type = load_test_samples(args.test_jsonl, Path(VSI_IMAGES_ROOT))
    route_split = load_route_split(args.route_split_json)
    excluded_ids = load_excluded_ids(args.exclude_ids_file)
    effective_exclude_types = set(EXCLUDE_TYPES)
    if args.include_route_planning or route_split is not None:
        effective_exclude_types -= ROUTE_PLANNING_TYPES

    available = [qt for qt in by_type if qt not in effective_exclude_types]
    run_types = args.types if args.types else sorted(available)
    run_types = [qt for qt in run_types if qt not in effective_exclude_types]

    if route_split is not None:
        if args.route_partition == "train":
            selected_ids = set(route_split.get("train_ids", []))
        elif args.route_partition == "holdout":
            selected_ids = set(route_split.get("holdout_ids", []))
        else:
            selected_ids = set(route_split.get("train_ids", [])) | set(route_split.get("holdout_ids", []))
        if not selected_ids:
            raise SystemExit(f"[ERROR] no ids found for route partition: {args.route_partition}")
    else:
        selected_ids = set()

    print(f"\nTest file : {args.test_jsonl}")
    print(f"Types     : {run_types}")
    print(f"Excluded  : {sorted(effective_exclude_types)}")
    if args.exclude_ids_file:
        print(f"Exclude IDs: {args.exclude_ids_file}  (n={len(excluded_ids)})")
    if route_split is not None:
        print(f"Route split: {args.route_split_json}  partition={args.route_partition}  (n={len(selected_ids)})")
    print(f"Results   : {results_dir}\n")

    per_type_scores: dict[str, list[float]] = {}
    g_start = time.time()

    for qt in run_types:
        samples = by_type.get(qt, [])
        if qt in ROUTE_PLANNING_TYPES and route_split is not None:
            samples = [s for s in samples if s.id in selected_ids]
        if excluded_ids:
            samples = [s for s in samples if s.id not in excluded_ids]
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
                    result = None
                    try:
                        with _silence():                   # hide SAM3/DA3 chatter
                            result = orch.run(sample)      # summarizer -> answer
                        pred = result.predicted_answer
                    except Exception:
                        traceback.print_exc()
                        pred = ""
                    summary_text = ""
                    if result is not None:
                        summary_text = str(getattr(result, "final_context", "") or "")
                    stages = collect_stages(tools)
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
                    if summary_text:
                        print(summary_text)
                    print_stages(stages)
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
