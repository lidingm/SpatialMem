"""Method (SpatialMem framework) evaluation and training driver.

Two modes, both driven by explicit ckpts to keep the project's base
`skills/` and `memory/` pristine.

  --mode train : evolve SKILL library + Memory on N samples.
                 REQUIRED: --save-ckpt DST  (destination ckpt to write to)
                 OPTIONAL: --ckpt SRC       (source ckpt to init from;
                                             default = 'base' project state)
                 Base skills/memory are NEVER mutated.

  --mode eval  : run SpatialMem with evolve=False on N samples.
                 OPTIONAL: --ckpt SRC       (state to eval; default = 'base')
                 SKILL library + Memory are read-only.

Uses mocked DA3/SAM3 (same mocks as scripts/debug_llm.py) — GPU-free.
Because tools are mocked, absolute accuracy is not meaningful, but the
framework's decision-making, SKILL selection, and evolution mechanics are
faithfully exercised end-to-end.

Usage:
    # Fresh training from base
    python -m scripts.train_and_eval --mode train --split train \\
        --model qwen3-vl-30b-a3b-instruct --max-samples 1000 \\
        --save-ckpt run_v1

    # Continue training from a ckpt
    python -m scripts.train_and_eval --mode train --split train \\
        --ckpt run_v1 --save-ckpt run_v2 --max-samples 1000

    # Eval a specific ckpt
    python -m scripts.train_and_eval --mode eval --split test \\
        --ckpt run_v1 --max-samples 1000 --workers 20

    # Eval base (untrained seeds)
    python -m scripts.train_and_eval --mode eval --split test \\
        --ckpt base --max-samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Import mocks from debug_llm.py so we don't duplicate
from scripts.debug_llm import FakeDA3, FakeSAM3, _monkey_patch_point_cloud, _monkey_patch_code_tools

from agent.checkpoint import (
    BASE_MARKER, ckpt_exists, ckpt_paths, clone_ckpt, clone_from_base, update_meta,
)
from agent.config import ensure_dirs, RESULTS_DIR
from agent.data_loader import VSIBenchDataLoader
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


def build_orchestrator(model: str, ckpt: str, temperature: float = 0.3, verbose: bool = False):
    """Build a full agent stack pointing at a specific ckpt.

    Args:
        model: LLM model name
        ckpt: ckpt name to load SKILL/Memory from (use 'base' for pristine seeds)
    """
    skills_root, memory_root = ckpt_paths(ckpt)
    llm = LLMClient(
        base_url=os.environ.get("LLM_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"),
        api_key=os.environ.get("LLM_API_KEY", os.environ.get("MODEL_PROXY_TOKEN", "")),
        model=model,
        temperature=temperature,
    )
    memory = Memory(root=memory_root)
    skill_lib = SkillLib(root=skills_root, pending_dir=skills_root / "pending")
    tools = ToolRegistry(da3_tool=FakeDA3(), sam3_tool=FakeSAM3())
    planner = Planner(llm)
    reasoner = Reasoner(llm)
    reflector = Reflector(llm)
    orch = Orchestrator(tools, memory, skill_lib, planner, reasoner, reflector,
                        verbose=verbose)
    return orch, llm, memory, skill_lib, reflector


def snapshot_state(memory: Memory, skill_lib: SkillLib) -> dict:
    return {
        "n_seeded_skills": sum(1 for s in skill_lib.load_all() if s.frontmatter.get("seeded")),
        "n_evolved_skills": sum(1 for s in skill_lib.load_all() if not s.frontmatter.get("seeded")),
        "n_pending_skills": sum(1 for d in skill_lib.pending_dir.iterdir() if d.is_dir()),
        "size_priors_objects": len(json.load(open(memory.root / "object_size_priors.json"))["objects"]),
        "size_priors_candidates": len(json.load(open(memory.root / "object_size_priors.json"))["candidates"]),
        "scene_priors_scenes": len(json.load(open(memory.root / "scene_scale_priors.json"))["scenes"]),
        "scene_priors_candidates": len(json.load(open(memory.root / "scene_scale_priors.json"))["candidates"]),
        "unsolved_count": sum(1 for _ in open(memory.root / "unsolved_cases.jsonl")),
    }


def run_train(args):
    """Parallel training run. Reads init state from --ckpt (default: base),
    clones to --save-ckpt, then mutates ONLY that destination ckpt."""
    ensure_dirs()
    _monkey_patch_point_cloud()
    _monkey_patch_code_tools()

    # Materialize destination ckpt: clone from source (or base) into ckpts/<save_ckpt>/.
    src = args.ckpt or BASE_MARKER
    if not ckpt_exists(src):
        raise SystemExit(f"source ckpt {src!r} not found. Run `python -m scripts.manage_ckpts list` to inspect.")
    if not args.save_ckpt:
        raise SystemExit("--save-ckpt is REQUIRED in train mode "
                         "(training must not mutate base or overwrite existing ckpts silently)")
    if ckpt_exists(args.save_ckpt) and args.save_ckpt != BASE_MARKER:
        raise SystemExit(f"destination ckpt {args.save_ckpt!r} already exists. "
                         f"Pick a different name or delete first with "
                         f"`python -m scripts.manage_ckpts delete {args.save_ckpt}`.")

    print(f"[TRAIN] cloning ckpt {src!r} → {args.save_ckpt!r}")
    clone_ckpt(src, args.save_ckpt, notes=args.notes)

    # Build orchestrator etc. pointing at the destination ckpt (all mutations land here)
    orch, llm, memory, skill_lib, reflector = build_orchestrator(
        args.model, ckpt=args.save_ckpt, temperature=args.temperature, verbose=args.verbose)

    def build_orch_shared_state():
        # Each worker needs its own orchestrator (fresh tools context per sample)
        # but the same Memory/SkillLib instances so their locks/state coordinate.
        skills_root, memory_root = ckpt_paths(args.save_ckpt)
        llm_ = LLMClient(
            base_url=os.environ.get("LLM_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"),
            api_key=os.environ.get("LLM_API_KEY", os.environ.get("MODEL_PROXY_TOKEN", "")),
            model=args.model, temperature=args.temperature,
        )
        tools_ = ToolRegistry(da3_tool=FakeDA3(), sam3_tool=FakeSAM3())
        planner_ = Planner(llm_)
        reasoner_ = Reasoner(llm_)
        reflector_ = Reflector(llm_)
        orch_ = Orchestrator(tools_, memory, skill_lib, planner_, reasoner_, reflector_,
                             verbose=args.verbose)
        return orch_, llm_, reflector_

    ids_path = _HERE / "splits" / f"{args.split}_ids.txt"
    ids = [int(x.strip()) for x in ids_path.read_text().split() if x.strip()]
    if args.max_samples:
        ids = ids[: args.max_samples]
    loader = VSIBenchDataLoader()
    samples = loader.load_ids(ids)
    print(f"[TRAIN] {len(samples)} samples with {args.model}  workers={args.workers}  ckpt={args.save_ckpt}")
    print(f"[TRAIN] Initial state: {snapshot_state(memory, skill_lib)}")

    out_path = Path(args.out or RESULTS_DIR / f"train_{args.save_ckpt}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    log_lock = Lock()      # serialize results.jsonl append + counters
    n_done = [0]
    n_correct = [0]
    t_start = time.time()

    def _one(sample):
        orch, llm, reflector = build_orch_shared_state()
        try:
            result = orch.run(sample)
            result.success = evaluate_answer(
                result.predicted_answer, sample.gt_answer,
                sample.answer_format, sample.task_type)
        except Exception as e:
            with log_lock:
                print(f"[RUN FAIL {sample.id}] {e}")
            return None

        # Fine-grained locks inside SkillLib/Memory handle concurrent evolve
        # writes; LLM calls (distill, reconstruct, verify) happen without a
        # big global lock.
        try:
            upd = evolve_after_sample(
                result, sample, memory, skill_lib, reflector, orch, llm,
                evaluate_answer, verbose=False)
        except Exception as e:
            with log_lock:
                print(f"[EVOLVE FAIL {sample.id}] {e}")
            upd = {"path": None, "skill_updates": [], "pending_saved": None,
                   "bootstrapped": None, "memory_updates": []}

        entry = {
            "sample_id": sample.id, "task_type": result.task_type,
            "task_category": result.task_category,
            "chosen_skill": result.chosen_skill,
            "success": bool(result.success),
            "predicted_answer": result.predicted_answer,
            "gt_answer": result.gt_answer,
            "num_rounds": result.num_rounds,
            "duration_seconds": round(result.duration_seconds, 2),
            "evolve_path": upd.get("path"),
            "skill_updates": upd.get("skill_updates", []),
            "pending_saved": upd.get("pending_saved"),
            "bootstrapped": upd.get("bootstrapped"),
        }
        with log_lock:
            n_done[0] += 1
            n_correct[0] += int(result.success)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if n_done[0] % 20 == 0 or n_done[0] == len(samples):
                elapsed = time.time() - t_start
                rate = n_done[0] / max(elapsed, 1e-6)
                eta = (len(samples) - n_done[0]) / max(rate, 1e-6)
                state = snapshot_state(memory, skill_lib)
                print(f"  [{n_done[0]:4d}/{len(samples)}]  acc={n_correct[0]/n_done[0]:.3f}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min  {state}")
        return entry

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(_one, s) for s in samples]))
    skill_lib.promote_eligible_pending()

    final_state = snapshot_state(memory, skill_lib)
    update_meta(args.save_ckpt,
                samples_seen=n_done[0],
                training_accuracy=round(n_correct[0] / max(n_done[0], 1), 4),
                final_state=final_state,
                trained_with=args.model)
    print(f"\n[TRAIN] DONE. Final state: {final_state}")
    print(f"[TRAIN] Ckpt saved: ckpts/{args.save_ckpt}/")


def _eval_sample_worker(sample, orch_factory):
    """Each worker builds its own orchestrator (fresh tool ctx per call).
    SkillLib and Memory are read-mostly at eval time so shared instances are fine."""
    orch, llm, memory, skill_lib, reflector = orch_factory()
    t0 = time.time()
    try:
        result = orch.run(sample)
        success = evaluate_answer(
            result.predicted_answer, sample.gt_answer,
            sample.answer_format, sample.task_type)
        return {
            "sample_id": sample.id,
            "task_type": sample.task_type,
            "task_category": sample.task_category,
            "answer_format": sample.answer_format,
            "gt_answer": sample.gt_answer,
            "predicted_answer": result.predicted_answer,
            "chosen_skill": result.chosen_skill,
            "num_rounds": result.num_rounds,
            "success": bool(success),
            "duration_seconds": round(time.time() - t0, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "sample_id": sample.id, "task_type": sample.task_type,
            "task_category": sample.task_category,
            "answer_format": sample.answer_format,
            "gt_answer": sample.gt_answer,
            "predicted_answer": "", "chosen_skill": None,
            "num_rounds": 0, "success": False,
            "duration_seconds": round(time.time() - t0, 2),
            "error": str(e),
        }


def run_eval(args):
    """Parallel eval run — reads SKILL/Memory from --ckpt in read-only mode.
    The ckpt is NEVER mutated during eval (Path A/B/C/D and Bootstrap require
    training mode).
    """
    ensure_dirs()
    _monkey_patch_point_cloud()
    _monkey_patch_code_tools()

    ckpt = args.ckpt or BASE_MARKER
    if not ckpt_exists(ckpt):
        raise SystemExit(f"ckpt {ckpt!r} not found. Run `python -m scripts.manage_ckpts list`.")

    ids_path = _HERE / "splits" / f"{args.split}_ids.txt"
    ids = [int(x.strip()) for x in ids_path.read_text().split() if x.strip()]
    if args.max_samples:
        ids = ids[: args.max_samples]
    loader = VSIBenchDataLoader()
    samples = loader.load_ids(ids)
    print(f"[EVAL] {len(samples)} samples with {args.model}  workers={args.workers}  ckpt={ckpt}")

    out_path = Path(args.out or RESULTS_DIR / f"eval_{ckpt}_{args.model}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    def orch_factory():
        return build_orchestrator(args.model, ckpt=ckpt,
                                  temperature=args.temperature, verbose=False)

    lock = Lock()
    n_done = [0]
    n_correct = [0]
    t_start = time.time()

    def _wrapped(sample):
        r = _eval_sample_worker(sample, orch_factory)
        with lock:
            n_done[0] += 1
            n_correct[0] += int(r["success"])
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n_done[0] % 50 == 0 or n_done[0] == len(samples):
                elapsed = time.time() - t_start
                rate = n_done[0] / max(elapsed, 1e-6)
                eta = (len(samples) - n_done[0]) / max(rate, 1e-6)
                print(f"  [{n_done[0]:5d}/{len(samples):5d}]  "
                      f"acc={n_correct[0]/n_done[0]:.3f}  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min")
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(_wrapped, s) for s in samples]))

    # Final per-type summary
    per_type = {}
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = per_type.setdefault(r["task_type"], {"n": 0, "correct": 0, "skill_hit": 0})
            d["n"] += 1
            d["correct"] += int(r["success"])
            d["skill_hit"] += int(r.get("chosen_skill") is not None)

    print(f"\n{'='*60}\nFINAL SUMMARY  ({args.model}, {args.split}, method)")
    print(f"  Overall: {n_correct[0]}/{n_done[0]} = {n_correct[0]/max(n_done[0],1):.3f}")
    print(f"  Per-type:")
    for qt in sorted(per_type):
        d = per_type[qt]
        print(f"    {qt:36s}  correct={d['correct']:4d}/{d['n']:4d} = {d['correct']/max(d['n'],1):.3f}"
              f"   skill_hit={d['skill_hit']}/{d['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("train", "eval"))
    ap.add_argument("--split", default="test", choices=("test", "train"))
    ap.add_argument("--model", default="qwen3-vl-30b-a3b-instruct")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--verbose", action="store_true")
    # Checkpoint control
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Source ckpt name to load SKILL/Memory from. "
                         "In train mode: initial state to clone from. "
                         "In eval mode: state to evaluate. Default: 'base' (project seeded state).")
    ap.add_argument("--save-ckpt", type=str, default=None,
                    help="[TRAIN ONLY, REQUIRED] Destination ckpt name to save the evolved state.")
    ap.add_argument("--notes", type=str, default="",
                    help="Free-text notes to attach to the new ckpt's metadata.")
    args = ap.parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
