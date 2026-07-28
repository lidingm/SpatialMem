#!/usr/bin/env python3
"""
Run the evaluation-only pipeline after the route_planning94 split already exists.

Flow:
1. Evaluate all VSI-Bench task types except route_planning using local Qwen3-VL-8B.
2. Evaluate only the route_planning holdout split from the existing 94-sample split using local Qwen3-VL-8B.

Prerequisite:
- `results/train_route_planning94_v1/route_planning_split.json` must already exist.

Usage:
  python run_route94_then_eval_v2.py --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_NAME = "v2"
TRAIN_RESULTS_SUBDIR = "train_route_planning94_v1"
EVAL_RESULTS_SUBDIR = "eval_vsibench_v2_after_route94"
SPLIT_JSON = PROJECT_ROOT / "results" / TRAIN_RESULTS_SUBDIR / "route_planning_split.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "results" / EVAL_RESULTS_SUBDIR
FULL_EVAL_LOG = EVAL_RESULTS_DIR / "eval_non_route.log"
ROUTE_HOLDOUT_LOG = EVAL_RESULTS_DIR / "eval_route_holdout.log"
EVAL_LLM_BASE_URL = "http://127.0.0.1:18000/v1"
EVAL_LLM_API_KEY = "EMPTY"
EVAL_LLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0", help="CUDA device string, e.g. cuda:0")
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Pass --verbose through to eval scripts",
    )
    return p.parse_args()


def run_step(cmd: list[str], name: str, log_path: Path | None = None) -> None:
    print(f"\n{'=' * 80}")
    print(f"[PIPELINE] {name}")
    print("[CMD]", " ".join(cmd))
    if log_path is not None:
        print(f"[LOG] {log_path}")
    print(f"{'=' * 80}\n")

    kwargs = {"cwd": PROJECT_ROOT}
    log_f: TextIO | None = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("a", encoding="utf-8")
            log_f.write(f"\n{'=' * 80}\n")
            log_f.write(f"[PIPELINE] {name}\n")
            log_f.write(f"[CMD] {' '.join(cmd)}\n")
            log_f.write(f"{'=' * 80}\n")
            log_f.flush()
            kwargs["stdout"] = log_f
            kwargs["stderr"] = subprocess.STDOUT
        result = subprocess.run(cmd, **kwargs)
    finally:
        if log_f is not None:
            log_f.flush()
            log_f.close()

    if result.returncode != 0:
        raise SystemExit(f"[PIPELINE] step failed: {name} (exit={result.returncode})")


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not SPLIT_JSON.exists():
        raise SystemExit(
            f"[PIPELINE] missing split file: {SPLIT_JSON}. "
            "Create it first with the earlier route_planning94 training run."
        )

    eval_common = [
        py,
        "-u",
        str(PROJECT_ROOT / "eval_vsibench.py"),
        "--device",
        args.device,
        "--ckpt",
        CKPT_NAME,
        "--out",
        EVAL_RESULTS_SUBDIR,
        "--llm_base_url",
        EVAL_LLM_BASE_URL,
        "--llm_api_key",
        EVAL_LLM_API_KEY,
        "--llm_model",
        EVAL_LLM_MODEL,
    ]
    if args.verbose:
        eval_common.append("--verbose")

    run_step(eval_common, "Evaluate VSI-Bench except route_planning", FULL_EVAL_LOG)

    route_holdout_cmd = eval_common + [
        "--types",
        "route_planning",
        "--route_split_json",
        str(SPLIT_JSON),
        "--route_partition",
        "holdout",
    ]
    run_step(route_holdout_cmd, "Evaluate route_planning holdout", ROUTE_HOLDOUT_LOG)

    print(f"\n{'=' * 80}")
    print("[PIPELINE] all evaluation steps finished")
    print(f"Route split   : {SPLIT_JSON}")
    print(f"Eval results  : {PROJECT_ROOT / 'results' / EVAL_RESULTS_SUBDIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()