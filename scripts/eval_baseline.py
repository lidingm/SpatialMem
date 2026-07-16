"""Baseline evaluation: direct VLM query, no SpatialMem framework.

For each VSIBench sample: send question + a few sampled frames to the VLM
with a system prompt about answer format, parse the answer, compare to GT.

Runs LLM calls in parallel via ThreadPoolExecutor.

Usage:
    export LLM_BASE_URL=https://models-proxy.stepfun-inc.com/v1
    export LLM_API_KEY=$MODEL_PROXY_TOKEN
    python -m scripts.eval_baseline \\
        --model qwen3-vl-30b-a3b-instruct \\
        --split test \\
        --workers 20 \\
        --max-samples 4130 \\
        --out results/baseline_qwen3vl30b.jsonl
"""

from __future__ import annotations

import argparse
import base64
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

import numpy as np
from agent.config import RESULTS_DIR
from agent.data_loader import VSIBenchDataLoader, SpatialSample
from agent.llm_client import LLMClient
from agent.train_loop import evaluate_answer


BASELINE_SYSTEM_PROMPT = """You are a spatial reasoning expert answering questions about indoor scenes shown in a video.

Look at the frames carefully, then answer. Answer format rules (STRICT):
- Numeric fill: ONLY the number. E.g. "0.6", "3", "16". No units, no words.
- Multiple choice: ONLY one letter (A/B/C/D). Nothing else. No period.

Output JSON:
{
    "answer": "the answer value ONLY, in the required format",
    "reasoning": "1-2 sentences on how you decided"
}
"""


def _encode_image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _select_frames(paths: list[str], k: int) -> list[int]:
    if not paths:
        return []
    if len(paths) <= k:
        return list(range(len(paths)))
    return list(np.linspace(0, len(paths) - 1, k).astype(int))


def build_user_message(sample: SpatialSample, k_frames: int) -> list[dict]:
    content = [{"type": "text",
                "text": f"Question: {sample.question}\n\n"
                        f"Answer format: {sample.answer_format} "
                        f"({'single letter A/B/C/D' if sample.answer_format == 'select' else 'single number'})"}]
    for idx in _select_frames(sample.image_paths, k_frames):
        try:
            b64 = _encode_image_b64(sample.image_paths[idx])
            suffix = Path(sample.image_paths[idx]).suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        except Exception:
            pass
    return content


def eval_sample(sample: SpatialSample, llm: LLMClient, k_frames: int) -> dict:
    t0 = time.time()
    try:
        resp = llm.chat_json([
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(sample, k_frames)},
        ], max_tokens=512)
        predicted = str(resp.get("answer", "")).strip()
        reasoning = resp.get("reasoning", "")
        error = None
    except Exception as e:
        predicted, reasoning, error = "", "", str(e)

    success = evaluate_answer(
        predicted, sample.gt_answer, sample.answer_format, sample.task_type)
    return {
        "sample_id": sample.id,
        "task_type": sample.task_type,
        "task_category": sample.task_category,
        "answer_format": sample.answer_format,
        "question": sample.question[:200],
        "gt_answer": sample.gt_answer,
        "predicted_answer": predicted,
        "reasoning": reasoning,
        "success": bool(success),
        "duration_seconds": round(time.time() - t0, 2),
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-30b-a3b-instruct")
    ap.add_argument("--split", default="test", choices=("test", "train"),
                    help="which id list under splits/ to use")
    ap.add_argument("--split-file", default=None,
                    help="direct path to id list; overrides --split")
    ap.add_argument("--k-frames", type=int, default=4,
                    help="how many frames to send per sample")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    ids_path = Path(args.split_file) if args.split_file else _HERE / "splits" / f"{args.split}_ids.txt"
    ids = [int(x.strip()) for x in ids_path.read_text().split() if x.strip()]
    if args.max_samples:
        ids = ids[: args.max_samples]

    loader = VSIBenchDataLoader()
    samples = loader.load_ids(ids)
    print(f"Loaded {len(samples)} samples from {ids_path}")

    out_path = Path(args.out or RESULTS_DIR / f"baseline_{args.split}_{args.model}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(
        base_url=os.environ.get("LLM_BASE_URL", "https://models-proxy.stepfun-inc.com/v1"),
        api_key=os.environ.get("LLM_API_KEY", os.environ.get("MODEL_PROXY_TOKEN", "")),
        model=args.model,
        temperature=args.temperature,
    )

    lock = Lock()
    n_done = [0]
    n_correct = [0]
    t_start = time.time()

    def _worker(sample):
        r = eval_sample(sample, llm, args.k_frames)
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

    # Truncate previous partial results
    if out_path.exists():
        out_path.unlink()

    print(f"Running baseline eval with {args.workers} workers → {out_path}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(_worker, s) for s in samples]))

    # Final summary
    per_type = {}
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            per_type.setdefault(r["task_type"], {"n": 0, "correct": 0}).__setitem__("n", per_type[r["task_type"]]["n"] + 1)
            if r["success"]:
                per_type[r["task_type"]]["correct"] += 1

    print(f"\n{'='*60}\nFINAL SUMMARY  ({args.model}, {args.split})")
    print(f"  Overall accuracy: {n_correct[0]}/{n_done[0]} = {n_correct[0]/max(n_done[0],1):.3f}")
    print(f"  Per-type:")
    for qt in sorted(per_type):
        d = per_type[qt]
        print(f"    {qt:36s}  {d['correct']:4d}/{d['n']:4d} = {d['correct']/max(d['n'],1):.3f}")


if __name__ == "__main__":
    main()
