#!/usr/bin/env python3
"""Direct Qwen3-VL baseline on VSI-Bench.

This script evaluates a deployed OpenAI-compatible VLM directly on the 32 RGB
frames and the question, without SpatialMem tools, SKILLs, Checker, or memory.
It writes one JSONL row per sample and supports resume by skipping completed ids.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_vsibench import load_test_samples, score_sample  # noqa: E402

DEFAULT_TEST_JSONL = "/home/zhouruofan/datasets/VSI-Bench/test.jsonl"
DEFAULT_IMAGES_ROOT = "/home/zhouruofan/datasets/VSI-Bench/images"
DEFAULT_OUT_SUBDIR = "qwen_base_official_style"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:18000/v1"
DEFAULT_LLM_API_KEY = "EMPTY"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_SEED = 3407
DEFAULT_MAX_TOKENS = 64


def _encode_image(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")

def build_messages(sample) -> list[dict[str, Any]]:
    question = str(sample.raw.get("question_raw") or sample.question).strip()
    options = sample.raw.get("options") or []
    if sample.answer_format == "select":
        prompt = f"Question: {question}\n"
        if options:
            prompt += "Options:\n" + "\n".join(str(o) for o in options) + "\n"
        prompt += "Answer with the option letter only."
    else:
        prompt = f"Question: {question}\nPlease answer concisely with short words or phrases when possible."

    content: list[dict[str, Any]] = []
    for fp in sample.image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(fp)}"},
        })
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

def call_chat(base_url: str, api_key: str, model: str, messages: list[dict[str, Any]],
              temperature: float, top_p: float, top_k: int, repetition_penalty: float,
              presence_penalty: float, seed: int, max_tokens: int, timeout: int, retries: int) -> str:
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if "qwen" in model.lower():
        payload["enable_thinking"] = False

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")[:300]
            last_error = f"HTTP {e.code}: {err_text}"
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error) from e
        except Exception as e:  # noqa: BLE001
            last_error = repr(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
    raise RuntimeError(f"LLM request failed after {retries} retries: {last_error}")


def load_done(path: Path) -> dict[str, float]:
    done: dict[str, float] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = str(r.get("sample_id", ""))
            if sid:
                done[sid] = float(r.get("score", 0.0))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", default=DEFAULT_TEST_JSONL)
    ap.add_argument("--images_root", default=DEFAULT_IMAGES_ROOT)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--max_per_type", type=int, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT_SUBDIR)
    ap.add_argument("--out_root", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--llm_base_url", default=DEFAULT_LLM_BASE_URL)
    ap.add_argument("--llm_api_key", default=DEFAULT_LLM_API_KEY)
    ap.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    ap.add_argument("--presence_penalty", type=float, default=DEFAULT_PRESENCE_PENALTY)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    by_type = load_test_samples(args.test_jsonl, Path(args.images_root))
    type_aliases = {
        "absolute_distance": ["object_abs_distance"],
        "relative_distance": ["object_rel_distance"],
        "relative_direction": [
            "object_rel_direction_easy",
            "object_rel_direction_medium",
            "object_rel_direction_hard",
        ],
        "appearance_order": ["obj_appearance_order"],
        "object_size": ["object_size_estimation"],
        "room_size": ["room_size_estimation"],
    }
    requested_types = args.types or sorted(by_type)
    run_types = []
    for t in requested_types:
        run_types.extend(type_aliases.get(t, [t]))

    print("Qwen baseline eval")
    print("Model    :", args.llm_model)
    print("Base URL :", args.llm_base_url)
    print("Out dir  :", out_dir)
    print("Types    :", run_types)

    all_scores: list[float] = []
    for qt in run_types:
        samples = list(by_type.get(qt, []))
        if args.max_per_type is not None:
            samples = samples[:args.max_per_type]
        if not samples:
            print(f"[{qt}] no samples, skipping")
            continue

        out_file = out_dir / f"{qt}.jsonl"
        done = load_done(out_file)
        scores = list(done.values())
        remaining = [s for s in samples if s.id not in done]
        metric = "ACC" if samples[0].answer_format == "select" else "MRA"
        print(f"\n{'='*60}\n[{qt}] total={len(samples)} done={len(done)} remaining={len(remaining)} metric={metric}\n{'='*60}")

        with out_file.open("a", encoding="utf-8") as f:
            for i, sample in enumerate(remaining, start=len(done) + 1):
                t0 = time.time()
                print(f"\n# [{qt}] {i}/{len(samples)} id={sample.id} Q: {sample.question.splitlines()[0][:90]}")
                pred = ""
                error = None
                try:
                    messages = build_messages(sample)
                    pred = call_chat(
                        args.llm_base_url, args.llm_api_key, args.llm_model, messages,
                        args.temperature, args.top_p, args.top_k, args.repetition_penalty,
                        args.presence_penalty, args.seed, args.max_tokens, args.timeout, args.retries,
                    )
                except Exception:  # noqa: BLE001
                    error = traceback.format_exc()
                    print(error)
                options = sample.raw.get("options")
                score = score_sample(pred, sample.gt_answer, sample.answer_format, options)
                scores.append(score)
                row = {
                    "sample_id": sample.id,
                    "task_type": sample.task_type,
                    "answer_format": sample.answer_format,
                    "gt_answer": sample.gt_answer,
                    "predicted_answer": pred,
                    "score": score,
                    "error": error,
                    "duration_seconds": round(time.time() - t0, 3),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f">>> {metric}={score:.3f} pred={pred!r} gt={sample.gt_answer!r} time={row['duration_seconds']:.1f}s")

        avg = sum(scores) / len(scores) if scores else 0.0
        all_scores.extend(scores)
        print(f"\n[{qt}] avg {metric}={avg:.3f} | n={len(scores)}")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    summary = {"overall_avg": overall, "n": len(all_scores), "types": run_types}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nOverall avg:", f"{overall:.3f}", "n=", len(all_scores))


if __name__ == "__main__":
    main()
