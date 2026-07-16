"""Analyze a single eval result JSONL (baseline or method).

Usage:
    python -m scripts.analyze_results results/eval_v1_qwen3vl30b-a3b-instruct.jsonl

Reports:
  - Overall accuracy
  - Per task_type accuracy
  - Per answer_format accuracy (fill / select / ...)
  - SKILL usage stats (only if the results contain chosen_skill fields — i.e., method eval)
  - Duration percentiles
  - Error rate
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def format_pct(num: int, den: int) -> str:
    if den == 0:
        return "  n/a"
    return f"{num/den:.3f} ({num}/{den})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="path to eval results .jsonl")
    ap.add_argument("--markdown", action="store_true", help="output as markdown tables")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"file not found: {p}")
    rows = load(p)
    if not rows:
        raise SystemExit(f"empty file: {p}")

    total = len(rows)
    correct = sum(1 for r in rows if r.get("success"))
    errored = sum(1 for r in rows if r.get("error"))

    # Detect if this is method-style (has chosen_skill) or baseline-style
    is_method = any("chosen_skill" in r for r in rows)

    # Per task_type
    per_type = defaultdict(lambda: {"n": 0, "c": 0, "skill_hit": 0})
    per_fmt = defaultdict(lambda: {"n": 0, "c": 0})
    skills_used = Counter()
    durations = []
    for r in rows:
        per_type[r["task_type"]]["n"] += 1
        per_type[r["task_type"]]["c"] += int(r.get("success", False))
        if is_method and r.get("chosen_skill"):
            per_type[r["task_type"]]["skill_hit"] += 1
            skills_used[r["chosen_skill"]] += 1
        per_fmt[r.get("answer_format", "?")]["n"] += 1
        per_fmt[r.get("answer_format", "?")]["c"] += int(r.get("success", False))
        if "duration_seconds" in r:
            durations.append(r["duration_seconds"])

    # Print report
    print(f"=== {p} ===")
    print(f"Total samples:  {total}")
    print(f"Overall:        {format_pct(correct, total)}  acc")
    print(f"Errored:        {errored}/{total} ({errored/total:.1%})" if total else "0")
    if durations:
        print(f"Duration (s):   min={min(durations):.1f}  "
              f"p50={statistics.median(durations):.1f}  "
              f"p90={statistics.quantiles(durations, n=10)[-1] if len(durations)>=10 else max(durations):.1f}  "
              f"max={max(durations):.1f}  "
              f"mean={statistics.mean(durations):.1f}")

    fmt = "| {qt:36s} | {n:>4d} | {rate:>8.3f} | {c:>3d}/{n:<3d} |" if args.markdown else \
          "  {qt:36s}  n={n:>4d}  acc={rate:.3f}  ({c}/{n})"
    if args.markdown:
        print("\n| task_type                             |    n |  acc     | correct   |")
        print("|---------------------------------------|-----:|---------:|:----------|")
    else:
        print("\nPer task_type:")
    for qt in sorted(per_type):
        d = per_type[qt]
        print(fmt.format(qt=qt, n=d["n"], rate=d["c"]/max(d["n"],1), c=d["c"]))

    print("\nPer answer_format:")
    for fk in sorted(per_fmt):
        d = per_fmt[fk]
        print(f"  {fk:8s}  n={d['n']:>4d}  acc={d['c']/max(d['n'],1):.3f}  ({d['c']}/{d['n']})")

    if is_method:
        skill_hit_total = sum(int(bool(r.get("chosen_skill"))) for r in rows)
        print(f"\nSKILL usage (method-eval only):")
        print(f"  overall skill_hit_rate: {skill_hit_total}/{total} = {skill_hit_total/total:.3f}")
        if skills_used:
            print(f"  chosen SKILL distribution:")
            for name, cnt in skills_used.most_common():
                print(f"    {name:36s}  {cnt}")
        print(f"\nPer task_type skill_hit:")
        for qt in sorted(per_type):
            d = per_type[qt]
            if d["n"] > 0:
                print(f"  {qt:36s}  {d['skill_hit']}/{d['n']} = {d['skill_hit']/d['n']:.3f}")


if __name__ == "__main__":
    main()
