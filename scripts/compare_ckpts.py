"""Compare multiple eval result files side-by-side.

Common uses:
  - baseline vs method (on same 4K test)
  - base ckpt vs v1 vs v2 (evolution over training)
  - across driver models (gpt-4o-mini vs qwen3-vl-30b-a3b-instruct)

The comparison is only meaningful if all input files were run on the SAME
sample IDs. This script auto-restricts to the intersection of sample IDs
across all inputs and warns loudly if there's a mismatch.

Usage:
    python -m scripts.compare_ckpts \\
        --labels base v1 v2 \\
        --files results/eval_base.jsonl results/eval_v1.jsonl results/eval_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def load(path: Path) -> dict[str, dict]:
    """Return {sample_id: row}."""
    out = {}
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            out[str(r["sample_id"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True,
                    help="short label for each file (used as column header)")
    ap.add_argument("--files", nargs="+", required=True,
                    help="paths to eval .jsonl files, same order as --labels")
    ap.add_argument("--sort-by", default="delta", choices=("delta", "task_type"),
                    help="row sort order in the per-type table")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    if len(args.labels) != len(args.files):
        raise SystemExit(f"# labels ({len(args.labels)}) != # files ({len(args.files)})")

    results = {label: load(Path(f)) for label, f in zip(args.labels, args.files)}
    print(f"Loaded {len(results)} eval files:")
    for label, res in results.items():
        print(f"  {label:20s}  {len(res):5d} samples  ({args.files[list(results).index(label)]})")

    # Intersect sample IDs
    id_sets = [set(res.keys()) for res in results.values()]
    shared = set.intersection(*id_sets)
    total_any = len(set.union(*id_sets))
    print(f"\nShared sample IDs across all files: {len(shared)}/{total_any}")
    if len(shared) < total_any:
        print(f"  ⚠ dropping {total_any - len(shared)} non-shared samples for fair comparison")

    # Overall accuracy per label
    print(f"\n=== Overall accuracy on {len(shared)} shared samples ===")
    if args.markdown:
        print(f"| ckpt                | acc      | correct   |")
        print(f"|:--------------------|---------:|:----------|")
    for label, res in results.items():
        correct = sum(1 for sid in shared if res[sid].get("success"))
        if args.markdown:
            print(f"| {label:19s} | {correct/len(shared):>8.4f} | {correct}/{len(shared)} |")
        else:
            print(f"  {label:20s}  {correct/len(shared):.4f}  ({correct}/{len(shared)})")

    # Per task_type
    per_type = defaultdict(lambda: {label: {"n": 0, "c": 0} for label in args.labels})
    for sid in shared:
        for label, res in results.items():
            r = res[sid]
            qt = r["task_type"]
            per_type[qt][label]["n"] += 1
            per_type[qt][label]["c"] += int(r.get("success", False))

    # Compute delta (last label - first label) if applicable
    if len(args.labels) >= 2:
        first, last = args.labels[0], args.labels[-1]
        rows = []
        for qt, cells in per_type.items():
            first_acc = cells[first]["c"] / max(cells[first]["n"], 1)
            last_acc = cells[last]["c"] / max(cells[last]["n"], 1)
            rows.append((qt, cells, last_acc - first_acc))
        if args.sort_by == "delta":
            rows.sort(key=lambda x: x[2], reverse=True)
        else:
            rows.sort(key=lambda x: x[0])

        print(f"\n=== Per task_type ({first} → {last} delta) ===")
        header = "task_type".ljust(36) + " |  n  |"
        for label in args.labels:
            header += f"  {label:>10s} |"
        header += f"  Δ({last}−{first})"
        print(header)
        print("-" * len(header))
        for qt, cells, delta in rows:
            n = cells[args.labels[0]]["n"]
            row = f"{qt:36s} | {n:>3d} |"
            for label in args.labels:
                acc = cells[label]["c"] / max(cells[label]["n"], 1)
                row += f"  {acc:>10.4f} |"
            sign = "+" if delta > 0 else ""
            row += f"  {sign}{delta:.4f}"
            print(row)

    # SKILL usage (only if all files have chosen_skill — i.e. all method-eval)
    has_skill = all(any("chosen_skill" in res[sid] for sid in shared) for res in results.values())
    if has_skill:
        print(f"\n=== SKILL hit rate ({len(shared)} shared samples) ===")
        for label, res in results.items():
            hits = sum(1 for sid in shared if res[sid].get("chosen_skill"))
            print(f"  {label:20s}  {hits/len(shared):.4f}  ({hits}/{len(shared)})")


if __name__ == "__main__":
    main()
