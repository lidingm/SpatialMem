"""Split VSIBench into a training subset and test subset.

Deterministic (seed=42) shuffle → first N = train, rest = test. Written to
`splits/train_ids.txt` and `splits/test_ids.txt` (one sample id per line).

Usage:
    python -m scripts.split_vsi --train-size 1000 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
from agent.data_loader import VSIBenchDataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="splits")
    args = ap.parse_args()

    loader = VSIBenchDataLoader()
    df = loader.df
    all_ids = df["id"].tolist()
    print(f"Total VSIBench samples: {len(all_ids)}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(all_ids))
    shuffled = [all_ids[i] for i in order]

    train_ids = shuffled[: args.train_size]
    test_ids = shuffled[args.train_size:]
    print(f"Train: {len(train_ids)}  Test: {len(test_ids)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_ids.txt").write_text("\n".join(str(i) for i in train_ids) + "\n")
    (out_dir / "test_ids.txt").write_text("\n".join(str(i) for i in test_ids) + "\n")

    # Print per-type breakdown
    for name, ids in (("train", train_ids), ("test", test_ids)):
        sub = df[df["id"].isin(ids)]
        counts = sub["question_type"].value_counts().to_dict()
        print(f"\n{name.upper()} breakdown:")
        for qt, c in sorted(counts.items()):
            print(f"  {qt:36s} {c}")


if __name__ == "__main__":
    main()
