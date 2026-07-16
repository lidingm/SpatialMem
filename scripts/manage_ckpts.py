"""CLI for managing SpatialMem ckpts: list / inspect / delete / clone.

Usage:
    python -m scripts.manage_ckpts list
    python -m scripts.manage_ckpts inspect <name>
    python -m scripts.manage_ckpts delete <name>
    python -m scripts.manage_ckpts clone --src <src> --dst <dst>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agent.checkpoint import (
    ckpt_paths, clone_ckpt, clone_from_base, delete_ckpt, list_ckpts,
)


def cmd_list(args):
    ckpts = list_ckpts()
    if not ckpts:
        print("(no ckpts under ckpts/)")
        return
    print(f"{'NAME':30s}  {'CREATED':20s}  {'BASE':20s}  {'SKILLs':22s}  {'MEMORY':30s}  NOTES")
    for m in ckpts:
        import time
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("created_at", 0)))
        base = m.get("base_ckpt") or "-"
        st = m.get("_stats", {})
        skills = f"{st.get('n_seeded_skills',0)}s+{st.get('n_evolved_skills',0)}e+{st.get('n_pending_skills',0)}p"
        mem = f"{st.get('object_priors',0)}o+{st.get('object_candidates',0)}c objs; " \
              f"{st.get('unsolved_cases',0)} unsolved"
        print(f"{m['name']:30s}  {created:20s}  {base:20s}  {skills:22s}  {mem:30s}  {m.get('notes','')}")


def cmd_inspect(args):
    from agent.checkpoint import BASE_MARKER
    skills_dir, memory_dir = ckpt_paths(args.name)
    print(f"=== ckpt: {args.name} ===")
    print(f"  skills dir: {skills_dir}")
    print(f"  memory dir: {memory_dir}")

    if args.name != BASE_MARKER:
        meta_path = skills_dir.parent / "ckpt_meta.json"
        if meta_path.exists():
            print(f"\n  meta: {meta_path.read_text(encoding='utf-8')}")

    from agent.skill_lib import SkillLib
    sl = SkillLib(root=skills_dir, pending_dir=skills_dir / "pending")
    print(f"\nSKILLs ({len(sl.load_all())} main + "
          f"{sum(1 for d in sl.pending_dir.iterdir() if d.is_dir())} pending):")
    for s in sl.load_all():
        fm = s.frontmatter
        pit_chars = len(s.sections.get("Known Pitfalls", "").strip())
        print(f"  {s.name:34s}  seeded={fm.get('seeded'):>1}  v={fm.get('version')}  "
              f"calls={fm.get('total_calls')}  succ={fm.get('success_count')}  "
              f"rate={fm.get('success_rate')}  pitfalls={pit_chars}chars")

    from agent.memory import Memory
    m = Memory(root=memory_dir)
    obj_p = m._read_json("object_size_priors.json")
    sc_p = m._read_json("scene_scale_priors.json")
    print(f"\nMemory:")
    print(f"  object priors: {len(obj_p.get('objects', {}))} main + "
          f"{len(obj_p.get('candidates', {}))} candidates")
    print(f"  scene priors:  {len(sc_p.get('scenes', {}))} main + "
          f"{len(sc_p.get('candidates', {}))} candidates")
    uc = memory_dir / "unsolved_cases.jsonl"
    ucn = sum(1 for _ in uc.open(encoding="utf-8")) if uc.exists() else 0
    print(f"  unsolved_cases: {ucn} lines")
    ct_dir = memory_dir / "category_trajectories"
    if ct_dir.is_dir():
        for j in sorted(ct_dir.glob("*.jsonl")):
            n = sum(1 for _ in j.open(encoding="utf-8"))
            marker = j.parent / f"{j.stem}.bootstrapped"
            print(f"  category_trajectories/{j.name}: {n} lines"
                  f"{' [bootstrapped]' if marker.exists() else ''}")


def cmd_delete(args):
    delete_ckpt(args.name)
    print(f"deleted ckpt {args.name!r}")


def cmd_clone(args):
    target = clone_ckpt(args.src, args.dst, notes=args.notes)
    print(f"cloned {args.src!r} → {args.dst!r} at {target}")


def cmd_from_base(args):
    target = clone_from_base(args.dst, notes=args.notes)
    print(f"cloned base → {args.dst!r} at {target}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_list = sp.add_parser("list")
    sp_list.set_defaults(func=cmd_list)

    sp_ins = sp.add_parser("inspect")
    sp_ins.add_argument("name")
    sp_ins.set_defaults(func=cmd_inspect)

    sp_del = sp.add_parser("delete")
    sp_del.add_argument("name")
    sp_del.set_defaults(func=cmd_delete)

    sp_cln = sp.add_parser("clone")
    sp_cln.add_argument("--src", required=True, help="source ckpt name (or 'base')")
    sp_cln.add_argument("--dst", required=True, help="destination ckpt name")
    sp_cln.add_argument("--notes", default="")
    sp_cln.set_defaults(func=cmd_clone)

    sp_fb = sp.add_parser("from_base")
    sp_fb.add_argument("dst", help="destination ckpt name")
    sp_fb.add_argument("--notes", default="")
    sp_fb.set_defaults(func=cmd_from_base)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
