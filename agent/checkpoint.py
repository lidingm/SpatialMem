"""Checkpoint (ckpt) mechanism for SpatialMem's SKILL library + Memory state.

Rationale
---------
Every agent run mutates the SKILL library and Memory. If we let training
write directly to the project's `skills/` and `memory/` directories, then:
  - Every run pollutes the pristine seeded state.
  - Two training runs can't be compared (each starts from a different state).
  - Eval on a specific "trained" state is impossible.

Solution: treat the SKILL library + Memory as a **checkpoint** (like model
weights). The project's `skills/` and `memory/` are the **base** — read-only
seeded starting point. Training loads a source ckpt (or base), evolves state,
and saves to a destination ckpt. Eval loads a specific ckpt in read-only mode.

Ckpt layout
-----------
    ckpts/{name}/
    ├── skills/              # snapshot of SKILL library
    │   ├── {seed_or_evolved_skill}/
    │   │   ├── SKILL.md
    │   │   ├── execute.py
    │   │   └── trajectories.jsonl
    │   └── pending/
    ├── memory/              # snapshot of Memory
    │   ├── object_size_priors.json
    │   ├── scene_scale_priors.json
    │   ├── unsolved_cases.jsonl
    │   └── category_trajectories/
    └── ckpt_meta.json       # {name, created_at, base_ckpt, samples_seen, notes, ...}

The project's base `skills/` and `memory/` are NEVER mutated by training scripts
(only by the human editing seed SKILLs). Every training run must specify a
`--save-ckpt` destination; every eval must specify (or accept default of base
via `--ckpt base`).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

from agent.config import CKPTS_DIR, MEMORY_DIR, SKILLS_DIR


BASE_MARKER = "base"  # reserved name meaning "the project's seeded base state"


def _ckpt_dir(name: str) -> Path:
    return CKPTS_DIR / name


def ckpt_paths(name: str) -> tuple[Path, Path]:
    """Return (skills_root, memory_root) for a ckpt (or the project base).

    Use the return values as `SkillLib(root=skills_root)` and `Memory(root=memory_root)`.
    """
    if name == BASE_MARKER or name is None:
        return SKILLS_DIR, MEMORY_DIR
    d = _ckpt_dir(name)
    if not d.is_dir():
        raise FileNotFoundError(f"ckpt {name!r} not found at {d}")
    return d / "skills", d / "memory"


def ckpt_exists(name: str) -> bool:
    return name == BASE_MARKER or _ckpt_dir(name).is_dir()


def clone_from_base(dst: str, notes: str | None = None) -> Path:
    """Create a fresh ckpt initialized with the project's base seeded state."""
    target = _ckpt_dir(dst)
    if target.exists():
        raise FileExistsError(f"ckpt {dst!r} already exists at {target}")
    target.mkdir(parents=True)
    shutil.copytree(SKILLS_DIR, target / "skills")
    shutil.copytree(MEMORY_DIR, target / "memory")
    _write_meta(target, name=dst, base_ckpt=BASE_MARKER, notes=notes)
    return target


def clone_ckpt(src: str, dst: str, notes: str | None = None) -> Path:
    """Copy an existing ckpt to a new name (for continued training)."""
    if src == BASE_MARKER:
        return clone_from_base(dst, notes=notes)
    src_dir = _ckpt_dir(src)
    dst_dir = _ckpt_dir(dst)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source ckpt {src!r} not found at {src_dir}")
    if dst_dir.exists():
        raise FileExistsError(f"destination ckpt {dst!r} already exists at {dst_dir}")
    shutil.copytree(src_dir, dst_dir)
    # Rewrite meta to reflect new derivation
    _write_meta(dst_dir, name=dst, base_ckpt=src, notes=notes)
    return dst_dir


def save_current_as_ckpt(dst: str, notes: str | None = None) -> Path:
    """DEPRECATED / DEBUG-ONLY. Snapshot the project's current `skills/`+`memory/`
    (which SHOULD be the pristine base) to a ckpt. Prefer `clone_from_base` for
    fresh starts and `clone_ckpt` for continuations.
    """
    return clone_from_base(dst, notes=notes)  # base is at project skills/memory


def list_ckpts() -> list[dict]:
    """Return metadata for every ckpt under `ckpts/`, sorted by created_at DESC."""
    if not CKPTS_DIR.exists():
        return []
    out = []
    for d in CKPTS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "ckpt_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"name": d.name, "created_at": 0, "notes": "(corrupt meta)"}
        # Refresh live stats
        meta["_stats"] = _live_stats(d)
        out.append(meta)
    out.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return out


def delete_ckpt(name: str) -> None:
    if name == BASE_MARKER:
        raise ValueError("cannot delete the base ckpt (it's the project's seeded state)")
    d = _ckpt_dir(name)
    if d.exists():
        shutil.rmtree(d)


def update_meta(name: str, **fields) -> None:
    """Merge `fields` into ckpt_meta.json. Common uses: samples_seen, notes."""
    if name == BASE_MARKER:
        return
    d = _ckpt_dir(name)
    meta_path = d / "ckpt_meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    meta.update(fields)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Internals ────────────────────────────────────────────────────────

def _write_meta(target: Path, name: str, base_ckpt: str | None,
                notes: str | None = None) -> None:
    (target / "ckpt_meta.json").write_text(json.dumps({
        "name": name,
        "created_at": int(time.time()),
        "base_ckpt": base_ckpt,
        "notes": notes or "",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _live_stats(ckpt_dir: Path) -> dict:
    """Peek at the ckpt's state for at-a-glance inventory."""
    skills_dir = ckpt_dir / "skills"
    memory_dir = ckpt_dir / "memory"
    stats = {"n_seeded_skills": 0, "n_evolved_skills": 0, "n_pending_skills": 0}
    if skills_dir.exists():
        import yaml
        for d in skills_dir.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if d.name == "pending":
                stats["n_pending_skills"] = sum(
                    1 for p in d.iterdir() if p.is_dir())
                continue
            md = d / "SKILL.md"
            if not md.exists():
                continue
            try:
                parts = md.read_text(encoding="utf-8").split("---", 2)
                fm = yaml.safe_load(parts[1]) if len(parts) >= 3 else {}
                if fm.get("seeded", True):
                    stats["n_seeded_skills"] += 1
                else:
                    stats["n_evolved_skills"] += 1
            except Exception:
                pass
    if memory_dir.exists():
        try:
            obj = json.loads((memory_dir / "object_size_priors.json").read_text(encoding="utf-8"))
            stats["object_priors"] = len(obj.get("objects", {}))
            stats["object_candidates"] = len(obj.get("candidates", {}))
        except Exception:
            stats["object_priors"] = 0
            stats["object_candidates"] = 0
        try:
            sc = json.loads((memory_dir / "scene_scale_priors.json").read_text(encoding="utf-8"))
            stats["scene_priors"] = len(sc.get("scenes", {}))
            stats["scene_candidates"] = len(sc.get("candidates", {}))
        except Exception:
            stats["scene_priors"] = 0
            stats["scene_candidates"] = 0
        uc_path = memory_dir / "unsolved_cases.jsonl"
        stats["unsolved_cases"] = sum(1 for _ in uc_path.open(encoding="utf-8")) if uc_path.exists() else 0
    return stats
