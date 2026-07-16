"""Declarative memory for SpatialMem.

Stores physical-world priors (object size, scene scale) with Welford online
mean/std, and a running log of samples the agent has failed to solve even
after reflection. Layer3-style failure lessons are NOT stored here; they
live inside individual SKILL.md `Known Pitfalls` sections.
"""

from __future__ import annotations

import fcntl
import json
import math
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from agent.config import MEMORY_DIR, WARMUP_N


# Thread-level lock: guards in-process concurrent writes.
_MEM_LOCK = threading.RLock()


@contextmanager
def _file_lock(path: Path):
    """Cross-process exclusive lock backed by a .lock sidecar file.

    Combines with _MEM_LOCK so concurrent threads + processes are both safe.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _MEM_LOCK:
        with lock_path.open("a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


class Memory:
    def __init__(self, root: str | Path | None = None, warmup_n: int = WARMUP_N):
        self.root = Path(root or MEMORY_DIR)
        self.warmup_n = warmup_n
        self.root.mkdir(parents=True, exist_ok=True)
        # Ensure schema files exist
        for name, default in (
            ("object_size_priors.json", {"schema_version": "1.0", "objects": {}, "candidates": {}}),
            ("scene_scale_priors.json", {"schema_version": "1.0", "scenes": {}, "candidates": {}}),
        ):
            p = self.root / name
            if not p.exists():
                self._write_json(p, default)
        (self.root / "unsolved_cases.jsonl").touch(exist_ok=True)

    # ─── Object size priors ─────────────────────────────────────────────

    def get_object_size_prior(self, name: str) -> dict | None:
        data = self._read_json("object_size_priors.json")
        name = name.strip().lower()
        return data["objects"].get(name) or data.get("candidates", {}).get(name)

    def update_object_size_prior(self, name: str, **dims: float) -> None:
        self._update_prior("object_size_priors.json", "objects", name.strip().lower(), dims)

    # ─── Scene scale priors ─────────────────────────────────────────────

    def get_scene_scale_prior(self, scene_type: str) -> dict | None:
        data = self._read_json("scene_scale_priors.json")
        key = scene_type.strip().lower()
        return data["scenes"].get(key) or data.get("candidates", {}).get(key)

    def update_scene_scale_prior(self, scene_type: str, **dims: float) -> None:
        self._update_prior("scene_scale_priors.json", "scenes", scene_type.strip().lower(), dims)

    # ─── Unsolved cases ─────────────────────────────────────────────────

    def record_unsolved(self, sample_id: str, task_type: str,
                        predicted: str, gt: str, reason: str,
                        trajectory_summary: dict | None = None) -> None:
        unsolved_path = self.root / "unsolved_cases.jsonl"
        with _file_lock(unsolved_path):
            entry = {
                "ts": int(time.time()),
                "sample_id": sample_id,
                "task_type": task_type,
                "predicted": predicted,
                "gt": gt,
                "reason": reason,
                "trajectory": trajectory_summary or {},
            }
            with (self.root / "unsolved_cases.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ─── Format for LLM context ─────────────────────────────────────────

    def format_context(self, task_category: str,
                       object_categories: list[str] | None = None,
                       scene_type: str | None = None,
                       object_names: list[str] | None = None) -> str:
        """Build the memory context block appended to Planner/Reflector/Finalizer prompts.

        object_categories: canonical category names used during segmentation (e.g. ["chair", "cabinet"]).
            Preferred over object_names because they match what was actually stored in memory.
        object_names: fallback — question-text-parsed names used when categories aren't known yet
            (e.g. at planning time, before any segmentation has run).
        scene_type: inferred room type (e.g. "bedroom"). Used to retrieve scene-scale priors.
        """
        parts: list[str] = []

        # 优先使用 object_categories（规范类别名），回退到 object_names（正则解析）
        names_to_query = object_categories or object_names or []
        if names_to_query:
            size_lines = []
            seen: set[str] = set()
            for name in names_to_query:
                name = name.strip().lower()
                if not name or name in seen:
                    continue
                seen.add(name)
                p = self.get_object_size_prior(name)
                if not p:
                    continue
                dims = []
                for k in ("width", "height", "depth"):
                    stat = p.get(k)
                    if isinstance(stat, dict) and "mean" in stat:
                        dims.append(f"{k}={stat['mean']:.2f}±{stat.get('std', 0):.2f}m")
                if dims:
                    size_lines.append(f"  {name}: {', '.join(dims)}")
            if size_lines:
                parts.append("[Size Priors]\n" + "\n".join(size_lines))

        if scene_type:
            p = self.get_scene_scale_prior(scene_type)
            if p:
                dims = []
                for k in ("floor_area", "width", "height", "depth"):
                    stat = p.get(k)
                    if isinstance(stat, dict) and "mean" in stat:
                        unit = "m²" if k == "floor_area" else "m"
                        dims.append(f"{k}={stat['mean']:.2f}{unit}")
                if dims:
                    parts.append(f"[Scene Scale Prior — {scene_type}]\n  " + ", ".join(dims))

        return "\n\n".join(parts) if parts else "No relevant memory found."

    # ─── Internals ──────────────────────────────────────────────────────

    def _update_prior(self, filename: str, main_key: str,
                      entry_key: str, dims: dict[str, float]) -> None:
        path = self.root / filename
        with _file_lock(path):
            self._update_prior_locked(path, main_key, entry_key, dims)

    def _update_prior_locked(self, path: Path, main_key: str,
                              entry_key: str, dims: dict[str, float]) -> None:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.setdefault(main_key, {})
        data.setdefault("candidates", {})

        # Locate existing entry
        if entry_key in data[main_key]:
            entry = data[main_key][entry_key]
            in_main = True
        elif entry_key in data["candidates"]:
            entry = data["candidates"][entry_key]
            in_main = False
        else:
            entry = {}
            in_main = False

        # Welford update per dim
        for dim, val in dims.items():
            if val is None or not math.isfinite(val):
                continue
            stat = entry.get(dim, {"count": 0, "mean": 0.0, "M2": 0.0})
            n = stat["count"] + 1
            delta = val - stat["mean"]
            new_mean = stat["mean"] + delta / n
            new_M2 = stat.get("M2", 0.0) + delta * (val - new_mean)
            stat.update({
                "count": n,
                "mean": round(new_mean, 4),
                "M2": round(new_M2, 6),
                "std": round(math.sqrt(new_M2 / n) if n > 1 else 0.0, 4),
            })
            entry[dim] = stat

        entry["last_updated"] = int(time.time())

        # Warmup: promote to main pool once any dim reaches count >= warmup_n
        max_count = max(
            (v.get("count", 0) for v in entry.values() if isinstance(v, dict)),
            default=0,
        )
        if in_main or max_count >= self.warmup_n:
            data[main_key][entry_key] = entry
            data["candidates"].pop(entry_key, None)
        else:
            data["candidates"][entry_key] = entry

        self._write_json(path, data)

    def _read_json(self, filename: str) -> dict:
        p = self.root / filename
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
