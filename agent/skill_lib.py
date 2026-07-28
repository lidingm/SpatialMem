"""SKILL library: on-disk folder-based skills with retrieval, invocation,
and evolutionary bookkeeping (record_call, append_pitfall, rewrite_when_to_use,
save_pending, promote_from_pending).

The actual LLM-driven distillation lives in `agent/evolve.py`; this module
handles only the mechanical read/write and dispatch.

A SKILL is a directory containing:
  SKILL.md          -- YAML frontmatter (stats + task_categories) plus Markdown sections
  execute.py        -- must define execute(sample, tools, ctx, params=None) -> dict
  trajectories.jsonl -- one JSON summary per invocation, append-only
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.config import SKILLS_DIR, SKILLS_PENDING_DIR, PENDING_PROMOTE_K


SKILL_MD_SECTIONS = ("When to Use", "Parameters", "Tool Sequence",
                     "Known Pitfalls", "Examples", "Checker")


# Module-level lock: SkillLib writes SKILL.md and trajectories.jsonl files
# from multiple threads during parallel training. All read-modify-write
# operations acquire this lock so state stays consistent. LLM calls in
# evolve.py happen OUTSIDE this lock, keeping parallelism.
_SKILL_LOCK = threading.RLock()


@dataclass
class Skill:
    name: str
    dir: Path
    frontmatter: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    body: str = ""                     # raw MD body without frontmatter

    @property
    def task_categories(self) -> list[str]:
        v = self.frontmatter.get("task_categories", [])
        return v if isinstance(v, list) else [v]

    @property
    def success_rate(self) -> float:
        return float(self.frontmatter.get("success_rate", 0.0))

    @property
    def total_calls(self) -> int:
        return int(self.frontmatter.get("total_calls", 0))

    @property
    def success_count(self) -> int:
        return int(self.frontmatter.get("success_count", 0))

    def brief(self, max_chars_section: int = 900) -> str:
        """Compact SKILL.md rendering for planner prompts.

        Planner should see the skill's operational contract and accumulated
        experience, not only its name. Each section is capped independently so
        Examples/Pitfalls can grow without crowding out other retrieved skills.
        """
        # Detect if this SKILL is currently sitting in the pending dir
        is_pending = "pending" in self.dir.parts
        pending_tag = "  [AUTO-DISTILLED — this SKILL was self-summarized by the system for this task category; feel free to use it if it fits your current task]" if is_pending else ""

        def _section(name: str) -> str:
            body = self.sections.get(name, "").strip() or "None."
            if len(body) > max_chars_section:
                body = body[:max_chars_section].rstrip() + "..."
            return f"## {name}\n{body}"

        sections = "\n\n".join(
            _section(name)
            for name in ("When to Use", "Parameters", "Tool Sequence",
                         "Known Pitfalls", "Examples")
        )
        return (
            f"# SKILL: {self.name}{pending_tag}\n"
            f"metadata: task_categories={self.task_categories}, "
            f"success_rate={self.success_rate:.2f}, calls={self.total_calls}, "
            f"seeded={self.frontmatter.get('seeded', False)}, "
            f"version={self.frontmatter.get('version', 0)}\n\n"
            f"{sections}"
        )


class SkillLib:
    def __init__(self, root: str | Path | None = None,
                 pending_dir: str | Path | None = None):
        self.root = Path(root or SKILLS_DIR)
        self.pending_dir = Path(pending_dir or SKILLS_PENDING_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    # ─── Loading ────────────────────────────────────────────────────────

    def load_all(self) -> list[Skill]:
        skills: list[Skill] = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("_") or d.name == "pending":
                continue
            s = self._load_skill_dir(d)
            if s is not None:
                skills.append(s)
        return skills

    def get(self, name: str) -> Skill | None:
        d = self.root / name
        if not d.is_dir():
            d = self.pending_dir / name
        if not d.is_dir():
            return None
        return self._load_skill_dir(d)

    def retrieve(self, task_category: str,
                 question: str = "",
                 top_k: int = 3,
                 include_pending: bool = True) -> list[Skill]:
        """Return SKILLs whose frontmatter task_categories includes the given
        category. Ranking:
          1. Main-library SKILLs first (sorted by success_rate DESC, calls DESC)
          2. Pending SKILLs after (sorted the same way)
        Pending SKILLs are included by default so newly evolved drafts can
        actually be exercised — otherwise they can never accrue the successes
        needed for promotion (catch-22).
        """
        main_pool = self.load_all()
        main_matches = [s for s in main_pool if task_category in s.task_categories]
        main_matches.sort(key=lambda s: (s.success_rate, s.total_calls), reverse=True)

        pending_matches: list[Skill] = []
        if include_pending:
            for d in sorted(self.pending_dir.iterdir()):
                if d.is_dir():
                    s = self._load_skill_dir(d)
                    if s is not None and task_category in s.task_categories:
                        pending_matches.append(s)
            pending_matches.sort(key=lambda s: (s.success_rate, s.total_calls), reverse=True)

        return (main_matches + pending_matches)[:top_k]

    # ─── Invocation ─────────────────────────────────────────────────────

    def invoke(self, skill: Skill, sample: Any, tools: Any,
               ctx: dict, params: dict | None = None) -> dict:
        fn = self._import_execute(skill.dir)
        return fn(sample, tools, ctx, params or {})

    # ─── Bookkeeping (called by evolve.py) ──────────────────────────────

    def record_call(self, skill_name: str, success: bool,
                    trajectory_summary: dict) -> None:
        with _SKILL_LOCK:
            skill = self.get(skill_name)
            if skill is None:
                return
            fm = skill.frontmatter
            fm["total_calls"] = int(fm.get("total_calls", 0)) + 1
            if success:
                fm["success_count"] = int(fm.get("success_count", 0)) + 1
            else:
                fm["failure_count"] = int(fm.get("failure_count", 0)) + 1
            fm["success_rate"] = round(fm["success_count"] / max(fm["total_calls"], 1), 4)
            self._save_frontmatter(skill)

            line = json.dumps({"ts": int(time.time()), "success": success, **trajectory_summary},
                              ensure_ascii=False)
            (skill.dir / "trajectories.jsonl").open("a", encoding="utf-8").write(line + "\n")

    def append_pitfall(self, skill_name: str, entry_text: str) -> None:
        """Append a bullet to Known Pitfalls.

        Caller must have already decided this is a genuinely new pitfall.
        """
        with _SKILL_LOCK:
            skill = self.get(skill_name)
            if skill is None:
                return
            current = skill.sections.get("Known Pitfalls", "").strip()
            if current.lower() in ("", "none yet.", "none."):
                new_body = "- " + entry_text.strip()
            else:
                new_body = current + "\n- " + entry_text.strip()
            skill.sections["Known Pitfalls"] = new_body
            self._save_sections(skill)

    def append_example(self, skill_name: str, entry_text: str,
                       sample_id: str | None = None) -> bool:
        """Append one sample-specific insight to Examples.

        Examples are intentionally not generic tool pitfalls. They capture a
        concrete question/sample detail that is useful to remember.
        """
        entry_text = entry_text.strip()
        if not entry_text:
            return False
        with _SKILL_LOCK:
            skill = self.get(skill_name)
            if skill is None:
                return False
            current = skill.sections.get("Examples", "").strip()
            if sample_id and f"Sample {sample_id}" in current:
                return False
            if current.lower() in ("", "none yet.", "none."):
                new_body = entry_text
            else:
                new_body = current + "\n\n" + entry_text
            skill.sections["Examples"] = new_body
            self._save_sections(skill)
            return True

    def append_checker_note(self, skill_name: str, entry_text: str) -> bool:
        """Append one Checker-specific lesson to the Checker section."""
        entry_text = entry_text.strip()
        if not entry_text:
            return False
        with _SKILL_LOCK:
            skill = self.get(skill_name)
            if skill is None:
                return False
            current = skill.sections.get("Checker", "").strip()
            if current.lower() in ("", "none yet.", "none."):
                new_body = "- " + entry_text.lstrip("- ").strip()
            else:
                normalized = entry_text.lstrip("- ").strip()
                if normalized in current:
                    return False
                new_body = current + "\n- " + normalized
            skill.sections["Checker"] = new_body
            self._save_sections(skill)
            return True

    def rewrite_section(self, skill_name: str, section: str, new_text: str) -> None:
        """Overwrite an entire section body."""
        with _SKILL_LOCK:
            skill = self.get(skill_name)
            if skill is None or section not in SKILL_MD_SECTIONS:
                return
            skill.sections[section] = new_text.strip()
            self._save_sections(skill)

    def bump_version(self, skill_name: str) -> None:
        with _SKILL_LOCK:
            skill = self.get(skill_name)
            if skill is None:
                return
            skill.frontmatter["version"] = int(skill.frontmatter.get("version", 1)) + 1
            self._save_frontmatter(skill)

    # ─── Pending pool ───────────────────────────────────────────────────

    def save_pending(self, name: str, skill_md: str, execute_py: str) -> Path:
        with _SKILL_LOCK:
            target = self.pending_dir / name
            target.mkdir(parents=True, exist_ok=True)
            (target / "SKILL.md").write_text(skill_md, encoding="utf-8")
            (target / "execute.py").write_text(execute_py, encoding="utf-8")
            (target / "trajectories.jsonl").touch(exist_ok=True)
            return target

    def promote_from_pending(self, name: str) -> bool:
        with _SKILL_LOCK:
            src = self.pending_dir / name
            dst = self.root / name
            if not src.is_dir() or dst.exists():
                return False
            shutil.move(str(src), str(dst))
            return True

    def promote_eligible_pending(self, k: int = PENDING_PROMOTE_K) -> list[str]:
        """Promote every pending SKILL whose success_count >= k."""
        with _SKILL_LOCK:
            promoted: list[str] = []
            for d in list(self.pending_dir.iterdir()):
                if not d.is_dir():
                    continue
                s = self._load_skill_dir(d)
                if s is None:
                    continue
                if s.success_count >= k:
                    if self.promote_from_pending(d.name):
                        promoted.append(d.name)
            return promoted

    # ─── Internal file I/O ──────────────────────────────────────────────

    def _load_skill_dir(self, d: Path) -> Skill | None:
        md_path = d / "SKILL.md"
        exec_path = d / "execute.py"
        if not md_path.exists() or not exec_path.exists():
            return None
        fm, sections, body = _parse_skill_md(md_path.read_text(encoding="utf-8"))
        return Skill(name=d.name, dir=d, frontmatter=fm, sections=sections, body=body)

    def _import_execute(self, skill_dir: Path) -> Callable:
        exec_path = skill_dir / "execute.py"
        module_name = f"_skill_{skill_dir.name}_{skill_dir.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, exec_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {exec_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        if not hasattr(mod, "execute"):
            raise RuntimeError(f"{exec_path} does not define execute()")
        return getattr(mod, "execute")

    def _save_frontmatter(self, skill: Skill) -> None:
        text = _render_skill_md(skill.frontmatter, skill.sections)
        (skill.dir / "SKILL.md").write_text(text, encoding="utf-8")

    _save_sections = _save_frontmatter  # same operation


# ─── SKILL.md parsing / rendering ──────────────────────────────────────

def _parse_skill_md(text: str) -> tuple[dict[str, Any], dict[str, str], str]:
    """Return (frontmatter, sections_dict, body_without_frontmatter)."""
    fm: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                fm = {}
            body = parts[2].lstrip("\n")

    sections: dict[str, str] = {}
    current_name: str | None = None
    current_buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("# "):
            if current_name is not None:
                sections[current_name] = "\n".join(current_buf).strip()
            current_name = line[2:].strip()
            current_buf = []
        else:
            current_buf.append(line)
    if current_name is not None:
        sections[current_name] = "\n".join(current_buf).strip()

    return fm, sections, body


def _render_skill_md(frontmatter: dict[str, Any], sections: dict[str, str]) -> str:
    fm_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    parts = ["---", fm_text, "---", ""]
    for name in SKILL_MD_SECTIONS:
        body = sections.get(name, "None yet.").strip() or "None yet."
        parts.append(f"# {name}")
        parts.append("")
        parts.append(body)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
