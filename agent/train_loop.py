"""Training loop / evaluation driver for SpatialMem.

- Iterates VSIBench samples of a given question_type
- Runs the orchestrator on each
- Evaluates against GT with format-aware matching (numeric MRA, letter match)
- Triggers evolve.evolve_after_sample after each sample
- Periodically promotes eligible pending SKILLs

Also usable for pure evaluation (skip evolution with evolve=False).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from agent.config import PENDING_PROMOTE_K, RESULTS_DIR
from agent.data_loader import SpatialSample, VSIBenchDataLoader
from agent.evolve import evolve_after_sample
from agent.llm_client import LLMClient
from agent.memory import Memory
from agent.orchestrator import AgentResult, Orchestrator
from agent.reflector import Reflector
from agent.skill_lib import SkillLib


class TrainLoop:
    def __init__(
        self,
        orchestrator: Orchestrator,
        loader: VSIBenchDataLoader,
        memory: Memory,
        skill_lib: SkillLib,
        reflector: Reflector,
        llm: LLMClient,
        results_dir: str | Path | None = None,
        promote_every: int = 20,
    ):
        self.orch = orchestrator
        self.loader = loader
        self.memory = memory
        self.skill_lib = skill_lib
        self.reflector = reflector
        self.llm = llm
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.promote_every = promote_every

    # ─── Batch run ───────────────────────────────────────────────────

    def run_task_type(
        self,
        question_type: str,
        max_samples: int | None = 10,
        shuffle: bool = True,
        seed: int = 42,
        evolve: bool = True,
        save_name: str | None = None,
    ) -> list[AgentResult]:
        samples = self.loader.load_task_type(
            question_type, max_samples=max_samples, shuffle=shuffle, seed=seed)
        print(f"Loaded {len(samples)} samples for {question_type}")

        results: list[AgentResult] = []
        for i, sample in enumerate(samples):
            print(f"\n{'#'*60}\n# {i+1}/{len(samples)}  id={sample.id}\n{'#'*60}")
            result = self.run_single(sample, evolve=evolve)
            results.append(result)
            status = "CORRECT" if result.success else "WRONG"
            print(f">>> {status} | pred={result.predicted_answer!r} gt={result.gt_answer!r} "
                  f"skills={result.skills_used} time={result.duration_seconds:.1f}s")

            if evolve and (i + 1) % self.promote_every == 0:
                promoted = self.skill_lib.promote_eligible_pending(PENDING_PROMOTE_K)
                if promoted:
                    print(f"[PROMOTE] moved to main library: {promoted}")

        # Final promote sweep
        if evolve:
            promoted = self.skill_lib.promote_eligible_pending(PENDING_PROMOTE_K)
            if promoted:
                print(f"[PROMOTE final] moved to main library: {promoted}")

        out_path = self.results_dir / f"{save_name or question_type}.jsonl"
        self.save_results(results, out_path)
        self.print_summary(results)
        return results

    def run_single(self, sample: SpatialSample, evolve: bool = True) -> AgentResult:
        result = self.orch.run(sample)
        result.success = evaluate_answer(
            result.predicted_answer, result.gt_answer,
            result.answer_format, sample.task_type)
        if evolve:
            update = evolve_after_sample(
                result, sample, self.memory, self.skill_lib,
                self.reflector, self.orch, self.llm, evaluate_answer,
                verbose=self.orch.verbose,
            )
            if self.orch.verbose:
                print(f"[EVOLVE] path={update['path']} "
                      f"skill_updates={update['skill_updates']} "
                      f"memory={update['memory_updates']} "
                      f"pending={update['pending_saved']}")
        return result

    # ─── I/O ──────────────────────────────────────────────────────────

    def save_results(self, results: Sequence[AgentResult], path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in results:
                entry = {
                    "sample_id": r.sample_id,
                    "task_category": r.task_category,
                    "answer_format": r.answer_format,
                    "question": r.question,
                    "gt_answer": r.gt_answer,
                    "predicted_answer": r.predicted_answer,
                    "success": r.success,
                    "confidence": r.confidence,
                    "num_rounds": r.num_rounds,
                    "duration_seconds": r.duration_seconds,
                    "chosen_skill": r.chosen_skill,
                    "skills_used": r.skills_used,
                    "tool_calls": [{"tool": t["tool_name"], "success": t["success"]}
                                   for t in r.tool_calls],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Results saved to {path}")

    def print_summary(self, results: Sequence[AgentResult]):
        if not results:
            return
        correct = sum(1 for r in results if r.success)
        total = len(results)
        skill_hits = sum(1 for r in results if r.skills_used)
        print(f"\n{'='*40}")
        print(f"Accuracy:  {correct}/{total} = {correct/total:.1%}")
        print(f"SKILL used: {skill_hits}/{total} = {skill_hits/total:.1%}")
        print(f"Avg time:  {sum(r.duration_seconds for r in results)/total:.1f}s")
        print(f"Avg rounds:{sum(r.num_rounds for r in results)/total:.1f}")
        print(f"{'='*40}")


# ─── Answer evaluation ───────────────────────────────────────────────

def evaluate_answer(predicted: str, gt: str, answer_format: str, task_type: str) -> bool:
    predicted = (predicted or "").strip()
    gt = (gt or "").strip()

    if answer_format == "select":
        p = _extract_option(predicted)
        g = _extract_option(gt)
        return p is not None and p == g

    if answer_format == "judge":
        p = predicted.lower().strip(".")
        g = gt.lower().strip(".")
        return ("yes" in p) == ("yes" in g)

    if answer_format == "fill":
        pn = _extract_number(predicted)
        gn = _extract_number(gt)
        if pn is not None and gn is not None:
            # VSIBench evaluation: MRA-style relative tolerance for numeric
            if abs(gn) < 0.01:
                return abs(pn - gn) < 0.5
            return abs(pn - gn) / abs(gn) < 0.3
        return predicted.lower() == gt.lower()

    if answer_format == "sentence":
        pn = _extract_number(predicted)
        gn = _extract_number(gt)
        if pn is not None and gn is not None:
            if abs(gn) < 0.01:
                return abs(pn - gn) < 0.5
            return abs(pn - gn) / abs(gn) < 0.3
        return predicted.lower().strip(".") == gt.lower().strip(".")

    return predicted.strip().lower() == gt.strip().lower()


def _extract_number(text: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group()) if m else None


def _extract_option(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else None
