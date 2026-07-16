"""Unified configuration for SpatialMem.

All paths, LLM credentials, and evolution hyper-parameters live here.
Environment variables override defaults where indicated.
"""

from __future__ import annotations

import os
from pathlib import Path


# ─── Project layout ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMORY_DIR = PROJECT_ROOT / "memory"
SKILLS_DIR = PROJECT_ROOT / "skills"
SKILLS_PENDING_DIR = SKILLS_DIR / "pending"
SKILL_TEMPLATE_DIR = SKILLS_DIR / "_template"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
CKPTS_DIR = PROJECT_ROOT / "ckpts"


# ─── Dataset (VSIBench test) ──────────────────────────────────────────

VSI_PARQUET = Path(os.getenv(
    "VSI_PARQUET",
    "/home/zhouruofan/datasets/VSI-Bench/test_debiased.parquet",
))
VSI_IMAGES_ROOT = Path(os.getenv(
    "VSI_IMAGES_ROOT",
    "/home/zhouruofan/datasets/VSI-Bench/images",
))

# ─── Dataset (VSI-Train-10k) ──────────────────────────────────────────

VSI_TRAIN_JSONL = Path(os.getenv(
    "VSI_TRAIN_JSONL",
    "/home/zhouruofan/datasets/VSI-Train-10k/vsi_train_10k.jsonl",
))
VSI_TRAIN_IMAGES_ROOT = Path(os.getenv(
    "VSI_TRAIN_IMAGES_ROOT",
    "/home/zhouruofan/datasets/VSI-Train-10k/images",
))


# ─── Dataset (SPAR / ScanNet) ────────────────────────────────────────────

SPAR_JSONL_ROOT = Path(os.getenv(
    "SPAR_JSONL_ROOT",
    "/home/zhouruofan/datasets/spar/scannet/qa_jsonl/train",
))
SPAR_IMAGES_ROOT = Path(os.getenv(
    "SPAR_IMAGES_ROOT",
    "/home/zhouruofan/datasets/spar/scannet/images",
))
SPAR_RENDER_DIR = Path(os.getenv(
    "SPAR_RENDER_DIR",
    "/tmp/spar_rendered",
))


# ─── LLM (env-var driven, no defaults committed) ───────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))


# ─── Reasoning loop ────────────────────────────────────────────────────

MAX_ROUNDS = 3
CONFIDENCE_THRESHOLD = 0.8
MAX_FRAMES_PER_SAMPLE = 32   # uniformly sample 32 frames per scene


# ─── Evolution (SKILL/Memory) ──────────────────────────────────────────

# Memory: object size / scene scale prior warmup threshold — count required
# in a per-object candidate buffer before promoting to the main prior file.
WARMUP_N = 5

# SKILL: distillation is triggered every N successes of a given SKILL.
DISTILL_TRIGGER_N = 5

# SKILL: a pending SKILL is promoted from skills/pending/ to skills/ after
# being re-used successfully K times.
PENDING_PROMOTE_K = 3

# SKILL: how many recent trajectory summaries to feed the distillation LLM.
DISTILL_WINDOW = 20

# SKILL: category-level proactive bootstrap. When N samples of a given
# task_category have gone through the raw-tool path without any SKILL
# match (main or pending), the framework synthesizes a SKILL from the
# accumulated trajectories. Prevents unpaved task types from being
# permanently uncovered just because reactive Path B/D never fires.
BOOTSTRAP_TRIGGER_N = int(os.getenv("BOOTSTRAP_TRIGGER_N", "8"))

# Reflection: attempts to reconstruct a corrected plan on wrong answers.
REFLECT_MAX_ATTEMPTS = 1


# ─── Frame sampling ────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create runtime output directories if they don't exist."""
    for d in (MEMORY_DIR, SKILLS_DIR, SKILLS_PENDING_DIR, OUTPUTS_DIR, RESULTS_DIR, CKPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
