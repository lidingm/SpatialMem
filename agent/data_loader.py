"""Data loaders for SpatialMem.

VSIBenchDataLoader    — reads the parquet-format VSIBench test set.
VSITrain10KDataLoader — reads the JSONL-format VSI-Train-10k training set.
SPARDataLoader        — reads SPAR ScanNet subset; renders annotation points/bboxes.

All loaders produce `SpatialSample` objects compatible with the SpatialMem agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from agent.config import (
    VSI_IMAGES_ROOT, VSI_PARQUET,
    VSI_TRAIN_IMAGES_ROOT, VSI_TRAIN_JSONL,
    SPAR_IMAGES_ROOT, SPAR_JSONL_ROOT, SPAR_RENDER_DIR,
)


# SPAR task_type -> internal task_category
SPAR_TASK_CATEGORY_MAP: dict[str, str] = {
    "depth_prediction_oc":               "depth_estimation",
    "depth_prediction_oc_mv":            "depth_estimation",
    "depth_prediction_oo":               "depth_estimation",
    "depth_prediction_oo_mv":            "depth_estimation",
    "camera_motion_infer":               "camera_motion",
    "view_change_infer":                 "view_change",
    "obj_frame_locate":                  "obj_frame_locate",
    "position_matching":                 "position_matching",
    "spatial_imagination_oc":            "spatial_imagination",
    "spatial_imagination_oc_mv":         "spatial_imagination",
    "spatial_imagination_oc_video":      "spatial_imagination",
    "spatial_imagination_oo":            "spatial_imagination",
    "spatial_imagination_oo_mv":         "spatial_imagination",
    "spatial_imagination_oo_video":      "spatial_imagination",
    "spatial_imagination_map_mv":        "spatial_imagination",
}

# 17 SPAR task types not present in VSI-Bench / VSI-Train-10k
SPAR_UNIQUE_TASK_TYPES: list[str] = sorted(SPAR_TASK_CATEGORY_MAP.keys())

# VSIBench question_type -> internal task_category
VSI_TASK_CATEGORY_MAP = {
    "object_counting":               "counting",
    "object_abs_distance":           "distance",
    "object_rel_distance":           "distance",
    "object_size_estimation":        "object_size",
    "room_size_estimation":          "room_size",
    "object_rel_direction_easy":     "spatial_relation",
    "object_rel_direction_medium":   "spatial_relation",
    "object_rel_direction_hard":     "spatial_relation",
    "obj_appearance_order":          "appearance_order",
    "route_planning":                "route_planning",
    # VSI-Train-10k question_type -> same internal task_category
    "relative_direction":            "spatial_relation",
    "relative_distance":             "distance",
    "absolute_distance":             "distance",
    "object_size":                   "object_size",
    "object_count":                  "counting",
    "room_size":                     "room_size",
    "appearance_order":              "appearance_order",
}


def classify_task_category(task_type: str) -> str:
    return (
        VSI_TASK_CATEGORY_MAP.get(task_type)
        or SPAR_TASK_CATEGORY_MAP.get(task_type)
        or task_type
    )


@dataclass
class SpatialSample:
    id: str
    question: str
    gt_answer: str
    image_paths: list[str]
    task_type: str
    answer_format: str          # "fill" (numeric) or "select" (A/B/C/D)
    annotations: dict = field(default_factory=dict)   # unused for VSI
    raw: dict = field(default_factory=dict)

    @property
    def scene_id(self) -> str:
        return self.raw.get("scene_name", "")

    @property
    def task_category(self) -> str:
        return classify_task_category(self.task_type)

    @property
    def is_multi_frame(self) -> bool:
        return len(self.image_paths) > 1

    @property
    def object_names(self) -> list[str]:
        """Heuristic: pull noun-phrases the question references, so Memory can
        prefetch relevant size priors."""
        names: list[str] = []
        for pattern in (
            r"(?:the|a|an)\s+([\w\s]+?)(?:\s+(?:and|to|from|is|are|in)\b|[,\.\?])",
            r"how\s+many\s+([\w\s]+?)\s+(?:are|is|would)",
            r"between\s+the\s+([\w\s]+?)\s+and\s+the\s+([\w\s]+?)[,\.\?]",
        ):
            for m in re.finditer(pattern, self.question, re.IGNORECASE):
                for g in m.groups():
                    if g:
                        names.append(g.strip().lower())
        # Deduplicate while preserving order
        return list(dict.fromkeys(names))


class VSIBenchDataLoader:
    def __init__(
        self,
        parquet_path: str | Path | None = None,
        images_root: str | Path | None = None,
    ):
        self.parquet_path = Path(parquet_path or VSI_PARQUET)
        self.images_root = Path(images_root or VSI_IMAGES_ROOT)
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_parquet(self.parquet_path)
        return self._df

    def question_types(self) -> list[str]:
        return sorted(self.df["question_type"].unique().tolist())

    def load_task_type(
        self,
        question_type: str,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> list[SpatialSample]:
        rows = self.df[self.df["question_type"] == question_type]
        if shuffle:
            rows = rows.sample(frac=1.0, random_state=seed)
        if max_samples is not None:
            rows = rows.head(max_samples)
        return [self._row_to_sample(r) for _, r in rows.iterrows()]

    def load_ids(self, ids: list[str | int]) -> list[SpatialSample]:
        int_ids = [int(i) for i in ids]
        rows = self.df[self.df["id"].isin(int_ids)]
        return [self._row_to_sample(r) for _, r in rows.iterrows()]

    def _row_to_sample(self, row: pd.Series) -> SpatialSample:
        scene_dir = self.images_root / row["dataset"] / row["scene_name"]
        image_paths = _list_scene_frames(scene_dir)

        options = row["options"]
        has_options = options is not None and (
            not isinstance(options, float) or not np.isnan(options)
        )
        # numpy arrays are truthy-ambiguous; explicit len check for arrays
        if isinstance(options, np.ndarray):
            has_options = options.size > 0

        answer_format = "select" if has_options else "fill"

        question = str(row["question"])
        if answer_format == "select":
            question = question + "\n" + "\n".join(str(o) for o in options)

        return SpatialSample(
            id=str(row["id"]),
            question=question,
            gt_answer=str(row["ground_truth"]),
            image_paths=image_paths,
            task_type=str(row["question_type"]),
            answer_format=answer_format,
            annotations={},
            raw={
                "id": int(row["id"]),
                "dataset": str(row["dataset"]),
                "scene_name": str(row["scene_name"]),
                "question_type": str(row["question_type"]),
                "options": list(options) if has_options else None,
            },
        )


def _list_scene_frames(scene_dir: Path) -> list[str]:
    """VSI frames are named `frame-{i}-of-32.jpg`. Sort by numeric index."""
    frames = sorted(
        scene_dir.glob("frame-*.jpg"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    return [str(p) for p in frames]



class VSITrain10KDataLoader:
    """Loads VSI-Train-10k from its JSONL file and pre-extracted image frames.

    Frame directory layout (produced by extract_frames.py):
      images_root / <video_path_without_ext> / frame-{i}-of-32.jpg
    e.g.
      images_root/scannet_videos_128f/train/scene0335_02_128f/frame-0-of-32.jpg
    """

    def __init__(
        self,
        jsonl_path: str | Path | None = None,
        images_root: str | Path | None = None,
    ):
        self.jsonl_path  = Path(jsonl_path  or VSI_TRAIN_JSONL)
        self.images_root = Path(images_root or VSI_TRAIN_IMAGES_ROOT)
        self._records: list[dict] | None = None

    @property
    def records(self) -> list[dict]:
        if self._records is None:
            with open(self.jsonl_path, encoding="utf-8") as f:
                self._records = [
                    {**json.loads(line), "_idx": i}
                    for i, line in enumerate(f)
                    if line.strip()
                ]
        return self._records

    def question_types(self) -> list[str]:
        return sorted({r["question_type"] for r in self.records})

    def load_task_type(
        self,
        question_type: str,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> list[SpatialSample]:
        import random
        rows = [r for r in self.records if r["question_type"] == question_type]
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)
        if max_samples is not None:
            rows = rows[:max_samples]
        return [self._row_to_sample(r) for r in rows]

    def load_all(
        self,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> list[SpatialSample]:
        import random
        rows = list(self.records)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)
        if max_samples is not None:
            rows = rows[:max_samples]
        return [self._row_to_sample(r) for r in rows]

    def load_ids(self, ids: list[str | int]) -> list[SpatialSample]:
        id_set = {str(i) for i in ids}
        rows = [r for r in self.records if str(r["_idx"]) in id_set]
        return [self._row_to_sample(r) for r in rows]

    def _row_to_sample(self, row: dict) -> SpatialSample:
        video_path = row["video"]
        scene_dir  = self.images_root / Path(video_path).with_suffix("")
        image_paths = _list_scene_frames(scene_dir)

        sample_type   = row.get("type", "oe")
        answer_format = "select" if sample_type == "mc" else "fill"

        # Strip the <image> prefix that VSI-Train-10k embeds in the question
        raw_question = str(row.get("question", ""))
        question = raw_question.replace("<image>", "").strip()

        scene_name = Path(video_path).stem   # e.g. scene0335_02_128f

        return SpatialSample(
            id=str(row["_idx"]),
            question=question,
            gt_answer=str(row["ground_truth"]),
            image_paths=image_paths,
            task_type=str(row["question_type"]),
            answer_format=answer_format,
            annotations={},
            raw={
                "video": video_path,
                "scene_name": scene_name,
                "source": row.get("source", ""),
                "question_type": row["question_type"],
                "question_type_detail": row.get("question_type_detail", ""),
                "type": sample_type,
            },
        )


# ─── SPAR annotation rendering ─────────────────────────────────────────────

_NAMED_COLORS = {
    "red":    (220, 50,  50),
    "green":  (50,  200, 50),
    "blue":   (50,  100, 220),
    "yellow": (220, 200, 50),
}
_ANON_COLORS = [
    (220, 50,  50),
    (50,  200, 50),
    (50,  100, 220),
    (220, 200, 50),
    (50,  200, 200),
    (200, 50,  200),
]


def _render_spar_annotations(
    sample_id: str,
    image_paths: list[str],
    row: dict,
    render_dir: Path,
) -> list[str]:
    """Draw annotation points/bboxes onto image copies and return new paths.

    Returns original paths unchanged for images with no annotations.
    """
    from PIL import Image, ImageDraw  # type: ignore

    # Collect per-image draw commands: {img_idx: [(kind, color, coords)]}
    per_image: dict[int, list] = {}

    point_img_idx = (row.get("point_img_idx") or [[]])[0]
    bbox_img_idx  = (row.get("bbox_img_idx")  or [[]])[0]

    # Named color points / bboxes
    for ci, color_name in enumerate(("red", "green", "blue", "yellow")):
        color = _NAMED_COLORS[color_name]
        pt_raw = row.get(f"{color_name}_point")
        if pt_raw:
            pt = pt_raw[0]  # [[x,y]] -> [x,y]
            idx = point_img_idx[ci] if ci < len(point_img_idx) else 0
            per_image.setdefault(idx, []).append(("point", color, pt))
        bb_raw = row.get(f"{color_name}_bbox")
        if bb_raw:
            bb = bb_raw[0]  # [[x1,y1,x2,y2]] -> [x1,y1,x2,y2]
            idx = bbox_img_idx[ci] if ci < len(bbox_img_idx) else 0
            per_image.setdefault(idx, []).append(("bbox", color, bb))

    # Anonymous point_list / bbox_list (spatial_imagination_map_mv etc.)
    pl_raw = row.get("point_list")
    if pl_raw:
        for pi, pt in enumerate(pl_raw[0]):
            color = _ANON_COLORS[pi % len(_ANON_COLORS)]
            idx = point_img_idx[pi] if pi < len(point_img_idx) else 0
            per_image.setdefault(idx, []).append(("point", color, pt))

    bl_raw = row.get("bbox_list")
    if bl_raw:
        for bi, bb in enumerate(bl_raw[0]):
            color = _ANON_COLORS[bi % len(_ANON_COLORS)]
            idx = bbox_img_idx[bi] if bi < len(bbox_img_idx) else 0
            per_image.setdefault(idx, []).append(("bbox", color, bb))

    if not per_image:
        return image_paths

    sample_dir = render_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    rendered = list(image_paths)
    for img_idx, commands in per_image.items():
        if img_idx >= len(image_paths):
            continue
        orig = Path(image_paths[img_idx])
        img  = Image.open(orig).convert("RGB")
        draw = ImageDraw.Draw(img)
        for kind, color, coords in commands:
            if kind == "point":
                x, y, r = coords[0], coords[1], 6
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            else:  # bbox
                draw.rectangle(
                    [coords[0], coords[1], coords[2], coords[3]],
                    outline=color, width=2,
                )
        out = sample_dir / orig.name
        img.save(str(out))
        rendered[img_idx] = str(out)

    return rendered


# ─── SPAR DataLoader ────────────────────────────────────────────────────────

class SPARDataLoader:
    """Loads SPAR (ScanNet subset) training data.

    Directory layout::

        SPAR_JSONL_ROOT/
            {task_type}/
                {format}/       # select | fill | sentence
                    *.jsonl

    Annotation points/bboxes are rendered onto image copies stored in
    SPAR_RENDER_DIR before being returned as SpatialSample image_paths.
    Format preference: select > fill > sentence.
    """

    _FORMAT_PREFERENCE = ("select", "fill", "sentence")

    def __init__(
        self,
        jsonl_root: str | Path | None = None,
        images_root: str | Path | None = None,
        render_dir: str | Path | None = None,
    ):
        self.jsonl_root  = Path(jsonl_root  or SPAR_JSONL_ROOT)
        self.images_root = Path(images_root or SPAR_IMAGES_ROOT)
        self.render_dir  = Path(render_dir  or SPAR_RENDER_DIR)

    def available_formats(self, task_type: str) -> list[str]:
        """Return format subdirs that exist for this task type."""
        task_dir = self.jsonl_root / task_type
        return [
            fmt for fmt in self._FORMAT_PREFERENCE
            if (task_dir / fmt).is_dir()
        ]

    def best_format(self, task_type: str) -> str | None:
        fmts = self.available_formats(task_type)
        return fmts[0] if fmts else None

    def load_task_type(
        self,
        task_type: str,
        max_samples: int | None = 200,
        preferred_format: str | None = None,
    ) -> list[SpatialSample]:
        """Load up to `max_samples` from the best available format (in file order)."""
        fmt = preferred_format or self.best_format(task_type)
        if fmt is None:
            raise FileNotFoundError(
                f"No SPAR data found for task_type={task_type!r} under {self.jsonl_root}"
            )
        fmt_dir = self.jsonl_root / task_type / fmt
        records = self._load_records(fmt_dir)
        if max_samples is not None:
            records = records[:max_samples]
        return [self._row_to_sample(r, task_type, fmt) for r in records]

    # ── Internals ──────────────────────────────────────────────────────────

    def _load_records(self, fmt_dir: Path) -> list[dict]:
        """Read all JSONL records from a format directory (sorted by filename)."""
        records: list[dict] = []
        for jsonl_file in sorted(fmt_dir.glob("*.jsonl")):
            with jsonl_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def _row_to_sample(self, row: dict, task_type: str, fmt: str) -> SpatialSample:
        conversations = row.get("conversations", [])
        question  = conversations[0]["value"] if conversations else ""
        gt_answer = conversations[1]["value"] if len(conversations) > 1 else ""

        rel_paths  = row.get("image", [])
        orig_paths = [str(self.images_root / p) for p in rel_paths]
        image_paths = _render_spar_annotations(
            sample_id=row["id"],
            image_paths=orig_paths,
            row=row,
            render_dir=self.render_dir,
        )

        scene_id = rel_paths[0].split("/")[0] if rel_paths else ""

        return SpatialSample(
            id=row["id"],
            question=question,
            gt_answer=gt_answer,
            image_paths=image_paths,
            task_type=task_type,
            answer_format=fmt,
            annotations={
                k: row[k] for k in (
                    "red_point", "green_point", "blue_point", "yellow_point",
                    "red_bbox",  "green_bbox",  "blue_bbox",  "yellow_bbox",
                    "point_list", "bbox_list",
                    "point_img_idx", "bbox_img_idx",
                ) if k in row
            },
            raw={
                "id":        row["id"],
                "type":      row.get("type", task_type),
                "scene_id":  scene_id,
                "image":     rel_paths,
                "format":    fmt,
            },
        )

