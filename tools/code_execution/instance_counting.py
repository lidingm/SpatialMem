"""Instance counting: number of unique instances after 3D clustering,
optionally calibrated by a VLM reviewing the annotated frames.

The calibration deliberately does NOT ask the model for a final count — it only
reports structured, high-confidence corrections (missed / spurious / merge /
split) and the count is derived from them here. That keeps the arithmetic
deterministic and auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


# ─── Counting ───────────────────────────────────────────────────────────────

def count_unique_instances(
    results_3d: dict,
    llm=None,
    object_name: str | None = None,
    memory=None,
    conf_threshold: float = 0.7,
    checker_notes: str | None = None,
) -> dict:
    """Count unique instances from 3D localization results (after clustering).

    When `llm` and `object_name` are given and `results_3d` carries the annotated
    frames rendered by instance_3d_localization, a VLM reviews those frames and the
    count is corrected from its structured findings. Falls back to the raw cluster
    count if calibration is unavailable or fails.

    Returns:
        dict with total_unique, obj_id_list, and (when calibrated) `calibration`.
    """
    obj_id_list = results_3d.get("obj_id_list", [])
    result = {"total_unique": len(obj_id_list), "obj_id_list": obj_id_list}

    frames = (results_3d.get("annotated_frames_by_category") or {}).get(object_name or "") \
        or results_3d.get("annotated_frames")
    instances = (results_3d.get("instances_by_category") or {}).get(object_name or "") \
        or results_3d.get("instances")
    if instances is not None:
        result["final_instances"] = list(instances)   # no calibration -> all of them

    if llm is not None and object_name and frames and instances is not None:
        from agent.checker import Checker
        cal = Checker(llm, memory=memory, checker_notes=checker_notes).calibrate_count(
            object_name, results_3d, conf_threshold=conf_threshold)
        result["calibration"] = cal
        if cal.get("error") is None:
            result["total_unique"] = cal["adjusted_count"]
            # keep obj_id_list consistent with the calibrated count — a stale list
            # of pre-calibration ids contradicts total_unique and misleads readers.
            # (After split/missed the count may exceed the surviving ids; total_unique
            #  is always the authoritative answer.)
            result["obj_id_list"] = cal["kept_ids"]
            # The surviving instances WITH their per-frame 2D boxes — this is the
            # post-calibration state the reflector / skill distillation needs to see.
            # rebuild the instance list so it reflects the VLM's actual operations
            # (merge unions boxes, split partitions them, missed adds placeholders)
            result["final_instances"] = apply_corrections_to_instances(
                instances, cal, object_name)
            _save_calibration(results_3d, object_name, cal, result["final_instances"])
    return result


def apply_corrections_to_instances(instances: list[dict], cal: dict,
                                   object_name: str) -> list[dict]:
    """Rebuild the instance list so it actually reflects the VLM's operations.

    The count arithmetic alone is not enough for downstream reflection / skill
    distillation — those need the resulting BOXES:
      spurious -> instance dropped
      merge    -> the group becomes ONE instance whose 2D boxes / frames are unioned
      split    -> the instance is partitioned by `target_frame_groups` into one
                  instance per real object; `non_target_frames` boxes are dropped.
                  Without those groups the count is still fixed but the boxes
                  cannot be partitioned — the instance is flagged instead.
      missed   -> a box-less placeholder is appended (we only know the frames)
    """
    resp = cal.get("raw") or {}
    conf = cal.get("_conf_threshold", 0.7)
    by_id = {int(it["id"]): dict(it) for it in instances}
    kept = set(cal.get("kept_ids") or [])
    out: list[dict] = []
    consumed: set[int] = set()

    # merge — union the group's boxes/frames into the surviving instance
    for m in (resp.get("merge") or []):
        if float(m.get("confidence", 0)) < conf:
            continue
        grp = sorted({int(x) - 1 for x in (m.get("instances") or [])} & set(by_id))
        if len(grp) < 2:
            continue
        keep_id = grp[0]
        if keep_id not in kept:
            continue
        base = dict(by_id[keep_id])
        boxes, frames = list(base.get("boxes_2d") or []), set(base.get("frames") or [])
        for g in grp[1:]:
            boxes += list(by_id[g].get("boxes_2d") or [])
            frames |= set(by_id[g].get("frames") or [])
            consumed.add(g)
        base["boxes_2d"] = sorted(boxes, key=lambda b: b["frame"])
        base["frames"] = sorted(frames)
        base["merged_from"] = [g + 1 for g in grp]
        out.append(base)
        consumed.add(keep_id)

    # split — partition the boxes by the VLM's frame groups
    for s in (resp.get("split") or []):
        if float(s.get("confidence", 0)) < conf:
            continue
        gid = int(s.get("instance", 0)) - 1
        if gid not in by_id or gid in consumed or gid not in kept:
            continue
        base = by_id[gid]
        groups = [g for g in (s.get("target_frame_groups") or []) if g]
        non_target = set(int(f) for f in (s.get("non_target_frames") or []))
        if not groups:
            flagged = dict(base)
            flagged["split_unresolved"] = int(s.get("n_target", 2))
            out.append(flagged)
            consumed.add(gid)
            continue
        all_boxes = list(base.get("boxes_2d") or [])
        for gi, frames_g in enumerate(groups, start=1):
            fset = {int(f) for f in frames_g} - non_target
            piece = dict(base)
            piece["boxes_2d"] = [b for b in all_boxes if int(b["frame"]) in fset]
            piece["frames"] = sorted(fset)
            piece["label"] = f"{base.get('category', object_name)} {base['id'] + 1}.{gi}"
            piece["split_from"] = base["id"] + 1
            out.append(piece)
        consumed.add(gid)

    # untouched survivors
    for oid in sorted(kept):
        if oid not in consumed and oid in by_id:
            out.append(dict(by_id[oid]))

    # missed — the VLM only tells us the frames it saw the object in, so these
    # placeholders carry no boxes; they exist so the count and the instance list
    # stay in step and the reflector can see what was claimed to be missing.
    n_next = 1
    for mm in (resp.get("missed") or []):
        if float(mm.get("confidence", 0)) < conf:
            continue
        out.append({
            "label": f"{object_name} (missed {n_next})",
            "category": object_name,
            "id": -n_next,
            "color": "none",
            "frames": sorted({int(f) for f in (mm.get("frames") or [])}),
            "boxes_2d": [],
            "from_vlm_missed": True,
            "description": mm.get("description", ""),
        })
        n_next += 1
    return _renumber_final_instances(out, object_name)


def _first_frame(inst: dict) -> int:
    frames = [_safe_int(f) for f in (inst.get("frames") or [])]
    return min(frames) if frames else 10**9


def _label_number(label: str) -> int:
    m = re.search(r"\b(\d+)(?:\.\d+)?\b", str(label))
    return int(m.group(1)) if m else 10**9


def _safe_int(value, default: int = 10**9) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _renumber_final_instances(instances: list[dict], object_name: str) -> list[dict]:
    """Sort final count instances by first visible frame and relabel as object 1..N."""
    ordered = sorted(
        (dict(it) for it in instances),
        key=lambda it: (
            _first_frame(it),
            _label_number(it.get("label", "")),
            _safe_int(it.get("id")),
        ),
    )
    for idx, it in enumerate(ordered, start=1):
        old_label = str(it.get("label") or "")
        it["original_label"] = old_label
        it["final_index"] = idx
        it["label"] = f"{object_name} {idx}"
        for b in it.get("boxes_2d") or []:
            if isinstance(b, dict):
                b["label"] = it["label"]
    return ordered


def _save_calibration(results_3d: dict, object_name: str, cal: dict,
                      final_instances: list[dict]) -> None:
    """Persist the post-calibration instance set next to the annotated frames.

    Written so a later pass (reflection / skill distillation) can inspect exactly
    what the VLM changed and which boxes survived, without re-running anything.
    """
    ann_dir = results_3d.get("annotated_dir")
    if not ann_dir:
        return
    try:
        payload = {
            "object_name": object_name,
            "base_count": cal.get("base_count"),
            "adjusted_count": cal.get("adjusted_count"),
            "kept_ids": cal.get("kept_ids"),
            "applied": cal.get("applied"),
            "llm_raw": cal.get("raw"),
            "final_instances": final_instances,     # incl. per-frame boxes_2d
            "dropped_instances": [
                it for it in (results_3d.get("instances") or [])
                if int(it["id"]) not in set(cal.get("kept_ids") or [])
            ],
        }
        (Path(ann_dir) / "calibration.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - persistence must not break counting
        print(f"[instance_counting] could not save calibration.json: {e}")


def print_counting_summary(counting_result: dict, text_prompt: str) -> None:
    print(f'Scene has {counting_result["total_unique"]} unique "{text_prompt}" instances')
    print(f'Instance IDs: {counting_result["obj_id_list"]}')
    cal = counting_result.get("calibration")
    if cal:
        print(f'VLM calibration: {cal["base_count"]} -> {cal["adjusted_count"]}')
        for a in cal.get("applied", []):
            print(f'  - {a}')
