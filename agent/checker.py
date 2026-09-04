"""Checker role for SpatialMem.

The Checker is a passive VLM verifier called by task tools after perception.
It does not plan new tools or compute geometry; it validates visual grounding
decisions and, when a required target has no SAM3 track, directly answers from
the original frames.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from typing import Any


CALIBRATE_SYSTEM = (
    "You are a meticulous visual verifier for an automated visual reasoning "
    "pipeline. Follow the task-specific user instructions carefully, inspect the "
    "provided visual evidence, and correct clear visual mistakes when the evidence "
    "supports it. Respond with valid JSON only."
)



def _maybe_print_count_checker_raw(object_name: str, resp: dict) -> None:
    # Notebook-only debugging aid; regular train/eval scripts do not load ipykernel.
    if 'ipykernel' not in sys.modules:
        return
    print(f'[CHECKER RAW] count/{object_name}:')
    print(json.dumps(resp, ensure_ascii=False, indent=2))


def _b64_image(path: str, max_side: int = 640) -> str:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _size_prior_str(memory, object_name: str) -> str:
    if memory is None:
        return "no prior available"
    try:
        prior = memory.get_object_size_prior(object_name)
    except Exception:
        return "no prior available"
    if not prior:
        return "no prior available"
    dims = [f"{k}={st['mean']:.2f}±{st.get('std', 0):.2f} m"
            for k, st in prior.items()
            if isinstance(st, dict) and "mean" in st]
    return ", ".join(dims) if dims else "no prior available"


def _checker_notes_block(checker_notes: str | None) -> str:
    notes = str(checker_notes or "").strip()
    if not notes or notes.lower() in ("none.", "none yet."):
        return ""
    return f"""
Relevant Checker experience from the active SKILL:
{notes}

Use these notes only when they apply to the current visual evidence. They are
past Checker-specific lessons, not a substitute for inspecting the frames.
"""


def _context_summary_block(context_summary: str | None) -> str:
    summary = str(context_summary or "").strip()
    if not summary:
        return ""
    return f"""
Previous pipeline evidence summary:
{summary}

Use this summary as auxiliary context about earlier tool results and decisions.
If it conflicts with the current annotated images or candidate list, trust the
visual evidence and the current candidate list.
"""


def validate_count_split(split: dict, frames_by_id: dict[int, set[int]],
                         allowed_ids: set[int] | None = None) -> tuple[bool, str, dict]:
    """Validate and normalize a count-checker split before it can affect count.

    Model output is 1-based; internal ids are 0-based. A valid split must be a
    complete partition of that track's frames into target groups plus optional
    non-target frames, with no invented, duplicated, overlapping, or missing frames.
    """
    parsed: dict[str, Any] = {}
    try:
        gid = int(split.get("instance", 0)) - 1
    except Exception:
        return False, "instance is not an integer", parsed
    parsed["instance_id"] = gid
    if gid not in frames_by_id:
        return False, f"instance {gid + 1} is not in CURRENT RESULT", parsed
    if allowed_ids is not None and gid not in allowed_ids:
        return False, f"instance {gid + 1} was already removed or merged", parsed

    original_frames = set(int(f) for f in (frames_by_id.get(gid) or set()))
    if not original_frames:
        return False, f"instance {gid + 1} has no listed frames to split", parsed

    try:
        n_target = int(split.get("n_target", 0))
        n_non_target = int(split.get("n_non_target", 0))
    except Exception:
        return False, "n_target/n_non_target must be integers", parsed
    parsed["n_target"] = n_target
    parsed["n_non_target"] = n_non_target
    if n_target < 1:
        return False, "n_target must be at least 1", parsed
    if n_non_target < 0:
        return False, "n_non_target must be non-negative", parsed

    raw_groups = split.get("target_frame_groups") or []
    if not isinstance(raw_groups, list):
        return False, "target_frame_groups must be a list", parsed
    if len(raw_groups) != n_target:
        return False, "target_frame_groups must contain exactly n_target groups", parsed

    target_frames: set[int] = set()
    norm_groups: list[list[int]] = []
    for gi, group in enumerate(raw_groups, start=1):
        if not isinstance(group, list) or not group:
            return False, f"target frame group {gi} is empty or not a list", parsed
        try:
            frames = [int(f) for f in group]
        except Exception:
            return False, f"target frame group {gi} contains a non-integer frame", parsed
        if len(frames) != len(set(frames)):
            return False, f"target frame group {gi} repeats a frame", parsed
        fset = set(frames)
        invented = sorted(fset - original_frames)
        if invented:
            return False, f"target frame group {gi} uses frames not in instance {gid + 1}: {invented}", parsed
        overlap = sorted(target_frames & fset)
        if overlap:
            return False, f"frames assigned to multiple target groups: {overlap}", parsed
        target_frames |= fset
        norm_groups.append(sorted(fset))

    raw_non_target = split.get("non_target_frames") or []
    if not isinstance(raw_non_target, list):
        return False, "non_target_frames must be a list", parsed
    try:
        non_target_frames = [int(f) for f in raw_non_target]
    except Exception:
        return False, "non_target_frames contains a non-integer frame", parsed
    if len(non_target_frames) != len(set(non_target_frames)):
        return False, "non_target_frames repeats a frame", parsed
    non_target = set(non_target_frames)
    invented = sorted(non_target - original_frames)
    if invented:
        return False, f"non_target_frames uses frames not in instance {gid + 1}: {invented}", parsed
    overlap = sorted(target_frames & non_target)
    if overlap:
        return False, f"frames assigned to both target and non-target: {overlap}", parsed
    if n_non_target == 0 and non_target:
        return False, "n_non_target is 0 but non_target_frames is non-empty", parsed
    if n_non_target > 0 and not non_target:
        return False, "n_non_target is positive but non_target_frames is empty", parsed

    assigned = target_frames | non_target
    missing = sorted(original_frames - assigned)
    if missing:
        return False, f"some frames of instance {gid + 1} are unassigned: {missing}", parsed
    extra = sorted(assigned - original_frames)
    if extra:
        return False, f"split uses nonexistent frames for instance {gid + 1}: {extra}", parsed

    parsed.update({
        "target_frame_groups": norm_groups,
        "non_target_frames": sorted(non_target),
        "original_frames": sorted(original_frames),
    })
    return True, "", parsed


LEGACY_BUILD_CALIBRATION_PROMPT_REFERENCE = '''
def build_calibration_prompt(object_name: str, instances: list[dict],
                             n_frames: int, prior_str: str,
                             checker_notes: str | None = None,
                             context_summary: str | None = None) -> str:
    inst_lines = "\n".join(
        f"  - {it['label']}: color={it['color']}, appears in frames {it['frames']}, "
        f"3D size {it['bbox_size_m'][0]:.2f} x {it['bbox_size_m'][1]:.2f} x "
        f"{it['bbox_size_m'][2]:.2f} m"
        for it in instances) or "  (none - the pipeline found nothing)"

    return f"""GOAL: find every "{object_name}" in this room and count them EXACTLY.

The {n_frames} images are consecutive frames sampled from ONE walkthrough video of the
SAME room. An automated pipeline (open-vocabulary segmentation + cross-frame tracking + 3D
clustering) has already produced a result. Every detected instance is drawn as a colored
bounding box with a small same-colored label "{object_name} N".

CURRENT RESULT - {len(instances)} instance(s):
{inst_lines}

Typical real-world size for "{object_name}" (agent memory priors): {prior_str}
{_checker_notes_block(checker_notes)}

WHAT COUNTS AS A "{object_name}" - take the category in its NARROW, everyday sense:
  Count only standalone objects a person would plainly call a "{object_name}". Do NOT count
  look-alike parts of other objects. E.g. "door" = the room's doors only, NOT cabinet /
  wardrobe / oven / washing-machine doors. 
  The segmenter is open-vocabulary and readily latches onto such look-alikes - catching
  them is a main part of your job.

Review ALL frames and report ONLY HIGH-CONFIDENCE corrections in these four categories:
  1. "missed"   - a real {object_name} that the pipeline never counted AT ALL: it carries NO
                  colored box in ANY of the frames.
                  CRITICAL - check the other frames before reporting. If the SAME physical
                  object does carry a box in some OTHER frame (the tracker merely lost it in
                  the frame you happen to be looking at), it is ALREADY counted: do NOT report
                  it, that would double-count. A gap of a few frames is normal and harmless.
                  Only report an object that is boxed NOWHERE, in no frame at all.
                  But when such an object does exist, you MUST report it - the pipeline
                  systematically UNDER-counts crowded scenes. Whenever the current result
                  already lists MANY instances (say more than 4), scan the frames deliberately
                  for extra {object_name}s that never received a box: omissions are most
                  likely precisely there. Report each never-boxed object as its OWN entry,
                  and in "frames" list the frames where you can see it.
  2. "spurious" - a labeled instance that is DEFINITELY not a {object_name} but is clearly
                  some OTHER identifiable object, in EVERY frame it appears in.
                  Report this ONLY if you are 100% certain - you must be able to say what
                  the object actually IS. If you are even slightly unsure what it is, DO NOT
                  report it. Deleting a real {object_name} is far worse than leaving a
                  doubtful one in.
  3. "merge"    - two or more labeled instances that are in fact the SAME physical object
                  seen from different viewpoints (count is inflated). This typically happens
                  when the camera pans away and later comes back: the tracker loses the object
                  and re-registers it under a NEW label, so one object gets counted twice.
                  HOW TO DECIDE "one object seen twice" vs "two similar objects": do NOT rely
                  on the object's own appearance - a room often contains several identical
                  items. Instead check whether the SURROUNDING CONTEXT is consistent: the
                  neighbouring furniture, the wall / window / door / corner behind it, the
                  floor pattern, and its spatial relation to those landmarks.
                    * same object  -> its surroundings and its position in the room stay
                                      consistent across the two labels' frames;
                    * two objects  -> they sit in different parts of the room next to
                                      different landmarks, even if they look identical.
  4. "split"    - ONE labeled instance actually lumps together MORE THAN ONE physical object
                  (the tracker merged them under a single label, so the count is deflated).
                  Report how many of those objects are genuinely a "{object_name}" (n_target)
                  and how many are something else (n_non_target, which must NOT be counted).
                  The count then changes by (n_target - 1): the one label is replaced by
                  n_target real objects; non-target pieces are simply discarded.
                  Example: label "{object_name} 2" actually covers 3 separate objects -
                  2 real {object_name}s and 1 other thing -> n_target=2, n_non_target=1 -> +1.
                  ALSO fill "target_frame_groups": one list of frames per real {object_name}
                  (so 2 real objects -> 2 groups), and "non_target_frames": the frames where
                  the box sits on something that is not a {object_name}. These let the boxes
                  be split up correctly; without them only the count can be fixed.

DECISION PROCEDURE - for EACH labeled instance, look at ALL the frames it appears in and ask:
"how many DISTINCT physical objects do its boxes land on across those frames?"
  - exactly 1 object, and it IS a {object_name}       -> no correction, leave it alone
  - exactly 1 object, and it is NOT a {object_name}   -> "spurious"
  - 2 or more DISTINCT objects (the tracker jumped)   -> "split", reporting
        n_target     = how many of those objects are real {object_name}s
        n_non_target = how many of them are not
  CRITICAL: use "split" even when only ONE of the objects is a real {object_name}
  (n_target=1, n_non_target=k) - that keeps the real one and drops the rest.
  Do NOT report "spurious" merely because the track is messy, or because the FIRST frames
  show a non-target object: that would wrongly throw away a real {object_name} that the
  same label covers in later frames. "spurious" is ONLY for a label that contains NO real
  {object_name} in any frame at all.
Meanwhile, cross-reference all labeled instances and ask:
"which distinct labels actually land on the SAME physical {object_name} across their respective frames?"
  - 2 or more labels, and their surrounding landmarks and room positions stay consistent -> "merge"
  CRITICAL: ensure these labels never appear together in the SAME frame; if they do, do NOT merge.

CONFIDENCE CALIBRATION (do NOT inflate - most reported confidences are far too high):
  Base the number on EVIDENCE you can actually point to, not on how plausible your story sounds.
    0.9-1.0 : you can name at least TWO frames in which the object is large, sharp and
              unambiguous, and no other reading is reasonable.
    0.7-0.9 : clear evidence in at least one good frame, and no plausible alternative reading.
    < 0.7   : anything you are inferring, guessing, or judging from a box that is small,
              blurry, cropped or partially occluded; or where another reading is possible.
  Low-confidence corrections are DISCARDED, so if you are unsure just give a low confidence.
  The pipeline already used 3D geometry and multi-frame tracking that you cannot fully see.
  When your reading disagrees with the pipeline and the evidence is not decisive, the
  PIPELINE is more likely right - prefer reporting NO correction. Reporting nothing is a
  perfectly good answer and is much better than a confident mistake.

Hard rules:
  - Two boxes visible in the SAME frame are ALWAYS different physical objects - never merge them.
  - Use the size prior only as a sanity check for "spurious" (e.g. wildly implausible size),
    not as your main evidence.
  - If you are not highly confident about a correction, DO NOT report it. Empty lists are a
    perfectly good answer.

Respond with JSON ONLY, exactly this schema:
{{
  "missed":   [{{"description": "where it is", "frames": [0]}}],
  "spurious": [{{"instance": 1, "reason": "..."}}],
  "merge":    [{{"instances": [1, 2], "reason": "..."}}],
  "split":    [{{"instance": 1, "n_target": 2, "n_non_target": 0,
                "target_frame_groups": [[0, 1], [5, 6]],
                "non_target_frames": [8],
                "reason": "..."}}],
  "reasoning": "one short paragraph"
}}"""
'''


def build_calibration_prompt(object_name: str, instances: list[dict],
                             n_frames: int, prior_str: str,
                             checker_notes: str | None = None,
                             context_summary: str | None = None) -> str:
    inst_lines = "\n".join(
        f"  - {it['label']}: color={it['color']}, appears in frames {it['frames']}, "
        f"3D size {it['bbox_size_m'][0]:.2f} x {it['bbox_size_m'][1]:.2f} x "
        f"{it['bbox_size_m'][2]:.2f} m"
        for it in instances) or "  (none - the pipeline found nothing)"

    return f'''GOAL: review the annotated frames and correct the pipeline's instance count for "{object_name}".

The {n_frames} images are consecutive sampled frames from ONE walkthrough of the SAME room.
The pipeline has already detected candidate instances using segmentation, tracking, and 3D clustering.
Each candidate is drawn with a colored box and label like "{object_name} N".

CURRENT RESULT - {len(instances)} candidate instance(s):
{inst_lines}
{_checker_notes_block(checker_notes)}
{_context_summary_block(context_summary)}

Use the common everyday understanding of the category "{object_name}". Count objects
that people would normally recognize as members, common subtypes, or normal functional variants
of that category. For example, "table" includes dining tables, coffee tables, side tables, desk,
bedside/night tables, and TV/console tables, but not plant stands, shelves, racks, or counters.
For "door", room, passage, or entrance doors count, but cabinet, wardrobe, appliance, or
furniture-panel doors do not. Report "spurious" only when the labeled object is clearly not
the target category in ordinary usage; if it is a plausible common subtype or reasonable
borderline use of the category name, keep it. Use visual evidence together with everyday
common sense: For example, if the current result is implausibly high for large furniture in one
room (e.g. many sofas or tables), carefully check for duplicate tracks to merge and clear
non-target detections to remove.

Return CORRECTIONS ONLY. If a labeled track is already correct, omit it from all correction lists.
Do not use any field to confirm normal tracks.

The goal is to count physical object instances, not perfectly segmented full-object boxes.
Do NOT mark a label as "spurious" merely because the box shows only part of a real target
object, is cropped, or covers a surface/part of that target.

Correction types and fields:
  - "spurious": one labeled instance is clearly not {object_name} in ordinary usage. Use this
    only for clear non-targets; keep plausible common subtypes and reasonable borderline cases.
    Fields: "instance", "reason".
  - "merge": two or more labels are the SAME physical {object_name} and should count once;
    use it only when appearance, shape, and surrounding room context are all consistently the same.
    Use merge only when you are very certain the labels must be the same object; be especially cautious when there are few objects.
    Fields: "instances", "reason".
  - "split": one label covers multiple DIFFERENT physical objects across that label's OWN frames.
    Use split only for a tracker jump within a single label; if the same object reappears with
    another label after a camera-angle/viewpoint change, that is "merge", not "split".
    Fields: "instance", "n_target", "n_non_target", "target_frame_groups", "non_target_frames", "reason".
  - "missed": a real {object_name} is visible but never boxed in any frame.
    Fields: "description", "frames".

Pipeline failure modes to check carefully:
  - Because frames are sparsely sampled, tracking can be discontinuous: the SAME physical
    object may disappear for a while and later receive a new label. Check possible merges, but
    merge only when the labels point to the same room location with consistent surrounding
    landmarks; do not merge merely because objects look alike or appear in non-overlapping frames.
  - Because SAM/open-vocabulary segmentation can use a broad category meaning, some boxes may
    cover related but non-target objects. Remove clear non-target labels with "spurious", but
    keep common subtypes, normal functional variants, and reasonable borderline cases.

Decision procedure:
  1. Check each labeled track across only the frames listed for that track.
     - If it contains exactly one real {object_name}, leave it unchanged and output nothing for it.
     - If it contains exactly one object that is not reasonably recognized as {object_name}, report "spurious".
     - If this same track jumps between multiple DIFFERENT physical objects, report "split".
       Do not split temporal chunks of the same physical object.
  2. Compare different labels with each other.
     Check for possible merges, but require strong evidence that the labels refer to the same
     physical target object: same room location, consistent surrounding landmarks, no same-frame
     co-occurrence, and no clear conflict in stable visual attributes such as color/material/shape.
     Similar-looking objects should remain separate unless the spatial context makes the same-object
     interpretation clear. If matching labels are the same non-target object, report them as
     "spurious" instead. Keep labels separate when they are clearly different target objects.
  3. Check for completely missed targets.
     Report "missed" only for a real target that has no colored box in any frame.

Parameter validity rules:
  - Use ONLY the candidate labels listed in CURRENT RESULT. The instance id is the number in
    the visible label, e.g. "{object_name} 3" means instance=3. Do not invent ids, and do not
    use raw SAM track ids or frame numbers as instance ids.
  - For "spurious", "instance" must be one listed candidate id.
  - For "merge", "instances" must contain two or more listed candidate ids, and every id must
    be a real {object_name}. Never merge labels that appear in the same frame, labels also reported
    as "spurious", invented ids, or clearly different-looking objects.
  - For "split", "instance" must be one listed candidate id. Use split ONLY when that label's
    boxes land on multiple distinct complete objects; different parts, surfaces, bedding,
    panels, temporal chunks, camera-angle changes, or visible regions of the SAME object are not split.
  - For "split", all "target_frame_groups" and "non_target_frames" must use only frames listed
    for that same candidate in CURRENT RESULT. Never include frames belonging only to another label.
  - For "split", "target_frame_groups" must contain exactly n_target non-empty frame lists, one
    per distinct real target object. If you cannot assign legal separate frame groups for separate
    complete objects, do not split.
  - If a second physical object already has its own label, do not add that other label's frames
    to a split. Use "merge" if the labels are the same object, or leave them separate if they
    are different objects.
  - If your reason says the same object was assigned separate labels, or was split because of
    viewpoint/camera-angle changes, the operation must be "merge" instead of "split".
  - For "missed", use frame indices where the never-boxed target is visible.
  - Each listed candidate id should appear in at most one correction type. Choose the single
    operation that best describes the error. If the reason identifies an object as non-target,
    that id belongs in "spurious", not "merge".
  - Keep every "reason" to one short sentence. Do not repeat the same phrase or frame list.

Respond with JSON ONLY, exactly this schema:
{{
  "missed":   [{{"description": "where it is", "frames": [0]}}],
  "spurious": [{{"instance": 1, "reason": "..."}}],
  "merge":    [{{"instances": [1, 2], "reason": "..."}}],
  "split":    [{{"instance": 1, "n_target": 2, "n_non_target": 1,
                "target_frame_groups": [[0, 1], [5, 6]],
                "non_target_frames": [8],
                "reason": "..."}}],
  "reasoning": "one short paragraph"
}}'''

def llm_calibrate_count(
    llm,
    object_name: str,
    annotated_frames: list[str],
    instances: list[dict],
    memory=None,
    max_side: int = 640,
    max_tokens: int = 2000,
    checker_notes: str | None = None,
    context_summary: str | None = None,
) -> dict:
    base_ids = [int(it["id"]) for it in instances]
    out = {"base_count": len(base_ids), "adjusted_count": len(base_ids),
           "kept_ids": list(base_ids), "applied": [], "raw": None, "error": None}
    if llm is None or not annotated_frames:
        out["error"] = "no llm or no annotated frames"
        return out

    prior_str = _size_prior_str(memory, object_name)
    content: list[dict] = [{
        "type": "text",
        "text": build_calibration_prompt(object_name, instances,
                                         len(annotated_frames), prior_str,
                                         checker_notes=checker_notes,
                                         context_summary=context_summary),
    }]
    for i, fp in enumerate(annotated_frames):
        try:
            content.append({"type": "text", "text": f"frame {i}:"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_b64_image(fp, max_side)}"}})
        except Exception:
            pass

    try:
        resp = llm.chat_json(
            [{"role": "system", "content": CALIBRATE_SYSTEM},
             {"role": "user", "content": content}],
            max_tokens=max_tokens,
            include_raw=True,
        )
    except Exception as e:  # noqa: BLE001 - calibration must never break counting
        out["error"] = f"llm call failed: {e}"
        return out
    out["raw"] = resp
    _maybe_print_count_checker_raw(object_name, resp)

    alive = set(base_ids)
    applied: list[str] = []
    ignored: list[str] = []
    split_delta = 0
    frames_by_id = {
        int(it["id"]): {int(f) for f in (it.get("frames") or [])}
        for it in instances
    }

    def _cooccurs(group: list[int]) -> bool:
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if frames_by_id.get(a, set()) & frames_by_id.get(b, set()):
                    return True
        return False

    for s in (resp.get("spurious") or []):
        gid = int(s.get("instance", 0)) - 1
        if gid in alive:
            alive.discard(gid)
            applied.append(f"spurious: dropped {object_name} {gid + 1} - {s.get('reason', '')}")

    for m in (resp.get("merge") or []):
        grp = sorted({int(x) - 1 for x in (m.get("instances") or [])} & alive)
        if len(grp) >= 2 and not _cooccurs(grp):
            for g in grp[1:]:
                alive.discard(g)
            applied.append(f"merge: {[g + 1 for g in grp]} are one object - {m.get('reason', '')}")

    for s in (resp.get("split") or []):
        ok, why, parsed = validate_count_split(s, frames_by_id, allowed_ids=alive)
        if not ok:
            label = s.get("instance", "?") if isinstance(s, dict) else "?"
            ignored.append(f"split {object_name} {label}: {why}")
            continue
        gid = int(parsed["instance_id"])
        n_t = int(parsed["n_target"])
        n_nt = int(parsed["n_non_target"])
        split_delta += (n_t - 1)
        applied.append(f"split: {object_name} {gid + 1} covers {n_t} real "
                       f"+ {n_nt} non-target ({n_t - 1:+d}) - {s.get('reason', '')}")

    n_missed = len(resp.get("missed") or [])
    if n_missed:
        applied.append(f"missed: +{n_missed}")

    out["adjusted_count"] = max(0, len(alive) + split_delta + n_missed)
    out["kept_ids"] = sorted(alive)
    out["applied"] = applied
    out["ignored"] = ignored
    return out


def _answer_format_instruction(answer_format: str) -> str:
    return {
        "fill": "Answer with a SINGLE NUMBER only. No units, no words.",
        "select": (
            "Answer with EXACTLY ONE LETTER: A, B, C, or D. Nothing else. "
            "First derive the semantic answer, compare it against every option text in the question, "
            "and return the letter whose option text matches. For ordering questions, match the full comma-separated order exactly."
        ),
        "judge": "Answer with exactly 'Yes' or 'No'.",
        "sentence": "Answer in one complete sentence.",
    }.get(answer_format, "Answer in the required format.")


class Checker:
    def __init__(self, llm, memory=None, checker_notes: str | None = None,
                 context_summary: str | None = None):
        self.llm = llm
        self.memory = memory
        self.checker_notes = str(checker_notes or "").strip()
        self.context_summary = str(context_summary or "").strip()

    def set_checker_notes(self, checker_notes: str | None) -> None:
        self.checker_notes = str(checker_notes or "").strip()

    def set_context_summary(self, context_summary: str | None) -> None:
        self.context_summary = str(context_summary or "").strip()

    @staticmethod
    def _instances_for(results_3d: dict, object_name: str) -> list[dict]:
        by_cat = results_3d.get("instances_by_category") or {}
        if object_name in by_cat:
            return list(by_cat[object_name])
        return [it for it in (results_3d.get("instances") or [])
                if it.get("category") == object_name]

    @staticmethod
    def _frames_for(results_3d: dict, object_name: str) -> list[str]:
        by_cat = results_3d.get("annotated_frames_by_category") or {}
        return list(by_cat.get(object_name) or results_3d.get("annotated_frames") or [])

    def calibrate_count(
        self,
        object_name: str,
        results_3d: dict,
    ) -> dict:
        frames = self._frames_for(results_3d, object_name)
        instances = self._instances_for(results_3d, object_name)
        cal = llm_calibrate_count(
            self.llm, object_name, frames, instances,
            memory=self.memory,
            checker_notes=self.checker_notes,
            context_summary=self.context_summary,
        )
        cal["summary"] = self.count_summary(object_name, cal)
        return cal

    def locate_single_object(
        self,
        object_name: str,
        results_3d: dict,
        max_side: int = 640,
        max_tokens: int = 1000,
    ) -> dict:
        instances = self._instances_for(results_3d, object_name)
        frames = self._frames_for(results_3d, object_name)
        out = {"object": object_name, "instance_id": None, "instance_label": None,
               "keep_frames": [], "dropped_frames": [], "reason": "",
               "raw": None, "error": None}
        if self.llm is None or not frames:
            out["error"] = "no llm or no annotated frames"
            out["summary"] = self.location_summary(out)
            return out
        if not instances:
            out["error"] = "no candidate"
            out["summary"] = self.location_summary(out)
            return out

        lines = "\n".join(
            f"  - {c['label']}: color={c['color']}, boxed in frames {c['frames']}, "
            f"3D size {c['bbox_size_m'][0]:.2f} x {c['bbox_size_m'][1]:.2f} x "
            f"{c['bbox_size_m'][2]:.2f} m"
            for c in instances)

        sys = "You are a precise visual grounding checker. Respond with valid JSON only."
        txt = f"""This room contains EXACTLY ONE target "{object_name}" for the downstream spatial task.

The images are consecutive frames from ONE walkthrough of the SAME room. An automated
pipeline drew colored boxes only for candidate "{object_name}" tracks, each with a
small same-colored label like "{object_name} 1".

Candidate tracks:
{lines}
{_checker_notes_block(self.checker_notes)}
{_context_summary_block(self.context_summary)}

Pick the ONE track that is genuinely the target "{object_name}". Then split that picked
track's frames into:
  keep_frames    : frames where the box clearly sits on the real target object.
  discard_frames : frames where you are HIGHLY CERTAIN the box is on the WRONG thing.
Discard ONLY when certain. If in any doubt, KEEP the frame.
HARD RULE: if the picked track is boxed in only ONE frame, you may NOT discard it.

Respond with JSON ONLY:
{{
  "chosen_instance": "<label or null>",
  "keep_frames": [],
  "discard_frames": [],
  "reason": "short reason"
}}"""

        content: list[dict[str, Any]] = [{"type": "text", "text": txt}]
        for i, fp in enumerate(frames):
            try:
                content.append({"type": "text", "text": f"frame {i}:"})
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{_b64_image(fp, max_side)}"}})
            except Exception:
                pass

        try:
            resp = self.llm.chat_json(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            best = max(instances, key=lambda c: c.get("score", 0))
            out.update({
                "instance_id": int(best["id"]),
                "instance_label": best["label"],
                "keep_frames": list(best["frames"]),
                "dropped_frames": [],
                "reason": f"checker error ({e}); fell back to top-score candidate",
                "error": str(e),
            })
            out["summary"] = self.location_summary(out)
            return out

        out["raw"] = resp
        label = str((resp or {}).get("chosen_instance") or "").strip()
        chosen = next((c for c in instances if c["label"] == label), None)
        if chosen is None:
            chosen = max(instances, key=lambda c: c.get("score", 0))
        tf = {int(f) for f in chosen["frames"]}
        keep = {int(f) for f in (resp or {}).get("keep_frames", [])} & tf
        discard = {int(f) for f in (resp or {}).get("discard_frames", [])} & tf
        if len(tf) <= 1:
            keep, discard = set(tf), set()
        if not keep:
            keep = tf - discard
        out.update({
            "instance_id": int(chosen["id"]),
            "instance_label": chosen["label"],
            "keep_frames": sorted(keep),
            "dropped_frames": sorted(tf - keep),
            "reason": str((resp or {}).get("reason", "")),
        })
        out["summary"] = self.location_summary(out)
        return out

    def answer_from_raw_sample(
        self,
        question: str,
        frame_paths: list[str],
        answer_format: str,
        reason: str,
        max_side: int = 640,
        max_tokens: int = 1200,
    ) -> dict:
        out = {"answer": "", "reasoning": "", "confidence": 0.0,
               "reason": reason, "raw": None, "error": None}
        if self.llm is None:
            out["error"] = "no llm"
            out["summary"] = self.direct_answer_summary(out)
            return out

        sys = "You are a visual spatial reasoning checker. Respond with valid JSON only."
        txt = f"""A required target object had zero SAM3 tracks, so the geometry pipeline cannot be trusted.
Answer the original question directly from the raw input frames.

Question:
{question}

Required answer format: {_answer_format_instruction(answer_format)}
{_checker_notes_block(self.checker_notes)}
{_context_summary_block(self.context_summary)}

Respond with JSON ONLY:
{{
  "answer": "final answer in the required format",
  "reasoning": "brief visual reasoning",
  "confidence": 0.0
}}"""
        content: list[dict[str, Any]] = [{"type": "text", "text": txt}]
        for i, fp in enumerate(frame_paths):
            try:
                content.append({"type": "text", "text": f"raw frame {i}:"})
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{_b64_image(fp, max_side)}"}})
            except Exception:
                pass
        try:
            resp = self.llm.chat_json(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            out["error"] = str(e)
            out["summary"] = self.direct_answer_summary(out)
            return out
        out.update({
            "answer": str(resp.get("answer", "")),
            "reasoning": str(resp.get("reasoning", "")),
            "confidence": float(resp.get("confidence", 0.0) or 0.0),
            "raw": resp,
        })
        out["summary"] = self.direct_answer_summary(out)
        return out

    @staticmethod
    def count_summary(object_name: str, cal: dict) -> str:
        if cal.get("error"):
            return f"Checker count calibration for {object_name}: unavailable ({cal['error']})"
        lines = [f"Checker count calibration for {object_name}: {cal['base_count']} -> {cal['adjusted_count']}"]
        applied = cal.get("applied") or []
        if applied:
            lines.extend(f"- {a}" for a in applied)
        else:
            lines.append("- no corrections")
        ignored = cal.get("ignored") or []
        lines.extend(f"- ignored invalid correction: {a}" for a in ignored[:3])
        return "\n".join(lines)

    @staticmethod
    def location_summary(loc: dict) -> str:
        if loc.get("instance_id") is None:
            return f"Checker location for {loc.get('object')}: no candidate ({loc.get('error') or 'not selected'})"
        return (f"Checker location for {loc['object']}: chose {loc['instance_label']} "
                f"keep={loc['keep_frames']} dropped={loc['dropped_frames']}"
                + (f" reason={loc['reason']}" if loc.get("reason") else ""))

    @staticmethod
    def direct_answer_summary(ans: dict) -> str:
        if ans.get("error"):
            return f"Checker raw-sample direct answer failed: {ans['error']}"
        return (f"Checker raw-sample direct answer: {ans.get('answer')!r} "
                f"confidence={float(ans.get('confidence', 0.0)):.2f}; "
                f"reason={ans.get('reason')}")
