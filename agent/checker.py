"""Checker role for SpatialMem.

The Checker is a passive VLM verifier called by task tools after perception.
It does not plan new tools or compute geometry; it validates visual grounding
decisions and, when a required target has no SAM3 track, directly answers from
the original frames.
"""

from __future__ import annotations

import base64
import io
from typing import Any


CALIBRATE_SYSTEM = (
    "You are a meticulous visual verifier for an automated 3D object-counting "
    "pipeline. Be conservative: report ONLY errors you are highly confident about. "
    "Respond with valid JSON only."
)


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


LEGACY_BUILD_CALIBRATION_PROMPT_REFERENCE = '''
def build_calibration_prompt(object_name: str, instances: list[dict],
                             n_frames: int, prior_str: str,
                             checker_notes: str | None = None) -> str:
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
  "missed":   [{{"description": "where it is", "frames": [0], "confidence": 0.0}}],
  "spurious": [{{"instance": 1, "reason": "...", "confidence": 0.0}}],
  "merge":    [{{"instances": [1, 2], "reason": "...", "confidence": 0.0}}],
  "split":    [{{"instance": 1, "n_target": 2, "n_non_target": 0,
                "target_frame_groups": [[0, 1], [5, 6]],
                "non_target_frames": [8],
                "reason": "...", "confidence": 0.0}}],
  "reasoning": "one short paragraph"
}}"""
'''


def build_calibration_prompt(object_name: str, instances: list[dict],
                             n_frames: int, prior_str: str,
                             checker_notes: str | None = None) -> str:
    inst_lines = "\n".join(
        f"  - {it['label']}: color={it['color']}, appears in frames {it['frames']}, "
        f"3D size {it['bbox_size_m'][0]:.2f} x {it['bbox_size_m'][1]:.2f} x "
        f"{it['bbox_size_m'][2]:.2f} m"
        for it in instances) or "  (none - the pipeline found nothing)"

    return f'''GOAL: inspect the frames carefully, correct the pipeline when the visual evidence supports it, and count every "{object_name}" in this room as accurately as possible.

The {n_frames} images are consecutive frames sampled from ONE walkthrough video of the
SAME room. An automated pipeline (open-vocabulary segmentation + cross-frame tracking + 3D
clustering) has already produced a result. Every detected instance is drawn as a colored
bounding box with a small same-colored label "{object_name} N".

CURRENT RESULT - {len(instances)} instance(s):
{inst_lines}

Typical real-world size for "{object_name}" (agent memory priors): {prior_str}
{_checker_notes_block(checker_notes)}

WHAT COUNTS AS A "{object_name}" - take the category in its narrow, everyday sense:
  Count only standalone objects a person would plainly call a "{object_name}". Do NOT count
  look-alike parts of other objects. E.g. "door" = the room's doors only, NOT cabinet /
  wardrobe / oven / washing-machine doors.

Your job is to CHECK and CORRECT the current result from the images, not to defer to the
pipeline by default. Review ALL frames and actively look for count errors in these four categories:
  1. "missed"   - a real {object_name} that the pipeline never counted at all: it carries NO
                  colored box in ANY frame. Search carefully for such objects.
  2. "spurious" - a labeled instance that is not actually a {object_name}.
  3. "merge"    - two or more labeled instances are actually the SAME physical object seen in
                  different frames or viewpoints. If two tracks stay in the same room location
                  with the same surrounding context and landmarks, merge them.
  4. "split"    - one labeled instance actually covers multiple physical objects across frames.

DECISION PROCEDURE - inspect the actual images, not just the listed tracks.
For EACH labeled instance, look across all of its frames and ask:
"Across all frames of this one track, how many DISTINCT physical objects do its boxes actually land on?"
  - exactly 1 physical object, and it IS a real {object_name}
      -> leave it unchanged
  - exactly 1 physical object, and it is NOT a real {object_name}
      -> report "spurious"
      -> fill:
         * "instance": the track id number
         * "reason": why this track is not a real {object_name}
         * "confidence": your confidence in this correction
  - 2 or more DISTINCT physical objects across frames
      -> report "split"
      -> this means one track jumped across multiple objects and merged them incorrectly

For every "split", you MUST fill the fields precisely:
  - "instance": the track id number being split
  - "n_target": how many DISTINCT real {object_name} objects this track covers across all its frames
  - "n_non_target": how many DISTINCT non-target objects this track covers across all its frames
  - "target_frame_groups": one frame-index list for each real target object
      * the number of inner lists MUST equal n_target
      * each inner list must contain the frames where the box is on that one real target object
      * if n_target = 1, provide exactly one inner list
  - "non_target_frames": all frames where the box is on something that is NOT a real {object_name}
      * if n_non_target = 0, use []
      * if n_non_target > 0, this list should not be empty
  - "reason": explain briefly how the track jumps across different objects
  - "confidence": your confidence in this correction

CRITICAL RULES FOR "split" VS "spurious":
  - If a track contains ANY real {object_name} in any frame, do NOT call it "spurious".
  - Use "spurious" ONLY when the track contains NO real {object_name} in any frame at all.
  - Even if only ONE covered object is a real {object_name} and the rest are wrong objects,
    this is still "split", with n_target=1 and n_non_target=k.

Then compare DIFFERENT labels against each other and ask:
"Do these different track IDs actually point to the SAME physical {object_name}?"
  - If two labels appear in different frames but match the SAME {object_name} at the SAME room location and 
  surrounding environment, report "merge".
  - For "merge", fill:
      * "instances": the list of track id numbers that refer to the same real object
      * "reason": why these tracks correspond to the same object
      * "confidence": your confidence in this correction
  - If two visually similar objects sit in different places, keep them separate.
  - Two boxes visible in the SAME frame are always different physical objects, so never merge them.

Before finalizing your answer, also check for completely missed targets:
"Is there any real {object_name} visible in the frames that never receives any box in any frame at all?"
  - If YES, report "missed".
  - Use "missed" ONLY for a real target object that is never boxed in any frame.
  - If the same physical object is boxed in some other frame, it is already counted and must NOT be reported as missed.
  - For every "missed", fill:
      * "description": a short description of where the missed object is
      * "frames": the frame indices where this missed object is visible
      * "confidence": your confidence in this correction
  - If you find multiple never-boxed target objects, report each one as a separate entry in "missed".

CONFIDENCE GUIDANCE:
  Base confidence on visible evidence in the frames. You do NOT need impossible certainty.
  If the images give clear support for a correction, report it with an honest confidence.
  Do not suppress a visually well-supported correction merely because the pipeline said otherwise.

Respond with JSON ONLY, exactly this schema:
{{
  "missed":   [{{"description": "where it is", "frames": [0], "confidence": 0.0}}],
  "spurious": [{{"instance": 1, "reason": "...", "confidence": 0.0}}],
  "merge":    [{{"instances": [1, 2], "reason": "...", "confidence": 0.0}}],
  "split":    [{{"instance": 1, "n_target": 2, "n_non_target": 0,
                "target_frame_groups": [[0, 1], [5, 6]],
                "non_target_frames": [8],
                "reason": "...", "confidence": 0.0}}],
  "reasoning": "one short paragraph"
}}'''

def llm_calibrate_count(
    llm,
    object_name: str,
    annotated_frames: list[str],
    instances: list[dict],
    memory=None,
    conf_threshold: float = 0.7,
    max_side: int = 640,
    max_tokens: int = 2000,
    checker_notes: str | None = None,
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
                                         checker_notes=checker_notes),
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
        print(f"[CHECKER RAW] count/{object_name} ERROR: {out['error']}")
        return out
    print(f"[CHECKER RAW] count/{object_name}:\n{resp.get('_raw_response', resp)}")
    out["raw"] = resp

    alive = set(base_ids)
    applied: list[str] = []
    split_delta = 0

    for s in (resp.get("spurious") or []):
        if float(s.get("confidence", 0)) >= conf_threshold:
            gid = int(s.get("instance", 0)) - 1
            if gid in alive:
                alive.discard(gid)
                applied.append(f"spurious: dropped {object_name} {gid + 1} — {s.get('reason', '')}")

    for m in (resp.get("merge") or []):
        if float(m.get("confidence", 0)) >= conf_threshold:
            grp = sorted({int(x) - 1 for x in (m.get("instances") or [])} & alive)
            if len(grp) >= 2:
                for g in grp[1:]:
                    alive.discard(g)
                applied.append(f"merge: {[g + 1 for g in grp]} are one object — {m.get('reason', '')}")

    for s in (resp.get("split") or []):
        if float(s.get("confidence", 0)) >= conf_threshold and (int(s.get("instance", 0)) - 1) in alive:
            n_t = int(s.get("n_target", 2))
            n_nt = int(s.get("n_non_target", 0))
            split_delta += (n_t - 1)
            applied.append(f"split: {object_name} {int(s['instance'])} covers {n_t} real "
                           f"+ {n_nt} non-target ({n_t - 1:+d}) — {s.get('reason', '')}")

    n_missed = sum(1 for m in (resp.get("missed") or [])
                   if float(m.get("confidence", 0)) >= conf_threshold)
    if n_missed:
        applied.append(f"missed: +{n_missed}")

    out["adjusted_count"] = max(0, len(alive) + split_delta + n_missed)
    out["kept_ids"] = sorted(alive)
    out["applied"] = applied
    out["_conf_threshold"] = conf_threshold
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
    def __init__(self, llm, memory=None, checker_notes: str | None = None):
        self.llm = llm
        self.memory = memory
        self.checker_notes = str(checker_notes or "").strip()

    def set_checker_notes(self, checker_notes: str | None) -> None:
        self.checker_notes = str(checker_notes or "").strip()

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
        conf_threshold: float = 0.7,
    ) -> dict:
        frames = self._frames_for(results_3d, object_name)
        instances = self._instances_for(results_3d, object_name)
        cal = llm_calibrate_count(
            self.llm, object_name, frames, instances,
            memory=self.memory, conf_threshold=conf_threshold,
            checker_notes=self.checker_notes,
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
        return (f"Checker count calibration for {object_name}: "
                f"{cal['base_count']} -> {cal['adjusted_count']}; "
                f"{'; '.join(cal.get('applied', [])) or 'no corrections'}")

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
