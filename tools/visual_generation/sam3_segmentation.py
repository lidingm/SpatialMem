"""SAM3 text-prompted image segmentation tool (per-frame detection).

Uses SAM3 image model for per-frame open-vocabulary detection.
Cross-frame instance association is handled downstream by 3D clustering.

Follows the official SAM3 example notebook pattern:
  1. enable bfloat16 autocast globally
  2. build_sam3_image_model
  3. Sam3Processor.set_image → set_text_prompt → read state
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# Defaults from env vars; callers can pass explicit paths to the constructor.
DEFAULT_SAM3_REPO = Path(os.environ.get(
    "SAM3_REPO", "/home/zhouruofan/Training-Free/tool_model/sam3"))
DEFAULT_SAM3_CHECKPOINT = Path(os.environ.get(
    "SAM3_CHECKPOINT", "/home/zhouruofan/Training-Free/tool_model/sam3.1/sam3.1_multiplex.pt"))
DEFAULT_BPE_PATH = DEFAULT_SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def _ensure_sam3_importable(sam3_repo: Path) -> None:
    repo = str(sam3_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


class SAM3SegmentationTool:
    """SAM3 per-frame image segmentation tool with text prompts.

    Also exposes `segment_video_track` which uses the SAM3.1 video predictor
    to track instances across all frames simultaneously — much more accurate
    for object counting because temporal context disambiguates duplicates.
    The video predictor is built lazily on first use.
    """

    def __init__(
        self,
        sam3_repo: str | Path = DEFAULT_SAM3_REPO,
        checkpoint_path: str | Path | None = None,
        bpe_path: str | Path | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
    ) -> None:
        self.sam3_repo = Path(sam3_repo).expanduser()
        _ensure_sam3_importable(self.sam3_repo)

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe = str(bpe_path or DEFAULT_BPE_PATH)
        ckpt = str(checkpoint_path or DEFAULT_SAM3_CHECKPOINT)

        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        self.device = device
        self.model = build_sam3_image_model(
            bpe_path=bpe,
            device="cuda",
            checkpoint_path=ckpt,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=confidence_threshold,
        )
        self.confidence_threshold = confidence_threshold
        self._video_predictor = None  # lazy-initialised on first video call

    # ── Video tracking ────────────────────────────────────────────────────

    def _get_video_predictor(self):
        """Build the SAM3.1 multiplex video predictor (once, then cached)."""
        if self._video_predictor is not None:
            return self._video_predictor

        _ensure_sam3_importable(self.sam3_repo)
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        bpe = str(DEFAULT_BPE_PATH)
        ckpt = str(DEFAULT_SAM3_CHECKPOINT)

        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=ckpt,
            bpe_path=bpe,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
        # --- instance-counting tuning ----------------------------------------
        # Moderately looser than the official defaults for recall, but pulled
        # back from the earlier fully-permissive values (which admitted too many
        # spurious / fragmented tracks). Official defaults shown in parens.
        #
        # (1) Masklet confirmation: require this many consecutive detections
        #     before a track is confirmed/output. 1 = confirm on first detection
        #     (max recall — downstream 3D merge handles fragments). (default 3)
        predictor.model.masklet_confirmation_consecutive_det_thresh = 1
        # (2) Unmatch-removal: remove a track with >= this many unmatched frames
        #     in its hotstart window. Set == hotstart_delay (=15) so the removal
        #     condition is unsatisfiable — no track is ever dropped for being
        #     briefly undetected; the downstream 3D stage handles everything.
        predictor.model.hotstart_unmatch_thresh = predictor.model.hotstart_delay
        # (3) Same-instance judgment INSIDE SAM: mask-overlap thresholds for
        #     re-associating a detection to an existing tracklet / merging
        #     tracklets. Higher = stricter (harder to call two things the same
        #     instance). (defaults: assoc=0.1, trk_assoc=0.5)
        predictor.model.assoc_iou_thresh = 0.07
        predictor.model.trk_assoc_iou_thresh = 0.35
        # ----------------------------------------------------------------------

        # Flash Attention 3 (flash_attn_interface) may not be installed.
        # Patch every ViT Attention block to fall back to SDPA instead.
        for mod in predictor.model.modules():
            if hasattr(mod, "use_fa3"):
                mod.use_fa3 = False

        self._video_predictor = predictor
        return predictor

    def segment_video_track(
        self,
        frame_paths: Sequence[str | Path],
        text_prompt: str,
    ) -> dict:
        """Track instances across video frames using SAM3.1 video mode.

        Adds a text prompt on frame 0, propagates across all frames, and
        collects unique tracked instance IDs.  Compared to per-frame image
        segmentation + 3D clustering, this is more robust for object counting
        because temporal context handles occlusion and re-appearance.

        Args:
            frame_paths: ordered list of frame image paths (e.g. 32 frames).
            text_prompt: text description of the object to track (e.g. "chair").

        Returns:
            dict with keys:
              total_count   — int, number of unique instances tracked.
              unique_obj_ids — list[int], all obj_ids seen across the video.
              per_frame     — list[dict] (one per frame), each with:
                  obj_ids : list[int]
                  masks   : (N, H, W) bool ndarray  (original resolution)
                  boxes   : (N, 4) float32 ndarray  (xyxy pixel coords)
                  scores  : (N,)   float32 ndarray
        """
        predictor = self._get_video_predictor()

        # Load frames as PIL images (resource_path accepts list[PIL.Image])
        pil_frames = [Image.open(p).convert("RGB") for p in frame_paths]
        orig_w, orig_h = pil_frames[0].size  # pixel dimensions

        # Start session
        resp = predictor.handle_request({
            "type": "start_session",
            "resource_path": pil_frames,
        })
        session_id = resp["session_id"]

        n_frames = len(pil_frames)
        empty_frame = {
            "obj_ids": [],
            "masks":   np.empty((0, orig_h, orig_w), dtype=bool),
            "boxes":   np.empty((0, 4), dtype=np.float32),
            "scores":  np.empty((0,), dtype=np.float32),
        }

        try:
            # Register the text query on frame 0 (official SAM3 pattern). SAM3
            # runs FA detection on EVERY frame during propagate_in_video, so this
            # single add_prompt is all that's needed — objects that first appear
            # on later frames are still detected and tracked.
            resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt,
            })
            n_det_frame0 = int(len(resp["outputs"]["out_obj_ids"]))
            print(f"  [SAM3 video] add_prompt frame 0: {n_det_frame0} '{text_prompt}' detected")

            # Propagate across all frames, collecting per-frame instances.
            all_obj_ids: set[int] = set()
            per_frame: list[dict] = [None] * len(pil_frames)  # type: ignore

            for out in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
            }):
                fi   = out["frame_index"]
                outs = out["outputs"]

                # Outputs may be torch tensors or numpy arrays depending on the
                # SAM3 code path — normalise to numpy either way.
                def _np(x):
                    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

                obj_ids = _np(outs["out_obj_ids"]).tolist()          # list[int]
                scores  = _np(outs["out_probs"]).astype(np.float32)  # (N,)

                # Convert masks to numpy bool at original resolution
                masks_np = _np(outs["out_binary_masks"]).astype(bool)  # (N, H_model, W_model)

                # Convert normalised xywh → xyxy pixel coords
                if len(obj_ids) > 0:
                    bx = _np(outs["out_boxes_xywh"]).astype(np.float32)  # (N,4) xywh normed
                    cx = (bx[:, 0] + bx[:, 2] / 2) * orig_w
                    cy = (bx[:, 1] + bx[:, 3] / 2) * orig_h
                    hw = bx[:, 2] * orig_w / 2
                    hh = bx[:, 3] * orig_h / 2
                    boxes_xyxy = np.stack(
                        [cx - hw, cy - hh, cx + hw, cy + hh], axis=1
                    )
                else:
                    boxes_xyxy = np.empty((0, 4), dtype=np.float32)

                all_obj_ids.update(obj_ids)
                per_frame[fi] = {
                    "obj_ids": obj_ids,
                    "masks":   masks_np,
                    "boxes":   boxes_xyxy,
                    "scores":  scores,
                }

            # Fill any frames that received no output (no objects visible)
            per_frame = [f if f is not None else empty_frame for f in per_frame]

        finally:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })

        unique_ids = sorted(all_obj_ids)
        return {
            "total_count":    len(unique_ids),
            "unique_obj_ids": unique_ids,
            "per_frame":      per_frame,
        }

    def segment(
        self,
        image: str | Path | Image.Image,
        text_prompt: str,
        confidence_threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-frame text-prompted instance segmentation.

        Returns:
            dict with keys:
              masks  - (N, H, W) bool
              boxes  - (N, 4) float, xyxy pixel coords
              scores - (N,) float
        """
        if confidence_threshold is not None and confidence_threshold != self.confidence_threshold:
            self.processor.set_confidence_threshold(confidence_threshold)
            self.confidence_threshold = confidence_threshold

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        state = self.processor.set_image(image)
        self.processor.reset_all_prompts(state)
        state = self.processor.set_text_prompt(state=state, prompt=text_prompt)

        masks_tensor = state.get("masks")
        boxes_tensor = state.get("boxes")
        scores_tensor = state.get("scores")

        if masks_tensor is None or len(masks_tensor) == 0:
            h, w = np.array(image).shape[:2]
            return {"masks": np.empty((0, h, w), dtype=bool),
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "scores": np.empty((0,), dtype=np.float32)}

        masks = masks_tensor.squeeze(1).cpu().float().numpy().astype(bool)
        boxes = boxes_tensor.cpu().float().numpy()
        scores = scores_tensor.cpu().float().numpy()

        return {"masks": masks, "boxes": boxes, "scores": scores}

    def segment_frames(
        self,
        frame_paths: Sequence[str | Path],
        text_prompt: str,
        output_dir: str | Path | None = None,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Segment all frames independently, save results."""
        results = [
            self.segment(p, text_prompt, confidence_threshold)
            for p in frame_paths
        ]
        if output_dir is not None:
            save_segmentation_results(results, frame_paths, text_prompt, output_dir)
        return results


def save_segmentation_results(
    results: list[dict[str, np.ndarray]],
    frame_paths: Sequence[str | Path],
    text_prompt: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_masks, all_boxes, all_scores, all_counts = [], [], [], []
    for res in results:
        all_masks.append(res["masks"])
        all_boxes.append(res["boxes"])
        all_scores.append(res["scores"])
        all_counts.append(len(res["scores"]))

    total = sum(all_counts)
    np.savez_compressed(
        output_dir / "sam3_segmentation.npz",
        masks=np.concatenate(all_masks, axis=0) if total > 0 else np.empty((0,), dtype=bool),
        boxes=np.concatenate(all_boxes, axis=0) if total > 0 else np.empty((0, 4), dtype=np.float32),
        scores=np.concatenate(all_scores, axis=0) if total > 0 else np.empty((0,), dtype=np.float32),
        counts=np.array(all_counts, dtype=np.int32),
    )

    metadata = {
        "text_prompt": text_prompt,
        "frames": [str(p) for p in frame_paths],
        "per_frame_counts": all_counts,
        "total_detections": total,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def visualize_segmentation(
    frame_paths: Sequence[str | Path],
    results: list[dict[str, np.ndarray]],
    text_prompt: str,
    output_dir: str | Path | None = None,
    max_cols: int = 5,
) -> None:
    """Visualize per-frame detection results in a matplotlib grid and optionally save."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = len(frame_paths)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).ravel()

    cmap = plt.get_cmap("tab10")
    for i in range(n):
        ax = axes[i]
        ax.imshow(Image.open(frame_paths[i]).convert("RGB"))
        res = results[i]
        for j in range(len(res["scores"])):
            color = cmap(j % 10)
            x0, y0, x1, y1 = res["boxes"][j]
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x0, max(y0 - 4, 0), f'#{j} {res["scores"][j]:.2f}', color="white",
                    fontsize=9, fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.7, pad=2))
        ax.set_title(f'Frame {Path(frame_paths[i]).stem} | {len(res["scores"])} det')
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f'SAM3: "{text_prompt}"', fontsize=16)
    plt.tight_layout()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out / "segmentation_overview.png"), dpi=100, bbox_inches="tight")

    plt.show()
