"""Run Depth Anything 3 Giant for geometry, BEV, and local NVS tools.

The class in this module is meant to be loaded once and reused across many
SpatialMem samples. A typical flow is:

1. instantiate ``DA3GeometryTool`` once, which loads DA3-Giant on GPU;
2. call ``infer`` for one sample, with 3DGS enabled by default;
3. call ``make_bev`` or ``render_nvs`` on the returned prediction as needed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


# Defaults come from environment variables; callers can pass explicit paths
# to the tool constructor. Values here are placeholders that should NOT be
# committed pointing at any particular user's home directory.
DEFAULT_DA3_REPO = Path(os.environ.get("DA3_REPO", "/opt/models/depth-anything-3"))
DEFAULT_MODEL_DIR = Path(os.environ.get("DA3_MODEL_DIR", "/opt/models/DA3NESTED-GIANT-LARGE-1.1"))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Movement = Literal["yaw_left", "yaw_right", "pitch_up", "pitch_down", "move_left", "move_right", "move_up", "move_down", "move_forward", "move_backward"]


@contextlib.contextmanager
def _quiet_da3_output():
    """Suppress verbose DA3 library logs during batch runs."""
    if os.environ.get("SPATIALMEM_QUIET_DA3", "1") in {"0", "false", "False"}:
        yield
        return

    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                yield
            finally:
                logging.disable(previous_disable)


def _ensure_da3_importable(da3_repo: Path) -> None:
    src_dir = da3_repo / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def collect_image_paths(inputs: Sequence[str]) -> list[str]:
    """Expand image files/directories into a sorted frame path list."""
    paths: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(
                p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image suffix: {path}")
            paths.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

    if not paths:
        raise ValueError("No input frames were found.")
    return [str(p) for p in paths]


def save_depth_visualizations(depth: np.ndarray, output_dir: Path) -> None:
    """Save simple per-frame depth visualizations as uint8 PNG files."""
    import cv2

    vis_dir = output_dir / "depth_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, depth_map in enumerate(depth):
        finite = np.isfinite(depth_map)
        valid = finite & (depth_map > 0)
        if not np.any(valid):
            depth_u8 = np.zeros(depth_map.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(depth_map[valid], [2, 98])
            if hi <= lo:
                hi = lo + 1e-6
            norm = np.clip((depth_map - lo) / (hi - lo), 0, 1)
            depth_u8 = (norm * 255).astype(np.uint8)
            depth_u8[~valid] = 0
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(vis_dir / f"{idx:04d}.png"), colored)


def _to_numpy_scalar(value: object, default: float = np.nan) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=np.float32)
    detach = getattr(value, "detach", None)
    item = getattr(value, "item", None)
    if callable(detach):
        value = value.detach().cpu().item()
    elif callable(item):
        value = item()
    if isinstance(value, (str, bytes, dict, list, tuple)):
        return np.array(str(value))
    return np.array(value)


def _resolve_device(device: str):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"{device} was requested but torch.cuda.is_available() is False. "
            "Check the active conda environment, CUDA driver, and GPU visibility."
        )
    torch_device = torch.device(device)
    print(f"Using device: {torch_device}")
    if torch_device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(torch_device)}")
    return torch_device


def _as_homogeneous_np(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=ext.dtype)
        out[:3, :4] = ext
        return out
    raise ValueError(f"Expected extrinsic shape (3, 4) or (4, 4), got {ext.shape}")


def _save_rgb_image(image: np.ndarray, path: str | Path) -> None:
    from PIL import Image

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def prediction_to_point_cloud(
    prediction,
    use_conf: bool = False,
    conf_percentile: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project every valid depth pixel into a colored world-space point cloud."""
    from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors

    conf = prediction.conf if use_conf else None
    if use_conf and conf is not None:
        conf_values = conf[np.isfinite(conf)]
        conf_thr = float(np.percentile(conf_values, conf_percentile)) if conf_values.size else -np.inf
    else:
        conf_thr = -np.inf

    points, colors = _depths_to_world_points_with_colors(
        depth=prediction.depth,
        K=prediction.intrinsics,
        ext_w2c=prediction.extrinsics,
        images_u8=prediction.processed_images,
        conf=conf,
        conf_thr=conf_thr,
    )
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def save_prediction_outputs(
    prediction,
    frame_paths: Sequence[str],
    output_dir: str | Path,
    model_dir: str | Path,
    da3_repo: str | Path,
    device: str,
    process_res: int,
    process_res_method: str,
    use_ray_pose: bool,
    ref_view_strategy: str,
    infer_gs: bool,
    save_depth_vis: bool = True,
) -> dict[str, np.ndarray | int | float | None]:
    """Save DA3 prediction arrays and return them as a dictionary."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "depth": prediction.depth,
        "conf": prediction.conf,
        "intrinsics": prediction.intrinsics,
        "extrinsics": prediction.extrinsics,
        "processed_images": prediction.processed_images,
        "is_metric": prediction.is_metric,
        "scale_factor": prediction.scale_factor,
        "prediction": prediction,
        "has_gaussians": prediction.gaussians is not None,
    }

    np.savez_compressed(
        output_dir / "da3_geometry.npz",
        depth=prediction.depth,
        conf=prediction.conf,
        intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,
        processed_images=prediction.processed_images,
        is_metric=_to_numpy_scalar(prediction.is_metric),
        scale_factor=_to_numpy_scalar(prediction.scale_factor),
    )

    metadata = {
        "frames": list(frame_paths),
        "model_dir": str(model_dir),
        "da3_repo": str(da3_repo),
        "device": str(device),
        "process_res": process_res,
        "process_res_method": process_res_method,
        "use_ray_pose": use_ray_pose,
        "ref_view_strategy": ref_view_strategy,
        "infer_gs": infer_gs,
        "outputs": {
            "depth": list(prediction.depth.shape),
            "conf": None if prediction.conf is None else list(prediction.conf.shape),
            "intrinsics": None if prediction.intrinsics is None else list(prediction.intrinsics.shape),
            "extrinsics": None if prediction.extrinsics is None else list(prediction.extrinsics.shape),
            "processed_images": None if prediction.processed_images is None else list(prediction.processed_images.shape),
            "has_gaussians": prediction.gaussians is not None,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if save_depth_vis:
        save_depth_visualizations(prediction.depth, output_dir)

    return result


class DA3GeometryTool:
    """Reusable DA3-Giant geometry, BEV, and local NVS tool."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        da3_repo: str | Path = DEFAULT_DA3_REPO,
        device: str = "cuda",
    ) -> None:
        self.da3_repo = Path(da3_repo).expanduser()
        self.model_dir = Path(model_dir).expanduser()
        _ensure_da3_importable(self.da3_repo)

        from depth_anything_3.api import DepthAnything3

        self.device = _resolve_device(device)
        with _quiet_da3_output():
            self.model = DepthAnything3.from_pretrained(str(self.model_dir))
            self.model = self.model.to(device=self.device)
        self.model.eval()

    def infer(
        self,
        frame_paths: Sequence[str],
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        infer_gs: bool = True,
    ):
        """Run one DA3 forward pass. 3DGS is enabled by default for Giant."""
        frames = collect_image_paths(frame_paths)
        with _quiet_da3_output():
            return self.model.inference(
                frames,
                infer_gs=infer_gs,
                process_res=process_res,
                process_res_method=process_res_method,
                use_ray_pose=use_ray_pose,
                ref_view_strategy=ref_view_strategy,
                export_dir=None,
                export_format="mini_npz",
            )

    def run(
        self,
        frame_paths: Sequence[str],
        output_dir: str | Path,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        infer_gs: bool = True,
        save_depth_vis: bool = True,
    ) -> dict[str, np.ndarray | int | float | None]:
        """Run inference and save depth, pose, intrinsics, and metadata."""
        frames = collect_image_paths(frame_paths)
        prediction = self.infer(
            frames,
            process_res=process_res,
            process_res_method=process_res_method,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            infer_gs=infer_gs,
        )
        return save_prediction_outputs(
            prediction=prediction,
            frame_paths=frames,
            output_dir=output_dir,
            model_dir=self.model_dir,
            da3_repo=self.da3_repo,
            device=str(self.device),
            process_res=process_res,
            process_res_method=process_res_method,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            infer_gs=infer_gs,
            save_depth_vis=save_depth_vis,
        )

    def export_gaussian_ply(
        self,
        prediction,
        output_dir: str | Path,
        gs_views_interval: int | None = 1,
    ) -> Path:
        """Export DA3 3DGS PLY without triggering DA3's automatic gs_video export."""
        if prediction.gaussians is None:
            raise ValueError("prediction.gaussians is None. Run infer(..., infer_gs=True).")
        from depth_anything_3.utils.export.gs import export_to_gs_ply

        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        export_to_gs_ply(
            prediction=prediction,
            export_dir=str(output_dir),
            gs_views_interval=gs_views_interval,
        )
        return output_dir / "gs_ply" / "0000.ply"

    def make_bev(
        self,
        prediction,
        output_dir: str | Path | None = None,
        horizontal_axes: tuple[int, int] = (0, 2),
        target_max_side_px: int = 1200,
        use_conf: bool = True,
        conf_percentile: float = 25.0,
        max_points: int | None = None,
        rotate_tall_display: bool = True,
        pad_to_square_display: bool = True,
    ) -> dict[str, np.ndarray | float | tuple[int, int]]:
        """Create a color BEV by depth back-projection, not by Gaussian rendering."""
        points, colors = prediction_to_point_cloud(
            prediction,
            use_conf=use_conf,
            conf_percentile=conf_percentile,
        )
        if max_points is not None and len(points) > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(points), size=max_points, replace=False)
            points, colors = points[idx], colors[idx]

        if points.shape[0] == 0:
            raise ValueError("No valid points were produced from prediction depth.")

        xy = points[:, horizontal_axes]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_extent = np.maximum(xy_max - xy_min, 1e-6)
        cell_size = float(xy_extent.max() / target_max_side_px)
        cell_size = max(cell_size, 1e-6)

        bev_size = np.ceil(xy_extent / cell_size).astype(int) + 1
        width, height = int(bev_size[0]), int(bev_size[1])
        pix = np.floor((xy - xy_min) / cell_size).astype(np.int64)
        px = np.clip(pix[:, 0], 0, width - 1)
        py = np.clip(pix[:, 1], 0, height - 1)

        color_sum = np.zeros((height, width, 3), dtype=np.float64)
        count = np.zeros((height, width), dtype=np.float64)
        np.add.at(color_sum, (py, px), colors.astype(np.float64))
        np.add.at(count, (py, px), 1)

        bev_color = np.zeros((height, width, 3), dtype=np.uint8)
        valid = count > 0
        bev_color[valid] = np.clip(color_sum[valid] / count[valid, None], 0, 255).astype(np.uint8)

        bev_display = np.flipud(bev_color)
        if rotate_tall_display and bev_display.shape[0] > bev_display.shape[1]:
            bev_display = np.rot90(bev_display, k=1)
        if pad_to_square_display:
            h, w = bev_display.shape[:2]
            side = max(h, w)
            padded = np.zeros((side, side, 3), dtype=bev_display.dtype)
            y0 = (side - h) // 2
            x0 = (side - w) // 2
            padded[y0 : y0 + h, x0 : x0 + w] = bev_display
            bev_display = padded

        result = {
            "points": points,
            "colors": colors,
            "bev_color": bev_color,
            "bev_display": bev_display,
            "xy_min": xy_min,
            "xy_max": xy_max,
            "xy_extent": xy_extent,
            "cell_size": cell_size,
            "horizontal_axes": horizontal_axes,
            "raw_size_hw": (height, width),
            "use_conf": use_conf,
            "conf_percentile": conf_percentile,
        }

        if output_dir is not None:
            output_dir = Path(output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_dir / "bev_from_depth.npz",
                bev_color=bev_color,
                bev_display=bev_display,
                xy_min=xy_min,
                xy_max=xy_max,
                xy_extent=xy_extent,
                cell_size=np.array(cell_size, dtype=np.float32),
                horizontal_axes=np.array(horizontal_axes, dtype=np.int32),
            )
            _save_rgb_image(bev_display, output_dir / "bev_from_depth.png")

        return result

    def render_nvs(
        self,
        prediction,
        frame_index: int = 0,
        movement: Movement = "yaw_left",
        angle_deg: float = 30.0,
        distance: float = 0.2,
        output_dir: str | Path | None = None,
        output_name: str | None = None,
        render_hw: tuple[int, int] | None = None,
        use_sh: bool = True,
        color_mode: str = "RGB+D",
        show_reference: bool = False,
    ) -> dict[str, np.ndarray]:
        """Render a local NVS image from a source camera pose using DA3 3DGS.

        Rotation movements use ``angle_deg``. Translation movements use ``distance``
        in DA3 world units along the selected camera axis.
        """
        if prediction.gaussians is None:
            raise ValueError("prediction.gaussians is None. Run infer(..., infer_gs=True).")

        import torch
        from depth_anything_3.model.utils.gs_renderer import render_3dgs
        from depth_anything_3.utils.geometry import as_homogeneous

        gaussians = prediction.gaussians
        device = gaussians.means.device
        height, width = prediction.depth.shape[-2:] if render_hw is None else render_hw

        base_w2c = torch.from_numpy(prediction.extrinsics[frame_index : frame_index + 1]).float().to(device)
        base_w2c = as_homogeneous(base_w2c)[0].float()
        base_c2w = torch.linalg.inv(base_w2c)

        base_intr = torch.from_numpy(prediction.intrinsics[frame_index : frame_index + 1]).float().to(device)
        intr_norm = base_intr.clone()
        intr_norm[:, 0, :] /= prediction.depth.shape[-1]
        intr_norm[:, 1, :] /= prediction.depth.shape[-2]

        target_c2w = self._apply_camera_movement(base_c2w, movement, angle_deg, distance).float()
        target_w2c = torch.linalg.inv(target_c2w).unsqueeze(0)

        with torch.inference_mode():
            color, depth = render_3dgs(
                extrinsics=target_w2c,
                intrinsics=intr_norm,
                image_shape=(height, width),
                gaussian=gaussians,
                use_sh=use_sh,
                num_view=1,
                color_mode=color_mode,
            )
        image = color[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        depth_np = depth[0].detach().cpu().numpy()

        result = {"image": image, "depth": depth_np, "target_w2c": target_w2c.detach().cpu().numpy()[0]}

        if show_reference:
            with torch.inference_mode():
                ref_color, ref_depth = render_3dgs(
                    extrinsics=base_w2c.unsqueeze(0),
                    intrinsics=intr_norm,
                    image_shape=(height, width),
                    gaussian=gaussians,
                    use_sh=use_sh,
                    num_view=1,
                    color_mode=color_mode,
                )
            result["reference_image"] = ref_color[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
            result["reference_depth"] = ref_depth[0].detach().cpu().numpy()

        if output_dir is not None:
            output_dir = Path(output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = output_name or f"nvs_f{frame_index:04d}_{movement}"
            _save_rgb_image(image, output_dir / f"{stem}.png")
            np.savez_compressed(
                output_dir / f"{stem}.npz",
                image=image,
                depth=depth_np,
                target_w2c=result["target_w2c"],
                movement=np.array(movement),
                angle_deg=np.array(angle_deg, dtype=np.float32),
                distance=np.array(distance, dtype=np.float32),
            )

        return result

    @staticmethod
    def _apply_camera_movement(base_c2w, movement: Movement, angle_deg: float, distance: float):
        import torch

        target = base_c2w.clone()
        dtype = target.dtype
        device = target.device
        angle = torch.tensor(np.deg2rad(angle_deg), dtype=dtype, device=device)
        c = torch.cos(angle)
        s = torch.sin(angle)

        def rot_x(theta_sign: float):
            ss = s * theta_sign
            rot = torch.eye(4, dtype=dtype, device=device)
            rot[:3, :3] = torch.stack(
                [
                    torch.stack([torch.tensor(1.0, dtype=dtype, device=device), torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(0.0, dtype=dtype, device=device)]),
                    torch.stack([torch.tensor(0.0, dtype=dtype, device=device), c, -ss]),
                    torch.stack([torch.tensor(0.0, dtype=dtype, device=device), ss, c]),
                ]
            )
            return rot

        def rot_y(theta_sign: float):
            ss = s * theta_sign
            rot = torch.eye(4, dtype=dtype, device=device)
            rot[:3, :3] = torch.stack(
                [
                    torch.stack([c, torch.tensor(0.0, dtype=dtype, device=device), ss]),
                    torch.stack([torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(1.0, dtype=dtype, device=device), torch.tensor(0.0, dtype=dtype, device=device)]),
                    torch.stack([-ss, torch.tensor(0.0, dtype=dtype, device=device), c]),
                ]
            )
            return rot

        # OpenCV-style camera frame: x right, y down, z forward.
        if movement == "yaw_left":
            target = target @ rot_y(-1.0)
        elif movement == "yaw_right":
            target = target @ rot_y(1.0)
        elif movement == "pitch_up":
            target = target @ rot_x(1.0)
        elif movement == "pitch_down":
            target = target @ rot_x(-1.0)
        elif movement in {"move_left", "move_right", "move_up", "move_down", "move_forward", "move_backward"}:
            axis = {
                "move_left": (0, -1.0),
                "move_right": (0, 1.0),
                "move_up": (1, -1.0),
                "move_down": (1, 1.0),
                "move_forward": (2, 1.0),
                "move_backward": (2, -1.0),
            }[movement]
            delta_cam = torch.zeros(3, dtype=dtype, device=device)
            delta_cam[axis[0]] = axis[1] * distance
            target[:3, 3] = target[:3, 3] + target[:3, :3] @ delta_cam
        else:
            raise ValueError(f"Unsupported movement: {movement}")
        return target


def run_da3_geometry(
    frame_paths: Sequence[str],
    output_dir: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    da3_repo: str | Path = DEFAULT_DA3_REPO,
    device: str = "cuda",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    infer_gs: bool = True,
    save_depth_vis: bool = True,
) -> dict[str, np.ndarray | int | float | None]:
    """Run DA3 once from a one-shot CLI-like call."""
    tool = DA3GeometryTool(model_dir=model_dir, da3_repo=da3_repo, device=device)
    return tool.run(
        frame_paths=frame_paths,
        output_dir=output_dir,
        process_res=process_res,
        process_res_method=process_res_method,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        infer_gs=infer_gs,
        save_depth_vis=save_depth_vis,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DA3-Giant on multiple frames and save depth/camera pose outputs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input image paths or directories. Directories are expanded by image suffix.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where da3_geometry.npz and metadata.json will be saved.",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--da3-repo", default=str(DEFAULT_DA3_REPO))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--ref-view-strategy", default="saddle_balanced")
    parser.add_argument("--no-gs", action="store_true", help="Disable DA3 3DGS inference.")
    parser.add_argument("--no-depth-vis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_da3_geometry(
        frame_paths=args.inputs,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        da3_repo=args.da3_repo,
        device=args.device,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        infer_gs=not args.no_gs,
        save_depth_vis=not args.no_depth_vis,
    )
    print("Saved DA3 geometry outputs.")
    print(f"depth: {result['depth'].shape}")
    if result["extrinsics"] is not None:
        print(f"extrinsics: {result['extrinsics'].shape}")
    if result["intrinsics"] is not None:
        print(f"intrinsics: {result['intrinsics'].shape}")
    print(f"has_gaussians: {result['has_gaussians']}")


if __name__ == "__main__":
    main()
