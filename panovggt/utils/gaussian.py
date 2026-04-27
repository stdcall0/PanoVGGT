"""
Utilities for flattening and saving predicted Gaussian parameters.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

GSPLAT_SH_C0 = 0.28209479177387814


GAUSSIAN_PREDICTION_KEYS = {
    "gaussian_keep_logits",
    "gaussian_keep_prob",
    "gaussian_split_logits",
    "gaussian_split_prob",
    "gaussian_split_count",
    "gaussian_token_means",
    "gaussian_token_log_scales",
    "gaussian_token_scales",
    "gaussian_token_rotations",
    "gaussian_token_opacity_logits",
    "gaussian_token_opacity",
    "gaussian_token_sh",
    "gaussian_child_offsets",
    "gaussian_child_log_scales",
    "gaussian_child_scales",
    "gaussian_child_rotations",
    "gaussian_child_opacity_logits",
    "gaussian_child_opacity",
    "gaussian_child_sh",
    "gaussian_means",
    "gaussian_scales",
    "gaussian_rotations",
    "gaussian_opacity",
    "gaussian_sh",
    "gaussian_valid_mask",
}


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def gaussian_keys_in(predictions: Dict[str, object]):
    return [key for key in predictions.keys() if key.startswith("gaussian_")]


def flatten_gaussian_predictions(
    predictions: Dict[str, object],
    keep_threshold: Optional[float] = None,
) -> Optional[Dict[str, np.ndarray]]:
    required = {
        "gaussian_means",
        "gaussian_scales",
        "gaussian_rotations",
        "gaussian_opacity",
        "gaussian_sh",
        "gaussian_valid_mask",
    }
    if not required.issubset(predictions):
        return None

    means = _to_numpy(predictions["gaussian_means"])
    scales = _to_numpy(predictions["gaussian_scales"])
    rotations = _to_numpy(predictions["gaussian_rotations"])
    opacity = _to_numpy(predictions["gaussian_opacity"])
    sh = _to_numpy(predictions["gaussian_sh"])
    mask = _to_numpy(predictions["gaussian_valid_mask"]).astype(bool)

    if mask.ndim == 4:
        means = means[None]
        scales = scales[None]
        rotations = rotations[None]
        opacity = opacity[None]
        sh = sh[None]
        mask = mask[None]

    if keep_threshold is not None and "gaussian_keep_prob" in predictions:
        keep_prob = _to_numpy(predictions["gaussian_keep_prob"]).squeeze(-1)
        if keep_prob.ndim == 3:
            keep_prob = keep_prob[None]
        mask &= keep_prob[..., None] >= float(keep_threshold)

    if mask.size == 0 or not mask.any():
        return {
            "means": np.zeros((0, 3), dtype=np.float32),
            "scales": np.zeros((0, 3), dtype=np.float32),
            "rotations": np.zeros((0, 4), dtype=np.float32),
            "opacity": np.zeros((0, 1), dtype=np.float32),
            "sh": np.zeros((0, sh.shape[-2], 3), dtype=np.float32),
            "batch_index": np.zeros((0,), dtype=np.int64),
            "frame_index": np.zeros((0,), dtype=np.int64),
            "patch_row": np.zeros((0,), dtype=np.int64),
            "patch_col": np.zeros((0,), dtype=np.int64),
            "slot_index": np.zeros((0,), dtype=np.int64),
        }

    batch_idx, frame_idx, patch_row, patch_col, slot_idx = np.nonzero(mask)
    flat = {
        "means": means[mask].astype(np.float32),
        "scales": scales[mask].astype(np.float32),
        "rotations": rotations[mask].astype(np.float32),
        "opacity": opacity[mask].astype(np.float32),
        "sh": sh[mask].astype(np.float32),
        "batch_index": batch_idx.astype(np.int64),
        "frame_index": frame_idx.astype(np.int64),
        "patch_row": patch_row.astype(np.int64),
        "patch_col": patch_col.astype(np.int64),
        "slot_index": slot_idx.astype(np.int64),
    }

    if "gaussian_keep_prob" in predictions:
        keep_prob = _to_numpy(predictions["gaussian_keep_prob"]).squeeze(-1)
        if keep_prob.ndim == 3:
            keep_prob = keep_prob[None]
        flat["keep_prob"] = keep_prob[batch_idx, frame_idx, patch_row, patch_col].astype(
            np.float32
        )
    if "gaussian_split_count" in predictions:
        split_count = _to_numpy(predictions["gaussian_split_count"])
        if split_count.ndim == 3:
            split_count = split_count[None]
        flat["split_count"] = split_count[
            batch_idx, frame_idx, patch_row, patch_col
        ].astype(np.int64)
    return flat


def save_gaussian_predictions(
    output_path: str,
    predictions: Dict[str, object],
    keep_threshold: Optional[float] = None,
) -> Optional[Dict[str, np.ndarray]]:
    flattened = flatten_gaussian_predictions(
        predictions, keep_threshold=keep_threshold
    )
    if flattened is None:
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **flattened)
    return flattened


def gaussian_color_from_sh(flattened: Dict[str, np.ndarray]) -> np.ndarray:
    sh = flattened.get("sh")
    if sh is None or sh.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rgb = np.clip(sh[:, 0] * GSPLAT_SH_C0 + 0.5, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def save_gaussian_centers_ply(
    output_path: str,
    flattened: Dict[str, np.ndarray],
) -> None:
    xyz = flattened.get("means")
    if xyz is None:
        return
    rgb = gaussian_color_from_sh(flattened)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    num_points = xyz.shape[0]

    with output.open("wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        data = np.empty(
            num_points,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("r", np.uint8),
                ("g", np.uint8),
                ("b", np.uint8),
            ],
        )
        data["x"] = xyz[:, 0]
        data["y"] = xyz[:, 1]
        data["z"] = xyz[:, 2]
        data["r"] = rgb[:, 0] if len(rgb) else 0
        data["g"] = rgb[:, 1] if len(rgb) else 0
        data["b"] = rgb[:, 2] if len(rgb) else 0
        f.write(data.tobytes())


def _inverse_sigmoid(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def save_gaussian_splat_ply(
    output_path: str,
    flattened: Dict[str, np.ndarray],
) -> None:
    """
    Save Gaussians in the de-facto standard 3DGS PLY layout used by common
    Gaussian Splat viewers.

    The exported properties follow the usual GraphDECO-compatible schema:
    xyz, dummy normals, SH DC / SH rest, opacity (logit), log-scales, rotation.
    """
    xyz = flattened.get("means")
    scales = flattened.get("scales")
    rotations = flattened.get("rotations")
    opacity = flattened.get("opacity")
    sh = flattened.get("sh")

    if any(v is None for v in (xyz, scales, rotations, opacity, sh)):
        return

    xyz = np.asarray(xyz, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    opacity = np.asarray(opacity, dtype=np.float32).reshape(-1, 1)
    sh = np.asarray(sh, dtype=np.float32)

    num_points = xyz.shape[0]
    if num_points == 0:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as f:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex 0\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                "property float f_dc_0\n"
                "property float f_dc_1\n"
                "property float f_dc_2\n"
                "property float opacity\n"
                "property float scale_0\n"
                "property float scale_1\n"
                "property float scale_2\n"
                "property float rot_0\n"
                "property float rot_1\n"
                "property float rot_2\n"
                "property float rot_3\n"
                "end_header\n"
            )
            f.write(header.encode("ascii"))
        return

    if sh.ndim != 3 or sh.shape[-1] != 3:
        raise ValueError(f"Expected SH array of shape (N, C, 3), got {sh.shape}")

    normals = np.zeros_like(xyz, dtype=np.float32)
    f_dc = sh[:, 0, :].astype(np.float32)
    f_rest = sh[:, 1:, :].reshape(num_points, -1).astype(np.float32)
    opacity_logit = _inverse_sigmoid(opacity).astype(np.float32)
    log_scales = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)

    rot_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / np.clip(rot_norm, 1e-8, None)

    dtype_fields = [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("nx", np.float32),
        ("ny", np.float32),
        ("nz", np.float32),
        ("f_dc_0", np.float32),
        ("f_dc_1", np.float32),
        ("f_dc_2", np.float32),
    ]
    dtype_fields.extend((f"f_rest_{i}", np.float32) for i in range(f_rest.shape[1]))
    dtype_fields.extend(
        [
            ("opacity", np.float32),
            ("scale_0", np.float32),
            ("scale_1", np.float32),
            ("scale_2", np.float32),
            ("rot_0", np.float32),
            ("rot_1", np.float32),
            ("rot_2", np.float32),
            ("rot_3", np.float32),
        ]
    )

    data = np.empty(num_points, dtype=dtype_fields)
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["nx"] = normals[:, 0]
    data["ny"] = normals[:, 1]
    data["nz"] = normals[:, 2]
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]
    for i in range(f_rest.shape[1]):
        data[f"f_rest_{i}"] = f_rest[:, i]
    data["opacity"] = opacity_logit[:, 0]
    data["scale_0"] = log_scales[:, 0]
    data["scale_1"] = log_scales[:, 1]
    data["scale_2"] = log_scales[:, 2]
    data["rot_0"] = rotations[:, 0]
    data["rot_1"] = rotations[:, 1]
    data["rot_2"] = rotations[:, 2]
    data["rot_3"] = rotations[:, 3]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
        )
        for name, np_type in dtype_fields:
            if np_type is not np.float32:
                raise TypeError(f"Unexpected dtype for Gaussian PLY field {name}: {np_type}")
            header += f"property float {name}\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
