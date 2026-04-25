"""
Utilities for flattening and saving predicted Gaussian parameters.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


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
    rgb = np.clip(sh[:, 0], 0.0, 1.0)
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
