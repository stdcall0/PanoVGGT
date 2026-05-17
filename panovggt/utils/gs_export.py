"""
Export per-frame Gaussians to a standard 3DGS .ply format.

The .ply layout matches the inria 3DGS / SuperSplat / Niagara viewers:
- vertex header lists `x y z nx ny nz f_dc_0..2 f_rest_0..3K opacity scale_0..2 rot_0..3`
- normals are unused (zeros)
"""

from typing import Optional, Dict
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def gs_to_ply(
    means: torch.Tensor,        # (N, 3)
    scales: torch.Tensor,       # (N, 3) positive (we log them on write)
    rotations: torch.Tensor,    # (N, 4) wxyz (3DGS uses wxyz)
    opacities: torch.Tensor,    # (N,) sigmoid space (we logit them on write)
    sh_dc: torch.Tensor,        # (N, 3)
    sh_rest: Optional[torch.Tensor],  # (N, 3, sh_extra) or (N, sh_extra, 3)
    out_path: str,
):
    means = means.detach().cpu().numpy().astype(np.float32)
    scales = scales.detach().cpu().numpy().astype(np.float32)
    rotations = rotations.detach().cpu().numpy().astype(np.float32)
    opacities = opacities.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
    sh_dc = sh_dc.detach().cpu().numpy().astype(np.float32)

    if sh_rest is None:
        sh_rest_np = np.zeros((means.shape[0], 0), dtype=np.float32)
    else:
        sr = sh_rest.detach().cpu().numpy().astype(np.float32)
        if sr.ndim == 3 and sr.shape[1] == 3:           # (N, 3, K-1)
            sr = sr.transpose(0, 2, 1)                  # → (N, K-1, 3)
        sh_rest_np = sr.reshape(sr.shape[0], -1)        # (N, 3*(K-1))

    log_scales = np.log(np.clip(scales, 1e-8, None))
    inv_sigmoid_op = np.log(np.clip(opacities, 1e-6, 1 - 1e-6) /
                            (1 - np.clip(opacities, 1e-6, 1 - 1e-6)))

    n_dc = sh_dc.shape[1]
    n_rest = sh_rest_np.shape[1]
    dtype_full = (
        [("x", "f4"), ("y", "f4"), ("z", "f4"),
         ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        + [(f"f_dc_{i}", "f4") for i in range(n_dc)]
        + [(f"f_rest_{i}", "f4") for i in range(n_rest)]
        + [("opacity", "f4"),
           ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
           ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    )

    elements = np.empty(means.shape[0], dtype=dtype_full)
    cols = [means, np.zeros_like(means), sh_dc, sh_rest_np, inv_sigmoid_op,
            log_scales, rotations]
    flat = np.concatenate(cols, axis=1)
    for i, name in enumerate(elements.dtype.names):
        elements[name] = flat[:, i]

    ply_el = PlyElement.describe(elements, "vertex")
    PlyData([ply_el]).write(out_path)
    return out_path


def aggregate_predictions(
    pred: Dict[str, torch.Tensor],
    sh_degree: int = 1,
) -> Dict[str, torch.Tensor]:
    """Flatten the GS branch's (B,S,Hp,Wp,*) tensors into a single set in world frame.

    Centers come from `world_points` patch-pooled to (B,S,Hp,Wp,3) — done by
    averaging 14x14 windows.
    """
    from panovggt.render.gs_utils import patch_pool

    g = pred["gaussian"]
    B, S, Hp, Wp, _ = g["sh_dc"].shape
    H, W = pred["world_points"].shape[2:4]
    patch = H // Hp
    centers_pp = patch_pool(pred["world_points"], patch).reshape(-1, 3)
    means = centers_pp + g["offset"].reshape(-1, 3)
    scales = g["scale"].reshape(-1, 3)
    quats = g["rotation"].reshape(-1, 4)
    opacities = g["opacity"].reshape(-1)
    sh_dc = g["sh_dc"].reshape(-1, 3)
    sh_rest = g["sh_rest"]
    if sh_rest.shape[-1] > 0:
        sh_rest = sh_rest.reshape(-1, 3, sh_rest.shape[-1])
    else:
        sh_rest = None
    return dict(
        means=means, scales=scales, rotations=quats, opacities=opacities,
        sh_dc=sh_dc, sh_rest=sh_rest,
    )
