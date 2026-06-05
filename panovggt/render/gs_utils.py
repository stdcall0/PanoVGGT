"""
Helpers for the Gaussian-splatting branch.

- patch_pool / unpool helpers between full-resolution maps and patch tokens
- depth-footprint / kNN scale initialization
- sigmoid/softplus inverses
"""

import math
import torch
import torch.nn.functional as F


def softplus_inv(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.expm1(y.clamp_min(eps)) + eps)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def patch_pool(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Average-pool a per-pixel map into per-patch tokens.

    Args:
        x: (B, S, H, W, C) or (B, S, C, H, W). Returns matching layout.
        patch_size: integer pool/stride.
    Returns:
        Pooled tensor with H' = H // patch_size, W' = W // patch_size.
    """
    if x.dim() != 5:
        raise ValueError(f"expected 5-D tensor, got {x.shape}")
    if x.shape[-1] in (1, 3) and x.shape[-3] != x.shape[-1]:
        # (B, S, H, W, C) layout
        B, S, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(B * S, C, H, W)
        x = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
        Hp, Wp = x.shape[-2:]
        return x.reshape(B, S, C, Hp, Wp).permute(0, 1, 3, 4, 2).contiguous()
    else:
        # (B, S, C, H, W) layout
        B, S, C, H, W = x.shape
        x = x.reshape(B * S, C, H, W)
        x = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
        Hp, Wp = x.shape[-2:]
        return x.reshape(B, S, C, Hp, Wp).contiguous()


def patch_valid_ratio(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Average-pool a boolean/float valid mask to per-patch valid ratios."""
    if mask.dim() == 5:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"unexpected 5-D mask shape {mask.shape}")
    if mask.dim() != 4:
        raise ValueError(f"expected mask shape (B,S,H,W), got {mask.shape}")
    pooled = F.avg_pool2d(
        mask.float().flatten(0, 1).unsqueeze(1),
        kernel_size=patch_size,
        stride=patch_size,
    ).squeeze(1)
    return pooled.view(*mask.shape[:2], *pooled.shape[-2:])


def depth_footprint_scale(
    depth: torch.Tensor,
    patch_size: int,
    H: int,
    valid_mask=None,
) -> torch.Tensor:
    """
    Compute an isotropic init scale per patch from depth and angular pixel size.

    For an ERP image of height H, one row spans pi / H radians. A 14-pixel
    patch therefore subtends `patch_size * pi / H`. Multiplied by depth this
    gives a tangent-plane footprint in world units. We pool depth to patch
    resolution first.

    Args:
        depth: (B, S, H, W) or (B, S, H, W, 1).
        patch_size: int.
        H: ERP height in pixels.
    Returns:
        Tensor (B, S, Hp, Wp, 3) of isotropic scales.
    """
    if depth.dim() == 5:
        depth = depth.squeeze(-1)
    angular = patch_size * math.pi / float(H)
    if valid_mask is not None:
        if valid_mask.dim() == 5:
            if valid_mask.shape[-1] == 1:
                valid_mask = valid_mask[..., 0]
            elif valid_mask.shape[2] == 1:
                valid_mask = valid_mask[:, :, 0]
            else:
                raise ValueError(f"unexpected 5-D valid_mask shape {valid_mask.shape}")
        valid = valid_mask.to(dtype=depth.dtype)
        depth_sum = F.avg_pool2d(
            (depth * valid).flatten(0, 1).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
        ).squeeze(1)
        valid_ratio = F.avg_pool2d(
            valid.flatten(0, 1).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
        ).squeeze(1)
        pooled = depth_sum / valid_ratio.clamp_min(1e-6)
        pooled = torch.where(valid_ratio > 0, pooled, torch.zeros_like(pooled))
    else:
        pooled = F.avg_pool2d(
            depth.flatten(0, 1).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
        ).squeeze(1)  # (B*S, Hp, Wp)
    s = (pooled * angular).clamp_min(1e-4)
    s = s.view(*depth.shape[:2], *s.shape[-2:])  # (B,S,Hp,Wp)
    return s.unsqueeze(-1).expand(*s.shape, 3).contiguous()


@torch.no_grad()
def knn_scale(centers: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Compute per-Gaussian initialization scale as the mean distance to the
    k nearest neighbours in world space. Per-frame batched implementation.

    Args:
        centers: (B, S, N, 3) or (B, S, Hp, Wp, 3).
        k:       number of nearest neighbours (excluding self).
    Returns:
        Tensor of shape `centers.shape[:-1] + (3,)` with isotropic scale.
    """
    orig_shape = centers.shape
    B, S = orig_shape[:2]
    pts = centers.reshape(B, S, -1, 3)
    N = pts.shape[2]
    k = min(k, max(N - 1, 1))
    # pairwise distances per (B,S)
    d2 = torch.cdist(pts, pts, p=2)  # (B, S, N, N)
    d2.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
    nn_d, _ = d2.topk(k, dim=-1, largest=False)  # (B,S,N,k)
    s = nn_d.mean(dim=-1).clamp_min(1e-4)        # (B,S,N)
    s = s.unsqueeze(-1).expand(B, S, N, 3)
    return s.reshape(*orig_shape[:-1], 3).contiguous()
