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


def _mask_to_4d(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 5:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"unexpected 5-D mask shape {mask.shape}")
    if mask.dim() != 4:
        raise ValueError(f"expected mask shape (B,S,H,W), got {mask.shape}")
    return mask


def _avg_pool_4d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    pooled = F.avg_pool2d(
        x.flatten(0, 1).unsqueeze(1),
        kernel_size=patch_size,
        stride=patch_size,
    ).squeeze(1)
    return pooled.view(*x.shape[:2], *pooled.shape[-2:])


def patch_weighted_pool(x: torch.Tensor, valid_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Average-pool a per-pixel map using only valid pixels.

    Invalid pixels are zeroed with `torch.where` before multiplication so NaN or
    Inf values outside the mask cannot leak into the pooled result.
    """
    if x.dim() != 5:
        raise ValueError(f"expected 5-D tensor, got {x.shape}")
    valid = _mask_to_4d(valid_mask).to(dtype=x.dtype, device=x.device)
    valid = valid * torch.isfinite(valid).to(dtype=x.dtype)

    if x.shape[-1] in (1, 3) and x.shape[-3] != x.shape[-1]:
        B, S, H, W, C = x.shape
        weights = valid.unsqueeze(-1)
        safe_x = torch.where(weights > 0, x, torch.zeros_like(x))
        weighted = (safe_x * weights).permute(0, 1, 4, 2, 3).reshape(B * S, C, H, W)
        numer = F.avg_pool2d(weighted, kernel_size=patch_size, stride=patch_size)
        denom = F.avg_pool2d(
            valid.flatten(0, 1).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
        )
        pooled = numer / denom.clamp_min(1e-6)
        pooled = torch.where(denom > 0, pooled, torch.zeros_like(pooled))
        Hp, Wp = pooled.shape[-2:]
        return pooled.reshape(B, S, C, Hp, Wp).permute(0, 1, 3, 4, 2).contiguous()

    B, S, C, H, W = x.shape
    weights = valid.unsqueeze(2)
    safe_x = torch.where(weights > 0, x, torch.zeros_like(x))
    weighted = (safe_x * weights).reshape(B * S, C, H, W)
    numer = F.avg_pool2d(weighted, kernel_size=patch_size, stride=patch_size)
    denom = F.avg_pool2d(
        valid.flatten(0, 1).unsqueeze(1),
        kernel_size=patch_size,
        stride=patch_size,
    )
    pooled = numer / denom.clamp_min(1e-6)
    pooled = torch.where(denom > 0, pooled, torch.zeros_like(pooled))
    Hp, Wp = pooled.shape[-2:]
    return pooled.reshape(B, S, C, Hp, Wp).contiguous()


def patch_valid_ratio(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Average-pool a boolean/float valid mask to per-patch valid ratios."""
    mask = _mask_to_4d(mask)
    pooled = F.avg_pool2d(
        mask.float().flatten(0, 1).unsqueeze(1),
        kernel_size=patch_size,
        stride=patch_size,
    ).squeeze(1)
    return pooled.view(*mask.shape[:2], *pooled.shape[-2:])


def _reshape_subgrid_pooled(x: torch.Tensor, subgrid_size: int, channel_last: bool) -> torch.Tensor:
    if subgrid_size == 1:
        if channel_last:
            return x.unsqueeze(-2)
        return x.permute(0, 1, 3, 4, 2).contiguous().unsqueeze(-2)
    if channel_last:
        B, S, Hs, Ws, C = x.shape
        Hp, Wp = Hs // subgrid_size, Ws // subgrid_size
        x = x.view(B, S, Hp, subgrid_size, Wp, subgrid_size, C)
        return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, S, Hp, Wp, subgrid_size * subgrid_size, C
        )
    B, S, C, Hs, Ws = x.shape
    Hp, Wp = Hs // subgrid_size, Ws // subgrid_size
    x = x.view(B, S, C, Hp, subgrid_size, Wp, subgrid_size)
    return x.permute(0, 1, 3, 5, 4, 6, 2).reshape(
        B, S, Hp, Wp, subgrid_size * subgrid_size, C
    )


def subpatch_pool(x: torch.Tensor, patch_size: int, subgrid_size: int) -> torch.Tensor:
    """Average-pool a map into a per-patch subgrid with Q = subgrid_size^2."""
    if patch_size % subgrid_size != 0:
        raise ValueError(
            f"patch_size={patch_size} must be divisible by subgrid_size={subgrid_size}."
        )
    subpatch_size = patch_size // subgrid_size
    pooled = patch_pool(x, subpatch_size)
    channel_last = pooled.shape[-1] in (1, 3) and pooled.shape[-3] != pooled.shape[-1]
    return _reshape_subgrid_pooled(pooled, subgrid_size, channel_last=channel_last)


def subpatch_weighted_pool(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    patch_size: int,
    subgrid_size: int,
) -> torch.Tensor:
    """Valid-weighted pool into a per-patch subgrid with Q = subgrid_size^2."""
    if patch_size % subgrid_size != 0:
        raise ValueError(
            f"patch_size={patch_size} must be divisible by subgrid_size={subgrid_size}."
        )
    subpatch_size = patch_size // subgrid_size
    pooled = patch_weighted_pool(x, valid_mask, subpatch_size)
    channel_last = pooled.shape[-1] in (1, 3) and pooled.shape[-3] != pooled.shape[-1]
    return _reshape_subgrid_pooled(pooled, subgrid_size, channel_last=channel_last)


def subpatch_valid_ratio(mask: torch.Tensor, patch_size: int, subgrid_size: int) -> torch.Tensor:
    if patch_size % subgrid_size != 0:
        raise ValueError(
            f"patch_size={patch_size} must be divisible by subgrid_size={subgrid_size}."
        )
    subpatch_size = patch_size // subgrid_size
    pooled = patch_valid_ratio(mask, subpatch_size)
    return _reshape_subgrid_pooled(pooled.unsqueeze(-1), subgrid_size, channel_last=True).squeeze(-1)


def depth_footprint_scale(
    depth: torch.Tensor,
    patch_size: int,
    H: int,
    valid_mask=None,
) -> torch.Tensor:
    """
    Compute a tangent-footprint init scale per patch from depth and ERP latitude.

    For an ERP image of height H, one row spans pi / H radians. A 14-pixel
    patch therefore subtends `patch_size * pi / H`. Multiplied by depth this
    gives a vertical tangent-plane footprint in world units. Horizontal ERP
    footprint shrinks by cos(latitude), so the local x scale is row-aware.

    Args:
        depth: (B, S, H, W) or (B, S, H, W, 1).
        patch_size: int.
        H: ERP height in pixels.
    Returns:
        Tensor (B, S, Hp, Wp, 3) of local x/y/z scales.
    """
    if depth.dim() == 5:
        depth = depth.squeeze(-1)
    angular = patch_size * math.pi / float(H)
    if valid_mask is not None:
        valid = _mask_to_4d(valid_mask).to(dtype=depth.dtype, device=depth.device)
        finite = torch.isfinite(depth)
        valid = valid * finite.to(dtype=depth.dtype)
        safe_depth = torch.where(finite & (valid > 0), depth, torch.zeros_like(depth))
        depth_sum = _avg_pool_4d(safe_depth * valid, patch_size)
        valid_ratio = _avg_pool_4d(valid, patch_size)
        pooled = depth_sum / valid_ratio.clamp_min(1e-6)
        pooled = torch.where(valid_ratio > 0, pooled, torch.zeros_like(pooled))
    else:
        pooled = F.avg_pool2d(
            depth.flatten(0, 1).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
        ).squeeze(1)  # (B*S, Hp, Wp)
        pooled = pooled.view(*depth.shape[:2], *pooled.shape[-2:])
    if pooled.dim() == 3:
        pooled = pooled.view(*depth.shape[:2], *pooled.shape[-2:])

    vertical = (pooled * angular).clamp_min(1e-4)
    Hp = vertical.shape[-2]
    row_centers = (torch.arange(Hp, device=depth.device, dtype=depth.dtype) + 0.5) * patch_size
    latitude = (row_centers / float(H) - 0.5) * math.pi
    cos_lat = torch.cos(latitude).abs().view(1, 1, Hp, 1)
    horizontal = (vertical * cos_lat).clamp_min(1e-4)
    return torch.stack([horizontal, vertical, vertical], dim=-1).contiguous()


def subpatch_depth_footprint_scale(
    depth: torch.Tensor,
    patch_size: int,
    subgrid_size: int,
    H: int,
    valid_mask=None,
) -> torch.Tensor:
    if patch_size % subgrid_size != 0:
        raise ValueError(
            f"patch_size={patch_size} must be divisible by subgrid_size={subgrid_size}."
        )
    subpatch_size = patch_size // subgrid_size
    scale = depth_footprint_scale(depth, subpatch_size, H, valid_mask=valid_mask)
    return _reshape_subgrid_pooled(scale, subgrid_size, channel_last=True)


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
