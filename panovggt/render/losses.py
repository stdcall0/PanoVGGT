"""
Photometric losses for the GS branch (RGB L1 + SSIM + optional depth L1).

We avoid pulling kornia in by using a small differentiable SSIM impl.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d


def ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """SSIM over a 4-D image batch (B, C, H, W). Returns per-pixel map."""
    device = pred.device
    dtype = pred.dtype
    C = pred.shape[1]
    win = _gaussian_window(window_size, sigma, device, dtype)
    win = win.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_p = F.conv2d(pred, win, padding=pad, groups=C)
    mu_g = F.conv2d(gt, win, padding=pad, groups=C)
    mu_p2 = mu_p ** 2
    mu_g2 = mu_g ** 2
    mu_pg = mu_p * mu_g

    sigma_p = F.conv2d(pred * pred, win, padding=pad, groups=C) - mu_p2
    sigma_g = F.conv2d(gt * gt, win, padding=pad, groups=C) - mu_g2
    sigma_pg = F.conv2d(pred * gt, win, padding=pad, groups=C) - mu_pg

    num = (2 * mu_pg + C1) * (2 * sigma_pg + C2)
    den = (mu_p2 + mu_g2 + C1) * (sigma_p + sigma_g + C2)
    return num / den


def masked_l1(
    pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6
) -> torch.Tensor:
    err = (pred - gt).abs()
    if mask is None:
        return err.mean()
    if mask.dim() == err.dim() - 1:
        mask = mask.unsqueeze(1)
    err = err * mask
    return err.sum() / (mask.sum() * pred.shape[1] + eps)


class GaussianRenderLoss(nn.Module):
    """RGB-L1 + SSIM + optional depth-L1, all masked by ERP validity mask."""

    def __init__(
        self,
        rgb_weight: float = 1.0,
        ssim_weight: float = 0.2,
        depth_weight: float = 0.0,
        ssim_window: int = 11,
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.ssim_weight = ssim_weight
        self.depth_weight = depth_weight
        self.ssim_window = ssim_window

    def forward(
        self,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        depth_pred: Optional[torch.Tensor] = None,
        depth_gt: Optional[torch.Tensor] = None,
        depth_mask: Optional[torch.Tensor] = None,
    ):
        details = {}
        rgb_l1 = masked_l1(rgb_pred, rgb_gt, mask)
        details["rgb_l1"] = rgb_l1
        total = self.rgb_weight * rgb_l1

        if self.ssim_weight > 0:
            ssim_map = ssim(rgb_pred, rgb_gt, window_size=self.ssim_window)
            if mask is not None:
                if mask.dim() == ssim_map.dim() - 1:
                    mask = mask.unsqueeze(1)
                ssim_map = ssim_map * mask
                ssim_loss = 1.0 - (ssim_map.sum() / (mask.sum() * ssim_map.shape[1] + 1e-6))
            else:
                ssim_loss = 1.0 - ssim_map.mean()
            details["ssim"] = ssim_loss
            total = total + self.ssim_weight * ssim_loss

        if self.depth_weight > 0 and depth_pred is not None and depth_gt is not None:
            d_l1 = masked_l1(depth_pred, depth_gt, depth_mask)
            details["depth_l1"] = d_l1
            total = total + self.depth_weight * d_l1

        details["total"] = total
        return total, details
