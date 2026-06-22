"""
Photometric losses for the GS branch (RGB L1/MSE + SSIM + optional depth L1).

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


def erp_solid_angle_weights(height: int, device, dtype) -> torch.Tensor:
    """ERP row weights proportional to per-pixel solid angle."""
    rows = torch.arange(height, device=device, dtype=dtype) + 0.5
    phi = (rows / height - 0.5) * torch.pi
    return torch.cos(phi).clamp_min(0.0).view(1, 1, height, 1)


def _weight_mask_like(
    ref: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
):
    weight_mask = torch.ones_like(ref[:, :1])
    if mask is not None:
        if mask.dim() == ref.dim() - 1:
            mask = mask.unsqueeze(1)
        weight_mask = weight_mask * mask.to(device=ref.device, dtype=ref.dtype)
    if weight is not None:
        weight_mask = weight_mask * weight.to(device=ref.device, dtype=ref.dtype)
    return weight_mask


def _sanitize_masked_pair(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
):
    if mask is None and weight is None:
        return pred, gt
    keep = _weight_mask_like(pred, mask=mask, weight=weight) > 0
    pred = torch.where(keep, pred, torch.zeros_like(pred))
    gt = torch.where(keep, gt, torch.zeros_like(gt))
    return pred, gt


def _masked_weighted_mean(
    err: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if mask is None and weight is None:
        return err.mean()
    weight_mask = _weight_mask_like(err, mask=mask, weight=weight)
    err = torch.where(weight_mask > 0, err * weight_mask, torch.zeros_like(err))
    return err.sum() / (weight_mask.sum() * err.shape[1] + eps)


def _ssim_valid_window_mask(mask: torch.Tensor, window_size: int) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    valid = (mask > 0).to(dtype=mask.dtype)
    kernel = mask.new_ones(1, 1, window_size, window_size)
    pad = window_size // 2
    valid_count = F.conv2d(valid, kernel, padding=pad)
    full_window = (valid_count >= window_size * window_size).to(dtype=mask.dtype)
    return mask * full_window


def masked_mse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred, gt = _sanitize_masked_pair(pred, gt, mask=mask, weight=weight)
    err = (pred - gt).square()
    return _masked_weighted_mean(err, mask=mask, weight=weight, eps=eps)


def masked_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred, gt = _sanitize_masked_pair(pred, gt, mask=mask, weight=weight)
    err = (pred - gt).abs()
    return _masked_weighted_mean(err, mask=mask, weight=weight, eps=eps)


def masked_charbonnier(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    charbonnier_eps: float = 1e-3,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred, gt = _sanitize_masked_pair(pred, gt, mask=mask, weight=weight)
    residual = pred - gt
    err = torch.sqrt(residual.square() + charbonnier_eps ** 2) - charbonnier_eps
    return _masked_weighted_mean(err, mask=mask, weight=weight, eps=eps)


class GaussianRenderLoss(nn.Module):
    """RGB photometric loss + SSIM + optional depth-L1, masked by ERP validity mask."""

    def __init__(
        self,
        rgb_weight: float = 1.0,
        ssim_weight: float = 0.2,
        depth_weight: float = 0.0,
        ssim_window: int = 11,
        rgb_loss_type: str = "l1",
        charbonnier_eps: float = 1e-3,
        solid_angle_weight: bool = False,
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.ssim_weight = ssim_weight
        self.depth_weight = depth_weight
        self.ssim_window = ssim_window
        self.rgb_loss_type = rgb_loss_type.lower()
        self.charbonnier_eps = float(charbonnier_eps)
        self.solid_angle_weight = bool(solid_angle_weight)
        if self.rgb_loss_type not in ("l1", "mse", "l2", "charbonnier"):
            raise ValueError(
                f"Unsupported rgb_loss_type '{rgb_loss_type}'. "
                "Use 'l1', 'mse', or 'charbonnier'."
            )

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
        weight = None
        if self.solid_angle_weight:
            weight = erp_solid_angle_weights(
                rgb_pred.shape[-2], device=rgb_pred.device, dtype=rgb_pred.dtype
            )
        if self.rgb_loss_type == "l1":
            rgb_loss = masked_l1(rgb_pred, rgb_gt, mask, weight=weight)
            details["rgb_l1"] = rgb_loss
        elif self.rgb_loss_type in ("mse", "l2"):
            rgb_loss = masked_mse(rgb_pred, rgb_gt, mask, weight=weight)
            details["rgb_mse"] = rgb_loss
        else:
            rgb_loss = masked_charbonnier(
                rgb_pred,
                rgb_gt,
                mask,
                weight=weight,
                charbonnier_eps=self.charbonnier_eps,
            )
            details["rgb_charbonnier"] = rgb_loss
        details["rgb_loss"] = rgb_loss
        total = self.rgb_weight * rgb_loss

        if self.ssim_weight > 0:
            if mask is not None:
                ssim_pred, ssim_gt = _sanitize_masked_pair(rgb_pred, rgb_gt, mask=mask)
                ssim_map = ssim(ssim_pred, ssim_gt, window_size=self.ssim_window)
                ssim_mask = _ssim_valid_window_mask(mask, self.ssim_window)
                ssim_weight_mask = _weight_mask_like(ssim_map, mask=ssim_mask, weight=weight)
                ssim_map = torch.where(
                    ssim_weight_mask > 0,
                    ssim_map * ssim_weight_mask,
                    torch.zeros_like(ssim_map),
                )
                denom = ssim_weight_mask.sum() * ssim_map.shape[1]
                ssim_loss = torch.where(
                    denom > 1e-6,
                    1.0 - (ssim_map.sum() / denom.clamp_min(1e-6)),
                    ssim_map.new_zeros(()),
                )
            else:
                ssim_map = ssim(rgb_pred, rgb_gt, window_size=self.ssim_window)
                if weight is not None:
                    ssim_weight_mask = _weight_mask_like(ssim_map, weight=weight)
                    denom = ssim_weight_mask.sum() * ssim_map.shape[1]
                    ssim_loss = torch.where(
                        denom > 1e-6,
                        1.0 - ((ssim_map * ssim_weight_mask).sum() / denom.clamp_min(1e-6)),
                        ssim_map.new_zeros(()),
                    )
                else:
                    ssim_loss = 1.0 - ssim_map.mean()
            details["ssim"] = ssim_loss
            total = total + self.ssim_weight * ssim_loss

        if self.depth_weight > 0 and depth_pred is not None and depth_gt is not None:
            d_l1 = masked_l1(depth_pred, depth_gt, depth_mask, weight=weight)
            details["depth_l1"] = d_l1
            total = total + self.depth_weight * d_l1

        details["total"] = total
        return total, details
