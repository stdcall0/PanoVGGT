"""
Gaussian prediction heads for patch-token 3D Gaussian Splatting outputs.

The implementation stays on the patch lattice to keep memory bounded while still
supporting adaptive densification through per-token split offsets.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _standardize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat, dim=-1, eps=1e-6)
    return torch.where(quat[..., :1] < 0, -quat, quat)


class _TokenMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianStage1Head(nn.Module):
    """
    Predicts token-level keep probabilities and split counts.

    Split logits classify the number of active Gaussians per token in the range
    [1, max_gaussians_per_token].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        max_gaussians_per_token: int = 4,
    ):
        super().__init__()
        if max_gaussians_per_token < 1:
            raise ValueError("max_gaussians_per_token must be >= 1")

        hidden_dim = hidden_dim or in_dim
        self.max_gaussians_per_token = max_gaussians_per_token
        self.keep_head = _TokenMLP(in_dim, hidden_dim, 1)
        self.split_head = _TokenMLP(
            in_dim, hidden_dim, max_gaussians_per_token
        )

    def forward(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        patch_h: int,
        patch_w: int,
        enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if enabled:
            keep_logits = self.keep_head(tokens)
            split_logits = self.split_head(tokens)
            keep_prob = torch.sigmoid(keep_logits)
            split_count = split_logits.argmax(dim=-1) + 1
        else:
            device = tokens.device
            dtype = tokens.dtype
            keep_logits = torch.ones(
                (*tokens.shape[:-1], 1), device=device, dtype=dtype
            )
            split_logits = torch.zeros(
                (*tokens.shape[:-1], self.max_gaussians_per_token),
                device=device,
                dtype=dtype,
            )
            split_logits[..., 0] = 1.0
            keep_prob = torch.ones_like(keep_logits)
            split_count = torch.ones(
                tokens.shape[:-1], device=device, dtype=torch.long
            )

        keep_prob_grid = keep_prob.view(batch_size, num_frames, patch_h, patch_w, 1)
        keep_logits_grid = keep_logits.view(
            batch_size, num_frames, patch_h, patch_w, 1
        )
        split_logits_grid = split_logits.view(
            batch_size,
            num_frames,
            patch_h,
            patch_w,
            self.max_gaussians_per_token,
        )
        split_prob_grid = torch.softmax(split_logits_grid, dim=-1)
        split_count_grid = split_count.view(
            batch_size, num_frames, patch_h, patch_w
        )

        return {
            "gaussian_keep_logits": keep_logits_grid,
            "gaussian_keep_prob": keep_prob_grid,
            "gaussian_split_logits": split_logits_grid,
            "gaussian_split_prob": split_prob_grid,
            "gaussian_split_count": split_count_grid,
        }


class GaussianParameterHead(nn.Module):
    """
    Decodes patch tokens into base Gaussian parameters plus split offsets.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        max_gaussians_per_token: int = 4,
        sh_degree: int = 3,
        log_scale_min: float = -6.0,
        log_scale_max: float = 1.5,
    ):
        super().__init__()
        if max_gaussians_per_token < 1:
            raise ValueError("max_gaussians_per_token must be >= 1")

        hidden_dim = hidden_dim or in_dim
        self.max_gaussians_per_token = max_gaussians_per_token
        self.num_split_offsets = max(0, max_gaussians_per_token - 1)
        self.sh_degree = sh_degree
        self.num_sh_bases = (sh_degree + 1) ** 2
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)

        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 3)
        self.log_scale_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 4)
        self.opacity_head = nn.Linear(hidden_dim, 1)
        self.sh_head = nn.Linear(hidden_dim, self.num_sh_bases * 3)
        self.offset_head = nn.Linear(hidden_dim, self.num_split_offsets * 3)
        self.child_log_scale_head = nn.Linear(hidden_dim, self.num_split_offsets * 3)
        self.child_rotation_head = nn.Linear(hidden_dim, self.num_split_offsets * 4)
        self.child_opacity_head = nn.Linear(hidden_dim, self.num_split_offsets)
        self.child_sh_head = nn.Linear(
            hidden_dim, self.num_split_offsets * self.num_sh_bases * 3
        )

    def forward(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        patch_h: int,
        patch_w: int,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(tokens)

        means = self.mean_head(hidden)
        log_scales = self.log_scale_head(hidden).clamp(
            min=self.log_scale_min, max=self.log_scale_max
        )
        scales = torch.exp(log_scales)
        rotations = _standardize_quaternion(self.rotation_head(hidden))
        opacity_logits = self.opacity_head(hidden)
        opacity = torch.sigmoid(opacity_logits)
        sh_coeffs = self.sh_head(hidden).view(
            *hidden.shape[:-1], self.num_sh_bases, 3
        )

        if self.num_split_offsets > 0:
            child_offsets = self.offset_head(hidden).view(
                *hidden.shape[:-1], self.num_split_offsets, 3
            )
            child_log_scales = self.child_log_scale_head(hidden).view(
                *hidden.shape[:-1], self.num_split_offsets, 3
            ).clamp(min=self.log_scale_min, max=self.log_scale_max)
            child_scales = torch.exp(child_log_scales)
            child_rotations = _standardize_quaternion(
                self.child_rotation_head(hidden).view(
                    *hidden.shape[:-1], self.num_split_offsets, 4
                )
            )
            child_opacity_logits = self.child_opacity_head(hidden).view(
                *hidden.shape[:-1], self.num_split_offsets, 1
            )
            child_opacity = torch.sigmoid(child_opacity_logits)
            child_sh = self.child_sh_head(hidden).view(
                *hidden.shape[:-1], self.num_split_offsets, self.num_sh_bases, 3
            )
        else:
            child_offsets = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_log_scales = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_scales = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_rotations = hidden.new_zeros(*hidden.shape[:-1], 0, 4)
            child_opacity_logits = hidden.new_zeros(*hidden.shape[:-1], 0, 1)
            child_opacity = hidden.new_zeros(*hidden.shape[:-1], 0, 1)
            child_sh = hidden.new_zeros(*hidden.shape[:-1], 0, self.num_sh_bases, 3)

        grid_shape = (batch_size, num_frames, patch_h, patch_w)
        means = means.view(*grid_shape, 3)
        log_scales = log_scales.view(*grid_shape, 3)
        scales = scales.view(*grid_shape, 3)
        rotations = rotations.view(*grid_shape, 4)
        opacity_logits = opacity_logits.view(*grid_shape, 1)
        opacity = opacity.view(*grid_shape, 1)
        sh_coeffs = sh_coeffs.view(*grid_shape, self.num_sh_bases, 3)
        child_offsets = child_offsets.view(
            *grid_shape, self.num_split_offsets, 3
        )
        child_log_scales = child_log_scales.view(
            *grid_shape, self.num_split_offsets, 3
        )
        child_scales = child_scales.view(*grid_shape, self.num_split_offsets, 3)
        child_rotations = child_rotations.view(
            *grid_shape, self.num_split_offsets, 4
        )
        child_opacity_logits = child_opacity_logits.view(
            *grid_shape, self.num_split_offsets, 1
        )
        child_opacity = child_opacity.view(*grid_shape, self.num_split_offsets, 1)
        child_sh = child_sh.view(
            *grid_shape, self.num_split_offsets, self.num_sh_bases, 3
        )

        return {
            "gaussian_token_means": means,
            "gaussian_token_log_scales": log_scales,
            "gaussian_token_scales": scales,
            "gaussian_token_rotations": rotations,
            "gaussian_token_opacity_logits": opacity_logits,
            "gaussian_token_opacity": opacity,
            "gaussian_token_sh": sh_coeffs,
            "gaussian_child_offsets": child_offsets,
            "gaussian_child_log_scales": child_log_scales,
            "gaussian_child_scales": child_scales,
            "gaussian_child_rotations": child_rotations,
            "gaussian_child_opacity_logits": child_opacity_logits,
            "gaussian_child_opacity": child_opacity,
            "gaussian_child_sh": child_sh,
        }


def assemble_gaussian_outputs(
    stage1_outputs: Dict[str, torch.Tensor],
    param_outputs: Dict[str, torch.Tensor],
    keep_threshold: Optional[float] = None,
    use_stage1: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Materialize per-token Gaussian parameters into a fixed maximum number of slots.
    """

    means = param_outputs["gaussian_token_means"]
    scales = param_outputs["gaussian_token_scales"]
    rotations = param_outputs["gaussian_token_rotations"]
    opacity = param_outputs["gaussian_token_opacity"]
    sh_coeffs = param_outputs["gaussian_token_sh"]
    child_offsets = param_outputs["gaussian_child_offsets"]
    child_scales = param_outputs["gaussian_child_scales"]
    child_rotations = param_outputs["gaussian_child_rotations"]
    child_opacity = param_outputs["gaussian_child_opacity"]
    child_sh = param_outputs["gaussian_child_sh"]

    batch_size, num_frames, patch_h, patch_w, _ = means.shape
    max_gaussians = child_offsets.shape[-2] + 1

    zero_offset = torch.zeros_like(means).unsqueeze(-2)
    all_offsets = torch.cat([zero_offset, child_offsets], dim=-2)
    materialized_means = means.unsqueeze(-2) + all_offsets
    materialized_scales = torch.cat(
        [scales.unsqueeze(-2), child_scales], dim=-2
    )
    materialized_rotations = torch.cat(
        [rotations.unsqueeze(-2), child_rotations], dim=-2
    )
    materialized_sh = torch.cat(
        [sh_coeffs.unsqueeze(-3), child_sh], dim=-3
    )
    materialized_opacity = torch.cat(
        [opacity.unsqueeze(-2), child_opacity], dim=-2
    )

    if use_stage1:
        keep_prob = stage1_outputs["gaussian_keep_prob"]
        split_count = stage1_outputs["gaussian_split_count"]
    else:
        keep_prob = torch.ones_like(opacity)
        split_count = torch.ones_like(
            stage1_outputs["gaussian_split_count"], dtype=torch.long
        )

    slot_ids = torch.arange(max_gaussians, device=means.device).view(
        1, 1, 1, 1, max_gaussians
    )
    active_mask = slot_ids < split_count.unsqueeze(-1)
    if keep_threshold is not None:
        active_mask = active_mask & (
            keep_prob.squeeze(-1).unsqueeze(-1) >= float(keep_threshold)
        )

    materialized_opacity = materialized_opacity * keep_prob.unsqueeze(-2)
    materialized_opacity = materialized_opacity * active_mask.unsqueeze(-1).to(
        materialized_opacity.dtype
    )

    return {
        "gaussian_means": materialized_means,
        "gaussian_scales": materialized_scales,
        "gaussian_rotations": materialized_rotations,
        "gaussian_opacity": materialized_opacity,
        "gaussian_sh": materialized_sh,
        "gaussian_valid_mask": active_mask,
    }
