"""
Gaussian prediction heads for patch-token 3D Gaussian Splatting outputs.

The implementation stays on the patch lattice to keep memory bounded while still
supporting adaptive densification through per-token split offsets.
"""

import math
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
        self._init_parameters()

    @staticmethod
    def _init_linear(linear: nn.Linear, std: float = 1e-3, bias: float = 0.0) -> None:
        nn.init.normal_(linear.weight, mean=0.0, std=std)
        nn.init.constant_(linear.bias, bias)

    def _init_parameters(self) -> None:
        self._init_linear(self.keep_head.net[-1], std=1e-3, bias=1.5)
        self._init_linear(self.split_head.net[-1], std=1e-3, bias=0.0)
        with torch.no_grad():
            # Start from one Gaussian per valid patch.  Densification should be
            # earned by the split loss; initializing to the max split makes the
            # render path see every child slot from the first step and tends to
            # produce over-dense grey blobs.
            self.split_head.net[-1].bias[0] = 3.0

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
        token_opacity_init: float = 0.78,
        child_opacity_init: float = 0.72,
        max_child_offset: float = 0.25,
        mean_anchor_scale: float = 1.0,
        max_mean_residual: float = 0.25,
        child_anchor_scale: float = 1.0,
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
        self.token_opacity_init = float(token_opacity_init)
        self.child_opacity_init = float(child_opacity_init)
        self.max_child_offset = float(max_child_offset)
        self.mean_anchor_scale = float(mean_anchor_scale)
        self.max_mean_residual = float(max_mean_residual)
        self.child_anchor_scale = float(child_anchor_scale)

        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_anchor_head = nn.Linear(in_dim, 3)
        self.mean_head = nn.Linear(hidden_dim, 3)
        self.log_scale_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 4)
        self.opacity_head = nn.Linear(hidden_dim, 1)
        self.sh_head = nn.Linear(hidden_dim, self.num_sh_bases * 3)
        self.child_anchor_offset_head = nn.Linear(
            in_dim, self.num_split_offsets * 3
        )
        self.offset_head = nn.Linear(hidden_dim, self.num_split_offsets * 3)
        self.child_log_scale_head = nn.Linear(hidden_dim, self.num_split_offsets * 3)
        self.child_rotation_head = nn.Linear(hidden_dim, self.num_split_offsets * 4)
        self.child_opacity_head = nn.Linear(hidden_dim, self.num_split_offsets)
        self.child_sh_head = nn.Linear(
            hidden_dim, self.num_split_offsets * self.num_sh_bases * 3
        )
        self._init_parameters()

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _init_linear(linear: nn.Linear, std: float = 1e-3, bias: float = 0.0) -> None:
        nn.init.normal_(linear.weight, mean=0.0, std=std)
        nn.init.constant_(linear.bias, bias)

    def _init_parameters(self) -> None:
        self._init_linear(self.mean_anchor_head, std=0.0, bias=0.0)
        self._init_linear(self.mean_head, std=1e-3, bias=0.0)
        # Patch targets in the normalized scene are typically around 0.1-0.3m.
        # Starting from extremely small Gaussians under-covers cube faces and
        # leaves the render-alpha loss with weak gradients.
        self._init_linear(self.log_scale_head, std=1e-3, bias=-1.5)
        self._init_linear(self.rotation_head, std=1e-3, bias=0.0)
        self._init_linear(
            self.opacity_head, std=1e-3, bias=self._logit(self.token_opacity_init)
        )
        self._init_linear(self.sh_head, std=1e-3, bias=0.0)
        self._init_linear(self.child_anchor_offset_head, std=0.0, bias=0.0)
        self._init_linear(self.offset_head, std=1e-3, bias=0.0)
        self._init_linear(self.child_log_scale_head, std=1e-3, bias=-1.6)
        self._init_linear(self.child_rotation_head, std=1e-3, bias=0.0)
        self._init_linear(
            self.child_opacity_head, std=1e-3, bias=self._logit(self.child_opacity_init)
        )
        self._init_linear(self.child_sh_head, std=1e-3, bias=0.0)
        with torch.no_grad():
            self.rotation_head.bias[0] = 1.0
            for child_idx in range(self.num_split_offsets):
                self.child_rotation_head.bias[child_idx * 4] = 1.0

    def forward(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        patch_h: int,
        patch_w: int,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(tokens)

        mean_anchor = self.mean_anchor_scale * self.mean_anchor_head(tokens)
        mean_residual = torch.tanh(self.mean_head(hidden)) * self.max_mean_residual
        means = mean_anchor + mean_residual
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
            child_anchor_offsets = self.child_anchor_offset_head(tokens).view(
                *hidden.shape[:-1], self.num_split_offsets, 3
            )
            raw_child_offsets = self.offset_head(hidden).view(
                *hidden.shape[:-1], self.num_split_offsets, 3
            )
            child_residual_offsets = torch.tanh(raw_child_offsets) * self.max_child_offset
            child_offsets = (
                self.child_anchor_scale * child_anchor_offsets
                + child_residual_offsets
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
            child_anchor_offsets = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_residual_offsets = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_offsets = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_log_scales = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_scales = hidden.new_zeros(*hidden.shape[:-1], 0, 3)
            child_rotations = hidden.new_zeros(*hidden.shape[:-1], 0, 4)
            child_opacity_logits = hidden.new_zeros(*hidden.shape[:-1], 0, 1)
            child_opacity = hidden.new_zeros(*hidden.shape[:-1], 0, 1)
            child_sh = hidden.new_zeros(*hidden.shape[:-1], 0, self.num_sh_bases, 3)

        grid_shape = (batch_size, num_frames, patch_h, patch_w)
        means = means.view(*grid_shape, 3)
        mean_anchor = mean_anchor.view(*grid_shape, 3)
        mean_residual = mean_residual.view(*grid_shape, 3)
        log_scales = log_scales.view(*grid_shape, 3)
        scales = scales.view(*grid_shape, 3)
        rotations = rotations.view(*grid_shape, 4)
        opacity_logits = opacity_logits.view(*grid_shape, 1)
        opacity = opacity.view(*grid_shape, 1)
        sh_coeffs = sh_coeffs.view(*grid_shape, self.num_sh_bases, 3)
        child_anchor_offsets = child_anchor_offsets.view(
            *grid_shape, self.num_split_offsets, 3
        )
        child_residual_offsets = child_residual_offsets.view(
            *grid_shape, self.num_split_offsets, 3
        )
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
            "gaussian_token_mean_anchor": mean_anchor,
            "gaussian_token_mean_residual": mean_residual,
            "gaussian_token_log_scales": log_scales,
            "gaussian_token_scales": scales,
            "gaussian_token_rotations": rotations,
            "gaussian_token_opacity_logits": opacity_logits,
            "gaussian_token_opacity": opacity,
            "gaussian_token_sh": sh_coeffs,
            "gaussian_child_anchor_offsets": child_anchor_offsets,
            "gaussian_child_residual_offsets": child_residual_offsets,
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
    soft_split: bool = False,
    min_active_prob: float = 1e-4,
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
        split_prob = stage1_outputs["gaussian_split_prob"]
    else:
        keep_prob = torch.ones_like(opacity)
        split_count = torch.ones_like(
            stage1_outputs["gaussian_split_count"], dtype=torch.long
        )
        split_prob = None

    slot_ids = torch.arange(max_gaussians, device=means.device).view(
        1, 1, 1, 1, max_gaussians
    )
    hard_active_mask = slot_ids < split_count.unsqueeze(-1)

    if use_stage1 and soft_split and split_prob is not None:
        split_slot_weight = torch.flip(
            torch.cumsum(torch.flip(split_prob, dims=[-1]), dim=-1),
            dims=[-1],
        ).clamp_(0.0, 1.0)
        active_mask = split_slot_weight > float(min_active_prob)
    else:
        split_slot_weight = hard_active_mask.to(materialized_opacity.dtype)
        active_mask = hard_active_mask

    if keep_threshold is not None:
        keep_mask = keep_prob.squeeze(-1).unsqueeze(-1) >= float(keep_threshold)
        active_mask = active_mask & keep_mask
        split_slot_weight = split_slot_weight * keep_mask.to(split_slot_weight.dtype)

    # keep_prob/split_prob are selection signals, not physical opacity.  The
    # renderer should see the predicted alpha of selected slots directly;
    # multiplying these probabilities into opacity made early Gaussians nearly
    # transparent and prevented render alpha from recovering.
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
