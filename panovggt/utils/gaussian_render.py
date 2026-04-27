"""
Utilities for supervising Gaussian predictions with cubemap rendering.

The repository operates on equirectangular panoramas, while gsplat expects
standard projective cameras. This module bridges the two by projecting each
panorama frame to 90-degree cube faces and rasterizing the predicted Gaussians
against those pinhole views.
"""

import logging
import math
from typing import Dict, Optional, Sequence, Tuple

from panovggt.utils.runtime_env import (
    bootstrap_gsplat_runtime,
    loaded_libstdcpp_path,
)

bootstrap_gsplat_runtime()

import torch
import torch.nn as nn
import torch.nn.functional as F

from panovggt.Projection.Equirec2Cube import Equirec2Cube

logger = logging.getLogger(__name__)
GSPLAT_SH_C0 = 0.28209479177387814


def _masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(pred.dtype)
    denom = weight.sum().clamp_min(1.0)
    return ((pred - target).abs() * weight).sum() / denom


def _balanced_soft_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred = pred.float().clamp(1e-5, 1.0 - 1e-5)
    target = target.float().clamp(0.0, 1.0)

    with torch.no_grad():
        pos_mass = target.sum().clamp_min(1.0)
        neg_mass = (1.0 - target).sum().clamp_min(1.0)
        total_mass = pos_mass + neg_mass

        pos_weight = neg_mass / total_mass
        neg_weight = pos_mass / total_mass
        balance = target * pos_weight + (1.0 - target) * neg_weight

        # Boundary pixels are ambiguous after cubemap projection; downweight them.
        confidence = 0.5 + (target - 0.5).abs()
        weight = balance * confidence
        weight = weight / weight.mean().clamp_min(1e-6)

    pred_logits = torch.logit(pred)
    return F.binary_cross_entropy_with_logits(pred_logits, target, weight=weight)


class GaussianCubemapRenderLoss(nn.Module):
    """Renderer-backed loss using gsplat on cubemap faces."""

    def __init__(
        self,
        cube_dim: int = 64,
        face_indices: Optional[Sequence[int]] = None,
        rgb_weight: float = 1.0,
        depth_weight: float = 0.25,
        alpha_weight: float = 0.05,
        opacity_floor: float = 1e-4,
        fov_degrees: float = 90.0,
        sh_degree: int = 3,
        sh_warmup_start: float = 0.0,
        sh_warmup_end: float = 0.35,
    ):
        super().__init__()
        self.cube_dim = int(cube_dim)
        self.face_indices = tuple(face_indices or (0, 1, 2, 3, 4, 5))
        self.rgb_weight = float(rgb_weight)
        self.depth_weight = float(depth_weight)
        self.alpha_weight = float(alpha_weight)
        self.opacity_floor = float(opacity_floor)
        self.fov_degrees = float(fov_degrees)
        self.sh_degree = int(sh_degree)
        self.sh_warmup_start = float(sh_warmup_start)
        self.sh_warmup_end = float(sh_warmup_end)

        face_rotations = torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],  # back
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],  # down
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],   # front
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # left
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],  # right
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],  # top
            ],
            dtype=torch.float32,
        )
        self.register_buffer("_face_rotations", face_rotations, persistent=False)
        self._cubemap_projectors = nn.ModuleDict()
        self._warned_missing_gsplat = False

    def _high_order_scale(self, progress: object) -> float:
        if isinstance(progress, torch.Tensor):
            progress = float(progress.detach().float().mean().item())
        elif progress is None:
            progress = 1.0
        else:
            progress = float(progress)

        if self.sh_warmup_end <= self.sh_warmup_start:
            return 1.0
        scale = (progress - self.sh_warmup_start) / (
            self.sh_warmup_end - self.sh_warmup_start
        )
        return float(max(0.0, min(1.0, scale)))

    def _get_rasterization(self):
        bootstrap_gsplat_runtime(preload_runtime_libs=False)
        try:
            from gsplat import rasterization
        except Exception as exc:
            if (
                not self._warned_missing_gsplat
                and "CXXABI" in str(exc)
            ):
                logger.warning(
                    "gsplat failed to load against '%s'. "
                    "Bootstrap the runtime before importing torch, e.g. via training/launch.py.",
                    loaded_libstdcpp_path() or "unknown libstdc++",
                )
            if not self._warned_missing_gsplat:
                logger.warning(
                    "gsplat is unavailable; Gaussian render loss is disabled."
                )
                self._warned_missing_gsplat = True
            return None
        return rasterization

    def _get_e2c(self, equ_h: int, device: torch.device) -> Equirec2Cube:
        key = f"h{equ_h}"
        if key not in self._cubemap_projectors:
            self._cubemap_projectors[key] = Equirec2Cube(
                self.cube_dim, equ_h, FoV=self.fov_degrees
            )
        return self._cubemap_projectors[key].to(device)

    def _cube_intrinsics(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        focal = 0.5 * self.cube_dim / math.tan(0.5 * math.radians(self.fov_degrees))
        cx = (self.cube_dim - 1) * 0.5
        K = torch.tensor(
            [[focal, 0.0, cx], [0.0, focal, cx], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        return K

    def _prepare_targets(
        self, gt: Dict[str, torch.Tensor], frame_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = gt["imgs"].index_select(1, frame_indices)
        local_points = gt["local_points"].index_select(1, frame_indices)
        valid_masks = gt["valid_masks"].index_select(1, frame_indices).float()

        batch_size, num_frames, _, equ_h, equ_w = images.shape
        projector = self._get_e2c(equ_h, images.device)

        flat_images = images.reshape(batch_size * num_frames, 3, equ_h, equ_w)
        flat_points = (
            local_points.permute(0, 1, 4, 2, 3)
            .contiguous()
            .reshape(batch_size * num_frames, 3, equ_h, equ_w)
        )
        flat_masks = valid_masks.reshape(batch_size * num_frames, 1, equ_h, equ_w)

        cube_images = projector(flat_images, mode="bilinear").view(
            batch_size, num_frames, 6, 3, self.cube_dim, self.cube_dim
        )
        cube_points = projector(flat_points, mode="bilinear").view(
            batch_size, num_frames, 6, 3, self.cube_dim, self.cube_dim
        )
        cube_masks = projector(flat_masks, mode="bilinear").view(
            batch_size, num_frames, 6, 1, self.cube_dim, self.cube_dim
        )

        face_ids = torch.as_tensor(
            self.face_indices, device=images.device, dtype=torch.long
        )
        cube_images = cube_images.index_select(2, face_ids)
        cube_points = cube_points.index_select(2, face_ids)
        cube_masks = cube_masks.index_select(2, face_ids)

        rot = self._face_rotations.to(device=images.device).index_select(0, face_ids).to(
            dtype=cube_points.dtype
        )
        cube_points = cube_points.permute(0, 1, 2, 4, 5, 3).contiguous()
        cube_points = torch.einsum(
            "bfvhwc,vcd->bfvhwd", cube_points, rot.transpose(-1, -2)
        )

        target_rgb = cube_images.permute(0, 1, 2, 4, 5, 3).contiguous()
        target_alpha = cube_masks.permute(0, 1, 2, 4, 5, 3).contiguous().clamp(0.0, 1.0)
        raw_depth = cube_points[..., 2:3]
        depth_valid = torch.isfinite(raw_depth) & (raw_depth > 1e-4)
        target_depth = torch.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
        target_support = target_alpha * depth_valid.to(target_alpha.dtype)
        return target_rgb, target_depth, target_alpha, target_support

    def _flatten_batch_gaussians(
        self,
        pred: Dict[str, torch.Tensor],
        batch_idx: int,
        high_order_scale: float = 1.0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        means = pred["gaussian_means"][batch_idx].reshape(-1, 3)
        scales = pred["gaussian_scales"][batch_idx].reshape(-1, 3)
        rotations = pred["gaussian_rotations"][batch_idx].reshape(-1, 4)
        opacity = pred["gaussian_opacity"][batch_idx].reshape(-1, 1)
        sh = pred["gaussian_sh"][batch_idx].reshape(-1, pred["gaussian_sh"].shape[-2], 3)
        valid = pred["gaussian_valid_mask"][batch_idx].reshape(-1)

        opacity_scalar = opacity.squeeze(-1)
        finite = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(scales).all(dim=-1)
            & torch.isfinite(rotations).all(dim=-1)
            & torch.isfinite(opacity_scalar)
            & torch.isfinite(sh).all(dim=(-1, -2))
        )
        keep = valid & finite & (opacity_scalar > self.opacity_floor)
        if not keep.any().item():
            return None

        slot_count = keep.numel()
        means = means[keep]
        scales = scales[keep].clamp_min(1e-5)
        rotations = F.normalize(rotations[keep], dim=-1, eps=1e-6)
        opacities = opacity_scalar[keep].clamp(1e-5, 1.0 - 1e-5)
        sh_coeffs = sh[keep]
        if sh_coeffs.shape[-2] > 1 and high_order_scale < 1.0:
            sh_coeffs = sh_coeffs.clone()
            sh_coeffs[..., 1:, :] *= high_order_scale
        return {
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
            "sh_coeffs": sh_coeffs,
            "slot_count": slot_count,
        }

    def _build_face_cameras(
        self, c2w_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = c2w_frames.device
        face_ids = torch.as_tensor(self.face_indices, device=device, dtype=torch.long)

        with torch.autocast(device_type=device.type, enabled=False):
            face_rot = self._face_rotations.to(device=device, dtype=torch.float32).index_select(
                0, face_ids
            )
            face_rot4 = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(
                c2w_frames.shape[0], face_rot.shape[0], 1, 1
            )
            face_rot4[..., :3, :3] = face_rot.view(1, face_rot.shape[0], 3, 3)
            face_c2w = c2w_frames.float().unsqueeze(1) @ face_rot4
            viewmats = torch.linalg.inv(face_c2w)
            K = self._cube_intrinsics(device, torch.float32)
            Ks = K.view(1, 1, 3, 3).expand(
                c2w_frames.shape[0], face_rot.shape[0], 3, 3
            )
        return viewmats.reshape(-1, 4, 4), Ks.reshape(-1, 3, 3)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = gt["imgs"].new_tensor(0.0)
        rasterization = self._get_rasterization()
        required = {
            "gaussian_means",
            "gaussian_scales",
            "gaussian_rotations",
            "gaussian_opacity",
            "gaussian_sh",
            "gaussian_valid_mask",
        }
        if rasterization is None or not required.issubset(pred.keys()):
            return zero, {
                "gaussian_render_loss": zero,
                "gaussian_render_rgb_loss": zero,
                "gaussian_render_depth_loss": zero,
                "gaussian_render_alpha_loss": zero,
                "gaussian_active_count": zero,
                "gaussian_active_ratio": zero,
                "gaussian_mean_opacity": zero,
                "gaussian_mean_scale": zero,
                "gaussian_target_alpha_mean": zero,
                "gaussian_render_alpha_mean": zero,
                "gaussian_sh_high_order_scale": zero,
            }

        frame_indices = torch.arange(
            gt["camera_poses"].shape[1], device=gt["imgs"].device, dtype=torch.long
        )
        if frame_indices.numel() == 0:
            return zero, {
                "gaussian_render_loss": zero,
                "gaussian_render_rgb_loss": zero,
                "gaussian_render_depth_loss": zero,
                "gaussian_render_alpha_loss": zero,
                "gaussian_active_count": zero,
                "gaussian_active_ratio": zero,
                "gaussian_mean_opacity": zero,
                "gaussian_mean_scale": zero,
                "gaussian_target_alpha_mean": zero,
                "gaussian_render_alpha_mean": zero,
                "gaussian_sh_high_order_scale": zero,
            }
        target_rgb, target_depth, target_alpha, target_support = self._prepare_targets(
            gt, frame_indices
        )
        high_order_scale = self._high_order_scale(gt.get("gaussian_progress", 1.0))

        rgb_terms = []
        depth_terms = []
        alpha_terms = []
        active_counts = []
        active_ratios = []
        opacity_means = []
        scale_means = []
        target_alpha_means = []
        render_alpha_means = []

        for batch_idx in range(gt["imgs"].shape[0]):
            gaussian_pack = self._flatten_batch_gaussians(
                pred, batch_idx, high_order_scale=high_order_scale
            )
            if gaussian_pack is None:
                continue

            viewmats, Ks = self._build_face_cameras(
                gt["camera_poses"][batch_idx].index_select(0, frame_indices)
            )
            target_rgb_b = target_rgb[batch_idx].reshape(-1, self.cube_dim, self.cube_dim, 3)
            target_depth_b = target_depth[batch_idx].reshape(-1, self.cube_dim, self.cube_dim, 1)
            target_alpha_b = target_alpha[batch_idx].reshape(-1, self.cube_dim, self.cube_dim, 1)
            target_support_b = target_support[batch_idx].reshape(
                -1, self.cube_dim, self.cube_dim, 1
            )
            target_valid_b = target_support_b > 1e-4

            if not target_valid_b.any().item():
                continue

            far_plane = float(target_depth_b[target_valid_b].amax().item() + 1.0)
            render_colors, render_alphas, _ = rasterization(
                means=gaussian_pack["means"],
                quats=gaussian_pack["rotations"],
                scales=gaussian_pack["scales"],
                opacities=gaussian_pack["opacities"],
                colors=gaussian_pack["sh_coeffs"],
                viewmats=viewmats,
                Ks=Ks,
                width=self.cube_dim,
                height=self.cube_dim,
                near_plane=0.01,
                far_plane=far_plane,
                render_mode="RGB+ED",
                sh_degree=self.sh_degree,
                packed=True,
            )

            render_rgb = render_colors[..., :3]
            render_depth = render_colors[..., 3:4]
            rgb_mask = target_support_b.expand_as(render_rgb)
            rgb_terms.append(_masked_l1(render_rgb, target_rgb_b, rgb_mask))
            depth_terms.append(_masked_l1(render_depth, target_depth_b, target_support_b))
            alpha_terms.append(_balanced_soft_bce(render_alphas, target_alpha_b))
            active_counts.append(render_alphas.new_tensor(float(gaussian_pack["means"].shape[0])))
            active_ratios.append(
                render_alphas.new_tensor(
                    float(gaussian_pack["means"].shape[0]) / max(gaussian_pack["slot_count"], 1)
                )
            )
            opacity_means.append(gaussian_pack["opacities"].mean())
            scale_means.append(gaussian_pack["scales"].mean())
            target_alpha_means.append(target_alpha_b.mean())
            render_alpha_means.append(render_alphas.mean())

        if not rgb_terms:
            return zero, {
                "gaussian_render_loss": zero,
                "gaussian_render_rgb_loss": zero,
                "gaussian_render_depth_loss": zero,
                "gaussian_render_alpha_loss": zero,
                "gaussian_active_count": zero,
                "gaussian_active_ratio": zero,
                "gaussian_mean_opacity": zero,
                "gaussian_mean_scale": zero,
                "gaussian_target_alpha_mean": zero,
                "gaussian_render_alpha_mean": zero,
                "gaussian_sh_high_order_scale": zero.new_tensor(high_order_scale),
            }

        rgb_loss = torch.stack(rgb_terms).mean()
        depth_loss = torch.stack(depth_terms).mean() if depth_terms else zero
        alpha_loss = torch.stack(alpha_terms).mean() if alpha_terms else zero
        active_count = torch.stack(active_counts).mean() if active_counts else zero
        active_ratio = torch.stack(active_ratios).mean() if active_ratios else zero
        mean_opacity = torch.stack(opacity_means).mean() if opacity_means else zero
        mean_scale = torch.stack(scale_means).mean() if scale_means else zero
        target_alpha_mean = torch.stack(target_alpha_means).mean() if target_alpha_means else zero
        render_alpha_mean = torch.stack(render_alpha_means).mean() if render_alpha_means else zero
        total_loss = (
            self.rgb_weight * rgb_loss
            + self.depth_weight * depth_loss
            + self.alpha_weight * alpha_loss
        )
        return total_loss, {
            "gaussian_render_loss": total_loss,
            "gaussian_render_rgb_loss": rgb_loss,
            "gaussian_render_depth_loss": depth_loss,
            "gaussian_render_alpha_loss": alpha_loss,
            "gaussian_active_count": active_count,
            "gaussian_active_ratio": active_ratio,
            "gaussian_mean_opacity": mean_opacity,
            "gaussian_mean_scale": mean_scale,
            "gaussian_target_alpha_mean": target_alpha_mean,
            "gaussian_render_alpha_mean": render_alpha_mean,
            "gaussian_sh_high_order_scale": zero.new_tensor(high_order_scale),
        }
