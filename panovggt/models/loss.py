"""
Loss Functions for 3D Vision Tasks

This module provides loss functions for joint camera pose estimation and 3D point prediction.
It includes scale-invariant point losses, normal consistency losses, and relative pose losses.
"""

import math
import logging
from typing import Dict, Optional, Sequence, Tuple, Union

from panovggt.utils.runtime_env import bootstrap_gsplat_runtime

bootstrap_gsplat_runtime()

import torch
import torch.nn as nn
import torch.nn.functional as F

from panovggt.utils.geometry import homogenize_points, se3_inverse, depth_edge
from panovggt.utils.alignment import align_points_scale
from panovggt.utils.rotation import mat_to_quat, quat_to_mat
from panovggt.utils.gaussian_render import (
    GSPLAT_SH_C0,
    GaussianCubemapRenderLoss,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def weighted_mean(
    x: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    dim: Optional[Union[int, torch.Size]] = None,
    keepdim: bool = False,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute weighted mean along specified dimensions.
    
    Args:
        x (torch.Tensor): Input tensor.
        w (torch.Tensor, optional): Weight tensor with same shape as x.
        dim (int or torch.Size, optional): Dimension(s) to reduce.
        keepdim (bool): Whether to keep reduced dimensions.
        eps (float): Small constant for numerical stability.
    
    Returns:
        torch.Tensor: Weighted mean of x.
    """
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def smooth_l1(
    err: torch.Tensor,
    beta: float = 0.0
) -> torch.Tensor:
    """
    Smooth L1 loss (Huber loss variant).
    
    Args:
        err (torch.Tensor): Error tensor.
        beta (float): Threshold for switching between L1 and L2. If 0, returns err.
    
    Returns:
        torch.Tensor: Smoothed error.
    """
    if beta == 0:
        return err
    else:
        return torch.where(
            err < beta,
            0.5 * err.square() / beta,
            err - 0.5 * beta
        )


def angle_diff_vec3(
    v1: torch.Tensor,
    v2: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute angular difference between two 3D vectors using atan2 for numerical stability.
    
    Args:
        v1 (torch.Tensor): First vector of shape [..., 3].
        v2 (torch.Tensor): Second vector of shape [..., 3].
        eps (float): Small constant for numerical stability.
    
    Returns:
        torch.Tensor: Angular difference in radians, shape [...], range [0, π].
    """
    # Compute cross product norm (proportional to sin(theta))
    cross_product = torch.cross(v1, v2, dim=-1)
    cross_norm = torch.norm(cross_product, dim=-1)
    
    # Compute dot product (proportional to cos(theta))
    dot_product = (v1 * v2).sum(dim=-1)
    
    # Use atan2 for numerically stable angle computation
    angle = torch.atan2(cross_norm + eps, dot_product)
    
    # Ensure angle is in [0, π]
    angle = torch.abs(angle)
    angle = torch.clamp(angle, min=0.0, max=math.pi)
    
    return angle


# =============================================================================
# Point Loss
# =============================================================================

class PointLoss(nn.Module):
    """
    Scale-Invariant Point Loss with Normal Consistency.
    
    This loss computes:
    1. L1 loss on scale-aligned 3D points (local and optional global).
    2. Normal consistency loss based on cross products of neighboring points.
    3. Optional confidence loss for uncertainty estimation.
    
    Args:
        local_align_res (int): Number of points to sample for scale alignment. Default: 4096.
        train_conf (bool): Whether to train confidence prediction. Default: False.
        expected_dist_thresh (float): Distance threshold for positive confidence samples. Default: 0.02.
    """
    
    def __init__(
        self,
        local_align_res: int = 4096,
        train_conf: bool = False,
        expected_dist_thresh: float = 0.02
    ):
        super().__init__()
        self.local_align_res = local_align_res
        self.train_conf = train_conf
        self.expected_dist_thresh = expected_dist_thresh
        
        # Loss functions
        self.criteria_local = nn.L1Loss(reduction='none')
        if self.train_conf:
            self.conf_loss_fn = nn.BCEWithLogitsLoss()
    
    def prepare_sampling(
        self,
        pts: torch.Tensor,
        mask: torch.Tensor,
        target_size: int = 4096
    ) -> torch.Tensor:
        """
        Sample fixed number of valid points for robust scale estimation.
        
        Args:
            pts (torch.Tensor): Points of shape [B, N, H, W, C].
            mask (torch.Tensor): Valid mask of shape [B, N, H, W].
            target_size (int): Number of points to sample.
        
        Returns:
            torch.Tensor: Sampled points of shape [B, target_size, C].
        """
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]  # [M, C]
            
            if valid_pts.shape[0] > 0:
                # Resample to target size
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # [1, C, M]
                valid_pts = F.interpolate(
                    valid_pts, size=target_size, mode='nearest'
                )  # [1, C, target_size]
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # [target_size, C]
            else:
                # Fallback to ones if no valid points
                valid_pts = torch.ones(
                    (target_size, C),
                    device=pts.device,
                    dtype=pts.dtype
                )
            
            output.append(valid_pts)
        
        return torch.stack(output, dim=0)
    
    def compute_normal_loss(
        self,
        points: torch.Tensor,
        gt_points: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normal consistency loss using cross products of neighboring points.
        
        Args:
            points (torch.Tensor): Predicted points of shape [B, N, H, W, 3].
            gt_points (torch.Tensor): Ground truth points of shape [B, N, H, W, 3].
            mask (torch.Tensor): Valid mask of shape [B, N, H, W].
        
        Returns:
            torch.Tensor: Scalar normal loss.
        """
        # Detect edges using radial distance
        radial_distance = torch.norm(gt_points, dim=-1)
        not_edge = ~depth_edge(radial_distance, rtol=0.03)
        mask = torch.logical_and(mask, not_edge)
        
        # Extract 4 corner points for each pixel
        leftup = points[..., :-1, :-1, :]
        rightup = points[..., :-1, 1:, :]
        leftdown = points[..., 1:, :-1, :]
        rightdown = points[..., 1:, 1:, :]
        
        # Compute normals via cross products (4 triangles per pixel)
        upxleft = torch.cross(
            rightup - rightdown, leftdown - rightdown, dim=-1
        )
        leftxdown = torch.cross(
            leftup - rightup, rightdown - rightup, dim=-1
        )
        downxright = torch.cross(
            leftdown - leftup, rightup - leftup, dim=-1
        )
        rightxup = torch.cross(
            rightdown - leftdown, leftup - leftdown, dim=-1
        )
        
        # Same for ground truth
        gt_leftup = gt_points[..., :-1, :-1, :]
        gt_rightup = gt_points[..., :-1, 1:, :]
        gt_leftdown = gt_points[..., 1:, :-1, :]
        gt_rightdown = gt_points[..., 1:, 1:, :]
        
        gt_upxleft = torch.cross(
            gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1
        )
        gt_leftxdown = torch.cross(
            gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1
        )
        gt_downxright = torch.cross(
            gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1
        )
        gt_rightxup = torch.cross(
            gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1
        )
        
        # Compute validity masks for each triangle
        mask_leftup = mask[..., :-1, :-1]
        mask_rightup = mask[..., :-1, 1:]
        mask_leftdown = mask[..., 1:, :-1]
        mask_rightdown = mask[..., 1:, 1:]
        
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown
        
        # Compute angular differences with smoothing
        MIN_ANGLE = math.radians(1)
        MAX_ANGLE = math.radians(90)
        BETA_RAD = math.radians(3)
        
        loss = (
            mask_upxleft * smooth_l1(
                angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            ) +
            mask_leftxdown * smooth_l1(
                angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            ) +
            mask_downxright * smooth_l1(
                angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            ) +
            mask_rightxup * smooth_l1(
                angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            )
        )
        
        # Average over all pixels and triangles
        loss = loss.mean() / (4 * max(points.shape[-3:-1]))
        
        # Safety check for NaN/Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            loss = torch.tensor(0.0, device=points.device, dtype=points.dtype)
        
        return loss
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute point loss.
        
        Args:
            pred (dict): Predictions containing:
                - local_points: [B, N, H, W, 3]
                - conf (optional): [B, N, H, W, 1] or [B, N, H, W]
                - global_points (optional): [B, N, H, W, 3]
            gt (dict): Ground truth containing:
                - local_points: [B, N, H, W, 3]
                - global_points: [B, N, H, W, 3]
                - valid_masks: [B, N, H, W]
        
        Returns:
            Tuple containing:
                - total_loss (torch.Tensor): Scalar total loss.
                - details (dict): Dictionary with individual loss components.
                - scale (torch.Tensor): Estimated scale factors of shape [B].
        """
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        
        B, N, H, W, _ = pred_local_pts.shape
        
        # Compute depth weights (inverse radial distance)
        weights = torch.norm(gt_local_pts, dim=-1).sqrt()
        weights = weights.clamp_min(
            0.1 * weighted_mean(weights, valid_masks, dim=(-2, -1), keepdim=True)
        )
        weights = 1.0 / (weights + 1e-6)
        
        # Scale alignment using sampled points
        with torch.no_grad():
            xyz_pred_sampled = self.prepare_sampling(
                pred_local_pts, valid_masks, target_size=self.local_align_res
            )
            xyz_gt_sampled = self.prepare_sampling(
                gt_local_pts, valid_masks, target_size=self.local_align_res
            )
            xyz_weights_sampled = self.prepare_sampling(
                weights[..., None], valid_masks, target_size=self.local_align_res
            )[..., 0]
            
            scale = align_points_scale(
                xyz_pred_sampled, xyz_gt_sampled, xyz_weights_sampled
            )
            scale = scale.abs().clamp_min(1e-6)
        
        # Apply scale alignment
        aligned_local_pts = scale.view(B, 1, 1, 1, 1) * pred_local_pts
        
        # Compute L1 point loss with depth weighting
        local_pts_loss = self.criteria_local(
            aligned_local_pts[valid_masks].float(),
            gt_local_pts[valid_masks].float()
        ) * weights[valid_masks].float()[..., None]
        
        details = {}
        total_loss = local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()
        
        # Confidence loss (optional)
        if self.train_conf:
            pred_conf = pred['conf']
            if pred_conf.dim() == 5 and pred_conf.shape[-1] == 1:
                pred_conf = pred_conf[..., 0]
            
            # Positive samples: error < threshold
            valid_samples = (
                local_pts_loss.detach().mean(-1) < self.expected_dist_thresh
            )
            
            conf_logits = pred_conf[valid_masks]
            local_conf_loss = self.conf_loss_fn(conf_logits, valid_samples.float())
            
            total_loss += 0.05 * local_conf_loss
            details['local_conf_loss'] = local_conf_loss
        
        # Normal consistency loss
        normal_loss = self.compute_normal_loss(
            aligned_local_pts, gt_local_pts, valid_masks
        )
        total_loss += normal_loss
        details['normal_loss'] = normal_loss
        
        # Global points loss (optional)
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_global_pts = gt['global_points']
            pred_global_pts = pred['global_points'] * scale.view(B, 1, 1, 1, 1)
            
            global_pts_loss = self.criteria_local(
                pred_global_pts[valid_masks].float(),
                gt_global_pts[valid_masks].float()
            ) * weights[valid_masks].float()[..., None]
            
            total_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()
        
        return total_loss, details, scale


# =============================================================================
# Camera Pose Loss
# =============================================================================

class CameraLoss(nn.Module):
    """
    Relative Camera Pose Loss with Adaptive Huber Loss.
    
    Computes relative pose errors between all pairs of frames using:
    1. Rotation angular error (geodesic distance on SO(3)).
    2. Translation Huber loss with adaptive threshold.
    
    Args:
        alpha (float): Weight of translation loss relative to rotation loss. Default: 100.
        delta_real_world (float): Huber loss threshold in real-world units (meters). Default: 0.25.
    """
    
    def __init__(
        self,
        alpha: float = 100.0,
        delta_real_world: float = 0.25
    ):
        super().__init__()
        self.alpha = alpha
        self.delta_real_world = delta_real_world
    
    @staticmethod
    def rotation_angular_error(
        R: torch.Tensor,
        R_gt: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute rotation angular error in radians.
        
        Args:
            R (torch.Tensor): Predicted rotation matrices of shape [B, 3, 3].
            R_gt (torch.Tensor): Ground truth rotation matrices of shape [B, 3, 3].
            eps (float): Small constant for numerical stability.
        
        Returns:
            torch.Tensor: Scalar angular error in radians.
        """
        # Compute relative rotation: R_rel = R^T @ R_gt
        R_rel = torch.matmul(R.transpose(1, 2), R_gt)
        
        # Extract rotation angle from trace: cos(θ) = (trace(R_rel) - 1) / 2
        trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1.0) / 2.0
        
        # Clamp to valid range for numerical stability
        cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
        
        # Compute angle in [0, π]
        angle = torch.acos(cosine)
        
        return angle.mean()
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        scale: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute camera pose loss.
        
        Args:
            pred (dict): Predictions containing:
                - camera_poses: [B, N, 4, 4] in c2w format.
            gt (dict): Ground truth containing:
                - camera_poses: [B, N, 4, 4] in c2w format.
                - norm_factors: [B] normalization factors.
            scale (torch.Tensor): Scale factors from point alignment of shape [B].
        
        Returns:
            Tuple containing:
                - total_loss (torch.Tensor): Scalar total loss.
                - details (dict): Dictionary with 'trans_loss' and 'rot_loss'.
        """
        pred_pose = pred['camera_poses']  # [B, N, 4, 4] c2w
        gt_pose = gt['camera_poses']      # [B, N, 4, 4] c2w
        norm_factors = gt['norm_factors'] # [B]
        
        B, N, _, _ = pred_pose.shape
        
        # Safety check for invalid scale
        if torch.isnan(scale).any() or torch.isinf(scale).any():
            logger.error("Invalid scale in CameraLoss; using fallback value 1.0")
            scale = torch.ones_like(scale)
        
        # Apply scale alignment to predicted translations
        pred_pose_aligned = pred_pose.clone()
        pred_pose_aligned[..., :3, 3] *= scale.view(B, 1, 1)
        
        # Convert to w2c format
        pred_w2c = se3_inverse(pred_pose_aligned)
        gt_w2c = se3_inverse(gt_pose)
        
        # Safety check for NaN in w2c matrices
        if torch.isnan(pred_w2c).any() or torch.isnan(gt_w2c).any():
            logger.error("NaN detected in w2c matrices; returning zero loss")
            zero = torch.zeros((), device=pred_pose.device, dtype=pred_pose.dtype)
            return zero, {'trans_loss': zero, 'rot_loss': zero}
        
        # Compute relative poses: T_j_i = T_w_i @ T_j_w
        pred_rel_all = torch.matmul(
            pred_w2c.unsqueeze(2), pred_pose_aligned.unsqueeze(1)
        )  # [B, N, N, 4, 4]
        gt_rel_all = torch.matmul(
            gt_w2c.unsqueeze(2), gt_pose.unsqueeze(1)
        )  # [B, N, N, 4, 4]
        
        # Exclude diagonal (i == j)
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)
        
        # Extract translations and rotations
        t_pred = pred_rel_all[..., :3, 3][:, mask]   # [B, N*(N-1), 3]
        R_pred = pred_rel_all[..., :3, :3][:, mask]  # [B, N*(N-1), 3, 3]
        t_gt = gt_rel_all[..., :3, 3][:, mask]
        R_gt = gt_rel_all[..., :3, :3][:, mask]
        
        # Adaptive Huber loss for translation
        delta_normalized = self.delta_real_world / (norm_factors + 1e-8)  # [B]
        delta_expanded = delta_normalized.view(B, 1, 1)  # [B, 1, 1]
        
        error = t_pred - t_gt  # [B, P, 3] where P = N*(N-1)
        abs_error = torch.abs(error)
        
        quadratic = 0.5 * (error ** 2)
        linear = delta_expanded * abs_error - 0.5 * (delta_expanded ** 2)
        
        loss_per_element = torch.where(
            abs_error <= delta_expanded, quadratic, linear
        )
        trans_loss = loss_per_element.mean()
        
        # Rotation angular error
        rot_loss = self.rotation_angular_error(
            R_pred.reshape(-1, 3, 3),
            R_gt.reshape(-1, 3, 3)
        )
        
        # Total loss
        total_loss = self.alpha * trans_loss + rot_loss
        
        # Safety check for NaN/Inf
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            logger.error("NaN/Inf detected in total camera loss; returning zero")
            zero = torch.zeros((), device=pred_pose.device, dtype=pred_pose.dtype)
            return zero, {'trans_loss': zero, 'rot_loss': zero}
        
        return total_loss, {'trans_loss': trans_loss, 'rot_loss': rot_loss}


# =============================================================================
# Gaussian Loss
# =============================================================================

class GaussianLoss(nn.Module):
    """
    Geometry-derived supervision for the 3DGS branch.

    Since the training data in this repository does not include rendered
    Gaussian-ground-truth assets, this loss derives token-level pseudo-targets
    from patch-wise canonical-global point statistics and RGB appearance.
    """

    def __init__(
        self,
        patch_size: int = 14,
        max_gaussians_per_token: int = 4,
        sh_degree: int = 3,
        enable_stage1: bool = True,
        keep_target_threshold: float = 0.35,
        min_valid_fraction: float = 0.15,
        center_weight: float = 0.75,
        scale_weight: float = 0.15,
        rotation_weight: float = 0.08,
        opacity_weight: float = 0.08,
        keep_weight: float = 0.05,
        split_weight: float = 0.05,
        sh_weight: float = 0.08,
        sh_rest_reg_weight: float = 0.002,
        sh_rest_warmup_start: float = 0.0,
        sh_rest_warmup_end: float = 0.35,
        offset_reg_weight: float = 0.01,
        opacity_target_min: float = 0.65,
        opacity_target_max: float = 0.95,
        scale_target_min: float = 0.06,
        scale_target_max: float = 0.35,
        scale_target_angular_factor: float = 1.0,
        scale_under_weight: float = 2.0,
        scale_over_weight: float = 0.25,
        scale_floor_weight: float = 0.0,
        opacity_floor_weight: float = 0.0,
        split_detail_thresholds: Optional[Sequence[float]] = None,
        keep_importance_threshold: float = 0.0,
        render_config: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_gaussians_per_token = max_gaussians_per_token
        self.num_sh_bases = (sh_degree + 1) ** 2
        self.enable_stage1 = enable_stage1
        self.keep_target_threshold = float(keep_target_threshold)
        self.min_valid_fraction = min_valid_fraction

        self.center_weight = center_weight
        self.scale_weight = scale_weight
        self.rotation_weight = rotation_weight
        self.opacity_weight = opacity_weight
        self.keep_weight = keep_weight
        self.split_weight = split_weight
        self.sh_weight = sh_weight
        self.sh_rest_reg_weight = sh_rest_reg_weight
        self.sh_rest_warmup_start = float(sh_rest_warmup_start)
        self.sh_rest_warmup_end = float(sh_rest_warmup_end)
        self.offset_reg_weight = offset_reg_weight
        self.opacity_target_min = float(opacity_target_min)
        self.opacity_target_max = float(opacity_target_max)
        self.scale_target_min = float(scale_target_min)
        self.scale_target_max = float(scale_target_max)
        self.scale_target_angular_factor = float(scale_target_angular_factor)
        self.scale_under_weight = float(scale_under_weight)
        self.scale_over_weight = float(scale_over_weight)
        self.scale_floor_weight = float(scale_floor_weight)
        self.opacity_floor_weight = float(opacity_floor_weight)
        if split_detail_thresholds is None:
            split_detail_thresholds = (0.35, 0.55, 0.75)
        self.split_detail_thresholds = tuple(
            float(x) for x in split_detail_thresholds
        )[: max(0, self.max_gaussians_per_token - 1)]
        self.keep_importance_threshold = float(keep_importance_threshold)

        render_config = render_config or {}
        self.render_weight = float(render_config.get("weight", 0.0))
        self.render_loss = None
        if render_config.get("enabled", False) and self.render_weight > 0:
            self.render_loss = GaussianCubemapRenderLoss(
                cube_dim=render_config.get("cube_dim", 64),
                face_indices=render_config.get("face_indices", (0, 1, 2, 3, 4, 5)),
                rgb_weight=render_config.get("rgb_weight", 1.0),
                depth_weight=render_config.get("depth_weight", 0.25),
                alpha_weight=render_config.get("alpha_weight", 0.05),
                opacity_floor=render_config.get("opacity_floor", 1e-4),
                fov_degrees=render_config.get("fov_degrees", 90.0),
                sh_degree=sh_degree,
                sh_warmup_start=render_config.get("sh_warmup_start", 0.0),
                sh_warmup_end=render_config.get("sh_warmup_end", 0.35),
                weight_warmup_start=render_config.get("weight_warmup_start", 0.05),
                weight_warmup_end=render_config.get("weight_warmup_end", 0.25),
            )

    def _high_order_scale(self, progress: object) -> float:
        if isinstance(progress, torch.Tensor):
            progress = float(progress.detach().float().mean().item())
        elif progress is None:
            progress = 1.0
        else:
            progress = float(progress)

        if self.sh_rest_warmup_end <= self.sh_rest_warmup_start:
            return 1.0
        scale = (progress - self.sh_rest_warmup_start) / (
            self.sh_rest_warmup_end - self.sh_rest_warmup_start
        )
        return float(max(0.0, min(1.0, scale)))

    def _patchify_scalar(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H, W = x.shape
        ps = self.patch_size
        Hp, Wp = H // ps, W // ps
        return x.view(B, N, Hp, ps, Wp, ps).permute(0, 1, 2, 4, 3, 5).reshape(
            B, N, Hp, Wp, ps * ps
        )

    def _patchify_vector(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H, W, C = x.shape
        ps = self.patch_size
        Hp, Wp = H // ps, W // ps
        return (
            x.view(B, N, Hp, ps, Wp, ps, C)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, N, Hp, Wp, ps * ps, C)
        )

    def _rotation_loss(
        self,
        pred_quat: torch.Tensor,
        target_quat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        zero = pred_quat.new_tensor(0.0)
        if not mask.any().item():
            return zero

        pred_rot = quat_to_mat(pred_quat)
        target_rot = quat_to_mat(target_quat)
        rel = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
        trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
        cosine = ((trace - 1.0) / 2.0).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angles = torch.acos(cosine)
        return angles[mask].mean()

    def _scale_loss(
        self,
        pred_log_scales: torch.Tensor,
        target_log_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize under-sized Gaussians more strongly than over-sized ones."""
        zero = pred_log_scales.new_tensor(0.0)
        if pred_log_scales.numel() == 0:
            return zero

        diff = pred_log_scales - target_log_scales
        under = F.relu(-diff)
        over = F.relu(diff)
        return (
            self.scale_under_weight * under.mean()
            + self.scale_over_weight * over.mean()
        )

    def _slot_assignments(self, device: torch.device) -> Tuple[torch.Tensor, int, int]:
        num_slots = int(self.max_gaussians_per_token)
        ps = int(self.patch_size)
        if num_slots <= 1:
            return torch.zeros(ps * ps, device=device, dtype=torch.long), 1, 1

        cols = int(math.ceil(math.sqrt(num_slots)))
        rows = int(math.ceil(num_slots / cols))
        y = torch.arange(ps, device=device)
        x = torch.arange(ps, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        row_id = torch.clamp((yy * rows) // ps, max=rows - 1)
        col_id = torch.clamp((xx * cols) // ps, max=cols - 1)
        slot_id = torch.clamp(row_id * cols + col_id, max=num_slots - 1)
        return slot_id.reshape(-1).long(), rows, cols

    def _scale_floor_from_means(
        self,
        means: torch.Tensor,
        image_h: int,
        image_w: int,
        rows: int = 1,
        cols: int = 1,
    ) -> torch.Tensor:
        angular_y = math.pi * self.patch_size / max(float(image_h * rows), 1.0)
        angular_x = 2.0 * math.pi * self.patch_size / max(float(image_w * cols), 1.0)
        angular = max(angular_x, angular_y) * self.scale_target_angular_factor
        radial = means.norm(dim=-1).clamp_min(0.25)
        return (radial * angular).clamp(self.scale_target_min, self.scale_target_max)

    def _cov_to_log_scales_rotations(
        self,
        cov: torch.Tensor,
        valid: torch.Tensor,
        scale_floor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        leading_shape = cov.shape[:-2]
        log_scales = (
            scale_floor.clamp_min(1e-6)
            .log()
            .unsqueeze(-1)
            .expand(*leading_shape, 3)
            .clone()
        )
        rotations = torch.zeros(
            *leading_shape, 4, device=cov.device, dtype=cov.dtype
        )
        rotations[..., 0] = 1.0

        finite_cov = torch.isfinite(cov).all(dim=(-1, -2))
        valid = valid & finite_cov
        if not valid.any().item():
            return log_scales, rotations

        eye = torch.eye(3, device=cov.device, dtype=cov.dtype)
        cov_valid = cov[valid] + 1e-5 * eye
        floor_valid = scale_floor[valid].float()
        valid_log_scales = []
        valid_rotations = []
        cpu_fallback_used = False

        for cov_chunk, floor_chunk in zip(cov_valid.split(8192), floor_valid.split(8192)):
            cov_chunk = cov_chunk.float()
            try:
                eigvals_chunk, eigvecs_chunk = torch.linalg.eigh(cov_chunk)
            except RuntimeError:
                eigvals_chunk_cpu, eigvecs_chunk_cpu = torch.linalg.eigh(cov_chunk.cpu())
                eigvals_chunk = eigvals_chunk_cpu.to(cov_chunk.device)
                eigvecs_chunk = eigvecs_chunk_cpu.to(cov_chunk.device)
                cpu_fallback_used = True

            eigvals_chunk, order = torch.sort(eigvals_chunk, dim=-1, descending=True)
            gather_index = order.unsqueeze(-2).expand(-1, 3, 3)
            eigvecs_chunk = torch.gather(eigvecs_chunk, -1, gather_index)

            det = torch.det(eigvecs_chunk)
            last_col = eigvecs_chunk[..., :, -1] * torch.where(
                det[..., None] < 0, -1.0, 1.0
            )
            eigvecs_chunk = torch.cat(
                [eigvecs_chunk[..., :, :2], last_col.unsqueeze(-1)], dim=-1
            )

            scales_chunk = torch.sqrt(eigvals_chunk.clamp_min(1e-6))
            floor_chunk = floor_chunk[:, None].expand_as(scales_chunk)
            scales_chunk = torch.maximum(scales_chunk, floor_chunk)
            scales_chunk = scales_chunk.clamp(
                min=self.scale_target_min, max=self.scale_target_max
            )
            log_scales_chunk = torch.log(scales_chunk.clamp_min(1e-6))
            rotations_chunk = mat_to_quat(eigvecs_chunk)
            rotations_chunk = torch.nan_to_num(
                rotations_chunk, nan=0.0, posinf=0.0, neginf=0.0
            )
            rotations_norm = rotations_chunk.norm(dim=-1, keepdim=True)
            default_quat = torch.zeros_like(rotations_chunk)
            default_quat[..., 0] = 1.0
            rotations_chunk = torch.where(
                rotations_norm > 1e-6,
                rotations_chunk / rotations_norm.clamp_min(1e-6),
                default_quat,
            )

            valid_log_scales.append(log_scales_chunk.to(log_scales.dtype))
            valid_rotations.append(rotations_chunk.to(rotations.dtype))

        if cpu_fallback_used:
            logging.warning("Gaussian target eigendecomposition fell back to CPU for at least one chunk.")

        log_scales[valid] = torch.cat(valid_log_scales, dim=0)
        rotations[valid] = torch.cat(valid_rotations, dim=0)
        return log_scales, rotations

    def _prepare_targets(self, gt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_points = gt["global_points"]
        valid_masks = gt["valid_masks"].bool()
        images = gt["imgs"].permute(0, 1, 3, 4, 2).contiguous()
        image_h, image_w = images.shape[2], images.shape[3]

        finite_global = torch.isfinite(global_points).all(dim=-1)
        finite_rgb = torch.isfinite(images).all(dim=-1)
        valid_masks = valid_masks & finite_global & finite_rgb

        global_points = torch.nan_to_num(global_points, nan=0.0, posinf=0.0, neginf=0.0)
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)

        global_patches = self._patchify_vector(global_points)
        image_patches = self._patchify_vector(images)
        valid_patches = self._patchify_scalar(valid_masks.float())

        weights = valid_patches
        counts = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        valid_fraction = counts.squeeze(-1) / float(self.patch_size * self.patch_size)
        patch_valid = valid_fraction > self.min_valid_fraction

        patch_means = (global_patches * weights[..., None]).sum(dim=-2) / counts
        patch_rgb = (image_patches * weights[..., None]).sum(dim=-2) / counts

        centered_points = (global_patches - patch_means.unsqueeze(-2)) * weights[..., None]
        cov = torch.einsum(
            "bnhwpd,bnhwpe->bnhwde", centered_points, centered_points
        ) / counts[..., None]
        eye = torch.eye(3, device=cov.device, dtype=cov.dtype).view(1, 1, 1, 1, 3, 3)
        cov = cov + 1e-5 * eye
        finite_cov = torch.isfinite(cov).all(dim=(-1, -2))
        cov = torch.where(finite_cov[..., None, None], cov, eye.expand_as(cov))
        patch_valid = patch_valid & finite_cov

        scale_floor = self._scale_floor_from_means(
            patch_means, image_h=image_h, image_w=image_w
        )
        log_scales, rotations = self._cov_to_log_scales_rotations(
            cov, patch_valid, scale_floor
        )

        slot_ids, slot_rows, slot_cols = self._slot_assignments(global_points.device)
        one_hot = F.one_hot(slot_ids, num_classes=self.max_gaussians_per_token).to(
            dtype=weights.dtype
        )
        slot_weights = weights.unsqueeze(-1) * one_hot.view(
            1, 1, 1, 1, self.patch_size * self.patch_size, self.max_gaussians_per_token
        )
        slot_counts = slot_weights.sum(dim=-2).clamp_min(1.0)
        min_slot_pixels = max(
            1.0,
            float(self.patch_size * self.patch_size)
            / self.max_gaussians_per_token
            * self.min_valid_fraction,
        )
        slot_valid = patch_valid.unsqueeze(-1) & (slot_counts > min_slot_pixels)
        slot_means = torch.einsum(
            "bnhwpk,bnhwpc->bnhwkc", slot_weights, global_patches
        ) / slot_counts[..., None]
        slot_rgb = torch.einsum(
            "bnhwpk,bnhwpc->bnhwkc", slot_weights, image_patches
        ) / slot_counts[..., None]
        slot_means = torch.where(
            slot_valid[..., None], slot_means, patch_means.unsqueeze(-2)
        )
        slot_rgb = torch.where(
            slot_valid[..., None], slot_rgb, patch_rgb.unsqueeze(-2)
        )
        slot_centered = global_patches.unsqueeze(-2) - slot_means.unsqueeze(-3)
        slot_cov = torch.einsum(
            "bnhwpkd,bnhwpke->bnhwkde",
            slot_centered * slot_weights.unsqueeze(-1),
            slot_centered,
        ) / slot_counts[..., None, None]
        slot_scale_floor = self._scale_floor_from_means(
            slot_means,
            image_h=image_h,
            image_w=image_w,
            rows=slot_rows,
            cols=slot_cols,
        )
        slot_log_scales, slot_rotations = self._cov_to_log_scales_rotations(
            slot_cov, slot_valid, slot_scale_floor
        )

        rgb_var = (
            ((image_patches - patch_rgb.unsqueeze(-2)) ** 2) * weights[..., None]
        ).sum(dim=(-2, -1)) / counts.squeeze(-1)
        geom_var = log_scales.exp().mean(dim=-1)
        rgb_var = torch.where(patch_valid, rgb_var, torch.zeros_like(rgb_var))
        geom_var = torch.where(patch_valid, geom_var, torch.zeros_like(geom_var))
        rgb_detail_abs = (rgb_var.clamp_min(0.0).sqrt() / 0.25).clamp(0.0, 1.0)
        geom_excess = (
            (geom_var - scale_floor).clamp_min(0.0)
            / scale_floor.clamp_min(1e-6)
        )
        geom_detail_abs = (geom_excess / 2.0).clamp(0.0, 1.0)
        split_detail = (
            0.40 * rgb_detail_abs + 0.60 * geom_detail_abs
        ) * patch_valid.float()

        importance_target = (
            0.50 * valid_fraction + 0.50 * split_detail
        ).clamp_(0.0, 1.0)
        if self.keep_importance_threshold > 0:
            keep_target = patch_valid & (
                importance_target >= self.keep_importance_threshold
            )
        else:
            keep_target = patch_valid
        opacity_span = max(0.0, self.opacity_target_max - self.opacity_target_min)
        opacity_target = self.opacity_target_min + opacity_span * importance_target
        opacity_target = torch.where(
            patch_valid,
            opacity_target.clamp(0.0, 1.0),
            torch.zeros_like(importance_target),
        )
        split_target = torch.ones_like(
            slot_valid.long().sum(dim=-1),
            dtype=torch.long,
        )
        for threshold in self.split_detail_thresholds:
            split_target = split_target + (split_detail >= threshold).long()
        slot_valid_count = slot_valid.long().sum(dim=-1).clamp_min(1)
        split_target = torch.minimum(split_target, slot_valid_count).clamp(
            min=1, max=self.max_gaussians_per_token
        )
        split_target = torch.where(
            patch_valid, split_target, torch.ones_like(split_target)
        )

        sh_target = torch.zeros(
            *patch_rgb.shape[:-1],
            self.num_sh_bases,
            3,
            device=patch_rgb.device,
            dtype=patch_rgb.dtype,
        )
        sh_target[..., 0, :] = (
            patch_rgb.clamp(0.0, 1.0) - 0.5
        ) / GSPLAT_SH_C0
        slot_sh_target = torch.zeros(
            *slot_rgb.shape[:-1],
            self.num_sh_bases,
            3,
            device=slot_rgb.device,
            dtype=slot_rgb.dtype,
        )
        slot_sh_target[..., 0, :] = (
            slot_rgb.clamp(0.0, 1.0) - 0.5
        ) / GSPLAT_SH_C0

        return {
            "patch_valid": patch_valid,
            "means": patch_means,
            "log_scales": log_scales,
            "rotations": rotations,
            "slot_valid": slot_valid,
            "slot_means": slot_means,
            "slot_log_scales": slot_log_scales,
            "slot_rotations": slot_rotations,
            "importance": importance_target,
            "detail": split_detail,
            "keep_target": keep_target,
            "opacity_target": opacity_target,
            "split_target": split_target,
            "sh_target": sh_target,
            "slot_sh_target": slot_sh_target,
        }

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if "gaussian_token_means" not in pred:
            zero = gt["global_points"].new_tensor(0.0)
            return zero, {}

        target = self._prepare_targets(gt)
        patch_valid = target["patch_valid"]
        zero = pred["gaussian_token_means"].new_tensor(0.0)
        center_loss = zero
        scale_loss = zero
        rotation_loss = zero
        opacity_loss = zero
        keep_loss = zero
        split_loss = zero
        sh_loss = zero
        sh_dc_loss = zero
        sh_rest_reg = zero
        offset_reg = zero
        scale_floor_loss = zero
        opacity_floor_loss = zero
        target_keep_ratio = zero
        target_importance_mean = zero
        target_opacity_mean = zero
        pred_keep_prob_mean = zero
        target_split_mean = zero
        pred_split_mean = zero
        sh_high_order_scale = self._high_order_scale(gt.get("gaussian_progress", 1.0))
        keep_target = target["keep_target"]

        if self.enable_stage1 and "gaussian_keep_logits" in pred:
            keep_loss = F.binary_cross_entropy_with_logits(
                pred["gaussian_keep_logits"].squeeze(-1),
                keep_target.float(),
            )
            if "gaussian_keep_prob" in pred:
                pred_keep_prob_mean = pred["gaussian_keep_prob"].squeeze(-1).mean()
            if "gaussian_split_count" in pred and patch_valid.any().item():
                pred_split_mean = pred["gaussian_split_count"][patch_valid].float().mean()
            target_keep_ratio = keep_target.float().mean()

        slot_valid = target.get("slot_valid")
        base_valid = patch_valid

        if patch_valid.any().item():
            center_target = target["means"]
            scale_target = target["log_scales"]
            rotation_target = target["rotations"]
            if slot_valid is not None:
                center_target = target["slot_means"][..., 0, :]
                scale_target = target["slot_log_scales"][..., 0, :]
                rotation_target = target["slot_rotations"][..., 0, :]

            center_loss = F.smooth_l1_loss(
                pred["gaussian_token_means"][base_valid],
                center_target[base_valid],
            )
            scale_loss = self._scale_loss(
                pred["gaussian_token_log_scales"][base_valid],
                scale_target[base_valid],
            )
            if (
                "gaussian_child_log_scales" in pred
                and pred["gaussian_child_log_scales"].numel() > 0
            ):
                child_valid = patch_valid.unsqueeze(-1).expand(
                    pred["gaussian_child_log_scales"].shape[:-1]
                )
                child_scale_target = target["log_scales"].unsqueeze(-2).expand_as(
                    pred["gaussian_child_log_scales"]
                )
                if slot_valid is not None and target["slot_log_scales"].shape[-2] > 1:
                    child_valid = slot_valid[..., 1:]
                    child_scale_target = target["slot_log_scales"][..., 1:, :]
                if child_valid.any().item():
                    scale_loss = 0.5 * (
                        scale_loss
                        + self._scale_loss(
                            pred["gaussian_child_log_scales"][child_valid],
                            child_scale_target[child_valid],
                        )
                    )
            if self.scale_floor_weight > 0:
                all_log_scales = [pred["gaussian_token_log_scales"][base_valid]]
                if (
                    "gaussian_child_log_scales" in pred
                    and pred["gaussian_child_log_scales"].numel() > 0
                ):
                    all_log_scales.append(
                        pred["gaussian_child_log_scales"][patch_valid].reshape(-1, 3)
                    )
                pred_log_scale_all = torch.cat(all_log_scales, dim=0)
                scale_floor = math.log(max(self.scale_target_min, 1e-6))
                scale_floor_loss = F.relu(
                    pred_log_scale_all.new_tensor(scale_floor) - pred_log_scale_all
                ).mean()
            rotation_loss = self._rotation_loss(
                pred["gaussian_token_rotations"],
                rotation_target,
                base_valid,
            )
            if (
                "gaussian_child_rotations" in pred
                and pred["gaussian_child_rotations"].numel() > 0
            ):
                child_rotation_target = target["rotations"].unsqueeze(-2).expand_as(
                    pred["gaussian_child_rotations"]
                )
                child_rotation_mask = patch_valid.unsqueeze(-1).expand(
                    pred["gaussian_child_rotations"].shape[:-1]
                )
                if slot_valid is not None and target["slot_rotations"].shape[-2] > 1:
                    child_rotation_target = target["slot_rotations"][..., 1:, :]
                    child_rotation_mask = slot_valid[..., 1:]
                child_rotation_loss = self._rotation_loss(
                    pred["gaussian_child_rotations"],
                    child_rotation_target,
                    child_rotation_mask,
                )
                rotation_loss = 0.5 * (rotation_loss + child_rotation_loss)

            if (
                "gaussian_child_offsets" in pred
                and pred["gaussian_child_offsets"].numel() > 0
                and slot_valid is not None
                and target["slot_means"].shape[-2] > 1
            ):
                child_means = (
                    pred["gaussian_token_means"].unsqueeze(-2)
                    + pred["gaussian_child_offsets"]
                )
                child_center_mask = slot_valid[..., 1:]
                if child_center_mask.any().item():
                    center_loss = 0.5 * (
                        center_loss
                        + F.smooth_l1_loss(
                            child_means[child_center_mask],
                            target["slot_means"][..., 1:, :][child_center_mask],
                        )
                    )

            opacity_logits = pred["gaussian_token_opacity_logits"].squeeze(-1)
            opacity_target = target["opacity_target"]
            target_importance_mean = target["importance"][patch_valid].mean()
            target_opacity_mean = opacity_target[patch_valid].mean()
            opacity_mask = patch_valid & keep_target
            if opacity_mask.any().item():
                opacity_loss = F.binary_cross_entropy_with_logits(
                    opacity_logits[opacity_mask],
                    opacity_target[opacity_mask],
                )
                if (
                    "gaussian_child_opacity_logits" in pred
                    and pred["gaussian_child_opacity_logits"].numel() > 0
                ):
                    child_opacity_logits = pred["gaussian_child_opacity_logits"][
                        opacity_mask
                    ]
                    child_opacity_target = opacity_target[opacity_mask].unsqueeze(-1)
                    child_opacity_target = child_opacity_target.expand_as(
                        child_opacity_logits.squeeze(-1)
                    )
                    opacity_loss = 0.5 * (
                        opacity_loss
                        + F.binary_cross_entropy_with_logits(
                            child_opacity_logits.squeeze(-1),
                            child_opacity_target,
                        )
                    )
                if self.opacity_floor_weight > 0:
                    all_opacity_logits = [opacity_logits[opacity_mask]]
                    if (
                        "gaussian_child_opacity_logits" in pred
                        and pred["gaussian_child_opacity_logits"].numel() > 0
                    ):
                        all_opacity_logits.append(
                            pred["gaussian_child_opacity_logits"][opacity_mask]
                            .squeeze(-1)
                            .reshape(-1)
                        )
                    opacity_all = torch.sigmoid(torch.cat(all_opacity_logits, dim=0))
                    opacity_floor = max(self.opacity_target_min, 0.0)
                    opacity_floor_loss = F.relu(
                        opacity_all.new_tensor(opacity_floor) - opacity_all
                    ).mean()

            if self.enable_stage1 and "gaussian_split_logits" in pred:
                split_mask = patch_valid & keep_target
                if split_mask.any().item():
                    split_logits = pred["gaussian_split_logits"][split_mask]
                    split_target = target["split_target"][split_mask] - 1
                    target_split_mean = target["split_target"][split_mask].float().mean()
                    split_loss = F.cross_entropy(split_logits, split_target)

            pred_sh = pred["gaussian_token_sh"][patch_valid]
            target_sh = target["sh_target"][patch_valid]
            if slot_valid is not None:
                pred_sh = pred["gaussian_token_sh"][base_valid]
                target_sh = target["slot_sh_target"][..., 0, :, :][base_valid]
            sh_dc_loss = F.l1_loss(pred_sh[:, 0], target_sh[:, 0])
            if "gaussian_child_sh" in pred and pred["gaussian_child_sh"].numel() > 0:
                if slot_valid is not None and target["slot_sh_target"].shape[-3] > 1:
                    child_sh_mask = slot_valid[..., 1:]
                    if child_sh_mask.any().item():
                        child_sh = pred["gaussian_child_sh"][child_sh_mask]
                        child_target_sh = target["slot_sh_target"][..., 1:, :, :][
                            child_sh_mask
                        ]
                        sh_dc_loss = 0.5 * (
                            sh_dc_loss
                            + F.l1_loss(
                                child_sh[..., 0, :],
                                child_target_sh[..., 0, :],
                            )
                        )
                else:
                    child_sh = pred["gaussian_child_sh"][patch_valid]
                    child_target_sh = target_sh[:, None, 0, :].expand_as(
                        child_sh[..., 0, :]
                    )
                    sh_dc_loss = 0.5 * (
                        sh_dc_loss + F.l1_loss(child_sh[..., 0, :], child_target_sh)
                    )
            sh_loss = sh_dc_loss
            if pred_sh.shape[1] > 1:
                sh_rest_reg = pred_sh[:, 1:].pow(2).mean()
                if (
                    "gaussian_child_sh" in pred
                    and pred["gaussian_child_sh"].numel() > 0
                ):
                    child_sh = pred["gaussian_child_sh"][patch_valid]
                    sh_rest_reg = 0.5 * (
                        sh_rest_reg + child_sh[..., 1:, :].pow(2).mean()
                    )

            if (
                "gaussian_child_offsets" in pred
                and pred["gaussian_child_offsets"].numel() > 0
            ):
                offset_source = pred.get(
                    "gaussian_child_residual_offsets",
                    pred["gaussian_child_offsets"],
                )
                offset_reg = offset_source[patch_valid].norm(dim=-1).mean()

        render_loss = zero
        render_details = {
            "gaussian_render_loss": zero,
            "gaussian_render_rgb_loss": zero,
            "gaussian_render_depth_loss": zero,
            "gaussian_render_alpha_loss": zero,
        }
        if self.render_loss is not None:
            render_loss, render_details = self.render_loss(pred, gt)

        total_loss = (
            self.center_weight * center_loss
            + self.scale_weight * scale_loss
            + self.rotation_weight * rotation_loss
            + self.opacity_weight * opacity_loss
            + self.keep_weight * keep_loss
            + self.split_weight * split_loss
            + self.sh_weight * sh_loss
            + self.sh_rest_reg_weight * (1.0 - sh_high_order_scale) * sh_rest_reg
            + self.offset_reg_weight * offset_reg
            + self.scale_floor_weight * scale_floor_loss
            + self.opacity_floor_weight * opacity_floor_loss
            + self.render_weight * render_loss
        )

        details = {
            "gaussian_center_loss": center_loss,
            "gaussian_scale_loss": scale_loss,
            "gaussian_rotation_loss": rotation_loss,
            "gaussian_opacity_loss": opacity_loss,
            "gaussian_keep_loss": keep_loss,
            "gaussian_split_loss": split_loss,
            "gaussian_sh_loss": sh_loss,
            "gaussian_sh_dc_loss": sh_dc_loss,
            "gaussian_sh_rest_reg": sh_rest_reg,
            "gaussian_sh_high_order_scale": zero.new_tensor(sh_high_order_scale),
            "gaussian_offset_reg": offset_reg,
            "gaussian_scale_floor_loss": scale_floor_loss,
            "gaussian_opacity_floor_loss": opacity_floor_loss,
            "gaussian_target_keep_ratio": target_keep_ratio,
            "gaussian_target_importance": target_importance_mean,
            "gaussian_target_opacity": target_opacity_mean,
            "gaussian_pred_keep_prob": pred_keep_prob_mean,
            "gaussian_target_split": target_split_mean,
            "gaussian_pred_split": pred_split_mean,
        }
        details.update(render_details)
        return total_loss, details


# =============================================================================
# Combined Loss
# =============================================================================

class Loss(nn.Module):
    """
    Combined Loss for Joint Point and Camera Estimation.
    
    This module combines:
    1. Point loss (local/global points + normal consistency + optional confidence).
    2. Camera pose loss (relative rotation + translation).
    
    Args:
        train_conf (bool): Whether to train confidence prediction. Default: False.
    """
    
    def __init__(
        self,
        train_conf: bool = False,
        enable_3dgs: bool = False,
        gaussian_head: Optional[Dict[str, Union[int, float, bool]]] = None,
        gaussian_render: Optional[Dict[str, object]] = None,
        gaussian_supervision: Optional[Dict[str, object]] = None,
        point_loss_weight: float = 1.0,
        camera_loss_weight: float = 0.1,
        gaussian_loss_weight: float = 0.2,
        patch_size: int = 14,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
        self.enable_3dgs = enable_3dgs
        self.point_loss_weight = point_loss_weight
        self.camera_loss_weight = camera_loss_weight
        self.gaussian_loss_weight = gaussian_loss_weight

        gaussian_head = gaussian_head or {}
        self.gaussian_loss = (
            GaussianLoss(
                patch_size=patch_size,
                max_gaussians_per_token=gaussian_head.get(
                    "max_gaussians_per_token", 4
                ),
                sh_degree=gaussian_head.get("sh_degree", 3),
                enable_stage1=gaussian_head.get("enable_stage1", True),
                keep_target_threshold=gaussian_head.get("keep_threshold", 0.35),
                render_config=gaussian_render,
                **(gaussian_supervision or {}),
            )
            if enable_3dgs
            else None
        )

    @staticmethod
    def _compute_norm_factor(
        local_points: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        B = local_points.shape[0]
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, local_points.shape[1], -1, 3)
        all_dis = all_pts.norm(dim=-1)
        denom = masks.float().sum(dim=[-1, -2, -3]).clamp(min=1e-8)
        return all_dis.sum(dim=[-1, -2]) / denom

    @staticmethod
    def _transform_points_to_cam0(
        points: torch.Tensor,
        camera_poses: torch.Tensor,
        norm_factor: torch.Tensor,
    ) -> torch.Tensor:
        B = points.shape[0]
        R0 = camera_poses[:, 0, :3, :3]
        t0 = camera_poses[:, 0, :3, 3]
        t_w2c = -torch.matmul(t0.unsqueeze(-2), R0).squeeze(-2)
        rot = R0.view(B, *([1] * (points.dim() - 2)), 3, 3)
        trans = t_w2c.view(B, *([1] * (points.dim() - 2)), 3)
        scale = norm_factor.view(B, *([1] * (points.dim() - 1)))
        return (torch.einsum("...j,...jk->...k", points, rot) + trans) / scale

    def prepare_gaussian_gt(
        self, gt: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        existing_norm = gt.get("norm_factors")
        if isinstance(existing_norm, torch.Tensor):
            gaussian_gt = dict(gt)
            # The training pipeline normalizes points/depths/cameras before the
            # loss and keeps the original scale in norm_factors for reference.
            # Gaussian rendering consumes already-normalized local depths here,
            # so expose unit factors to avoid scaling the depth targets twice.
            gaussian_gt["norm_factors"] = torch.ones_like(
                existing_norm,
                device=gt["global_points"].device,
                dtype=gt["global_points"].dtype,
            )
            return gaussian_gt

        norm_factor = self._compute_norm_factor(
            gt["local_points"], gt["valid_masks"]
        ).clamp_min(1e-6)
        canonical_global_points = self._transform_points_to_cam0(
            gt["global_points"], gt["camera_poses"], norm_factor
        )
        canonical_camera_poses = se3_inverse(gt["camera_poses"][:, :1]) @ gt["camera_poses"]
        canonical_camera_poses = canonical_camera_poses.clone()
        canonical_camera_poses[..., :3, 3] /= norm_factor.view(-1, 1, 1)

        gaussian_gt = dict(gt)
        gaussian_gt["global_points"] = canonical_global_points
        gaussian_gt["camera_poses"] = canonical_camera_poses
        gaussian_gt["norm_factors"] = norm_factor
        return gaussian_gt
    
    def prepare_gt(self, gt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare ground truth data for loss computation.
        
        Converts extrinsics from w2c to c2w format and ensures 4x4 matrices.
        
        Args:
            gt (dict): Raw ground truth containing:
                - extrinsics: [B, N, 3, 4] or [B, N, 4, 4] in w2c format.
                - world_points: [B, N, H, W, 3].
                - cam_points: [B, N, H, W, 3].
                - point_masks: [B, N, H, W].
                - images: [B, N, C, H, W].
                - norm_factors (optional): [B].
                - depths (optional): [B, N, H, W].
        
        Returns:
            dict: Processed ground truth with c2w camera poses.
        """
        poses_w2c = gt['extrinsics']
        
        # Convert [B, N, 3, 4] to [B, N, 4, 4] if needed
        if poses_w2c.shape[-2:] == (3, 4):
            B, N, _, _ = poses_w2c.shape
            bottom = poses_w2c.new_zeros((B, N, 1, 4))
            bottom[..., 0, 3] = 1.0
            poses_w2c = torch.cat([poses_w2c, bottom], dim=-2)
        
        # Convert w2c to c2w
        poses_c2w = se3_inverse(poses_w2c)
        
        return {
            'imgs': gt['images'],
            'global_points': gt['world_points'],
            'local_points': gt['cam_points'],
            'valid_masks': gt['point_masks'],
            'camera_poses': poses_c2w,
            'depths': gt.get('depths', None),
            'norm_factors': gt.get('norm_factors', None),
            'gaussian_progress': gt.get('gaussian_progress', None),
        }
    
    def normalize_pred(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Normalize predictions using scale computed from local points.
        
        Args:
            pred (dict): Predictions containing:
                - local_points: [B, N, H, W, 3].
                - camera_poses: [B, N, 4, 4] in c2w format.
                - global_points (optional): [B, N, H, W, 3].
            gt (dict): Ground truth with 'valid_masks'.
        
        Returns:
            dict: Normalized predictions.
        """
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        masks = gt['valid_masks']
        
        B, N, H, W, _ = local_points.shape
        
        # Compute normalization scale from local points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        denom = masks.float().sum(dim=[-1, -2, -3]).clamp(min=1e-8)
        norm_factor = all_dis.sum(dim=[-1, -2]) / denom  # [B]
        
        scale = norm_factor.view(B, 1, 1, 1, 1)
        R0 = camera_poses[:, 0, :3, :3]  # [B, 3, 3]
        t0 = camera_poses[:, 0, :3, 3]   # [B, 3]
        t_w2c = -torch.matmul(t0.unsqueeze(-2), R0).squeeze(-2)  # [B, 3]

        def expand_like(tensor: torch.Tensor) -> torch.Tensor:
            return norm_factor.view(B, *([1] * (tensor.dim() - 1)))

        def transform_points_to_cam0(points: torch.Tensor) -> torch.Tensor:
            rot = R0.view(B, *([1] * (points.dim() - 2)), 3, 3)
            trans = t_w2c.view(B, *([1] * (points.dim() - 2)), 3)
            return torch.einsum("...j,...jk->...k", points, rot) + trans

        def transform_rotations_to_cam0(quaternions: torch.Tensor) -> torch.Tensor:
            rot_mats = quat_to_mat(quaternions)
            rot = R0.view(B, *([1] * (rot_mats.dim() - 3)), 3, 3)
            rot_mats = torch.einsum("...ij,...jk->...ik", rot, rot_mats)
            return mat_to_quat(rot_mats)

        # Normalize local points
        pred['local_points'] = local_points / scale

        # Normalize global points if present
        if 'global_points' in pred and pred['global_points'] is not None:
            global_points = pred['global_points']
            pred['global_points'] = transform_points_to_cam0(global_points) / scale

        # Normalize camera translations
        camera_poses_normalized = se3_inverse(camera_poses[:, :1]) @ camera_poses
        camera_poses_normalized = camera_poses_normalized.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)
        pred['camera_poses'] = camera_poses_normalized
        
        return pred
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_raw: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components.
        
        Args:
            pred (dict): Model predictions.
            gt_raw (dict): Raw ground truth data.
        
        Returns:
            dict: Loss dictionary containing:
                - loss_objective: Total loss for optimization.
                - loss_camera: Camera pose loss.
                - loss_T: Translation loss.
                - loss_R: Rotation loss.
                - loss_conf_point: Point confidence loss.
                - loss_reg_point: Point regression loss.
                - loss_grad_point: Normal consistency loss.
                - loss_local_point: Local point loss.
                - loss_global_point: Global point loss.
                - loss_conf_depth: Depth confidence loss (unused, set to 0).
                - loss_reg_depth: Depth regression loss (unused, set to 0).
                - loss_grad_depth: Depth gradient loss (unused, set to 0).
        """
        # Prepare ground truth
        gt = self.prepare_gt(gt_raw)
        gaussian_gt = self.prepare_gaussian_gt(gt)

        gaussian_loss = loss_objective = None
        gaussian_details = {}
        if self.gaussian_loss is not None:
            gaussian_loss, gaussian_details = self.gaussian_loss(pred, gaussian_gt)
        else:
            gaussian_loss = gt["global_points"].new_tensor(0.0)

        zero = gaussian_loss.new_tensor(0.0)
        point_loss = zero
        cam_loss = zero
        point_details = {}
        cam_details = {}

        # Normalize predictions for point/camera supervision only. Gaussian-only
        # training freezes or disables these branches, so avoid spending time on
        # losses that have zero objective weight.
        needs_point_scale = self.point_loss_weight > 0 or self.camera_loss_weight > 0
        if needs_point_scale:
            point_pred = self.normalize_pred(dict(pred), gt)
            point_loss, point_details, scale = self.point_loss(point_pred, gt)
            if self.camera_loss_weight > 0:
                cam_loss, cam_details = self.camera_loss(point_pred, gt, scale)

        # Total objective loss
        loss_objective = (
            self.point_loss_weight * point_loss
            + self.camera_loss_weight * cam_loss
            + self.gaussian_loss_weight * gaussian_loss
        )
        
        # Helper to convert to tensor
        def as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(
                x, device=loss_objective.device, dtype=loss_objective.dtype
            )
        
        zero = loss_objective.new_tensor(0.0)
        
        # Construct unified loss dictionary
        loss_dict = {
            # Total objective
            'loss_objective': loss_objective,
            
            # Camera losses
            'loss_camera': as_tensor(cam_loss),
            'loss_T': as_tensor(cam_details.get('trans_loss', zero)),
            'loss_R': as_tensor(cam_details.get('rot_loss', zero)),
            
            # Depth losses (not currently used, set to zero)
            'loss_conf_depth': zero,
            'loss_reg_depth': zero,
            'loss_grad_depth': zero,
            
            # Point losses
            'loss_conf_point': as_tensor(point_details.get('local_conf_loss', zero)),
            'loss_reg_point': (
                as_tensor(point_details.get('local_pts_loss', zero)) +
                as_tensor(point_details.get('global_pts_loss', zero))
            ),
            'loss_grad_point': as_tensor(point_details.get('normal_loss', zero)),
            'loss_local_point': as_tensor(point_details.get('local_pts_loss', zero)),
            'loss_global_point': as_tensor(point_details.get('global_pts_loss', zero)),
            'loss_gaussian': as_tensor(gaussian_loss),
            'loss_gaussian_center': as_tensor(gaussian_details.get('gaussian_center_loss', zero)),
            'loss_gaussian_scale': as_tensor(gaussian_details.get('gaussian_scale_loss', zero)),
            'loss_gaussian_rotation': as_tensor(gaussian_details.get('gaussian_rotation_loss', zero)),
            'loss_gaussian_opacity': as_tensor(gaussian_details.get('gaussian_opacity_loss', zero)),
            'loss_gaussian_keep': as_tensor(gaussian_details.get('gaussian_keep_loss', zero)),
            'loss_gaussian_split': as_tensor(gaussian_details.get('gaussian_split_loss', zero)),
            'loss_gaussian_sh': as_tensor(gaussian_details.get('gaussian_sh_loss', zero)),
            'loss_gaussian_sh_dc': as_tensor(gaussian_details.get('gaussian_sh_dc_loss', zero)),
            'loss_gaussian_sh_rest_reg': as_tensor(gaussian_details.get('gaussian_sh_rest_reg', zero)),
            'loss_gaussian_sh_high_order_scale': as_tensor(gaussian_details.get('gaussian_sh_high_order_scale', zero)),
            'loss_gaussian_offset_reg': as_tensor(gaussian_details.get('gaussian_offset_reg', zero)),
            'loss_gaussian_scale_floor': as_tensor(gaussian_details.get('gaussian_scale_floor_loss', zero)),
            'loss_gaussian_opacity_floor': as_tensor(gaussian_details.get('gaussian_opacity_floor_loss', zero)),
            'loss_gaussian_render': as_tensor(gaussian_details.get('gaussian_render_loss', zero)),
            'loss_gaussian_render_rgb': as_tensor(gaussian_details.get('gaussian_render_rgb_loss', zero)),
            'loss_gaussian_render_depth': as_tensor(gaussian_details.get('gaussian_render_depth_loss', zero)),
            'loss_gaussian_render_alpha': as_tensor(gaussian_details.get('gaussian_render_alpha_loss', zero)),
            'loss_gaussian_render_weight_scale': as_tensor(gaussian_details.get('gaussian_render_weight_scale', zero)),
            'loss_gaussian_target_keep_ratio': as_tensor(gaussian_details.get('gaussian_target_keep_ratio', zero)),
            'loss_gaussian_target_importance': as_tensor(gaussian_details.get('gaussian_target_importance', zero)),
            'loss_gaussian_target_opacity': as_tensor(gaussian_details.get('gaussian_target_opacity', zero)),
            'loss_gaussian_pred_keep_prob': as_tensor(gaussian_details.get('gaussian_pred_keep_prob', zero)),
            'loss_gaussian_target_split': as_tensor(gaussian_details.get('gaussian_target_split', zero)),
            'loss_gaussian_pred_split': as_tensor(gaussian_details.get('gaussian_pred_split', zero)),
            'loss_gaussian_active_count': as_tensor(gaussian_details.get('gaussian_active_count', zero)),
            'loss_gaussian_active_ratio': as_tensor(gaussian_details.get('gaussian_active_ratio', zero)),
            'loss_gaussian_mean_opacity': as_tensor(gaussian_details.get('gaussian_mean_opacity', zero)),
            'loss_gaussian_mean_scale': as_tensor(gaussian_details.get('gaussian_mean_scale', zero)),
            'loss_gaussian_target_alpha_mean': as_tensor(gaussian_details.get('gaussian_target_alpha_mean', zero)),
            'loss_gaussian_render_alpha_mean': as_tensor(gaussian_details.get('gaussian_render_alpha_mean', zero)),
            'loss_gaussian_target_visible_mean': as_tensor(gaussian_details.get('gaussian_target_visible_mean', zero)),
            'loss_gaussian_render_visible_mean': as_tensor(gaussian_details.get('gaussian_render_visible_mean', zero)),
        }
        
        return loss_dict
