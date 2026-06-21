"""
Loss Functions for 3D Vision Tasks

This module provides loss functions for joint camera pose estimation and 3D point prediction.
It includes scale-invariant point losses, normal consistency losses, and relative pose losses.
"""

import math
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from panovggt.utils.geometry import homogenize_points, se3_inverse, depth_edge
from panovggt.utils.alignment import align_points_scale
from panovggt.render.gs_branch import GSBranch
from panovggt.render.losses import GaussianRenderLoss


logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def invert_homogeneous_matrix(T: torch.Tensor) -> torch.Tensor:
    """Invert batched 4x4 homogeneous transforms without assuming SO(3)."""
    return torch.linalg.inv(T.float()).to(dtype=T.dtype)


def transform_points_homogeneous(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply batched column-convention 4x4 transforms to [..., 3] points."""
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1).float()
    out_h = torch.einsum("bij,bnhwj->bnhwi", T.float(), points_h)
    w = out_h[..., 3:4]
    safe_w = torch.where(w.abs() > 1e-8, w, torch.ones_like(w))
    return (out_h[..., :3] / safe_w).to(dtype=points.dtype)


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
        
        if not valid_masks.any():
            zero = pred_local_pts.new_tensor(0.0)
            scale = torch.ones(B, device=pred_local_pts.device, dtype=pred_local_pts.dtype)
            return zero, {"local_pts_loss": zero, "normal_loss": zero}, scale
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
# Combined Loss
# =============================================================================

class Loss(nn.Module):
    """
    Combined Loss for Joint Point and Camera Estimation.

    This module combines:
    1. Point loss (local/global points + normal consistency + optional confidence).
    2. Camera pose loss (relative rotation + translation).
    3. Optional Gaussian render loss when `gs.enabled` is true.

    Args:
        train_conf (bool): Whether to train confidence prediction. Default: False.
        gs (dict|None): Gaussian-branch config. When provided and enabled the
            module renders the predicted Gaussians and adds a photometric loss.
    """

    def __init__(self, train_conf: bool = False, gs: Optional[Dict] = None):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
        self.gs_conf = gs or {}
        self.gs_enabled = bool(self.gs_conf.get("enabled", False))
        self.gs_branch = None
        self.gs_loss = None
        if self.gs_enabled:
            self.gs_loss = GaussianRenderLoss(
                rgb_weight=float(self.gs_conf.get("rgb_weight", 1.0)),
                ssim_weight=float(self.gs_conf.get("ssim_weight", 0.2)),
                depth_weight=float(self.gs_conf.get("depth_weight", 0.0)),
                rgb_loss_type=str(self.gs_conf.get("rgb_loss_type", "l1")),
                charbonnier_eps=float(self.gs_conf.get("charbonnier_eps", 1e-3)),
                solid_angle_weight=bool(self.gs_conf.get("solid_angle_weight", False)),
            )
        self._gs_loss_weight = float(self.gs_conf.get("loss_weight", 1.0))
        self._point_loss_weight = float(self.gs_conf.get("point_loss_weight", 1.0))
        self._camera_loss_weight = float(self.gs_conf.get("camera_loss_weight", 0.1))
        self._gs_photometric_mode = str(self.gs_conf.get("photometric_mode", "source_recon"))
        self._gs_target_policy = str(self.gs_conf.get("target_policy", "last"))
        self._gs_num_target_views = int(self.gs_conf.get("num_target_views", 1))
        self._gs_self_recon = bool(self.gs_conf.get("self_recon", False))
        self._gs_mask_rgb_by_valid = bool(self.gs_conf.get("mask_rgb_by_valid", False))
        self._gs_scale_reg_weight = float(self.gs_conf.get("scale_reg_weight", 0.0))
        self._gs_scale_reg_target = float(self.gs_conf.get("scale_reg_target", 1.0))
        self._gs_coverage_weight = float(self.gs_conf.get("coverage_weight", 0.0))
        self._gs_coverage_target_alpha = float(self.gs_conf.get("coverage_target_alpha", 1.0))
        self._gs_front_floater_weight = float(self.gs_conf.get("front_floater_weight", 0.0))
        self._gs_front_floater_margin = float(self.gs_conf.get("front_floater_margin", 0.03))
        self._gs_front_floater_depth_source = str(
            self.gs_conf.get("front_floater_depth_source", "pred")
        ).lower()
        self._gs_offset_reg_weight = float(self.gs_conf.get("offset_reg_weight", 0.0))
    
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
        
        point_masks = gt['point_masks']
        rgb_masks = gt.get('rgb_masks', gt.get('image_masks', None))
        if rgb_masks is None:
            rgb_masks = torch.ones_like(point_masks, dtype=torch.bool)
        depth_masks = gt.get('depth_masks', gt.get('valid_depth_masks', point_masks))
        source_gs_masks = gt.get('source_gs_masks', gt.get('gs_masks', point_masks))

        return {
            'imgs': gt['images'],
            'global_points': gt['world_points'],
            'local_points': gt['cam_points'],
            'valid_masks': point_masks,
            'rgb_masks': rgb_masks,
            'depth_masks': depth_masks,
            'source_gs_masks': source_gs_masks,
            'camera_poses': poses_c2w,
            'depths': gt.get('depths', None),
            'norm_factors': gt.get('norm_factors', None),
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
        source_idx = pred.get("gs_source_indices", None)
        if source_idx is not None:
            source_idx = source_idx.to(device=local_points.device, dtype=torch.long)
            masks = gt['valid_masks'].index_select(1, source_idx)
        else:
            masks = gt['valid_masks']
        
        B, N, H, W, _ = local_points.shape
        pred['camera_poses_original'] = camera_poses
        if 'world_points' in pred and pred['world_points'] is not None:
            pred['world_points_original'] = pred['world_points']
        
        # Compute normalization scale from local points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        valid_count = masks.float().sum(dim=[-1, -2, -3])
        raw_norm_factor = all_dis.sum(dim=(1, 2)) / valid_count.clamp(min=1e-8)
        norm_factor = torch.where(
            valid_count > 0,
            raw_norm_factor,
            torch.ones_like(raw_norm_factor),
        )
        norm_factor = torch.nan_to_num(
            norm_factor, nan=1.0, posinf=1e6, neginf=1.0
        ).clamp(min=1e-6, max=1e6)

        scale = norm_factor.view(B, 1, 1, 1, 1)

        # Normalize local points
        pred['local_points'] = local_points / scale
        if 'depth' in pred and pred['depth'] is not None:
            pred['depth_original'] = pred['depth']
            pred['depth'] = pred['depth'] / scale

        # Transform world-like predictions to the same normalized cam0 frame.
        # Use the full homogeneous inverse here instead of R^T/t. The camera
        # head is projected to SE(3), but tiny numerical drift can otherwise
        # make cam0_w2c @ c2w0 differ enough to trip the GS coordinate check.
        c2w0 = camera_poses[:, :1]
        cam0_w2c = invert_homogeneous_matrix(c2w0)
        cam0_w2c_single = cam0_w2c[:, 0]

        def normalize_world_like(points: torch.Tensor) -> torch.Tensor:
            points_cam0 = transform_points_homogeneous(points, cam0_w2c_single)
            return points_cam0 / scale

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] = normalize_world_like(pred['global_points'])
        if 'world_points' in pred and pred['world_points'] is not None:
            pred['world_points'] = normalize_world_like(pred['world_points'])
            pred['gs_world_points'] = pred['world_points']
        
        # Normalize camera translations
        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)
        pred['camera_poses'] = camera_poses_normalized
        pred['norm_factor'] = norm_factor

        # Camera poses for GS rendering live in normalized cam0 coordinates.
        # The first view is identity, and other views are target-camera poses
        # relative to cam0 with translations divided by the same norm_factor.
        gs_camera_poses = torch.matmul(
            cam0_w2c.float(), camera_poses.float()
        ).to(camera_poses.dtype)
        gs_camera_poses[..., :3, 3] /= norm_factor.view(B, 1, 1)
        eye = torch.eye(4, device=gs_camera_poses.device, dtype=gs_camera_poses.dtype)
        gs_camera_poses = torch.cat(
            [eye.view(1, 1, 4, 4).expand(B, 1, 4, 4), gs_camera_poses[:, 1:]],
            dim=1,
        )
        pred['gs_camera_poses'] = gs_camera_poses
        
        return pred
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_raw: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Prepare ground truth
        gt = self.prepare_gt(gt_raw)

        # Normalize predictions
        pred = self.normalize_pred(pred, gt)

        # Compute point and camera losses only when they can affect the objective.
        # In GS-only training, evaluating these branches can turn bad-depth samples
        # into 0*NaN and poison the render loss.
        zero = pred["local_points"].new_tensor(0.0)
        B = pred["local_points"].shape[0]
        scale = torch.ones(B, device=zero.device, dtype=zero.dtype)
        point_details: Dict[str, torch.Tensor] = {}
        cam_details: Dict[str, torch.Tensor] = {}
        point_loss = zero
        cam_loss = zero

        need_point_loss = self._point_loss_weight != 0.0 or self._camera_loss_weight != 0.0
        if need_point_loss:
            point_loss, point_details, scale = self.point_loss(pred, gt)
        if self._camera_loss_weight != 0.0:
            cam_loss, cam_details = self.camera_loss(pred, gt, scale)

        # Total objective loss
        loss_objective = zero
        if self._point_loss_weight != 0.0:
            loss_objective = loss_objective + self._point_loss_weight * point_loss
        if self._camera_loss_weight != 0.0:
            loss_objective = loss_objective + self._camera_loss_weight * cam_loss

        # ---- Gaussian render loss ----------------------------------------
        gs_total = loss_objective.new_tensor(0.0)
        gs_details: Dict[str, torch.Tensor] = {}
        if self.gs_enabled and "gaussian" in pred and pred["gaussian"] is not None:
            gs_total, gs_details = self._compute_gs_loss(pred, gt)
            loss_objective = loss_objective + self._gs_loss_weight * gs_total

        # Helper to convert to tensor
        def as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(
                x, device=loss_objective.device, dtype=loss_objective.dtype
            )

        # Construct unified loss dictionary
        loss_dict = {
            'loss_objective': loss_objective,
            'loss_camera': as_tensor(cam_loss),
            'loss_T': as_tensor(cam_details.get('trans_loss', zero)),
            'loss_R': as_tensor(cam_details.get('rot_loss', zero)),
            'loss_conf_depth': zero,
            'loss_reg_depth': zero,
            'loss_grad_depth': zero,
            'loss_conf_point': as_tensor(point_details.get('local_conf_loss', zero)),
            'loss_reg_point': (
                as_tensor(point_details.get('local_pts_loss', zero)) +
                as_tensor(point_details.get('global_pts_loss', zero))
            ),
            'loss_grad_point': as_tensor(point_details.get('normal_loss', zero)),
            'loss_local_point': as_tensor(point_details.get('local_pts_loss', zero)),
            'loss_global_point': as_tensor(point_details.get('global_pts_loss', zero)),
            # GS-branch loss components (zero when disabled)
            'loss_gs': as_tensor(gs_total),
            'loss_gs_rgb': as_tensor(gs_details.get('rgb_loss', gs_details.get('rgb_l1', zero))),
            'loss_gs_ssim': as_tensor(gs_details.get('ssim', zero)),
            'loss_gs_depth': as_tensor(gs_details.get('depth_l1', zero)),
            'loss_gs_scale_reg': as_tensor(gs_details.get('scale_reg', zero)),
            'loss_gs_coverage': as_tensor(gs_details.get('coverage', zero)),
            'loss_gs_front_floater': as_tensor(gs_details.get('front_floater', zero)),
            'loss_gs_offset_reg': as_tensor(gs_details.get('offset_reg', zero)),
        }

        return loss_dict

    # ------------------------------------------------------------------
    def _build_gs_branch(self, equ_h: int, sh_degree: int):
        """Lazily build the GSBranch once we know image dimensions."""
        gs_conf = self.gs_conf
        self.gs_branch = GSBranch(
            renderer=str(gs_conf.get("renderer", "cube")),
            equ_h=int(equ_h),
            face_res=int(gs_conf.get("face_res", 256)),
            fov_deg=float(gs_conf.get("fov_deg", 95.0)),
            boundary_px=int(gs_conf.get("boundary_px", 4)),
            sh_degree=int(sh_degree),
            scale_init_mode=str(gs_conf.get("scale_init_mode", "depth_footprint")),
            scale_init_value=float(gs_conf.get("scale_init_value", 0.01)),
            scale_init_factor=float(gs_conf.get("scale_init_factor", 1.0)),
            scale_mult_min=gs_conf.get("scale_mult_min", None),
            scale_mult_max=gs_conf.get("scale_mult_max", None),
            use_offset=bool(gs_conf.get("use_offset", False)),
            detach_centers=bool(gs_conf.get("detach_centers", True)),
            detach_camera=bool(gs_conf.get("detach_camera", True)),
            train_dc=bool(gs_conf.get("train_dc", True)),
            train_opacity=bool(gs_conf.get("train_opacity", True)),
            train_scale=bool(gs_conf.get("train_scale", False)),
            train_rotation=bool(gs_conf.get("train_rotation", False)),
            train_sh_rest=bool(gs_conf.get("train_sh_rest", False)),
            min_valid_ratio=float(gs_conf.get("min_valid_ratio", 0.25)),
        )

    def _compute_gs_loss(self, pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor]):
        """Render the predicted Gaussians and compute photometric loss."""
        if "gs_world_points" not in pred or "gs_camera_poses" not in pred:
            raise KeyError("GS loss requires normalized `gs_world_points` and `gs_camera_poses`.")
        world_points = pred["gs_world_points"]         # (B,S,H,W,3), normalized cam0 frame
        camera_poses = pred["gs_camera_poses"]         # (B,S,4,4), c2w in normalized cam0 frame
        depth_pred = pred.get("depth", None)           # (B,S,H,W,1)
        gs_params = pred["gaussian"]
        images = pred.get("images", gt.get("imgs"))    # (B,S,3,H,W)
        B, S, H, W, _ = world_points.shape
        if world_points.shape[:4] != camera_poses.shape[:2] + (H, W):
            raise ValueError(
                "GS coordinate contract violated: world_points must be [B,S,H,W,3] "
                f"and camera_poses [B,S,4,4], got {world_points.shape} / {camera_poses.shape}"
            )
        eye = torch.eye(4, device=camera_poses.device, dtype=camera_poses.dtype)
        first_pose_err = (camera_poses[:, 0].float() - eye.float()).abs().max()
        if first_pose_err > 5e-3:
            raise ValueError(
                "GS camera poses must be first-frame-relative; "
                f"pose[:,0] is not identity (max error {first_pose_err.item():.3e})."
            )

        mode = self._gs_photometric_mode.lower()
        split_before_forward = (
            pred.get("gs_source_indices", None) is not None
            and pred.get("gs_target_indices", None) is not None
        )
        source_full_idx = None
        if split_before_forward:
            if mode not in ("novel_view", "nvs", "source_target"):
                raise ValueError(
                    "trainer-side GS view split is only valid for novel_view/source_target modes."
                )
            source_full_idx = pred["gs_source_indices"].to(
                device=world_points.device, dtype=torch.long
            )
            target_idx = pred["gs_target_indices"].to(
                device=world_points.device, dtype=torch.long
            )
            if source_full_idx.numel() != S:
                raise ValueError(
                    "trainer-side GS source indices must match prediction view count, "
                    f"got {source_full_idx.numel()} indices for S={S}."
                )
            source_idx = torch.arange(S, device=world_points.device)
        elif mode in ("source_recon", "reconstruction", "observed"):
            source_idx = target_idx = torch.arange(S, device=world_points.device)
        elif mode in ("novel_view", "nvs", "source_target"):
            if S < 2:
                raise ValueError("GS novel_view loss requires at least 2 views in each sample.")
            if self._gs_num_target_views < 1 or self._gs_num_target_views >= S:
                raise ValueError(
                    "num_target_views must be in [1, S-1] for novel_view GS loss, "
                    f"got {self._gs_num_target_views} with S={S}."
                )
            target_idx = self._select_gs_target_indices(S, world_points.device)
            source_mask = torch.ones(S, dtype=torch.bool, device=world_points.device)
            source_mask[target_idx] = False
            source_idx = torch.arange(S, device=world_points.device)[source_mask]
        else:
            raise ValueError(
                "Unknown GS photometric_mode "
                f"'{self._gs_photometric_mode}'. Use 'source_recon' or 'novel_view'."
            )

        def select_views(x: Optional[torch.Tensor], idx: torch.Tensor):
            if x is None:
                return None
            return x.index_select(1, idx)

        def select_gs_params(params: Dict[str, torch.Tensor], idx: torch.Tensor):
            selected = {}
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() < 2 or value.shape[1] != S:
                        raise ValueError(
                            f"Gaussian param '{key}' must have source-view dim S={S}; "
                            f"got {tuple(value.shape)}."
                        )
                    selected[key] = value.index_select(1, idx)
                else:
                    selected[key] = value
            return selected

        def flatten_views(x: Optional[torch.Tensor]):
            if x is None:
                return None
            return x.reshape(B * S, 1, *x.shape[2:])

        def flatten_gs_params(params: Dict[str, torch.Tensor]):
            flattened = {}
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() < 2 or value.shape[1] != S:
                        raise ValueError(
                            f"Gaussian param '{key}' must have source-view dim S={S}; "
                            f"got {tuple(value.shape)}."
                        )
                    flattened[key] = value.reshape(B * S, 1, *value.shape[2:])
                else:
                    flattened[key] = value
            return flattened

        def canonical_depth(depth: Optional[torch.Tensor], num_views: int):
            if depth is None:
                return None
            if depth.dim() == 5 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            elif depth.dim() == 5 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            elif depth.dim() != 4:
                raise ValueError(
                    f"unexpected tensor shape for GS depth target: {tuple(depth.shape)}"
                )
            return depth.reshape(B * num_views, 1, H, W).to(
                dtype=world_points.dtype, device=world_points.device
            )

        def canonical_mask(mask_tensor: Optional[torch.Tensor], batch_views: int):
            if mask_tensor is None:
                return None
            if mask_tensor.dim() == 5 and mask_tensor.shape[2] == 1:
                mask_tensor = mask_tensor[:, :, 0]
            elif mask_tensor.dim() == 5 and mask_tensor.shape[-1] == 1:
                mask_tensor = mask_tensor[..., 0]
            elif mask_tensor.dim() != 4:
                raise ValueError(
                    f"unexpected tensor shape for GS valid mask: {tuple(mask_tensor.shape)}"
                )
            return mask_tensor.reshape(batch_views, 1, H, W).to(
                dtype=world_points.dtype, device=world_points.device
            )

        def apply_depth_normalization(depth_gt: Optional[torch.Tensor], num_views: int):
            if depth_gt is None:
                return None
            if gt_norm_factor is not None and pred_norm_factor is not None:
                depth_ratio = (
                    gt_norm_factor.to(depth_gt.device, dtype=depth_gt.dtype)
                    / pred_norm_factor.detach().to(depth_gt.device, dtype=depth_gt.dtype)
                )
                depth_ratio = depth_ratio.view(B, 1, 1, 1, 1).expand(B, num_views, 1, 1, 1)
                depth_gt = depth_gt * depth_ratio.reshape(B * num_views, 1, 1, 1)
            return depth_gt

        gt_norm_factor = gt.get("norm_factors", None)
        pred_norm_factor = pred.get("norm_factor", None)

        def gt_target_camera_poses(target_view_idx: torch.Tensor):
            pose_source = str(
                self.gs_conf.get("bootstrap_geometry_source", "gt")
            ).lower()
            if pose_source not in ("gt", "ground_truth"):
                raise ValueError(
                    "trainer-side GS novel-view split requires "
                    "loss.gs.bootstrap_geometry_source: gt."
                )
            if source_full_idx is None or source_full_idx.numel() == 0:
                raise ValueError("missing GS source indices for target camera pose bootstrap.")
            gt_camera_poses = gt["camera_poses"].to(
                device=world_points.device, dtype=camera_poses.dtype
            )
            anchor_pose = gt_camera_poses.index_select(1, source_full_idx[:1])
            anchor_w2c = invert_homogeneous_matrix(anchor_pose)
            target_c2w = select_views(gt_camera_poses, target_view_idx)
            rel = torch.matmul(anchor_w2c.float(), target_c2w.float()).to(camera_poses.dtype)
            if pred_norm_factor is not None:
                rel[..., :3, 3] /= pred_norm_factor.detach().view(B, 1, 1).to(
                    device=rel.device, dtype=rel.dtype
                )
            return rel

        sh_dim = gs_params["sh_rest"].shape[-1]        # 3*sh_extra/3 = sh_extra
        # Recover sh_degree from rest dim (sh_extra = (deg+1)^2 - 1)
        sh_extra = sh_dim
        from math import isqrt
        deg = isqrt(sh_extra + 1) - 1 if sh_extra > 0 else 0

        if self.gs_branch is None or self.gs_branch.renderer.equ_h != H:
            self._build_gs_branch(equ_h=H, sh_degree=deg)
            self.gs_branch.renderer.to(world_points.device)
        else:
            self.gs_branch.renderer.to(world_points.device)

        valid_masks = gt.get("valid_masks", None)
        depth_masks = gt.get("depth_masks", valid_masks)
        source_gs_masks = gt.get("source_gs_masks", valid_masks)
        rgb_masks = gt.get("rgb_masks", None)
        if split_before_forward and source_full_idx is not None:
            source_gs_masks_for_pred = select_views(source_gs_masks, source_full_idx)
        else:
            source_gs_masks_for_pred = source_gs_masks

        if mode in ("source_recon", "reconstruction", "observed") and self._gs_self_recon:
            # Render each source view only from its own Gaussian set. This avoids
            # letting a union of all observed views explain every training target
            # during the fragile bootstrap phase.
            render_B = B * S
            render_T = 1
            T = S
            flat_world_points = flatten_views(world_points)
            flat_depth_pred = flatten_views(depth_pred)
            flat_images = flatten_views(images)
            flat_masks = flatten_views(source_gs_masks_for_pred)
            flat_camera_poses = flatten_views(camera_poses)
            flat_gs_params = flatten_gs_params(gs_params)

            out = self.gs_branch.render(
                gs_params=flat_gs_params,
                world_points=flat_world_points,
                camera_poses_c2w=flat_camera_poses,
                images=flat_images,
                depth=flat_depth_pred,
                point_masks=flat_masks,
            )

            rgb_pred = out["rgb_erp"].reshape(render_B * render_T, 3, H, W)
            depth_render = out["depth_erp"].reshape(render_B * render_T, 1, H, W)
            render_mask = out["mask_erp"].reshape(render_B * render_T, 1, H, W)
            rgb_gt = images.reshape(render_B * render_T, 3, H, W).to(
                dtype=rgb_pred.dtype, device=rgb_pred.device
            )
            target_depths = canonical_depth(gt.get("depths", None), S)
            target_depths = apply_depth_normalization(target_depths, S)
            pred_target_depths = canonical_depth(
                depth_pred.detach() if depth_pred is not None else None, S
            )
            target_depth_masks = canonical_mask(depth_masks, render_B)
            target_rgb_masks = canonical_mask(rgb_masks, render_B)
        else:
            source_world_points = select_views(world_points, source_idx)
            source_depth_pred = select_views(depth_pred, source_idx)
            source_images = select_views(images, source_idx)
            source_masks = select_views(source_gs_masks_for_pred, source_idx)
            source_gs_params = select_gs_params(gs_params, source_idx)
            render_camera_poses = (
                gt_target_camera_poses(target_idx)
                if split_before_forward
                else select_views(camera_poses, target_idx)
            )
            target_image_bank = gt.get("imgs", images) if split_before_forward else images
            target_images = select_views(target_image_bank, target_idx)
            target_depths_raw = select_views(gt.get("depths", None), target_idx)
            target_pred_depths_raw = (
                None
                if split_before_forward or depth_pred is None
                else select_views(depth_pred.detach(), target_idx)
            )
            target_depth_masks_raw = select_views(depth_masks, target_idx)
            target_rgb_masks_raw = select_views(rgb_masks, target_idx)
            T = int(target_idx.numel())

            out = self.gs_branch.render(
                gs_params=source_gs_params,
                world_points=source_world_points,
                camera_poses_c2w=render_camera_poses,
                images=source_images,
                depth=source_depth_pred,
                point_masks=source_masks,
            )

            rgb_pred = out["rgb_erp"].reshape(B * T, 3, H, W)
            depth_render = out["depth_erp"].reshape(B * T, 1, H, W)
            render_mask = out["mask_erp"].reshape(B * T, 1, H, W)
            rgb_gt = target_images.reshape(B * T, 3, H, W).to(
                dtype=rgb_pred.dtype, device=rgb_pred.device
            )
            target_depths = canonical_depth(target_depths_raw, T)
            target_depths = apply_depth_normalization(target_depths, T)
            pred_target_depths = canonical_depth(target_pred_depths_raw, T)
            target_depth_masks = canonical_mask(target_depth_masks_raw, B * T)
            target_rgb_masks = canonical_mask(target_rgb_masks_raw, B * T)

        alpha_render = out["alpha_erp"].reshape(B * T, 1, H, W).clamp(0.0, 1.0)
        if target_depth_masks is not None:
            depth_render_mask = render_mask * target_depth_masks
        else:
            depth_render_mask = render_mask
        if target_rgb_masks is not None:
            rgb_render_mask = render_mask * target_rgb_masks
        else:
            rgb_render_mask = render_mask
        rgb_mask = (
            rgb_render_mask * target_depth_masks
            if self._gs_mask_rgb_by_valid and target_depth_masks is not None
            else rgb_render_mask
        )

        gs_total, gs_details = self.gs_loss(
            rgb_pred=rgb_pred,
            rgb_gt=rgb_gt,
            mask=rgb_mask,
            depth_pred=depth_render,
            depth_gt=target_depths,
            depth_mask=depth_render_mask if target_depths is not None else None,
        )
        if self._gs_coverage_weight > 0.0:
            coverage_target = alpha_render.new_tensor(self._gs_coverage_target_alpha)
            coverage_err = F.relu(coverage_target - alpha_render).square()
            coverage = (coverage_err * rgb_mask).sum() / (rgb_mask.sum() + 1e-6)
            gs_details["coverage"] = coverage
            gs_total = gs_total + self._gs_coverage_weight * coverage
            gs_details["total"] = gs_total

        if self._gs_front_floater_weight > 0.0:
            if self._gs_front_floater_depth_source == "gt" and target_depths is not None:
                surface_depth = target_depths.detach()
            else:
                surface_depth = pred_target_depths.detach() if pred_target_depths is not None else None
            if surface_depth is not None:
                margin = depth_render.new_tensor(self._gs_front_floater_margin)
                front_err = F.relu(surface_depth - depth_render - margin).square()
                front_weight = alpha_render * depth_render_mask
                front_floater = (front_err * front_weight).sum() / (front_weight.sum() + 1e-6)
                gs_details["front_floater"] = front_floater
                gs_total = gs_total + self._gs_front_floater_weight * front_floater
                gs_details["total"] = gs_total

        if self._gs_offset_reg_weight > 0.0:
            offset = out.get("offset", None)
            scale_init = out.get("scale_init", None)
            if offset is not None and scale_init is not None:
                scale_ref = scale_init.detach().norm(dim=-1).clamp(min=1e-6)
                offset_reg = (offset.norm(dim=-1) / scale_ref).square().mean()
                gs_details["offset_reg"] = offset_reg
                gs_total = gs_total + self._gs_offset_reg_weight * offset_reg
                gs_details["total"] = gs_total

        if self._gs_scale_reg_weight > 0.0:
            scale_mult = out.get("scale_mult", None)
            if scale_mult is not None:
                scale_reg = F.relu(scale_mult - self._gs_scale_reg_target).square().mean()
                gs_details["scale_reg"] = scale_reg
                gs_total = gs_total + self._gs_scale_reg_weight * scale_reg
                gs_details["total"] = gs_total
        return gs_total, gs_details

    def _select_gs_target_indices(self, num_views: int, device: torch.device) -> torch.Tensor:
        count = self._gs_num_target_views
        policy = self._gs_target_policy.lower()
        if policy == "first":
            target_idx = torch.arange(count, device=device)
        elif policy == "last":
            target_idx = torch.arange(num_views - count, num_views, device=device)
        elif policy == "random":
            if self.training:
                target_idx = torch.randperm(num_views, device=device)[:count]
                target_idx = target_idx.sort().values
            else:
                target_idx = torch.arange(num_views - count, num_views, device=device)
        else:
            raise ValueError(
                f"Unknown GS target_policy '{self._gs_target_policy}'. "
                "Use 'first', 'last', or 'random'."
            )
        return target_idx
