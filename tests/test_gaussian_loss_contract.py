import torch
import torch.nn as nn

from panovggt.models.loss import Loss
from panovggt.render.losses import GaussianRenderLoss


def test_gaussian_render_loss_normalizes_mask_by_visible_pixels_and_channels():
    rgb_pred = torch.zeros(1, 3, 2, 2)
    rgb_gt = torch.zeros_like(rgb_pred)
    rgb_gt[:, :, 0, 1] = 2.0
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, 0, 1] = 1.0

    loss_fn = GaussianRenderLoss(
        rgb_weight=1.0,
        ssim_weight=0.0,
        depth_weight=0.0,
        rgb_loss_type="mse",
    )
    total, details = loss_fn(rgb_pred, rgb_gt, mask=mask)

    torch.testing.assert_close(total, torch.tensor(4.0))
    torch.testing.assert_close(details["rgb_loss"], torch.tensor(4.0))
    torch.testing.assert_close(details["rgb_mse"], torch.tensor(4.0))


def test_combined_loss_initializes_gaussian_branch_loss_without_trainer():
    loss = Loss(
        train_conf=False,
        gs={
            "enabled": True,
            "rgb_weight": 1.0,
            "ssim_weight": 0.0,
            "depth_weight": 0.0,
            "rgb_loss_type": "l1",
            "point_loss_weight": 0.0,
            "camera_loss_weight": 0.0,
        },
    )

    assert loss.gs_enabled is True
    assert loss.gs_loss is not None
    assert loss._gs_photometric_mode == "source_recon"


def _identity_poses(batch=1, views=1):
    return torch.eye(4).view(1, 1, 4, 4).expand(batch, views, 4, 4).clone()


def _minimal_gaussian_params(batch=1, views=1, patch_h=1, patch_w=1):
    return {
        "offset": torch.zeros(batch, views, patch_h, patch_w, 3),
        "scale": torch.full((batch, views, patch_h, patch_w, 3), 0.01),
        "rotation": torch.zeros(batch, views, patch_h, patch_w, 4),
        "opacity": torch.ones(batch, views, patch_h, patch_w, 1),
        "sh_dc": torch.zeros(batch, views, patch_h, patch_w, 3),
        "sh_rest": torch.zeros(batch, views, patch_h, patch_w, 3, 3),
    }


class _FakeRenderer:
    equ_h = 2

    def to(self, device):
        return self


class _FakeBranch:
    def __init__(self):
        self.renderer = _FakeRenderer()
        self.point_masks = None

    def render(self, gs_params, world_points, camera_poses_c2w, images, depth=None, point_masks=None):
        batch, targets, _, width, _ = world_points.shape
        height = width
        self.point_masks = point_masks.detach().clone() if point_masks is not None else None
        return {
            "rgb_erp": torch.zeros(batch, targets, 3, height, width),
            "depth_erp": torch.zeros(batch, targets, 1, height, width),
            "alpha_erp": torch.ones(batch, targets, 1, height, width),
            "mask_erp": torch.ones(batch, targets, 1, height, width),
        }


class _CaptureRenderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_mask = None
        self.depth_mask = None

    def forward(self, rgb_pred, rgb_gt, mask=None, depth_pred=None, depth_gt=None, depth_mask=None):
        self.rgb_mask = mask.detach().clone() if mask is not None else None
        self.depth_mask = depth_mask.detach().clone() if depth_mask is not None else None
        zero = rgb_pred.sum() * 0.0
        return zero, {"rgb_loss": zero, "total": zero}


def test_gaussian_loss_uses_separate_rgb_depth_and_source_masks():
    loss = Loss(
        train_conf=False,
        gs={
            "enabled": True,
            "mask_rgb_by_valid": False,
            "rgb_weight": 1.0,
            "ssim_weight": 0.0,
            "depth_weight": 1.0,
            "point_loss_weight": 0.0,
            "camera_loss_weight": 0.0,
        },
    )
    fake_branch = _FakeBranch()
    capture_loss = _CaptureRenderLoss()
    loss.gs_branch = fake_branch
    loss.gs_loss = capture_loss

    rgb_masks = torch.tensor([[[[1, 0], [1, 1]]]], dtype=torch.bool)
    depth_masks = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.bool)
    source_gs_masks = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
    pred = {
        "gs_world_points": torch.zeros(1, 1, 2, 2, 3),
        "gs_camera_poses": _identity_poses(),
        "depth": torch.ones(1, 1, 2, 2, 1),
        "gaussian": _minimal_gaussian_params(),
        "images": torch.zeros(1, 1, 3, 2, 2),
    }
    gt = {
        "imgs": torch.zeros(1, 1, 3, 2, 2),
        "depths": torch.ones(1, 1, 2, 2),
        "valid_masks": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "rgb_masks": rgb_masks,
        "depth_masks": depth_masks,
        "source_gs_masks": source_gs_masks,
    }

    loss._compute_gs_loss(pred, gt)

    torch.testing.assert_close(fake_branch.point_masks.float(), source_gs_masks.float())
    torch.testing.assert_close(capture_loss.rgb_mask, rgb_masks.float().reshape(1, 1, 2, 2))
    torch.testing.assert_close(capture_loss.depth_mask, depth_masks.float().reshape(1, 1, 2, 2))
