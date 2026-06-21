import torch
import torch.nn as nn

from panovggt.models.loss import Loss
from panovggt.render.losses import (
    GaussianRenderLoss,
    erp_solid_angle_weights,
)


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


def test_gaussian_render_loss_supports_masked_charbonnier():
    rgb_pred = torch.zeros(1, 3, 2, 2)
    rgb_gt = torch.zeros_like(rgb_pred)
    rgb_gt[:, :, 0, 1] = 2.0
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, 0, 1] = 1.0

    loss_fn = GaussianRenderLoss(
        rgb_weight=1.0,
        ssim_weight=0.0,
        depth_weight=0.0,
        rgb_loss_type="charbonnier",
        charbonnier_eps=0.1,
    )
    total, details = loss_fn(rgb_pred, rgb_gt, mask=mask)

    expected = torch.sqrt(torch.tensor(4.0 + 0.01)) - 0.1
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(details["rgb_charbonnier"], expected)


def test_gaussian_render_loss_ignores_nan_outside_rgb_and_depth_masks():
    rgb_pred = torch.zeros(1, 3, 2, 2)
    rgb_gt = torch.zeros_like(rgb_pred)
    rgb_pred[:, :, 0, 1] = float("nan")
    depth_pred = torch.zeros(1, 1, 2, 2)
    depth_gt = torch.zeros_like(depth_pred)
    depth_gt[:, :, 1, 0] = float("nan")
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, 0, 0] = 1.0
    depth_mask = torch.zeros(1, 1, 2, 2)
    depth_mask[:, :, 1, 1] = 1.0

    loss_fn = GaussianRenderLoss(
        rgb_weight=1.0,
        ssim_weight=0.0,
        depth_weight=1.0,
        rgb_loss_type="mse",
    )
    total, details = loss_fn(
        rgb_pred,
        rgb_gt,
        mask=mask,
        depth_pred=depth_pred,
        depth_gt=depth_gt,
        depth_mask=depth_mask,
    )

    torch.testing.assert_close(details["rgb_loss"], torch.tensor(0.0))
    torch.testing.assert_close(details["depth_l1"], torch.tensor(0.0))
    torch.testing.assert_close(total, torch.tensor(0.0))


def test_gaussian_render_loss_ssim_requires_fully_valid_window():
    rgb_pred = torch.zeros(1, 3, 5, 5)
    rgb_gt = torch.zeros_like(rgb_pred)
    rgb_pred[:, :, 1:4, 1:4] = 1.0
    rgb_pred[:, :, 2, 2] = 0.0
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, 2, 2] = 1.0

    loss_fn = GaussianRenderLoss(
        rgb_weight=0.0,
        ssim_weight=1.0,
        depth_weight=0.0,
        ssim_window=3,
        rgb_loss_type="mse",
    )
    total, details = loss_fn(rgb_pred, rgb_gt, mask=mask)

    torch.testing.assert_close(details["ssim"], torch.tensor(0.0))
    torch.testing.assert_close(total, torch.tensor(0.0))


def test_gaussian_render_loss_ssim_ignores_nan_outside_valid_windows():
    rgb_pred = torch.zeros(1, 3, 5, 5)
    rgb_gt = torch.zeros_like(rgb_pred)
    rgb_pred[:, :, 0, 0] = float("nan")
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, 1:4, 1:4] = 1.0

    loss_fn = GaussianRenderLoss(
        rgb_weight=0.0,
        ssim_weight=1.0,
        depth_weight=0.0,
        ssim_window=3,
        rgb_loss_type="mse",
    )
    total, details = loss_fn(rgb_pred, rgb_gt, mask=mask)

    torch.testing.assert_close(details["ssim"], torch.tensor(0.0))
    torch.testing.assert_close(total, torch.tensor(0.0))


def test_erp_solid_angle_weights_are_symmetric_and_downweight_poles():
    weights = erp_solid_angle_weights(4, device=torch.device("cpu"), dtype=torch.float32)

    assert weights.shape == (1, 1, 4, 1)
    torch.testing.assert_close(weights[..., 0, :], weights[..., -1, :])
    torch.testing.assert_close(weights[..., 1, :], weights[..., 2, :])
    assert weights[..., 0, :].item() < weights[..., 1, :].item()


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
        self.camera_poses_c2w = None
        self.source_camera_poses_c2w = None
        self.world_points = None
        self.depth = None
        self.images = None
        self.extra_out = {}

    def render(
        self,
        gs_params,
        world_points,
        camera_poses_c2w,
        images,
        depth=None,
        point_masks=None,
        source_camera_poses_c2w=None,
    ):
        batch = world_points.shape[0]
        targets = camera_poses_c2w.shape[1]
        height, width = world_points.shape[2:4]
        self.point_masks = point_masks.detach().clone() if point_masks is not None else None
        self.camera_poses_c2w = camera_poses_c2w.detach().clone()
        self.source_camera_poses_c2w = (
            source_camera_poses_c2w.detach().clone()
            if source_camera_poses_c2w is not None
            else None
        )
        self.world_points = world_points.detach().clone()
        self.depth = depth.detach().clone() if depth is not None else None
        self.images = images.detach().clone()
        out = {
            "rgb_erp": torch.zeros(batch, targets, 3, height, width),
            "depth_erp": torch.zeros(batch, targets, 1, height, width),
            "alpha_erp": torch.ones(batch, targets, 1, height, width),
            "mask_erp": torch.ones(batch, targets, 1, height, width),
        }
        out.update(self.extra_out)
        return out


class _CaptureRenderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_mask = None
        self.depth_mask = None
        self.rgb_gt = None

    def forward(self, rgb_pred, rgb_gt, mask=None, depth_pred=None, depth_gt=None, depth_mask=None):
        self.rgb_mask = mask.detach().clone() if mask is not None else None
        self.depth_mask = depth_mask.detach().clone() if depth_mask is not None else None
        self.rgb_gt = rgb_gt.detach().clone()
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


def test_gaussian_loss_regularizers_use_patch_valid_mask():
    loss = Loss(
        train_conf=False,
        gs={
            "enabled": True,
            "rgb_weight": 0.0,
            "ssim_weight": 0.0,
            "depth_weight": 0.0,
            "offset_reg_weight": 1.0,
            "scale_reg_weight": 1.0,
            "scale_reg_target": 1.0,
            "point_loss_weight": 0.0,
            "camera_loss_weight": 0.0,
        },
    )
    fake_branch = _FakeBranch()
    fake_branch.extra_out = {
        "offset_ratio": torch.tensor([[[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]),
        "scale_mult": torch.tensor([[[[10.0, 10.0, 10.0], [1.0, 1.0, 1.0]]]]),
        "patch_valid": torch.tensor([[[[0.0], [1.0]]]]),
    }
    loss.gs_branch = fake_branch
    loss.gs_loss = _CaptureRenderLoss()
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
    }

    _, details = loss._compute_gs_loss(pred, gt)

    torch.testing.assert_close(details["offset_reg"], torch.tensor(0.0))
    torch.testing.assert_close(details["scale_reg"], torch.tensor(0.0))


def test_gaussian_front_floater_ignores_nan_outside_depth_mask():
    loss = Loss(
        train_conf=False,
        gs={
            "enabled": True,
            "rgb_weight": 0.0,
            "ssim_weight": 0.0,
            "depth_weight": 0.0,
            "front_floater_weight": 1.0,
            "front_floater_depth_source": "gt",
            "front_floater_margin": 0.0,
            "point_loss_weight": 0.0,
            "camera_loss_weight": 0.0,
        },
    )
    fake_branch = _FakeBranch()
    fake_branch.extra_out = {
        "depth_erp": torch.tensor([[[[[0.0, float("nan")], [0.0, 0.0]]]]]),
        "alpha_erp": torch.ones(1, 1, 1, 2, 2),
        "mask_erp": torch.ones(1, 1, 1, 2, 2),
    }
    loss.gs_branch = fake_branch
    loss.gs_loss = _CaptureRenderLoss()
    pred = {
        "gs_world_points": torch.zeros(1, 1, 2, 2, 3),
        "gs_camera_poses": _identity_poses(),
        "depth": torch.ones(1, 1, 2, 2, 1),
        "gaussian": _minimal_gaussian_params(),
        "images": torch.zeros(1, 1, 3, 2, 2),
    }
    depth_masks = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    depth_masks[:, :, 0, 1] = False
    gt = {
        "imgs": torch.zeros(1, 1, 3, 2, 2),
        "depths": torch.zeros(1, 1, 2, 2),
        "valid_masks": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "depth_masks": depth_masks,
    }

    _, details = loss._compute_gs_loss(pred, gt)

    torch.testing.assert_close(details["front_floater"], torch.tensor(0.0))


def test_trainer_side_novel_view_split_uses_full_gt_target_only_as_target():
    loss = Loss(
        train_conf=False,
        gs={
            "enabled": True,
            "photometric_mode": "novel_view",
            "view_split_location": "trainer",
            "bootstrap_geometry_source": "gt",
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

    source_masks = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    source_masks[:, 2] = False
    gt_poses = _identity_poses(views=3)
    gt_poses[:, 2, 0, 3] = 2.0
    gt_images = torch.zeros(1, 3, 3, 2, 2)
    gt_images[:, 2] = 0.7
    pred = {
        "gs_world_points": torch.zeros(1, 2, 2, 2, 3),
        "gs_camera_poses": _identity_poses(views=2),
        "depth": torch.ones(1, 2, 2, 2, 1),
        "gaussian": _minimal_gaussian_params(views=2),
        "images": torch.full((1, 2, 3, 2, 2), 0.1),
        "gs_source_indices": torch.tensor([0, 1]),
        "gs_target_indices": torch.tensor([2]),
        "norm_factor": torch.ones(1),
    }
    gt = {
        "imgs": gt_images,
        "depths": torch.ones(1, 3, 2, 2),
        "global_points": torch.zeros(1, 3, 2, 2, 3),
        "valid_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "rgb_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "depth_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "source_gs_masks": source_masks,
        "camera_poses": gt_poses,
    }

    loss._compute_gs_loss(pred, gt)

    assert fake_branch.point_masks.shape[1] == 2
    torch.testing.assert_close(fake_branch.point_masks.float(), source_masks[:, :2].float())
    torch.testing.assert_close(capture_loss.rgb_gt, torch.full((1, 3, 2, 2), 0.7))
    torch.testing.assert_close(
        fake_branch.camera_poses_c2w[:, :, 0, 3],
        torch.tensor([[2.0]]),
    )


def test_trainer_side_novel_view_gt_target_pose_matches_pred_scale():
    loss = Loss(
        train_conf=False,
        gs={
            "enabled": True,
            "photometric_mode": "novel_view",
            "view_split_location": "trainer",
            "bootstrap_geometry_source": "gt",
            "mask_rgb_by_valid": False,
            "rgb_weight": 1.0,
            "ssim_weight": 0.0,
            "depth_weight": 0.0,
            "point_loss_weight": 0.0,
            "camera_loss_weight": 0.0,
        },
    )
    fake_branch = _FakeBranch()
    loss.gs_branch = fake_branch
    loss.gs_loss = _CaptureRenderLoss()

    gt_poses = _identity_poses(views=3)
    gt_poses[:, 2, 0, 3] = 2.0
    pred = {
        "gs_world_points": torch.zeros(1, 2, 2, 2, 3),
        "gs_camera_poses": _identity_poses(views=2),
        "depth": torch.ones(1, 2, 2, 2, 1),
        "gaussian": _minimal_gaussian_params(views=2),
        "images": torch.zeros(1, 2, 3, 2, 2),
        "gs_source_indices": torch.tensor([0, 1]),
        "gs_target_indices": torch.tensor([2]),
        "norm_factor": torch.tensor([4.0]),
    }
    gt = {
        "imgs": torch.zeros(1, 3, 3, 2, 2),
        "depths": torch.ones(1, 3, 2, 2),
        "global_points": torch.zeros(1, 3, 2, 2, 3),
        "valid_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "rgb_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "depth_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "source_gs_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "camera_poses": gt_poses,
        "norm_factors": torch.tensor([6.0]),
    }

    loss._compute_gs_loss(pred, gt)

    torch.testing.assert_close(
        fake_branch.camera_poses_c2w[:, :, 0, 3],
        torch.tensor([[3.0]]),
    )


def test_trainer_side_novel_view_gt_bootstrap_uses_gt_source_geometry():
    loss = Loss(
        train_conf=False,
        gs={
            "enabled": True,
            "photometric_mode": "novel_view",
            "view_split_location": "trainer",
            "bootstrap_geometry_source": "gt",
            "mask_rgb_by_valid": False,
            "rgb_weight": 1.0,
            "ssim_weight": 0.0,
            "depth_weight": 0.0,
            "point_loss_weight": 0.0,
            "camera_loss_weight": 0.0,
        },
    )
    fake_branch = _FakeBranch()
    loss.gs_branch = fake_branch
    loss.gs_loss = _CaptureRenderLoss()

    gt_poses = _identity_poses(views=3)
    gt_poses[:, 1, 0, 3] = 10.0
    gt_poses[:, 2, 0, 3] = 13.0
    gt_points = torch.zeros(1, 3, 2, 2, 3)
    gt_points[:, 1, ..., 0] = 12.0
    gt_points[:, 2, ..., 0] = 14.0
    gt_depths = torch.ones(1, 3, 2, 2)
    gt_depths[:, 1] = 2.0
    gt_depths[:, 2] = 3.0
    pred = {
        "gs_world_points": torch.full((1, 2, 2, 2, 3), 100.0),
        "gs_camera_poses": _identity_poses(views=2),
        "depth": torch.full((1, 2, 2, 2, 1), 100.0),
        "gaussian": _minimal_gaussian_params(views=2),
        "images": torch.zeros(1, 2, 3, 2, 2),
        "gs_source_indices": torch.tensor([1, 2]),
        "gs_target_indices": torch.tensor([0]),
        "norm_factor": torch.ones(1),
    }
    gt = {
        "imgs": torch.zeros(1, 3, 3, 2, 2),
        "depths": gt_depths,
        "global_points": gt_points,
        "valid_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "rgb_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "depth_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "source_gs_masks": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "camera_poses": gt_poses,
        "norm_factors": torch.ones(1),
    }

    loss._compute_gs_loss(pred, gt)

    torch.testing.assert_close(
        fake_branch.world_points[:, :, ..., 0],
        torch.tensor([[[[2.0, 2.0], [2.0, 2.0]], [[4.0, 4.0], [4.0, 4.0]]]]),
    )
    torch.testing.assert_close(fake_branch.depth, gt_depths[:, 1:3])
    torch.testing.assert_close(
        fake_branch.source_camera_poses_c2w[:, :, 0, 3],
        torch.tensor([[0.0, 3.0]]),
    )
