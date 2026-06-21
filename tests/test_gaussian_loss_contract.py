import torch

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
