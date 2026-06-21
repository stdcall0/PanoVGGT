import torch

from training.train_utils.view_split import build_gs_view_split, select_source_views


def test_build_gs_view_split_selects_last_target_before_model_forward():
    batch = {
        "images": torch.zeros(2, 4, 3, 2, 2),
        "depths": torch.zeros(2, 4, 2, 2),
        "norm_factors": torch.ones(2),
    }
    gs_conf = {
        "photometric_mode": "novel_view",
        "view_split_location": "trainer",
        "target_policy": "last",
        "num_target_views": 1,
    }

    source_idx, target_idx = build_gs_view_split(batch, gs_conf, training=True)
    source_batch = select_source_views(batch, source_idx)

    torch.testing.assert_close(source_idx.cpu(), torch.tensor([0, 1, 2]))
    torch.testing.assert_close(target_idx.cpu(), torch.tensor([3]))
    assert source_batch["images"].shape[1] == 3
    assert source_batch["depths"].shape[1] == 3
    assert source_batch["norm_factors"].shape == (2,)


def test_build_gs_view_split_is_inactive_for_source_reconstruction():
    batch = {"images": torch.zeros(1, 4, 3, 2, 2)}
    gs_conf = {
        "photometric_mode": "source_recon",
        "view_split_location": "trainer",
    }

    assert build_gs_view_split(batch, gs_conf, training=True) is None
