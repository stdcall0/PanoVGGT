from types import SimpleNamespace

import torch

from training.data.datasets.stanford2d3ds import Stanford2D3DSDataset


def _common_conf():
    return SimpleNamespace(
        img_size=518,
        patch_size=14,
        rescale=False,
        rescale_aug=False,
        landscape_check=False,
        training=True,
        get_nearby=False,
        inside_random=False,
        allow_duplicate_img=False,
        augs={},
    )


def test_stanford_default_pole_mask_excludes_erp_top_and_bottom(tmp_path):
    dataset = Stanford2D3DSDataset(
        common_conf=_common_conf(),
        Stanford2D3DS_DIR=str(tmp_path),
        len_train=1,
    )
    frame_data = {
        "valid_mask": torch.ones(100, 200, dtype=torch.bool),
    }
    frame_data["valid_mask"][50, 10] = False

    dataset._apply_pano_pole_mask(frame_data)

    assert dataset.mask_pano_poles is True
    assert dataset.pano_pole_mask_ratio == 0.14
    assert frame_data["valid_mask"][:14].any().item() is False
    assert frame_data["valid_mask"][-14:].any().item() is False
    assert frame_data["valid_mask"][14:86, 0].all().item() is True
    assert frame_data["valid_mask"][50, 10].item() is False


def test_stanford_pole_mask_can_be_disabled(tmp_path):
    dataset = Stanford2D3DSDataset(
        common_conf=_common_conf(),
        Stanford2D3DS_DIR=str(tmp_path),
        len_train=1,
        mask_pano_poles=False,
    )
    frame_data = {
        "valid_mask": torch.ones(100, 200, dtype=torch.bool),
    }

    dataset._apply_pano_pole_mask(frame_data)

    assert frame_data["valid_mask"].all().item() is True
