import torch

from panovggt.render.gs_branch import GSBranch


def _make_branch(**overrides):
    branch = object.__new__(GSBranch)
    branch.renderer_name = "cube"
    branch.sh_degree = 1
    branch.scale_init_mode = "constant"
    branch.scale_init_value = 0.01
    branch.scale_init_factor = 1.0
    branch.scale_mult_min = None
    branch.scale_mult_max = None
    branch.use_offset = False
    branch.detach_centers = True
    branch.detach_camera = True
    branch.min_valid_ratio = 0.25
    branch.gs_head = None
    branch.train_flags = {
        "sh_dc": True,
        "opacity": True,
        "scale": False,
        "rotation": False,
        "sh_rest": False,
        "offset": False,
    }
    for key, value in overrides.items():
        setattr(branch, key, value)
    return branch


def _make_gs_params(batch=1, views=1, patch_h=2, patch_w=2, sh_degree=1):
    sh_rest_channels = (sh_degree + 1) ** 2 - 1
    return {
        "offset": torch.zeros(batch, views, patch_h, patch_w, 3),
        "scale": torch.full((batch, views, patch_h, patch_w, 3), 0.01),
        "rotation": torch.zeros(batch, views, patch_h, patch_w, 4),
        "opacity": torch.ones(batch, views, patch_h, patch_w, 1),
        "sh_dc": torch.zeros(batch, views, patch_h, patch_w, 3),
        "sh_rest": torch.zeros(batch, views, patch_h, patch_w, 3, sh_rest_channels),
    }


def test_materialize_applies_patch_mask_and_dc_bootstrap():
    batch, views, height, width = 1, 1, 4, 4
    gs_params = _make_gs_params(batch=batch, views=views)
    world_points = torch.zeros(batch, views, height, width, 3)
    images = torch.full((batch, views, 3, height, width), 0.75)
    point_masks = torch.ones(batch, views, height, width, dtype=torch.bool)
    point_masks[..., :2, :2] = False

    branch = _make_branch()
    out = branch.materialize(
        gs_params=gs_params,
        world_points=world_points,
        images=images,
        depth=None,
        point_masks=point_masks,
    )

    assert out["centers"].shape == (batch, views, 2, 2, 3)
    assert out["colors_sh"].shape == (batch, views, 2, 2, 4, 3)
    assert out["patch_valid"].shape == (batch, views, 2, 2, 1)
    assert out["opacities"][0, 0, 0, 0, 0].item() == 0.0
    assert out["opacities"][0, 0, 1, 1, 0].item() == 1.0

    c0 = 0.28209479177387814
    expected_dc = torch.full((3,), (0.75 - 0.5) / c0)
    torch.testing.assert_close(out["sh_dc"][0, 0, 1, 1], expected_dc)


def test_materialize_freezes_rotation_to_identity_when_not_trainable():
    gs_params = _make_gs_params()
    gs_params["rotation"].normal_()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)

    out = _make_branch().materialize(gs_params, world_points, images)

    expected = torch.zeros(1, 1, 2, 2, 4)
    expected[..., 0] = 1.0
    torch.testing.assert_close(out["rotations"], expected)
