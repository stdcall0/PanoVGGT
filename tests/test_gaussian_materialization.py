import torch

from panovggt.render.gs_branch import GSBranch, materialize_gaussians


def _make_branch(**overrides):
    branch = object.__new__(GSBranch)
    branch.renderer_name = "cube"
    branch.sh_degree = 1
    branch.scale_init_mode = "constant"
    branch.scale_init_value = 0.01
    branch.scale_init_factor = 1.0
    branch.scale_mult_min = None
    branch.scale_mult_max = None
    branch.offset_max_ratio = 0.5
    branch.use_offset = False
    branch.detach_centers = True
    branch.detach_camera = True
    branch.rotation_init_mode = "identity"
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


def _make_gs_params(batch=1, views=1, patch_h=2, patch_w=2, sh_degree=1, subgrid_size=1):
    sh_rest_channels = (sh_degree + 1) ** 2 - 1
    q = subgrid_size * subgrid_size
    if q > 1:
        return {
            "offset": torch.zeros(batch, views, patch_h, patch_w, q, 3),
            "scale": torch.full((batch, views, patch_h, patch_w, q, 3), 0.01),
            "rotation": torch.zeros(batch, views, patch_h, patch_w, q, 4),
            "opacity": torch.ones(batch, views, patch_h, patch_w, q, 1),
            "sh_dc": torch.zeros(batch, views, patch_h, patch_w, q, 3),
            "sh_rest": torch.zeros(batch, views, patch_h, patch_w, q, 3, sh_rest_channels),
        }
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


def test_free_materialize_matches_branch_masking_contract():
    gs_params = _make_gs_params()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.full((1, 1, 3, 4, 4), 0.25)
    point_masks = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    point_masks[..., :2, :2] = False
    gs_conf = {
        "renderer": "cube",
        "scale_init_mode": "constant",
        "scale_init_value": 0.01,
        "min_valid_ratio": 0.25,
        "train_dc": True,
        "train_opacity": True,
        "train_scale": False,
        "train_rotation": False,
        "train_sh_rest": False,
        "use_offset": False,
    }

    branch_out = _make_branch().materialize(
        gs_params=gs_params,
        world_points=world_points,
        images=images,
        depth=None,
        point_masks=point_masks,
    )
    free_out = materialize_gaussians(
        gs_params=gs_params,
        world_points=world_points,
        images=images,
        depth=None,
        gs_conf=gs_conf,
        sh_degree=1,
        point_masks=point_masks,
    )

    torch.testing.assert_close(free_out["opacities"], branch_out["opacities"])
    torch.testing.assert_close(free_out["patch_valid"], branch_out["patch_valid"])


def test_aggregate_predictions_passes_point_masks_to_materialization():
    from panovggt.utils.gs_export import aggregate_predictions

    gs_params = _make_gs_params()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)
    point_masks = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    point_masks[..., :2, :2] = False

    agg = aggregate_predictions(
        {
            "gaussian": gs_params,
            "world_points": world_points,
            "images": images,
            "point_masks": point_masks,
        },
        sh_degree=1,
        gs_conf={
            "scale_init_mode": "constant",
            "scale_init_value": 0.01,
            "min_valid_ratio": 0.25,
            "train_opacity": True,
        },
    )

    assert agg["opacities"][0].item() == 0.0
    assert agg["opacities"][-1].item() == 1.0


def test_smooth_bounded_scale_starts_at_unit_multiplier():
    gs_params = _make_gs_params()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)
    branch = _make_branch(scale_mult_min=0.25, scale_mult_max=1.25)
    branch.train_flags["scale"] = True

    out = branch.materialize(gs_params, world_points, images)

    torch.testing.assert_close(out["scale_mult"], torch.ones_like(out["scale_mult"]))


def test_smooth_bounded_scale_keeps_gradients_near_upper_bound():
    gs_params = _make_gs_params()
    gs_params["scale"] = (
        torch.full_like(gs_params["scale"], 0.01) * torch.exp(torch.tensor(3.0))
    )
    gs_params["scale"].requires_grad_()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)
    branch = _make_branch(scale_mult_min=0.25, scale_mult_max=1.25)
    branch.train_flags["scale"] = True

    out = branch.materialize(gs_params, world_points, images)
    out["scales"].sum().backward()

    assert out["scale_mult"].max().item() < 1.25
    assert gs_params["scale"].grad.abs().sum().item() > 0.0


def test_scale_relative_offset_starts_at_zero():
    gs_params = _make_gs_params()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)
    branch = _make_branch(use_offset=True)
    branch.train_flags["offset"] = True

    out = branch.materialize(gs_params, world_points, images)

    torch.testing.assert_close(out["offset"], torch.zeros_like(out["offset"]))
    torch.testing.assert_close(out["offset_ratio"], torch.zeros_like(out["offset_ratio"]))


def test_scale_relative_offset_is_bounded_and_keeps_gradients():
    gs_params = _make_gs_params()
    gs_params["offset"] = torch.full_like(gs_params["offset"], 3.0)
    gs_params["offset"].requires_grad_()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)
    branch = _make_branch(use_offset=True, offset_max_ratio=0.5)
    branch.train_flags["offset"] = True

    out = branch.materialize(gs_params, world_points, images)
    out["offset"].sum().backward()

    assert out["offset"].abs().max().item() < 0.005
    assert out["offset_ratio"].abs().max().item() < 0.5
    assert gs_params["offset"].grad.abs().sum().item() > 0.0


def test_tangent_rotation_init_aligns_local_z_to_center_ray():
    from panovggt.utils.rotation import quat_to_mat

    gs_params = _make_gs_params()
    world_points = torch.zeros(1, 1, 4, 4, 3)
    world_points[..., 0] = 1.0
    images = torch.zeros(1, 1, 3, 4, 4)
    branch = _make_branch(rotation_init_mode="tangent")

    out = branch.materialize(gs_params, world_points, images)
    rot = quat_to_mat(out["rotations"])

    torch.testing.assert_close(rot[..., :, 2], torch.tensor([1.0, 0.0, 0.0]).expand(1, 1, 2, 2, 3))
    torch.testing.assert_close(out["rotations"].norm(dim=-1), torch.ones(1, 1, 2, 2))


def test_linear_gaussian_head_can_emit_2x2_subgrid_tensors():
    from panovggt.layers.gaussian_head import LinearGaussianHead

    head = LinearGaussianHead(dec_embed_dim=8, sh_degree=1, subgrid_size=2)
    out = head(torch.zeros(1, 1, 8), Hp=1, Wp=1, B=1, S=1)

    assert out["sh_dc"].shape == (1, 1, 1, 1, 4, 3)
    assert out["sh_rest"].shape == (1, 1, 1, 1, 4, 3, 3)
    assert out["opacity"].shape == (1, 1, 1, 1, 4, 1)


def test_materialize_2x2_subgrid_uses_subpatch_centers_and_color_bootstrap():
    gs_params = _make_gs_params(patch_h=1, patch_w=1, subgrid_size=2)
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)
    values = [0.1, 0.2, 0.3, 0.4]
    for idx, (y0, x0) in enumerate(((0, 0), (0, 2), (2, 0), (2, 2))):
        world_points[:, :, y0:y0 + 2, x0:x0 + 2, 0] = float(idx)
        images[:, :, :, y0:y0 + 2, x0:x0 + 2] = values[idx]

    out = _make_branch().materialize(gs_params, world_points, images)

    assert out["centers"].shape == (1, 1, 1, 1, 4, 3)
    torch.testing.assert_close(out["centers"][0, 0, 0, 0, :, 0], torch.arange(4, dtype=torch.float32))
    c0 = 0.28209479177387814
    expected_dc = (torch.tensor(values) - 0.5) / c0
    torch.testing.assert_close(out["sh_dc"][0, 0, 0, 0, :, 0], expected_dc)


def test_aggregate_predictions_flattens_2x2_subgrid_gaussians():
    from panovggt.utils.gs_export import aggregate_predictions

    gs_params = _make_gs_params(patch_h=1, patch_w=1, subgrid_size=2)
    world_points = torch.zeros(1, 1, 4, 4, 3)
    images = torch.zeros(1, 1, 3, 4, 4)

    agg = aggregate_predictions(
        {"gaussian": gs_params, "world_points": world_points, "images": images},
        sh_degree=1,
        gs_conf={"scale_init_mode": "constant", "scale_init_value": 0.01},
    )

    assert agg["means"].shape == (4, 3)


class _RecordingRenderer:
    equ_h = 4

    def __init__(self):
        self.calls = []

    def render(self, means, quats, scales, opacities, colors_sh, w2c_per_view):
        self.calls.append(
            {
                "means_shape": tuple(means.shape),
                "quats_shape": tuple(quats.shape),
                "scales_shape": tuple(scales.shape),
                "opacities_shape": tuple(opacities.shape),
                "colors_shape": tuple(colors_sh.shape),
                "views": int(w2c_per_view.shape[0]),
            }
        )
        views = w2c_per_view.shape[0]
        return {
            "rgb_erp": torch.zeros(views, 3, 4, 8),
            "depth_erp": torch.zeros(views, 1, 4, 8),
            "alpha_erp": torch.ones(views, 1, 4, 8),
            "mask_erp": torch.ones(views, 1, 4, 8),
        }


def test_render_flattens_2x2_subgrid_gaussians_per_batch():
    branch = _make_branch()
    branch.renderer = _RecordingRenderer()
    gs_params = _make_gs_params(batch=2, views=1, patch_h=1, patch_w=1, subgrid_size=2)
    world_points = torch.zeros(2, 1, 4, 4, 3)
    images = torch.zeros(2, 1, 3, 4, 4)
    camera_poses = torch.eye(4).view(1, 1, 4, 4).expand(2, 1, 4, 4).clone()

    out = branch.render(
        gs_params=gs_params,
        world_points=world_points,
        camera_poses_c2w=camera_poses,
        images=images,
    )

    assert len(branch.renderer.calls) == 2
    assert branch.renderer.calls[0] == {
        "means_shape": (4, 3),
        "quats_shape": (4, 4),
        "scales_shape": (4, 3),
        "opacities_shape": (4,),
        "colors_shape": (4, 4, 3),
        "views": 1,
    }
    assert branch.renderer.calls[1] == branch.renderer.calls[0]
    assert out["rgb_erp"].shape == (2, 1, 3, 4, 8)
