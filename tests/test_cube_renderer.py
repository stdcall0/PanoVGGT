import torch
import torch.nn as nn

from panovggt.render.cube_renderer import CubePanoRenderer


class _SumFacesCubeToEqui(nn.Module):
    def forward(self, cube):
        return cube.sum(dim=2)

    def get_valid_mask(self, views):
        return torch.ones(views, 1, 1, 1)


def test_cube_renderer_passes_background_and_normalizes_depth_moment_after_projection():
    renderer = CubePanoRenderer(
        equ_h=1,
        face_res=1,
        fov_deg=90.0,
        boundary_px=0,
        sh_degree=1,
        bg_color=0.25,
    )
    renderer.cube2equi = _SumFacesCubeToEqui()
    calls = {}

    def fake_rasterization(**kwargs):
        calls.update(kwargs)
        rgb_depth = torch.zeros(6, 1, 1, 4)
        rgb_depth[0, 0, 0, 3] = 2.0
        rgb_depth[1, 0, 0, 3] = 4.0
        alpha = torch.zeros(6, 1, 1, 1)
        alpha[0, 0, 0, 0] = 1.0
        alpha[1, 0, 0, 0] = 1.0
        return rgb_depth, alpha, {}

    renderer._rasterization = fake_rasterization
    out = renderer.render(
        means=torch.zeros(1, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.ones(1, 3),
        opacities=torch.ones(1),
        colors_sh=torch.zeros(1, 4, 3),
        w2c_per_view=torch.eye(4).view(1, 4, 4),
    )

    assert calls["render_mode"] == "RGB+D"
    torch.testing.assert_close(calls["backgrounds"], torch.full((6, 3), 0.25))
    torch.testing.assert_close(out["alpha_erp"], torch.full((1, 1, 1, 1), 2.0))
    torch.testing.assert_close(out["depth_erp"], torch.full((1, 1, 1, 1), 3.0))


def test_cube_renderer_reuses_static_tensors_for_same_device_and_dtype():
    renderer = CubePanoRenderer(
        equ_h=2,
        face_res=4,
        fov_deg=90.0,
        boundary_px=1,
        sh_degree=1,
    )
    device = torch.device("cpu")
    dtype = torch.float32

    K1 = renderer._intrinsics(device=device, dtype=dtype)
    K2 = renderer._intrinsics(device=device, dtype=dtype)
    ray1 = renderer._face_ray_norm(device=device, dtype=dtype)
    ray2 = renderer._face_ray_norm(device=device, dtype=dtype)
    mask1 = renderer._boundary_mask(device=device, dtype=dtype)
    mask2 = renderer._boundary_mask(device=device, dtype=dtype)

    assert K1.data_ptr() == K2.data_ptr()
    assert ray1.data_ptr() == ray2.data_ptr()
    assert mask1.data_ptr() == mask2.data_ptr()
