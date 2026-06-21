import builtins
import sys
import types

import torch
import torch.nn as nn

from panovggt.render import cube_renderer
from panovggt.render.cube_renderer import CubePanoRenderer
from panovggt.render.cube_to_equi import Cube2Equirec


class _SumFacesCubeToEqui(nn.Module):
    def forward(self, cube):
        return cube.sum(dim=2)

    def get_valid_mask(self, views):
        return torch.ones(views, 1, 1, 1)


def test_cube_renderer_normalizes_depth_moment_after_projection():
    renderer = CubePanoRenderer(
        equ_h=1,
        face_res=1,
        fov_deg=90.0,
        boundary_px=0,
        sh_degree=1,
        bg_color=None,
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
    assert calls["backgrounds"] is None
    torch.testing.assert_close(out["alpha_erp"], torch.full((1, 1, 1, 1), 2.0))
    torch.testing.assert_close(out["depth_erp"], torch.full((1, 1, 1, 1), 3.0))


def test_cube_renderer_post_composites_nonzero_background_for_packed_rgbd():
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
        alpha = torch.zeros(6, 1, 1, 1)
        alpha[0, 0, 0, 0] = 0.25
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
    assert calls["packed"] is True
    assert calls["backgrounds"] is None
    torch.testing.assert_close(out["alpha_erp"], torch.full((1, 1, 1, 1), 0.25))
    torch.testing.assert_close(out["rgb_erp"], torch.full((1, 3, 1, 1), 0.1875))


def test_cube_renderer_omits_explicit_zero_background_for_packed_rgbd():
    renderer = CubePanoRenderer(
        equ_h=1,
        face_res=1,
        fov_deg=90.0,
        boundary_px=0,
        sh_degree=1,
        bg_color=0.0,
    )
    renderer.cube2equi = _SumFacesCubeToEqui()
    calls = {}

    def fake_rasterization(**kwargs):
        calls.update(kwargs)
        return torch.zeros(6, 1, 1, 4), torch.zeros(6, 1, 1, 1), {}

    renderer._rasterization = fake_rasterization
    renderer.render(
        means=torch.zeros(1, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.ones(1, 3),
        opacities=torch.ones(1),
        colors_sh=torch.zeros(1, 4, 3),
        w2c_per_view=torch.eye(4).view(1, 4, 4),
    )

    assert calls["render_mode"] == "RGB+D"
    assert calls["packed"] is True
    assert calls["backgrounds"] is None


def test_cube2equirec_matches_input_dtype_for_grid_sample_and_valid_mask():
    cube2equi = Cube2Equirec(face_w=2, equ_h=2, equ_w=4, fov_deg=90.0)
    cube = torch.zeros(1, 1, 6, 2, 2, dtype=torch.float64)

    out = cube2equi(cube)
    mask = cube2equi.get_valid_mask(1, device=cube.device, dtype=cube.dtype)

    assert out.dtype == torch.float64
    assert mask.dtype == torch.float64


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


def test_gsplat_backend_loader_falls_back_when_fcntl_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(cube_renderer, "_GSPLAT_BACKEND_READY", False)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))

    gsplat_module = types.ModuleType("gsplat")
    cuda_module = types.ModuleType("gsplat.cuda")
    cuda_module._backend = object()
    gsplat_module.cuda = cuda_module
    monkeypatch.setitem(sys.modules, "gsplat", gsplat_module)
    monkeypatch.setitem(sys.modules, "gsplat.cuda", cuda_module)

    real_import = builtins.__import__
    import_attempts = []

    def import_without_fcntl(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fcntl":
            import_attempts.append(name)
            raise ImportError("No module named fcntl")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_fcntl)

    cube_renderer._ensure_gsplat_backend_loaded_once()

    assert import_attempts == ["fcntl"]
    assert cube_renderer._GSPLAT_BACKEND_READY is True
