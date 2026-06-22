import torch
from omegaconf import OmegaConf

import inference


def test_normalize_gs_export_frame_uses_point_masks_for_scale():
    world_points = torch.tensor([[[[[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]]]])
    local_points = world_points.clone()
    camera_poses = torch.eye(4).view(1, 1, 4, 4)
    depth = torch.tensor([[[[[1.0], [100.0]]]]])
    point_masks = torch.tensor([[[[True, False]]]])

    world_norm, depth_norm, norm_factor, gs_camera_poses = inference.normalize_gs_export_frame(
        world_points=world_points,
        local_points=local_points,
        camera_poses=camera_poses,
        depth=depth,
        point_masks=point_masks,
    )

    torch.testing.assert_close(norm_factor, torch.tensor([1.0]))
    torch.testing.assert_close(world_norm[..., 0], world_points[..., 0])
    torch.testing.assert_close(depth_norm, depth)
    torch.testing.assert_close(gs_camera_poses[:, 0], torch.eye(4).view(1, 4, 4))


def test_load_model_rejects_missing_gaussian_head_when_gs_export_requested(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 8,
            "model": {
                "enable_camera": True,
                "enable_depth": True,
                "enable_point": True,
                "enable_gaussian": False,
                "aggregator": {},
                "gs_sh_degree": 1,
                "gs_scale_init": 0.01,
                "gs_opacity_init": 0.1,
                "gs_subgrid_size": 2,
            },
        }
    )
    monkeypatch.setattr(inference, "load_config", lambda _path: cfg)

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict, strict=False):
            return ["gaussian_head.sh_dc.weight", "backbone.weight"], []

    monkeypatch.setattr(inference, "PanoVGGTModel", FakeModel)
    ckpt_path = tmp_path / "no_gs.pt"
    torch.save({"model_state_dict": {"backbone.weight": torch.zeros(1)}}, ckpt_path)

    try:
        inference.load_model("dummy.yaml", str(ckpt_path), "cpu", enable_gaussian=True)
    except RuntimeError as exc:
        assert "gaussian_head" in str(exc)
    else:
        raise AssertionError("load_model should reject missing Gaussian head weights")
