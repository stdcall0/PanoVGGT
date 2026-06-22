from training.data.datasets.matterport3d import Matterport3DDataset


def test_matterport_prefers_fixed_skybox_rgb_when_available(tmp_path):
    scene_path = tmp_path / "scene"
    perspective = scene_path / "pano_color_perspective"
    fixed = scene_path / "pano_skybox_color_fixed"
    perspective.mkdir(parents=True)
    fixed.mkdir(parents=True)
    (perspective / "view_0.png").write_bytes(b"perspective")
    (fixed / "view_0.png").write_bytes(b"fixed")

    dataset = object.__new__(Matterport3DDataset)

    assert dataset._resolve_color_path(str(scene_path), "view_0") == str(fixed / "view_0.png")
