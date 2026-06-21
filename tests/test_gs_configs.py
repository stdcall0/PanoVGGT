from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(name):
    return OmegaConf.load(REPO_ROOT / "training" / "config" / "gs" / name)


def test_local_gs_config_enables_2x2_tangent_and_robust_rgb_defaults():
    cfg = _load_config("local.yaml")

    assert cfg.model.gs_subgrid_size == 2
    assert cfg.loss.gs.mask_rgb_by_valid is False
    assert cfg.loss.gs.rgb_loss_type == "charbonnier"
    assert cfg.loss.gs.solid_angle_weight is True
    assert cfg.loss.gs.rotation_init_mode == "tangent"


def test_stage2c_novel_view_config_uses_trainer_side_bootstrap_contract():
    cfg = _load_config("stage2c_novel_view.yaml")

    assert cfg.loss.gs.photometric_mode == "novel_view"
    assert cfg.loss.gs.view_split_location == "trainer"
    assert cfg.loss.gs.bootstrap_geometry_source == "gt"
    assert cfg.loss.gs.num_target_views == 1
    assert cfg.loss.gs.train_rotation is False
    assert cfg.loss.gs.train_sh_rest is False
