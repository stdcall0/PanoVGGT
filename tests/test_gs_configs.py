from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(name):
    return OmegaConf.load(REPO_ROOT / "training" / "config" / "gs" / name)


def _compose_config(name):
    config_dir = str(REPO_ROOT / "training" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=f"gs/{name.removesuffix('.yaml')}")


def test_local_gs_config_enables_2x2_tangent_and_masked_rgb_defaults():
    cfg = _load_config("local.yaml")

    assert cfg.model.gs_subgrid_size == 2
    assert cfg.loss.gs.mask_rgb_by_valid is True
    assert cfg.loss.gs.rgb_loss_type == "charbonnier"
    assert cfg.loss.gs.solid_angle_weight is True
    assert cfg.loss.gs.rotation_init_mode == "tangent"


def test_stage2c_novel_view_config_uses_trainer_side_bootstrap_contract():
    cfg = _load_config("stage2c_novel_view.yaml")

    assert cfg.loss.gs.photometric_mode == "novel_view"
    assert cfg.loss.gs.view_split_location == "trainer"
    assert cfg.loss.gs.bootstrap_geometry_source == "gt"
    assert cfg.loss.gs.mask_rgb_by_valid is True
    assert cfg.loss.gs.num_target_views == 1
    assert cfg.loss.gs.train_rotation is False
    assert cfg.loss.gs.train_sh_rest is False


def test_gs_stage_configs_use_init_checkpoint_for_stage_handoff():
    expected_init_paths = {
        "stage1_bootstrap.yaml": "./checkpoints/model.pt",
        "stage2_refine.yaml": "./outputs/gs_stage1_bootstrap/ckpts/checkpoint.pt",
        "stage2b_coverage.yaml": "./outputs/gs_stage2_refine/ckpts/checkpoint.pt",
        "stage2c_novel_view.yaml": "./outputs/gs_stage2b_coverage/ckpts/checkpoint.pt",
        "stage3_full_head.yaml": "./outputs/gs_stage2c_novel_view/ckpts/checkpoint.pt",
        "local.yaml": "./outputs/gs_stage1_bootstrap/ckpts/checkpoint.pt",
    }

    for config_name, init_path in expected_init_paths.items():
        cfg = _load_config(config_name)

        assert cfg.checkpoint.resume_checkpoint_path is None
        assert cfg.checkpoint.init_checkpoint_path == init_path


def test_gs_stage_configs_compose_with_gaussian_training_defaults():
    config_names = [
        "stage1_bootstrap.yaml",
        "stage2_refine.yaml",
        "stage2b_coverage.yaml",
        "stage2c_novel_view.yaml",
        "stage3_full_head.yaml",
        "local.yaml",
    ]

    for config_name in config_names:
        cfg = _compose_config(config_name)

        assert cfg.model.enable_gaussian is True
        assert cfg.model.gs_subgrid_size == 2
        assert cfg.loss.gs.enabled is True
        assert cfg.loss.gs.mask_rgb_by_valid is True
