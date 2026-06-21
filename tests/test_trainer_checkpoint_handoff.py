import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from training import trainer as trainer_module
from training.trainer import Trainer


def _checkpoint_conf(tmp_path, *, resume=None, init=None):
    return SimpleNamespace(
        save_dir=str(tmp_path / "ckpts"),
        resume_checkpoint_path=resume,
        init_checkpoint_path=init,
        strict=True,
    )


def test_checkpoint_selection_prefers_explicit_resume_over_stage_init(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "get_resume_checkpoint", lambda save_dir: None)
    conf = _checkpoint_conf(
        tmp_path,
        resume=str(tmp_path / "same_stage.pt"),
        init=str(tmp_path / "previous_stage.pt"),
    )

    resume_path, init_path = trainer_module._select_checkpoint_paths(conf)

    assert resume_path == str(tmp_path / "same_stage.pt")
    assert init_path is None


def test_checkpoint_selection_auto_resumes_same_stage_before_stage_init(tmp_path, monkeypatch):
    same_stage = str(tmp_path / "ckpts" / "checkpoint.pt")
    monkeypatch.setattr(trainer_module, "get_resume_checkpoint", lambda save_dir: same_stage)
    conf = _checkpoint_conf(tmp_path, resume=None, init=str(tmp_path / "previous_stage.pt"))

    resume_path, init_path = trainer_module._select_checkpoint_paths(conf)

    assert resume_path == same_stage
    assert init_path is None


def test_checkpoint_selection_uses_stage_init_only_without_same_stage_resume(tmp_path, monkeypatch):
    previous_stage = str(tmp_path / "previous_stage.pt")
    monkeypatch.setattr(trainer_module, "get_resume_checkpoint", lambda save_dir: None)
    conf = _checkpoint_conf(tmp_path, resume=None, init=previous_stage)

    resume_path, init_path = trainer_module._select_checkpoint_paths(conf)

    assert resume_path is None
    assert init_path == previous_stage


def test_model_init_checkpoint_does_not_restore_training_state(tmp_path):
    source_model = nn.Linear(1, 1)
    with torch.no_grad():
        source_model.weight.fill_(2.0)
        source_model.bias.fill_(3.0)
    ckpt_path = tmp_path / "previous_stage.pt"
    torch.save(
        {
            "model": source_model.state_dict(),
            "epoch": 17,
            "steps": {"train": 1234, "val": 56},
            "time_elapsed": 789.0,
        },
        ckpt_path,
    )

    target_model = nn.Linear(1, 1)
    with torch.no_grad():
        target_model.weight.zero_()
        target_model.bias.zero_()

    trainer = object.__new__(Trainer)
    trainer.model = target_model
    trainer.model_conf = SimpleNamespace(train_conf=False)
    trainer.checkpoint_conf = SimpleNamespace(strict=True)
    trainer.optim_conf = SimpleNamespace(amp=SimpleNamespace(enabled=False))
    trainer.mode = "train"
    trainer.rank = 0
    trainer.epoch = 0
    trainer.steps = {"train": 0, "val": 0}
    trainer.ckpt_time_elapsed = 0

    trainer._load_resuming_checkpoint(
        str(ckpt_path),
        load_model=True,
        load_optim=False,
        load_train_state=False,
    )

    assert torch.allclose(target_model.weight, torch.full_like(target_model.weight, 2.0))
    assert torch.allclose(target_model.bias, torch.full_like(target_model.bias, 3.0))
    assert trainer.epoch == 0
    assert trainer.steps == {"train": 0, "val": 0}
    assert trainer.ckpt_time_elapsed == 0
