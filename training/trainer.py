# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

# --- Environment Variable Setup for Performance and Debugging ---

import contextlib
import gc
import json
import logging as python_logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence
import inspect

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers


def _safe_barrier(local_rank: int):
    """A safe barrier with explicit device binding when NCCL supports it."""
    if not dist.is_available() or not dist.is_initialized():
        return
    sig = inspect.signature(dist.barrier)
    if "device_ids" in sig.parameters:
        dist.barrier(device_ids=[local_rank])
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize(local_rank)
        dist.barrier()


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.
    """

    EPSILON = 1e-8

    def __init__(
            self,
            *,
            data: Dict[str, Any],
            model: Dict[str, Any],
            logging: Dict[str, Any],
            checkpoint: Dict[str, Any],
            max_epochs: int,
            mode: str = "train",
            device: str = "cuda",
            seed_value: int = 123,
            val_epoch_freq: int = 1,
            distributed: Dict[str, bool] = None,
            cuda: Dict[str, bool] = None,
            limit_train_batches: Optional[int] = None,
            limit_val_batches: Optional[int] = None,
            optim: Optional[Dict[str, Any]] = None,
            loss: Optional[Dict[str, Any]] = None,
            env_variables: Optional[Dict[str, Any]] = None,
            accum_steps: int = 1,
            **kwargs,
    ):
        """
        Initializes the Trainer.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store configs
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Hparams
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        self.where = 0.0

        
        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            exp_name=self.logging_conf.log_exp,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        assert is_dist_avail_and_initialized(), "Torch distributed must be initialized before using Trainer."

        # 3) Build model/loss/logger/AMP components before creating dataloaders.
        self._setup_components()

        
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # 5) Restore model weights before DDP wrapping.
        ckpt_path = self.checkpoint_conf.resume_checkpoint_path or get_resume_checkpoint(self.checkpoint_conf.save_dir)
        init_ckpt_path = getattr(self.checkpoint_conf, "init_checkpoint_path", None)
        if ckpt_path is None and init_ckpt_path is not None:
            self._load_initial_checkpoint(init_ckpt_path)
        if ckpt_path is not None:
            
            self._load_resuming_checkpoint(ckpt_path, load_model=True, load_optim=False)

        # 6) Wrap model with DDP after device placement.
        self._setup_ddp_distributed_training(distributed, device)

        # 7) Build optimizers after DDP wrapping.
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Restore optimizer/scaler state if resuming training.
        if ckpt_path is not None and self.mode == "train":
            self._load_resuming_checkpoint(ckpt_path, load_model=False, load_optim=True)

        # 8) DDP DataLoader / Dataset
        self._setup_dataloaders()

        # 9) barrier
        _safe_barrier(self.local_rank)

    def _setup_timers(self):
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str, load_model: bool = True, load_optim: bool = True):
        """Loads model/optimizer/scaler from checkpoint with DDP-awareness."""
        logging.info(f"Resuming from {ckpt_path} (rank {self.rank})")
        with g_pathmgr.open(ckpt_path, "rb") as f:
            # Training checkpoints include optimizer scheduler metadata that can
            # contain OmegaConf containers. They are produced locally by this
            # trainer, so use the full loader for resume instead of PyTorch's
            # restricted weights-only path.
            checkpoint = torch.load(f, map_location="cpu", weights_only=False)

        model_state_dict = checkpoint.get("model", checkpoint)

        
        train_conf = getattr(self.model_conf, 'train_conf', False)

        
        if load_model:
            target_model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            if train_conf:
                missing, unexpected = target_model.load_state_dict(model_state_dict, strict=False)
                logging.info(f"[Confidence] Loaded model (non-strict). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                if self.rank == 0:
                    if missing:
                        logging.info(f"Missing keys (confidence params): {missing}")
                    if unexpected:
                        logging.warning(f"Unexpected keys: {unexpected}")
            else:
                missing, unexpected = target_model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)
                if self.rank == 0:
                    logging.info(f"Loaded model. Missing: {missing or 'None'}, Unexpected: {unexpected or 'None'}.")

        
        if load_optim and self.mode == "train" and "optimizer" in checkpoint and hasattr(self, "optims"):
            if train_conf:
                logging.info(f"[Confidence] Skip loading optimizer state (start fresh).")
            else:
                logging.info(f"Loading optimizer state dict (rank {self.rank})")
                
                try:
                    self.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    logging.warning(f"Failed to load optimizer state: {e}")

        
        if train_conf:
            self.epoch = 0
            self.steps = {"train": 0, "val": 0}
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)
        else:
            if "prev_epoch" in checkpoint:
                self.epoch = checkpoint["prev_epoch"] + 1
            elif "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
            else:
                self.epoch = 0
            self.steps = checkpoint.get("steps", {"train": 0, "val": 0})
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # 4) AMP scaler
        if hasattr(self, "scaler") and self.optim_conf.amp.enabled and "scaler" in checkpoint:
            if train_conf:
                logging.info(f"[Confidence] Reset AMP scaler state (skip loading).")
            else:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                except Exception as e:
                    logging.warning(f"Failed to load AMP scaler: {e}")

    def _load_initial_checkpoint(self, ckpt_path: str):
        """Load model weights only, without restoring optimizer or training progress."""
        logging.info(f"Initializing model weights from {ckpt_path} (rank {self.rank})")
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=True)

        model_state_dict = checkpoint.get("model", checkpoint)
        target_model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        model_state_dict = self._adapt_initial_state_dict(model_state_dict, target_model)
        missing, unexpected = target_model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(
                f"Initialized model. Missing: {missing or 'None'}, Unexpected: {unexpected or 'None'}."
            )

    def _adapt_initial_state_dict(self, state_dict: Mapping[str, Any], target_model: nn.Module) -> Dict[str, Any]:
        """Reuse compatible pretrained branches for newly added 3DGS modules."""
        adapted = dict(state_dict)
        target_state = target_model.state_dict()
        copied = []

        def copy_if_shape_matches(src_key: str, dst_key: str) -> None:
            if dst_key in adapted or src_key not in state_dict or dst_key not in target_state:
                return
            src_value = state_dict[src_key]
            if hasattr(src_value, "shape") and src_value.shape == target_state[dst_key].shape:
                adapted[dst_key] = src_value
                copied.append((src_key, dst_key))

        for key in list(target_state.keys()):
            if key.startswith("gaussian_decoder."):
                suffix = key[len("gaussian_decoder."):]
                copy_if_shape_matches(f"global_points_decoder.{suffix}", key)

        point_w_key = "global_point_head.proj.weight"
        point_b_key = "global_point_head.proj.bias"

        point_weight = state_dict.get(point_w_key)
        point_bias = state_dict.get(point_b_key)
        patch_size = int(getattr(target_model, "patch_size", 14))
        expected_rows = 3 * patch_size * patch_size
        slot_count = int(
            getattr(target_model, "gaussian_head_config", {}).get(
                "max_gaussians_per_token", 4
            )
        )

        def slot_ids_for_patch(device: torch.device) -> torch.Tensor:
            if slot_count <= 1:
                return torch.zeros(patch_size * patch_size, device=device, dtype=torch.long)
            cols = int(math.ceil(math.sqrt(slot_count)))
            rows = int(math.ceil(slot_count / cols))
            y = torch.arange(patch_size, device=device)
            x = torch.arange(patch_size, device=device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            row_id = torch.clamp((yy * rows) // patch_size, max=rows - 1)
            col_id = torch.clamp((xx * cols) // patch_size, max=cols - 1)
            return torch.clamp(row_id * cols + col_id, max=slot_count - 1).reshape(-1).long()

        def per_slot_average_patch_tensor(value: torch.Tensor) -> Optional[torch.Tensor]:
            if (
                value is None
                or not hasattr(value, "shape")
                or value.shape[0] != expected_rows
            ):
                return None
            reshaped = value.view(3, patch_size * patch_size, *value.shape[1:])
            slot_ids = slot_ids_for_patch(reshaped.device)
            slots = []
            for slot_idx in range(slot_count):
                mask = slot_ids == slot_idx
                if not mask.any().item():
                    slots.append(reshaped.mean(dim=1))
                else:
                    slots.append(reshaped[:, mask].mean(dim=1))
            return torch.stack(slots, dim=0)

        slot_weight = per_slot_average_patch_tensor(point_weight)
        slot_bias = per_slot_average_patch_tensor(point_bias)

        anchor_w_key = "gaussian_param_head.mean_anchor_head.weight"
        anchor_b_key = "gaussian_param_head.mean_anchor_head.bias"
        if (
            anchor_w_key not in adapted
            and anchor_w_key in target_state
            and slot_weight is not None
            and slot_weight.shape[0] >= 1
            and slot_weight.shape[1:] == target_state[anchor_w_key].shape
        ):
            adapted[anchor_w_key] = slot_weight[0]
            copied.append((point_w_key, anchor_w_key))

        if (
            anchor_b_key not in adapted
            and anchor_b_key in target_state
            and slot_bias is not None
            and slot_bias.shape[0] >= 1
            and slot_bias.shape[1:] == target_state[anchor_b_key].shape
        ):
            adapted[anchor_b_key] = slot_bias[0]
            copied.append((point_b_key, anchor_b_key))

        child_w_key = "gaussian_param_head.child_anchor_offset_head.weight"
        child_b_key = "gaussian_param_head.child_anchor_offset_head.bias"
        if (
            child_w_key not in adapted
            and child_w_key in target_state
            and slot_weight is not None
            and slot_weight.shape[0] > 1
            and slot_weight.shape[1:] == target_state[anchor_w_key].shape
        ):
            child_offsets = (slot_weight[1:] - slot_weight[:1]).reshape(
                target_state[child_w_key].shape
            )
            adapted[child_w_key] = child_offsets
            copied.append((point_w_key, child_w_key))

        if (
            child_b_key not in adapted
            and child_b_key in target_state
            and slot_bias is not None
            and slot_bias.shape[0] > 1
            and slot_bias.shape[1:] == target_state[anchor_b_key].shape
        ):
            child_bias_offsets = (slot_bias[1:] - slot_bias[:1]).reshape(
                target_state[child_b_key].shape
            )
            adapted[child_b_key] = child_bias_offsets
            copied.append((point_b_key, child_b_key))

        if copied and self.rank == 0:
            logging.info(
                "Adapted %d pretrained global-point tensors for Gaussian initialization.",
                len(copied),
            )
        return adapted

    def _setup_device(self, device: str):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # TB writer / model / loss / clipper / scaler
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.optim_conf.amp.enabled)

        
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}")
            self.model = freeze_modules(self.model, patterns=self.optim_conf.frozen_module_names)
            logging.info(f"[Done]  Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}")

        # rank0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, self.logging_conf.log_exp, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Instantiates datasets/dataloaders AFTER DDP is ready."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get('val', None), _recursive_=False)
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        assert isinstance(self.model, torch.nn.Module)
        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        saver.save_checkpoint(
            model=model,
            ema_models=None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            dataloader = self.train_dataset.get_loader(epoch=self.epoch)

            final_train_loss_meters = self.train_epoch(dataloader)
            self._log_epoch_metrics(final_train_loss_meters, 'train', self.epoch)
            self.save_checkpoint(self.epoch)

            del dataloader, final_train_loss_meters
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()

            self.epoch += 1

        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=self.epoch)
        final_val_loss_meters = self.val_epoch(dataloader)
        self._log_epoch_metrics(final_val_loss_meters, 'val', self.epoch)

        del dataloader, final_val_loss_meters
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _log_epoch_metrics(self, loss_meters: Dict[str, AverageMeter], phase: str, epoch: int):
        """Synchronize and log epoch metrics to TensorBoard on rank 0."""
        _safe_barrier(self.local_rank)

        for name, meter in loss_meters.items():
            if f"Loss/{phase}" in name:
                try:
                    if hasattr(meter, 'sum'):
                        local_sum = meter.sum
                    else:
                        local_sum = meter.avg * meter.count if meter.count > 0 else 0.0
                    local_count = meter.count if hasattr(meter, 'count') else 0

                    sync_tensor = torch.tensor([local_sum, local_count], dtype=torch.float32, device=self.device)
                    dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)

                    global_sum = sync_tensor[0].item()
                    global_count = sync_tensor[1].item()
                    global_avg = (global_sum / global_count) if global_count > 0 else 0.0

                    if self.rank == 0:
                        metric_name = name.replace(f"Loss/{phase}_", "")
                        self.tb_writer.log(f"{phase}/epoch_metrics/{metric_name}", global_avg, epoch)
                except Exception as e:
                    logging.error(f"Error in _log_epoch_metrics for {name}: {e}")
                    continue

        _safe_barrier(self.local_rank)

    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = iters_per_epoch if self.limit_val_batches is None else self.limit_val_batches
        progress_denom = max(self.max_epochs - 1, 1)

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break

            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.amp.autocast('cuda', enabled=False):
                batch = self._process_batch(batch)
            batch["gaussian_progress"] = min(
                (self.epoch + float(data_iter) / max(limit_val_batches, 1)) / progress_denom,
                1.0,
            )
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"]
            amp_dtype = torch.bfloat16 if amp_type == "bfloat16" else torch.float16

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=amp_dtype):
                    val_loss_dict = self._step(batch, self.model, phase, loss_meters)

            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return loss_meters

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}

        for config in self.gradient_clipper.configs:
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = iters_per_epoch if self.limit_train_batches is None else self.limit_train_batches
        progress_denom = max(self.max_epochs - 1, 1)

        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break

            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.amp.autocast('cuda', enabled=False):
                batch = self._process_batch(batch)
            batch["gaussian_progress"] = min(
                (self.epoch + float(data_iter) / max(limit_train_batches, 1)) / progress_denom,
                1.0,
            )
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # from training.train_utils.debug_vis_batch import save_batch_visualization

            # is_rank0 = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
            # print("is_rank0", is_rank0)
            # if is_rank0:
            # try:
            # vis_steps = int(os.environ.get("DEBUG_VIS_STEPS", "4"))
            # vis_idx = int(os.environ.get("DEBUG_VIS_IDX", "0"))
            # # Debug visualization uses current epoch and iteration counters.
            # if data_iter < vis_steps:
            # out_dir = getattr(self, "log_dir", "./logs")
            # out_dir = os.path.join(out_dir, "debug_vis")
            # save_batch_visualization(batch, batch_idx=vis_idx, output_base_dir=out_dir)

            # # Optional debugger break at step N via DEBUG_VIS_BREAK_AT=N.
            # break_at = os.environ.get("DEBUG_VIS_BREAK_AT", None)
            # if break_at is not None and data_iter == int(break_at):
            # import pdb; pdb.set_trace()
            # except Exception as e:
            # print(f"[DEBUG_VIS] Visualization failed at step {data_iter}: {e}")


            accum_steps = self.accum_steps
            chunked_batches = [batch] if accum_steps == 1 else chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(chunked_batches, phase, loss_meters)

            # step schedulers
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(f"Skipping scheduler update at end of training. where={self.where}")

            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (f"{i}_" if len(self.optims) > 1
                                            else (f"{j}_" if len(optim.optimizer.param_groups) > 1 else ""))
                            self.tb_writer.log(os.path.join("Optim", f"{optim_prefix}", option),
                                               param_group[option], self.steps[phase])
                self.tb_writer.log(os.path.join("Optim", "where"), self.where, self.steps[phase])

            # Grad clipping / nan guard
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)
                grad_norm_dict = self.gradient_clipper(model=self.model)
                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optim step
            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return loss_meters

    def _run_steps_on_batch_chunks(self, chunked_batches: List[Any], phase: str, loss_meters: Dict[str, AverageMeter]):
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"]
        amp_dtype = torch.bfloat16 if amp_type == "bfloat16" else torch.float16

        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (self.model.no_sync() if i < accum_steps - 1 else contextlib.nullcontext())
            with ddp_context:
                with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=amp_dtype):
                    loss_dict = self._step(chunked_batch, self.model, phase, loss_meters)

                loss = loss_dict["loss_objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    logging.error(f"Loss is {loss.item()}, stop training")
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()

    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        tensor_keys = ["images", "depths", "extrinsics", "intrinsics", "cam_points", "world_points", "point_masks"]
        string_keys = ["seq_name"]
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, torch.flip(original_tensor, dims=[1])], dim=0)
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        return batch

    def _process_batch(self, batch: Mapping):
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths, avg_scale = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths
        batch["norm_factors"] = avg_scale
        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        y_hat = model(images=batch["images"])
        loss_dict = self.loss(y_hat, batch)

        log_data = {**y_hat, **loss_dict, **batch}
        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase]["keys_to_log"]
            assert len(keys_to_log) > 0, "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase]["modality"]
            assert modality in ["image", "video"], "Currently only support video or image logging"

            name = f"Visuals/{phase}"
            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(name, visuals_to_log, step, self.logging_conf.video_logging_fps)


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    if accum_steps == 1:
        return [batch]
    batch_size = get_chunkable_length(batch)
    if batch_size is None or batch_size <= 1:
        return [batch]

    effective_chunks = min(accum_steps, batch_size)
    chunked_batches = [get_chunk_from_data(batch, i, effective_chunks) for i in range(effective_chunks)]
    return [chunk for chunk in chunked_batches if get_chunkable_length(chunk) not in (None, 0)]


def is_sequence_of_primitives(data: Any) -> bool:
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )


def get_chunkable_length(data: Any) -> Optional[int]:
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        return len(data)
    if isinstance(data, Mapping):
        for value in data.values():
            length = get_chunkable_length(value)
            if length is not None:
                return length
        return None
    if isinstance(data, Sequence) and not isinstance(data, str):
        for value in data:
            length = get_chunkable_length(value)
            if length is not None:
                return length
        return None
    return None


def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        start = (len(data) * chunk_id) // num_chunks
        end = (len(data) * (chunk_id + 1)) // num_chunks
        return data[start:end]
    elif isinstance(data, Mapping):
        return {key: get_chunk_from_data(value, chunk_id, num_chunks) for key, value in data.items()}
    elif isinstance(data, str):
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
