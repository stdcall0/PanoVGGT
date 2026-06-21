from typing import Mapping, Optional, Tuple

import torch


NOVEL_VIEW_MODES = {"novel_view", "nvs", "source_target"}


def select_gs_target_indices(
    num_views: int,
    device: torch.device,
    num_target_views: int,
    target_policy: str,
    training: bool,
) -> torch.Tensor:
    if num_target_views < 1 or num_target_views >= num_views:
        raise ValueError(
            "num_target_views must be in [1, S-1] for trainer-side GS view split, "
            f"got {num_target_views} with S={num_views}."
        )
    policy = target_policy.lower()
    if policy == "first":
        target_idx = torch.arange(num_target_views, device=device)
    elif policy == "last":
        target_idx = torch.arange(num_views - num_target_views, num_views, device=device)
    elif policy == "random":
        if training:
            target_idx = torch.randperm(num_views, device=device)[:num_target_views]
            target_idx = target_idx.sort().values
        else:
            target_idx = torch.arange(num_views - num_target_views, num_views, device=device)
    else:
        raise ValueError(
            f"Unknown GS target_policy '{target_policy}'. Use 'first', 'last', or 'random'."
        )
    return target_idx


def build_gs_view_split(
    batch: Mapping,
    gs_conf: Mapping,
    training: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    mode = str(gs_conf.get("photometric_mode", "source_recon")).lower()
    split_location = str(gs_conf.get("view_split_location", "loss")).lower()
    if mode not in NOVEL_VIEW_MODES or split_location != "trainer":
        return None

    images = batch["images"]
    num_views = images.shape[1]
    target_idx = select_gs_target_indices(
        num_views=num_views,
        device=images.device,
        num_target_views=int(gs_conf.get("num_target_views", 1)),
        target_policy=str(gs_conf.get("target_policy", "last")),
        training=training,
    )
    source_mask = torch.ones(num_views, dtype=torch.bool, device=images.device)
    source_mask[target_idx] = False
    source_idx = torch.arange(num_views, device=images.device)[source_mask]
    if source_idx.numel() == 0:
        raise ValueError("trainer-side GS view split produced zero source views.")
    return source_idx, target_idx


def select_source_views(batch: Mapping, source_idx: torch.Tensor) -> dict:
    source_batch = dict(batch)
    num_views = batch["images"].shape[1]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[1] == num_views:
            source_batch[key] = value.index_select(1, source_idx)
    return source_batch
