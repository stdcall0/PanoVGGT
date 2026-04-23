# panovggt.py

import math
import os

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from panovggt.models.aggregator import Aggregator
from panovggt.layers.transformer_head import (
    TransformerDecoder,
    LinearPts3d,
    ContextTransformerDecoder,
)
from panovggt.layers.gaussian_head import (
    GaussianParameterHead,
    GaussianStage1Head,
    assemble_gaussian_outputs,
)
from panovggt.layers.camera_head import CameraHead


def _homogenize_points(xyz: torch.Tensor) -> torch.Tensor:
    """Convert (…, 3) points to homogeneous (…, 4)."""
    ones = torch.ones_like(xyz[..., :1])
    return torch.cat([xyz, ones], dim=-1)


def _build_pano_pos_embed_single(
    Hp: int,
    Wp: int,
    Cpos: int,
    patch_start_idx: int,
    device: torch.device,
    pano_pos_mlp: nn.Module,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build absolute spherical position embedding for a single frame."""
    ys = torch.arange(Hp, device=device, dtype=dtype) + 0.5
    xs = torch.arange(Wp, device=device, dtype=dtype) + 0.5
    theta = (ys[:, None] / Hp - 0.5) * math.pi
    phi = (xs[None, :] / Wp - 0.5) * (2 * math.pi)
    theta = theta.expand(Hp, Wp)
    phi = phi.expand(Hp, Wp)

    pos_feats = torch.stack(
        [torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)],
        dim=-1,
    ).reshape(Hp * Wp, 4)

    wparam = next(pano_pos_mlp.parameters(), None)
    if wparam is not None:
        pos_feats = pos_feats.to(device=wparam.device, dtype=wparam.dtype)

    pos_patch = pano_pos_mlp(pos_feats)

    # Register tokens get zero positional encoding
    zeros_reg = torch.zeros(
        patch_start_idx, Cpos, device=pos_patch.device, dtype=pos_patch.dtype
    )
    return torch.cat([zeros_reg, pos_patch], dim=0)  # (P, Cpos)


class PositionAdapter(nn.Module):
    """Adapter that maps aggregator position embeddings to decoder-specific dimensions."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bottleneck_factor: int = 4,
        use_layernorm: bool = True,
        dropout: float = 0.0,
        init_as_identity: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        hidden_dim = max(32, min(in_dim, out_dim) // bottleneck_factor)

        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        self.projection = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )
        self.ln = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.register_buffer(
            "_res_scale", torch.tensor(float(residual_scale)), persistent=False
        )

        if init_as_identity:
            if isinstance(self.net[-1], nn.Linear):
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)
            if isinstance(self.projection, nn.Linear):
                nn.init.xavier_uniform_(self.projection.weight)
                nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id = self.projection(self.ln(x))
        return x_id + self._res_scale * self.net(x)


class PanoVGGTModel(nn.Module, PyTorchModelHubMixin):
    """
    PanoVGGT: Panoramic Visual Geometry Grounded Transformer.

    Uses absolute spherical position encoding for panoramic images,
    combined with RoPE for relative positional relationships.
    """

    DEFAULT_AGGREGATOR_CONFIG = {
        "depth": 36,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "patch_embed": "dinov2_vitl14_reg",
        "num_register_tokens": 5,
        "qkv_bias": True,
        "proj_bias": True,
        "ffn_bias": True,
        "qk_norm": True,
        "rope_freq": 100,
        "init_values": 0.01,
        "use_pano_pos": True,
        "pos_mlp_hidden": 1024,
    }
    DEFAULT_GAUSSIAN_HEAD_CONFIG = {
        "enable_stage1": True,
        "keep_threshold": 0.35,
        "max_gaussians_per_token": 4,
        "sh_degree": 3,
        "dec_embed_dim": 1024,
        "dec_num_heads": 16,
        "out_dim": 1024,
        "stage1_hidden_dim": 512,
        "mlp_hidden_dim": 1024,
        "log_scale_min": -6.0,
        "log_scale_max": 1.5,
    }

    def __init__(
        self,
        aggregator: dict = None,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_depth: bool = True,
        enable_global_points: bool = True,
        enable_3dgs: bool = False,
        gaussian_head: dict = None,
        **kwargs,
    ):
        super().__init__()

        if aggregator is None:
            aggregator = self.DEFAULT_AGGREGATOR_CONFIG.copy()
        if gaussian_head is None:
            gaussian_head = self.DEFAULT_GAUSSIAN_HEAD_CONFIG.copy()
        else:
            merged_gaussian_head = self.DEFAULT_GAUSSIAN_HEAD_CONFIG.copy()
            merged_gaussian_head.update(gaussian_head)
            gaussian_head = merged_gaussian_head

        # 1) Aggregator
        self.aggregator = Aggregator(
            **aggregator, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.patch_size = patch_size

        # 2) Branch switches
        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_depth = enable_depth
        self.enable_global_points = enable_global_points
        self.enable_3dgs = enable_3dgs
        self.gaussian_head_config = gaussian_head

        # 3) Decoder heads
        in_dim_for_decoders = 2 * embed_dim

        if self.enable_point:
            self.point_decoder = TransformerDecoder(
                in_dim=in_dim_for_decoders,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=getattr(self.aggregator, "rope", None),
            )
            self.point_head = LinearPts3d(
                patch_size=self.patch_size, dec_embed_dim=1024, output_dim=3,
            )

        if self.enable_camera:
            self.camera_decoder = TransformerDecoder(
                in_dim=in_dim_for_decoders,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=512,
                rope=getattr(self.aggregator, "rope", None),
                use_checkpoint=False,
            )
            self.camera_head = CameraHead(dim=512)

        if self.enable_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=in_dim_for_decoders,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=getattr(self.aggregator, "rope", None),
            )
            self.global_point_head = LinearPts3d(
                patch_size=self.patch_size, dec_embed_dim=1024, output_dim=3,
            )

        if self.enable_3dgs:
            self.gaussian_stage1_enabled = bool(
                gaussian_head.get("enable_stage1", True)
            )
            self.gaussian_keep_threshold = float(
                gaussian_head.get("keep_threshold", 0.35)
            )
            self.gaussian_decoder = ContextTransformerDecoder(
                in_dim=in_dim_for_decoders,
                dec_embed_dim=gaussian_head["dec_embed_dim"],
                dec_num_heads=gaussian_head["dec_num_heads"],
                out_dim=gaussian_head["out_dim"],
                rope=getattr(self.aggregator, "rope", None),
            )
            self.gaussian_stage1_head = GaussianStage1Head(
                in_dim=gaussian_head["out_dim"],
                hidden_dim=gaussian_head["stage1_hidden_dim"],
                max_gaussians_per_token=gaussian_head["max_gaussians_per_token"],
            )
            self.gaussian_param_head = GaussianParameterHead(
                in_dim=gaussian_head["out_dim"],
                hidden_dim=gaussian_head["mlp_hidden_dim"],
                max_gaussians_per_token=gaussian_head["max_gaussians_per_token"],
                sh_degree=gaussian_head["sh_degree"],
                log_scale_min=gaussian_head["log_scale_min"],
                log_scale_max=gaussian_head["log_scale_max"],
            )

        # 4) Absolute spherical position encoding adapters
        if not hasattr(self.aggregator, "pano_pos_mlp"):
            raise AttributeError(
                "Aggregator missing 'pano_pos_mlp'. "
                "Set use_pano_pos=True in Aggregator config."
            )

        # Auto-detect position embedding dimension
        mlp = self.aggregator.pano_pos_mlp
        if isinstance(mlp, nn.Sequential) and len(mlp) > 0 and isinstance(mlp[-1], nn.Linear):
            self.Cpos = mlp[-1].out_features
        else:
            self.Cpos = 1024

        # Per-branch position adapters
        self.pos_adapters = nn.ModuleDict()

        if self.enable_point:
            dim, _ = self._get_dec_cfg(self.point_decoder)
            self.pos_adapters["point"] = PositionAdapter(
                in_dim=self.Cpos, out_dim=dim,
                init_as_identity=True, residual_scale=0.0,
            )

        if self.enable_camera:
            dim, _ = self._get_dec_cfg(self.camera_decoder)
            self.pos_adapters["camera"] = PositionAdapter(
                in_dim=self.Cpos, out_dim=dim,
                init_as_identity=True, residual_scale=0.0,
            )

        if self.enable_global_points:
            dim, _ = self._get_dec_cfg(self.global_points_decoder)
            self.pos_adapters["global"] = PositionAdapter(
                in_dim=self.Cpos, out_dim=dim,
                init_as_identity=True, residual_scale=0.0,
            )

        if self.enable_3dgs:
            dim, _ = self._get_dec_cfg(self.gaussian_decoder)
            self.pos_adapters["gaussian"] = PositionAdapter(
                in_dim=self.Cpos,
                out_dim=dim,
                init_as_identity=True,
                residual_scale=0.0,
            )

        # Direction vector cache for equirectangular projection
        self.register_buffer("_direction_vectors_cache", None, persistent=False)
        self.register_buffer("_cache_hw", None, persistent=False)

    @staticmethod
    def _get_dec_cfg(decoder):
        dim = getattr(decoder, "dec_embed_dim", None)
        heads = getattr(decoder, "dec_num_heads", None)
        return dim, heads

    def _get_branch_pos_embed(
        self, Hp, Wp, patch_start_idx, device, dtype, branch_name, BS
    ):
        """Generate branch-specific absolute position embedding (BS, P, Dim)."""
        pos_single_core = _build_pano_pos_embed_single(
            Hp, Wp, self.Cpos, patch_start_idx, device,
            self.aggregator.pano_pos_mlp, dtype,
        )
        adapter = self.pos_adapters[branch_name]
        pos_single = adapter(pos_single_core)
        return pos_single.unsqueeze(0).expand(BS, -1, -1)

    def _get_direction_vectors(self, H, W, device, dtype):
        """Compute or retrieve cached ERP direction vectors."""
        needs_recompute = (
            self._direction_vectors_cache is None
            or self._cache_hw is None
            or self._cache_hw.numel() != 2
            or self._cache_hw[0].item() != H
            or self._cache_hw[1].item() != W
        )
        if needs_recompute:
            u = torch.arange(W, device=device, dtype=dtype) + 0.5
            v = torch.arange(H, device=device, dtype=dtype) + 0.5
            phi = (u / W - 0.5) * 2 * torch.pi
            theta = -(v / H - 0.5) * torch.pi
            grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing="ij")
            dir_z = torch.cos(grid_theta) * torch.cos(grid_phi)
            dir_x = torch.cos(grid_theta) * torch.sin(grid_phi)
            dir_y = -torch.sin(grid_theta)
            self._direction_vectors_cache = torch.stack([dir_x, dir_y, dir_z], dim=-1)
            self._cache_hw = torch.tensor([H, W], device=device, dtype=torch.int64)
        elif self._direction_vectors_cache.device != device:
            self._direction_vectors_cache = self._direction_vectors_cache.to(device)
        return self._direction_vectors_cache

    def _build_anchor_context(
        self, tokens: torch.Tensor, batch_size: int, num_frames: int
    ) -> torch.Tensor:
        _, num_tokens, channels = tokens.shape
        tokens_4d = tokens.reshape(batch_size, num_frames, num_tokens, channels)

        if self.training:
            anchor_idx = torch.randint(0, num_frames, (batch_size,), device=tokens.device)
            context = (
                tokens_4d[torch.arange(batch_size, device=tokens.device), anchor_idx]
                .unsqueeze(1)
                .expand(batch_size, num_frames, num_tokens, channels)
            )
        else:
            mid_idx = num_frames // 2
            context = tokens_4d[:, mid_idx : mid_idx + 1].expand(
                batch_size, num_frames, num_tokens, channels
            )
        return context.reshape(batch_size * num_frames, num_tokens, channels)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Aggregator forward
        out = self.aggregator(images)
        if isinstance(out, (list, tuple)):
            tokens = out[0][-1] if isinstance(out[0], list) else out[0]
            patch_start_idx = out[1]
        else:
            tokens = out
            patch_start_idx = 0

        if tokens.dim() == 4:
            tokens = tokens.view(B * S, tokens.shape[2], tokens.shape[3])

        # RoPE position indices
        pos_2d = None
        if getattr(self.aggregator, "rope", None) is not None:
            pos_2d = self.aggregator.position_getter(B * S, patch_h, patch_w, tokens.device)
            pos_2d = pos_2d + 1
            pos_special = torch.zeros(
                B * S, patch_start_idx, 2, device=tokens.device, dtype=pos_2d.dtype
            )
            pos_2d = torch.cat([pos_special, pos_2d], dim=1)

        predictions = {}

        # --- Point branch ---
        if self.enable_point:
            pos_bs_point = self._get_branch_pos_embed(
                patch_h, patch_w, patch_start_idx,
                tokens.device, tokens.dtype, "point", B * S,
            )
            point_hidden = self.point_decoder(
                tokens, pos_embed=pos_bs_point, xpos=pos_2d,
            )
            with torch.amp.autocast(device_type="cuda", enabled=False):
                point_hidden = point_hidden.float()
                ret = self.point_head(
                    [point_hidden[:, patch_start_idx:]], (H, W)
                ).reshape(B, S, H, W, 3)

                log_d = ret[..., 2]
                d_pred = torch.exp(log_d)[..., None]

                directions = self._get_direction_vectors(H, W, d_pred.device, d_pred.dtype)
                local_points = directions.view(1, 1, H, W, 3) * d_pred

            predictions["local_points"] = local_points
            if self.enable_depth:
                predictions["depth"] = d_pred

        # --- Camera branch ---
        if self.enable_camera:
            pos_bs_cam = self._get_branch_pos_embed(
                patch_h, patch_w, patch_start_idx,
                tokens.device, tokens.dtype, "camera", B * S,
            )
            camera_hidden = self.camera_decoder(
                tokens, pos_embed=pos_bs_cam, xpos=pos_2d,
            )
            with torch.amp.autocast(device_type="cuda", enabled=False):
                camera_hidden = camera_hidden.float()
                camera_poses = self.camera_head(
                    camera_hidden[:, patch_start_idx:], patch_h, patch_w,
                ).reshape(B, S, 4, 4)
            predictions["camera_poses"] = camera_poses

            if "local_points" in predictions:
                P_homo = _homogenize_points(predictions["local_points"])
                world = torch.einsum(
                    "bsij, bshwj -> bshwi", camera_poses, P_homo
                )[..., :3]
                predictions["world_points"] = world
                predictions["points"] = world

        context = None
        if self.enable_global_points or self.enable_3dgs:
            context = self._build_anchor_context(tokens, B, S)

        # --- Global points branch ---
        if self.enable_global_points:
            pos_bs_global = self._get_branch_pos_embed(
                patch_h, patch_w, patch_start_idx,
                tokens.device, tokens.dtype, "global", B * S,
            )
            global_point_hidden = self.global_points_decoder(
                tokens, context, pos_embed=pos_bs_global, xpos=pos_2d, ypos=pos_2d,
            )
            with torch.amp.autocast(device_type="cuda", enabled=False):
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head(
                    [global_point_hidden[:, patch_start_idx:]], (H, W)
                ).reshape(B, S, H, W, 3)
            predictions["global_points"] = global_points
        else:
            predictions["global_points"] = None

        # --- 3D Gaussian Splatting branch ---
        if self.enable_3dgs:
            pos_bs_gaussian = self._get_branch_pos_embed(
                patch_h,
                patch_w,
                patch_start_idx,
                tokens.device,
                tokens.dtype,
                "gaussian",
                B * S,
            )
            gaussian_hidden = self.gaussian_decoder(
                tokens,
                context,
                pos_embed=pos_bs_gaussian,
                xpos=pos_2d,
                ypos=pos_2d,
            )
            with torch.amp.autocast(device_type="cuda", enabled=False):
                gaussian_hidden = gaussian_hidden.float()
                gaussian_patch_tokens = gaussian_hidden[:, patch_start_idx:]
                stage1_outputs = self.gaussian_stage1_head(
                    gaussian_patch_tokens,
                    batch_size=B,
                    num_frames=S,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    enabled=self.gaussian_stage1_enabled,
                )
                param_outputs = self.gaussian_param_head(
                    gaussian_patch_tokens,
                    batch_size=B,
                    num_frames=S,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
                gaussian_outputs = assemble_gaussian_outputs(
                    stage1_outputs,
                    param_outputs,
                    keep_threshold=(
                        self.gaussian_keep_threshold if not self.training else None
                    ),
                    use_stage1=self.gaussian_stage1_enabled,
                )

            predictions.update(stage1_outputs)
            predictions.update(param_outputs)
            predictions.update(gaussian_outputs)

        if not self.training:
            predictions["images"] = images

        return predictions
