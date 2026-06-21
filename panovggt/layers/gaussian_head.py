"""
LinearGaussianHead — patch-token Gaussian-parameter head for PanoVGGT.

Each ViT patch token (1024-D) is projected by a *separate* nn.Linear per
parameter group: offset / scale / rotation / opacity / sh_dc / sh_rest.
This makes per-stage freezing trivial (regex match on `gaussian_head.<name>`
in `optim.frozen_module_names`).

Activation conventions follow PixelSplat / Splatt3R but use softplus for
scale (more stable than exp at our patch-pooled depth scales).
"""

from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inv(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.expm1(y.clamp_min(eps)) + eps)


def _init_linear(layer: nn.Linear, weight_scale: float, bias: float, bias_vec=None):
    if weight_scale > 0:
        nn.init.uniform_(layer.weight, -weight_scale, weight_scale)
    else:
        nn.init.zeros_(layer.weight)
    if bias_vec is not None:
        with torch.no_grad():
            layer.bias.copy_(torch.as_tensor(bias_vec))
    else:
        nn.init.constant_(layer.bias, bias)


class LinearGaussianHead(nn.Module):
    """
    Patch-level Gaussian parameter head with per-group sub-modules.

    Each forward returns a dict of (B, S, Hp, Wp, *) tensors.
    """

    def __init__(
        self,
        dec_embed_dim: int = 1024,
        sh_degree: int = 1,
        scale_init: float = 0.01,
        opacity_init: float = 0.1,
    ):
        super().__init__()
        self.dec_embed_dim = dec_embed_dim
        self.sh_degree = int(sh_degree)
        self.sh_extra = (sh_degree + 1) ** 2 - 1 if sh_degree > 0 else 0
        self.scale_init = float(scale_init)
        self.opacity_init = float(opacity_init)

        self.offset = nn.Linear(dec_embed_dim, 3)
        self.scale = nn.Linear(dec_embed_dim, 3)
        self.rotation = nn.Linear(dec_embed_dim, 4)
        self.opacity = nn.Linear(dec_embed_dim, 1)
        self.sh_dc = nn.Linear(dec_embed_dim, 3)
        self.sh_rest = nn.Linear(dec_embed_dim, 3 * self.sh_extra) if self.sh_extra > 0 else None

        scale_bias = float(_softplus_inv(torch.tensor(self.scale_init)).item())
        opacity_bias = float(math.log(self.opacity_init / (1.0 - self.opacity_init)))

        _init_linear(self.offset, weight_scale=1e-4, bias=0.0)
        _init_linear(self.scale, weight_scale=1e-4, bias=scale_bias)
        _init_linear(self.rotation, weight_scale=1e-4, bias=0.0,
                     bias_vec=[1.0, 0.0, 0.0, 0.0])
        _init_linear(self.opacity, weight_scale=1e-4, bias=opacity_bias)
        _init_linear(self.sh_dc, weight_scale=1e-4, bias=0.0)
        if self.sh_rest is not None:
            _init_linear(self.sh_rest, weight_scale=0.0, bias=0.0)

    @property
    def out_dim_summary(self) -> Dict[str, int]:
        return dict(
            offset=3, scale=3, rotation=4, opacity=1,
            sh_dc=3, sh_rest=3 * self.sh_extra,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        Hp: int,
        Wp: int,
        B: int,
        S: int,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            tokens: (B*S, Hp*Wp, dec_embed_dim)
        Returns:
            dict of (B, S, Hp, Wp, *) tensors with raw activations applied.
        """
        BS, N, D = tokens.shape
        assert BS == B * S and N == Hp * Wp, (BS, N, B, S, Hp, Wp)

        def _r(x, last):
            return x.view(B, S, Hp, Wp, last)

        offset_raw = _r(self.offset(tokens), 3)
        scale_raw = _r(self.scale(tokens), 3)
        rot_raw = _r(self.rotation(tokens), 4)
        opacity_raw = _r(self.opacity(tokens), 1)
        dc_raw = _r(self.sh_dc(tokens), 3)
        if self.sh_rest is not None:
            rest_raw = _r(self.sh_rest(tokens), 3 * self.sh_extra)
            sh_rest = rest_raw.view(B, S, Hp, Wp, 3, self.sh_extra)
        else:
            sh_rest = dc_raw.new_zeros(B, S, Hp, Wp, 3, 0)

        scale = F.softplus(scale_raw) + 1e-6
        rotation = rot_raw / (rot_raw.norm(dim=-1, keepdim=True) + 1e-8)
        opacity = torch.sigmoid(opacity_raw)
        sh_dc = dc_raw

        return dict(
            offset=offset_raw,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
        )
