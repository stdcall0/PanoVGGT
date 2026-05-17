"""
Optional native pano renderer using the ODGS CUDA rasterizer.

This module is a thin adapter around the rasterizer at
`references/ODGS/submodules/odgs-gaussian-rasterization/`. It is only
exercised when `gs.renderer == 'odgs'` in the config; if the extension is
not installed, instantiation raises `NotImplementedError` and the user
must fall back to `cube`.
"""

import torch
import torch.nn as nn


class ODGSPanoRenderer(nn.Module):
    def __init__(self, equ_h: int, sh_degree: int = 1):
        super().__init__()
        try:
            from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "ODGS rasterizer not built. Install via `pip install -e "
                "references/ODGS/submodules/odgs-gaussian-rasterization` or "
                "set gs.renderer='cube'."
            ) from e
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.sh_degree = sh_degree

    def render(self, *args, **kwargs):
        # Hook left intentionally minimal; full implementation tracked as
        # follow-up work. The cube path is canonical for v1.
        raise NotImplementedError("ODGS render path is a follow-up; use 'cube' for now.")
