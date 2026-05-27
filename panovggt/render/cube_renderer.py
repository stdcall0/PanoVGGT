"""
Cubemap Gaussian renderer.

For each input view, we rasterize all Gaussians onto 6 perspective faces
covering 360 degrees with a configurable wider-than-90 FoV (default 95°)
to suppress seam artifacts. The 6 face renders are stitched back to ERP
via cube_to_equi.Cube2Equirec for photometric loss in pano space.

This renderer is *differentiable wrt all GS parameters and centers*.
Camera poses are stop-gradient'ed by default (controlled at the caller).
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .cube_to_equi import Cube2Equirec, opencv_face_rotations


def _make_intrinsic(face_res: int, fov_rad: float, device, dtype) -> torch.Tensor:
    f = 0.5 * face_res / math.tan(0.5 * fov_rad)
    cx = cy = (face_res - 1) / 2.0
    K = torch.zeros(3, 3, device=device, dtype=dtype)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K


class CubePanoRenderer(nn.Module):
    """
    Cubemap-based equirectangular renderer using gsplat.

    Args:
        equ_h: ERP height for output rendering.
        face_res: per-face resolution.
        fov_deg: per-face field of view in degrees (default 95° to suppress seams).
        boundary_px: pixels to mask at each face border (in face space) when
                     building the validity mask in ERP space.
        sh_degree: SH degree to pass to gsplat. None falls back to N-D features.
        bg_color: optional (3,) background color, defaults to mid-gray.
    """

    def __init__(
        self,
        equ_h: int,
        face_res: int = 256,
        fov_deg: float = 95.0,
        boundary_px: int = 4,
        sh_degree: int = 1,
        bg_color: Optional[float] = 0.0,
    ):
        super().__init__()
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.face_res = face_res
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.boundary_px = boundary_px
        self.sh_degree = sh_degree
        self.bg_color = bg_color

        self.cube2equi = Cube2Equirec(
            face_w=face_res, equ_h=equ_h, equ_w=self.equ_w, fov_deg=fov_deg
        )
        self.register_buffer("face_R", opencv_face_rotations())  # (6, 3, 3)
        self._rasterization = None

    def _load_rasterization(self):
        if self._rasterization is None:
            try:
                from gsplat import rasterization
            except Exception as exc:
                raise ImportError(
                    "CubePanoRenderer requires the optional `gsplat` package. "
                    "Install it with `pip install gsplat`, or disable GS loss / "
                    "choose a renderer that is available in this environment."
                ) from exc
            self._rasterization = rasterization
        return self._rasterization

    def reset(self, equ_h: int, face_res: int):
        """Rebuild the cube↔equi grid (used when stage changes face_res)."""
        if equ_h == self.equ_h and face_res == self.face_res:
            return
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.face_res = face_res
        self.cube2equi = Cube2Equirec(
            face_w=face_res, equ_h=equ_h, equ_w=self.equ_w, fov_deg=self.fov_deg
        ).to(self.face_R.device)

    def _build_face_viewmats(self, w2c: torch.Tensor) -> torch.Tensor:
        """Compose face_w2c = face_R^T @ w2c for all 6 faces.

        Args:
            w2c: (4, 4) camera-from-world for one frame.
        Returns:
            (6, 4, 4) face cameras.
        """
        device = w2c.device
        dtype = w2c.dtype
        Rs = self.face_R.to(device=device, dtype=dtype)             # (6, 3, 3)
        # face_w2c rotation = face_R^T @ R_w2c
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        face_R_w2c = torch.einsum("fji,jk->fik", Rs, R_w2c)         # (6, 3, 3)
        face_t = torch.einsum("fji,j->fi", Rs, t_w2c)               # (6, 3)
        face_w2c = torch.zeros(6, 4, 4, device=device, dtype=dtype)
        face_w2c[:, :3, :3] = face_R_w2c
        face_w2c[:, :3, 3] = face_t
        face_w2c[:, 3, 3] = 1.0
        return face_w2c

    def _boundary_mask(self) -> torch.Tensor:
        m = torch.ones(1, 1, 6, self.face_res, self.face_res, dtype=torch.float32)
        bp = self.boundary_px
        if bp > 0:
            m[..., :bp, :] = 0.0
            m[..., -bp:, :] = 0.0
            m[..., :, :bp] = 0.0
            m[..., :, -bp:] = 0.0
        return m

    def _face_ray_norm(self, device, dtype) -> torch.Tensor:
        """Per-face factor converting perspective z-depth to radial depth."""
        f = 0.5 * self.face_res / math.tan(0.5 * self.fov_rad)
        c = (self.face_res - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(self.face_res, device=device, dtype=dtype),
            torch.arange(self.face_res, device=device, dtype=dtype),
            indexing="ij",
        )
        x = (xs - c) / f
        y = (ys - c) / f
        return torch.sqrt(x.square() + y.square() + 1.0).view(
            1, self.face_res, self.face_res, 1
        )

    def render(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors_sh: torch.Tensor,
        w2c_per_view: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            means:       (N, 3)
            quats:       (N, 4)  w-x-y-z
            scales:      (N, 3)
            opacities:   (N,)
            colors_sh:   (N, K, 3)  K = (sh_degree+1)**2
            w2c_per_view:(V, 4, 4)
        Returns:
            dict:
                rgb_erp:   (V, 3, equ_h, equ_w)
                depth_erp: (V, 1, equ_h, equ_w)
                alpha_erp: (V, 1, equ_h, equ_w)
                mask_erp:  (V, 1, equ_h, equ_w)  (boundary + cube valid)
        """
        device = means.device
        dtype = means.dtype
        V = w2c_per_view.shape[0]
        K = _make_intrinsic(self.face_res, self.fov_rad, device, dtype)  # (3,3)
        Ks = K.unsqueeze(0).expand(6 * V, 3, 3).contiguous()             # (6V,3,3)

        # Build per-face viewmats for every input view.
        face_w2c_all = []
        for v in range(V):
            face_w2c_all.append(self._build_face_viewmats(w2c_per_view[v]))
        face_w2c_all = torch.cat(face_w2c_all, dim=0)                    # (6V,4,4)

        bg = (
            None
            if self.bg_color is None
            else torch.full(
                (6 * V, 4), float(self.bg_color), device=device, dtype=dtype
            )
        )

        rasterization = self._load_rasterization()
        rgb, alpha, _info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_sh,
            viewmats=face_w2c_all,
            Ks=Ks,
            width=self.face_res,
            height=self.face_res,
            sh_degree=self.sh_degree,
            render_mode="RGB+D",
            packed=True,
        )
        # rgb is (6V, H, W, 4)  (RGB+D)
        rgb_d = rgb
        rgb_only = rgb_d[..., :3]              # (6V, H, W, 3)
        depth = rgb_d[..., 3:4]                # (6V, H, W, 1)
        depth = depth * self._face_ray_norm(device=device, dtype=dtype)
        alpha = alpha                          # (6V, H, W, 1)

        # reshape to (V, C, 6, face, face) for cube2equi
        def _to_cube(x):
            x = x.permute(0, 3, 1, 2).contiguous()  # (6V, C, H, W)
            x = x.view(V, 6, x.shape[1], self.face_res, self.face_res)
            return x.permute(0, 2, 1, 3, 4).contiguous()  # (V, C, 6, h, w)

        rgb_cube = _to_cube(rgb_only)
        depth_cube = _to_cube(depth)
        alpha_cube = _to_cube(alpha)

        boundary = self._boundary_mask().to(device=device, dtype=dtype)     # (1,1,6,h,w)
        mask_cube = boundary.expand(V, 1, 6, self.face_res, self.face_res)  # (V,1,6,h,w)

        rgb_erp = self.cube2equi(rgb_cube)
        depth_erp = self.cube2equi(depth_cube)
        alpha_erp = self.cube2equi(alpha_cube)
        mask_erp = self.cube2equi(mask_cube) * self.cube2equi.get_valid_mask(V).to(device=device, dtype=dtype)

        return dict(
            rgb_erp=rgb_erp,
            depth_erp=depth_erp,
            alpha_erp=alpha_erp,
            mask_erp=mask_erp,
        )
