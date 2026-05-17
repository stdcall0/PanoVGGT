"""
GS-branch orchestrator.

Given:
    - PanoVGGT predictions (world_points, camera_poses, depth)
    - GS head outputs (offset, scale, rotation, opacity, sh_dc, sh_rest)
    - GT images (and optional GT depth) at ERP resolution

This module:
    1. Builds patch-level Gaussian centers from world_points (avg-pool).
    2. Constructs final GS parameter tensors (with stage gating).
    3. Calls the configured renderer (cube or odgs) to render at all S
       input-frame poses.
    4. Returns (rendered_rgb, rendered_depth, render_mask, gs_dict).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from panovggt.utils.geometry import se3_inverse
from .cube_renderer import CubePanoRenderer
from .gs_utils import patch_pool, depth_footprint_scale, knn_scale, softplus_inv


class GSBranch:
    """Stateless helper bound to a stage config. Renders and returns tensors."""

    def __init__(
        self,
        renderer: str = "cube",
        equ_h: int = 518,
        face_res: int = 256,
        fov_deg: float = 95.0,
        boundary_px: int = 4,
        sh_degree: int = 1,
        scale_init_mode: str = "depth_footprint",   # or "knn" or "constant"
        scale_init_value: float = 0.01,
        use_offset: bool = False,
        detach_centers: bool = True,
        detach_camera: bool = True,
        train_dc: bool = True,
        train_opacity: bool = True,
        train_scale: bool = False,
        train_rotation: bool = False,
        train_sh_rest: bool = False,
        gs_head=None,                                # used to gate trainables
    ):
        self.renderer_name = renderer
        self.sh_degree = sh_degree
        self.scale_init_mode = scale_init_mode
        self.scale_init_value = scale_init_value
        self.use_offset = use_offset
        self.detach_centers = detach_centers
        self.detach_camera = detach_camera
        self.gs_head = gs_head

        self.train_flags = dict(
            sh_dc=train_dc,
            opacity=train_opacity,
            scale=train_scale,
            rotation=train_rotation,
            sh_rest=train_sh_rest,
            offset=use_offset,
        )

        if renderer == "cube":
            self.renderer = CubePanoRenderer(
                equ_h=equ_h,
                face_res=face_res,
                fov_deg=fov_deg,
                boundary_px=boundary_px,
                sh_degree=sh_degree,
                bg_color=0.0,
            )
        elif renderer == "odgs":
            from .odgs_renderer import ODGSPanoRenderer
            self.renderer = ODGSPanoRenderer(equ_h=equ_h, sh_degree=sh_degree)
        else:
            raise ValueError(f"unknown renderer '{renderer}'")

    # ---------------------------------------------------------------------
    def _gate(self, raw: torch.Tensor, init: torch.Tensor, name: str):
        """If train_flag is False, replace gradients with the initializer."""
        if self.train_flags.get(name, True):
            return raw
        return init.detach()

    # ---------------------------------------------------------------------
    def render(
        self,
        gs_params: Dict[str, torch.Tensor],   # output of LinearGaussianHead
        world_points: torch.Tensor,           # (B,S,H,W,3) — predicted
        camera_poses_c2w: torch.Tensor,       # (B,S,4,4)
        images: torch.Tensor,                 # (B,S,3,H,W) GT
        depth: Optional[torch.Tensor] = None, # (B,S,H,W,1) predicted
    ):
        B, S, H, W, _ = world_points.shape
        patch_size = H // gs_params["sh_dc"].shape[2]
        Hp = gs_params["sh_dc"].shape[2]
        Wp = gs_params["sh_dc"].shape[3]
        assert patch_size * Hp == H and patch_size * Wp == W, (patch_size, Hp, Wp, H, W)

        # ---- centers ------------------------------------------------------
        centers_pp = patch_pool(world_points, patch_size)  # (B,S,Hp,Wp,3)
        if self.detach_centers:
            centers_pp = centers_pp.detach()

        if self.use_offset and self.train_flags["offset"]:
            centers = centers_pp + gs_params["offset"]
        else:
            centers = centers_pp

        # ---- color init ---------------------------------------------------
        # patch-pool the GT image to get a per-Gaussian DC bootstrap.
        img_pp = patch_pool(images, patch_size)              # (B,S,3,Hp,Wp)
        img_pp = img_pp.permute(0, 1, 3, 4, 2).contiguous()  # (B,S,Hp,Wp,3)
        # gsplat with sh_degree expects DC stored as (rgb-0.5)/C0; we use the
        # PixelSplat convention where the coefficient lives at K=0.
        C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
        dc_init = (img_pp - 0.5) / C0
        sh_dc = self._gate(gs_params["sh_dc"] + dc_init.detach(), dc_init, "sh_dc")
        # For sh_rest, init is zeros which is exactly the gate's default
        sh_rest = self._gate(
            gs_params["sh_rest"], torch.zeros_like(gs_params["sh_rest"]), "sh_rest"
        )

        # ---- scale init ---------------------------------------------------
        if self.scale_init_mode == "depth_footprint" and depth is not None:
            d = depth if depth.dim() == 4 else depth.squeeze(-1)
            scale_init = depth_footprint_scale(d, patch_size, H)        # (B,S,Hp,Wp,3)
        elif self.scale_init_mode == "knn":
            scale_init = knn_scale(centers_pp.detach(), k=3)
        else:
            scale_init = torch.full_like(gs_params["scale"], float(self.scale_init_value))
        # For training scale we pass the raw network output (already softplus'd).
        # For "frozen" scale stages, replace by scale_init.
        if self.train_flags["scale"]:
            scale_final = gs_params["scale"]
        else:
            scale_final = scale_init.detach()

        # rotation
        if self.train_flags["rotation"]:
            rotation_final = gs_params["rotation"]
        else:
            rotation_final = gs_params["rotation"].new_zeros(gs_params["rotation"].shape)
            rotation_final[..., 0] = 1.0

        # opacity
        opacity_final = gs_params["opacity"]
        if not self.train_flags["opacity"]:
            opacity_final = opacity_final.detach()

        # ---- pack SH ------------------------------------------------------
        # gsplat wants colors of shape (N, K, 3) where K = (sh_degree+1)**2.
        K = (self.sh_degree + 1) ** 2
        sh_rest_chan = sh_rest.shape[-1]
        if K - 1 != sh_rest_chan:
            # head was built with a different sh_degree; pad / trim to match
            if K - 1 > sh_rest_chan:
                pad = sh_rest.new_zeros(*sh_rest.shape[:-1], K - 1 - sh_rest_chan)
                sh_rest = torch.cat([sh_rest, pad], dim=-1)
            else:
                sh_rest = sh_rest[..., : K - 1]
        # combine to (B,S,Hp,Wp,K,3)
        sh_dc_kx = sh_dc.unsqueeze(-2)                             # (...,1,3)
        sh_rest_kx = sh_rest.permute(0, 1, 2, 3, 5, 4).contiguous()  # (...,K-1,3)
        colors_sh = torch.cat([sh_dc_kx, sh_rest_kx], dim=-2)      # (...,K,3)

        # ---- flatten to (N, *) for gsplat ---------------------------------
        means = centers.view(-1, 3)
        scales = scale_final.view(-1, 3)
        quats = rotation_final.view(-1, 4)
        opacities = opacity_final.view(-1)
        colors = colors_sh.view(-1, K, 3)

        # ---- viewmats: w2c per input view --------------------------------
        c2w = camera_poses_c2w
        if self.detach_camera:
            c2w = c2w.detach()
        w2c = se3_inverse(c2w)                                     # (B,S,4,4)
        # render per batch element separately to avoid cross-batch leakage
        out_rgb, out_depth, out_alpha, out_mask = [], [], [], []
        for b in range(B):
            n_b = means.shape[0] // B
            sl = slice(b * n_b, (b + 1) * n_b)
            r = self.renderer.render(
                means=means[sl],
                quats=quats[sl],
                scales=scales[sl],
                opacities=opacities[sl],
                colors_sh=colors[sl],
                w2c_per_view=w2c[b],                               # (S,4,4)
            )
            out_rgb.append(r["rgb_erp"])
            out_depth.append(r["depth_erp"])
            out_alpha.append(r["alpha_erp"])
            out_mask.append(r["mask_erp"])
        return dict(
            rgb_erp=torch.stack(out_rgb, dim=0),       # (B,S,3,H,W)
            depth_erp=torch.stack(out_depth, dim=0),
            alpha_erp=torch.stack(out_alpha, dim=0),
            mask_erp=torch.stack(out_mask, dim=0),
            centers=centers,
            scales=scale_final,
            rotations=rotation_final,
            opacities=opacity_final,
            colors_sh=colors_sh,
        )
