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

from panovggt.utils.geometry import se3_inverse
from .gs_geometry import normalize_quaternion, quat_multiply, tangent_frame_quaternions
from .gs_utils import (
    patch_pool,
    patch_valid_ratio,
    depth_footprint_scale,
    knn_scale,
    subpatch_depth_footprint_scale,
    subpatch_pool,
    subpatch_valid_ratio,
)


def _smooth_bounded_scale_multiplier(
    raw_scale: torch.Tensor,
    scale_init_value: float,
    scale_mult_min: Optional[float],
    scale_mult_max: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert positive head scale output into a smooth bounded multiplier."""
    scale_ref = max(float(scale_init_value), 1e-6)
    scale_log_raw = torch.log(raw_scale.clamp_min(1e-8) / scale_ref)
    log_mult = scale_log_raw
    if scale_mult_min is not None:
        min_log = abs(torch.log(raw_scale.new_tensor(float(scale_mult_min))).item())
        log_mult = torch.where(
            scale_log_raw < 0,
            -min_log * torch.tanh(-scale_log_raw),
            log_mult,
        )
    if scale_mult_max is not None:
        max_log = torch.log(raw_scale.new_tensor(float(scale_mult_max))).item()
        log_mult = torch.where(
            scale_log_raw >= 0,
            max_log * torch.tanh(scale_log_raw),
            log_mult,
        )
    return torch.exp(log_mult), scale_log_raw


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
        scale_init_factor: float = 1.0,
        scale_mult_min: Optional[float] = None,
        scale_mult_max: Optional[float] = None,
        offset_max_ratio: float = 0.5,
        use_offset: bool = False,
        detach_centers: bool = True,
        detach_camera: bool = True,
        train_dc: bool = True,
        train_opacity: bool = True,
        train_scale: bool = False,
        train_rotation: bool = False,
        train_sh_rest: bool = False,
        rotation_init_mode: str = "identity",
        min_valid_ratio: float = 0.25,
        gs_head=None,                                # used to gate trainables
    ):
        self.renderer_name = renderer
        self.sh_degree = sh_degree
        self.scale_init_mode = scale_init_mode
        self.scale_init_value = scale_init_value
        self.scale_init_factor = float(scale_init_factor)
        self.scale_mult_min = scale_mult_min
        self.scale_mult_max = scale_mult_max
        self.offset_max_ratio = float(offset_max_ratio)
        self.use_offset = use_offset
        self.detach_centers = detach_centers
        self.detach_camera = detach_camera
        self.rotation_init_mode = str(rotation_init_mode)
        self.min_valid_ratio = float(min_valid_ratio)
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
            from .cube_renderer import CubePanoRenderer
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
    def materialize(
        self,
        gs_params: Dict[str, torch.Tensor],   # output of LinearGaussianHead
        world_points: torch.Tensor,           # (B,S,H,W,3) — predicted
        images: torch.Tensor,                 # (B,S,3,H,W) GT
        depth: Optional[torch.Tensor] = None, # (B,S,H,W,1) predicted
        point_masks: Optional[torch.Tensor] = None, # (B,S,H,W) valid geometry mask
    ) -> Dict[str, torch.Tensor]:
        B, S, H, W, _ = world_points.shape
        patch_size = H // gs_params["sh_dc"].shape[2]
        Hp = gs_params["sh_dc"].shape[2]
        Wp = gs_params["sh_dc"].shape[3]
        has_subgrid = gs_params["sh_dc"].dim() == 6
        Q = gs_params["sh_dc"].shape[4] if has_subgrid else 1
        subgrid_size = int(round(Q ** 0.5))
        if subgrid_size * subgrid_size != Q:
            raise ValueError(f"subgrid Gaussian count must be square, got Q={Q}.")
        assert patch_size * Hp == H and patch_size * Wp == W, (patch_size, Hp, Wp, H, W)

        def with_q(x: torch.Tensor) -> torch.Tensor:
            return x if has_subgrid else x.unsqueeze(4)

        def maybe_squeeze_q(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None or has_subgrid:
                return x
            return x.squeeze(4)

        gs_offset = with_q(gs_params["offset"])
        gs_scale = with_q(gs_params["scale"])
        gs_rotation = with_q(gs_params["rotation"])
        gs_opacity = with_q(gs_params["opacity"])
        gs_sh_dc = with_q(gs_params["sh_dc"])
        gs_sh_rest = gs_params["sh_rest"] if has_subgrid else gs_params["sh_rest"].unsqueeze(4)

        # ---- centers ------------------------------------------------------
        patch_valid = None
        if point_masks is not None:
            if has_subgrid:
                patch_valid = subpatch_valid_ratio(
                    point_masks, patch_size, subgrid_size
                ).unsqueeze(-1)
            else:
                patch_valid = patch_valid_ratio(point_masks, patch_size).unsqueeze(-1).unsqueeze(4)

        if has_subgrid:
            centers_pp = subpatch_pool(world_points, patch_size, subgrid_size)
        else:
            centers_pp = patch_pool(world_points, patch_size).unsqueeze(4)
        if self.detach_centers:
            centers_pp = centers_pp.detach()

        # ---- color init ---------------------------------------------------
        # patch-pool the GT image to get a per-Gaussian DC bootstrap.
        if has_subgrid:
            img_pp = subpatch_pool(images, patch_size, subgrid_size)
        else:
            img_pp = patch_pool(images, patch_size)
            img_pp = img_pp.permute(0, 1, 3, 4, 2).contiguous().unsqueeze(4)
        # gsplat with sh_degree expects DC stored as (rgb-0.5)/C0; we use the
        # PixelSplat convention where the coefficient lives at K=0.
        C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
        dc_init = (img_pp - 0.5) / C0
        sh_dc = self._gate(gs_sh_dc + dc_init.detach(), dc_init, "sh_dc")
        # For sh_rest, init is zeros which is exactly the gate's default
        sh_rest = self._gate(
            gs_sh_rest, torch.zeros_like(gs_sh_rest), "sh_rest"
        )

        # ---- scale init ---------------------------------------------------
        if self.scale_init_mode == "depth_footprint" and depth is not None:
            d = depth if depth.dim() == 4 else depth.squeeze(-1)
            if has_subgrid:
                scale_init = subpatch_depth_footprint_scale(
                    d, patch_size, subgrid_size, H, valid_mask=point_masks
                )
            else:
                scale_init = depth_footprint_scale(
                    d, patch_size, H, valid_mask=point_masks
                ).unsqueeze(4)
        elif self.scale_init_mode == "knn":
            scale_init = knn_scale(centers_pp.detach(), k=3)
        else:
            scale_init = torch.full_like(gs_scale, float(self.scale_init_value))
        scale_init = scale_init * self.scale_init_factor
        # Keep the configured scale init as the geometric prior. The head's
        # softplus output is initialized to scale_init_value, so this starts at
        # scale_init and learns a positive multiplicative correction.
        if self.train_flags["scale"]:
            scale_mult, scale_log_raw = _smooth_bounded_scale_multiplier(
                raw_scale=gs_scale,
                scale_init_value=self.scale_init_value,
                scale_mult_min=self.scale_mult_min,
                scale_mult_max=self.scale_mult_max,
            )
            scale_final = scale_init.detach() * scale_mult
        else:
            scale_log_raw = torch.zeros_like(scale_init)
            scale_mult = torch.ones_like(scale_init)
            scale_final = scale_init.detach()

        # ---- offset -------------------------------------------------------
        # Interpret the head output as a local offset ratio rather than an
        # absolute world-space displacement. Binding it to scale_init prevents
        # offset freedom from silently growing when learned scales grow.
        if self.use_offset and self.train_flags["offset"]:
            offset_ratio = torch.tanh(gs_offset) * self.offset_max_ratio
            offset = offset_ratio * scale_init.detach()
            centers = centers_pp + offset
        else:
            offset_ratio = torch.zeros_like(centers_pp)
            offset = torch.zeros_like(centers_pp)
            centers = centers_pp

        # rotation
        if self.rotation_init_mode == "tangent":
            rotation_init = tangent_frame_quaternions(centers_pp.detach())
        elif self.rotation_init_mode == "identity":
            rotation_init = gs_rotation.new_zeros(gs_rotation.shape)
            rotation_init[..., 0] = 1.0
        else:
            raise ValueError(
                f"unknown rotation_init_mode '{self.rotation_init_mode}', "
                "expected 'identity' or 'tangent'."
            )
        if self.train_flags["rotation"]:
            rotation_final = quat_multiply(rotation_init, gs_rotation)
        else:
            rotation_final = normalize_quaternion(rotation_init)

        # opacity
        opacity_final = gs_opacity
        if not self.train_flags["opacity"]:
            opacity_final = opacity_final.detach()
        if patch_valid is not None:
            gaussian_valid = (patch_valid >= self.min_valid_ratio).to(opacity_final.dtype)
            opacity_final = opacity_final * gaussian_valid

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
        # combine to (B,S,Hp,Wp,Q,K,3)
        sh_dc_kx = sh_dc.unsqueeze(-2)                             # (...,1,3)
        sh_rest_kx = sh_rest.permute(0, 1, 2, 3, 4, 6, 5).contiguous()  # (...,K-1,3)
        colors_sh = torch.cat([sh_dc_kx, sh_rest_kx], dim=-2)      # (...,K,3)

        return dict(
            centers=maybe_squeeze_q(centers),
            offset=maybe_squeeze_q(offset),
            offset_ratio=maybe_squeeze_q(offset_ratio),
            scales=maybe_squeeze_q(scale_final),
            scale_init=maybe_squeeze_q(scale_init),
            scale_mult=maybe_squeeze_q(scale_mult),
            scale_log_raw=maybe_squeeze_q(scale_log_raw),
            scale_mult_clamped=maybe_squeeze_q(scale_mult),
            rotations=maybe_squeeze_q(rotation_final),
            opacities=maybe_squeeze_q(opacity_final),
            patch_valid=maybe_squeeze_q(patch_valid),
            sh_dc=maybe_squeeze_q(sh_dc),
            sh_rest=maybe_squeeze_q(sh_rest),
            colors_sh=maybe_squeeze_q(colors_sh),
            patch_size=torch.as_tensor(patch_size, device=world_points.device),
            subgrid_size=torch.as_tensor(subgrid_size, device=world_points.device),
        )

    # ---------------------------------------------------------------------
    def render(
        self,
        gs_params: Dict[str, torch.Tensor],   # output of LinearGaussianHead
        world_points: torch.Tensor,           # (B,S,H,W,3) — predicted
        camera_poses_c2w: torch.Tensor,       # (B,S,4,4)
        images: torch.Tensor,                 # (B,S,3,H,W) GT
        depth: Optional[torch.Tensor] = None, # (B,S,H,W,1) predicted
        point_masks: Optional[torch.Tensor] = None, # (B,S,H,W) valid geometry mask
    ):
        B = world_points.shape[0]
        materialized = self.materialize(
            gs_params=gs_params,
            world_points=world_points,
            images=images,
            depth=depth,
            point_masks=point_masks,
        )
        centers = materialized["centers"]
        scale_final = materialized["scales"]
        scale_init = materialized["scale_init"]
        scale_mult = materialized["scale_mult_clamped"]
        offset = materialized["offset"]
        rotation_final = materialized["rotations"]
        opacity_final = materialized["opacities"]
        colors_sh = materialized["colors_sh"]

        # ---- flatten to (N, *) for gsplat ---------------------------------
        means = centers.view(-1, 3)
        scales = scale_final.view(-1, 3)
        quats = rotation_final.view(-1, 4)
        opacities = opacity_final.view(-1)
        colors = colors_sh.view(-1, colors_sh.shape[-2], 3)

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
            offset=offset,
            offset_ratio=materialized.get("offset_ratio"),
            scales=scale_final,
            scale_init=scale_init,
            scale_mult=scale_mult,
            scale_log_raw=materialized.get("scale_log_raw"),
            scale_mult_clamped=scale_mult,
            rotations=rotation_final,
            opacities=opacity_final,
            colors_sh=colors_sh,
            patch_valid=materialized.get("patch_valid"),
        )


def materialize_gaussians(
    gs_params: Dict[str, torch.Tensor],
    world_points: torch.Tensor,
    images: torch.Tensor,
    depth: Optional[torch.Tensor],
    gs_conf: Dict,
    sh_degree: int,
    point_masks: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Build the exact Gaussian tensors used by GSBranch.render for export/tests."""
    branch = object.__new__(GSBranch)
    branch.renderer_name = str(gs_conf.get("renderer", "cube"))
    branch.sh_degree = int(sh_degree)
    branch.scale_init_mode = str(gs_conf.get("scale_init_mode", "depth_footprint"))
    branch.scale_init_value = float(gs_conf.get("scale_init_value", 0.01))
    branch.scale_init_factor = float(gs_conf.get("scale_init_factor", 1.0))
    branch.scale_mult_min = gs_conf.get("scale_mult_min", None)
    branch.scale_mult_max = gs_conf.get("scale_mult_max", None)
    branch.offset_max_ratio = float(gs_conf.get("offset_max_ratio", 0.5))
    branch.use_offset = bool(gs_conf.get("use_offset", False))
    branch.detach_centers = bool(gs_conf.get("detach_centers", True))
    branch.detach_camera = bool(gs_conf.get("detach_camera", True))
    branch.rotation_init_mode = str(gs_conf.get("rotation_init_mode", "identity"))
    branch.gs_head = None
    branch.train_flags = dict(
        sh_dc=bool(gs_conf.get("train_dc", True)),
        opacity=bool(gs_conf.get("train_opacity", True)),
        scale=bool(gs_conf.get("train_scale", False)),
        rotation=bool(gs_conf.get("train_rotation", False)),
        sh_rest=bool(gs_conf.get("train_sh_rest", False)),
        offset=bool(gs_conf.get("use_offset", False)),
    )
    branch.min_valid_ratio = float(gs_conf.get("min_valid_ratio", 0.25))
    return branch.materialize(
        gs_params=gs_params,
        world_points=world_points,
        images=images,
        depth=depth,
        point_masks=point_masks,
    )
