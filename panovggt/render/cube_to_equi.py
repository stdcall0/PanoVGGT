"""
Cube ↔ Equirectangular conversion using 3-D grid_sample.

We define our own canonical OpenCV-camera face rotations:
    0: front  (+z)
    1: right  (+x)
    2: back   (-z)
    3: left   (-x)
    4: up     (-y)
    5: down   (+y)

Camera convention: +x right, +y down, +z forward (standard OpenCV).
World convention is *the same as the camera* — PanoVGGT's
`_get_direction_vectors` uses dir_y = -sin(theta) so that +y points
"down" the sphere, matching OpenCV.

Faces are rendered at `fov_deg` (e.g. 95°) for seam-tolerant overlap.
The cube → ERP grid pre-computes a normalized sampling grid that maps
each ERP pixel to (u, v, face_id) coordinates inside the cube tensor.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def opencv_face_rotations() -> torch.Tensor:
    """Returns (6, 3, 3) world-from-camera rotation matrices in OpenCV.

    Each R maps a *camera-frame* unit vector (e.g. (0,0,1) for forward)
    to its corresponding *world-frame* direction.
    """

    def Ry(deg):
        a = math.radians(deg)
        return torch.tensor(
            [[math.cos(a), 0.0, math.sin(a)],
             [0.0, 1.0, 0.0],
             [-math.sin(a), 0.0, math.cos(a)]],
            dtype=torch.float32,
        )

    def Rx(deg):
        a = math.radians(deg)
        return torch.tensor(
            [[1.0, 0.0, 0.0],
             [0.0, math.cos(a), -math.sin(a)],
             [0.0, math.sin(a), math.cos(a)]],
            dtype=torch.float32,
        )

    return torch.stack([
        torch.eye(3),    # front
        Ry(+90),         # right
        Ry(180),         # back
        Ry(-90),         # left
        Rx(+90),         # up    (looks toward -y in world)
        Rx(-90),         # down  (looks toward +y in world)
    ], dim=0)


class Cube2Equirec(nn.Module):
    """Differentiable cube → ERP using 3-D grid_sample.

    Input:  cube tensor (B, C, 6, face_h, face_w) — face order matches
            `opencv_face_rotations()` above.
    Output: ERP tensor  (B, C, equ_h, equ_w).
    """

    def __init__(
        self,
        face_w: int,
        equ_h: int,
        equ_w: int = None,
        fov_deg: float = 90.0,
    ):
        super().__init__()
        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w if equ_w is not None else equ_h * 2
        self.fov = math.radians(fov_deg)
        self._build_grid()

    def _build_grid(self):
        H, W = self.equ_h, self.equ_w
        device = "cpu"
        # ERP pixel directions matching PanoVGGT's _get_direction_vectors
        u = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        phi = (u / W - 0.5) * 2 * math.pi      # (W,)
        theta = -(v / H - 0.5) * math.pi       # (H,)
        gphi, gtheta = torch.meshgrid(phi, theta, indexing="xy")  # (H, W)
        dir_z = torch.cos(gtheta) * torch.cos(gphi)
        dir_x = torch.cos(gtheta) * torch.sin(gphi)
        dir_y = -torch.sin(gtheta)
        d_world = torch.stack([dir_x, dir_y, dir_z], dim=-1)       # (H, W, 3)

        Rs = opencv_face_rotations()  # (6, 3, 3) world-from-cam
        # camera-frame direction = R^T @ world-frame direction
        d_face_all = torch.einsum("fji,hwj->fhwi", Rs, d_world)    # (6, H, W, 3)

        z = d_face_all[..., 2]
        # Identify the chosen face: face whose +z component is the largest
        # (within the cube's "principal axis" cone).
        face_id = z.argmax(dim=0)                                  # (H, W)

        x = d_face_all[..., 0].gather(0, face_id.unsqueeze(0)).squeeze(0)
        y = d_face_all[..., 1].gather(0, face_id.unsqueeze(0)).squeeze(0)
        z_sel = z.gather(0, face_id.unsqueeze(0)).squeeze(0).clamp_min(1e-6)
        u_norm = (x / z_sel) / math.tan(self.fov / 2)
        v_norm = (y / z_sel) / math.tan(self.fov / 2)

        # Build normalized z (face_id) for grid_sample's depth axis with
        # align_corners=True: face_id ∈ [0,5] -> z ∈ [-1, +1].
        face_norm = (face_id.float() / 2.5) - 1.0  # (H, W)

        valid = (
            (z_sel > 0)
            & (u_norm.abs() <= 1.0)
            & (v_norm.abs() <= 1.0)
        )

        grid = torch.stack([u_norm, v_norm, face_norm], dim=-1)    # (H, W, 3)
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0))  # (1,1,H,W,3)
        self.register_buffer("valid_mask", valid.float()[None, None])   # (1,1,H,W)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cube: (B, C, 6, face_h, face_w).
        Returns:
            erp: (B, C, equ_h, equ_w)
        """
        B, C, F_, fh, fw = cube.shape
        assert F_ == 6, f"expected 6 faces, got {F_}"
        # grid_sample 3-D expects (N, C, D_in, H_in, W_in) and grid (N, D_out, H_out, W_out, 3)
        # we map D_out=1, H_out=equ_h, W_out=equ_w.
        grid = self.grid.expand(B, 1, self.equ_h, self.equ_w, 3)
        out = F.grid_sample(
            cube, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return out.squeeze(2)  # drop D_out=1 → (B, C, equ_h, equ_w)

    def get_valid_mask(self, batch_size: int = 1) -> torch.Tensor:
        return self.valid_mask.expand(batch_size, 1, self.equ_h, self.equ_w).contiguous()
