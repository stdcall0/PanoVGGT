import torch
import torch.nn.functional as F

from panovggt.utils.rotation import mat_to_quat


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for WXYZ quaternions."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    out = torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )
    return normalize_quaternion(out)


def tangent_frames_from_centers(centers: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build right-handed tangent frames whose third axis follows the center ray."""
    normal_norm = centers.norm(dim=-1, keepdim=True)
    fallback_normal = centers.new_tensor([0.0, 0.0, 1.0])
    normal = torch.where(
        normal_norm > eps,
        centers / normal_norm.clamp_min(eps),
        fallback_normal.view(*([1] * (centers.dim() - 1)), 3),
    )

    up = centers.new_tensor([0.0, 1.0, 0.0]).view(*([1] * (centers.dim() - 1)), 3)
    alt = centers.new_tensor([1.0, 0.0, 0.0]).view(*([1] * (centers.dim() - 1)), 3)
    tangent = torch.cross(up.expand_as(normal), normal, dim=-1)
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    alt_tangent = torch.cross(alt.expand_as(normal), normal, dim=-1)
    tangent = torch.where(tangent_norm > eps, tangent, alt_tangent)
    tangent = F.normalize(tangent, dim=-1, eps=eps)
    bitangent = F.normalize(torch.cross(normal, tangent, dim=-1), dim=-1, eps=eps)
    normal = F.normalize(torch.cross(tangent, bitangent, dim=-1), dim=-1, eps=eps)
    return torch.stack([tangent, bitangent, normal], dim=-1)


def tangent_frame_quaternions(centers: torch.Tensor) -> torch.Tensor:
    frames = tangent_frames_from_centers(centers)
    return normalize_quaternion(mat_to_quat(frames))
