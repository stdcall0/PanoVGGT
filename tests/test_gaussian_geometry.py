import torch

from panovggt.render.gs_geometry import (
    quat_multiply,
    tangent_frame_quaternions,
    tangent_frames_from_centers,
)
from panovggt.utils.rotation import quat_to_mat


def test_tangent_frames_are_orthonormal_and_right_handed():
    centers = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    frames = tangent_frames_from_centers(centers)
    should_be_eye = frames.transpose(-1, -2) @ frames

    torch.testing.assert_close(should_be_eye, torch.eye(3).expand(4, 3, 3), atol=1e-6, rtol=1e-6)
    assert torch.linalg.det(frames).min().item() > 0.999


def test_tangent_frame_quaternions_match_frame_matrices():
    centers = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    frames = tangent_frames_from_centers(centers)
    quats = tangent_frame_quaternions(centers)

    torch.testing.assert_close(quats.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(quat_to_mat(quats), frames, atol=1e-6, rtol=1e-6)


def test_quat_multiply_identity_residual_preserves_base():
    base = tangent_frame_quaternions(torch.tensor([[1.0, 0.0, 0.0]]))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    torch.testing.assert_close(quat_multiply(base, identity), base)
