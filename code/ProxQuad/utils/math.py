"""Batched rotation helpers. Quaternions are (w, x, y, z), MuJoCo's convention."""

import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) unit quaternion -> (..., 3, 3) rotation matrix (body -> world)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def vee(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) skew-symmetric matrix -> (..., 3) vector."""
    return torch.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], dim=-1)


def normalize(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def normalize_with_rate(v: torch.Tensor, v_dot: torch.Tensor,
                        eps: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor]:
    """u = v / |v| and its time derivative u_dot = (v_dot - u (u . v_dot)) / |v|."""
    n = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    u = v / n
    u_dot = (v_dot - u * (u * v_dot).sum(-1, keepdim=True)) / n
    return u, u_dot


def uniform(rng: tuple[float, float], shape, device) -> torch.Tensor:
    """Uniform sample in [low, high)."""
    lo, hi = rng
    return lo + (hi - lo) * torch.rand(shape, device=device)
