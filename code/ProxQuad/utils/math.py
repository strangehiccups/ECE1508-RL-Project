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


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b, (..., 4) wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vector [rad] -> (..., 4) unit quaternion wxyz."""
    angle = v.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    k = torch.where(angle > 1e-8, torch.sin(half) / angle.clamp_min(1e-8), 0.5 - angle**2 / 48)
    return torch.cat([torch.cos(half), k * v], dim=-1)


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
