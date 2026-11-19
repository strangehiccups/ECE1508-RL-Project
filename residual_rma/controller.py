"""
Nominal controller: geometric tracking controller on SE(3) (Lee, Leok & McClamroch, CDC 2010)
(https://arxiv.org/pdf/1003.2005), batched in PyTorch so it runs on all simulated drones at once.

It only knows the *nominal* model: 27 g, nominal inertia, four identical ideal motors, no wind,
no downwash. Everything it does not know is the "residual" that the RL policy has to handle.
It has no integral term, so a steady push leaves a steady position error (error = force / kp).
"""
import math
import torch


def _vee(S):
    """(N,3,3) skew-symmetric -> (N,3)."""
    return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=-1)


class GeometricController:
    def __init__(self, device,
                 mass=0.027, inertia=(1.4e-5, 1.4e-5, 2.17e-5),
                 arm=0.028, k_yaw=0.006, f_max=0.15, g=9.81,
                 kp=0.675, kv=0.23,                     # position loop: ~5 rad/s, damping ~0.85
                 kR=(8.75e-3, 8.75e-3, 1.4e-3),          # attitude loop: ~25 rad/s (yaw ~8 rad/s)
                 kW=(5.6e-4, 5.6e-4, 2.8e-4),
                 max_tilt_deg=35.0):
        self.device = device
        self.m, self.g, self.f_max = mass, g, f_max
        self.J = torch.tensor(inertia, device=device)
        self.kp, self.kv = kp, kv
        self.kR = torch.tensor(kR, device=device)
        self.kW = torch.tensor(kW, device=device)
        self.tan_tilt = math.tan(math.radians(max_tilt_deg))
        self.e3 = torch.tensor([0.0, 0.0, 1.0], device=device)
        L, k = arm, k_yaw
        # wrench [T, tx, ty, tz] = A @ [f1, f2, f3, f4]; motors at (+x+y), (-x+y), (-x-y), (+x-y)
        A = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                          [L,   L,  -L,  -L],
                          [-L,  L,   L,  -L],
                          [k,  -k,   k,  -k]], device=device)
        self.A_inv = torch.linalg.inv(A)

    def __call__(self, pos, vel, R, omega, pos_ref, vel_ref=None, acc_ref=None, yaw_ref=None):
        """pos, vel: (N,3) world; R: (N,3,3) body->world; omega: (N,3) body rates.
        Returns nominal motor thrusts (N,4) in newtons."""
        n = pos.shape[0]
        vel_ref = torch.zeros_like(pos) if vel_ref is None else vel_ref
        acc_ref = torch.zeros_like(pos) if acc_ref is None else acc_ref

        # ---- position loop: desired force in the world frame
        F = (-self.kp * (pos - pos_ref) - self.kv * (vel - vel_ref)
             + self.m * self.g * self.e3 + self.m * acc_ref)
        Fz = F[:, 2].clamp(min=0.3 * self.m * self.g)
        Fxy = F[:, :2]
        lim = (self.tan_tilt * Fz).unsqueeze(-1)
        Fxy = Fxy * torch.clamp(lim / (Fxy.norm(dim=-1, keepdim=True) + 1e-9), max=1.0)
        F = torch.cat([Fxy, Fz.unsqueeze(-1)], dim=-1)

        b3 = R[:, :, 2]
        thrust = (F * b3).sum(-1)

        # ---- desired attitude
        b3d = F / F.norm(dim=-1, keepdim=True)
        if yaw_ref is None:
            b1c = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, 3)
        else:
            b1c = torch.stack([torch.cos(yaw_ref), torch.sin(yaw_ref), torch.zeros_like(yaw_ref)], -1)
        b2d = torch.linalg.cross(b3d, b1c)
        b2d = b2d / b2d.norm(dim=-1, keepdim=True)
        b1d = torch.linalg.cross(b2d, b3d)
        Rd = torch.stack([b1d, b2d, b3d], dim=-1)

        # ---- attitude loop (desired body rates = 0 for hover)
        eR = 0.5 * _vee(Rd.transpose(1, 2) @ R - R.transpose(1, 2) @ Rd)
        eW = omega
        M = -self.kR * eR - self.kW * eW + torch.linalg.cross(omega, omega * self.J)

        wrench = torch.cat([thrust.unsqueeze(-1), M], dim=-1)
        f = wrench @ self.A_inv.T
        return f.clamp(0.0, self.f_max)
