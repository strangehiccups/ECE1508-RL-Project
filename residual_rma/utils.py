"""Small helpers shared by the residual-RMA scripts."""
import csv
import torch


def quat_to_rotmat(q):
    """q: (N,4) quaternion (w,x,y,z) -> (N,3,3) rotation matrix (body -> world)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


class RunningNorm:
    """Running mean/std observation normalizer."""
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class CsvLog:
    """Append-only CSV log (one row per training iteration), easy to plot later."""
    def __init__(self, path, fields):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def write(self, **row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
