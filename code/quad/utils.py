import torch
import torch.nn as nn

class RunningNorm:
    """Running mean/std observation normalizer on the GPU."""
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

# ---- checkpoint helpers ---------------------------------------------------
def save_ckpt(path, enc, policy, norm, phi=None, adapt_every: int=None):
    d = {"enc": enc.state_dict(), "policy": policy.state_dict(),
         "mean": norm.mean, "var": norm.var, "count": norm.count}
    if phi is not None:
        d["phi"] = phi.state_dict()
    if adapt_every is not None:
        d["ADAPT_EVERY"] = adapt_every
    torch.save(d, path)

def load_ckpt(path,
              enc_type: type[nn.Module],
              policy_type: type[nn.Module],
              adaptor_type: type[nn.Module]=None,
              enc_args: dict={},
              policy_args: dict={},
              adaptor_args: dict={},
              device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(path, map_location=device)
    enc = enc_type(**enc_args).to(device)
    enc.load_state_dict(ck["enc"])
    policy = policy_type(**policy_args).to(device)
    policy.load_state_dict(ck["policy"])
    norm = RunningNorm(policy.obs_dim, device)
    norm.mean, norm.var, norm.count = ck["mean"], ck["var"], ck["count"]
    phi = None
    if ("phi" in ck) and (adaptor_type is not None):
        phi = adaptor_type(**adaptor_args).to(device)
        phi.load_state_dict(ck["phi"])
    for m in (enc, policy, phi):
        if m is not None:
            m.eval()
    ADAPT_EVERY = None
    if "ADAPT_EVERY" in ck:
        ADAPT_EVERY = ck["ADAPT_EVERY"]
    return enc, policy, phi, norm, ADAPT_EVERY