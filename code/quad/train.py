import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from env import QuadHoverEnv
from policy import ActorCritic, RunningNorm

N, H = 2048, 24                       # envs, rollout length
ITERS, EPOCHS, MINIBATCHES = 1000, 5, 4
GAMMA, LAM, CLIP, LR = 0.99, 0.95, 0.2, 3e-4

env = QuadHoverEnv(num_envs=N)
dev = env.dev
model = ActorCritic().to(dev)
norm = RunningNorm(22, dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter("runs/hover")

obs = env.obs()
ep_ret = torch.zeros(N, device=dev)

for it in range(ITERS):
    t0 = time.time()
    O = torch.zeros(H, N, 22, device=dev)
    A = torch.zeros(H, N, 4, device=dev)
    LP = torch.zeros(H, N, device=dev)
    V = torch.zeros(H, N, device=dev)
    R = torch.zeros(H, N, device=dev)
    D = torch.zeros(H, N, device=dev)
    finished = []

    for t in range(H):
        norm.update(obs)
        o = norm(obs)
        with torch.no_grad():
            d = model.dist(o)
            a = d.sample()
            O[t], A[t], V[t] = o, a, model.value(o)
            LP[t] = d.log_prob(a).sum(-1)
        obs, rew, done = env.step(a)
        R[t], D[t] = rew, done.float()
        ep_ret += rew
        if done.any():
            finished.append(ep_ret[done].clone())
            ep_ret[done] = 0.0

    with torch.no_grad():
        last_v = model.value(norm(obs))
    adv = torch.zeros(H, N, device=dev)
    last = torch.zeros(N, device=dev)
    for t in reversed(range(H)):
        nonterm = 1.0 - D[t]
        next_v = V[t + 1] if t < H - 1 else last_v
        delta = R[t] + GAMMA * next_v * nonterm - V[t]
        last = delta + GAMMA * LAM * nonterm * last
        adv[t] = last
    ret = adv + V

    o, a, lp = O.reshape(-1, 22), A.reshape(-1, 4), LP.reshape(-1)
    adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
    B = o.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(B, device=dev)
        for i in range(0, B, B // MINIBATCHES):
            idx = perm[i:i + B // MINIBATCHES]
            d = model.dist(o[idx])
            ratio = (d.log_prob(a[idx]).sum(-1) - lp[idx]).exp()
            ab = adv_f[idx]
            ab = (ab - ab.mean()) / (ab.std() + 1e-8)
            pg = -torch.min(ratio * ab, ratio.clamp(1 - CLIP, 1 + CLIP) * ab).mean()
            vl = 0.5 * (model.value(o[idx]) - ret_f[idx]).pow(2).mean()
            loss = pg + 0.5 * vl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    mean_ret = torch.cat(finished).mean().item() if finished else float("nan")
    writer.add_scalar("reward/step", R.mean().item(), it)
    writer.add_scalar("reward/episode_return", mean_ret, it)
    writer.add_scalar("policy/action_std", model.log_std.exp().mean().item(), it)
    print(f"it {it:4d}  step_rew {R.mean().item():6.3f}  ep_ret {mean_ret:8.2f}  "
          f"{N * H / (time.time() - t0):,.0f} steps/s")

    if it % 50 == 0 or it == ITERS - 1:
        torch.save({"model": model.state_dict(), "mean": norm.mean, "var": norm.var,
                    "count": norm.count}, "policy.pt")