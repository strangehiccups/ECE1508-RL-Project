"""
RMA training (Fig. 2 of Kumar et al. 2021)

  Phase 1: PPO jointly trains the Env Factor Encoder mu and the Base Policy pi,
           with privileged e_t: z_t = mu(e_t), a_t = pi(x_t, a_{t-1}, z_t)
  Phase 2: supervised, on-policy training of the Adaptation Module phi (DAgger-style):
           zhat_t = phi(x_{t-k:t-1}, a_{t-k:t-1}), the base policy is rolled out with zhat
           (refreshed at 10 Hz) and phi regresses the true z_t = mu(e_t).

  python train.py                 # phase 1 then phase 2
  python train.py --phase 1       # -> rma_phase1.pt
  python train.py --phase 2       # loads rma_phase1.pt -> rma.pt
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from env import QuadHoverEnv
from policy import EnvEncoder, ActorCritic, AdaptationModule
from utils import RunningNorm, save_ckpt, load_ckpt

N = 2048    # envs
# phase 1 (PPO) ===============================================================
H1 = 24     # rollout length
EPOCHS1 = 5
MINIBATCHES1 = 4
GAMMA, LAM, CLIP, LR1 = 0.99, 0.95, 0.2, 3e-4
# phase 2 (supervised) ========================================================
H2 = 16         # rollout length
EPOCHS2 = 4
MINIBATCHES2 = 4
LR2 = 5e-4
ADAPT_DT = 0.1 # adaptation module (phi) runs at 10 Hz, base policy (pi) at 100 Hz

PHASE1_CKPT, PHASE2_CKPT, FINAL_CKPT = "rma_phase1", "rma_phase2", "rma"

# =============================================================================
# Phase 1: base policy + env factor encoder with PPO
# =============================================================================
def phase1(env: QuadHoverEnv, iters: int):
    device = env.device
    enc = EnvEncoder().to(device)
    policy = ActorCritic().to(device)
    norm = RunningNorm(env.obs_dim, device)
    params = list(enc.parameters()) + list(policy.parameters())
    opt = torch.optim.Adam(params, lr=LR1)
    writer = SummaryWriter("runs/" + PHASE1_CKPT)

    env.reset(torch.arange(N, device=device))
    obs = env.obs()
    ep_ret = torch.zeros(N, device=device)

    for it in range(iters):
        t0 = time.time()
        O = torch.zeros(H1, N, env.obs_dim, device=device)      # observations
        E = torch.zeros(H1, N, env.force_dim, device=device)    # environmental forces (4 thrust + 3 gust)
        A = torch.zeros(H1, N, env.n_actions, device=device)    # actions
        LP = torch.zeros(H1, N, device=device)                  # ?
        V = torch.zeros(H1, N, device=device)
        R = torch.zeros(H1, N, device=device)
        D = torch.zeros(H1, N, device=device)
        finished = []

        for t in range(H1):
            norm.update(obs)
            o = norm(obs)
            e = env.privileged()
            with torch.no_grad():
                z = enc(e)
                d = policy.dist(o, z)
                a = d.sample()
                O[t], E[t], A[t], V[t] = o, e, a, policy.value(o, z)
                LP[t] = d.log_prob(a).sum(-1)
            obs, rew, done = env.step(a)
            R[t], D[t] = rew, done.float()
            ep_ret += rew
            if done.any():
                finished.append(ep_ret[done].clone())
                ep_ret[done] = 0.0

        with torch.no_grad():
            last_v = policy.value(norm(obs), enc(env.privileged()))
        adv = torch.zeros(H1, N, device=device)
        last = torch.zeros(N, device=device)
        for t in reversed(range(H1)):
            nonterm = 1.0 - D[t]
            next_v = V[t + 1] if t < H1 - 1 else last_v
            delta = R[t] + GAMMA * next_v * nonterm - V[t]
            last = delta + GAMMA * LAM * nonterm * last
            adv[t] = last
        ret = adv + V

        o_f, e_f = O.reshape(-1, env.obs_dim), E.reshape(-1, env.force_dim)
        a_f, lp_f = A.reshape(-1, env.n_actions), LP.reshape(-1)
        adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
        B = o_f.shape[0]
        for _ in range(EPOCHS1):
            perm = torch.randperm(B, device=device)
            for i in range(0, B, B // MINIBATCHES1):
                idx = perm[i:i + B // MINIBATCHES1]
                z = enc(e_f[idx])                       # gradients flow into mu as well
                d = policy.dist(o_f[idx], z)
                ratio = (d.log_prob(a_f[idx]).sum(-1) - lp_f[idx]).exp()
                ab = adv_f[idx]
                ab = (ab - ab.mean()) / (ab.std() + 1e-8)
                pg = -torch.min(ratio * ab, ratio.clamp(1 - CLIP, 1 + CLIP) * ab).mean()
                vl = 0.5 * (policy.value(o_f[idx], z) - ret_f[idx]).pow(2).mean()
                loss = pg + 0.5 * vl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

        mean_ret = torch.cat(finished).mean().item() if finished else float("nan")
        writer.add_scalar("reward/step", R.mean().item(), it)
        writer.add_scalar("reward/episode_return", mean_ret, it)
        writer.add_scalar("policy/action_std", policy.log_std.exp().mean().item(), it)
        print(f"[P1] it {it:4d}  step_rew {R.mean().item():6.3f}  ep_ret {mean_ret:8.2f}  "
              f"{N * H1 / (time.time() - t0):,.0f} steps/s")

        if it % 50 == 0 or it == iters - 1:
            save_ckpt(PHASE1_CKPT + ".pt", enc, policy, norm)


# =============================================================================
# Phase 2: adaptation module with on-policy supervised regression
# =============================================================================
def phase2(env: QuadHoverEnv, iters: int):
    ADAPT_EVERY = int(ADAPT_DT // env.control_dt)
    device = env.device
    enc, policy, _, norm, _ = load_ckpt(path=PHASE1_CKPT + ".pt",
                                        enc_type=EnvEncoder, policy_type=ActorCritic,
                                        device=device)
    for p in list(enc.parameters()) + list(policy.parameters()):
        p.requires_grad_(False)
    phi = AdaptationModule().to(device)
    opt = torch.optim.Adam(phi.parameters(), lr=LR2)
    writer = SummaryWriter("runs/" + PHASE2_CKPT)

    env.reset(torch.arange(N, device=device))
    obs = env.obs()
    z_hat = torch.zeros(N, policy.z_dim, device=device)
    step_i = 0                                            # global clock for the 10 Hz phi updates

    for it in range(iters):
        t0 = time.time()
        Hb = torch.zeros(H2, N, env.hist_len, env.obs_dim, device=device)   # normalized histories
        Zb = torch.zeros(H2, N, policy.z_dim, device=device)                # targets mu(e_t)
        rew_sum = 0.0

        # ---- on-policy rollout with z_hat from the (initially random) phi ----
        for t in range(H2):
            o = norm(obs)
            hn = env.norm_hist(norm)
            with torch.no_grad():
                Zb[t] = enc(env.privileged())
                if step_i % ADAPT_EVERY == 0:             # asynchronous: phi at 10 Hz, pi at 100 Hz
                    z_hat = phi(hn)
                a = policy.dist(o, z_hat).sample()         # sampled actions -> exploratory trajectories
            Hb[t] = hn
            obs, rew, done = env.step(a)
            rew_sum += rew.mean().item()
            step_i += 1

        # ---- supervised regression  MSE(z_hat, z) ----
        Hf, Zf = Hb.reshape(-1, env.hist_len, env.obs_dim), Zb.reshape(-1, policy.z_dim)
        B = Hf.shape[0]
        with torch.no_grad():                             # error on fresh on-policy data (before the update)
            pre_mse = F.mse_loss(phi(Hf), Zf).item()
        z_var = Zf.var(0).mean().item()
        for _ in range(EPOCHS2):
            perm = torch.randperm(B, device=device)
            for i in range(0, B, B // MINIBATCHES2):
                idx = perm[i:i + B // MINIBATCHES2]
                loss = F.mse_loss(phi(Hf[idx]), Zf[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        r2 = 1.0 - pre_mse / max(z_var, 1e-8)
        writer.add_scalar("adapt/mse_on_policy", pre_mse, it)
        writer.add_scalar("adapt/r2", r2, it)
        writer.add_scalar("reward/step", rew_sum / H2, it)
        print(f"[P2] it {it:4d}  mse {pre_mse:7.4f}  R2 {r2:6.3f}  step_rew {rew_sum / H2:6.3f}  "
              f"{N * H2 / (time.time() - t0):,.0f} steps/s")

        if it % 25 == 0 or it == iters - 1:
            save_ckpt(FINAL_CKPT + ".pt", enc, policy, norm, phi, ADAPT_EVERY)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--iters1", type=int, default=1000)
    ap.add_argument("--iters2", type=int, default=300)
    args = ap.parse_args()

    env = QuadHoverEnv(num_envs=N)        # training distribution, resample_prob = 0.004
    if args.phase in ("1", "all"):
        phase1(env, args.iters1)
    if args.phase in ("2", "all"):
        phase2(env, args.iters2)
