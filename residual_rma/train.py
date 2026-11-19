"""
Residual RMA training.

  Phase 1 (PPO): base policy pi(x_t, a_{t-1}, z_t) and encoder mu(e_t) -> z_t, trained together
                 with privileged e_t. Curriculum: disturbance scale 0.3 -> 1.0 over the first third.
  Phase 2 (supervised, on-policy): adaptation module phi(last 50 state/command pairs) -> z_hat,
                 regressing mu(e_t); pi and mu are frozen; the rollouts use z_hat.

  python train.py --phase 1 --mode residual --name res_z          # residual + z   (the main method)
  python train.py --phase 1 --mode residual --no-z --name res_noz # residual, no z (robust baseline)
  python train.py --phase 1 --mode e2e --name e2e_z --iters 600   # end-to-end + z (v1-style, new env)
  python train.py --phase 2 --name res_z                          # adds phi -> checkpoints/res_z.pt

Outputs: checkpoints/<name>_phase1.pt, checkpoints/<name>.pt, logs/<name>_phase{1,2}.csv
"""
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import ResidualQuadEnv
from policy import EnvEncoder, ActorCritic, AdaptationModule
from utils import RunningNorm, CsvLog

N = 2048
H1, EPOCHS1, MB1 = 24, 5, 4
GAMMA, LAMBDA, CLIP, LR1 = 0.99, 0.95, 0.2, 3e-4
H2, EPOCHS2, MB2, LR2 = 16, 4, 4, 5e-4
SCALE0, CURRIC_FRAC = 0.3, 1 / 3

CKPT_DIR, LOG_DIR = "checkpoints", "logs"


def save(path, enc, policy, norm, cfg, adaptor=None, adapt_every=None):
    d = {"enc": enc.state_dict(), "policy": policy.state_dict(), "norm": norm.state_dict(), "cfg": cfg}
    if adaptor is not None:
        d["adaptor"], d["adapt_every"] = adaptor.state_dict(), adapt_every
    torch.save(d, path)


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    enc = EnvEncoder().to(device); enc.load_state_dict(ck["enc"])
    policy = ActorCritic(use_z=cfg["use_z"]).to(device); policy.load_state_dict(ck["policy"])
    norm = RunningNorm(policy.obs_dim, device); norm.load_state_dict(ck["norm"])
    adaptor = None
    if "adaptor" in ck:
        adaptor = AdaptationModule().to(device); adaptor.load_state_dict(ck["adaptor"])
    for m in (enc, policy, adaptor):
        if m is not None:
            m.eval()
    return enc, policy, adaptor, norm, cfg, ck.get("adapt_every", 1)


# =================================================================================================
def phase1(args):
    torch.manual_seed(args.seed)
    env = ResidualQuadEnv(num_envs=N, mode=args.mode, rand_scale=SCALE0)
    device = env.device
    enc, policy = EnvEncoder().to(device), ActorCritic(use_z=not args.no_z).to(device)
    norm = RunningNorm(env.obs_dim, device)
    params = list(enc.parameters()) + list(policy.parameters())
    opt = torch.optim.Adam(params, lr=LR1)
    cfg = {"mode": args.mode, "use_z": not args.no_z, "seed": args.seed, "iters": args.iters}
    log = CsvLog(f"{LOG_DIR}/{args.name}_phase1.csv",
                 ["it", "scale", "step_rew", "ep_ret", "ep_len_s", "crash_frac", "pos_err", "act_std", "sps", "wall_min"])

    # spread episode clocks so that resets are not synchronized across envs
    env.steps[:] = torch.randint(0, env.max_steps, (N,), device=device)
    obs = env.obs()
    ep_ret = torch.zeros(N, device=device)
    ep_len = torch.zeros(N, device=device)
    t_start = time.time()

    for it in range(args.iters):
        t0 = time.time()
        env.rand_scale = SCALE0 + (1.0 - SCALE0) * min(1.0, it / (CURRIC_FRAC * args.iters))
        O = torch.zeros(H1, N, env.obs_dim, device=device)
        E = torch.zeros(H1, N, env.env_dim, device=device)
        A = torch.zeros(H1, N, 4, device=device)
        LP, V, R, D = (torch.zeros(H1, N, device=device) for _ in range(4))
        fin_ret, fin_len, fin_crash, perr = [], [], [], []

        for t in range(H1):
            norm.update(obs)
            o, e = norm(obs), env.privileged()
            with torch.no_grad():
                z = enc(e)
                d = policy.dist(o, z)
                a = d.sample()
                O[t], E[t], A[t], V[t] = o, e, a, policy.value(o, z)
                LP[t] = d.log_prob(a).sum(-1)
            obs, rew, done = env.step(a)
            perr.append(env._state18()[:, :3].norm(dim=-1).mean())
            # time-outs are not failures: bootstrap them with the value of the current state
            R[t] = rew + GAMMA * V[t] * env.last_timeout.float()
            D[t] = done.float()
            ep_ret += rew; ep_len += 1
            if done.any():
                fin_ret.append(ep_ret[done].clone()); fin_len.append(ep_len[done].clone())
                fin_crash.append(env.last_crashed[done].float())
                ep_ret[done] = 0.0; ep_len[done] = 0.0

        with torch.no_grad():
            last_v = policy.value(norm(obs), enc(env.privileged()))
        adv = torch.zeros(H1, N, device=device)
        last = torch.zeros(N, device=device)
        for t in reversed(range(H1)):
            nonterm = 1.0 - D[t]
            next_v = V[t + 1] if t < H1 - 1 else last_v
            delta = R[t] + GAMMA * next_v * nonterm - V[t]
            last = delta + GAMMA * LAMBDA * nonterm * last
            adv[t] = last
        ret = adv + V

        o_f, e_f, a_f = O.reshape(-1, env.obs_dim), E.reshape(-1, env.env_dim), A.reshape(-1, 4)
        lp_f, adv_f, ret_f = LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        B = o_f.shape[0]
        mb = B // MB1
        for _ in range(EPOCHS1):
            perm = torch.randperm(B, device=device)
            for i in range(0, B, mb):
                idx = perm[i:i + mb]
                z = enc(e_f[idx])
                d = policy.dist(o_f[idx], z)
                ratio = (d.log_prob(a_f[idx]).sum(-1) - lp_f[idx]).exp()
                ab = adv_f[idx]
                ab = (ab - ab.mean()) / (ab.std() + 1e-8)
                pg = -torch.min(ratio * ab, ratio.clamp(1 - CLIP, 1 + CLIP) * ab).mean()
                vl = 0.5 * (policy.value(o_f[idx], z) - ret_f[idx]).pow(2).mean()
                loss = pg + 0.5 * vl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

        mr = torch.cat(fin_ret).mean().item() if fin_ret else float("nan")
        ml = torch.cat(fin_len).mean().item() * env.control_dt if fin_len else float("nan")
        cf = torch.cat(fin_crash).mean().item() if fin_crash else float("nan")
        sps = N * H1 / (time.time() - t0)
        log.write(it=it, scale=round(env.rand_scale, 3), step_rew=R.mean().item(), ep_ret=mr, ep_len_s=ml,
                  crash_frac=cf, pos_err=torch.stack(perr).mean().item(),
                  act_std=policy.log_std.exp().mean().item(), sps=round(sps), wall_min=(time.time() - t_start) / 60)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"[P1 {args.name}] it {it:4d} scale {env.rand_scale:.2f} rew {R.mean().item():6.3f} "
                  f"ep_ret {mr:8.1f} crash {cf:5.2f} err {torch.stack(perr).mean().item():.3f} "
                  f"std {policy.log_std.exp().mean().item():.3f} {sps:,.0f} sps", flush=True)
        if it % 50 == 0 or it == args.iters - 1:
            save(f"{CKPT_DIR}/{args.name}_phase1.pt", enc, policy, norm, cfg)
    log.close()


# =================================================================================================
def phase2(args):
    torch.manual_seed(args.seed + 100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, policy, _, norm, cfg, _ = load(f"{CKPT_DIR}/{args.name}_phase1.pt", device)
    assert cfg["use_z"], "a policy trained without z has nothing to adapt"
    for p in list(enc.parameters()) + list(policy.parameters()):
        p.requires_grad_(False)
    env = ResidualQuadEnv(num_envs=N, mode=cfg["mode"], rand_scale=1.0)
    adaptor = AdaptationModule().to(device)
    opt = torch.optim.Adam(adaptor.parameters(), lr=LR2)
    log = CsvLog(f"{LOG_DIR}/{args.name}_phase2.csv", ["it", "z_mse", "z_r2", "e_r2", "step_rew", "crash_frac", "sps"])

    env.steps[:] = torch.randint(0, env.max_steps, (N,), device=device)
    obs = env.obs()
    z_hat = torch.zeros(N, policy.z_dim, device=device)
    k = 0
    for it in range(args.iters2):
        t0 = time.time()
        Hb = torch.zeros(H2, N, env.hist_len, env.hist_dim, device=device)
        Zb = torch.zeros(H2, N, policy.z_dim, device=device)
        Eb = torch.zeros(H2, N, env.env_dim, device=device)
        rew_sum, crashes, dones = 0.0, 0.0, 0.0
        for t in range(H2):
            hn = env.norm_hist(norm)
            with torch.no_grad():
                e = env.privileged()
                Zb[t], Eb[t] = enc(e), e
                if k % args.adapt_every == 0:
                    z_hat = adaptor(hn)
                a = policy.dist(norm(obs), z_hat).sample()
            Hb[t] = hn
            obs, rew, done = env.step(a)
            rew_sum += rew.mean().item()
            crashes += env.last_crashed.sum().item(); dones += done.sum().item()
            k += 1

        Hf = Hb.reshape(-1, env.hist_len, env.hist_dim)
        Zf, Ef = Zb.reshape(-1, policy.z_dim), Eb.reshape(-1, env.env_dim)
        B = Hf.shape[0]
        with torch.no_grad():
            zp, ep = adaptor.forward_both(Hf)
            z_mse = F.mse_loss(zp, Zf).item()
            z_r2 = 1 - z_mse / Zf.var(0).mean().item()
            e_r2 = (1 - ((ep - Ef) ** 2).mean(0) / Ef.var(0).clamp(min=1e-6)).mean().item()
        mb = B // MB2
        for _ in range(EPOCHS2):
            perm = torch.randperm(B, device=device)
            for i in range(0, B, mb):
                idx = perm[i:i + mb]
                zp, ep = adaptor.forward_both(Hf[idx])
                loss = F.mse_loss(zp, Zf[idx]) + 0.1 * F.mse_loss(ep, Ef[idx])
                opt.zero_grad(); loss.backward(); opt.step()

        sps = N * H2 / (time.time() - t0)
        cf = crashes / max(dones, 1)
        log.write(it=it, z_mse=z_mse, z_r2=z_r2, e_r2=e_r2, step_rew=rew_sum / H2, crash_frac=cf, sps=round(sps))
        if it % 10 == 0 or it == args.iters2 - 1:
            print(f"[P2 {args.name}] it {it:4d} z_mse {z_mse:.4f} z_R2 {z_r2:.3f} e_R2 {e_r2:.3f} "
                  f"rew {rew_sum / H2:.3f} {sps:,.0f} sps", flush=True)
        if it % 25 == 0 or it == args.iters2 - 1:
            save(f"{CKPT_DIR}/{args.name}.pt", enc, policy, norm, cfg, adaptor, args.adapt_every)
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--mode", choices=["residual", "e2e"], default="residual")
    ap.add_argument("--no-z", action="store_true", help="policy without the extrinsics z (robust baseline)")
    ap.add_argument("--name", default="res_z")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--iters2", type=int, default=150)
    ap.add_argument("--adapt-every", type=int, default=1, help="phi update period in control steps (1 = 100 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    if args.phase in ("1", "all"):
        phase1(args)
    if args.phase in ("2", "all") and not args.no_z:
        phase2(args)
