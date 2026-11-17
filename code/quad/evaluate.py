"""
Evaluate the RMA stack in simulation (analogue of Table II / Fig. S4 in the paper).

Modes
  expert    : z_t = mu(e_t)                    (privileged upper bound)
  rma       : z_hat = phi(history), 10 Hz       (what gets deployed)
  no_adapt  : z = mu(0), the nominal extrinsics (base policy without the adaptation module)

Each mode sees identical environments (same RNG seed). Metrics are for the first episode of every env:
  success  = no crash during the 5 s episode
  ttf      = time-to-fall / episode length (1 = never fell)
  pos_err  = mean distance to the hover target while alive (m)
  return   = undiscounted episode return
  z_mse    = |z_hat - mu(e)|^2 (rma only)

  python evaluate.py --ckpt rma.pt --scales 1 2 3 --resample 0.01
"""
import argparse
import torch
from env import QuadHoverEnv
from policy import load_ckpt, norm_hist, ENV_DIM, ADAPT_EVERY


@torch.no_grad()
def rollout(env, enc, model, phi, norm, mode, seed):
    n, dev, T = env.n, env.dev, env.max_steps
    torch.manual_seed(seed)
    env.reset(torch.arange(n, device=dev))
    obs = env.obs()
    z_nom = enc(torch.zeros(1, ENV_DIM, device=dev)).expand(n, -1)
    z = z_nom

    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    crashed_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    ttf = torch.ones(n, device=dev)
    ret = torch.zeros(n, device=dev)
    err_sum = torch.zeros(n, device=dev)
    err_cnt = torch.zeros(n, device=dev)
    zmse_sum, zmse_cnt = 0.0, 0.0

    for t in range(T):
        z_true = enc(env.privileged())
        if mode == "expert":
            z = z_true
        elif mode == "rma":
            if t % ADAPT_EVERY == 0:
                z = phi(norm_hist(env.hist, norm))
            zmse_sum += ((z - z_true) ** 2).mean(-1)[~ended].sum().item()
            zmse_cnt += (~ended).sum().item()
        else:
            z = z_nom
        a = model.dist(norm(obs), z).mean                 # deterministic policy
        obs, rew, done = env.step(a)

        active = ~ended
        ret += rew * active
        ok = active & ~done                               # obs of finished envs already belongs to the next episode
        err_sum += obs[:, :3].norm(dim=-1) * ok
        err_cnt += ok
        new_crash = env.last_crashed & active
        ttf[new_crash] = (t + 1) / T
        crashed_ever |= new_crash
        ended |= done

    return {"success": 100.0 * (~crashed_ever).float().mean().item(),
            "ttf": ttf.mean().item(),
            "pos_err": (err_sum.sum() / err_cnt.sum().clamp(min=1)).item(),
            "return": ret.mean().item(),
            "z_mse": zmse_sum / zmse_cnt if zmse_cnt > 0 else float("nan")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rma.pt")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                    help="multipliers on the training randomization ranges (>1 = out of distribution)")
    ap.add_argument("--resample", type=float, default=0.01, help="per-step re-sampling probability (paper test: 0.01)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = QuadHoverEnv(num_envs=args.envs, resample_prob=args.resample)
    enc, model, phi, norm = load_ckpt(args.ckpt, env.dev)
    modes = ["expert", "no_adapt"] + (["rma"] if phi is not None else [])

    print(f"{'scale':>5} {'mode':>9} {'success%':>9} {'ttf':>6} {'pos_err':>8} {'return':>8} {'z_mse':>8}")
    for s in args.scales:
        env.rand_scale = s
        for mode in modes:
            r = rollout(env, enc, model, phi, norm, mode, args.seed)
            print(f"{s:5.1f} {mode:>9} {r['success']:9.1f} {r['ttf']:6.3f} {r['pos_err']:8.3f} "
                  f"{r['return']:8.1f} {r['z_mse']:8.4f}")
        print()
