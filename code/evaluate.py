"""
Evaluate the RMA stack in simulation (analogue of Table II / Fig. S4 in the paper).

Modes
  expert    : z_t = mu(e_t)                     (privileged upper bound)
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
from policy import EnvEncoder, ActorCritic, AdaptationModule
from utils import RunningNorm, load_ckpt

@torch.no_grad()
def rollout(env: QuadHoverEnv,
            enc: EnvEncoder,
            policy: ActorCritic,
            adaptor: AdaptationModule,
            norm: RunningNorm,
            ADAPT_EVERY: int,
            mode: str,
            seed: int):
    n, device, T = env.n, env.device, env.max_steps
    torch.manual_seed(seed)
    env.reset(torch.arange(n, device=device))
    obs = env.obs()
    z_nom = enc(torch.zeros(1, env.env_dim, device=device)).expand(n, -1)
    z = z_nom

    ended = torch.zeros(n, dtype=torch.bool, device=device)
    crashed_ever = torch.zeros(n, dtype=torch.bool, device=device)
    ttf = torch.ones(n, device=device)
    ret = torch.zeros(n, device=device)
    err_sum = torch.zeros(n, device=device)
    err_cnt = torch.zeros(n, device=device)
    zmse_sum, zmse_cnt = 0.0, 0.0

    for t in range(T):
        z_true = enc(env.privileged())
        if mode == "expert":
            z = z_true
        elif mode == "rma":
            if t % ADAPT_EVERY == 0:
                z = adaptor(env.norm_hist(norm))
            zmse_sum += ((z - z_true) ** 2).mean(-1)[~ended].sum().item()
            zmse_cnt += (~ended).sum().item()
        else:
            z = z_nom
        a = policy.dist(norm(obs), z).mean                 # deterministic policy
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
    enc, policy, adaptor, norm, ADAPT_EVERY = load_ckpt(path=args.ckpt,
                                                        enc_type=EnvEncoder,
                                                        policy_type=ActorCritic,
                                                        adaptor_type=AdaptationModule,
                                                        device=env.device)
    modes = ["expert", "no_adapt"] + (["rma"] if adaptor is not None else [])

    print(f"{'mode':>9}\t{'randomisation factor':>5}\t{'success %':>9}\t{'time-to-fall [0,1]':>6}\t{'position error':>8}\t{'undiscounted return':>8}\t{'z-mse':>8}")
    for s in args.scales:
        env.rand_scale = s
        for mode in modes:
            r = rollout(env, enc, policy, adaptor, norm, ADAPT_EVERY, mode, args.seed)
            print(f"{mode:>9}\t{s:5.1f}\t\t\t{r['success']:9.1f}\t{r['ttf']:6.3f}\t\t\t{r['pos_err']:8.3f}\t"
                  f"{r['return']:8.1f}\t\t{r['z_mse']:8.4f}")
        print()
