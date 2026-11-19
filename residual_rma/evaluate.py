"""
Compare all controllers on the same test scenarios (same random draws for every method).

Methods
  nominal     geometric controller only (a = 0)
  res_robust  nominal + residual policy trained WITHOUT z            (checkpoints/res_noz_phase1.pt)
  res_z0      nominal + residual policy, z = mu(0) (adaptation off)  (checkpoints/res_z.pt)
  res_expert  nominal + residual policy, z = mu(e_t) (privileged)    (checkpoints/res_z.pt)
  res_rma     nominal + residual policy, z_hat = phi(history)        (checkpoints/res_z.pt)   <- the method
  e2e_expert  end-to-end policy (v1 style) retrained in this env, z = mu(e_t)   (checkpoints/e2e_z.pt)
  e2e_rma     same, z_hat = phi(history)
  v1_rma      Edwin's trained code/rma.pt, unchanged, dropped into the new env

Scenarios (all 10 s unless noted, 100 Hz)
  calm        no disturbance at all
  v1_range    about v1's training world: gains +-10%, steady force +-0.025 N, no gusts, no passes
  train       training ranges (scale 1), 60% of episodes have an overhead pass, re-draw prob 0.01/step
  strong      2x the training ranges (out of distribution)
  pass_X      one overhead pass at height X m (0.25 / 0.5 / 0.75), nothing else; 8 s
  pass_0.15   closer than anything seen in training (unseen)
  pass_0.25x2 pass at 0.25 m with a 2x stronger downwash (unseen)

  python evaluate.py                       # everything, writes results/results.json + results.md
  python evaluate.py --methods nominal res_rma --scenarios train pass_0.25
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import torch

from env import ResidualQuadEnv
from train import load

ALL_METHODS = ["nominal", "res_robust", "res_z0", "res_expert", "res_rma", "e2e_expert", "e2e_rma", "v1_rma"]
ALL_SCENARIOS = ["calm", "v1_range", "train", "strong", "pass_0.75", "pass_0.5", "pass_0.25", "pass_0.15", "pass_0.25x2"]


# ---- agents ---------------------------------------------------------------------------------------
class Zero:
    def __init__(self): self.env_mode = "residual"
    def reset(self, n, device): pass
    def act(self, env, obs, t): return torch.zeros(env.n, 4, device=env.device)


class RMA:
    """kind: expert | rma | z0 | robust"""
    def __init__(self, path, kind, device):
        self.enc, self.policy, self.adaptor, self.norm, cfg, self.k = load(path, device)
        self.kind, self.env_mode = kind, cfg["mode"]
        assert kind != "rma" or self.adaptor is not None, f"{path} has no adaptation module (run phase 2)"

    def reset(self, n, device):
        self.z = torch.zeros(n, 8, device=device)
        self.e_hat = None

    def act(self, env, obs, t):
        if self.kind == "expert":
            self.z = self.enc(env.privileged())
        elif self.kind == "z0":
            self.z = self.enc(torch.zeros(env.n, env.env_dim, device=env.device))
        elif self.kind == "rma" and t % self.k == 0:
            self.z, self.e_hat = self.adaptor.forward_both(env.norm_hist(self.norm))
        return self.policy.dist(self.norm(obs), self.z).mean


class V1:
    """Edwin's v1 checkpoint (code/rma.pt) with his own classes, observation (22) and 10 Hz adaptor."""
    def __init__(self, path, code_dir, device):
        spec = importlib.util.spec_from_file_location("v1_policy", Path(code_dir) / "policy.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        ck = torch.load(path, map_location=device, weights_only=False)
        self.policy = m.ActorCritic().to(device); self.policy.load_state_dict(ck["policy"])
        self.adaptor = m.AdaptationModule().to(device); self.adaptor.load_state_dict(ck["adaptor"])
        self.mean, self.var = ck["mean"].to(device), ck["var"].to(device)
        self.k = ck.get("ADAPT_EVERY", 10)
        self.env_mode = "e2e"

    def reset(self, n, device):
        self.z = torch.zeros(n, 8, device=device)

    def _n(self, x, d):
        return ((x - self.mean[:d]) / torch.sqrt(self.var[:d] + 1e-8)).clamp(-10, 10)

    def act(self, env, obs, t):
        o22 = torch.cat([obs[:, :18], obs[:, 22:26]], -1)          # [pos_err, R, v, w, a_prev]
        if t % self.k == 0:
            h = env.hist                                            # [state 18, a 4], same layout as v1
            self.z = self.adaptor(torch.cat([self._n(h[..., :18], 18), h[..., 18:]], -1))
        return self.policy.dist(self._n(o22, 22), self.z).mean


def make_agent(name, args, device):
    """name may end in _s1, _s2 ... to load the run trained with that seed (e.g. res_rma_s1)."""
    ck = args.ckpt_dir
    seed = ""
    if "_s" in name and name.rsplit("_s", 1)[1].isdigit():
        name, seed = name.rsplit("_s", 1)[0], "_s" + name.rsplit("_s", 1)[1]
    if name == "nominal":    return Zero()
    if name == "res_robust": return RMA(f"{ck}/res_noz{seed}_phase1.pt", "robust", device)
    if name == "res_z0":     return RMA(f"{ck}/res_z{seed}.pt", "z0", device)
    if name == "res_expert": return RMA(f"{ck}/res_z{seed}.pt", "expert", device)
    if name == "res_rma":    return RMA(f"{ck}/res_z{seed}.pt", "rma", device)
    if name == "e2e_expert": return RMA(f"{ck}/e2e_z{seed}.pt", "expert", device)
    if name == "e2e_rma":    return RMA(f"{ck}/e2e_z{seed}.pt", "rma", device)
    if name == "v1_rma":     return V1(args.v1_ckpt, args.v1_code, device)
    raise ValueError(name)


# ---- scenarios ------------------------------------------------------------------------------------
RANDOM = {"calm": (0.0, 0.0, 0.0), "v1_range": (0.5, 0.0, 0.01), "train": (1.0, 0.6, 0.01), "strong": (2.0, 0.6, 0.01)}
PASS_BATCH = ["pass_0.75", "pass_0.5", "pass_0.25", "pass_0.15"]      # simulated together (same downwash scale)


def make_env(scns, mode, n_rand, n_pass, seed):
    """scns: one random scenario, or a list of pass scenarios that share a downwash scale.
    Returns env, T, {scenario: env indices}."""
    torch.manual_seed(seed)
    if scns[0] in RANDOM:
        scale, pp, rs = RANDOM[scns[0]]
        env = ResidualQuadEnv(num_envs=n_rand, mode=mode, rand_scale=scale, pass_prob=pp, resample_prob=rs)
        if scns[0] == "v1_range":                   # v1's training world: no gusts (v1 never saw them)
            env.dist.gust_on[:] = 0.0; env.dist.gust[:] = 0.0
        T, groups = env.max_steps, {scns[0]: torch.arange(n_rand)}
    else:
        n = n_pass * len(scns)
        env = ResidualQuadEnv(num_envs=n, mode=mode, rand_scale=2.0 if scns[0].endswith("x2") else 1.0,
                              pass_prob=0.0, resample_prob=0.0)
        ids = torch.arange(n, device=env.device)
        env.dist.clear(ids)
        g = torch.Generator(device="cpu").manual_seed(seed)
        heading = torch.rand(n_pass, generator=g) * 2 * math.pi
        offset = (torch.rand(n_pass, generator=g) * 2 - 1) * 0.1
        heading[0], offset[0] = 0.0, 0.0                       # first drone of each group: straight over (plotted)
        groups = {}
        for j, scn in enumerate(scns):
            dz = float(scn.split("_")[1].rstrip("x2"))
            groups[scn] = torch.arange(j * n_pass, (j + 1) * n_pass)
            for i in range(n_pass):
                env.dist.force_pass(ids[j * n_pass + i:j * n_pass + i + 1], dz, speed=0.3, offset=float(offset[i]),
                                    t0=0.5, heading=float(heading[i]))
        env.qpos[:, 0:3] = env.target
        env.qvel[:] = 0.0
        T = 800
    env.steps[:] = 0
    env._after_change()
    env.max_steps = T + 1          # no time-out inside the evaluation window
    env.auto_reset = False         # keep the random stream identical for every method
    return env, T, groups


@torch.no_grad()
def rollout(agent, scns, args, device):
    env, T, groups = make_env(scns, agent.env_mode, args.envs, args.pass_envs, args.seed)
    n = env.n
    agent.reset(n, device)
    obs = env.obs()
    z0 = lambda: torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    ttf, se_sum, se_cnt, drop, max_err, eff_sum, jerk_sum, zm, zc = torch.ones(n, device=device), *(z0() for _ in range(8))
    prev_cmd = None
    e_true, e_pred, e_mask = [], [], []
    traces = {s: [] for s in groups}
    t_settle = 0 if scns[0].startswith("pass") else int(1.0 / env.control_dt)   # skip the reset transient

    for t in range(T):
        a = agent.act(env, obs, t)
        if isinstance(agent, RMA) and agent.kind == "rma":
            z_true = agent.enc(env.privileged())
            zm += ((agent.z - z_true) ** 2).mean(-1) * alive; zc += alive.float()
            if agent.e_hat is not None and t % agent.k == 0 and t >= 50:
                e_true.append(env.privileged()); e_pred.append(agent.e_hat.clone()); e_mask.append(alive.clone())
        obs, _, _ = env.step(a)
        cmd = env.last_cmd / env.hover
        crashed = env.last_crashed & alive
        ttf[crashed] = (t + 1) / T
        alive &= ~env.last_crashed
        dead = ~alive
        if dead.any():
            # park crashed drones at the target so a tumbling drone cannot feed NaNs to the policy.
            # They are already excluded from every metric, and this uses no random numbers.
            env.qpos[dead, 0:3] = env.target; env.qpos[dead, 3] = 1.0; env.qpos[dead, 4:7] = 0.0
            env.qvel[dead] = 0.0; env.hist[dead] = 0.0; env.prev_action[dead] = 0.0
            env._after_change()
            obs = env.obs()
        pos_err = obs[:, :3]
        e = pos_err.norm(dim=-1)
        m = alive.float()
        if t >= t_settle:
            se_sum += e ** 2 * m; se_cnt += m
            max_err = torch.maximum(max_err, e * m)
        drop = torch.maximum(drop, (-pos_err[:, 2]) * m)
        eff_sum += ((cmd - 1.0) ** 2).mean(-1) * m
        if prev_cmd is not None:
            jerk_sum += (cmd - prev_cmd).abs().mean(-1) * m
        prev_cmd = cmd
        for s, idx in groups.items():
            traces[s].append(env.qpos[idx[0], 2].item())

    alive_steps = (ttf * T).clamp(min=1)
    out = {}
    for s, idx in groups.items():
        al = alive[idx]
        r = {"success": 100.0 * al.float().mean().item(),
             "ttf": ttf[idx].mean().item(),
             "rmse": torch.sqrt(se_sum[idx].sum() / se_cnt[idx].sum().clamp(min=1)).item(),
             "max_err": max_err[idx][al].mean().item() if al.any() else float("nan"),
             "max_drop": drop[idx][al].mean().item() if al.any() else float("nan"),
             "effort": torch.sqrt(eff_sum[idx] / alive_steps[idx]).mean().item(),
             "jerk": (jerk_sum[idx] / alive_steps[idx]).mean().item(),
             "z_mse": (zm[idx].sum() / zc[idx].sum()).item() if zc[idx].sum() > 0 else float("nan"),
             "trace_z": traces[s]}
        if e_true:
            msk = torch.stack(e_mask)[:, idx].reshape(-1)
            et = torch.stack(e_true)[:, idx].reshape(-1, env.env_dim)[msk]
            ep = torch.stack(e_pred)[:, idx].reshape(-1, env.env_dim)[msk]
            var = et.var(0)
            r2 = 1 - ((ep - et) ** 2).mean(0) / var.clamp(min=1e-8)
            ok = var > 1e-6                     # skip constant entries (yaw torque from downwash is always 0)
            for key, sl in [("e_r2_gain", slice(0, 4)), ("e_r2_force", slice(4, 7)), ("e_r2_torque", slice(7, 10))]:
                r[key] = r2[sl][ok[sl]].mean().item() if ok[sl].any() else float("nan")
        out[s] = r
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--scenarios", nargs="+", default=ALL_SCENARIOS)
    ap.add_argument("--envs", type=int, default=512, help="drones per random scenario")
    ap.add_argument("--pass-envs", type=int, default=64, help="drones per pass scenario")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--v1-ckpt", default="../code/rma.pt")
    ap.add_argument("--v1-code", default="../code")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}

    for name in args.methods:
        try:
            agent = make_agent(name, args, device)
        except (FileNotFoundError, AssertionError) as ex:
            print(f"skip {name}: {ex}")
            continue
        batches = [[s] for s in args.scenarios if s in RANDOM or s.endswith("x2")]
        batches += [[s for s in args.scenarios if s in PASS_BATCH]]
        for scns in [b for b in batches if b]:
            for scn, r in rollout(agent, scns, args, device).items():
                results.setdefault(name, {})[scn] = r
                print(f"{name:>11} {scn:>12}  success {r['success']:5.1f}%  rmse {100 * r['rmse']:5.1f} cm  "
                      f"drop {100 * r['max_drop']:5.1f} cm  effort {r['effort']:.3f}  jerk {r['jerk']:.4f}  "
                      f"z_mse {r['z_mse']:.3f}", flush=True)
            json.dump(results, open(path, "w"))

    # ---- markdown tables
    rows = [m for m in ALL_METHODS if m in results]
    with open(f"{args.out}/results.md", "w") as f:
        for title, scns, cols in [
            ("Random scenarios (512 drones each, 10 s)", ["calm", "v1_range", "train", "strong"], ["success", "rmse"]),
            ("Overhead passes (64 drones each, 8 s)", ["pass_0.75", "pass_0.5", "pass_0.25", "pass_0.15", "pass_0.25x2"],
             ["success", "max_drop"]),
        ]:
            scns = [s for s in scns if any(s in results[m] for m in rows)]
            f.write(f"### {title}\n\n| method | " + " | ".join(f"{s} {c}" for s in scns for c in cols) + " |\n")
            f.write("|---|" + "---|" * (len(scns) * len(cols)) + "\n")
            for m in rows:
                cells = []
                for s in scns:
                    for c in cols:
                        v = results[m].get(s, {}).get(c, float("nan"))
                        cells.append(f"{v:.0f}%" if c == "success" else (f"{100 * v:.1f} cm" if c in ("rmse", "max_drop") else f"{v:.3f}"))
                f.write(f"| {m} | " + " | ".join(cells) + " |\n")
            f.write("\n")
    print(open(f"{args.out}/results.md").read())
