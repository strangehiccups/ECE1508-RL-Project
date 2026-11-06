import time
import torch
from env import QuadHoverEnv

env = QuadHoverEnv(num_envs=2048)
obs = env.obs()
print("obs shape:", obs.shape)

# open-loop hover thrust: drone should hold altitude roughly
t0 = time.time()
for i in range(200):
    obs, rew, done = env.step(torch.zeros(env.n, 4, device=env.dev))
torch.cuda.synchronize()
dt = time.time() - t0
print("z mean after 200 steps:", env.qpos[:, 2].mean().item())
print("env steps/s:", 200 * env.n / dt)

# disturbance test: 0.05 N sideways gust on all envs
env.apply_disturbance(torch.tensor([0.05, 0.0, 0.0], device=env.dev).expand(env.n, 3))
for i in range(50):
    env.step(torch.zeros(env.n, 4, device=env.dev))
print("x mean after gust:", env.qpos[:, 0].mean().item())