"""State-action histories for the Phase 2 adaptation modules, one buffer per drone.

At time t each buffer holds steps t-L .. t-1 (oldest first). After a reset every slot
holds the first step's entry, with the hover command as the action: as if the drone had
hovered at its start state for the whole window (on hardware the buffer is full when the
policy engages).
    bottom: [x_k, a_k]            proprio (measured state) + the action taken at step k
    top:    [s_k^top, u_{k-1}^top] TOP_LAYOUT: the top's state and the command it flew
                                   with over the step before (same entry pi sees)
The adaptation module phi reads both. The lengths are independent (ObsCfg) and must cover
the longest wake transport delay once that is on (README "Planned").
A length of 0 disables that buffer.
"""

import torch


class HistoryBuffer:
    """Batched circular history buffer for a single tensor, every step is stored twice
    Continuous slices of the last L steps are returned by get()."""

    def __init__(self, num_envs: int, length: int, dim: int, device: torch.device):
        assert length > 0, "use no buffer instead of a zero-length one"
        self.L = length
        self.buf = torch.zeros(num_envs, 2 * length, dim, device=device)
        self.p = 0          # slot of the oldest step = the one the next push overwrites

    def push(self, x: torch.Tensor) -> None:
        self.buf[:, self.p] = x
        self.buf[:, self.p + self.L] = x
        self.p = (self.p + 1) % self.L

    def fill(self, ids: torch.Tensor, x: torch.Tensor) -> None:
        self.buf[ids] = x[:, None]

    def get(self) -> torch.Tensor:
        return self.buf[:, self.p:self.p + self.L]


class TransitionHistory:
    """The bottom and top drones' histories, each with its own length."""

    def __init__(self, num_envs: int, bottom_len: int, bottom_dim: int, top_len: int,
                 top_dim: int, device: torch.device):
        spec = {"bottom": (bottom_len, bottom_dim), "top": (top_len, top_dim)}
        self.buffers = {k: HistoryBuffer(num_envs, L, d, device)
                        for k, (L, d) in spec.items() if L > 0}
        self.fresh = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _entries(self, obs: dict[str, torch.Tensor], action: torch.Tensor):
        x = {"bottom": lambda: torch.cat([obs["proprio"], action], dim=-1),
             "top": lambda: obs["top"]}
        return {k: x[k]() for k in self.buffers}

    def push(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> None:
        for k, x in self._entries(obs, action).items():
            self.buffers[k].push(x)

    def reset(self, ids: torch.Tensor) -> None:
        """The next fill_fresh() fills these envs' buffers."""
        self.fresh[ids] = True

    def fill_fresh(self, obs: dict[str, torch.Tensor], prev_action: torch.Tensor) -> None:
        """Every slot of the envs reset since the last call <- this step's entry."""
        ids = self.fresh.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        for k, x in self._entries(obs, prev_action).items():
            self.buffers[k].fill(ids, x[ids])
        self.fresh[ids] = False

    def get(self) -> dict[str, torch.Tensor]:
        """{"bottom": (N, L_bottom, 22), "top": (N, L_top, 22)}; enabled buffers only.
        Views into the buffers: clone() to keep them past the next push."""
        return {k: b.get() for k, b in self.buffers.items()}
