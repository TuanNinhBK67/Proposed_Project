"""DDPG and TD3 baselines: ONE centralized agent on the global state with a deterministic bounded action.

Action a in [-1, 1]^{L PU + LMN + K Np} (see action_maps.physical_from_unit): power logits -> softmax over
PU = U+1 shares (env.power_map "softmax") or PU = U per-feed levels ("linear"), phases pi (a + 1), ports = top-n_k port scores per UE. The port selection is a non-differentiable
projection of the continuous scores; the critic is learned on the continuous score vector (proto-action).
Exploration: uniform random actions for warmup_steps, then Gaussian noise. TD3 adds twin critics, target policy
smoothing and delayed actor/target updates (Fujimoto et al., 2018). Same reward, episodes (T steps, terminal,
no bootstrap at the end) and held-out evaluation as the on-policy learners.
"""
import copy
import os

import torch
import torch.nn as nn

from ..action_maps import physical_from_unit, unit_action_dim
from ..config import save_config
from ..env import SimFasEnv
from ..evaluate import evaluate_policy
from ..training import EpisodeTracker, run_training_loop, summarise_episodes
from ..utils import SliceNormalizer, atomic_write, make_generator, seed_everything


def mlp(in_dim, hidden, out_dim):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DetActor(nn.Module):
    def __init__(self, state_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(state_dim, hidden, act_dim)
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s):
        return torch.tanh(self.net(s))


class QNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(state_dim + act_dim, hidden, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class ReplayBuffer:
    """Ring buffer over (row, env) of RAW states. Episodes end with a true terminal, so the successor of row t
    is row t+1 of the same environment, and it is only used (bootstrapped) when done[t] = 0."""

    def __init__(self, capacity, E, state_dim, act_dim, device):
        self.cap, self.E, self.device = max(2, capacity // E), E, device
        self.s = torch.zeros(self.cap, E, state_dim, device=device)
        self.a = torch.zeros(self.cap, E, act_dim, device=device)
        self.r = torch.zeros(self.cap, E, device=device)
        self.d = torch.zeros(self.cap, E, device=device)
        self.ptr = self.size = 0

    def add(self, s, a, r, d):
        self.s[self.ptr], self.a[self.ptr], self.r[self.ptr], self.d[self.ptr] = s, a, r, d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, generator=None):
        oldest = (self.ptr - self.size) % self.cap
        off = torch.randint(0, self.size - 1, (n,), device=self.device, generator=generator)  # never the newest row
        t = (oldest + off) % self.cap
        e = torch.randint(0, self.E, (n,), device=self.device, generator=generator)
        tn = (t + 1) % self.cap
        return self.s[t, e], self.a[t, e], self.r[t, e], self.d[t, e], self.s[tn, e]


class OffPolicyAgent:
    """Inference policy: deterministic actor on the normalised global state (frozen statistics)."""

    def __init__(self, actor, cfg, state_norm):
        self.actor, self.cfg, self.state_norm = actor, cfg, state_norm

    @torch.no_grad()
    def __call__(self, env, o):
        a = self.actor(self.state_norm(env.global_state(), update=False))
        return physical_from_unit(a, self.cfg, env.pmax, self.cfg.offpolicy.power_scale, env.nk, env.n_max)

    @classmethod
    def from_saved(cls, d, cfg, device):
        from ..config import dims
        dm, s = dims(cfg), cfg.system
        actor = DetActor(dm["state"], unit_action_dim(s.L, s.U, s.M, s.N, s.K, s.Np, cfg.env.power_map),
                         list(cfg.offpolicy.hidden))
        actor.load_state_dict(d["actor"])
        actor.to(device).eval()
        sn = SliceNormalizer(*dm["state_channel"], device, cfg.offpolicy.obs_clip)
        sn.rms.load_state_dict(d["state_norm"])
        return cls(actor, cfg, sn)


class OffPolicyTrainer:
    def __init__(self, cfg, algo: str, run_dir: str | None = None, device: str | None = None):
        if algo not in ("ddpg", "td3"):
            raise ValueError("off-policy algo must be 'ddpg' or 'td3'")
        self.cfg, self.algo, self.td3 = cfg, algo, algo == "td3"
        self.device = torch.device(device or cfg.resolved_device())
        seed_everything(cfg.seed)
        oc, s = cfg.offpolicy, cfg.system
        self.env = SimFasEnv(cfg, oc.num_envs, cfg.seed, self.device)
        e = self.env
        self.act_dim = unit_action_dim(s.L, s.U, s.M, s.N, s.K, s.Np, cfg.env.power_map)
        hid = list(oc.hidden)
        self.actor = DetActor(e.state_dim, self.act_dim, hid).to(self.device)
        self.critics = nn.ModuleList([QNet(e.state_dim, self.act_dim, hid) for _ in range(2 if self.td3 else 1)])
        self.critics.to(self.device)
        self.actor_t, self.critics_t = copy.deepcopy(self.actor), copy.deepcopy(self.critics)
        for p in list(self.actor_t.parameters()) + list(self.critics_t.parameters()):
            p.requires_grad_(False)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=oc.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critics.parameters(), lr=oc.critic_lr)
        self.state_norm = SliceNormalizer(*e.state_channel, self.device, oc.obs_clip)
        self.replay = ReplayBuffer(oc.buffer_size, e.E, e.state_dim, self.act_dim, self.device)
        self.gen = make_generator(cfg.seed + 29, self.device)
        self.policy_delay = oc.policy_delay if self.td3 else 1
        self.env_steps = self.updates = self.episodes = 0
        self.best = -float("inf")
        self.run_dir = run_dir
        self.tracker = EpisodeTracker(e.E, e.T, self.device)
        self._o = None
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            save_config(cfg, os.path.join(run_dir, "config.yaml"))

    # ------------------------------------------------------------------ interaction
    @torch.no_grad()
    def env_step(self, finished: list):
        oc, env = self.cfg.offpolicy, self.env
        if self._o is None:
            self._o = env.reset()
        s_raw = env.global_state()
        s_n = self.state_norm(s_raw, update=True)
        if self.env_steps < oc.warmup_steps:
            a = torch.rand(env.E, self.act_dim, device=self.device, generator=self.gen) * 2 - 1
        else:
            a = self.actor(s_n)
            a = (a + oc.expl_noise * torch.randn(a.shape, device=self.device, generator=self.gen)).clamp(-1, 1)
        p, phase, ports = physical_from_unit(a, self.cfg, env.pmax, oc.power_scale)
        self._o, info = env.step(p, phase, ports)
        done = info["terminated"]
        self.replay.add(s_raw, a, info["reward"].float(), done.float())
        self.env_steps += env.E
        finished += self.tracker.step(self, info, self.env_steps)

    # ------------------------------------------------------------------ learning
    def update(self) -> dict:
        oc = self.cfg.offpolicy
        s, a, r, d, s2 = self.replay.sample(oc.batch_size, self.gen)
        s, s2 = self.state_norm(s, update=False), self.state_norm(s2, update=False)
        with torch.no_grad():
            a2 = self.actor_t(s2)
            if self.td3:
                noise = (oc.target_noise * torch.randn(a2.shape, device=self.device, generator=self.gen)
                         ).clamp(-oc.noise_clip, oc.noise_clip)
                a2 = (a2 + noise).clamp(-1, 1)
            q2 = torch.min(torch.stack([q(s2, a2) for q in self.critics_t]), 0).values
            y = oc.reward_scale * r + oc.discount * (1.0 - d) * q2
        qs = [q(s, a) for q in self.critics]
        c_loss = sum(((qi - y) ** 2).mean() for qi in qs)
        self.opt_critic.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), oc.max_grad_norm)
        self.opt_critic.step()
        self.updates += 1
        out = {"critic_loss": c_loss.item(), "q_mean": qs[0].mean().item()}
        if self.updates % self.policy_delay == 0:
            a_loss = -self.critics[0](s, self.actor(s)).mean()
            self.opt_actor.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), oc.max_grad_norm)
            self.opt_actor.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_t), (self.critics, self.critics_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - oc.tau).add_(oc.tau * p)
            out["actor_loss"] = a_loss.item()
        return out

    def _iteration(self, frac_done: float) -> dict:
        oc = self.cfg.offpolicy
        finished, stats = [], {}
        n_vec = max(1, 1024 // self.env.E)                 # ~1k environment steps per logged iteration
        for _ in range(n_vec):
            self.env_step(finished)
            if self.env_steps >= oc.warmup_steps and self.replay.size > 2:
                for _ in range(oc.gradient_steps):
                    for k, v in self.update().items():
                        stats.setdefault(k, []).append(v)
        out = {k: sum(v) / len(v) for k, v in stats.items()}
        return out | summarise_episodes(finished) | {"finished": finished}

    # ------------------------------------------------------------------ evaluation / saving
    def policy(self) -> OffPolicyAgent:
        return OffPolicyAgent(self.actor, self.cfg, self.state_norm)

    def evaluate(self, episodes=None) -> dict:
        tr = self.cfg.training
        return evaluate_policy(self.cfg, self.policy(), episodes or tr.eval_episodes,
                               self.cfg.seed + tr.eval_seed_offset, self.device)

    def export_model(self, path: str, eval_result: dict | None = None):
        d = {"algo": self.algo, "config": self.cfg.to_dict(), "actor": self.actor.state_dict(),
             "state_norm": self.state_norm.rms.state_dict(), "env_steps": self.env_steps, "eval": eval_result}
        atomic_write(path, lambda fn: torch.save(d, fn))

    def train(self) -> dict:
        return run_training_loop(self, max(1, 1024 // self.env.E) * self.env.E, self._iteration)
