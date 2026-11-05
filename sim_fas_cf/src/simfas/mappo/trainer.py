"""On-policy learners: MAPPO (CTDE, decentralized AP actors on local observations, centralized critic) and
single-agent PPO (one centralized actor on the global state, same critic). Common shared reward.

Per update: num_envs x rollout_steps transitions are collected on-policy (all AP actions applied jointly and
simultaneously), GAE (Eq. (38)) is computed once per transition with a terminal at the end of every T-step
episode (bootstrap 0) and V(s_next) at the rollout boundary, and every agent's PPO-clip loss (Eq. (36)-(37))
uses its JOINT latent log-likelihood (Eq. (32)) with the common normalised advantage (ppo.per_head_clip: one
ratio and clipped surrogate per action head, summed); the loss is averaged over agents (ppo.fas_stats_active_only: the port-head terms only over the agents with
port draws). Stored latents and ordered FAS draws are re-scored, never re-sampled.
"""
import math
import os
import torch
import torch.nn as nn
from ..action_maps import phase_from_latent, power_from_latent
from ..config import power_dims, save_config
from ..env import SimFasEnv
from ..evaluate import evaluate_policy
from ..fas import merge_port_proposals, topk_ports
from ..phy import desired_port_power, desired_port_power_local
from ..training import EpisodeTracker, run_training_loop, summarise_episodes
from ..utils import SliceNormalizer, ValueNorm, atomic_write, make_generator, seed_everything
from .actor import Critic, build_actor

def gae(rewards, values, next_values, terminated, gamma, lam):
    """Generalised advantage estimation over (T, E); terminated[t] stops bootstrapping and the recursion."""
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nonterm = 1.0 - terminated[t]
        delta = rewards[t] + gamma * nonterm * next_values[t] - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values

def masked_mean(v, w=None):
    """Mean of v (mb, A); w (mb, A) 0/1 -> mean over the entries with w = 1 only (0 if there are none)."""
    return v.mean() if w is None else (v * w).sum() / w.sum().clamp_min(1)

def head_means(v, w=None):
    """(mb, A, 3) -> (3,) per-head means [power, phase, ports]; w (mb, A) -> the ports entry is averaged over the
    agents with w = 1 only (ppo.fas_stats_active_only: agents with >= 1 port draw)."""
    return v.mean((0, 1)) if w is None else torch.cat([v[..., :2].mean((0, 1)), masked_mean(v[..., 2], w)[None]])

def active_mask(env, central: bool, fas_learn: bool) -> torch.Tensor:
    """(E, A, K, n_max): which FAS draws an agent makes (controlled UEs x used slots)."""
    sv = env.slot_valid
    act = sv.expand(env.E, 1, *sv.shape) if central else env.ctrl_mask[..., None] & sv
    return act if fas_learn else torch.zeros_like(act)

def to_env_action(a, env, cfg, central):
    """Latent actions -> joint physical action (p, phase, ports), applying the fixed parts of ablations."""
    pc = cfg.ppo
    E, L = env.E, env.L
    p = power_from_latent(a["z_power"], env.pmax, cfg.env.power_map).reshape(E, L, env.U)
    phase = phase_from_latent(a["z_phase"], env.M, env.N).reshape(E, L, env.M, env.N)
    ports = a["draws"][:, 0] if central else merge_port_proposals(a["draws"], env.ctrl)
    if not pc.learn_power:
        p = torch.full_like(p, env.pmax / env.U)
    if not pc.learn_phase:
        phase = env.phase.clone()                    # random phases drawn at the episode start, kept for T steps
    if pc.fas_mode == "fixed_random":
        ports = env.ports.clone()
    elif pc.fas_mode == "strongest":                 # port gains of the configuration in effect (in o_t)
        g = desired_port_power(env.Hbar, env.Q1, env.Qinter, env.phase, env.p, env.F)
        ports = topk_ports(g, env.nk, env.n_max)
    elif pc.fas_mode == "strongest_new":             # port gains through the new p, phase of this step
        g = desired_port_power(env.Hbar, env.Q1, env.Qinter, phase, p, env.F)
        ports = topk_ports(g, env.nk, env.n_max)
    elif pc.fas_mode == "strongest_local":           # control AP's own contribution through its new p, phase
        g = desired_port_power_local(env.Hbar, env.Q1, env.Qinter, phase, p, env.F, env.ctrl)
        ports = topk_ports(g, env.nk, env.n_max)
    return p, phase, ports

class OnPolicyAgent:
    """Inference policy of a MAPPO / PPO actor with frozen normalisation statistics. deterministic = Gaussian
    means and greedy sequential FAS draws."""

    def __init__(self, actor, cfg, algo, obs_norm, state_norm, device, deterministic=True, seed=0):
        self.actor, self.cfg, self.algo = actor, cfg, algo
        self.central = algo == "ppo"
        self.obs_norm, self.state_norm, self.det = obs_norm, state_norm, deterministic
        self.gen = make_generator(seed + 23, device)

    @torch.no_grad()
    def __call__(self, env, o):
        if self.central:
            x = self.state_norm(env.global_state(), update=False)[:, None]
        else:
            x = self.obs_norm(o["obs"], update=False)
        a = self.actor.act(x, active_mask(env, self.central, self.cfg.ppo.fas_mode == "learned"), self.det,
                           None if self.det else self.gen)
        return to_env_action(a, env, self.cfg, self.central)

    @classmethod
    def from_saved(cls, d, cfg, device, deterministic=True):
        from ..config import dims
        from ..utils import SliceNormalizer
        dm = dims(cfg)
        algo = d["algo"]
        actor = build_actor(cfg, algo, dm["state"] if algo == "ppo" else dm["obs"]).to(device)
        actor.load_state_dict(d["actor"])
        actor.eval()
        on = SliceNormalizer(*dm["obs_channel"], device, cfg.ppo.obs_clip)
        sn = SliceNormalizer(*dm["state_channel"], device, cfg.ppo.obs_clip)
        on.rms.load_state_dict(d["obs_norm"])
        sn.rms.load_state_dict(d["state_norm"])
        return cls(actor, cfg, algo, on, sn, device, deterministic)

class OnPolicyTrainer:
    def __init__(self, cfg, algo: str, run_dir: str | None = None, device: str | None = None):
        if algo not in ("mappo", "ppo"):
            raise ValueError("on-policy algo must be 'mappo' or 'ppo'")
        self.cfg, self.algo, self.central = cfg, algo, algo == "ppo"
        self.device = torch.device(device or cfg.resolved_device())
        seed_everything(cfg.seed)
        pc = cfg.ppo
        self.env = SimFasEnv(cfg, pc.num_envs, cfg.seed, self.device)
        e = self.env
        self.A = 1 if self.central else e.L
        self.in_dim = e.state_dim if self.central else e.obs_dim
        self.actor = build_actor(cfg, algo, self.in_dim).to(self.device)
        nets = self.actor.nets                            # parameter groups for head_stop_freeze
        self._trunk_params = [q for n in nets for q in n.body.parameters()]
        self._head_params = [[q for n in nets for q in (*n.power_mean.parameters(), n.power_log_std)],
                             [q for n in nets for q in (*n.phase_mean.parameters(), n.phase_log_std)],
                             [q for n in nets for q in n.fas_logits.parameters()]]
        if pc.std_schedule != "none":                     # std set by the schedule, not by the optimizer
            for net in self.actor.nets:
                net.power_log_std.requires_grad_(False)
                net.phase_log_std.requires_grad_(False)
        self.critic = Critic(e.state_dim, list(pc.hidden)).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=pc.actor_lr, eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=pc.critic_lr, eps=1e-5)
        self.obs_norm = SliceNormalizer(*e.obs_channel, self.device, pc.obs_clip)
        self.state_norm = SliceNormalizer(*e.state_channel, self.device, pc.obs_clip)
        self.vnorm = ValueNorm(self.device, pc.value_norm)
        self.fas_learn = pc.fas_mode == "learned"
        T, E, A = pc.rollout_steps, e.E, self.A
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.device)
        self.buf = {"x": z(T, E, A, self.in_dim), "state": z(T, E, e.state_dim),
                    "active": z(T, E, A, e.K, e.n_max, dt=torch.bool),
                    "z_power": z(T, E, A, self.actor.n_ap, power_dims(cfg)), "z_phase": z(T, E, A, self.actor.n_ap, e.M * e.N),
                    "draws": z(T, E, A, e.K, e.n_max, dt=torch.long), "logp": z(T, E, A),
                    "logp_parts": z(T, E, A, 3), "value": z(T, E),
                    "reward": z(T, E), "terminated": z(T, E)}
        self.env_steps = self.updates = self.episodes = 0
        self.best = -float("inf")
        self.run_dir = run_dir
        self._o = None
        self.tracker = EpisodeTracker(E, e.T, self.device)
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            save_config(cfg, os.path.join(run_dir, "config.yaml"))

    # ------------------------------------------------------------------ rollout
    def _inputs(self, o, update: bool):
        st = self.state_norm(self.env.global_state(), update)
        x = st[:, None] if self.central else self.obs_norm(o["obs"], update)
        return x, st

    @torch.no_grad()
    def collect(self) -> dict:
        env, b, T = self.env, self.buf, self.cfg.ppo.rollout_steps
        if self._o is None:
            self._o = env.reset()
        o = self._o
        finished = []
        for t in range(T):
            x, st = self._inputs(o, update=True)
            active = active_mask(env, self.central, self.fas_learn)
            a = self.actor.act(x, active)
            v = self.vnorm.denormalize(self.critic(st))
            p, phase, ports = to_env_action(a, env, self.cfg, self.central)
            o, info = env.step(p, phase, ports)
            r = info["reward"].float()
            for k, val in (("x", x), ("state", st), ("active", active), ("z_power", a["z_power"]),
                           ("z_phase", a["z_phase"]), ("draws", a["draws"]), ("logp", a["logp"]),
                           ("logp_parts", a["logp_parts"]), ("value", v),
                           ("reward", r), ("terminated", info["terminated"].float())):
                b[k][t] = val
            finished += self.tracker.step(self, info, self.env_steps + (t + 1) * env.E)
        self._o = o
        _, st_next = self._inputs(o, update=False)
        self.last_value = self.vnorm.denormalize(self.critic(st_next))
        self.env_steps += T * env.E
        return {"rollout_reward": b["reward"].mean().item(), **summarise_episodes(finished), "finished": finished}

    # ------------------------------------------------------------------ PPO update
    def update(self) -> dict:
        pc, b = self.cfg.ppo, self.buf
        next_values = torch.cat([b["value"][1:], self.last_value[None]], 0)
        adv, ret = gae(b["reward"], b["value"], next_values, b["terminated"], pc.discount, pc.gae_lambda)
        self.vnorm.update(ret)
        ret_n = self.vnorm.normalize(ret)
        T, E = adv.shape
        n = T * E
        f = lambda x: x.reshape((n,) + x.shape[2:])
        x, st, active, zp, zf, dr, logp_old, parts_old = (f(b[k]) for k in ("x", "state", "active", "z_power",
                                                                             "z_phase", "draws", "logp", "logp_parts"))
        adv, ret_n = f(adv), f(ret_n)
        mb = min(pc.minibatch_size, n)
        starts = list(range(0, n, mb))
        if len(starts) > 1 and n - starts[-1] < 2:   # a 1-sample tail has no advantage std: merge it into the
            starts.pop()                             #   previous minibatch (other batch sizes unchanged)
        bounds = list(zip(starts, starts[1:] + [n]))
        heads = ("power", "phase", "fas")
        stats = {k: 0.0 for k in ("pg_loss", "v_loss", "ent_power", "ent_phase", "ent_fas", "approx_kl", "clipfrac",
                                  *(f"clipfrac_{h}" for h in heads), *(f"approx_kl_{h}" for h in heads))}
        w_head = torch.tensor(pc.head_loss_weights, device=self.device)
        live = torch.ones(3, dtype=torch.bool, device=self.device)        # heads still updated (head_target_kl)
        learned = torch.tensor([self.actor.learn_power, self.actor.learn_phase, self.fas_learn], device=self.device)
        head_epochs = [0, 0, 0]
        count = epochs = 0
        for _ in range(pc.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_kl, epoch_mb = 0.0, 0
            epoch_kl_h = torch.zeros(3, device=self.device)
            epochs += 1
            for h in range(3):
                head_epochs[h] += int(live[h] and learned[h])
            for s, s_end in bounds:
                i = perm[s:s_end]
                ev = self.actor.evaluate(x[i], active[i], zp[i], zf[i], dr[i])
                logratio = ev["logp"] - logp_old[i]
                ratio = logratio.exp()
                a = adv[i]
                a = ((a - a.mean()) / (a.std() + 1e-8))[:, None]          # common advantage for all agents
                ratio_h = (ev["logp_parts"] - parts_old[i]).exp()         # (mb, A, 3): power, phase, ports
                # (mb, A) agents with >= 1 port draw; fas_stats_active_only: port-head means over these agents only
                w_fas = active[i].flatten(2).any(-1).float() if pc.fas_stats_active_only else None
                on = live.float()
                if pc.per_head_clip:                                      # a fixed head has ratio 1: no gradient
                    a3 = a[..., None]
                    pg_h = head_means(torch.max(-a3 * ratio_h, -a3 * ratio_h.clamp(1 - pc.clip, 1 + pc.clip)), w_fas)
                    pg = (w_head * on * pg_h).sum()                       # weighted, stopped heads removed
                else:
                    pg = torch.max(-a * ratio, -a * ratio.clamp(1 - pc.clip, 1 + pc.clip)).mean()
                ent = (on[0] * pc.entropy_coef_power * ev["ent_power"].mean()
                       + on[1] * pc.entropy_coef_phase * ev["ent_phase"].mean()
                       + on[2] * pc.entropy_coef_fas * masked_mean(ev["ent_fas"], w_fas))
                loss_a = pg - ent
                if not torch.isfinite(loss_a):
                    raise FloatingPointError("non-finite actor loss")
                self.opt_actor.zero_grad()
                loss_a.backward()
                stopped = learned & ~live
                if pc.head_stop_freeze and stopped.any():          # no grad -> Adam leaves the parameter as is
                    frozen = self._trunk_params + [q for h in range(3) if stopped[h] for q in self._head_params[h]]
                    for q in frozen:
                        q.grad = None
                nn.utils.clip_grad_norm_(self.actor.parameters(), pc.max_grad_norm)
                self.opt_actor.step()
                v_loss = 0.5 * ((self.critic(st[i]) - ret_n[i]) ** 2).mean()
                if not torch.isfinite(v_loss):
                    raise FloatingPointError("non-finite critic loss")
                self.opt_critic.zero_grad()
                (pc.value_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), pc.max_grad_norm)
                self.opt_critic.step()
                with torch.no_grad():
                    stats["pg_loss"] += pg.item()
                    stats["v_loss"] += v_loss.item()
                    for k in ("ent_power", "ent_phase"):
                        stats[k] += ev[k].mean().item()
                    stats["ent_fas"] += masked_mean(ev["ent_fas"], w_fas).item()
                    kl = ((ratio - 1) - logratio).mean().item()
                    stats["approx_kl"] += kl
                    epoch_kl, epoch_mb = epoch_kl + kl, epoch_mb + 1
                    stats["clipfrac"] += ((ratio - 1).abs() > pc.clip).float().mean().item()
                    lr_h = ev["logp_parts"] - parts_old[i]
                    kl_h = head_means((lr_h.exp() - 1) - lr_h, w_fas)     # per-head KL estimate
                    epoch_kl_h += kl_h
                    clip_h = head_means(((ratio_h - 1).abs() > pc.clip).float(), w_fas)
                    for j, h in enumerate(heads):
                        stats[f"clipfrac_{h}"] += clip_h[j].item()
                        stats[f"approx_kl_{h}"] += kl_h[j].item()
                count += 1
            if pc.target_kl is not None and epoch_kl / max(epoch_mb, 1) > pc.target_kl:
                break
            if pc.head_target_kl is not None:
                live &= ~(epoch_kl_h / max(epoch_mb, 1) > torch.tensor(pc.head_target_kl, device=self.device))
                if not (live & learned).any():
                    break
        self.updates += 1
        out = {k: v / max(count, 1) for k, v in stats.items()}
        out["ppo_epochs_run"] = epochs
        for h, e in zip(heads, head_epochs):
            out[f"ppo_epochs_{h}"] = e
        var_r = ret.var()
        out["explained_var"] = (1 - (ret - b["value"]).var() / var_r).item() if var_r > 0 else float("nan")
        out["power_std"], out["phase_std"] = self.actor.mean_std()
        return out

    # ------------------------------------------------------------------ evaluation / saving
    def policy(self, deterministic=True) -> OnPolicyAgent:
        return OnPolicyAgent(self.actor, self.cfg, self.algo, self.obs_norm, self.state_norm, self.device,
                             deterministic, self.cfg.seed)

    def evaluate(self, episodes=None) -> dict:
        tr = self.cfg.training
        return evaluate_policy(self.cfg, self.policy(), episodes or tr.eval_episodes,
                               self.cfg.seed + tr.eval_seed_offset, self.device)

    def export_model(self, path: str, eval_result: dict | None = None):
        d = {"algo": self.algo, "config": self.cfg.to_dict(), "actor": self.actor.state_dict(),
             "obs_norm": self.obs_norm.rms.state_dict(), "state_norm": self.state_norm.rms.state_dict(),
             "env_steps": self.env_steps, "eval": eval_result}
        atomic_write(path, lambda fn: torch.save(d, fn))

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict:
        return run_training_loop(self, self.cfg.ppo.rollout_steps * self.env.E, self._iteration)

    def scheduled_std(self, frac_done: float) -> float:
        pc = self.cfg.ppo
        s0 = math.exp(pc.init_log_std)
        return s0 + (pc.final_std - s0) * min(max(frac_done, 0.0), 1.0)

    def _iteration(self, frac_done: float) -> dict:
        pc = self.cfg.ppo
        if pc.std_schedule == "linear":                   # same std for the rollout and the update that follows
            log_std = math.log(self.scheduled_std(frac_done))
            with torch.no_grad():
                for net in self.actor.nets:
                    net.power_log_std.fill_(log_std)
                    net.phase_log_std.fill_(log_std)
        if pc.lr_decay == "linear":
            for opt, lr in ((self.opt_actor, pc.actor_lr), (self.opt_critic, pc.critic_lr)):
                for g in opt.param_groups:
                    g["lr"] = lr * max(1.0 - frac_done, 0.0)
        return self.collect() | self.update()
