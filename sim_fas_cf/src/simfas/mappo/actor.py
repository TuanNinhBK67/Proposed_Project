"""Hybrid continuous-discrete actors and the centralized critic.

ActorGroup handles the three on-policy layouts with one agent axis A:
  MAPPO, shared actor     : A = L agents, one network, local observation o_l, n_ap = 1 AP per agent
  MAPPO, separate actors  : A = L agents, L networks
  PPO (centralized)       : A = 1 agent, one network on the global state, n_ap = L APs per agent
Each agent outputs Gaussian(z_power (n_ap, PU)) with PU = U+1 (env.power_map "softmax") or U ("linear"),
Gaussian(z_phase (n_ap, M N)) and FAS logits (K, Np);
the FAS logits of a UE are used only if the agent controls it (active mask).
"""
import math
import torch
import torch.nn as nn
from .distributions import fas_evaluate, fas_sample, gaussian_entropy, gaussian_log_prob

def layer_init(layer, std=math.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

def mlp_trunk(in_dim, hidden):
    layers, d = [], in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
        d = h
    return nn.Sequential(*layers), d

class HybridPolicy(nn.Module):
    def __init__(self, in_dim, n_ap, U, M, N, K, Np, hidden, init_log_std, ue_index=None, fas_hidden=128,
                 power_dims=None):
        """ue_index (K, d_ue) positions of each UE's features in x -> per-UE FAS head (logits of UE k from
        [trunk output, features of UE k], one MLP shared over UEs); None -> flat linear K*Np head."""
        super().__init__()
        self.n_ap, self.U, self.MN, self.K, self.Np = n_ap, U, M * N, K, Np
        self.PU = U + 1 if power_dims is None else power_dims
        self.body, h = mlp_trunk(in_dim, hidden)
        self.power_mean = layer_init(nn.Linear(h, n_ap * self.PU), std=0.01)
        self.phase_mean = layer_init(nn.Linear(h, n_ap * M * N), std=0.01)
        if ue_index is None:
            self.ue_index = None
            self.fas_logits = layer_init(nn.Linear(h, K * Np), std=0.01)
        else:
            self.register_buffer("ue_index", torch.as_tensor(ue_index, dtype=torch.long))
            self.fas_logits = nn.Sequential(layer_init(nn.Linear(h + self.ue_index.shape[1], fas_hidden)), nn.Tanh(),
                                            layer_init(nn.Linear(fas_hidden, Np), std=0.01))
        self.power_log_std = nn.Parameter(torch.full((n_ap, self.PU), float(init_log_std)))
        self.phase_log_std = nn.Parameter(torch.full((n_ap, M * N), float(init_log_std)))

    def heads(self, x):
        h = self.body(x)
        b = x.shape[:-1]
        if self.ue_index is None:
            logits = self.fas_logits(h).view(*b, self.K, self.Np)
        else:
            ue = x[..., self.ue_index]                                          # (*b, K, d_ue)
            logits = self.fas_logits(torch.cat([h[..., None, :].expand(*b, self.K, h.shape[-1]), ue], -1))
        return (self.power_mean(h).view(*b, self.n_ap, self.PU), self.phase_mean(h).view(*b, self.n_ap, self.MN),
                logits)

class ActorGroup(nn.Module):
    def __init__(self, in_dim, n_agents, n_ap, shared, U, M, N, K, Np, hidden, init_log_std,
                 learn_power=True, learn_phase=True, ue_index=None, fas_hidden=128, power_dims=None):
        super().__init__()
        self.A, self.n_ap = n_agents, n_ap
        self.learn_power, self.learn_phase = learn_power, learn_phase
        n_nets = 1 if shared else n_agents
        self.nets = nn.ModuleList([HybridPolicy(in_dim, n_ap, U, M, N, K, Np, hidden, init_log_std, ue_index,
                                                fas_hidden, power_dims) for _ in range(n_nets)])

    def _heads(self, x):
        """x (B, A, d) -> means (B, A, n_ap, PU), (B, A, n_ap, MN), logits (B, A, K, Np)."""
        if len(self.nets) == 1:
            return self.nets[0].heads(x)
        outs = [net.heads(x[:, a]) for a, net in enumerate(self.nets)]
        return tuple(torch.stack(o, 1) for o in zip(*outs))

    def _log_stds(self):
        """(A or 1, n_ap, dim) for broadcasting over (B, A, n_ap, dim)."""
        st = lambda name: torch.stack([getattr(n, name) for n in self.nets], 0)
        return st("power_log_std"), st("phase_log_std")

    def _gauss(self, zp, mp, zf, mf):
        lsp, lsf = self._log_stds()
        B, A = mp.shape[:2]
        zero = torch.zeros(B, A, device=mp.device)
        lp = gaussian_log_prob(zp, mp, lsp) if self.learn_power else zero
        lf = gaussian_log_prob(zf, mf, lsf) if self.learn_phase else zero
        ep = gaussian_entropy(lsp).expand(B, A) if self.learn_power else zero
        ef = gaussian_entropy(lsf).expand(B, A) if self.learn_phase else zero
        return lp, lf, ep, ef

    def act(self, x, active, deterministic=False, generator=None):
        """x (B, A, d), active (B, A, K, n) -> latent actions, ordered draws, joint log-prob and entropies."""
        mp, mf, logits = self._heads(x)
        lsp, lsf = self._log_stds()
        if deterministic:
            zp, zf = mp, mf
        else:
            zp = mp + lsp.exp() * torch.randn(mp.shape, device=mp.device, generator=generator)
            zf = mf + lsf.exp() * torch.randn(mf.shape, device=mf.device, generator=generator)
        B, A, K, Np = logits.shape
        draws, lq, eq = fas_sample(logits.reshape(B * A, K, Np), active.reshape(B * A, K, -1), deterministic,
                                   generator)
        lp, lf, ep, ef = self._gauss(zp, mp, zf, mf)
        lq, eq = lq.view(B, A), eq.view(B, A)
        return {"z_power": zp, "z_phase": zf, "draws": draws.view(B, A, K, -1), "logp": lp + lf + lq,
                "logp_parts": torch.stack([lp, lf, lq], -1), "ent_power": ep, "ent_phase": ef, "ent_fas": eq}

    def evaluate(self, x, active, z_power, z_phase, draws):
        """Re-score stored latent actions and ordered draws under the current parameters (same masks)."""
        mp, mf, logits = self._heads(x)
        B, A, K, Np = logits.shape
        lq, eq = fas_evaluate(logits.reshape(B * A, K, Np), active.reshape(B * A, K, -1), draws.reshape(B * A, K, -1))
        lp, lf, ep, ef = self._gauss(z_power, mp, z_phase, mf)
        lq = lq.view(B, A)
        return {"logp": lp + lf + lq, "logp_parts": torch.stack([lp, lf, lq], -1), "ent_power": ep, "ent_phase": ef,
                "ent_fas": eq.view(B, A)}

    def mean_std(self):
        lsp, lsf = self._log_stds()
        return lsp.exp().mean().item(), lsf.exp().mean().item()

class Critic(nn.Module):
    """V_psi(s_t): one scalar per environment time step (training only)."""

    def __init__(self, state_dim, hidden):
        super().__init__()
        body, h = mlp_trunk(state_dim, hidden)
        self.net = nn.Sequential(body, layer_init(nn.Linear(h, 1), std=1.0))

    def forward(self, s):
        return self.net(s).squeeze(-1)

def build_actor(cfg, algo: str, in_dim: int) -> ActorGroup:
    from ..config import power_dims, ue_feature_index
    s, p = cfg.system, cfg.ppo
    central = algo == "ppo"
    ue_index = ue_feature_index(cfg, central) if p.fas_head == "per_ue" else None
    return ActorGroup(in_dim, 1 if central else s.L, s.L if central else 1, True if central else p.shared_actor,
                      s.U, s.M, s.N, s.K, s.Np, list(p.hidden), p.init_log_std, p.learn_power, p.learn_phase,
                      ue_index, p.fas_head_hidden, power_dims(cfg))
