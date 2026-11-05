"""Deterministic maps from sampled (latent) actions to feasible physical actions.

Stochastic policies (MAPPO / PPO), manuscript power and phase maps:
  env.power_map = "softmax":
  p_{l,u}   = P_max * softmax(z^p_l)_u,  u = 1..U, from U+1 logits; share U+1 is unused power
             -> p >= 0 and sum_u p_{l,u} <= P_max by construction (many-to-one: PPO scores the latent z^p)
  env.power_map = "linear" (one independent level per feed, U entries):
  p_{l,u}   = P_max * (clip(z^p_{l,u}, -1, 1) + 1) / 2, then p_l <- p_l * P_max / sum_u p_{l,u} if the sum
             exceeds P_max -> p >= 0 and sum_u p_{l,u} <= P_max by construction, unused power allowed
  phi_{l,m}^n = 2 pi sigmoid(z^phi_{l,m,n})
Deterministic actor-critic baselines (DDPG / TD3) output a in [-1, 1]^dim:
  power logits = power_scale * a ("softmax") or levels a ("linear"),   phi = pi (a + 1),
  ports = top-n_k of the K x Np score block.
"""
import math
import torch
from .fas import topk_ports

def power_from_latent(z: torch.Tensor, pmax: float, power_map: str = "softmax") -> torch.Tensor:
    """z (..., U+1) ("softmax") or (..., U) ("linear") -> p (..., U) in W, sum over the last dim <= pmax."""
    z = z.to(torch.float64)
    if power_map == "softmax":
        return pmax * torch.softmax(z, dim=-1)[..., :-1]
    p = pmax * (z.clamp(-1.0, 1.0) + 1.0) / 2.0
    total = p.sum(-1, keepdim=True)
    return p * torch.where(total > pmax, pmax / total.clamp_min(1e-300), torch.ones_like(total))

def phase_from_latent(z: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """z (..., M*N) -> phase (..., M, N) in [0, 2 pi]."""
    return (2 * math.pi * torch.sigmoid(z.to(torch.float64))).reshape(z.shape[:-1] + (M, N))

def split_unit_action(a: torch.Tensor, L: int, U: int, M: int, N: int, K: int, Np: int, power_map: str = "softmax"):
    """a (B, L PU + LMN + K Np) in [-1, 1] with PU = U+1 ("softmax") or U ("linear") -> (power block (B, L, PU),
    phase block (B, L, MN), scores (B, K, Np))."""
    B, PU = a.shape[0], U + (1 if power_map == "softmax" else 0)
    i1, i2 = L * PU, L * PU + L * M * N
    return a[:, :i1].view(B, L, PU), a[:, i1:i2].view(B, L, M * N), a[:, i2:].view(B, K, Np)

def unit_action_dim(L, U, M, N, K, Np, power_map: str = "softmax") -> int:
    return L * (U + (1 if power_map == "softmax" else 0)) + L * M * N + K * Np

def physical_from_unit(a, cfg, pmax, power_scale, nk_list=None, n_max=None):
    """Deterministic [-1, 1] action -> (p (B, L, U), phase (B, L, M, N), ports (B, K, n_max)).
    nk_list / n_max default to cfg; pass the environment's values to run a policy with other n_k."""
    s = cfg.system
    pm = cfg.env.power_map
    ap, aphi, sc = split_unit_action(a, s.L, s.U, s.M, s.N, s.K, s.Np, pm)
    p = power_from_latent(power_scale * ap if pm == "softmax" else ap, pmax, pm)
    phase = (math.pi * (aphi.to(torch.float64) + 1.0)).reshape(a.shape[0], s.L, s.M, s.N)
    nk = s.nk_list if nk_list is None else nk_list
    return p, phase, topk_ports(sc, nk, max(nk) if n_max is None else n_max)
