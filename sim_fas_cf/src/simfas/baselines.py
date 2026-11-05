"""Non-learning reference policies (same feasibility as the learned policies, own RNG)."""
import math

import torch

from . import fas, phy
from .action_maps import power_from_latent

BASELINE_ALGOS = ("random", "rp_ep", "rp_ep_sp")         # non-learning methods of the benchmark


def make_baseline(algo: str, seed: int = 0):
    return {"random": lambda: RandomPolicy(seed), "rp_ep": EqualPowerInitialPolicy,
            "rp_ep_sp": StrongestPortsPolicy}[algo]()


class RandomPolicy:
    """Random feasible action at every step: power shares ~ Dirichlet(1) over the U+1 slots (unused share
    included; env.power_map "softmax") or uniform per-feed levels a ~ U[-1, 1] through the linear map ("linear"),
    uniform SIM phases in [0, 2 pi), uniformly random distinct port sets."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.g = torch.Generator().manual_seed(int(seed))

    def __call__(self, env, o):
        E, L, U = env.E, env.L, env.U
        if env.cfg.env.power_map == "softmax":
            w = -torch.log(torch.rand(E, L, U + 1, generator=self.g, dtype=torch.float64))
            p = env.pmax * (w / w.sum(-1, keepdim=True))[..., :U]
        else:
            p = power_from_latent(2 * torch.rand(E, L, U, generator=self.g, dtype=torch.float64) - 1, env.pmax, "linear")
        phase = 2 * math.pi * torch.rand(E, L, env.M, env.N, generator=self.g, dtype=torch.float64)
        ports = fas.random_ports(E, env.nk, env.Np, env.n_max, self.g, "cpu")
        return p.to(env.device), phase.to(env.device), ports.to(env.device)


class EqualPowerInitialPolicy:
    """RP-EP (Shi et al., arXiv:2409.12870): keeps the initial configuration of the episode, i.e. random SIM
    phases, equal power P_max/U per feed, and (FAS) random port sets: the 'no optimisation' reference."""

    name = "rp_ep"

    def __call__(self, env, o):
        return env.p.clone(), env.phase.clone(), env.ports.clone()


def desired_port_power(env, p, phase) -> torch.Tensor:
    """(E, K, Np) power of UE k's own stream at every candidate port for the given powers and SIM phases."""
    return phy.desired_port_power(env.Hbar, env.Q1, env.Qinter, phase, p, env.F)


class StrongestPortsPolicy:
    """RP-EP + strongest ports: random SIM phases and equal power as in RP-EP, and every UE activates the n_k
    ports with the largest desired-signal power through the current SIM configuration."""

    name = "rp_ep_sp"

    def __call__(self, env, o):
        scores = desired_port_power(env, env.p, env.phase)
        return env.p.clone(), env.phase.clone(), fas.topk_ports(scores, env.nk, env.n_max)
