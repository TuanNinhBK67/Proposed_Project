"""Policy evaluation on held-out scenarios (identical for every algorithm) and model loading for inference.

A policy is a callable (env, obs_dict) -> (p (E,L,U), phase (E,L,M,N), ports (E,K,n_max)). Policies keep their
own random generators, so the environment's scenario stream (geometry, shadowing, fading, initial configuration)
depends only on the evaluation seed. Reported numbers are those of the TERMINAL configuration (step T) of each
episode; episode means and the per-step refinement curve are reported as well.
"""
import math

import torch

from .env import SimFasEnv

STEP_METRICS = ("sum_se", "reward", "qos_ue_frac", "qos_all", "shortfall", "min_se", "power_frac")


@torch.no_grad()
def evaluate_policy(cfg, policy, episodes: int, seed: int, device="cpu", max_parallel: int = 32,
                    samples: bool = False) -> dict:
    """samples: also return the per-episode terminal metrics (final_<k>_samples) and the terminal per-UE rates
    (final_ue_se_samples, episodes x K) as lists."""
    E = max(1, min(episodes, max_parallel))
    env = SimFasEnv(cfg, E, seed, device)
    o = env.reset()
    T = env.T
    final = {k: [] for k in STEP_METRICS}
    mean = {k: [] for k in STEP_METRICS}
    curve_se, curve_r, ep_ret, rates = [], [], [], []
    for _ in range(math.ceil(episodes / E)):
        steps = []
        for _ in range(T):
            p, phase, ports = policy(env, o)
            o, info = env.step(p, phase, ports)
            steps.append({k: info[k].double() for k in STEP_METRICS} | {"rates": info["rates"].double()})
        for k in STEP_METRICS:
            final[k].append(steps[-1][k])
            mean[k].append(torch.stack([s[k] for s in steps]).mean(0))
        curve_se.append(torch.stack([s["sum_se"] for s in steps], 1))        # (E, T)
        curve_r.append(torch.stack([s["reward"] for s in steps], 1))
        ep_ret.append(torch.stack([s["reward"] for s in steps], 1).sum(1))
        rates.append(steps[-1]["rates"])
    cut = lambda xs: torch.cat(xs)[:episodes]
    out = {"episodes": episodes}
    for k in STEP_METRICS:
        v = cut(final[k])
        out[f"final_{k}"] = v.mean().item()
        out[f"final_{k}_std"] = v.std().item() if v.numel() > 1 else 0.0
        out[f"mean_{k}"] = cut(mean[k]).mean().item()
    out["episode_return"] = cut(ep_ret).mean().item()
    r = cut(rates)
    out["final_ue_se_p5"] = torch.quantile(r.flatten().float(), 0.05).item()
    out["final_ue_se_p50"] = torch.quantile(r.flatten().float(), 0.50).item()
    out["curve_sum_se"] = cut(curve_se).mean(0).tolist()
    out["curve_reward"] = cut(curve_r).mean(0).tolist()
    if samples:
        for k in STEP_METRICS:
            out[f"final_{k}_samples"] = cut(final[k]).tolist()
        out["final_ue_se_samples"] = r.tolist()
    return out


def load_policy(path: str, device="cpu", deterministic: bool = True):
    """Rebuild the inference policy saved by any trainer (best_model.pt / final_model.pt)."""
    d = torch.load(path, map_location=device, weights_only=False)
    from .config import config_from_dict
    cfg = config_from_dict(d["config"])
    algo = d["algo"]
    if algo in ("mappo", "ppo"):
        from .mappo.trainer import OnPolicyAgent
        return OnPolicyAgent.from_saved(d, cfg, device, deterministic), cfg, d
    if algo in ("ddpg", "td3"):
        from .offpolicy.trainer import OffPolicyAgent
        return OffPolicyAgent.from_saved(d, cfg, device), cfg, d
    from .baselines import BASELINE_ALGOS, make_baseline
    if algo in BASELINE_ALGOS:
        return make_baseline(algo, d.get("seed", 0)), cfg, d
    raise ValueError(f"unknown algo '{algo}' in {path}")
