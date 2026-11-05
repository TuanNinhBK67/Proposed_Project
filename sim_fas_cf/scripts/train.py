"""Train one algorithm / seed and keep the best model for inference.

python scripts/train.py --config configs/default.yaml --algo mappo --seed 1 --out runs/bench
  algos: mappo (proposed, CTDE) | ppo (centralized single agent) | ddpg | td3
         MAPPO with one variable fixed (MAPPO learns the other two): mappo_ep (equal power P_max/U) |
           mappo_rp (random SIM phases of the episode start) | mappo_rport (random ports of the episode start)
         non-learning: random | rp_ep (random phases, equal power, random ports) | rp_ep_sp (rp_ep + strongest ports)
Outputs in <out>/<algo>/seed<seed>/: config.yaml, metrics.csv (convergence), best_model.pt (best held-out
terminal metric), final_model.pt, final.json, progress.json.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

from simfas.baselines import BASELINE_ALGOS, EqualPowerInitialPolicy, make_baseline
from simfas.config import load_config, save_config
from simfas.env import SimFasEnv
from simfas.evaluate import evaluate_policy
from simfas.plotting import LivePlot
from simfas.training import EpisodeTracker
from simfas.utils import CSVLogger, atomic_json, atomic_write


def train_baseline(cfg, run_dir, device, algo):
    """No learning: run a non-learning policy (random, rp_ep, rp_ep_sp) for the same step budget as the learners
    (num_envs = ppo.num_envs), logging every episode to episodes.csv (its sum-rate level in the convergence
    figure), then evaluate it and the untouched initial configuration on the held-out scenarios."""
    tr = cfg.training
    E = cfg.ppo.num_envs
    env = SimFasEnv(cfg, E, cfg.seed, device)
    policy = make_baseline(algo, cfg.seed)
    counter = SimpleNamespace(episodes=0)
    tracker = EpisodeTracker(E, env.T, env.device)
    log_ep = CSVLogger(os.path.join(run_dir, "episodes.csv"))
    live_plot = LivePlot(run_dir, tr.plot_interval_s)
    o, steps = env.reset(), 0
    while steps < tr.total_env_steps:
        o, info = env.step(*policy(env, o))
        steps += E
        for r in tracker.step(counter, info, steps):
            log_ep.log(r)
        if steps % (E * 2000) == 0:
            atomic_json(os.path.join(run_dir, "progress.json"), {"status": "running", "algo": algo,
                                                                 "env_steps": steps, "episodes": counter.episodes})
        live_plot()
    seed = cfg.seed + tr.eval_seed_offset
    ev = evaluate_policy(cfg, make_baseline(algo, cfg.seed), tr.eval_episodes, seed, device)
    ev_init = evaluate_policy(cfg, EqualPowerInitialPolicy(), tr.eval_episodes, seed, device)
    log = CSVLogger(os.path.join(run_dir, "metrics.csv"))
    for s in (0, steps):
        log.log({"iter": 0, "env_steps": s, **{f"eval_{k}": v for k, v in ev.items() if isinstance(v, float)}})
    model = {"algo": algo, "config": cfg.to_dict(), "seed": cfg.seed, "eval": ev}
    for name in ("best_model.pt", "final_model.pt"):
        atomic_write(os.path.join(run_dir, name), lambda fn: torch.save(model, fn))
    final = {"best_" + tr.best_metric: ev[tr.best_metric], "best_eval": ev, "initial_config_eval": ev_init,
             "env_steps": steps, "episodes": counter.episodes}
    atomic_json(os.path.join(run_dir, "final.json"), final)
    atomic_json(os.path.join(run_dir, "progress.json"), {"status": "done", "algo": algo, **final})
    live_plot(force=True)
    print(f"[{algo} s{cfg.seed}] eval final sumSE {ev['final_sum_se']:.3f} reward {ev['final_reward']:.3f} | "
          f"initial config sumSE {ev_init['final_sum_se']:.3f}")
    return final


# MAPPO variants: one optimisation variable fixed by a simple rule, MAPPO learns the other two
# (mappo_sp: strongest port-gain ports of the step t-1 configuration; mappo_spn: same through the new p, phase of step t; mappo_sp_ep: strongest ports + equal power, only the phases learned;
# mappo_phc: all three learned, per-head ratio and clip)
MAPPO_VARIANTS = {"mappo_ep": ["ppo.learn_power=false"], "mappo_rp": ["ppo.learn_phase=false"],
                  "mappo_rport": ["ppo.fas_mode=fixed_random"], "mappo_sp": ["ppo.fas_mode=strongest"], "mappo_spn": ["ppo.fas_mode=strongest_new"], "mappo_spl": ["ppo.fas_mode=strongest_local"],
                  "mappo_sp_ep": ["ppo.fas_mode=strongest", "ppo.learn_power=false"],
                  "mappo_phc": ["ppo.per_head_clip=true"]}
# centralized PPO variants (one agent on the global state): ppo_sp = strongest port-gain ports of the step t-1
# configuration (as mappo_sp), ppo_spn = same through the new p, phase of step t (as mappo_spn); PPO learns p and phases
PPO_VARIANTS = {"ppo_sp": ["ppo.fas_mode=strongest"], "ppo_spn": ["ppo.fas_mode=strongest_new"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--algo", required=True, choices=["mappo", "ppo", "ddpg", "td3", *MAPPO_VARIANTS, *PPO_VARIANTS, *BASELINE_ALGOS])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="total environment steps (overrides the YAML)")
    ap.add_argument("--out", default=None, help="output root (overrides training.out_dir)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="YAML overrides, e.g. ppo.num_envs=8")
    a = ap.parse_args(argv)
    ov = MAPPO_VARIANTS.get(a.algo, []) + PPO_VARIANTS.get(a.algo, []) + list(a.set)
    if a.seed is not None:
        ov.append(f"seed={a.seed}")
    if a.steps is not None:
        ov.append(f"training.total_env_steps={a.steps}")
    cfg = load_config(a.config, [o for o in ov if not o.startswith("seed=")])
    if a.seed is not None:
        cfg.seed = a.seed
    if a.device:
        cfg.device = a.device
    out = a.out or cfg.training.out_dir
    run_dir = os.path.join(out, a.algo, f"seed{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)
    device = cfg.resolved_device()
    print(f"run {run_dir} on {device}", flush=True)
    if a.algo in BASELINE_ALGOS:
        save_config(cfg, os.path.join(run_dir, "config.yaml"))
        return train_baseline(cfg, run_dir, device, a.algo)
    if a.algo in ("mappo", "ppo", *MAPPO_VARIANTS, *PPO_VARIANTS):
        from simfas.mappo.trainer import OnPolicyTrainer
        trainer = OnPolicyTrainer(cfg, "ppo" if a.algo in ("ppo", *PPO_VARIANTS) else "mappo", run_dir, device)
    else:
        from simfas.offpolicy.trainer import OffPolicyTrainer
        trainer = OffPolicyTrainer(cfg, a.algo, run_dir, device)
    return trainer.train()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
