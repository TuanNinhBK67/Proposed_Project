"""Run a saved model (best_model.pt / final_model.pt of any algorithm) on held-out scenarios.

python scripts/inference.py --model runs/bench/mappo/seed1/best_model.pt --episodes 200
Reports the terminal-step metrics (after T optimisation steps) and the per-step refinement curve, optionally
dumps one episode's final configuration (powers, phases, ports) to JSON. --seed changes the scenario stream
(default: the training run's held-out evaluation seed, identical for every algorithm).
"""
import argparse
import json
import os

import torch

from simfas.env import SimFasEnv
from simfas.evaluate import evaluate_policy, load_policy


def dump_episode(cfg, policy, seed, device, path):
    env = SimFasEnv(cfg, 1, seed, device)
    o = env.reset()
    traj = []
    for t in range(env.T):
        p, phase, ports = policy(env, o)
        o, info = env.step(p, phase, ports)
        traj.append({"step": t + 1, "sum_se": info["sum_se"].item(), "reward": info["reward"].item(),
                     "ue_se": info["rates"][0].tolist()})
    final = {"power_w": p[0].tolist(), "phase_rad": phase[0].tolist(),
             "ports_0based": [[int(v) for v in row if v >= 0] for row in ports[0].tolist()]}
    with open(path, "w") as f:
        json.dump({"trajectory": traj, "final_configuration": final}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stochastic", action="store_true", help="sample from MAPPO/PPO policies instead of the mode")
    ap.add_argument("--dump-episode", action="store_true")
    a = ap.parse_args()
    policy, cfg, d = load_policy(a.model, a.device, deterministic=not a.stochastic)
    seed = a.seed if a.seed is not None else cfg.seed + cfg.training.eval_seed_offset
    r = evaluate_policy(cfg, policy, a.episodes, seed, a.device)
    print(f"{d['algo']} ({os.path.basename(a.model)}, trained {d.get('env_steps', 0)} env steps) | "
          f"{a.episodes} episodes x T={cfg.env.episode_steps}")
    for k in ("final_sum_se", "final_reward", "final_qos_ue_frac", "final_qos_all", "final_shortfall",
              "final_min_se", "final_ue_se_p5", "final_power_frac"):
        std = r.get(k + "_std")
        print(f"  {k:<20} {r[k]:8.4f}" + (f" ± {std:.4f}" if std is not None else ""))
    curve = r["curve_sum_se"]
    print("  sum SE over the T steps:", " ".join(f"{v:.2f}" for v in curve[:: max(1, len(curve) // 10)]))
    base = os.path.splitext(a.model)[0]
    with open(base + "_inference.json", "w") as f:
        json.dump({"model": a.model, "seed": seed, **r}, f, indent=2)
    if a.dump_episode:
        dump_episode(cfg, policy, seed, a.device, base + "_episode.json")
    print("saved", base + "_inference.json")


if __name__ == "__main__":
    with torch.no_grad():
        main()
