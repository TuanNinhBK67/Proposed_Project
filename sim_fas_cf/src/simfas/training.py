"""Shared training driver for every learner (MAPPO, PPO, DDPG, TD3)."""
import math
import os
import time

import torch

from .plotting import LivePlot
from .utils import CSVLogger, atomic_json

EPISODE_KEYS = ("ep_sum_se", "ep_reward", "ep_qos_satisfied", "return", "final_sum_se", "final_reward",
                "final_qos_all", "final_shortfall")


class EpisodeTracker:
    """Per-environment accumulators of the training episodes (stochastic / exploring policy). One row per
    finished episode: episode-mean sum SE, reward and per-UE QoS satisfaction over the T steps (ep_*), the
    undiscounted return, and the terminal-step metrics (final_*)."""

    def __init__(self, E, T, device):
        self.T = T
        self.acc = {k: torch.zeros(E, dtype=torch.float64, device=device) for k in ("se", "r", "qos")}

    def step(self, trainer, info, env_steps_after) -> list:
        self.acc["se"] += info["sum_se"].double()
        self.acc["r"] += info["reward"].double()
        self.acc["qos"] += info["qos_ue_frac"].double()
        done = info["terminated"]
        rows = []
        for i in torch.nonzero(done).flatten().tolist():
            trainer.episodes += 1
            rows.append({"episode": trainer.episodes, "env_steps": env_steps_after,
                         "ep_sum_se": self.acc["se"][i].item() / self.T, "ep_reward": self.acc["r"][i].item() / self.T,
                         "ep_qos_satisfied": self.acc["qos"][i].item() / self.T, "return": self.acc["r"][i].item(),
                         "final_sum_se": info["sum_se"][i].item(), "final_reward": info["reward"][i].item(),
                         "final_qos_all": info["qos_all"][i].item(), "final_shortfall": info["shortfall"][i].item()})
        for v in self.acc.values():
            v[done] = 0.0
        return rows


def summarise_episodes(rows: list) -> dict:
    return {f"train_{k}": sum(r[k] for r in rows) / len(rows) for k in EPISODE_KEYS} if rows else {}


def run_training_loop(trainer, steps_per_iter: int, iteration) -> dict:
    """Shared driver: iterate, evaluate on held-out scenarios every eval_interval environment steps (and at
    step 0), keep the best model (training.best_metric of the terminal configuration), log metrics.csv (one
    row per iteration) and episodes.csv (one row per finished training episode), and redraw the benchmark
    convergence figure of the output root every training.plot_interval_s seconds."""
    cfg, rd = trainer.cfg, trainer.run_dir
    tr = cfg.training
    metrics = CSVLogger(os.path.join(rd, "metrics.csv"))
    episodes = CSVLogger(os.path.join(rd, "episodes.csv"))
    live_plot = LivePlot(rd, tr.plot_interval_s)
    best_eval = {}
    n_iter = math.ceil(tr.total_env_steps / steps_per_iter)
    key = tr.best_metric
    t0 = time.time()

    def do_eval(row):
        ev = trainer.evaluate()
        row.update({f"eval_{k}": v for k, v in ev.items() if isinstance(v, float)})
        if ev[key] > trainer.best:
            trainer.best = ev[key]
            best_eval.clear()
            best_eval.update({k: v for k, v in ev.items() if isinstance(v, (int, float))}, env_steps=trainer.env_steps)
            trainer.export_model(os.path.join(rd, "best_model.pt"), ev)
            row["new_best"] = 1
        return ev

    row0 = {"iter": 0, "env_steps": 0, "wall_s": 0.0}
    do_eval(row0)
    metrics.log(row0)
    next_eval = tr.eval_interval
    for it in range(n_iter):
        row = {"iter": it + 1}
        res = iteration(it / n_iter)
        for r in res.pop("finished", []):
            episodes.log(r)
        row.update(res)
        row["env_steps"] = trainer.env_steps
        if trainer.env_steps >= next_eval or it == n_iter - 1:
            do_eval(row)
            next_eval = trainer.env_steps + tr.eval_interval
        row["wall_s"] = round(time.time() - t0, 1)
        metrics.log(row)
        el = time.time() - t0
        atomic_json(os.path.join(rd, "progress.json"), {
            "status": "running", "algo": trainer.algo, "iter": it + 1, "iters_total": n_iter,
            "env_steps": trainer.env_steps, "best": trainer.best, "elapsed_s": round(el, 1),
            "eta_s": round(el / (it + 1) * (n_iter - it - 1), 1)})
        live_plot()
        if it == 0 or (it + 1) % 20 == 0 or it == n_iter - 1:
            msg = f"[{trainer.algo} s{cfg.seed}] it {it + 1}/{n_iter} steps {trainer.env_steps}"
            if "train_final_sum_se" in row:
                msg += f" | train final sumSE {row['train_final_sum_se']:.3f}"
            if "eval_final_sum_se" in row:
                msg += f" | eval final sumSE {row['eval_final_sum_se']:.3f} reward {row['eval_final_reward']:.3f}"
            print(msg + f" | best {trainer.best:.3f} | {el:.0f}s", flush=True)
    trainer.export_model(os.path.join(rd, "final_model.pt"))
    final = {"best_" + key: trainer.best, "env_steps": trainer.env_steps, "episodes": trainer.episodes,
             "wall_s": round(time.time() - t0, 1), "best_eval": best_eval}
    atomic_json(os.path.join(rd, "final.json"), final)
    atomic_json(os.path.join(rd, "progress.json"), {"status": "done", "algo": trainer.algo, **final})
    live_plot(force=True)
    return final
