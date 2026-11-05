"""Inference sweeps: the saved best models (trained at the default configuration) evaluated when one system
parameter changes. Only parameters that keep the network input/output sizes are swept; the number of APs L
changes the global state, so it is run for MAPPO (AP-local actor) and Random only.

python scripts/sweep_inference.py --root runs/bench --out runs/bench/inference_sweeps [--episodes 128]
Every point: held-out scenarios (training seed + eval offset, identical for every method), deterministic
policies, metrics of the configuration after the T steps. Writes sweep_<name>.{png,pdf} and sweep_results.csv.
"""
import argparse
import copy
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from simfas.baselines import RandomPolicy  # noqa: E402
from simfas.config import dims, validate  # noqa: E402
from simfas.evaluate import evaluate_policy, load_policy  # noqa: E402

INK, GRID = "#0b0b0b", "#b9b8b3"
METHODS = [("mappo", "MAPPO (three optimized variable)", "#2a78d6", "o", "-"), ("ppo", "Centralized PPO", "#eb6834", "s", "-"),
           ("ddpg", "DDPG", "#1baf7a", "^", "-"), ("td3", "TD3", "#eda100", "D", "-"),
           ("random", "Random", "#e87ba4", "v", "-")]


def set_param(cfg, name, v):
    s = cfg.system
    if name == "p_max_dbm":
        s.p_max_dbm = float(v)
    elif name == "nk":
        s.nk = int(v)
    elif name == "aperture":
        s.Wx = s.Wy = float(v)
    elif name == "area":
        s.area_m = [float(v), float(v)]
    elif name == "L":
        s.L = int(v)
    elif name == "min_se":
        s.min_se = float(v)
    validate(cfg)
    return cfg


SWEEPS = [  # name, values, x label, metric, y label
    ("p_max_dbm", [10, 15, 20, 25, 30], r"Per-AP power budget $P^{\max}$ (dBm)", "final_sum_se", "Sum rate (bps/Hz)"),
    ("nk", [1, 2, 3, 4], r"Active ports per UE $n_k$", "final_sum_se", "Sum rate (bps/Hz)"),
    ("aperture", [0.5, 1, 2, 3], r"FAS aperture $W_x = W_y$ ($\lambda$)", "final_sum_se", "Sum rate (bps/Hz)"),
    ("area", [60, 80, 100, 125, 150], "Side of the square area (m)", "final_sum_se", "Sum rate (bps/Hz)"),
    ("L", [2, 3, 4, 5], r"Number of APs $L$", "final_sum_se", "Sum rate (bps/Hz)"),
    ("min_se", [0.25, 0.5, 1.0, 1.5], r"QoS requirement $R^{\min}$ (bps/Hz)", "final_qos_ue_frac",
     "UEs meeting $R^{\\min}$ (fraction)"),
]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 6.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42, "ps.fonttype": 42})


def plot(name, values, xlabel, ylabel, res, default, out):
    style()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for key, label, color, marker, ls in METHODS:
        pts = [(v, res[(v, key)]) for v in values if (v, key) in res]
        if len(pts) < 2:                                   # e.g. centralized methods in the L sweep
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, ls=ls, lw=1.3, marker=marker, ms=4.5, mfc="white", mew=1.0, label=label)
    ax.axvline(default, color=GRID, lw=0.8, ls=":", zorder=0)        # training configuration
    ax.set_xticks(values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.5, ls=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, handlelength=2.2,
              columnspacing=1.2)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"sweep_{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/bench")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1, help="training seed of the runs to load")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", nargs="*", default=None, help="subset of sweeps, e.g. p_max_dbm nk")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "inference_sweeps")
    os.makedirs(out, exist_ok=True)
    policies, base_cfg = {}, None
    for key, *_ in METHODS[:4]:
        path = os.path.join(a.root, key, f"seed{a.seed}", "best_model.pt")
        if os.path.exists(path):
            pol, cfg, _ = load_policy(path, a.device)
            policies[key] = pol
            base_cfg = base_cfg or cfg
    if base_cfg is None:
        raise SystemExit(f"no best_model.pt under {a.root}")
    base_dims = dims(base_cfg)
    seed = base_cfg.seed + base_cfg.training.eval_seed_offset
    rows, t0 = [], time.time()
    for name, values, xlabel, metric, ylabel in SWEEPS:
        if a.only and name not in a.only:
            continue
        default = {"p_max_dbm": base_cfg.system.p_max_dbm, "nk": base_cfg.system.nk_list[0], "aperture": base_cfg.system.Wx,
                   "area": base_cfg.system.area_m[0], "L": base_cfg.system.L, "min_se": base_cfg.system.min_se_list[0]}[name]
        res = {}
        for v in values:
            cfg = set_param(copy.deepcopy(base_cfg), name, v)
            d = dims(cfg)
            cands = dict(policies) | {"random": RandomPolicy(base_cfg.seed)}
            for key, pol in cands.items():
                if key in ("ppo", "ddpg", "td3") and d["state"] != base_dims["state"]:
                    continue                                # centralized input size changed (L): not applicable
                if key == "mappo" and d["obs"] != base_dims["obs"]:
                    continue
                r = evaluate_policy(cfg, pol, a.episodes, seed, a.device)
                res[(v, key)] = r[metric]
                rows.append({"sweep": name, "value": v, "method": key, "final_sum_se": r["final_sum_se"],
                             "final_sum_se_std": r["final_sum_se_std"], "final_reward": r["final_reward"],
                             "final_qos_ue_frac": r["final_qos_ue_frac"], "final_qos_all": r["final_qos_all"],
                             "final_shortfall": r["final_shortfall"], "final_ue_se_p5": r["final_ue_se_p5"]})
            print(f"  {name}={v:<6} " + " ".join(f"{k}:{res[(v, k)]:.3f}" for k, *_ in METHODS if (v, k) in res),
                  flush=True)
        plot(name, values, xlabel, ylabel, res, default, out)
    path = os.path.join(out, "sweep_results.csv")
    if a.only and os.path.exists(path):                    # keep the rows of the sweeps not re-run
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["sweep"] not in a.only] + rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
