"""Inference comparison of the saved best models of the perhead_v2 benchmark on the SAME unseen test scenarios.

python scripts/compare_inference.py [--episodes 1000] [--device cuda:0]
Every method: best_model.pt, deterministic policy, terminal configuration (after T steps). The test stream is
seed + test_seed_offset, different from the held-out stream used to pick best_model.pt during training, so the
numbers are not biased by that selection. Writes to <out>:
  inference_sumrate.{png,pdf}   mean terminal sum rate (95% CI) per method
  inference_qos.{png,pdf}       QoS satisfaction ratio (UEs with R_k >= R_min) per method (95% CI)
  inference_ue_rate_cdf.{png,pdf}  CDF of the terminal per-UE rate, R_min marked
  inference_results.csv         one row per method
"""
import argparse
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from simfas.evaluate import evaluate_policy, load_policy  # noqa: E402
from simfas.plotting import METHODS  # noqa: E402

INK, GRID = "#0b0b0b", "#b9b8b3"
ROOT = "runs/bench_L6U2K4"
RUNS = ["perhead_v2_6M/mappo", "perhead_v2_6M/ppo", "perhead_v2_6M/mappo_ep", "perhead_v2_6M/mappo_rp",
        "perhead_6M/mappo_rport", "perhead_6M/mappo_sp", "perhead_6M/mappo_spn", "perhead_v2_6M/mappo_spl",
        "perhead_v2_6M/ppo_spn"]                     # same runs as scripts/plot_perhead_v2.sh
STYLE = {key: (label, color) for key, label, color in METHODS}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, out, name):
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def bars(res, metric, xlabel, out, name, fmt):
    """Horizontal bars (one per method, top = first run), mean with a 95% CI whisker, value at the bar end."""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    y = np.arange(len(res))[::-1]
    for yi, r in zip(y, res):
        v = np.asarray(r[metric])
        m, ci = v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        ax.barh(yi, m, height=0.7, color=r["color"], edgecolor="white", linewidth=1.0)
        ax.errorbar(m, yi, xerr=ci, color=INK, lw=0.7, capsize=2)
        ax.text(m + ci, yi, " " + fmt.format(m), va="center", ha="left", fontsize=6.8, color=INK)
    ax.set_yticks(y, [r["label"] for r in res])
    ax.set_xlabel(xlabel)
    lo = min(np.mean(r[metric]) for r in res)
    hi = max(np.mean(r[metric]) for r in res)
    ax.set_xlim(max(0.0, lo - 0.25 * (hi - lo)) if lo > 0.5 * hi else 0.0, hi + 0.32 * (hi - lo + 1e-9))
    ax.grid(True, axis="x", color=GRID, lw=0.5, ls=":")
    ax.set_axisbelow(True)
    save(fig, out, name)


def cdf(res, rmin, out):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for r in res:
        v = np.sort(np.asarray(r["ue_se"]).ravel())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=r["color"], lw=1.2, label=r["label"])
    ax.axvline(rmin, color=INK, lw=0.7, ls="--")
    ax.text(rmin, 1.02, r" $R^{\min}$", fontsize=7, color=INK, va="bottom", ha="center")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Per-UE rate (bps/Hz)")
    ax.set_ylabel("CDF")
    ax.grid(True, color=GRID, lw=0.5, ls=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, handlelength=2.0,
              columnspacing=1.2)
    save(fig, out, "inference_ue_rate_cdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--test-seed-offset", type=int, default=2_000_003, help="test stream = training seed + this")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "perhead_v2_6M", "inference")
    os.makedirs(out, exist_ok=True)
    style()
    res, rows, t0, seed, rmin = [], [], time.time(), None, None
    for run in RUNS:
        path = os.path.join(a.root, run, "seed1", "best_model.pt")
        pol, cfg, d = load_policy(path, a.device)
        seed = cfg.seed + a.test_seed_offset if seed is None else seed       # identical stream for every method
        rmin = cfg.system.min_se_list[0]
        r = evaluate_policy(cfg, pol, a.episodes, seed, a.device, max_parallel=250, samples=True)
        label, color = STYLE[os.path.basename(run)]
        res.append({"label": label, "color": color, "sum_se": r["final_sum_se_samples"],
                    "qos": r["final_qos_ue_frac_samples"], "ue_se": r["final_ue_se_samples"]})
        rows.append({"run": run, "label": label, "trained_steps": d.get("env_steps", ""),
                     "sum_se": r["final_sum_se"], "sum_se_std": r["final_sum_se_std"],
                     "qos_ue_frac": r["final_qos_ue_frac"], "qos_all": r["final_qos_all"],
                     "ue_se_p5": r["final_ue_se_p5"], "ue_se_p50": r["final_ue_se_p50"], "reward": r["final_reward"]})
        print(f"  {label:<48} sum {r['final_sum_se']:6.3f}  qos {r['final_qos_ue_frac']:.3f}  "
              f"all {r['final_qos_all']:.3f}  p5 {r['final_ue_se_p5']:.3f}", flush=True)
    bars(res, "sum_se", "Sum rate (bps/Hz)", out, "inference_sumrate", "{:.2f}")
    bars(res, "qos", "QoS satisfaction ratio", out, "inference_qos", "{:.3f}")
    cdf(res, rmin, out)
    with open(os.path.join(out, "inference_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{a.episodes} test episodes (seed {seed}) per method, done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
