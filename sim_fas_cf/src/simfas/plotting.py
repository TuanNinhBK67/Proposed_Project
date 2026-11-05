"""Convergence figure: sum rate of the training episodes vs. environment steps, one line per method
(MAPPO, its variants with one variable fixed (equal power / random phases / random ports), PPO, DDPG, TD3: mean
over seeds of a moving average over episodes; the non-learning Random: dashed horizontal line at its mean over all
logged episodes; rp_ep / rp_ep_sp runs are not drawn).

Data: <root>/<algo>/seed<k>/episodes.csv (one row per finished training episode, written by every run).
Outputs in <root>/figures (or `out`):
  convergence_sumrate.png / .pdf   the figure
  convergence_sumrate.csv          the plotted curves in wide format: steps | one column per method
                                   (+ "<method> std" columns when a method has several seeds)
Used by scripts/plot_sumrate.py and, during training, by every run itself (LivePlot), so the figure is
refreshed while the benchmark is running. Files are written atomically: concurrent runs may redraw at once.
"""
import csv
import glob
import os
import time

import numpy as np

from .utils import atomic_write

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#b9b8b3"
# fixed categorical order of the validated palette (slots 1-8: the palette's maximum)
METHODS = [("mappo", "MAPPO (three optimized variable)", "#2a78d6"), ("ppo", "Centralized PPO", "#eb6834"), ("ddpg", "DDPG", "#1baf7a"),
           ("td3", "TD3", "#eda100"), ("random", "Random", "#e87ba4"), ("mappo_ep", "Equal power allocation", "#008300"),
           ("mappo_rp", "Random phase shift", "#6250d6"), ("mappo_rport", "Random FAS ports", "#e34948"),
           ("mappo_sp", "Proposed arch 1", "#8a5a2b"),   # slots 9-10: outside the palette
           ("mappo_spn", "Proposed arch 2", "#2b2b2b"),
           ("mappo_spl", "Proposed arch 3", "#00a6b5"),
           ("ppo_sp", "Centralized PPO based on port-gain (step t-1)", "#b5651d"),
           ("ppo_spn", "Centralized PPO based on port-gain", "#7b3294"),
           ("mappo_sp_ep", "Proposed arch 1, equal power", "#a0a000"),
           ("mappo_phc", "MAPPO, per-head clip", "#800020")]
YLABELS = {"ep_sum_se": "Sum rate (bps/Hz)", "final_sum_se": "Sum rate after T steps (bps/Hz)",
           "ep_reward": "Reward", "ep_qos_satisfied": "QoS satisfaction ratio"}
# non-learning methods: drawn as a dashed horizontal line at their mean over all logged episodes
STATIC = {"random", "rp_ep", "rp_ep_sp"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42, "ps.fonttype": 42})
    return matplotlib, plt


def read_episodes(path, metric):
    """(env_steps, metric) of every complete row; a row being appended by a running job is skipped."""
    xs, ys = [], []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    xs.append(float(row["env_steps"]))
                    ys.append(float(row[metric]))
                except (KeyError, TypeError, ValueError):
                    continue
    except FileNotFoundError:
        pass
    return np.asarray(xs), np.asarray(ys)


def smooth(y, w):
    """Trailing moving average over episodes; the first w-1 points average all episodes so far."""
    if w <= 1 or len(y) < 2:
        return y
    c = np.cumsum(np.insert(np.asarray(y, dtype=np.float64), 0, 0.0))
    idx = np.arange(1, len(y) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def draw(root, out=None, metric="ep_sum_se", smooth_w=5000, band=False, verbose=True, n_points=400,
         methods=None, name=None) -> bool:
    """Redraw the figure and its CSV from every run under `root`. Returns False if there is nothing to plot.
    methods: [(run sub-path under root, label, color)] (default METHODS); name: output file stem."""
    out = out or os.path.join(root, "figures")
    series, x_max = [], 0.0
    for key, label, color in (methods or METHODS):
        curves = []
        for d in sorted(glob.glob(os.path.join(root, key, "seed*"))):
            x, y = read_episodes(os.path.join(d, "episodes.csv"), metric)
            if len(y) >= 2:
                curves.append((x, np.full(len(y), y.mean()) if key in STATIC else smooth(y, smooth_w)))
        if not curves:
            continue
        start, end = max(c[0][0] for c in curves), min(c[0][-1] for c in curves)   # common range of the seeds
        if key in STATIC:
            start = 0.0
        series.append((key, label, color, curves, start, end))
        x_max = max(x_max, end)
    if not series:
        if verbose:
            print("nothing to plot yet", flush=True)
        return False

    matplotlib, plt = _pyplot()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    grid = np.linspace(0.0, x_max, n_points)
    header, cols = ["steps"], [grid]
    for key, label, color, curves, start, end in series:
        g = grid[(grid >= start) & (grid <= end)]
        Y = np.stack([np.interp(g, x, y) for x, y in curves])
        mu, sd = Y.mean(0), Y.std(0)
        if band and len(curves) > 1:
            ax.fill_between(g, mu - sd, mu + sd, color=color, alpha=0.12, lw=0)
        ax.plot(g, mu, color=color, lw=1.3, ls="--" if key in STATIC else "-", label=label)
        inside = (grid >= start) & (grid <= end)
        col = np.full(n_points, np.nan)
        col[inside] = mu
        header.append(label), cols.append(col)
        if len(curves) > 1:
            col = np.full(n_points, np.nan)
            col[inside] = sd
            header.append(f"{label} std"), cols.append(col)
        if verbose:
            print(f"  {key:<7} {len(curves)} seed(s) up to {int(end)} steps, last value {mu[-1]:.3f}")
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Steps")
    ax.set_ylabel(YLABELS.get(metric, metric))
    ax.grid(True, color=GRID, lw=0.5, ls=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, handlelength=2.0,
              columnspacing=1.2)                          # below the axes: never hides a curve
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else (f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k")))
    fig.tight_layout(pad=0.3)
    os.makedirs(out, exist_ok=True)
    name = os.path.join(out, name or ("convergence_sumrate" if metric == "ep_sum_se" else f"convergence_{metric}"))
    for ext in ("pdf", "png"):
        atomic_write(f"{name}.{ext}", lambda fn: fig.savefig(fn, dpi=300, bbox_inches="tight"))
    plt.close(fig)

    def write_csv(fn):
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(n_points):
                w.writerow([int(round(grid[i]))] + ["" if np.isnan(c[i]) else f"{c[i]:.5f}" for c in cols[1:]])
    atomic_write(f"{name}.csv", write_csv)
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] wrote {name}.png/.pdf/.csv", flush=True)
    return True


class LivePlot:
    """Redraws the benchmark figure of the output root (<root>/<algo>/seed<k> -> <root>) every `interval_s`
    seconds of training (0 = off) and once more at the end. A plotting error never stops training."""

    def __init__(self, run_dir, interval_s):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(run_dir))) if run_dir else None
        self.interval, self.last = interval_s, time.time()

    def __call__(self, force=False):
        if self.root is None or self.interval <= 0 or (not force and time.time() - self.last < self.interval):
            return
        self.last = time.time()
        try:
            draw(self.root, verbose=False)
        except Exception as exc:
            print(f"live plot failed: {exc}", flush=True)
