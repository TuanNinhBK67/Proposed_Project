"""Convergence figures and summary table of a benchmark root (<root>/<algo>/seed*/metrics.csv).

python scripts/plot_convergence.py --root runs/bench --out runs/bench/figures
Panels (one y-axis each): held-out terminal sum SE, terminal penalised reward, all-UE QoS satisfaction
probability, training episode return; mean over seeds with a +-1 std band. 'random' is drawn as a horizontal
reference. Also writes summary.csv / summary.md (table view of the same numbers).
"""
import argparse
import csv
import glob
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["mappo", "ppo", "ddpg", "td3"]                      # fixed categorical order (never cycled)
LABEL = {"mappo": "MAPPO (three optimized variable)", "ppo": "Centralized PPO", "ddpg": "DDPG", "td3": "TD3", "random": "Random"}
COLOR = {"mappo": "#2a78d6", "ppo": "#eb6834", "ddpg": "#1baf7a", "td3": "#eda100"}   # validated palette slots 1-4
REF_COLOR, INK, INK2, GRID, SURFACE = "#52514e", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PANELS = [("eval_final_sum_se", "Sum SE after T steps [bit/s/Hz]"),
          ("eval_final_reward", "Penalised reward after T steps"),
          ("eval_final_qos_all", "All-UE QoS satisfaction probability"),
          ("train_return", "Training episode return")]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def series(rows, key):
    pts = [(float(r["env_steps"]), float(r[key])) for r in rows if r.get(key) not in (None, "")]
    return np.array(pts).T if pts else None


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([y[: k - 1].cumsum() / np.arange(1, k), out])


def aggregate(runs, key, smooth_k=1):
    """Interpolate every seed on a common grid (up to the shortest run) -> grid, mean, std, n."""
    curves = [s for s in (series(r, key) for r in runs) if s is not None and len(s[0]) > 1]
    if not curves:
        return None
    x_end = min(c[0][-1] for c in curves)
    grid = np.linspace(0.0, x_end, 200)
    ys = np.stack([np.interp(grid, c[0], smooth(c[1], smooth_k)) for c in curves])
    return grid, ys.mean(0), ys.std(0), len(curves)


def load(root):
    out = {}
    for algo in ORDER + ["random"]:
        runs = [read_csv(p) for p in sorted(glob.glob(os.path.join(root, algo, "seed*", "metrics.csv")))]
        runs = [r for r in runs if r]
        if runs:
            out[algo] = runs
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def plot(data, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, PANELS):
        style_axes(ax)
        ends = []
        for algo in ORDER:
            if algo not in data:
                continue
            agg = aggregate(data[algo], key, smooth_k=25 if key == "train_return" else 1)
            if agg is None:
                continue
            x, m, s, n = agg
            ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.10, linewidth=0)
            ax.plot(x, m, color=COLOR[algo], linewidth=1.6, solid_capstyle="round", label=LABEL[algo])
            ends.append((m[-1], x[-1], algo))
        if "random" in data and key != "train_return":
            ref = series(data["random"][0], key)
            if ref is not None:
                vals = [series(r, key)[1][0] for r in data["random"] if series(r, key) is not None]
                ax.axhline(np.mean(vals), color=REF_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), label=LABEL["random"])
        if key == PANELS[0][0] and 0 < len(ends) <= 4:            # direct end labels on the headline panel
            ends.sort()
            lo, hi = ax.get_ylim()
            gap, last = 0.06 * (hi - lo), -np.inf
            for y, xe, algo in ends:
                y = max(y, last + gap)
                last = y
                ax.annotate(LABEL[algo].split(" ")[0], (xe, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Environment steps", fontsize=8.5, color=INK2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
    seen = {}
    for ax in axes.flat:                                          # union over panels, fixed series order
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    labels = [LABEL[a] for a in ORDER + ["random"] if LABEL[a] in seen]
    handles = [seen[lab] for lab in labels]
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"convergence.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def summary(data, out_dir):
    """Table view: per algorithm, mean +- std over seeds of the best and the last held-out evaluation."""
    rows = []
    for algo in ORDER + ["random"]:
        if algo not in data:
            continue
        best_se, last_se, best_r, last_qos, steps90, wall = [], [], [], [], [], []
        for r in data[algo]:
            se, rew, qos = series(r, "eval_final_sum_se"), series(r, "eval_final_reward"), series(r, "eval_final_qos_all")
            if se is None:
                continue
            best_se.append(se[1].max())
            last_se.append(se[1][-1])
            best_r.append(rew[1].max())
            last_qos.append(qos[1][-1])
            target = se[1][0] + 0.9 * (se[1][-1] - se[1][0])           # 90% of the improvement over step 0
            hit = np.nonzero(se[1] >= target)[0] if se[1][-1] > se[1][0] else []
            steps90.append(se[0][hit[0]] if len(hit) else np.nan)
            wall.append(float(r[-1].get("wall_s") or 0.0))
        f = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else ""
        rows.append({"algo": LABEL[algo], "seeds": len(best_se), "last eval sum SE": f(last_se),
                     "best eval sum SE": f(best_se), "best eval reward": f(best_r), "last all-UE QoS": f(last_qos),
                     "steps to 90% (k)": f"{np.nanmean(steps90) / 1e3:.0f}" if steps90 and not np.all(np.isnan(steps90)) else "",
                     "wall time (min)": f"{np.mean(wall) / 60:.1f}" if wall else ""})
    if not rows:
        return
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
    print(open(os.path.join(out_dir, "summary.md")).read())


def all_done(root, expect):
    files = glob.glob(os.path.join(root, "*", "seed*", "progress.json"))
    if len(files) < max(expect, 1):
        return False
    for f in files:
        try:
            with open(f) as fh:
                if json.load(fh).get("status") != "done":
                    return False
        except (OSError, ValueError):
            return False
    return True


def draw(root, out):
    data = load(root)
    if not data:
        print(f"no metrics.csv under {root} yet", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    plot(data, out)
    summary(data, out)
    print("saved", os.path.join(out, "convergence.png"), time.strftime("%H:%M:%S"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--watch", type=int, default=0, help="redraw every N seconds until every run is done")
    ap.add_argument("--expect", type=int, default=0, help="number of runs to wait for in --watch mode")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "figures")
    if not a.watch:
        draw(a.root, out)
        return
    while True:
        try:
            draw(a.root, out)
        except Exception as exc:                                   # a CSV being written; retry next round
            print("plot failed:", exc, flush=True)
        if all_done(a.root, a.expect):
            draw(a.root, out)
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
