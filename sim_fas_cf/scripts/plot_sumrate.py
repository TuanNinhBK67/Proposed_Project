"""Convergence figure: sum rate of the training episodes vs. environment steps, one line per method
(MAPPO, PPO, DDPG, TD3, Random), mean over seeds of a moving average over episodes.

python scripts/plot_sumrate.py --root runs/bench --out runs/bench/figures [--watch 300 --expect 15]
Writes convergence_sumrate.{pdf,png} and convergence_sumrate.csv (steps | one column per method).
Every training run already redraws this figure while it trains (training.plot_interval_s); this script is for
a manual redraw, other smoothing, or the final_sum_se metric.
The y value of an episode is its sum SE averaged over the T optimisation steps (ep_sum_se in episodes.csv);
--metric final_sum_se plots the sum SE after the T steps instead.
"""
import argparse
import glob
import json
import os
import time

from simfas.plotting import METHODS, draw


def all_done(root, expect, keys=None):
    """keys: run sub-paths under root (--methods); default every run directly under root."""
    files = [f for k in (keys or ["*"]) for f in glob.glob(os.path.join(root, k, "seed*", "progress.json"))]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/bench")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="ep_sum_se", choices=["ep_sum_se", "final_sum_se", "ep_reward", "ep_qos_satisfied"],
                    help="episodes.csv column: sum rate, sum rate after T steps, reward (per step), QoS satisfaction ratio")
    ap.add_argument("--smooth", type=int, default=5000, help="trailing moving average over episodes (5000 = 250k steps at T=50)")
    ap.add_argument("--band", action="store_true", help="also shade +-1 std over seeds")
    ap.add_argument("--watch", type=float, default=0, help="redraw every N seconds until every run is done")
    ap.add_argument("--expect", type=int, default=0, help="number of runs to wait for in --watch mode")
    ap.add_argument("--methods", default=None,
                    help='custom curves "sub/path=Label;sub/path2=Label 2" (run dirs under --root; colour of the '
                         'method named by the last path component, else the palette in order)')
    ap.add_argument("--name", default=None, help="output file stem (default convergence_sumrate)")
    a = ap.parse_args()
    methods = None
    if a.methods:
        pairs = [m.split("=", 1) for m in a.methods.split(";") if m.strip()]
        colour = {key: c for key, _, c in METHODS}
        methods = [(k.strip(), lb.strip(), colour.get(os.path.basename(k.strip()), METHODS[i % len(METHODS)][2]))
                   for i, (k, lb) in enumerate(pairs)]
    kw = dict(out=a.out, metric=a.metric, smooth_w=a.smooth, band=a.band, methods=methods, name=a.name)
    while True:
        try:
            draw(a.root, **kw)
        except Exception as exc:                                  # a CSV being written; retry next round
            print("plot failed:", exc, flush=True)
        if a.watch <= 0 or all_done(a.root, a.expect, [m[0] for m in methods] if methods else None):
            if a.watch > 0:
                draw(a.root, **kw)
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
