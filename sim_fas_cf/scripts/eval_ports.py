"""Port-selection comparison on the held-out scenarios: a trained model keeps its own powers and SIM phases, and
only the port rule is swapped.

  learned       the ports chosen by the policy (MAPPO port head)
  random_fixed  the random ports of the episode start, kept for the T steps (as the MAPPO + random ports variant)
  random_step   fresh random ports at every step
  strongest     every UE activates its n_k ports with the largest desired-signal power through the current SIM
                configuration and powers (UE pilot measurement, interference ignored)
  strongest_prev  the same, measured through the PREVIOUS step's SIM phases and powers (the information in the
                actor observation, env.port_gain_obs)

python scripts/eval_ports.py --runs runs/bench_L6U2K4/mappo/seed1 runs/bench_L6U2K4/mappo_rport/seed1 \
    --out runs/bench_L6U2K4/figures/port_rules.csv
"""
import argparse
import csv
import os

import torch

from simfas import fas
from simfas.baselines import desired_port_power
from simfas.evaluate import evaluate_policy, load_policy

RULES = ("learned", "random_fixed", "random_step", "strongest", "strongest_prev")
KEYS = ("final_sum_se", "final_sum_se_std", "final_qos_ue_frac", "final_qos_all", "final_reward", "mean_sum_se")


class PortRule:
    """Wraps a policy and replaces its ports; `overlap` = mean fraction of the learned active ports that are also
    among the n_k strongest ones."""

    def __init__(self, policy, rule, seed=0):
        self.policy, self.rule = policy, rule
        self.g = torch.Generator().manual_seed(seed)
        self.hits, self.total = 0.0, 0.0

    def __call__(self, env, o):
        prev = env.ports.clone()
        p, phase, ports = self.policy(env, o)
        best = fas.topk_ports(desired_port_power(env, p, phase), env.nk, env.n_max)
        valid = env.slot_valid.expand_as(ports)
        same = (ports[..., :, None] == best[..., None, :]).any(-1) & valid
        self.hits += same.sum().item()
        self.total += valid.sum().item()
        if self.rule == "random_fixed":
            ports = prev
        elif self.rule == "random_step":
            ports = fas.random_ports(env.E, env.nk, env.Np, env.n_max, self.g, "cpu").to(env.device)
        elif self.rule == "strongest":
            ports = best
        elif self.rule == "strongest_prev":
            ports = fas.topk_ports(desired_port_power(env, env.p, env.phase), env.nk, env.n_max)
        return p, phase, ports

    @property
    def overlap(self):
        return self.hits / max(self.total, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (<root>/<algo>/seed<k>)")
    ap.add_argument("--model", default="final_model.pt")
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="CSV of the table")
    a = ap.parse_args()
    rows = []
    for rd in a.runs:
        policy, cfg, _ = load_policy(os.path.join(rd, a.model), a.device)
        seed = cfg.seed + cfg.training.eval_seed_offset
        for rule in a.rules.split(","):
            w = PortRule(policy, rule)
            ev = evaluate_policy(cfg, w, cfg.training.eval_episodes, seed, a.device)
            row = {"run": rd, "rule": rule, **{k: ev[k] for k in KEYS}, "overlap_learned_strongest": w.overlap}
            rows.append(row)
            print(f"{rd:<45} {rule:<13} sum SE {ev['final_sum_se']:6.3f} (std {ev['final_sum_se_std']:.2f})  "
                  f"QoS UE {ev['final_qos_ue_frac']:.3f}  all {ev['final_qos_all']:.3f}  "
                  f"overlap {w.overlap:.3f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
