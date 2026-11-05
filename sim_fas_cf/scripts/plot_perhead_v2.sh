#!/usr/bin/env bash
# Combined figures (sum rate, reward, QoS) of runs/bench_L6U2K4/perhead_v2_6M (trunk-freeze fix, shared ports,
# local port gain, centralized PPO) with the converged baselines of perhead_6M. Extra args go to plot_sumrate.py
# (e.g. --watch 300 --expect 9).
cd "$(dirname "$0")/.."
PY=${PY:-/home/ncl/anaconda3/envs/tuanmanh/bin/python}
M="perhead_v2_6M/mappo=MAPPO (three optimized variable);perhead_v2_6M/ppo=Centralized PPO;perhead_v2_6M/mappo_ep=Equal power allocation;\
perhead_v2_6M/mappo_rp=Random phase shift;perhead_6M/mappo_rport=Random FAS ports;\
perhead_6M/mappo_sp=Proposed arch 1;perhead_6M/mappo_spn=Proposed arch 2;\
perhead_v2_6M/mappo_spl=Proposed arch 3;perhead_v2_6M/ppo_spn=Centralized PPO based on port-gain"
R=runs/bench_L6U2K4; OUT=$R/perhead_v2_6M/figures
for m in ep_sum_se:compare_sumrate ep_reward:compare_reward ep_qos_satisfied:compare_qos; do
  $PY scripts/plot_sumrate.py --root $R --out $OUT --methods "$M" --metric ${m%%:*} --name ${m##*:} "$@"
done
