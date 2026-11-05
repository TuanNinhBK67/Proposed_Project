#!/usr/bin/env bash
# Launch the full convergence benchmark detached from the terminal (survives SSH / VSCode disconnects):
# MAPPO, its 3 variants (mappo_ep, mappo_rp, mappo_rport), PPO and random on GPU_ON, DDPG/TD3 on GPU_OFF. Every run redraws $OUT/figures/convergence_sumrate.*
# while it trains (training.plot_interval_s), so no separate watcher process is needed.
#   CONFIG=configs/default.yaml STEPS=1000000 SEEDS="1" OUT=runs/bench GPU_ON=1 GPU_OFF=2 bash scripts/launch_detached.sh
cd "$(dirname "$0")/.."
PY=${PY:-/home/ncl/anaconda3/envs/tuanmanh/bin/python}
STEPS=${STEPS:-1000000}
SEEDS=${SEEDS:-"1 2 3"}
OUT=${OUT:-runs/bench}
GPU_ON=${GPU_ON:-2}
GPU_OFF=${GPU_OFF:-0}
mkdir -p "$OUT/logs"
export STEPS SEEDS OUT
ALGOS="mappo ppo random mappo_ep mappo_rp mappo_rport" GPU=$GPU_ON setsid nohup bash scripts/run_benchmark.sh > "$OUT/logs/launcher_onpolicy.log" 2>&1 < /dev/null &
ALGOS="ddpg td3" GPU=$GPU_OFF setsid nohup bash scripts/run_benchmark.sh > "$OUT/logs/launcher_offpolicy.log" 2>&1 < /dev/null &
echo "launched; logs in $OUT/logs, figure + CSV in $OUT/figures (redrawn by the runs during training)"
