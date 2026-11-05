#!/usr/bin/env bash
# Convergence benchmark: every algorithm x seed, launched in parallel, one log per run.
#   CONFIG=configs/default.yaml STEPS=500000 SEEDS="1" ALGOS="mappo ppo ddpg td3 random mappo_ep mappo_rp mappo_rport" \
#   OUT=runs/bench GPU=0 bash scripts/run_benchmark.sh
# GPU is the nvidia-smi index (CUDA_DEVICE_ORDER=PCI_BUS_ID). Waits for all runs, then draws the figures.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/ncl/anaconda3/envs/tuanmanh/bin/python}
CONFIG=${CONFIG:-configs/default.yaml}
STEPS=${STEPS:-500000}
SEEDS=${SEEDS:-"1 2 3"}
ALGOS=${ALGOS:-"mappo ppo ddpg td3 random mappo_ep mappo_rp mappo_rport"}
OUT=${OUT:-runs/bench}
GPU=${GPU:-0}
EXTRA=${EXTRA:-}
mkdir -p "$OUT/logs"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4
pids=()
for algo in $ALGOS; do
  for seed in $SEEDS; do
    log="$OUT/logs/${algo}_seed${seed}.log"
    $PY scripts/train.py --config "$CONFIG" --algo "$algo" --seed "$seed" --steps "$STEPS" --out "$OUT" $EXTRA \
      > "$log" 2>&1 &
    pids+=($!)
    echo "started $algo seed $seed (pid $!) -> $log"
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "all runs finished ($fail failed)"
$PY scripts/plot_sumrate.py --root "$OUT" --out "$OUT/figures"
