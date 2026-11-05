# simfas — SIM–FAS-aided cell-free massive MIMO, AP-centric MAPPO and baselines

**Full description of the model, every hyper-parameter and every decision (Vietnamese): `docs/PROJECT_GUIDE.md`.**

Implementation of the manuscript *Fluid Antennas and Stacked Intelligent Metasurfaces for Cell-Free Massive
MIMO Systems* (`demo_test.pdf`): SIM wave-domain beamforming at every AP (Rayleigh–Sommerfeld), UE-side 2-D FAS
with n_k simultaneously active ports and MMSE combining, AGA feed–UE association, and the CTDE MAPPO learner,
benchmarked against centralized PPO, DDPG, TD3 and a random policy.

The previous RZF-based code (`../sim_fas_mappo`) is untouched; this package replaces it for the manuscript model.

## Install
```bash
conda activate tuanmanh            # /home/ncl/anaconda3/envs/tuanmanh
cd FA_SIM/sim_fas_cf
PIP_CONFIG_FILE=/dev/null python -m pip install -e . --no-deps
python -m pytest -q                # 68 tests: physics vs. loop references, AGA, actions, log-probs, env, learners
```

## Train one algorithm
```bash
python scripts/train.py --config configs/default.yaml --algo mappo --seed 1 --out runs/bench
#   --algo mappo | ppo | ddpg | td3 | random      --steps 500000      --device cuda|cpu
#   --set ppo.target_kl=0.02 env.qos_penalty=10   (any YAML key)
```
Each run writes `<out>/<algo>/seed<k>/`: `metrics.csv` (training curves + held-out evaluations),
`best_model.pt` (best held-out terminal reward), `final_model.pt`, `config.yaml`, `final.json`, `progress.json`.

## Convergence benchmark (all algorithms x seeds, parallel) and figures
```bash
STEPS=1000000 SEEDS="1" OUT=runs/bench GPU_ON=0 GPU_OFF=0 bash scripts/launch_detached.sh   # detached
python scripts/plot_sumrate.py --root runs/bench              # convergence_sumrate.png/.pdf/.csv
```
`GPU` is the nvidia-smi index (the script sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`). On the RTX 5090 a single
MAPPO/PPO run does ~1.1k environment steps/s; DDPG/TD3 (one gradient step per transition) ~100 steps/s.

## Inference with a saved model
```bash
python scripts/inference.py --model runs/bench/mappo/seed1/best_model.pt --episodes 200 --dump-episode
```
Prints the terminal metrics after T steps (sum SE, reward, per-UE/all-UE QoS satisfaction, shortfall, 5th
percentile UE SE, power use) and the per-step refinement curve; writes `best_model_inference.json` and, with
`--dump-episode`, one episode's trajectory and final configuration (powers, phases, ports).
A MAPPO actor only sees AP-local inputs, so it can be run with a different number of APs L; changing U, K, M,
N or the port grid changes the network shapes and needs retraining.

## Layout
| Path | Content |
|---|---|
| `src/simfas/config.py` | YAML <-> dataclasses, validation (LU >= K, 1 <= n_k <= Np), input sizes |
| `geometry.py`, `fas.py`, `channels.py`, `sim_transfer.py` | positions, Eq.(1)-(8) |
| `association.py` | AGA (Sec. III-B) |
| `phy.py`, `reward.py` | Eq.(9)-(22), penalised reward of (P2) |
| `env.py` | batched fixed-channel refinement episodes, local observations, global state |
| `action_maps.py` | U+1 softmax power, 2 pi sigmoid phase, DDPG/TD3 bounded maps |
| `mappo/` | hybrid actor (Gaussian x2 + Plackett–Luce ports), critic, MAPPO / PPO trainer |
| `offpolicy/` | DDPG / TD3 trainer and replay buffer |
| `evaluate.py`, `baselines.py`, `training.py` | held-out evaluation, model loading, random policy, shared loop |
| `docs/PROJECT_GUIDE.md` | full guide: model, networks, hyper-parameters, decisions, usage |
| `docs/ASSUMPTIONS.md` | every assumption and manuscript issue |
| `docs/EQUATION_TO_CODE.md` | equation -> function -> test table |
