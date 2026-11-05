"""Experiment configuration (YAML <-> dataclasses).

Units: powers in W (dBm only in the YAML, converted with 10**((dBm-30)/10)), distances in m, rates in bit/s/Hz.
Every value that the manuscript does not fix is an implementation assumption listed in docs/ASSUMPTIONS.md.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import yaml

SPEED_OF_LIGHT = 299_792_458.0


def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


@dataclass
class SystemConfig:
    L: int = 3                              # SIM-aided APs
    U: int = 4                              # feeds (antennas) per AP
    K: int = 4                              # UEs
    M: int = 2                              # SIM layers per AP
    N: int = 16                             # meta-atoms per layer
    Nx: int = 4                             # FAS candidate ports along x
    Ny: int = 4                             # FAS candidate ports along y
    nk: Any = 2                             # active ports per UE: int (all UEs) or list of K ints
    Wx: float = 1.0                         # normalised FAS aperture along x (in wavelengths)
    Wy: float = 1.0
    fas_centered: bool = True               # [assumption] shift Eq.(1) coordinates to the aperture centre
    carrier_ghz: float = 28.0
    p_max_dbm: float = 20.0                 # per-AP power budget P_l^max (same for every AP)
    noise_dbm: float = -90.0                # noise variance per active RF chain (sigma_k^2, same for every UE)
    min_se: Any = 0.5                       # R_k^min: float (all UEs) or list of K floats [bit/s/Hz]
    # [assumption] deployment
    area_m: list = field(default_factory=lambda: [100.0, 100.0])
    ap_height_m: float = 10.0
    ue_height_m: float = 1.5
    ue_margin_m: float = 5.0                # UEs uniform in the area minus this margin
    # [assumption] SIM / feed geometry (Eq.(4)-(5))
    sim_grid: list | None = None            # [nx, ny] of a layer; default sqrt(N) x sqrt(N)
    meta_atom_spacing_lambda: float = 0.5   # d_a; meta-atom area S = d_a^2
    sim_interlayer_spacing_lambda: float = 1.0       # axial distance between consecutive layers
    feed_to_first_layer_spacing_lambda: float = 1.0  # axial distance feed plane -> layer 1
    feed_spacing_lambda: float = 0.5        # feeds as a ULA along x (lambda/2 in Eq.(5))
    sim_corr_model: str = "identity"        # R_l: "identity" (default) | "j0" (spherical Bessel over the atoms)
    # [assumption] large-scale fading
    pathloss_model: str = "log_distance"    # "log_distance" | "three_slope"
    pl_ref_db: float | None = None          # loss at pl_ref_distance_m; None -> 32.4 + 20 log10(fc/GHz)
    pl_exponent: float = 3.19               # 10*alpha; 3GPP TR 38.901 UMi NLOS optional model: 31.9 log10(d)
    pl_ref_distance_m: float = 1.0
    pl_const_db: float = -140.7             # three_slope only (Ngo et al. 2017, d in km)
    d0_m: float = 10.0
    d1_m: float = 50.0
    shadowing_std_db: float = 8.2

    # ------------------------------------------------------------------ derived quantities
    @property
    def Np(self) -> int:
        return self.Nx * self.Ny

    @property
    def nk_list(self) -> list[int]:
        return [int(self.nk)] * self.K if isinstance(self.nk, (int, float)) else [int(v) for v in self.nk]

    @property
    def n_max(self) -> int:
        return max(self.nk_list)

    @property
    def min_se_list(self) -> list[float]:
        return [float(self.min_se)] * self.K if isinstance(self.min_se, (int, float)) else [float(v) for v in self.min_se]

    @property
    def wavelength(self) -> float:
        return SPEED_OF_LIGHT / (self.carrier_ghz * 1e9)

    @property
    def p_max_w(self) -> float:
        return dbm_to_w(self.p_max_dbm)

    @property
    def noise_w(self) -> float:
        return dbm_to_w(self.noise_dbm)

    @property
    def sim_shape(self) -> tuple[int, int]:
        if self.sim_grid is not None:
            nx, ny = (int(v) for v in self.sim_grid)
        else:
            nx = ny = int(round(math.sqrt(self.N)))
        if nx * ny != self.N:
            raise ValueError(f"N={self.N} is not a {nx}x{ny} grid; set system.sim_grid explicitly")
        return nx, ny


@dataclass
class EnvConfig:
    episode_steps: int = 20                 # T optimisation iterations per fixed channel realisation
    qos_penalty: float = 5.0                # lambda_QoS of the penalised reward
    complex_dtype: str = "complex64"        # "complex128" for diagnostics / tests
    reset_geometry_each_episode: bool = True
    channel_encoding: str = "gain_direction"  # [assumption] lossless CSI encoding of every Hbar[l, k]:
                                              # "gain_direction": Hbar / rms (real, imag) + link SNR [dB/10]
                                              #   + per-port gain relative to the port mean [dB/10]
                                              # "raw_snr": Hbar * sqrt(Pmax / sigma2) (real, imag) only
    power_map: str = "softmax"              # per-AP power action map, both give sum_u p_lu <= P_max:
                                            #   "softmax": P_max softmax over U+1 shares, the last one unused
                                            #   "linear": p_lu = P_max (a_lu + 1) / 2 per feed (a in [-1, 1]),
                                            #     scaled down to P_max when the AP's sum exceeds it
    port_gain_obs: bool = False             # [assumption] append, per UE, the desired-signal power at every port
                                            #   through the CURRENT SIM phases and powers, coherently summed over
                                            #   the serving APs (what the UE measures per port from its pilots):
                                            #   Np gains relative to their mean + 1 level vs noise, all in dB/10.
                                            #   Actor: only for the UEs whose ports the AP controls; critic: all UEs
    obs_ue_scope: str = "connected"         # [assumption] UEs an AP actor knows about: "connected" = only the UEs
                                            #   with >= 1 of its feeds (F_l), others zeroed + K connection flags;
                                            #   "all" = CSI and QoS shortfall of every UE (ablation, old layout)
    phase_encoding: str = "cos_sin"         # SIM phases in obs and state: "cos_sin" (cos, sin per atom) or
                                            #   "raw" (phi / 2 pi in [0, 1), one value per atom)
    obs_ctrl_mask: bool = True              # actor obs: FAS control mask (K flags; implied by the active-port block
                                            #   only when obs_ports_scope = "controlled", required with "served")
    obs_ports_scope: str = "controlled"     # actor obs, active ports of UE k shown to AP l if l "controlled" its FAS,
                                            #   or if l "served" k (>= 1 feed in F_l; ports shared among serving APs)
    obs_conn_flag: bool = True              # actor obs, obs_ue_scope = "connected": K connection flags (False: the
                                            #   zeroed CSI of the unserved UEs is the only indication)
    obs_qos: bool = True                    # actor obs: QoS shortfall xi_{t-1} (False: QoS / SINR only in the state)
    state_distance: bool = True             # critic state: AP-UE distances D / D_max
    state_time: bool = True                 # critic state: normalised step index t / T (not in Eq. 34)


@dataclass
class PPOConfig:
    """On-policy learners: MAPPO (decentralized AP actors) and single-agent PPO (centralized actor)."""
    shared_actor: bool = True               # MAPPO: one parameter-shared actor for all AP agents
    discount: float = 0.99                  # delta
    gae_lambda: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 5
    rollout_steps: int = 64                 # per environment and update
    minibatch_size: int = 512
    num_envs: int = 32
    hidden: list = field(default_factory=lambda: [256, 256])
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    lr_decay: str = "none"                  # "none" | "linear"
    value_coef: float = 0.5
    value_norm: bool = True
    entropy_coef_power: float = 0.0
    entropy_coef_phase: float = 0.0
    entropy_coef_fas: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02          # stop the PPO epochs of an update once the epoch-mean KL exceeds it
    per_head_clip: bool = False             # True: one ratio and clipped surrogate per action head (power, phase,
                                            #   ports), summed, with the common advantage; False: joint ratio
    head_loss_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])   # per_head_clip: weights of the
                                            #   [power, phase, port] clipped surrogates in the actor loss
    head_target_kl: list | None = None      # per_head_clip: KL thresholds [power, phase, ports]. After an epoch whose
                                            #   mean KL of a head exceeds its threshold, that head's surrogate (and its
                                            #   entropy bonus) leaves the actor loss for the rest of the update; the
                                            #   update ends once every learned head has stopped. None: off
    head_stop_freeze: bool = False          # head_target_kl: once a head has stopped, the shared trunk and the
                                            #   stopped heads get NO update for the rest of the update (gradients
                                            #   dropped, so Adam momentum cannot move them either); only the heads
                                            #   still running update their own layers. False: old behaviour (the
                                            #   trunk keeps moving and drags the stopped heads with it)
    fas_stats_active_only: bool = False     # True: the port-head surrogate (per_head_clip), entropy, KL, clipfrac
                                            #   and the head_target_kl test are means over the agents with >= 1
                                            #   active port draw (controlling >= 1 UE); False: means over all agents
                                            #   (old behaviour; agents controlling no UE add ratio 1 and KL 0)
    init_log_std: float = -0.5
    std_schedule: str = "none"              # "none": Gaussian std of power/phase learned from init_log_std;
                                            #   "linear": std NOT learned, set before every rollout to
                                            #   exp(init_log_std) -> final_std linearly over the training steps
    final_std: float = 0.05
    obs_clip: float = 10.0
    # ablations (on-policy only)
    learn_power: bool = True                # False: equal power P_max/U
    learn_phase: bool = True                # False: random SIM phases of the episode start, kept for T steps
    fas_mode: str = "learned"               # "learned" | "fixed_random" (random ports of the episode start) |
                                            #   "strongest": every UE activates its n_k ports of largest port gain
                                            #   through the configuration in effect (step t-1), chosen together
                                            #   with the new p, phase (one phase; not learned) |
                                            #   "strongest_new": same rule through the NEW p, phase of step t
                                            #   (ports measured after the SIM is configured; not learned) |
                                            #   "strongest_local": the FAS-control AP picks the ports from its OWN
                                            #   contribution only (its CSI, new phases and powers of step t; local)
    fas_head: str = "per_ue"                # "per_ue": logits of UE k from [trunk features, features of UE k] by
                                            #   one small MLP shared over UEs; "flat": one linear K*Np head (old)
    fas_head_hidden: int = 128


@dataclass
class OffPolicyConfig:
    """DDPG / TD3 (single centralized agent on the global state, deterministic bounded actions)."""
    num_envs: int = 4
    hidden: list = field(default_factory=lambda: [256, 256])
    actor_lr: float = 1.0e-4
    critic_lr: float = 3.0e-4
    discount: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 100_000
    warmup_steps: int = 5_000               # environment steps with uniform random actions
    gradient_steps: int = 4                 # per vectorised environment step (= num_envs -> one update per transition)
    expl_noise: float = 0.1                 # Gaussian exploration noise on the [-1, 1] actions
    target_noise: float = 0.2               # TD3 target policy smoothing
    noise_clip: float = 0.5
    policy_delay: int = 2                   # TD3 delayed actor / target updates
    power_scale: float = 3.0                # power logits = power_scale * a  (softmax over U+1)
    reward_scale: float = 0.1
    max_grad_norm: float = 10.0
    obs_clip: float = 10.0


@dataclass
class TrainingConfig:
    total_env_steps: int = 500_000
    eval_interval: int = 10_000             # environment steps between held-out evaluations
    eval_episodes: int = 128
    eval_seed_offset: int = 1_000_003       # held-out scenarios: seed + offset (identical for every algorithm)
    best_metric: str = "final_reward"       # model selection on the terminal-step penalised reward
    checkpoint_interval: int = 50_000
    plot_interval_s: float = 300.0          # each run redraws <out>/figures/convergence_sumrate.* this often (0 = off)
    out_dir: str = "runs"
    run_name: str = ""


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    system: SystemConfig = field(default_factory=SystemConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    offpolicy: OffPolicyConfig = field(default_factory=OffPolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolved_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_SECTIONS = {"system": SystemConfig, "env": EnvConfig, "ppo": PPOConfig, "offpolicy": OffPolicyConfig,
             "training": TrainingConfig}


def _coerce(value, type_name: str, key: str):
    """YAML 1.1 reads e.g. '3.0e-4' as a string; convert to the declared scalar type."""
    if value is None:
        return value
    if type_name == "float":
        return float(value)
    if type_name == "int":
        f = float(value)
        if f != int(f):
            raise ValueError(f"{key} must be an integer, got {value!r}")
        return int(f)
    if type_name == "bool":
        if isinstance(value, str):
            if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{key} must be a boolean, got {value!r}")
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _build(cls, data: dict, where: str):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"unknown keys in '{where}': {sorted(unknown)}")
    return cls(**{k: _coerce(v, str(fields[k].type), f"{where}.{k}") for k, v in data.items()})


def config_from_dict(d: dict) -> Config:
    d = dict(d or {})
    sections = {k: _build(c, d.pop(k, {}) or {}, k) for k, c in _SECTIONS.items()}
    cfg = Config()
    for k, v in sections.items():
        setattr(cfg, k, v)
    for k in list(d):
        if k not in ("seed", "device"):
            raise KeyError(f"unknown top-level key '{k}'")
        setattr(cfg, k, d[k])
    validate(cfg)
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    for ov in overrides or []:                     # "ppo.num_envs=4"
        key, val = ov.split("=", 1)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return config_from_dict(d)


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def validate(cfg: Config) -> None:
    s = cfg.system
    for name in ("L", "U", "K", "M", "N", "Nx", "Ny"):
        if getattr(s, name) < 1:
            raise ValueError(f"system.{name} must be >= 1")
    if s.L * s.U < s.K:
        raise ValueError(f"AGA needs L*U >= K (every UE needs a dedicated feed); got L*U={s.L * s.U} < K={s.K}")
    nk = s.nk_list
    if len(nk) != s.K:
        raise ValueError(f"system.nk must have K={s.K} entries, got {len(nk)}")
    if any(not 1 <= v <= s.Np for v in nk):
        raise ValueError(f"every n_k must satisfy 1 <= n_k <= Np={s.Np}, got {nk}")
    if len(s.min_se_list) != s.K or any(v < 0 for v in s.min_se_list):
        raise ValueError("system.min_se must be a non-negative float or a list of K floats")
    if s.Wx <= 0 or s.Wy <= 0:
        raise ValueError("system.Wx and system.Wy must be > 0")
    s.sim_shape  # raises if N is not a grid
    if s.sim_corr_model not in ("identity", "j0"):
        raise ValueError("system.sim_corr_model must be 'identity' or 'j0'")
    if s.pathloss_model not in ("log_distance", "three_slope"):
        raise ValueError("system.pathloss_model must be 'log_distance' or 'three_slope'")
    if cfg.env.episode_steps < 1:
        raise ValueError("env.episode_steps must be >= 1")
    if cfg.env.qos_penalty < 0:
        raise ValueError("env.qos_penalty must be >= 0")
    if cfg.env.channel_encoding not in ("gain_direction", "raw_snr"):
        raise ValueError("env.channel_encoding must be 'gain_direction' or 'raw_snr'")
    if cfg.env.obs_ue_scope not in ("connected", "all"):
        raise ValueError("env.obs_ue_scope must be 'connected' or 'all'")
    if cfg.env.obs_ports_scope not in ("controlled", "served"):
        raise ValueError("env.obs_ports_scope must be 'controlled' or 'served'")
    if cfg.env.obs_ports_scope == "served" and not cfg.env.obs_ctrl_mask:   # port block no longer implies control
        raise ValueError("env.obs_ports_scope = 'served' needs env.obs_ctrl_mask = true (the actor must know "
                         "which UEs' port logits it decides)")
    if cfg.env.phase_encoding not in ("cos_sin", "raw"):
        raise ValueError("env.phase_encoding must be 'cos_sin' or 'raw'")
    if cfg.ppo.fas_head not in ("per_ue", "flat"):
        raise ValueError("ppo.fas_head must be 'per_ue' or 'flat'")
    if cfg.env.complex_dtype not in ("complex64", "complex128"):
        raise ValueError("env.complex_dtype must be complex64 or complex128")
    p = cfg.ppo
    for sec, d in (("ppo", p.discount), ("offpolicy", cfg.offpolicy.discount)):
        if not 0 <= d < 1:
            raise ValueError(f"{sec}.discount must satisfy 0 <= delta < 1 (Eq. (35)), got {d}")
    if not 0 <= p.gae_lambda <= 1:
        raise ValueError("ppo.gae_lambda must be in [0, 1]")
    if p.clip <= 0:
        raise ValueError("ppo.clip must be > 0 (Eq. (37))")
    for name in ("ppo_epochs", "rollout_steps", "minibatch_size", "num_envs"):
        if getattr(p, name) < 1:
            raise ValueError(f"ppo.{name} must be >= 1")
    if p.minibatch_size < 2 or p.num_envs * p.rollout_steps < 2:   # minibatch advantage normalisation needs a std
        raise ValueError("ppo.minibatch_size and ppo.num_envs * ppo.rollout_steps must be >= 2")
    if cfg.offpolicy.num_envs < 1:
        raise ValueError("offpolicy.num_envs must be >= 1")
    if cfg.env.power_map not in ("softmax", "linear"):
        raise ValueError("env.power_map must be 'softmax' or 'linear'")
    if p.fas_mode not in ("learned", "fixed_random", "strongest", "strongest_new", "strongest_local"):
        raise ValueError("ppo.fas_mode must be 'learned', 'fixed_random', 'strongest', 'strongest_new' or "
                         "'strongest_local'")
    if not (p.learn_power or p.learn_phase or p.fas_mode == "learned"):
        raise ValueError("at least one action branch must be learned")
    w = [float(v) for v in p.head_loss_weights]
    if len(w) != 3 or min(w) < 0:
        raise ValueError("ppo.head_loss_weights must be 3 non-negative numbers [power, phase, ports]")
    if not p.per_head_clip and w != [1.0, 1.0, 1.0]:
        raise ValueError("ppo.head_loss_weights needs ppo.per_head_clip = true")
    p.head_loss_weights = w
    if p.head_target_kl is not None:
        kl = [float(v) for v in p.head_target_kl]
        if len(kl) != 3 or min(kl) <= 0:
            raise ValueError("ppo.head_target_kl must be 3 positive numbers [power, phase, ports] or null")
        if not p.per_head_clip:
            raise ValueError("ppo.head_target_kl needs ppo.per_head_clip = true")
        p.head_target_kl = kl
    if p.head_stop_freeze and p.head_target_kl is None:
        raise ValueError("ppo.head_stop_freeze needs ppo.head_target_kl")
    if not isinstance(p.fas_stats_active_only, bool):
        raise ValueError("ppo.fas_stats_active_only must be true or false")
    if p.std_schedule not in ("none", "linear"):
        raise ValueError("ppo.std_schedule must be 'none' or 'linear'")
    if p.final_std <= 0:
        raise ValueError("ppo.final_std must be > 0")
    if p.lr_decay not in ("none", "linear"):
        raise ValueError("ppo.lr_decay must be 'none' or 'linear'")
    if cfg.training.best_metric not in ("final_reward", "final_sum_se"):
        raise ValueError("training.best_metric must be 'final_reward' or 'final_sum_se'")


def power_dims(cfg: Config) -> int:
    """Power action entries per AP: U+1 softmax shares ("softmax") or U per-feed levels ("linear")."""
    return cfg.system.U + (1 if cfg.env.power_map == "softmax" else 0)


def _blocks(cfg: Config, central: bool) -> list[tuple[str, int]]:
    """Ordered (name, size) blocks of the actor observation (central=False, Eq. obs) or of the critic state
    (central=True, Eq. state); env.actor_obs / env.global_state concatenate exactly these blocks."""
    s, e = cfg.system, cfg.env
    L, U, K, MN, Np = s.L, s.U, s.K, s.M * s.N, s.Np
    ch = 2 * Np * s.N + (1 + Np if e.channel_encoding == "gain_direction" else 0)       # per (l, k) link
    ph = MN * (2 if e.phase_encoding == "cos_sin" else 1)                               # per AP
    pg = [("port_gain", K * (Np + 1))] if e.port_gain_obs else []
    if not central:
        return ([("csi", K * ch), ("F", U * K), ("p", U), ("phase", ph), ("ports", K * Np)]
                + ([("ctrl", K)] if e.obs_ctrl_mask else []) + ([("xi", K)] if e.obs_qos else [])
                + ([("conn", K)] if e.obs_ue_scope == "connected" and e.obs_conn_flag else []) + pg)
    return (([("D", L * K)] if e.state_distance else []) + [("csi", L * K * ch), ("F", L * U * K), ("p", L * U),
            ("phase", L * ph), ("ports", K * Np), ("log_gamma", K), ("xi", K)]
            + ([("t", 1)] if e.state_time else []) + pg)


def _offsets(blocks: list[tuple[str, int]]) -> dict[str, int]:
    out, o = {}, 0
    for name, n in blocks:
        out[name] = o
        o += n
    return out


def dims(cfg: Config) -> dict[str, int]:
    """Flat real-valued input sizes of the actor observation (Eq. obs) and of the critic state (Eq. state)."""
    ob, st = _blocks(cfg, False), _blocks(cfg, True)
    oo, so = _offsets(ob), _offsets(st)
    return {"obs": sum(n for _, n in ob), "state": sum(n for _, n in st),
            "obs_channel": (oo["csi"], oo["csi"] + dict(ob)["csi"]),     # slices normalised with running statistics
            "state_channel": (so["csi"], so["csi"] + dict(st)["csi"])}


def ue_feature_index(cfg: Config, central: bool) -> list[list[int]]:
    """Positions of the features of each UE k inside the actor input (K rows of equal length), following
    _blocks (the optional blocks appear only when enabled):
      obs  : CSI of link (l, k) | F_l[:, k] | active ports of k if l controls k (obs_ports_scope: controlled) or
             serves k (served) | control flag | xi_k | connected flag | port gains of k
      state: D[:, k] | CSI of links (:, k) | F[:, :, k] | active ports of k | log1p(gamma_k) | xi_k | port gains of k"""
    s = cfg.system
    L, U, K, Np = s.L, s.U, s.K, s.Np
    bl = _blocks(cfg, central)
    o, size = _offsets(bl), dict(bl)
    ch = size["csi"] // (K if not central else L * K)
    rng = lambda a, n: list(range(a, a + n))
    rows = []
    for k in range(K):
        if not central:
            r = rng(o["csi"] + k * ch, ch) + [o["F"] + u * K + k for u in range(U)]
        else:
            r = [o["D"] + l * K + k for l in range(L)] if "D" in o else []
            r += [i for l in range(L) for i in rng(o["csi"] + (l * K + k) * ch, ch)]
            r += [o["F"] + (l * U + u) * K + k for l in range(L) for u in range(U)]
        r += rng(o["ports"] + k * Np, Np)
        for name in (("ctrl", "xi", "conn") if not central else ("log_gamma", "xi")):
            if name in o:
                r.append(o[name] + k)
        if "port_gain" in o:
            r += rng(o["port_gain"] + k * (Np + 1), Np + 1)
        rows.append(r)
    return rows
