"""Batched (E environments in lock-step) SIM-FAS cell-free environment for iterative resource optimisation.

Episode (manuscript Sec. III-C): reset draws the geometry, the large-scale gains and the small-scale fading,
runs AGA (F) and the FAS-control delegation (ctrl = nearest AP among those serving the UE), and installs an initial feasible configuration
(equal power P_max/U per feed, uniform random phases, random port sets). Hbar is then held FIXED for T
decision steps; only the effective channels change with the phases and ports. After step T the episode
terminates (terminated=True, bootstrap value 0) and the environment resets itself.

Time indexing:  o_t, s_t contain the configuration installed before a_t and the feedback (gamma, xi) of the
previous step;  step(a_t) evaluates r_t on Hbar, installs a_t and stores (gamma_t, xi_t) for o_{t+1}, s_{t+1}.

Actor observation of AP l (local CSI only, Eq. obs):
  [CSI of Hbar_l (all K, all ports) | F_l | p_l / Pmax | cos, sin(phase_l) | active-port indicators of UE k if
   l controls k (env.obs_ports_scope = "controlled") or serves k ("served") | control mask of l | xi_{t-1} / R^min
   | connection flags of l (obs_ue_scope = "connected") | port gains of the UEs controlled by l (env.port_gain_obs)]
  With obs_ue_scope = "connected" the CSI and xi entries of the UEs not served by AP l are zero.
  Optional blocks (config._blocks): control mask (env.obs_ctrl_mask, required with "served"), xi (env.obs_qos),
  connection flags (env.obs_conn_flag), D (env.state_distance); phases as cos, sin or phi / 2 pi
  (env.phase_encoding).
Critic global state (training only, Eq. state):
  [D / D_max | CSI of Hbar (all APs) | F | p / Pmax | cos, sin(phase) | active-port indicators of all UEs |
   log1p(gamma_{t-1}) | xi_{t-1} / R^min | t / T (env.state_time) | port gains of all UEs (env.port_gain_obs)]
CSI encoding of each link (env.channel_encoding): "gain_direction" (default) = [real, imag of Hbar / rms(Hbar),
link SNR 10log10(rms^2 Pmax / sigma2) / 10, per-port gain 10log10(||Hbar_p||^2 / mean_p) / 10], a lossless
re-parameterisation that keeps the 30+ dB large-scale spread out of the small-scale features; "raw_snr" =
Hbar * sqrt(Pmax / sigma2). The channel blocks are then standardised by training-only running statistics.
"""
import math
import torch
from . import association, channels, fas, geometry, sim_transfer
from .config import Config, dims
from .phy import desired_port_power
from .reward import evaluate_link, link_metrics
from .utils import complex_dtype, make_generator

GAMMA_FEATURE_CLIP = 20.0


class SimFasEnv:
    def __init__(self, cfg: Config, num_envs: int, seed: int, device="cpu"):
        self.cfg, self.E, self.device = cfg, num_envs, torch.device(device)
        s = cfg.system
        self.L, self.U, self.K, self.M, self.N = s.L, s.U, s.K, s.M, s.N
        self.Np, self.n_max, self.nk = s.Np, s.n_max, s.nk_list
        self.T = cfg.env.episode_steps
        self.cdtype = complex_dtype(cfg.env.complex_dtype)
        self.gen = make_generator(seed, "cpu")                 # device-independent scenario stream

        self.Q1, self.Qinter = sim_transfer.build_sim_matrices(cfg, self.device, self.cdtype)
        J, R = channels.correlation_matrices(cfg)
        t = lambda a: torch.as_tensor(a, dtype=self.cdtype, device=self.device)
        self.J_sqrt, self.R_sqrt = t(channels.psd_sqrt(J, name="J")), t(channels.psd_sqrt(R, name="R"))
        self.ap_xy = torch.as_tensor(geometry.ap_positions(s.L, s.area_m), dtype=torch.float64, device=self.device)
        self.feed_off = torch.as_tensor(geometry.feed_offsets(s.U, s.feed_spacing_lambda * s.wavelength),
                                        dtype=torch.float64, device=self.device)
        self.slot_valid = fas.slot_valid_mask(self.nk, self.n_max, self.device)
        self.pmax, self.sigma2 = s.p_max_w, s.noise_w
        self.r_min = torch.tensor(s.min_se_list, dtype=torch.float64, device=self.device)
        self.h_scale = math.sqrt(self.pmax / self.sigma2)
        self.d_max = math.sqrt(s.area_m[0] ** 2 + s.area_m[1] ** 2 + (s.ap_height_m - s.ue_height_m) ** 2)
        d = dims(cfg)
        self.obs_dim, self.state_dim = d["obs"], d["state"]
        self.obs_channel, self.state_channel = d["obs_channel"], d["state_channel"]
        self.t = torch.zeros(self.E, dtype=torch.long, device=self.device)
        self._drawn = False

    # ------------------------------------------------------------------ episode control
    def _draw_episode(self, mask: torch.Tensor):
        s, E = self.cfg.system, self.E
        if self.cfg.env.reset_geometry_each_episode or not self._drawn:
            ue = geometry.sample_ue_positions(E, self.K, s.area_m, s.ue_margin_m, self.gen, self.device)
            D = geometry.ap_ue_distances(self.ap_xy, ue, s.ap_height_m, s.ue_height_m)
            Df = geometry.feed_ue_distances(self.ap_xy, self.feed_off, ue, s.ap_height_m, s.ue_height_m)
            beta = channels.large_scale_gain(self.cfg, D, self.gen)
            F = association.aga(Df)
            ctrl, ctrl_mask = geometry.nearest_ap_control(D, F.any(2))
            new_geo = {"ue_xy": ue, "D": D, "D_feed": Df, "beta": beta, "F": F, "ctrl": ctrl, "ctrl_mask": ctrl_mask}
        else:
            new_geo = {}
        beta = new_geo.get("beta", getattr(self, "beta", None))
        new = dict(new_geo)
        new["Hbar"] = channels.draw_hbar(beta, self.J_sqrt, self.R_sqrt, self.gen)
        new["p"] = torch.full((E, self.L, self.U), self.pmax / self.U, dtype=torch.float64, device=self.device)
        new["phase"] = 2 * math.pi * torch.rand((E, self.L, self.M, self.N), generator=self.gen,
                                                dtype=torch.float64).to(self.device)
        new["ports"] = fas.random_ports(E, self.nk, self.Np, self.n_max, self.gen, self.device)
        for k, v in new.items():
            if not self._drawn:
                setattr(self, k, v)
            else:
                m = mask.view((E,) + (1,) * (v.dim() - 1))
                setattr(self, k, torch.where(m, v, getattr(self, k)))
        self._drawn = True
        res = self._evaluate(self.p, self.phase, self.ports)     # feedback of the initial configuration
        if not hasattr(self, "gamma_prev"):
            self.gamma_prev, self.xi_prev = res.gamma.double(), res.xi.double()
        else:
            self.gamma_prev = torch.where(mask[:, None], res.gamma.double(), self.gamma_prev)
            self.xi_prev = torch.where(mask[:, None], res.xi.double(), self.xi_prev)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)

    def reset(self) -> dict:
        self._draw_episode(torch.ones(self.E, dtype=torch.bool, device=self.device))
        return self.observe()

    def _evaluate(self, p, phase, ports):
        return evaluate_link(self.Hbar, ports, self.slot_valid, phase, p, self.F, self.Q1, self.Qinter,
                             self.sigma2, self.r_min, self.cfg.env.qos_penalty)

    # ------------------------------------------------------------------ observations
    def channel_features(self) -> torch.Tensor:
        """(E, L, K, ch) real features of every Hbar[l, k] (see module docstring)."""
        E, L, K = self.E, self.L, self.K
        if self.cfg.env.channel_encoding == "raw_snr":
            H = (self.Hbar * self.h_scale).reshape(E, L, K, -1)
            return torch.cat([H.real, H.imag], -1)
        port_pw = (self.Hbar.abs() ** 2).sum(-1).double()                        # (E, L, K, Np)
        mean_pw = port_pw.mean(-1).clamp_min(1e-300)                             # per-entry power x N
        rms = torch.sqrt(mean_pw / self.N)                                       # rms entry amplitude
        Hn = (self.Hbar / rms[..., None, None].to(self.Hbar.real.dtype)).reshape(E, L, K, -1)
        snr_db = 10 * torch.log10(rms ** 2 * self.pmax / self.sigma2) / 10
        port_db = 10 * torch.log10(port_pw.clamp_min(1e-300) / mean_pw[..., None]) / 10
        return torch.cat([Hn.real.double(), Hn.imag.double(), snr_db[..., None], port_db], -1)

    def port_gain_features(self) -> torch.Tensor:
        """(E, K, Np+1) for env.port_gain_obs: desired-signal power of UE k at every port through the current SIM
        phases and powers (coherent sum over its serving APs), as [10 log10(g_port / mean_port g) / 10 (Np),
        10 log10(mean_port g / sigma2) / 10]."""
        g = desired_port_power(self.Hbar, self.Q1, self.Qinter, self.phase, self.p, self.F).double()
        g = g.clamp_min(self.sigma2 * 1e-6)
        mean = g.mean(-1, keepdim=True)
        return torch.cat([torch.log10(g / mean), torch.log10(mean / self.sigma2)], -1)

    def phase_features(self) -> list[torch.Tensor]:
        """SIM phase blocks (E, L, M, N): [cos, sin] (env.phase_encoding = "cos_sin") or [phi / 2 pi] ("raw")."""
        if self.cfg.env.phase_encoding == "raw":
            return [self.phase / (2 * math.pi)]
        return [torch.cos(self.phase), torch.sin(self.phase)]

    def actor_obs(self) -> torch.Tensor:
        """(E, L, obs_dim); row l is the local observation of AP agent l (own CSI only), blocks as in
        config._blocks. With env.obs_ue_scope = "connected", AP l only knows the UEs served by >= 1 of its feeds:
        the CSI and QoS shortfall of the other UEs are zeroed and K connection flags are appended."""
        e, E, L, K = self.cfg.env, self.E, self.L, self.K
        connected = e.obs_ue_scope == "connected"
        conn = self.F.any(2) if connected else torch.ones(E, L, K, dtype=torch.bool, device=self.device)
        C = (self.channel_features() * conn[..., None]).reshape(E, L, -1)
        hot = fas.multi_hot(self.ports, self.slot_valid, self.Np)                       # (E, K, Np)
        own = self.ctrl_mask.to(torch.float32)                                          # (E, L, K)
        seen = own if e.obs_ports_scope == "controlled" else self.F.any(2).to(torch.float32)   # whose ports AP l sees
        ports_l = (hot[:, None] * seen[..., None]).reshape(E, L, -1)
        parts = [C, self.F.reshape(E, L, -1), self.p / self.pmax] + [x.reshape(E, L, -1) for x in self.phase_features()]
        parts.append(ports_l)
        if e.obs_ctrl_mask:
            parts.append(own)
        if e.obs_qos:
            parts.append((self.xi_prev / (self.r_min + 1e-6))[:, None, :].expand(E, L, K) * conn)
        if connected and e.obs_conn_flag:
            parts.append(conn)
        if e.port_gain_obs:                                # only for the UEs whose ports AP l decides
            parts.append((self.port_gain_features()[:, None] * own[..., None]).reshape(E, L, -1))
        return torch.cat([x.to(torch.float32) for x in parts], dim=-1)

    def global_state(self) -> torch.Tensor:
        """(E, state_dim) network-wide state for the centralized critic / centralized baselines (config._blocks)."""
        E = self.E
        parts = [(self.D / self.d_max).reshape(E, -1)] if self.cfg.env.state_distance else []
        parts += [self.channel_features().reshape(E, -1), self.F.reshape(E, -1), (self.p / self.pmax).reshape(E, -1)]
        parts += [x.reshape(E, -1) for x in self.phase_features()]
        parts += [fas.multi_hot(self.ports, self.slot_valid, self.Np).reshape(E, -1),
                  torch.log1p(self.gamma_prev).clamp(max=GAMMA_FEATURE_CLIP), self.xi_prev / (self.r_min + 1e-6)]
        if self.cfg.env.state_time:
            parts.append((self.t.double() / self.T)[:, None])
        if self.cfg.env.port_gain_obs:
            parts.append(self.port_gain_features().reshape(E, -1))
        return torch.cat([x.to(torch.float32) for x in parts], dim=-1)

    def observe(self) -> dict:
        return {"obs": self.actor_obs(), "ctrl_mask": self.ctrl_mask.clone()}

    # ------------------------------------------------------------------ step
    def validate_action(self, p, phase, ports):
        if not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError("powers must be finite and non-negative")
        if (p.sum(-1) > self.pmax * (1 + 1e-5)).any():
            raise ValueError("per-AP power budget violated")
        if not torch.isfinite(phase).all():
            raise ValueError("phases must be finite")
        fas.validate_ports(ports, self.slot_valid, self.Np)

    def step(self, p, phase, ports):
        """Apply the joint physical action of all APs simultaneously.
        p (E, L, U) W, phase (E, L, M, N) rad, ports (E, K, n_max) 0-based (-1 in unused slots)."""
        p = p.to(torch.float64)
        phase = torch.remainder(phase.to(torch.float64), 2 * math.pi)
        ports = ports.long()
        self.validate_action(p, phase, ports)
        res = self._evaluate(p, phase, ports)                    # r_t on the fixed Hbar of the episode
        info = link_metrics(res, self.r_min, p, self.pmax)
        info.update({"rates": res.R, "gamma": res.gamma, "xi": res.xi})
        self.p, self.phase, self.ports = p, phase, ports
        self.gamma_prev, self.xi_prev = res.gamma.double(), res.xi.double()
        self.t = self.t + 1
        terminated = self.t >= self.T
        info["terminated"], info["truncated"] = terminated, torch.zeros_like(terminated)
        if terminated.any():
            self._draw_episode(terminated)
        return self.observe(), info
