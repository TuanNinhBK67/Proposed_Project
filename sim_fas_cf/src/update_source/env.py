import math
import numpy as np
import torch
import yaml 
from types import SimpleNamespace
from .physics import *
from .utils import *

def load_config(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in raw.items()})
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    s = cfg.system
    s.n = s.nk
    s.wavelength = 300_000_000.0 / (s.carrier_ghz * 1e9)
    s.Np = s.Nx * s.Ny
    s.sim_shape = (round(math.sqrt(s.N)),) * 2
    s.p_max_w = 10 ** ((s.p_max_dbm - 30) / 10)
    s.noise_w = 10 ** ((s.noise_dbm - 30) / 10)
    return cfg

class SimFasEnv():
    def __init__(self, cfg, E, seed, device="cpu"):
        s = cfg.system
        self.cfg, self.E, self.seed, self.device = cfg, E, seed, torch.device(device)
        self.L, self.U, self.K, self.M, self.N, self.Np, self.n = s.L, s.U, s.K, s.M, s.N, s.Np, s.n
        self.T, self.lam = cfg.env.episode_steps, cfg.env.qos_penalty # lambda for penalty in reward
        self.pmax, self.sigma2, self.r_min = s.p_max_w, s.noise_w, s.min_se
        self.gen = torch.Generator("cpu").manual_seed(seed)

        # Đại lượng không đổi
        cd, fd = getattr(torch, cfg.env.complex_dtype), torch.float64
        self.Q1, self.Qinter = build_sim_matrices(cfg, self.device, cd)                   # (N,U), (N,N)
        J, R = correlation_matrices(cfg)
        self.J_sqrt = torch.as_tensor(psd_sqrt(J, name="J"), dtype=cd, device=self.device)  # (Np,Np)
        self.R_sqrt = torch.as_tensor(psd_sqrt(R, name="R"), dtype=cd, device=self.device)  # (N,N)
        self.ap_xy = torch.as_tensor(ap_positions(s.L, s.area_m), dtype=fd, device=self.device)  # (L,2)
        self.ant_off = torch.as_tensor(antenna_offsets(s.U, s.feed_spacing_lambda * s.wavelength),
                                       dtype=fd, device=self.device)                      # (U,)
        self.h_scale = math.sqrt(self.pmax / self.sigma2) # Hệ số nhân đổi H_bar sang thang đo 

        # Kích thước cho train.py và mappo.py
        L, U, K, M, N, Np = self.L, self.U, self.K, self.M, self.N, self.Np
        self.obs_csi_dim = K * 2 * Np * U
        self.state_csi_dim = L * self.obs_csi_dim
        self.obs_dim = self.obs_csi_dim + U * K + U + M * N + K * Np + K + K
        self.state_dim = self.state_csi_dim + L * U * K + L * U + L * M * N + K * Np + K + K + 1
        self.act_dims = dict(power=U + 1, phase=M * N, K=K, Np=Np, n=self.n)

    @torch.no_grad()
    def _link(self, p, phase, ports):
        return evaluate_link(self.Hbar, ports, phase, p, self.F, self.Q1, self.Qinter,
                             self.sigma2, self.r_min, self.lam)

    @torch.no_grad()
    def _update_csi(self):
        """Kênh hiệu dụng tới mọi port theo phase hiện tại."""
        E, L, K = self.E, self.L, self.K
        T = sim_apply(self.phase, self.Qinter, self.Q1)                # (E,L,N,U)
        H = (self.Hbar @ T[:, :, None]) * self.h_scale                 # (E,L,K,Np,U)
        H = H.reshape(E, L, K, -1)                                     # (E,L,K,Np*U)
        csi = torch.cat([H.real, H.imag], dim=-1)                      # (E,L,K,2*Np*U)
        self.csi_global = csi.reshape(E, -1)                           # (E, L*K*2*Np*U)
        self.csi_local = csi.reshape(E, L, -1)                         # (E, L, K*2*Np*U)

    def _new_episode(self):
        s, E = self.cfg.system, self.E
        if getattr(self.cfg.env, "fixed_channel", False):            # chỉ dùng để thử overfit một kênh
            self.gen.manual_seed(self.seed)

        # 1. Kịch bản: UE -> khoảng cách -> kênh -> gán antenna -> AP điều khiển
        ue = sample_ue_positions(E, self.K, s.area_m, s.ue_margin_m, self.gen, self.device)   # (E,K,2)
        D = ap_ue_distances(self.ap_xy, ue, s.ap_height_m, s.ue_height_m)                     # (E,L,K)
        D_ant = antenna_ue_distances(self.ap_xy, self.ant_off, ue, s.ap_height_m, s.ue_height_m)  # (E,L,U,K)
        beta = large_scale_gain(self.cfg, D, self.gen)                                        # (E,L,K)
        self.Hbar = draw_hbar(beta, self.J_sqrt, self.R_sqrt, self.gen)                       # (E,L,K,Np,N)
        self.F = aga(D_ant)                                                                   # (E,L,U,K)
        self.serve = self.F.any(2)                                                            # (E,L,K)
        self.ctrl, self.ctrl_mask = nearest_ap_control(D, self.serve)                         # (E,K), (E,L,K)
        
        # 3. Cấu hình đầu: công suất đều, phase và port ngẫu nhiên
        self.p = torch.full((E, self.L, self.U), self.pmax / self.U, dtype=torch.float64, device=self.device)
        self.phase = 2 * math.pi * torch.rand((E, self.L, self.M, self.N), generator=self.gen,
                                              dtype=torch.float64).to(self.device)
        self.ports = torch.rand((E, self.K, self.Np), generator=self.gen).topk(self.n, dim=-1).indices.to(self.device)

        # 4. Phản hồi của cấu hình đầu, dùng cho state đầu tiên
        res = self._link(self.p, self.phase, self.ports)
        self.gamma, self.xi = res["gamma"], res["xi"]
        self.t = 0
        self._update_csi()

    #phase2
    def _phase_feat(self):
        return torch.remainder(self.phase / (2 * math.pi), 1.0)

    def _port_hot(self):
        hot = torch.zeros(self.E, self.K, self.Np, device=self.device)
        return hot.scatter_(-1, self.ports, 1.0) # (E,K,Np)

    def _obs(self):
        """(E,L,obs_dim): kênh của AP tới mọi UE, port của mọi UE, cấu hình của AP và phản hồi SINR."""
        E, L = self.E, self.L
        ports = self._port_hot()[:, None].expand(-1, L, -1, -1)         # (E,L,K,Np)
        feedback = torch.log1p(self.gamma)[:, None].expand(-1, L, -1)   # (E,L,K)
        parts = [self.csi_local,                                        # K*2*Np*U
                self.F.reshape(E, L, -1),                               # U*K
                self.p / self.pmax,                                     # U
                self._phase_feat().reshape(E, L, -1),                   # M*N
                ports.reshape(E, L, -1),                                # K*Np
                self.ctrl_mask,                                         # K
                feedback]                                               # K
        return torch.cat([x.float() for x in parts], dim=-1)


    def _state(self):
        """(E,L,state_dim): state toàn mạng, lặp L lần theo quy ước share_obs. Chỉ dùng khi train."""
        E = self.E
        parts = [self.csi_global,                                            # L*K*2*Np*U
                 self.F.reshape(E, -1),                                      # L*U*K
                 (self.p / self.pmax).reshape(E, -1),                        # L*U
                 self._phase_feat().reshape(E, -1),                          # L*M*N
                 self._port_hot().reshape(E, -1),                            # K*Np
                 torch.log1p(self.gamma),                                    # K
                 self.xi / (self.r_min + 1e-6),                              # K
                 torch.full((E, 1), self.t / self.T, device=self.device)]    # 1                    
        state = torch.cat([x.float() for x in parts], dim=-1)                # (E,state_dim)
        return state[:, None].expand(-1, self.L, -1)

    #phase3
    def reset(self):
        self._new_episode()
        return self._obs(), self._state()

    def step(self, p, phase, ports):
        """p (E,L,U), phase (E,L,M,N), ports (E,K,n) -> obs, share_obs, reward (E,L,1), done (E,L), info."""
        # 1. Đánh giá hành động trên Hbar hiện tại
        res = self._link(p, phase, ports)

        # 2. Lưu cấu hình và phản hồi cho obs của bước sau
        self.p, self.phase, self.ports = p, phase, ports
        self.gamma, self.xi = res["gamma"], res["xi"]
        self.t += 1

        # 3. Reward chung và cờ kết thúc
        reward = res["reward"].float()[:, None, None].expand(-1, self.L, 1)
        terminal = self.t >= self.T
        done = torch.full((self.E, self.L), terminal, dtype=torch.bool, device=self.device)

        # 4. Tự reset SAU KHI đã có reward và done
        if terminal:
            self._new_episode()
        else:
            self._update_csi()   
        return self._obs(), self._state(), reward, done, res

    def decode(self, z_power, z_phase, draws):
        """z_power (E,L,U+1), z_phase (E,L,M*N), draws (E,L,K,n) -> p, phase, ports."""
        E, L, K = self.E, self.L, self.K
        p = self.pmax * torch.softmax(z_power.double(), dim=-1)[..., :-1]     # phần tử cuối là công suất bỏ không
        phase = 2 * math.pi * torch.sigmoid(z_phase.double()).reshape(E, L, self.M, self.N)
        idx = self.ctrl[:, None, :, None].expand(E, 1, K, self.n)             # UE k nghe AP điều khiển nó
        ports = draws.gather(1, idx)[:, 0]                                    # (E,K,n)
        return p, phase, ports

#Test code: 
if __name__ == "__main__":
    cfg = load_config("update_src/config.yaml")
    env = SimFasEnv(cfg, E=4, seed=cfg.seed, device=cfg.device)
    E, L, K, n = env.E, env.L, env.K, env.n

    # 1. Shape của reset và tính hợp lệ của AGA, ctrl
    obs, state = env.reset()
    assert obs.shape == (E, L, env.obs_dim) and state.shape == (E, L, env.state_dim)
    assert (env.F.sum(-1) == 1).all() and (env.F.sum((1, 2)) >= 1).all()
    assert (env.ctrl_mask.sum(1) == 1).all()
    assert (env.csi_local.reshape(E, L, K, -1)[~env.serve] != 0).any()    # AP thấy kênh tới UE nó không phục vụ
    assert not (env.ctrl_mask & ~env.serve).any()                         # AP chỉ điều khiển UE nó phục vụ

    # 2. decode thỏa ràng buộc và lấy đúng port của AP điều khiển
    z_p = torch.randn(E, L, env.U + 1, device=env.device)
    z_ph = torch.randn(E, L, env.M * env.N, device=env.device)
    draws = torch.rand(E, L, K, env.Np, device=env.device).topk(n, dim=-1).indices
    p, phase, ports = env.decode(z_p, z_ph, draws)
    assert (p >= 0).all() and (p.sum(-1) <= env.pmax * (1 + 1e-9)).all()
    assert ports.shape == (E, K, n)
    assert torch.equal(ports[0, 0], draws[0, env.ctrl[0, 0], 0])

    # 3. Vòng lặp một episode
    old_H = env.Hbar
    for t in range(env.T):
        H = env.Hbar
        obs, state, r, done, info = env.step(env.p, env.phase, env.ports)
        assert r.shape == (E, L, 1) and done.shape == (E, L)
        assert bool(done.all()) == (t == env.T - 1)        # chỉ done ở bước cuối
        assert t == env.T - 1 or env.Hbar is H             # kênh cố định trong episode
    assert not torch.equal(env.Hbar, old_H) and env.t == 0  # sang episode mới

    before, H = env.csi_global.clone(), env.Hbar
    env.step(*env.decode(z_p, z_ph, draws))
    assert env.Hbar is H and env.t == 1                       # kênh giữ nguyên trong episode
    assert not torch.equal(env.csi_global, before)            # CSI đổi theo phase

    # 4. Không có NaN hay inf
    assert torch.isfinite(obs).all() and torch.isfinite(state).all() and torch.isfinite(r).all()
    print("env OK:", obs.shape, state.shape, "sum_se =", info["sum_se"].mean().item())
