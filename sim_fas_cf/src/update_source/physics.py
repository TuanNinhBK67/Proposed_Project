# Physics: geometry, channel, SIM, FAS, ...
import numpy as np
import torch 
import math 
import logging
from .utils import *
#===== 1. Geometry =====

# Trả tọa độ 2-D của lưới phẳng nx x ny điểm, đặt đối xứng quanh (0, 0)
def planar_grid(nx: int, ny: int, spacing) -> np.ndarray:
    sx, sy = (spacing, spacing) if np.isscalar(spacing) else spacing # Kiểm tra là số (isscalar) hay là list/tuple
    cx = (np.arange(nx) - (nx - 1) / 2) * sx # Tập x
    cy = (np.arange(ny) - (ny - 1) / 2) * sy # Tập y
    pts = []
    for iy in range(ny):
        for ix in range(nx):
            pts.append((cx[ix], cy[iy]))
    return np.array(pts)

#AP position
def ap_positions(L: int, area) -> np.ndarray:
    ax, ay = float(area[0]), float(area[1])
    n_rows = max(1, round(math.sqrt(L))) # Số hàng

    # Chia đều L AP cho các hàng; AP dư dồn vào các hàng gần giữa nhất
    counts = [L // n_rows] * n_rows
    center = (n_rows - 1) / 2
    rows_by_centrality = sorted(range(n_rows), key=lambda r: abs(r - center))
    for r in rows_by_centrality[: L % n_rows]:
        counts[r] += 1

    # Mỗi AP đặt ở tâm ô của nó
    pos = []
    for r, n in enumerate(counts):
        y = (r + 0.5) * ay / n_rows
        for i in range(n):
            x = (i + 0.5) * ax / n
            pos.append([x, y])
    return np.array(pos, dtype=np.float64)

# AP-antenna position 
def antenna_offsets(U: int, spacing: float) -> np.ndarray:
    return (np.arange(1, U + 1) - (U + 1) / 2) * spacing

# Distribute UE position 
def sample_ue_positions(E: int, K: int, area, margin: float, generator, device) -> torch.Tensor:
    lo = torch.tensor([margin, margin], dtype=torch.float64, device=device)
    hi = torch.tensor([area[0] - margin, area[1] - margin], dtype=torch.float64, device=device)
    u = torch.rand((E, K, 2), generator=generator, device=generator.device, dtype=torch.float64).to(device)
    return lo + (hi - lo) * u

# AP-UE distances
def ap_ue_distances(ap_xy: torch.Tensor, ue_xy: torch.Tensor, h_ap: float, h_ue: float) -> torch.Tensor:
    ap = ap_xy[None, :, None, :] # (1, L, 1, 2)
    ue = ue_xy[:, None, :, :] # (E, 1, K, 2)
    dxy = ap - ue # (E, L, K, 2)
    horizontal_sq = (dxy ** 2).sum(-1) # (E, L, K)
    vertical_sq = (h_ap - h_ue) ** 2
    return torch.sqrt(horizontal_sq + vertical_sq)

# def ap_ue_distances(ap_xy: torch.Tensor, ue_xy: torch.Tensor, h_ap: float, h_ue: float) -> torch.Tensor:
#     E, K, _ = ue_xy.shape
#     L = ap_xy.shape[0]
#     dz = h_ap - h_ue
#     D = torch.zeros(E, L, K, dtype=ue_xy.dtype, device=ue_xy.device)
#     for e in range(E):
#         for l in range(L):
#             for k in range(K):
#                 dx = ap_xy[l, 0] - ue_xy[e, k, 0]
#                 dy = ap_xy[l, 1] - ue_xy[e, k, 1]
#                 D[e, l, k] = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#     return D

# UE-antenna distance -- for AGA
def antenna_ue_distances(ap_xy: torch.Tensor, offsets: torch.Tensor, ue_xy: torch.Tensor, h_ap: float, h_ue: float):
    feed_x = ap_xy[:, None, 0] + offsets[None, :] # (L, U)
    dx = feed_x[None, :, :, None] - ue_xy[:, None, None, :, 0]
    dy = ap_xy[None, :, None, None, 1] - ue_xy[:, None, None, :, 1]
    return torch.sqrt(dx ** 2 + dy ** 2 + (h_ap - h_ue) ** 2)

# Nearest-AP control 
def nearest_ap_control(D: torch.Tensor, serve: torch.Tensor): # D: (E, L, K)
    # serve (E, L, K): AP l có antenna phục vụ UE k (F.any(2)) -> chỉ xét các AP này
    D = torch.where(serve, D, torch.full_like(D, float("inf"))) # AP không phục vụ đổi thành inf
    E, L, K = D.shape
    ap_idx = torch.arange(L, device=D.device).view(1, L, 1) # (1, L, 1) - danh sách chỉ số AP
    is_min = D == D.min(dim=1, keepdim=True).values # (E, L, K) - các AP gần nhất của mỗi UE
    n_cand = is_min.sum(1) # (E, K) - số AP hòa nhau
    idx = ap_idx.expand_as(D)
    ctrl = torch.where(is_min, idx, torch.full_like(idx, L)).min(dim=1).values # (E, K) - tạm lấy AP gần nhất có chỉ số nhỏ nhất

    # Phá thế hòa: giao UE cho AP đang quản lý ít UE nhất; hòa tiếp thì lấy AP có chỉ số nhỏ nhất
    load = (is_min & (n_cand == 1)[:, None, :]).sum(-1) # (E, L) - số UE mỗi AP đã chắc chắn quản lý
    env_idx = torch.arange(E, device=D.device)
    for k in range(K): # Xét lần lượt các UE bị hòa theo thứ tự k
        tied = n_cand[:, k] > 1 # (E,)
        key = torch.where(is_min[:, :, k], load * L + idx[:, :, 0], torch.full_like(load, L * (K + 1))) # Tải x L + chỉ số AP: tải được ưu tiên trước
        pick = key.min(dim=1).values % L # (E,)
        ctrl[:, k] = torch.where(tied, pick, ctrl[:, k])
        load[env_idx, pick] += tied.long() # Cập nhật tải sau khi giao UE

    mask = ctrl[:, None, :] == ap_idx # (E, L, K)
    # for e in range(E):
    #     for l in range(L):
    #         for k in range(K):
    #             mask[e, l, k] = (ctrl[e, k] == l)

    # ctrl[e, k]: AP gần UE k nhất trong số các AP phục vụ UE k ở môi trường e
    # mask[e, l, k]: True nếu AP l điều khiển UE k 
    return ctrl, mask 

# # ===== 2. Kênh =====
# Activated port 
def gather_selected(Hbar: torch.Tensor, ports: torch.Tensor) -> torch.Tensor:
    E, L, K, Np, N = Hbar.shape
    n = ports.shape[-1]
    port_idx = ports[:, None, :, :, None] # (E, 1, K, n, 1)
    port_idx = port_idx.expand(E, L, K, n, N) # (E, L, K, n, N)
    return torch.gather(Hbar, 3, port_idx) # (E, L, K, n, N)

# correlation port FAS
def correlation_from_points(points: np.ndarray, wavelength: float) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :] # (P, P, 2)
    dist = np.linalg.norm(diff, axis=-1) # (P, P)
    phase = 2.0 * np.pi * dist / wavelength # (P, P)
    return spherical_j0(phase)

# correlation matrix for R(N, N) and J(Np, Np)
def port_coordinates(Nx: int, Ny: int, Wx: float, Wy: float, wavelength: float):  # (1)
    dx = Wx * wavelength / (Nx - 1) if Nx > 1 else 0.0
    dy = Wy * wavelength / (Ny - 1) if Ny > 1 else 0.0
    return planar_grid(Nx, Ny, (dx, dy)) #Do spacing nhận được tuple (dx, dy)

def correlation_matrices(cfg):
    s = cfg.system
    lam = s.wavelength
    port_xy = port_coordinates(s.Nx, s.Ny, s.Wx, s.Wy, lam)
    J = correlation_from_points(port_xy, lam)
    if s.sim_corr_model == "j0":
        nx, ny = s.sim_shape
        atom_xy = planar_grid(nx, ny, s.meta_atom_spacing_lambda * lam)
        R = correlation_from_points(atom_xy, lam)
    else:
        R = np.eye(s.N)
    return J, R

# large scale-fading
def pathloss_db(D: torch.Tensor, s) -> torch.Tensor:
    d = D / 1000.0
    d0, d1 = s.d0_m / 1000.0, s.d1_m / 1000.0
    term_d0 = -20.0 * torch.log10(d.clamp_min(d0))
    term_d1 = -15.0 * torch.log10(d.clamp_min(d1))
    return s.pl_const_db + term_d0 + term_d1

def large_scale_gain(cfg, D: torch.Tensor, generator) -> torch.Tensor:
    s = cfg.system
    z = torch.randn(D.shape, generator=generator, device=generator.device, dtype=torch.float64).to(D.device)
    return 10.0 ** ((pathloss_db(D, s) + s.shadowing_std_db * z) / 10.0)

# draw-hbar
def draw_hbar(beta: torch.Tensor, J_sqrt: torch.Tensor, R_sqrt: torch.Tensor, generator) -> torch.Tensor:
    E, L, K = beta.shape
    Np = J_sqrt.shape[0]
    N = R_sqrt.shape[0]
    G = crandn((E, L, K, Np, N), generator, beta.device, J_sqrt.dtype) # (E, L, K, Np, N)
    small_scale = J_sqrt @ G @ R_sqrt # (E, L, K, Np, N)
    large_scale = torch.sqrt(beta).to(J_sqrt.real.dtype)[..., None, None] # (E, L, K, 1, 1) [None, None] nhân với ma trận (Np, N)
    return large_scale * small_scale

# ===== 3. SIM =====
# antenna - atoms distance 
def antenna_layer_distances(cfg) -> np.ndarray:
    s = cfg.system
    lam = s.wavelength
    nx, ny = s.sim_shape
    atoms = planar_grid(nx, ny, s.meta_atom_spacing_lambda * lam)                 # (N, 2)
    antenna_x = antenna_offsets(s.U, s.feed_spacing_lambda * lam)                 # (U,)
    dx = atoms[:, 0, None] - antenna_x[None, :]                                   # (N, U)
    dy = atoms[:, 1, None]                                                        # (N, 1), antennas sit at y = 0
    dz = s.feed_to_first_layer_spacing_lambda * lam
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

# Build SIM matrix
def build_sim_matrices(cfg, device, cdtype):
    s = cfg.system
    lam = s.wavelength
    area = (s.meta_atom_spacing_lambda * lam) ** 2
    nx, ny = s.sim_shape
    atoms = planar_grid(nx, ny, s.meta_atom_spacing_lambda * lam)
    Q1 = rs_coefficient(antenna_layer_distances(cfg), s.feed_to_first_layer_spacing_lambda * lam, area, lam)
    dz = s.sim_interlayer_spacing_lambda * lam
    lateral = np.linalg.norm(atoms[:, None, :] - atoms[None, :, :], axis=-1)
    Qi = rs_coefficient(np.sqrt(lateral ** 2 + dz ** 2), dz, area, lam)
    t = lambda a: torch.as_tensor(a, dtype=cdtype, device=device)
    return t(Q1), t(Qi)

# def sim_apply(phase, Qinter, Q1)  
def sim_apply(phase: torch.Tensor, Qinter: torch.Tensor, X: torch.Tensor) -> torch.Tensor: #B_l Q_{l, 1}
    """B(phase) @ X, B = Phi_M Q ... Q Phi_1; phase (..., M, N), X (..., N, C) -> (..., N, C). With X = Q1: T_l."""
    M = phase.shape[-2]
    Phi = torch.diag_embed(phase_factors(phase, Qinter.dtype))                   # (..., M, N, N)
    Y = Phi[..., 0, :, :] @ X                                                     # Phi_1 X
    for m in range(1, M):
        Y = Phi[..., m, :, :] @ (Qinter @ Y)                                      # Phi_m Q (previous layers)
    return Y

# AGA - AP-UE association
def aga(D: torch.Tensor) -> torch.Tensor:
    E, L, U, K = D.shape
    if L * U < K: # Kiểm tra yêu cầu LU >= K
        raise ValueError(f"AGA needs L*U >= K, got L*U={L * U} < K={K}")
    F = torch.zeros_like(D, dtype=torch.bool) # association matrix 
    for e in range(E):
        # Giai đoạn 1: ưu tiên cặp (antenna, UE) gần nhất trên toàn mạng
        cost = D[e].clone() # (L, U, K) - tạo clone
        for _ in range(K):
            l, u, k = torch.unravel_index(cost.argmin(), cost.shape) # Đổi vị trí giá trị nhỏ nhất thành (l, u, k)
            F[e, l, u, k] = True
            cost[l, u, :] = float("inf") # Antenna này đã được dùng
            cost[:, :, k] = float("inf") # UE này đã có antenna

        # Giai đoạn 2: Dùng nốt antenna còn trống
        ue_order = torch.argsort(D[e].mean(1), dim=-1, stable=True) # (L, K), UE gần nhất đứng trước - đã có thứ tự
        for l in range(L):
            n_assigned = 0 # Đếm số antenna trống đã xử lý của AP 
            for u in range(U):
                if not F[e, l, u].any(): # Kiểm tra antenna này chưa được gán cho UE nào
                    F[e, l, u, ue_order[l, n_assigned % K]] = True
                    n_assigned += 1
    return F

# # ===== 4. Truyền dẫn =====
# effective channel 
def effective_channel(Hbar: torch.Tensor, ports: torch.Tensor, phase: torch.Tensor,
                      Q1: torch.Tensor, Qinter: torch.Tensor) -> torch.Tensor:
    T = sim_apply(phase, Qinter, Q1) # T_l = B_l Q1: (E, L, N, U)
    H_active = gather_selected(Hbar, ports) # A_k^H Hbar: (E, L, K, n, N)
    return H_active @ T[:, :, None] # (E, L, K, n, U)

# stream channel
def stream_channels(Heff: torch.Tensor, p: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    sqrt_p = p.clamp_min(0).sqrt()[..., None]                     
    PF = (sqrt_p * F).to(Heff.dtype)                           
    d_per_ap = Heff @ PF[:, :, None]                            
    d = d_per_ap.sum(dim=1)  # d[e, k, i] kênh từ luồng dữ liệu i đến các cổng đang bật UE k
    return d.transpose(-1, -2)                                    

# mnse sinr 
def mmse_sinr(d, sigma2):
    K, n = d.shape[1], d.shape[-1]
    # Bước 1: chia cho σ để nhiễu nhiệt σ²I trở thành I
    d = d / math.sqrt(sigma2)

    # Bước 2: lấy kênh tín hiệu có ích d_kk của mỗi UE
    ue = torch.arange(K, device=d.device)
    d_kk = d[:, ue, ue].unsqueeze(-1)

    # Bước 3: C_k = (tất cả các luồng) - (luồng của chính UE k) + I
    X = d.mT                                              
    all_streams = X @ X.mH                                
    own_stream = d_kk @ d_kk.mH                           
    I = torch.eye(n, dtype=d.dtype, device=d.device)
    C = all_streams - own_stream + I

    # Bước 4: tính u = C^{-1} d_kk bằng cách giải hệ C u = d_kk
    u = torch.linalg.solve(C, d_kk) 

    # Bước 5: gamma = d_kk^H u
    gamma = (d_kk.mH @ u).squeeze(-1).squeeze(-1).real    # (E,K)
    return gamma.clamp_min(0.0)

# evaluate link
def evaluate_link(Hbar, ports, phase, p, F, Q1, Qinter, sigma2, r_min, lam):
    """Toàn bộ chuỗi: Heff -> d -> gamma -> R (21), R_sum (22), xi, reward."""
    d = stream_channels(effective_channel(Hbar, ports, phase, Q1, Qinter), p, F)
    gamma = mmse_sinr(d, sigma2)
    R = torch.log2(1.0 + gamma)                                   # Eq.(21)
    xi = (r_min - R).clamp_min(0.0)
    sum_se = R.sum(-1)                                            # Eq.(22)
    return dict(gamma=gamma, R=R, xi=xi, sum_se=sum_se, reward=sum_se - lam * xi.sum(-1))

#For test SINR
def mmse_combiner(d, sigma2):
    K, n = d.shape[1], d.shape[-1]
    sig = math.sqrt(sigma2)
    d = d / sig
    ue = torch.arange(K, device=d.device)
    X = d.mT                                                        # (E,K,n,K)
    A = X @ X.mH + torch.eye(n, dtype=d.dtype, device=d.device)     # (E,K,n,n)
    d_kk = d[:, ue, ue].unsqueeze(-1)                               # (E,K,n,1)
    return torch.linalg.solve(A, d_kk).squeeze(-1) / sig

def explicit_sinr(d, w, sigma2):
    proj = (d @ w.conj().unsqueeze(-1)).squeeze(-1)                 # (E,K,K): w_k^H d_ki
    P = proj.abs() ** 2                                             # công suất của từng luồng
    signal = P.diagonal(dim1=1, dim2=2)                             # (E,K): phần có ích
    interference = P.sum(-1) - signal                               # (E,K): sum_{i≠k}
    noise = sigma2 * (w.abs() ** 2).sum(-1)                         # (E,K): σ²||w||²
    return signal / (interference + noise)