import torch
from .fas import gather_selected
from .sim_transfer import sim_apply

def effective_channel(Hbar, ports, slot_valid, phase, Q1, Qinter):
    # Kênh thô (kênh từ 4 feed tới các cổng đang bật) - Eq.(9)
    T = sim_apply(phase, Qinter, Q1) # T_l = B_l Q1: (E, L, N, U)
    return gather_selected(Hbar, ports, slot_valid) @ T[:, :, None]

def desired_port_contributions(Hbar, Q1, Qinter, phase, p, F):
    """(E, L, K, Np) complex contribution H_eff[l,k,port] P_l f_{l,k} of AP l to UE k's own stream at every
    candidate port (zero for an AP with no feed for k)."""
    E, L, K, Np, _ = Hbar.shape
    ports = torch.arange(Np, device=Hbar.device).expand(E, K, Np)
    valid = torch.ones(K, Np, dtype=torch.bool, device=Hbar.device)
    Heff = effective_channel(Hbar, ports, valid, phase, Q1, Qinter)                  # (E, L, K, Np, U)
    w = torch.sqrt(p.clamp_min(0))[..., None] * F.to(p.dtype)                       # (E, L, U, K)
    return torch.einsum("elkau,eluk->elka", Heff, w.to(Heff.dtype))


def desired_port_power(Hbar, Q1, Qinter, phase, p, F):
    """(E, K, Np) power of UE k's own stream at every candidate port, |sum_l H_eff[l,k,port] P_l f_{l,k}|^2, i.e.
    the coherent sum over the serving APs through their current SIM phases and powers (interference ignored)."""
    return desired_port_contributions(Hbar, Q1, Qinter, phase, p, F).sum(1).abs() ** 2


def desired_port_power_local(Hbar, Q1, Qinter, phase, p, F, ctrl):
    """(E, K, Np) power of UE k's own stream at every candidate port from its FAS-control AP ctrl[e, k] ONLY,
    i.e. what that AP computes from its own CSI, SIM phases and feed powers (other serving APs ignored)."""
    c = desired_port_contributions(Hbar, Q1, Qinter, phase, p, F)
    E, L, K, Np = c.shape
    return torch.gather(c, 1, ctrl[:, None, :, None].expand(E, 1, K, Np))[:, 0].abs() ** 2


def stream_channels(Heff, p, F):
    # Mỗi luồng dữ liệu tới mỗi UE - d_{k, i}
    PF = torch.sqrt(p.clamp_min(0))[..., None] * F.to(p.dtype) # P_l F_l: (E, L, U, K)
    return torch.einsum("elkau,elui->ekia", Heff, PF.to(Heff.dtype)) # Công suất + ma trận association 

def _scaled(d, sigma2): #Chia d cho căn \sigma 
    E, K = d.shape[:2]
    sig = torch.as_tensor(sigma2, dtype=d.real.dtype, device=d.device)
    sig = torch.broadcast_to(sig, (E, K))
    return d / torch.sqrt(sig)[:, :, None, None], sig

def mmse_sinr(d, sigma2):
    # Công thức SINR sử dụng bộ thu MMSE tối ưu (Eq.18)
    E, K, _, n = d.shape
    ds, _ = _scaled(d, sigma2)
    diag = torch.arange(K, device=d.device) # Tập tín hiệu mong muốn (d_1,1, d_2,2, ...)
    dkk = ds[:, diag, diag, :] # (E,K,n)
    # Ma trận nhiễu C_k
    C = torch.einsum("ekia,ekib->ekab", ds, ds.conj()) - torch.einsum("eka,ekb->ekab", dkk, dkk.conj())
    C = C + torch.eye(n, dtype=d.dtype, device=d.device)
    C = 0.5 * (C + C.conj().transpose(-1, -2)) #Ép đối xứng tuyệt đối 
    u = torch.linalg.solve(C, dkk[..., None])[..., 0] # Giải hệ thay vì nghịch đảo 
    gamma = torch.einsum("eka,eka->ek", dkk.conj(), u).real.clamp_min(0.0) # y = d_kk ^ H . u 
    desired = (dkk.abs() ** 2).sum(-1)
    return torch.where(desired > 0, gamma, torch.zeros_like(gamma))

def mmse_combiner(d, sigma2):
    # Chỉ dùng cho test - Eq.19 - tính chính bộ kết hợp w 
    E, K, _, n = d.shape
    ds, sig = _scaled(d, sigma2)
    diag = torch.arange(K, device=d.device)
    A = torch.einsum("ekia,ekib->ekab", ds, ds.conj()) + torch.eye(n, dtype=d.dtype, device=d.device)
    u = torch.linalg.solve(A, ds[:, diag, diag, :, None])[..., 0]      # (A_s)^{-1} d_kk / sigma
    return u / torch.sqrt(sig)[..., None]                                # = (sigma2 A_s)^{-1} d_kk

def explicit_sinr(d, w, sigma2):
    # Chỉ dùng cho test - Eq.17 - tính SINR 
    E, K = d.shape[:2]
    proj = torch.einsum("eka,ekia->eki", w.conj(), d).abs() ** 2
    diag = torch.arange(K, device=d.device)
    sig = torch.broadcast_to(torch.as_tensor(sigma2, dtype=proj.dtype, device=d.device), (E, K))
    num = proj[:, diag, diag]
    return num / (proj.sum(-1) - num + sig * (w.abs() ** 2).sum(-1))
