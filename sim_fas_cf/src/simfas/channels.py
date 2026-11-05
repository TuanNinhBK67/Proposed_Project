import logging
import math
import numpy as np
import torch
from .utils import crandn

#Tạo kênh truyền từ APs đến UEs
log = logging.getLogger(__name__)

def spherical_j0(x):
    # Hàm spherical_j0(x) - hàm Bessel cầu bậc 0 
    return np.sinc(np.asarray(x, dtype=np.float64) / np.pi)

def correlation_from_points(points: np.ndarray, wavelength: float) -> np.ndarray:
    # Ma trận tương quan của các port 
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return spherical_j0(2.0 * np.pi * dist / wavelength) # Khoảng cách tương quan 

def psd_sqrt(R: np.ndarray, tol: float = 1e-9, name: str = "R") -> np.ndarray:
    # Căn bậc 2 của ma trận tương quan - Căn bậc 2 phân rã trị riêng  J = V . diag(w) . V^T
    R = 0.5 * (R + R.conj().T) # --> Đảm bảo ma trận đối xứng hoàn toàn 
    w, V = np.linalg.eigh(R) # Phân rã chuyên cho ma trận đối xứng 
    scale = max(1.0, float(w.max())) # 
    if w.min() < -tol * scale: # Đảm bảo trị riêng không âm
        raise ValueError(f"{name} is not PSD (min eigenvalue {w.min():.3e})")
    n_clip = int((w < 0).sum())
    if n_clip:
        log.info("psd_sqrt(%s): clipped %d round-off negative eigenvalue(s), min %.3e", name, n_clip, w.min())
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T

def correlation_matrices(cfg):
    # Ghép 2 ma trận tương quan
    from .fas import port_coordinates
    from .geometry import planar_grid
    s = cfg.system # config parameter 
    lam = s.wavelength # lambda 
    J = correlation_from_points(port_coordinates(s.Nx, s.Ny, s.Wx, s.Wy, lam, s.fas_centered), lam)
    if s.sim_corr_model == "j0":
        nx, ny = s.sim_shape
        R = correlation_from_points(planar_grid(nx, ny, s.meta_atom_spacing_lambda * lam), lam)
    else:
        R = np.eye(s.N)
    return J, R

def pathloss_db(D: torch.Tensor, s) -> torch.Tensor:
    # Suy hao theo khoảng cách - path_loss Beta
    if s.pathloss_model == "log_distance":
        # Suy hao không gian tự do tại 1m --> Tần số càng cao, suy hao càng lớn 
        ref = s.pl_ref_db if s.pl_ref_db is not None else 32.4 + 20.0 * math.log10(s.carrier_ghz)
        d = D.clamp_min(s.pl_ref_distance_m) # Chặn khoảng cách tối thiểu 
        return -(ref + 10.0 * s.pl_exponent * torch.log10(d / s.pl_ref_distance_m))
    d, d0, d1 = D / 1000.0, s.d0_m / 1000.0, s.d1_m / 1000.0 # three-slope - Đổi sang km 
    phi0, phi1 = (d < d0).to(d.dtype), (d < d1).to(d.dtype) # Cờ 0/1
    return s.pl_const_db - 35.0 * torch.log10(d) + 20.0 * phi0 * torch.log10(d / d0) + 15.0 * phi1 * torch.log10(d / d1)

def large_scale_gain(cfg, D: torch.Tensor, generator) -> torch.Tensor:
    # Suy hao quy mô lớn 
    s = cfg.system
    # Bốc số ngẫu nhiên cho shadowing 
    z = torch.randn(D.shape, generator=generator, device=generator.device, dtype=torch.float64).to(D.device)
    # Cộng shadowing rồi đổi sang tuyến tính 
    return 10.0 ** ((pathloss_db(D, s) + s.shadowing_std_db * z) / 10.0)

def draw_hbar(beta: torch.Tensor, J_sqrt: torch.Tensor, R_sqrt: torch.Tensor, generator) -> torch.Tensor:
    # Ghép kênh H
    E, L, K = beta.shape # Kích thước theo: số kịch bản, AP, UE
    # Sinh fading ngẫu nhiên - Biểu diễn fading quy mô nhỏ 
    G = crandn((E, L, K, J_sqrt.shape[0], R_sqrt.shape[0]), generator, beta.device, J_sqrt.dtype)
    # \beta là hệ số công suất --> Lấy sqrt(\beta)
    sb = torch.sqrt(beta).to(J_sqrt.real.dtype)[..., None, None] # --> broadcasting 
    return sb * (J_sqrt @ G @ R_sqrt) # Co giãn cả ma trận theo \beta 
