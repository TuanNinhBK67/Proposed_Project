import numpy as np
import torch
from .geometry import feed_offsets, planar_grid

def rs_coefficient(d, dz, area, wavelength):
    # Hệ số lan truyền Rayleigh-Sommerfeld -- Eq.(4)
    return area * dz / d ** 2 * (1.0 / (2.0 * np.pi * d) - 1j / wavelength) * np.exp(1j * 2.0 * np.pi * d / wavelength)

def atom_positions(cfg) -> np.ndarray:
    # Vị trí các meta-atom -- Tạo luới 
    s = cfg.system
    nx, ny = s.sim_shape
    return planar_grid(nx, ny, s.meta_atom_spacing_lambda * s.wavelength)

def feed_layer_distances(cfg) -> np.ndarray:
    # Tính khoảng cách 3D từ mỗi feed tới mỗi atom của input layer 
    s = cfg.system
    lam = s.wavelength
    atoms = atom_positions(cfg)
    fx = feed_offsets(s.U, s.feed_spacing_lambda * lam)
    dx = atoms[:, 0, None] - fx[None, :] # Hiệu theo x, mọi cặp (atom, feed)
    dy = atoms[:, 1, None] # Hiệu theo y, (AP's atennas nằm ở y = 0)
    return np.sqrt(dx ** 2 + dy ** 2 + (s.feed_to_first_layer_spacing_lambda * lam) ** 2) # Khoảng cách không gian 3-D

def build_sim_matrices(cfg, device, cdtype):
    # Tạo các ma trận lan truyền 
    s = cfg.system
    lam = s.wavelength
    area = (s.meta_atom_spacing_lambda * lam) ** 2 # Diện tích 1 atom
    atoms = atom_positions(cfg)
    # Q1 - từ AP's antenna đến input layer 
    Q1 = rs_coefficient(feed_layer_distances(cfg), s.feed_to_first_layer_spacing_lambda * lam, area, lam)
    dz = s.sim_interlayer_spacing_lambda * lam # Khoảng cách 2 lớp 
    lateral = np.linalg.norm(atoms[:, None, :] - atoms[None, :, :], axis=-1) # Khoảng cách ngang 
    Qi = rs_coefficient(np.sqrt(lateral ** 2 + dz ** 2), dz, area, lam)
    t = lambda a: torch.as_tensor(a, dtype=cdtype, device=device)
    return t(Q1), t(Qi)

def _phase_factors(phase: torch.Tensor, cdtype) -> torch.Tensor:
    return torch.exp(1j * phase.to(torch.float64)).to(cdtype)

def sim_cascade(phase: torch.Tensor, Qinter: torch.Tensor) -> torch.Tensor:
    # Ma trận SIM B_l - Eq.(6) --> Dùng chạy test 
    e = _phase_factors(phase, Qinter.dtype)
    N = phase.shape[-1]
    B = e[..., 0, :, None] * torch.eye(N, dtype=Qinter.dtype, device=Qinter.device)
    for m in range(1, phase.shape[-2]):
        B = e[..., m, :, None] * (Qinter @ B)
    return B

def sim_apply(phase: torch.Tensor, Qinter: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # Tính B.X mà không cần tạo B 
    e = _phase_factors(phase, Qinter.dtype)
    Y = e[..., 0, :, None] * X
    for m in range(1, phase.shape[-2]):
        Y = e[..., m, :, None] * (Qinter @ Y)
    return Y
