"""Spectral efficiency, QoS deficits and the penalised reward of (P2).

R_k = log2(1 + gamma_k);  xi_k = [R_k^min - R_k]^+;  r = sum_k R_k - lambda_QoS sum_k xi_k.
The penalty is soft: it does NOT guarantee R_k >= R_k^min, so QoS satisfaction is reported separately.
"""
from dataclasses import dataclass
import torch
from .phy import effective_channel, mmse_sinr, stream_channels

def spectral_efficiency(gamma): # từ SINR ra rate (Eq.21)
    return torch.log2(1.0 + gamma)

def qos_deficit(R, r_min): # phần hụt QoS
    return (r_min - R).clamp_min(0.0)

def penalised_reward(R, r_min, qos_penalty): # reward 
    return R.sum(-1) - qos_penalty * qos_deficit(R, r_min).sum(-1)

@dataclass
class LinkResult: # Hộp đựng kết quả 
    gamma: torch.Tensor      # (E, K)
    R: torch.Tensor          # (E, K)
    xi: torch.Tensor         # (E, K)
    sum_se: torch.Tensor     # (E,)
    reward: torch.Tensor     # (E,)

def evaluate_link(Hbar, ports, slot_valid, phase, p, F, Q1, Qinter, sigma2, r_min, qos_penalty) -> LinkResult:
    # Nhận kênh và action, trả về mọi thứ 
    Heff = effective_channel(Hbar, ports, slot_valid, phase, Q1, Qinter)
    d = stream_channels(Heff, p, F)
    gamma = mmse_sinr(d, sigma2)
    R = spectral_efficiency(gamma)
    xi = qos_deficit(R, r_min)
    return LinkResult(gamma, R, xi, R.sum(-1), R.sum(-1) - qos_penalty * xi.sum(-1))

def link_metrics(res: LinkResult, r_min, p, pmax) -> dict:
    # Các chỉ số để theo dõi --> Mục tiêu tạo ra báo cáo, log ra các chỉ số 
    sat = (res.R >= r_min - 1e-9) # UE nào đạt QoS 
    return {"sum_se": res.sum_se, "reward": res.reward, "qos_ue_frac": sat.double().mean(-1),
            "qos_all": sat.all(-1).double(), "shortfall": res.xi.sum(-1), "min_se": res.R.min(-1).values,
            "power_frac": (p.sum(-1) / pmax).mean(-1)}